#!/usr/bin/env python3
"""
Phase 10f: Null-Space Correction via Weight-Weight Sign
=========================================================

The bias-aware decomposition captures 99.7% of routing signal but stacking
fails because the discarded weight-weight term carries DIRECTIONAL information
(like negative zero in the 4-state gate).

The full score has 4 terms:
  1. baseline(δ)           = b_qᵀ R(δ) b_k        [position, pre-computed]
  2. h(i)·c_q(δ)           = h(i)ᵀ W_qᵀ R(δ) b_k  [query cross-term]
  3. c_k(δ)·h(j)           = b_qᵀ R(δ) W_k h(j)    [key cross-term]
  4. h(i)ᵀ M(δ) h(j)       = weight-weight term     [NULL SPACE]

Terms 1-3 are what we compute. Term 4 is what we discard.
The hypothesis: the SIGN of term 4 is the missing correction signal.

sign(term4) ≈ sign(h(i)·d_q) × sign(h(j)·d_k) × sign(u1ᵀ R(δ) v1)

This gives us a 4-state encoding per score entry:
  {cross > 0, ww_sign > 0}  →  push further positive
  {cross > 0, ww_sign < 0}  →  positive but pull back (overshot)
  {cross < 0, ww_sign > 0}  →  negative but pull back
  {cross < 0, ww_sign < 0}  →  push further negative

Tests:
  1. Measure actual weight-weight term values
  2. Compare sign(ww_approx) vs sign(ww_actual)
  3. Test: bias-aware + ww_sign correction (per-layer and stacked)
  4. Test with zone-aware anchoring + ww_sign for COMB layers
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

print("=" * 80)
print("  PHASE 10f: NULL-SPACE CORRECTION")
print("  The weight-weight sign as the 4th state")
print("=" * 80)
print()

results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_classification = {}
for ls in phase9a['layer_summary']:
    layer_classification[ls['layer']] = {
        'fixed': set(ls['fixed_heads']),
        'routing': set(ls['routing_heads']),
    }

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager",
)
model.eval()

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7; HIDDEN_DIM = 3584
ROPE_THETA = 1000000.0
MAX_SEQ = 64

def phi_softmax_torch(scores, dim=-1):
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)

def apply_rotary_pos_emb(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

def get_rope_cache(seq_len, device, dtype):
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

def rope_rotate_vector(v, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    v1 = v[: len(v) // 2]
    v2 = v[len(v) // 2 :]
    return v * cos_d + torch.cat((-v2, v1)) * sin_d

def rope_rotate_matrix_cols(M, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    M1 = M[: HEAD_DIM // 2, :]
    M2 = M[HEAD_DIM // 2 :, :]
    return M * cos_d.unsqueeze(1) + torch.cat((-M2, M1), dim=0) * sin_d.unsqueeze(1)


# ================================================================
# Extract EVERYTHING: W_q, W_k, b_q, b_k, d_q, d_k, position_factor
# ================================================================
print("Extracting weights, biases, and null-space directions...")
inv_freq_cpu = 1.0 / (ROPE_THETA ** (
    torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

head_tables = {}
nullspace_data = {}  # {(layer, head): {d_q, d_k, position_factor, ww_scale}}

for layer_idx in range(N_LAYERS):
    attn = model.model.layers[layer_idx].self_attn
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    W_q_all = torch.zeros(NUM_HEADS, HEAD_DIM, HIDDEN_DIM, device="cpu", dtype=torch.float32)
    W_k_all = torch.zeros(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM, device="cpu", dtype=torch.float32)

    for s in range(0, HIDDEN_DIM, 512):
        e = min(s + 512, HIDDEN_DIM)
        chunk = identity[s:e].unsqueeze(0)
        with torch.no_grad():
            qo = attn.q_proj(chunk).float()
            ko = attn.k_proj(chunk).float()
        qr = qo[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = ko[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in range(NUM_HEADS):
            W_q_all[h, :, s:e] = qr[:, h, :].T
        for g in range(NUM_KV_HEADS):
            W_k_all[g, :, s:e] = kr[:, g, :].T

    zero_input = torch.zeros(1, 1, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        q_bias_raw = attn.q_proj(zero_input).float()[0, 0]
        k_bias_raw = attn.k_proj(zero_input).float()[0, 0]
    b_q_all = q_bias_raw.reshape(NUM_HEADS, HEAD_DIM).cpu()
    b_k_all = k_bias_raw.reshape(NUM_KV_HEADS, HEAD_DIM).cpu()

    for h in range(NUM_HEADS):
        W_q_all[h] -= b_q_all[h].unsqueeze(1)
    for g in range(NUM_KV_HEADS):
        W_k_all[g] -= b_k_all[g].unsqueeze(1)

    routing = layer_classification[layer_idx]['routing']
    for h in routing:
        g = h // HEADS_PER_KV
        W_q_h = W_q_all[h]; W_k_g = W_k_all[g]
        b_q_h = b_q_all[h]; b_k_g = b_k_all[g]
        scale = 1.0 / math.sqrt(HEAD_DIM)

        # Bias-aware tables (terms 1-3)
        baseline = torch.zeros(MAX_SEQ)
        c_q = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        c_k = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        for delta in range(MAX_SEQ):
            b_k_rotated = rope_rotate_vector(b_k_g, delta, inv_freq_cpu)
            W_k_rotated = rope_rotate_matrix_cols(W_k_g, delta, inv_freq_cpu)
            baseline[delta] = (b_q_h @ b_k_rotated) * scale
            c_q[delta] = (W_q_h.T @ b_k_rotated) * scale
            c_k[delta] = (W_k_rotated.T @ b_q_h) * scale

        head_tables[(layer_idx, h)] = {'baseline': baseline, 'c_q': c_q, 'c_k': c_k}

        # Null-space: SVD of MESH for dominant direction
        MESH = W_q_h @ W_k_g.T  # (HEAD_DIM, HEAD_DIM)
        U, S, Vt = torch.linalg.svd(MESH)
        u1 = U[:, 0]
        v1 = Vt[0, :]
        sigma1 = S[0].item()

        # d_q = W_q.T @ u1 (hidden-space query direction)
        # d_k = W_k.T @ v1 (hidden-space key direction)
        d_q = W_q_h.T @ u1  # (HIDDEN_DIM,)
        d_k = W_k_g.T @ v1  # (HIDDEN_DIM,)

        # Position factor for null space: u1ᵀ R(δ) v1
        pf = torch.zeros(MAX_SEQ)
        for delta in range(MAX_SEQ):
            v1_rotated = rope_rotate_vector(v1, delta, inv_freq_cpu)
            pf[delta] = u1 @ v1_rotated

        nullspace_data[(layer_idx, h)] = {
            'd_q': d_q, 'd_k': d_k,
            'position_factor': pf,
            'sigma1': sigma1,
            'scale': scale,
        }

    del W_q_all, W_k_all
    torch.cuda.empty_cache()
    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx} done")

print(f"  {len(head_tables)} head tables + {len(nullspace_data)} null-space directions")
print()


# ================================================================
# ANALYSIS 1: Measure actual ww term and sign accuracy
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: Weight-Weight Term Measurement")
print("=" * 80)
print()

DIAG_PROMPTS = [
    "The capital of France is",
    "Albert Einstein developed the theory of",
    "To be or not to",
    "The largest planet in our solar system is",
    "The color of grass is",
]

ww_actual_vals = []
ww_approx_signs = []
ww_actual_signs = []
cross_vals = []
per_layer_sign_agreement = defaultdict(list)

for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    real_data = {}
    def capture_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b, s, _ = h.shape
            with torch.no_grad():
                Q = module.q_proj(h).to(torch.bfloat16)
                K = module.k_proj(h).to(torch.bfloat16)
            Q = Q.reshape(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            K = K.reshape(b, s, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            cos, sin = get_rope_cache(s, h.device, torch.bfloat16)
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
            K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
            scores = {}
            for hd in range(NUM_HEADS):
                scores[hd] = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HEAD_DIM)).float().cpu()
            real_data[layer_idx] = {'scores': scores, 'h': h[0].cpu().float()}
            return output
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            capture_hook(li), with_kwargs=True))
    with torch.no_grad():
        model(ids, return_dict=True)
    for hk in hooks:
        hk.remove()

    for li in real_data:
        h_states = real_data[li]['h']
        s = h_states.shape[0]
        for hd in layer_classification[li]['routing']:
            if (li, hd) not in head_tables:
                continue
            tbl = head_tables[(li, hd)]
            ns = nullspace_data[(li, hd)]
            real_sc = real_data[li]['scores'][hd]

            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    # Terms 1-3
                    bl = tbl['baseline'][delta].item()
                    cq = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_states[j]).item()
                    cross = cq + ck
                    terms123 = bl + cross

                    # Actual ww term = real_score - terms123
                    ww_actual = real_sc[i, j].item() - terms123

                    # Approximate ww sign via dominant direction
                    sq = (h_states[i] @ ns['d_q']).item()
                    sk = (h_states[j] @ ns['d_k']).item()
                    pf = ns['position_factor'][delta].item()
                    ww_approx = sq * sk * pf * ns['sigma1'] * ns['scale']

                    ww_actual_vals.append(ww_actual)
                    cross_vals.append(cross)

                    # Sign comparison
                    actual_sign = 1 if ww_actual >= 0 else -1
                    approx_sign = 1 if ww_approx >= 0 else -1
                    ww_actual_signs.append(actual_sign)
                    ww_approx_signs.append(approx_sign)
                    per_layer_sign_agreement[li].append(1 if actual_sign == approx_sign else 0)

ww_arr = np.array(ww_actual_vals)
cross_arr = np.array(cross_vals)
actual_signs = np.array(ww_actual_signs)
approx_signs = np.array(ww_approx_signs)

print(f"  Total samples: {len(ww_arr)}")
print()
print(f"  Weight-weight term magnitude:")
print(f"    mean |ww| = {np.abs(ww_arr).mean():.4f}")
print(f"    mean |cross| = {np.abs(cross_arr).mean():.4f}")
print(f"    ratio |ww|/|cross| = {np.abs(ww_arr).mean() / max(np.abs(cross_arr).mean(), 1e-10):.4f}")
print()

# Sign agreement
overall_agree = np.mean(actual_signs == approx_signs)
print(f"  Sign agreement (approx vs actual):")
print(f"    Overall: {overall_agree:.1%}")
print()
print(f"    Per-layer:")
for li in sorted(per_layer_sign_agreement.keys()):
    vals = per_layer_sign_agreement[li]
    print(f"      L{li:2d}: {np.mean(vals):.1%} ({len(vals)} samples)")

print()

# Correlation of ww with residual
residual = ww_arr  # the ww term IS the residual from our decomposition
corr_ww_approx = np.corrcoef(ww_arr, [s*sk for s,sk in zip(ww_approx_signs, [1]*len(ww_approx_signs))])[0,1] if len(ww_arr) > 1 else 0
print(f"  ww_actual ↔ ww_approx correlation: {np.corrcoef(ww_arr, np.array([ww_approx_signs[i] * abs(ww_arr[i]) for i in range(len(ww_arr))]))[0,1]:.4f}")
print()


# ================================================================
# ANALYSIS 2: Does ww sign improve score correlation?
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: Score Improvement with Null-Space Sign")
print("=" * 80)
print()

# Collect paired data: (real_score, bias_only_score, bias+ww_sign_score)
all_real = []
all_bias = []
all_bias_ww = []

for pi, prompt in enumerate(DIAG_PROMPTS[:3]):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    real_data2 = {}
    def capture_hook2(layer_idx):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b, s, _ = h.shape
            with torch.no_grad():
                Q = module.q_proj(h).to(torch.bfloat16)
                K = module.k_proj(h).to(torch.bfloat16)
            Q = Q.reshape(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            K = K.reshape(b, s, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            cos, sin = get_rope_cache(s, h.device, torch.bfloat16)
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
            K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
            scores = {}
            for hd in range(NUM_HEADS):
                scores[hd] = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HEAD_DIM)).float().cpu()
            real_data2[layer_idx] = {'scores': scores, 'h': h[0].cpu().float()}
            return output
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            capture_hook2(li), with_kwargs=True))
    with torch.no_grad():
        model(ids, return_dict=True)
    for hk in hooks:
        hk.remove()

    for li in real_data2:
        h_states = real_data2[li]['h']
        s = h_states.shape[0]
        for hd in layer_classification[li]['routing']:
            if (li, hd) not in head_tables:
                continue
            tbl = head_tables[(li, hd)]
            ns = nullspace_data[(li, hd)]
            real_sc = real_data2[li]['scores'][hd]

            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cq = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_states[j]).item()
                    bias_score = bl + cq + ck

                    # Null-space sign correction
                    sq = (h_states[i] @ ns['d_q']).item()
                    sk = (h_states[j] @ ns['d_k']).item()
                    pf = ns['position_factor'][delta].item()
                    ww_approx = sq * sk * pf * ns['sigma1'] * ns['scale']

                    all_real.append(real_sc[i, j].item())
                    all_bias.append(bias_score)
                    all_bias_ww.append(bias_score + ww_approx)

real_arr = np.array(all_real)
bias_arr = np.array(all_bias)
bias_ww_arr = np.array(all_bias_ww)

corr_bias = np.corrcoef(real_arr, bias_arr)[0, 1]
corr_bias_ww = np.corrcoef(real_arr, bias_ww_arr)[0, 1]

print(f"  Score correlation with real attention:")
print(f"    Bias-aware only (terms 1-3):  {corr_bias:.6f}")
print(f"    Bias + ww_approx (all 4):     {corr_bias_ww:.6f}")
print(f"    Improvement:                  {corr_bias_ww - corr_bias:+.6f}")
print()

# MSE comparison
mse_bias = np.mean((real_arr - bias_arr) ** 2)
mse_bias_ww = np.mean((real_arr - bias_ww_arr) ** 2)
print(f"  MSE with real attention:")
print(f"    Bias-aware only:  {mse_bias:.6f}")
print(f"    Bias + ww_approx: {mse_bias_ww:.6f}")
print(f"    Reduction:        {(1 - mse_bias_ww/mse_bias):.1%}")
print()


# ================================================================
# END-TO-END: Bias-aware + null-space sign correction
# ================================================================
print("=" * 80)
print("  END-TO-END: Attention with Null-Space Correction")
print("=" * 80)
print()

def attn_bias_plus_nullspace(layer_idx, h_normed, attn_module):
    """Bias-aware + weight-weight approximation (all 4 terms)."""
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]

    h_float = h_normed[0].float().cpu()
    for h in routing:
        tbl = head_tables[(layer_idx, h)]
        ns = nullspace_data[(layer_idx, h)]

        # Pre-compute per-position projections
        proj_q = h_float @ ns['d_q']  # (seq_len,)
        proj_k = h_float @ ns['d_k']  # (seq_len,)

        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                delta = i - j
                # Terms 1-3 (bias-aware)
                bl = tbl['baseline'][delta].item()
                cq = (h_float[i] @ tbl['c_q'][delta]).item()
                ck = (tbl['c_k'][delta] @ h_float[j]).item()

                # Term 4 (null-space approximation)
                ww = proj_q[i].item() * proj_k[j].item() * ns['position_factor'][delta].item() * ns['sigma1'] * ns['scale']

                scores[i, j] = bl + cq + ck + ww

        scores = scores.to(h_normed.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_bias_plus_nullspace_sign(layer_idx, h_normed, attn_module):
    """Bias-aware + SIGN ONLY of weight-weight term (4-state)."""
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]

    h_float = h_normed[0].float().cpu()
    for h in routing:
        tbl = head_tables[(layer_idx, h)]
        ns = nullspace_data[(layer_idx, h)]
        proj_q = h_float @ ns['d_q']
        proj_k = h_float @ ns['d_k']

        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                delta = i - j
                bl = tbl['baseline'][delta].item()
                cq = (h_float[i] @ tbl['c_q'][delta]).item()
                ck = (tbl['c_k'][delta] @ h_float[j]).item()

                # Sign-only null space correction
                ww_sign = math.copysign(1.0,
                    proj_q[i].item() * proj_k[j].item() * ns['position_factor'][delta].item())

                # Scale the sign by cross-term magnitude (relative correction)
                cross_mag = abs(cq + ck)
                ww_correction = ww_sign * cross_mag * 0.1  # small push in ww direction

                scores[i, j] = bl + cq + ck + ww_correction

        scores = scores.to(h_normed.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_real_qk(layer_idx, h_normed, attn_module):
    batch, seq_len, _ = h_normed.shape
    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).to(torch.bfloat16)
        K = attn_module.k_proj(h_normed).to(torch.bfloat16)
        V_full = attn_module.v_proj(h_normed)
    Q = Q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    cos, sin = get_rope_cache(seq_len, h_normed.device, torch.bfloat16)
    Q = apply_rotary_pos_emb(Q, cos, sin)
    K = apply_rotary_pos_emb(K, cos, sin)
    K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
    for hd in range(NUM_HEADS):
        sc = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HEAD_DIM)).float()
        sc.masked_fill_(mask, float('-inf'))
        w = phi_softmax_torch(sc, dim=-1)
        attn_out[0, :, hd, :] = (w.to(torch.bfloat16) @ V_exp[0, :, hd, :].to(torch.bfloat16)).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def run_with_hooks(input_ids, attn_fn_map):
    hooks = []
    for layer_idx, attn_fn in attn_fn_map.items():
        def make_hook(li, fn):
            def hook_fn(module, args, kwargs, output):
                h = args[0] if args else kwargs.get('hidden_states')
                if h is None: return output
                geo = fn(li, h, module)
                return (geo,) + output[1:] if isinstance(output, tuple) else geo
            return hook_fn
        hk = model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(layer_idx, attn_fn), with_kwargs=True)
        hooks.append(hk)
    try:
        with torch.no_grad():
            out = model(input_ids, return_dict=True)
        logits = out.logits
    finally:
        for hk in hooks:
            hk.remove()
    return logits


TEST_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "Barack Obama was the",
    "To be or not to",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "Albert Einstein developed the theory of",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Shakespeare wrote many",
    "The square root of 144 is",
    "In mathematics, pi is approximately equal to",
    "The color of the sky is usually",
]

print("Collecting baselines...")
baseline_tokens = []
for p in TEST_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    baseline_tokens.append(out.logits[0, -1, :].float().argmax().item())
print(f"  {len(TEST_PROMPTS)} baselines ready.")
print()


def evaluate(name, attn_fn_map):
    n_match = 0; cos_list = []
    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits = run_with_hooks(ids, attn_fn_map)
        gl = logits[0, -1, :].float()
        if gl.argmax().item() == baseline_tokens[pi]:
            n_match += 1
        with torch.no_grad():
            bl = model(ids, return_dict=True).logits[0, -1, :].float()
        cos = F.cosine_similarity(bl.cpu().unsqueeze(0), gl.cpu().unsqueeze(0)).item()
        cos_list.append(cos)
    return n_match, len(TEST_PROMPTS), float(np.mean(cos_list))


print(f"  {'Config':>55s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 75)

# Baseline
n, t, c = evaluate("A", {i: attn_real_qk for i in range(N_LAYERS)})
print(f"  {'A: All real QK':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# All bias-aware (no null space)
n, t, c = evaluate("B", {i: attn_bias_plus_nullspace for i in range(N_LAYERS)})
print(f"  {'B: All bias + null-space approx (stacked)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# All bias + null-space sign
n, t, c = evaluate("C", {i: attn_bias_plus_nullspace_sign for i in range(N_LAYERS)})
print(f"  {'C: All bias + null-space SIGN (stacked)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Zone-aware: anchor L0-3,27 + bias+nullspace L4-26
cfg_d = {}
for i in range(4):
    cfg_d[i] = attn_real_qk
cfg_d[27] = attn_real_qk
for i in range(4, 27):
    cfg_d[i] = attn_bias_plus_nullspace
n, t, c = evaluate("D", cfg_d)
print(f"  {'D: Anchor(0-3,27) + bias+nullspace(4-26)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Zone-aware: DRUM + every4 COMB + MUSIC with null-space
cfg_e = {}
anchor_layers = set(range(4)) | {7, 11, 15, 19, 23, 27}
for i in anchor_layers:
    cfg_e[i] = attn_real_qk
for i in set(range(N_LAYERS)) - anchor_layers:
    cfg_e[i] = attn_bias_plus_nullspace
n, t, c = evaluate("E", cfg_e)
print(f"  {'E: Zone(DRUM+every4+MUSIC) + nullspace':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Same zone but with sign-only null space
cfg_f = {}
for i in anchor_layers:
    cfg_f[i] = attn_real_qk
for i in set(range(N_LAYERS)) - anchor_layers:
    cfg_f[i] = attn_bias_plus_nullspace_sign
n, t, c = evaluate("F", cfg_f)
print(f"  {'F: Zone(DRUM+every4+MUSIC) + nullspace SIGN':>55s}  {n:2d}/{t:2d}    {c:.4f}")

print()

# Per-layer comparison: bias-only vs bias+nullspace
print("  Per-layer comparison (replace one layer at a time):")
print(f"  {'Layer':>6s}  {'Bias-only':>10s}  {'Bias+NS':>10s}  {'Bias+NS_sign':>12s}")
print("  " + "-" * 45)

from functools import partial

# Helper to make single-layer configs
def attn_bias_only(layer_idx, h_normed, attn_module):
    """Original bias-aware from 10b."""
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]
    h_float = h_normed[0].float().cpu()
    for h in routing:
        tbl = head_tables[(layer_idx, h)]
        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                d = i - j
                bl = tbl['baseline'][d].item()
                cq = (h_float[i] @ tbl['c_q'][d]).item()
                ck = (tbl['c_k'][d] @ h_float[j]).item()
                scores[i, j] = bl + cq + ck
        scores = scores.to(h_normed.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)

# Test select layers
for li in [0, 3, 7, 13, 20, 27]:
    n1, _, c1 = evaluate(f"L{li}-bias", {li: attn_bias_only})
    n2, _, c2 = evaluate(f"L{li}-bias+ns", {li: attn_bias_plus_nullspace})
    n3, _, c3 = evaluate(f"L{li}-bias+ns_s", {li: attn_bias_plus_nullspace_sign})
    delta = "↑" if c2 > c1 else ("↓" if c2 < c1 else "=")
    print(f"  L{li:2d}:  {n1:2d}/15 ({c1:.4f})  {n2:2d}/15 ({c2:.4f}) {delta}  {n3:2d}/15 ({c3:.4f})")

print()

# Save
save_data = {
    'ww_magnitude': {
        'mean_abs_ww': float(np.abs(ww_arr).mean()),
        'mean_abs_cross': float(np.abs(cross_arr).mean()),
        'ratio': float(np.abs(ww_arr).mean() / max(np.abs(cross_arr).mean(), 1e-10)),
    },
    'sign_agreement': float(overall_agree),
    'score_correlation': {
        'bias_only': float(corr_bias),
        'bias_plus_ww': float(corr_bias_ww),
    },
    'mse': {
        'bias_only': float(mse_bias),
        'bias_plus_ww': float(mse_bias_ww),
    },
}
save_path = os.path.join(results_dir, 'phase10f_nullspace.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved to {save_path}")
print()
print("=" * 80)
print("  DONE — Phase 10f Null-Space Correction")
print("=" * 80)
