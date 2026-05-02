#!/usr/bin/env python3
"""
Phase 10b: Bias-Aware QK Decomposition
========================================

Finding 90 revealed that the rank-1 MESH is 99.94% the bias outer product
b_q ⊗ b_k — which is TOKEN-INDEPENDENT. The actual routing signal lives
in the 0.06% cross-terms (bias × weight).

The full score with RoPE decomposes as:

  score(i,j) = b_qᵀ R(δ) b_k                 [position baseline, pre-computable]
             + h(i)ᵀ W_qᵀ R(δ) b_k            [query-dependent, 1 dot product]
             + b_qᵀ R(δ) W_k h(j)              [key-dependent, 1 dot product]
             + h(i)ᵀ W_qᵀ R(δ) W_k h(j)       [both-dependent, tiny]

Where δ = i - j (relative position).

Pre-computable per head:
  baseline(δ)  = b_qᵀ R(δ) b_k / √d           — scalar table (seq_len entries)
  c_q(δ)       = W_qᵀ R(δ) b_k / √d           — vector table (seq_len × HIDDEN_DIM)  
  c_k(δ)       = (b_qᵀ R(δ) W_k)ᵀ / √d        — vector table (seq_len × HIDDEN_DIM)

Runtime cost per (i,j):
  score(i,j) = baseline(δ) + h(i)·c_q(δ) + c_k(δ)·h(j)
  = 1 table lookup + 2 dot products (HIDDEN_DIM each)

vs full QK: Q projection (HIDDEN_DIM × HEAD_DIM) + K projection + QK^T

This script:
  1. Validates the decomposition (score correlation with real QK)
  2. Measures the energy in each term
  3. Tests end-to-end with phi_softmax (per-layer and stacked)
  4. Compares with and without the weight-weight term

Requires: Qwen2-7B on GPU, phase9a results
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
print("  PHASE 10b: BIAS-AWARE QK DECOMPOSITION")
print("  Separating position baseline from token routing signal")
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
    """Apply RoPE rotation by delta positions to vector v (HEAD_DIM,)."""
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    v1 = v[: len(v) // 2]
    v2 = v[len(v) // 2 :]
    return v * cos_d + torch.cat((-v2, v1)) * sin_d

def rope_rotate_matrix_cols(M, delta, inv_freq):
    """Apply RoPE rotation by delta to each column of M (HEAD_DIM × N).
    R(δ) @ M — rotates each column."""
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    M1 = M[: HEAD_DIM // 2, :]
    M2 = M[HEAD_DIM // 2 :, :]
    return M * cos_d.unsqueeze(1) + torch.cat((-M2, M1), dim=0) * sin_d.unsqueeze(1)


# ================================================================
# Extract W_q, W_k, b_q, b_k for ALL heads (not just routing)
# ================================================================
print("Extracting W_q, W_k, b_q, b_k for all heads...")

inv_freq_cpu = 1.0 / (ROPE_THETA ** (
    torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

# Store per-head decomposition tables
# For each head: baseline(δ), c_q(δ), c_k(δ)
head_tables = {}  # {(layer, head): {baseline: (MAX_SEQ,), c_q: (MAX_SEQ, HIDDEN_DIM), c_k: (MAX_SEQ, HIDDEN_DIM)}}

for layer_idx in range(N_LAYERS):
    attn = model.model.layers[layer_idx].self_attn
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

    # Extract full W_q (NUM_HEADS × HEAD_DIM × HIDDEN_DIM) and W_k
    W_q_all = torch.zeros(NUM_HEADS, HEAD_DIM, HIDDEN_DIM, device="cpu", dtype=torch.float32)
    W_k_all = torch.zeros(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM, device="cpu", dtype=torch.float32)

    for s in range(0, HIDDEN_DIM, 512):
        e = min(s + 512, HIDDEN_DIM)
        chunk = identity[s:e].unsqueeze(0)
        with torch.no_grad():
            # Get raw Q/K without bias to separate weight from bias
            qo = attn.q_proj(chunk).float()
            ko = attn.k_proj(chunk).float()
        qr = qo[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = ko[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in range(NUM_HEADS):
            W_q_all[h, :, s:e] = qr[:, h, :].T
        for g in range(NUM_KV_HEADS):
            W_k_all[g, :, s:e] = kr[:, g, :].T

    # Extract biases
    # Q/K projections include bias; W_q @ 0 + b_q = b_q
    zero_input = torch.zeros(1, 1, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        q_bias_raw = attn.q_proj(zero_input).float()[0, 0]  # (NUM_HEADS * HEAD_DIM,)
        k_bias_raw = attn.k_proj(zero_input).float()[0, 0]  # (NUM_KV_HEADS * HEAD_DIM,)
    b_q_all = q_bias_raw.reshape(NUM_HEADS, HEAD_DIM).cpu()
    b_k_all = k_bias_raw.reshape(NUM_KV_HEADS, HEAD_DIM).cpu()

    # Now W_q includes the bias contribution. Fix: W_q from identity probe INCLUDES bias
    # since q_proj(e_i) = W_q[:, i] + b_q. So our extracted W_q is actually correct
    # (identity probe gives W_q columns), but we need to subtract bias contribution.
    # Actually: q_proj(identity[s:e]) = W_q[:, s:e] + b_q (broadcast over batch dim s:e)
    # The reshape + transpose gives us W_q + outer(ones, b_q) effectively.
    # No — the identity probe gives q_proj(e_i) for each basis vector e_i.
    # q_proj(e_i) = W_q @ e_i + b_q = W_q[:, i] + b_q
    # So our "W_q_all[h, :, i]" = W_q[h, :, i] + b_q[h, :]
    # We need to subtract b_q!
    for h in range(NUM_HEADS):
        W_q_all[h] -= b_q_all[h].unsqueeze(1)  # subtract bias from each column
    for g in range(NUM_KV_HEADS):
        W_k_all[g] -= b_k_all[g].unsqueeze(1)

    # Build decomposition tables for routing heads
    routing = layer_classification[layer_idx]['routing']
    for h in routing:
        g = h // HEADS_PER_KV
        W_q_h = W_q_all[h]   # (HEAD_DIM, HIDDEN_DIM)
        W_k_g = W_k_all[g]   # (HEAD_DIM, HIDDEN_DIM)
        b_q_h = b_q_all[h]   # (HEAD_DIM,)
        b_k_g = b_k_all[g]   # (HEAD_DIM,)

        baseline = torch.zeros(MAX_SEQ)
        c_q = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        c_k = torch.zeros(MAX_SEQ, HIDDEN_DIM)

        scale = 1.0 / math.sqrt(HEAD_DIM)

        for delta in range(MAX_SEQ):
            # R(δ) applied to b_k and W_k columns
            b_k_rotated = rope_rotate_vector(b_k_g, delta, inv_freq_cpu)
            W_k_rotated = rope_rotate_matrix_cols(W_k_g, delta, inv_freq_cpu)

            # Term 1: b_qᵀ R(δ) b_k (scalar)
            baseline[delta] = (b_q_h @ b_k_rotated) * scale

            # Term 2: W_qᵀ R(δ) b_k → project into hidden space
            # h(i)ᵀ W_qᵀ R(δ) b_k = h(i) · (W_q.T @ R(δ) b_k)
            # W_q is (HEAD_DIM, HIDDEN_DIM), so W_q.T is (HIDDEN_DIM, HEAD_DIM)
            c_q[delta] = (W_q_h.T @ b_k_rotated) * scale  # (HIDDEN_DIM,)

            # Term 3: b_qᵀ R(δ) W_k → project into hidden space
            # b_qᵀ R(δ) W_k h(j) = (W_k_rotated.T @ b_q) · h(j)
            c_k[delta] = (W_k_rotated.T @ b_q_h) * scale  # (HIDDEN_DIM,)

        head_tables[(layer_idx, h)] = {
            'baseline': baseline,
            'c_q': c_q,
            'c_k': c_k,
        }

    del W_q_all, W_k_all
    torch.cuda.empty_cache()

    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx}: {len(routing)} routing heads precomputed")

n_total = len(head_tables)
print(f"  Total: {n_total} head tables precomputed")
print()

# Memory estimate
mem_per_head = MAX_SEQ * (1 + HIDDEN_DIM + HIDDEN_DIM) * 4  # float32
print(f"  Memory per head table: {mem_per_head / 1024:.1f} KB")
print(f"  Total precomputed: {n_total * mem_per_head / 1024 / 1024:.1f} MB")
print()


# ================================================================
# VALIDATION 1: Score correlation with real attention
# ================================================================
print("=" * 80)
print("  VALIDATION: Score Correlation — Does the decomposition work?")
print("=" * 80)
print()

DIAG_PROMPTS = [
    "The capital of France is",
    "Albert Einstein developed the theory of",
    "To be or not to",
    "The largest planet in our solar system is",
    "Once upon a time in a land",
]

corr_decomposed = defaultdict(list)
corr_baseline_only = defaultdict(list)
corr_cross_only = defaultdict(list)
energy_breakdown = defaultdict(lambda: defaultdict(list))

for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    real_data = {}

    def capture_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None:
                return output
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
        h_states = real_data[li]['h']  # (s, HIDDEN_DIM)
        s = h_states.shape[0]
        routing = layer_classification[li]['routing']

        for hd in routing:
            if (li, hd) not in head_tables:
                continue
            real_sc = real_data[li]['scores'][hd]
            tbl = head_tables[(li, hd)]

            # Reconstruct scores using decomposition
            decomposed = torch.zeros(s, s)
            term_baseline = torch.zeros(s, s)
            term_cross = torch.zeros(s, s)

            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cq = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_states[j]).item()

                    term_baseline[i, j] = bl
                    term_cross[i, j] = cq + ck
                    decomposed[i, j] = bl + cq + ck

            mask = torch.tril(torch.ones(s, s)).bool()
            real_vals = real_sc[mask]
            dec_vals = decomposed[mask]
            bl_vals = term_baseline[mask]
            cr_vals = term_cross[mask]

            if dec_vals.std() > 0 and real_vals.std() > 0:
                corr_full = torch.corrcoef(torch.stack([real_vals, dec_vals]))[0, 1].item()
                corr_decomposed[li].append(corr_full)

            if bl_vals.std() > 0 and real_vals.std() > 0:
                corr_bl = torch.corrcoef(torch.stack([real_vals, bl_vals]))[0, 1].item()
                corr_baseline_only[li].append(corr_bl)

            if cr_vals.std() > 0 and real_vals.std() > 0:
                corr_cr = torch.corrcoef(torch.stack([real_vals, cr_vals]))[0, 1].item()
                corr_cross_only[li].append(corr_cr)

            # Energy breakdown
            bl_energy = bl_vals.abs().sum().item()
            cr_energy = cr_vals.abs().sum().item()
            total_energy = real_vals.abs().sum().item()
            energy_breakdown[li]['baseline'].append(bl_energy / max(total_energy, 1e-10))
            energy_breakdown[li]['cross'].append(cr_energy / max(total_energy, 1e-10))

# Summary
print(f"  {'Component':>25s}  {'Mean corr':>9s}  {'Min':>7s}  {'Max':>7s}")
print("  " + "-" * 55)

for name, data in [('Full decomposition', corr_decomposed),
                    ('Baseline only (b×b)', corr_baseline_only),
                    ('Cross terms only', corr_cross_only)]:
    all_vals = []
    for li in data:
        all_vals.extend(data[li])
    if all_vals:
        print(f"  {name:>25s}  {np.mean(all_vals):>9.4f}  {np.min(all_vals):>7.4f}  {np.max(all_vals):>7.4f}")

print()
print("  Per-layer decomposition correlation:")
for li in sorted(corr_decomposed.keys()):
    cd = np.mean(corr_decomposed[li])
    cb = np.mean(corr_baseline_only.get(li, [0]))
    cc = np.mean(corr_cross_only.get(li, [0]))
    be = np.mean(energy_breakdown[li]['baseline'])
    ce = np.mean(energy_breakdown[li]['cross'])
    print(f"    L{li:2d}: full={cd:+.4f}  baseline={cb:+.4f}  cross={cc:+.4f}  "
          f"energy: bl={be:.2%} cr={ce:.2%}")

print()


# ================================================================
# END-TO-END TEST: Bias-aware attention replacement
# ================================================================
print("=" * 80)
print("  END-TO-END: Bias-Aware Attention (15 prompts)")
print("=" * 80)
print()

def attn_bias_aware(layer_idx, h_normed, attn_module):
    """Bias-aware decomposed attention using precomputed tables."""
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
                delta = i - j
                bl = tbl['baseline'][delta]
                cq = h_float[i] @ tbl['c_q'][delta]
                ck = tbl['c_k'][delta] @ h_float[j]
                scores[i, j] = bl + cq + ck

        scores = scores.to(h_normed.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_full_qk_phi_bf16(layer_idx, h_normed, attn_module):
    """Full QK in bf16 + phi_softmax (known-good baseline)."""
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
                if h is None:
                    return output
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
baseline_margins = []
for p in TEST_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    logits = out.logits[0, -1, :].float()
    top2 = logits.topk(2)
    baseline_tokens.append(top2.indices[0].item())
    baseline_margins.append((top2.values[0] - top2.values[1]).item())
print(f"  {len(TEST_PROMPTS)} baselines ready.")
print()


def evaluate(name, attn_fn_map):
    n_match = 0; fails = []; cos_list = []
    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits = run_with_hooks(ids, attn_fn_map)
        gl = logits[0, -1, :].float()
        gid = gl.argmax().item()
        if gid == baseline_tokens[pi]:
            n_match += 1
        else:
            got = tokenizer.decode([gid])
            base = tokenizer.decode([baseline_tokens[pi]])
            fails.append((pi, prompt[:45], base, got, baseline_margins[pi]))
        with torch.no_grad():
            bl = model(ids, return_dict=True).logits[0, -1, :].float()
        cos = F.cosine_similarity(bl.cpu().unsqueeze(0), gl.cpu().unsqueeze(0)).item()
        cos_list.append(cos)
    return n_match, len(TEST_PROMPTS), float(np.mean(cos_list)), fails


# --- Shootout ---
print(f"  {'Config':>55s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 75)

# Baseline
cfg_base = {i: attn_full_qk_phi_bf16 for i in range(N_LAYERS)}
n, t, c, fails = evaluate("Baseline", cfg_base)
print(f"  {'Baseline: Full QK + phi_softmax (bf16)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Bias-aware: all layers
cfg_bias = {i: attn_bias_aware for i in range(N_LAYERS)}
n, t, c, fails = evaluate("Bias-aware all layers", cfg_bias)
print(f"  {'Bias-aware decomposition (all 28 layers)':>55s}  {n:2d}/{t:2d}    {c:.4f}")
if fails:
    for fi, fp, fb, fg, m in fails[:5]:
        print(f"    FAIL: \"{fp}\" base='{fb}' got='{fg}' margin={m:.3f}")

print()

# --- Per-layer ablation ---
print("  Per-layer ablation (replace ONE layer at a time):")
per_layer = {}
for li in range(N_LAYERS):
    cfg = {li: attn_bias_aware}
    n, t, c, _ = evaluate(f"L{li}", cfg)
    n_r = len(layer_classification[li]['routing'])
    status = "" if n == t else f"  ← FAIL({n}/{t})"
    per_layer[li] = {'n': n, 'cos': c}
    print(f"    L{li:2d}: {n:2d}/{t:2d}  cos={c:.4f}  routing={n_r:2d}{status}")

n_perfect = sum(1 for v in per_layer.values() if v['n'] == 15)
print(f"\n  Perfect layers: {n_perfect}/28")
print()

# --- Progressive stacking ---
print("  Progressive stacking (add layers one at a time, easiest first):")
# Sort layers by per-layer cos (best first)
sorted_layers = sorted(per_layer.keys(), key=lambda l: per_layer[l]['cos'], reverse=True)
# Only include layers that have routing heads
sorted_layers = [l for l in sorted_layers if layer_classification[l]['routing']]

for n_layers in [1, 2, 4, 8, 12, 16, 20, 24, 28]:
    chosen = sorted_layers[:min(n_layers, len(sorted_layers))]
    cfg = {li: attn_bias_aware for li in chosen}
    n, t, c, _ = evaluate(f"Top-{n_layers}", cfg)
    layers_str = ','.join(str(l) for l in sorted(chosen)[:6])
    if len(chosen) > 6:
        layers_str += ',...'
    print(f"    {n_layers:2d} layers: {n:2d}/{t:2d}  cos={c:.4f}  [{layers_str}]")

print()

# Save
save_data = {
    'per_layer': per_layer,
    'score_correlations': {
        'decomposed': {str(k): float(np.mean(v)) for k, v in corr_decomposed.items()},
        'baseline_only': {str(k): float(np.mean(v)) for k, v in corr_baseline_only.items()},
        'cross_only': {str(k): float(np.mean(v)) for k, v in corr_cross_only.items()},
    },
    'energy_breakdown': {
        str(k): {'baseline': float(np.mean(v['baseline'])), 'cross': float(np.mean(v['cross']))}
        for k, v in energy_breakdown.items()
    },
}
save_path = os.path.join(results_dir, 'phase10b_bias_aware.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved to {save_path}")

print()
print("=" * 80)
print("  DONE — Phase 10b Bias-Aware QK Decomposition")
print("=" * 80)
