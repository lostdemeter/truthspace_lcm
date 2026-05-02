#!/usr/bin/env python3
"""
Phase 10c: Routing Information Encoding Analysis
==================================================

Finding 90 showed the cross-terms (bias×weight) carry the routing signal
but are only 0.06% of weight-space energy. The hypothesis:

  The model minimized weight-space cost, then encoded routing decisions
  as SIGN FLIPS in the cross-term projections. The information may be:
    - Binary:  sign(cross_term) = {+1, -1}
    - Ternary: {+1, 0, -1} with a dead zone
    - 4-state: {+1, +0, -0, -1} matching the gate construction

This script:
  1. Computes cross-term projections h(i)·c_q(δ) and c_k(δ)·h(j)
  2. Analyzes their distributions — clustering, sign structure
  3. Tests quantized routing: sign-only, ternary, 4-state
  4. Measures score correlation at each quantization level
  5. End-to-end test with quantized cross-terms + phi_softmax

Connects to: Finding 41 (sign-only d_k works), Finding 61 (4-state gate)

Requires: Qwen2-7B on GPU, phase9a results, phase10b head_tables
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
print("  PHASE 10c: ROUTING INFORMATION ENCODING ANALYSIS")
print("  Is the routing signal binary, ternary, or 4-state?")
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
# Rebuild head_tables (same as phase10b)
# ================================================================
print("Building bias-aware decomposition tables...")

inv_freq_cpu = 1.0 / (ROPE_THETA ** (
    torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

head_tables = {}

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

        baseline = torch.zeros(MAX_SEQ)
        c_q = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        c_k = torch.zeros(MAX_SEQ, HIDDEN_DIM)

        for delta in range(MAX_SEQ):
            b_k_rotated = rope_rotate_vector(b_k_g, delta, inv_freq_cpu)
            W_k_rotated = rope_rotate_matrix_cols(W_k_g, delta, inv_freq_cpu)
            baseline[delta] = (b_q_h @ b_k_rotated) * scale
            c_q[delta] = (W_q_h.T @ b_k_rotated) * scale
            c_k[delta] = (W_k_rotated.T @ b_q_h) * scale

        head_tables[(layer_idx, h)] = {
            'baseline': baseline, 'c_q': c_q, 'c_k': c_k,
        }

    del W_q_all, W_k_all
    torch.cuda.empty_cache()
    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx} done")

print(f"  {len(head_tables)} head tables ready")
print()


# ================================================================
# ANALYSIS 1: Cross-term distribution
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: Cross-Term Projection Distributions")
print("=" * 80)
print()

DIAG_PROMPTS = [
    "The capital of France is",
    "Albert Einstein developed the theory of",
    "To be or not to",
    "The largest planet in our solar system is",
    "Once upon a time in a land",
    "The color of grass is",
    "Shakespeare wrote many",
    "Water freezes at zero degrees",
]

all_cq_vals = []  # all h(i)·c_q(δ) values
all_ck_vals = []  # all c_k(δ)·h(j) values
all_cross_sums = []  # cq + ck combined
all_real_scores = []
per_layer_cq = defaultdict(list)
per_layer_ck = defaultdict(list)

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
        routing = layer_classification[li]['routing']

        for hd in routing:
            if (li, hd) not in head_tables:
                continue
            tbl = head_tables[(li, hd)]
            real_sc = real_data[li]['scores'][hd]

            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    cq_val = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck_val = (tbl['c_k'][delta] @ h_states[j]).item()
                    cross_sum = cq_val + ck_val
                    real_val = real_sc[i, j].item()

                    all_cq_vals.append(cq_val)
                    all_ck_vals.append(ck_val)
                    all_cross_sums.append(cross_sum)
                    all_real_scores.append(real_val)
                    per_layer_cq[li].append(cq_val)
                    per_layer_ck[li].append(ck_val)

all_cq = np.array(all_cq_vals)
all_ck = np.array(all_ck_vals)
all_cross = np.array(all_cross_sums)
all_real = np.array(all_real_scores)

print(f"  Total score samples: {len(all_cq)}")
print()

# Distribution stats
print(f"  c_q projections (h·c_q):")
print(f"    mean={all_cq.mean():.6f}  std={all_cq.std():.6f}")
print(f"    min={all_cq.min():.4f}  max={all_cq.max():.4f}")
print(f"    |val| < 0.01: {(np.abs(all_cq) < 0.01).mean():.1%}")
print(f"    |val| < 0.001: {(np.abs(all_cq) < 0.001).mean():.1%}")
print()

print(f"  c_k projections (c_k·h):")
print(f"    mean={all_ck.mean():.6f}  std={all_ck.std():.6f}")
print(f"    min={all_ck.min():.4f}  max={all_ck.max():.4f}")
print(f"    |val| < 0.01: {(np.abs(all_ck) < 0.01).mean():.1%}")
print(f"    |val| < 0.001: {(np.abs(all_ck) < 0.001).mean():.1%}")
print()

print(f"  Cross sum (cq + ck):")
print(f"    mean={all_cross.mean():.6f}  std={all_cross.std():.6f}")
print(f"    min={all_cross.min():.4f}  max={all_cross.max():.4f}")
print()

# Sign distribution
n_pos_cq = (all_cq > 0).sum()
n_neg_cq = (all_cq < 0).sum()
n_zero_cq = (all_cq == 0).sum()
print(f"  Sign distribution (c_q):")
print(f"    positive: {n_pos_cq / len(all_cq):.1%}  negative: {n_neg_cq / len(all_cq):.1%}")
print()

# 4-state classification using log(φ) thresholds
threshold = LOG_PHI  # ~0.481
print(f"  4-state classification (threshold = log(φ) = {threshold:.4f}):")
for name, vals in [("c_q", all_cq), ("c_k", all_ck), ("cross_sum", all_cross)]:
    expand = (vals >= threshold).sum()
    preserve_plus = ((vals >= 0) & (vals < threshold)).sum()
    preserve_minus = ((vals < 0) & (vals > -threshold)).sum()
    contract = (vals <= -threshold).sum()
    print(f"    {name}: EXPAND={expand/len(vals):.1%}  +0={preserve_plus/len(vals):.1%}  "
          f"-0={preserve_minus/len(vals):.1%}  CONTRACT={contract/len(vals):.1%}")
print()

# Smaller threshold — maybe the routing signal is more subtle
for t_name, t_val in [("0.1", 0.1), ("0.01", 0.01), ("std/2", all_cross.std()/2)]:
    above = (all_cross > t_val).mean()
    below = (all_cross < -t_val).mean()
    dead = 1 - above - below
    print(f"  Threshold {t_name} ({t_val:.4f}): +={above:.1%}  dead={dead:.1%}  -={below:.1%}")
print()


# ================================================================
# ANALYSIS 2: Quantized score correlation
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: Quantized Routing — How Many Bits Carry the Signal?")
print("=" * 80)
print()

def quantize_binary(vals):
    """Sign-only: +1 / -1"""
    return np.sign(vals)

def quantize_ternary(vals, threshold):
    """Ternary: +1 / 0 / -1"""
    result = np.zeros_like(vals)
    result[vals > threshold] = 1.0
    result[vals < -threshold] = -1.0
    return result

def quantize_4state(vals, threshold):
    """+1, +0, -0, -1 using threshold"""
    result = np.zeros_like(vals)
    result[vals >= threshold] = 1.0
    result[(vals >= 0) & (vals < threshold)] = 0.1   # +0 (small positive)
    result[(vals < 0) & (vals > -threshold)] = -0.1  # -0 (small negative)
    result[vals <= -threshold] = -1.0
    return result

# For each quantization, compute how well quantized cross-terms
# reproduce the RELATIVE ordering of real scores (which is what softmax cares about)

# We need baseline + quantized cross to correlate with real
# First get all baseline values
all_baselines = []
for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]
    # Re-run to get h states (simplified - use stored data approach)
    # We already collected data above, but let's use the cross_sums directly

# Actually, let's compute correlation of cross-terms with (real - baseline)
# Since score = baseline + cross + weight_weight, and we want to test
# whether quantized cross captures the routing signal (real - baseline)
# We need baseline values too. Let me recompute properly.

print("  Recomputing with paired baseline values...")
all_residuals = []  # real_score - baseline (the token-dependent part)
all_cross_paired = []  # corresponding cross-term values

for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

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
        routing = layer_classification[li]['routing']
        for hd in routing:
            if (li, hd) not in head_tables:
                continue
            tbl = head_tables[(li, hd)]
            real_sc = real_data2[li]['scores'][hd]
            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cq = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_states[j]).item()
                    cross = cq + ck
                    residual = real_sc[i, j].item() - bl
                    all_residuals.append(residual)
                    all_cross_paired.append(cross)

residuals = np.array(all_residuals)
crosses = np.array(all_cross_paired)

print(f"  Samples: {len(residuals)}")
print(f"  Residual (real - baseline): mean={residuals.mean():.4f} std={residuals.std():.4f}")
print(f"  Cross-terms: mean={crosses.mean():.4f} std={crosses.std():.4f}")
print()

# Correlation of cross-terms with residual
base_corr = np.corrcoef(residuals, crosses)[0, 1]
print(f"  Cross-term ↔ residual correlation: {base_corr:.4f}")
print()

# Now test quantized versions
print(f"  {'Quantization':>35s}  {'Corr with residual':>18s}  {'Info retained':>13s}")
print("  " + "-" * 70)

# Full precision (reference)
print(f"  {'Full precision cross-terms':>35s}  {base_corr:>18.4f}  {'100%':>13s}")

# Binary (sign only)
binary = quantize_binary(crosses)
bc = np.corrcoef(residuals, binary)[0, 1]
print(f"  {'Binary: sign(cross)':>35s}  {bc:>18.4f}  {bc/base_corr:>12.1%}")

# Ternary at various thresholds
for t_name, t_val in [("std/4", crosses.std()/4),
                       ("std/2", crosses.std()/2),
                       ("std", crosses.std()),
                       ("0.01", 0.01),
                       ("0.05", 0.05),
                       ("0.1", 0.1),
                       ("log(φ)", LOG_PHI)]:
    tern = quantize_ternary(crosses, t_val)
    tc = np.corrcoef(residuals, tern)[0, 1]
    n_active = (tern != 0).mean()
    print(f"  {'Ternary (t=' + t_name + ')':>35s}  {tc:>18.4f}  {tc/base_corr:>12.1%}  active={n_active:.1%}")

# 4-state
for t_name, t_val in [("std/2", crosses.std()/2),
                       ("0.1", 0.1),
                       ("log(φ)", LOG_PHI)]:
    fs = quantize_4state(crosses, t_val)
    fc = np.corrcoef(residuals, fs)[0, 1]
    print(f"  {'4-state (t=' + t_name + ')':>35s}  {fc:>18.4f}  {fc/base_corr:>12.1%}")

# Multi-bit uniform quantization
for n_bits in [2, 3, 4, 8]:
    n_levels = 2 ** n_bits
    vmin, vmax = crosses.min(), crosses.max()
    step = (vmax - vmin) / n_levels
    quantized = np.round((crosses - vmin) / step) * step + vmin
    qc = np.corrcoef(residuals, quantized)[0, 1]
    print(f"  {'Uniform ' + str(n_bits) + '-bit':>35s}  {qc:>18.4f}  {qc/base_corr:>12.1%}")

print()


# ================================================================
# ANALYSIS 3: Per-layer information structure
# ================================================================
print("=" * 80)
print("  ANALYSIS 3: Per-Layer Cross-Term Structure")
print("=" * 80)
print()

# Collect per-layer cross-term stats
per_layer_crosses = defaultdict(list)
per_layer_residuals = defaultdict(list)

for pi, prompt in enumerate(DIAG_PROMPTS[:3]):  # fewer for speed
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    real_data3 = {}
    def capture_hook3(layer_idx):
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
            real_data3[layer_idx] = {'scores': scores, 'h': h[0].cpu().float()}
            return output
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            capture_hook3(li), with_kwargs=True))
    with torch.no_grad():
        model(ids, return_dict=True)
    for hk in hooks:
        hk.remove()

    for li in real_data3:
        h_states = real_data3[li]['h']
        s = h_states.shape[0]
        for hd in layer_classification[li]['routing']:
            if (li, hd) not in head_tables:
                continue
            tbl = head_tables[(li, hd)]
            real_sc = real_data3[li]['scores'][hd]
            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cq = (h_states[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_states[j]).item()
                    per_layer_crosses[li].append(cq + ck)
                    per_layer_residuals[li].append(real_sc[i, j].item() - bl)

print(f"  {'Layer':>6s}  {'Cross std':>10s}  {'Resid std':>10s}  "
      f"{'Full corr':>10s}  {'Sign corr':>10s}  {'Sign/Full':>10s}")
print("  " + "-" * 65)

layer_analysis = {}
for li in sorted(per_layer_crosses.keys()):
    cr = np.array(per_layer_crosses[li])
    re = np.array(per_layer_residuals[li])
    if len(cr) < 5:
        continue
    full_c = np.corrcoef(re, cr)[0, 1]
    sign_c = np.corrcoef(re, np.sign(cr))[0, 1]
    ratio = sign_c / full_c if abs(full_c) > 0.01 else 0
    layer_analysis[li] = {
        'cross_std': float(cr.std()),
        'resid_std': float(re.std()),
        'full_corr': float(full_c),
        'sign_corr': float(sign_c),
        'sign_ratio': float(ratio),
    }
    flag = " ← sign captures >90%" if ratio > 0.9 else ""
    print(f"  L{li:2d}:   {cr.std():>10.4f}  {re.std():>10.4f}  "
          f"{full_c:>+10.4f}  {sign_c:>+10.4f}  {ratio:>10.1%}{flag}")

# How many layers have sign_ratio > 90%?
n_sign_sufficient = sum(1 for v in layer_analysis.values() if v['sign_ratio'] > 0.9)
print(f"\n  Layers where sign carries >90% of info: {n_sign_sufficient}/{len(layer_analysis)}")
print()


# ================================================================
# END-TO-END: Quantized attention replacement
# ================================================================
print("=" * 80)
print("  END-TO-END: Best Quantization Strategy (15 prompts)")
print("=" * 80)
print()

def make_quantized_attn(quantize_fn):
    """Create attention function with quantized cross-terms."""
    def attn_fn(layer_idx, h_normed, attn_module):
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
                    bl = tbl['baseline'][delta].item()
                    cq = (h_float[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_float[j]).item()
                    cross = cq + ck
                    scores[i, j] = bl + quantize_fn(cross)

            scores = scores.to(h_normed.device)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
            weights = phi_softmax_torch(scores.float(), dim=-1)
            v_h = V_exp[0, :, h, :].float()
            attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

        combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
        with torch.no_grad():
            return attn_module.o_proj(combined)
    return attn_fn

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

# Test per-layer (replace one layer at a time) with different quantizations
# Pick a representative layer that was perfect in phase10b
test_layer = 20  # cos=0.997 in phase10b
print(f"  Single-layer test (L{test_layer}):")
print(f"  {'Strategy':>35s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 55)

strategies = [
    ("Full precision (baseline)", lambda x: x),
    ("Binary: sign(cross)", lambda x: float(np.sign(x))),
    ("Ternary (t=0.01)", lambda x: float(np.sign(x)) if abs(x) > 0.01 else 0.0),
    ("Ternary (t=std/2)", lambda x: float(np.sign(x)) if abs(x) > crosses.std()/2 else 0.0),
    ("4-state (t=log(φ))", lambda x: 1.0 if x >= LOG_PHI else (0.1 if x >= 0 else (-0.1 if x > -LOG_PHI else -1.0))),
    ("4-bit uniform", lambda x: round(x / (crosses.std()/8)) * (crosses.std()/8)),
]

for sname, sfn in strategies:
    cfg = {test_layer: make_quantized_attn(sfn)}
    n, t, c = evaluate(sname, cfg)
    print(f"  {sname:>35s}  {n:2d}/{t:2d}    {c:.4f}")

# Now test stacked (all 28 layers) with the most promising strategies
print()
print(f"  Stacked (all 28 layers):")
print(f"  {'Strategy':>35s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 55)

for sname, sfn in strategies:
    cfg = {i: make_quantized_attn(sfn) for i in range(N_LAYERS)}
    n, t, c = evaluate(sname, cfg)
    print(f"  {sname:>35s}  {n:2d}/{t:2d}    {c:.4f}")

print()

# Save
save_data = {
    'distribution': {
        'cq_mean': float(all_cq.mean()), 'cq_std': float(all_cq.std()),
        'ck_mean': float(all_ck.mean()), 'ck_std': float(all_ck.std()),
        'cross_mean': float(all_cross.mean()), 'cross_std': float(all_cross.std()),
    },
    'base_correlation': float(base_corr),
    'layer_analysis': layer_analysis,
}
save_path = os.path.join(results_dir, 'phase10c_routing_info.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved to {save_path}")

print()
print("=" * 80)
print("  DONE — Phase 10c Routing Information Encoding")
print("=" * 80)
