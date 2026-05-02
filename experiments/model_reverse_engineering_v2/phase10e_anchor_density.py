#!/usr/bin/env python3
"""
Phase 10e: Anchor Density Sweep
=================================

Finding from 10d: COMB replacement quality doesn't matter — what matters
is how many layers use real QK (anchors) vs precomputed.

This script sweeps anchor density to find the threshold:
  - Every Nth layer real QK
  - Critical-layer anchoring (worst per-layer layers from 10b)
  - Targeted anchoring by zone (DRUM + worst COMB + MUSIC)

Key question: What's the minimum number of real QK layers needed
for acceptable stacked accuracy?

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

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

print("=" * 80)
print("  PHASE 10e: ANCHOR DENSITY SWEEP")
print("  How many real QK layers are needed?")
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

# Build bias-aware tables
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
        scale = 1.0 / math.sqrt(HEAD_DIM)
        baseline = torch.zeros(MAX_SEQ)
        c_q = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        c_k = torch.zeros(MAX_SEQ, HIDDEN_DIM)
        for delta in range(MAX_SEQ):
            b_k_rotated = rope_rotate_vector(b_k_all[g], delta, inv_freq_cpu)
            W_k_rotated = rope_rotate_matrix_cols(W_k_all[g], delta, inv_freq_cpu)
            baseline[delta] = (b_q_all[h] @ b_k_rotated) * scale
            c_q[delta] = (W_q_all[h].T @ b_k_rotated) * scale
            c_k[delta] = (W_k_rotated.T @ b_q_all[h]) * scale
        head_tables[(layer_idx, h)] = {'baseline': baseline, 'c_q': c_q, 'c_k': c_k}
    del W_q_all, W_k_all
    torch.cuda.empty_cache()
    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx} done")

print(f"  {len(head_tables)} head tables ready")
print()


# ================================================================
# Attention functions
# ================================================================
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


def attn_bias_aware(layer_idx, h_normed, attn_module):
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


# ================================================================
# SWEEP 1: Uniform anchor density (every Nth layer)
# ================================================================
print("=" * 80)
print("  SWEEP 1: Uniform Anchor Density")
print("  Real QK every N layers, bias-aware elsewhere")
print("=" * 80)
print()

print(f"  {'Config':>45s}  {'Anchors':>7s}  {'Score':>7s}  {'Cos':>7s}  {'Saved':>7s}")
print("  " + "-" * 80)

all_results = {}

for stride in [1, 2, 3, 4, 7, 14]:
    anchor_layers = set(range(0, N_LAYERS, stride))
    # Always include L0 and L27
    anchor_layers.add(0)
    anchor_layers.add(27)
    approx_layers = set(range(N_LAYERS)) - anchor_layers

    cfg = {}
    for li in anchor_layers:
        cfg[li] = attn_real_qk
    for li in approx_layers:
        cfg[li] = attn_bias_aware

    n_anchors = len(anchor_layers)
    n_approx = len(approx_layers)
    name = f"Every {stride} (anchors={n_anchors})"

    n, t, c = evaluate(name, cfg)
    saved_pct = n_approx / N_LAYERS * 100
    print(f"  {name:>45s}  {n_anchors:2d}/28   {n:2d}/{t:2d}    {c:.4f}  {saved_pct:5.1f}%")
    all_results[f'uniform_stride_{stride}'] = {
        'n': n, 'cos': c, 'anchors': n_anchors, 'approx': n_approx,
        'anchor_layers': sorted(anchor_layers)
    }

print()

# ================================================================
# SWEEP 2: Per-layer failure-guided anchoring
# (Anchor at layers that fail in per-layer ablation from 10b)
# ================================================================
print("=" * 80)
print("  SWEEP 2: Failure-Guided Anchoring")
print("  Anchor at layers that fail per-layer, bias-aware elsewhere")
print("=" * 80)
print()

# Per-layer results from phase10b
per_layer_10b = {
    0: 1, 1: 14, 2: 15, 3: 15, 4: 14, 5: 15, 6: 14, 7: 13,
    8: 15, 9: 15, 10: 15, 11: 14, 12: 15, 13: 15, 14: 15, 15: 15,
    16: 15, 17: 15, 18: 15, 19: 15, 20: 15, 21: 15, 22: 15, 23: 15,
    24: 15, 25: 15, 26: 15, 27: 14
}

# Sort layers by failure count (worst first)
layers_by_failure = sorted(range(N_LAYERS), key=lambda l: per_layer_10b[l])

print(f"  Layers sorted by per-layer accuracy:")
print(f"    Worst: {[(l, per_layer_10b[l]) for l in layers_by_failure[:10]]}")
print()

print(f"  {'Config':>45s}  {'Anchors':>7s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 70)

# Progressively add anchors at worst layers
for n_anchors in [1, 2, 3, 5, 7, 10, 14]:
    anchor_layers = set(layers_by_failure[:n_anchors])
    approx_layers = set(range(N_LAYERS)) - anchor_layers
    cfg = {}
    for li in anchor_layers:
        cfg[li] = attn_real_qk
    for li in approx_layers:
        cfg[li] = attn_bias_aware

    name = f"Worst-{n_anchors} ({sorted(anchor_layers)})"
    n, t, c = evaluate(name, cfg)
    print(f"  {name:>45s}  {n_anchors:2d}/28   {n:2d}/{t:2d}    {c:.4f}")
    all_results[f'worst_{n_anchors}'] = {
        'n': n, 'cos': c, 'anchors': n_anchors,
        'anchor_layers': sorted(anchor_layers)
    }

print()

# ================================================================
# SWEEP 3: Zone-aware anchoring
# ================================================================
print("=" * 80)
print("  SWEEP 3: Zone-Aware Anchoring")
print("=" * 80)
print()

zone_configs = {
    'DRUM only (L0-3)': list(range(4)),
    'DRUM + MUSIC (L0-3,27)': list(range(4)) + [27],
    'DRUM + mid + MUSIC (L0-3,14,27)': list(range(4)) + [14, 27],
    'DRUM + every4 COMB + MUSIC': list(range(4)) + [7, 11, 15, 19, 23] + [27],
    'DRUM + every3 COMB + MUSIC': list(range(4)) + [7, 10, 13, 16, 19, 22, 25] + [27],
    'DRUM + every2 COMB + MUSIC': list(range(4)) + list(range(5, 27, 2)) + [27],
}

print(f"  {'Config':>50s}  {'Anchors':>7s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 75)

for name, anchors in zone_configs.items():
    anchor_set = set(anchors)
    approx_set = set(range(N_LAYERS)) - anchor_set
    cfg = {}
    for li in anchor_set:
        cfg[li] = attn_real_qk
    for li in approx_set:
        cfg[li] = attn_bias_aware

    n, t, c = evaluate(name, cfg)
    n_a = len(anchor_set)
    print(f"  {name:>50s}  {n_a:2d}/28   {n:2d}/{t:2d}    {c:.4f}")
    all_results[f'zone_{name[:20]}'] = {
        'n': n, 'cos': c, 'anchors': n_a,
        'anchor_layers': sorted(anchor_set)
    }

print()

# ================================================================
# Summary: Accuracy vs Anchor Count
# ================================================================
print("=" * 80)
print("  SUMMARY: Accuracy vs QK Compute Saved")
print("=" * 80)
print()

summary = []
for name, r in all_results.items():
    saved = (28 - r['anchors']) / 28 * 100
    summary.append((r['n'], r['cos'], r['anchors'], saved, name))

summary.sort(key=lambda x: (-x[0], -x[1]))

print(f"  {'Score':>7s}  {'Cos':>7s}  {'Anchors':>7s}  {'QK saved':>8s}  Config")
print("  " + "-" * 75)
for n, c, a, s, name in summary:
    print(f"  {n:2d}/15   {c:.4f}  {a:2d}/28    {s:5.1f}%    {name}")

# Save
save_path = os.path.join(results_dir, 'phase10e_anchor_density.json')
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved to {save_path}")
print()
print("=" * 80)
print("  DONE — Phase 10e Anchor Density")
print("=" * 80)
