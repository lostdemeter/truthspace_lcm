#!/usr/bin/env python3
"""
Phase 10d: Scaffold Trajectory Attention
==========================================

Combines three insights from prior design considerations:

1. MESH principle (Doc 169): Pre-compute the coupled product
2. Trajectory scaffolding (Doc 183): Navigation is 99.58% universal
3. Direction/magnitude separation (Doc 229): Separate WHAT from HOW MUCH

Architecture:
  - L0-3 (click point): Real QK + phi_softmax (establishes trajectory)
  - L4-26 (COMB): Scaffold scores + sign-based deviation
  - L27 (bottleneck): Real QK + phi_softmax (corrects drift)

For COMB layers:
  1. Pre-compute scaffold_score(i,j) = baseline(δ) + mean_h(i)·c_q(δ) + c_k(δ)·mean_h(j)
     from representative prompts (the 99.58% predictable part)
  2. At runtime: Δh = h_actual - h_mean
  3. deviation = Δh(i)·c_q(δ) + c_k(δ)·Δh(j)
  4. score = scaffold_score + deviation  (or scaffold_score + sign(deviation))

Also tests:
  - Anchor-only (real QK at L0-3,27, full bias-aware at L4-26)
  - Scaffold + full deviation
  - Scaffold + sign-only deviation
  - Various anchor configurations

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
print("  PHASE 10d: SCAFFOLD TRAJECTORY ATTENTION")
print("  Pre-computed trajectory + sign-based deviation routing")
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
# Build bias-aware tables (same as 10b)
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
        head_tables[(layer_idx, h)] = {'baseline': baseline, 'c_q': c_q, 'c_k': c_k}

    del W_q_all, W_k_all
    torch.cuda.empty_cache()
    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx} done")

print(f"  {len(head_tables)} head tables ready")
print()


# ================================================================
# STEP 1: Capture mean hidden-state trajectories
# ================================================================
print("=" * 80)
print("  STEP 1: Capturing Mean Hidden-State Trajectories")
print("=" * 80)
print()

SCAFFOLD_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "To be or not to",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "Albert Einstein developed the theory of",
    "Shakespeare wrote many",
]

# Collect hidden states at each layer for scaffold prompts
# We need per-position mean hidden states at each layer
mean_hidden_per_layer = {}  # {layer_idx: {seq_len: mean_h tensor (seq_len, HIDDEN_DIM)}}

for pi, prompt in enumerate(SCAFFOLD_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    layer_hidden = {}
    def capture_hook(li):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is not None:
                layer_hidden[li] = h[0].cpu().float()
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

    for li in layer_hidden:
        s = layer_hidden[li].shape[0]
        if li not in mean_hidden_per_layer:
            mean_hidden_per_layer[li] = {}
        if s not in mean_hidden_per_layer[li]:
            mean_hidden_per_layer[li][s] = {'sum': torch.zeros(s, HIDDEN_DIM), 'count': 0}
        mean_hidden_per_layer[li][s]['sum'] += layer_hidden[li]
        mean_hidden_per_layer[li][s]['count'] += 1

# Compute means
for li in mean_hidden_per_layer:
    for s in mean_hidden_per_layer[li]:
        d = mean_hidden_per_layer[li][s]
        d['mean'] = d['sum'] / d['count']

print(f"  Captured trajectories from {len(SCAFFOLD_PROMPTS)} prompts")
print(f"  Sequence lengths seen: {sorted(set(s for li in mean_hidden_per_layer for s in mean_hidden_per_layer[li]))}")
print()


# ================================================================
# STEP 2: Pre-compute scaffold scores for COMB layers
# ================================================================
print("=" * 80)
print("  STEP 2: Pre-computing Scaffold Scores")
print("=" * 80)
print()

# For each (layer, head, seq_len), pre-compute scaffold attention scores
# using mean hidden states
scaffold_scores = {}  # {(layer, head, seq_len): tensor (seq_len, seq_len)}

for li in range(4, 27):  # COMB layers only
    routing = layer_classification[li]['routing']
    for h in routing:
        if (li, h) not in head_tables:
            continue
        tbl = head_tables[(li, h)]
        for s in mean_hidden_per_layer.get(li, {}):
            mean_h = mean_hidden_per_layer[li][s]['mean']
            scores = torch.zeros(s, s)
            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cq = (mean_h[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ mean_h[j]).item()
                    scores[i, j] = bl + cq + ck
            scaffold_scores[(li, h, s)] = scores

print(f"  Pre-computed {len(scaffold_scores)} scaffold score matrices")
print()


# ================================================================
# Attention functions
# ================================================================

def attn_real_qk_phi_bf16(layer_idx, h_normed, attn_module):
    """Real QK in bf16 + phi_softmax (for anchor layers)."""
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
    """Full bias-aware decomposition (from phase10b)."""
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


def make_scaffold_attn(use_sign_only=False):
    """Create scaffold + deviation attention for COMB layers."""
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

        # Get mean hidden states for this seq_len
        mean_h = None
        if layer_idx in mean_hidden_per_layer and seq_len in mean_hidden_per_layer[layer_idx]:
            mean_h = mean_hidden_per_layer[layer_idx][seq_len]['mean']

        for h in routing:
            tbl = head_tables[(layer_idx, h)]

            # Try to get pre-computed scaffold
            scaffold = scaffold_scores.get((layer_idx, h, seq_len))

            if scaffold is not None and mean_h is not None:
                # Compute deviation from mean trajectory
                delta_h = h_float - mean_h  # (seq_len, HIDDEN_DIM)

                scores = torch.zeros(seq_len, seq_len)
                for i in range(seq_len):
                    for j in range(i + 1):
                        d = i - j
                        # Deviation scores from delta_h
                        dev_cq = (delta_h[i] @ tbl['c_q'][d]).item()
                        dev_ck = (tbl['c_k'][d] @ delta_h[j]).item()
                        deviation = dev_cq + dev_ck

                        if use_sign_only:
                            # Only the SIGN of deviation matters
                            if abs(deviation) > 0.001:
                                deviation = math.copysign(1.0, deviation)
                            else:
                                deviation = 0.0

                        scores[i, j] = scaffold[i, j].item() + deviation
            else:
                # Fallback: full bias-aware (no scaffold available)
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
    return attn_fn


# ================================================================
# Runner
# ================================================================
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


# ================================================================
# Test prompts & baselines
# ================================================================
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
    n_match = 0; cos_list = []; fails = []
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
            fails.append((pi, prompt[:40], base, got, baseline_margins[pi]))
        with torch.no_grad():
            bl = model(ids, return_dict=True).logits[0, -1, :].float()
        cos = F.cosine_similarity(bl.cpu().unsqueeze(0), gl.cpu().unsqueeze(0)).item()
        cos_list.append(cos)
    return n_match, len(TEST_PROMPTS), float(np.mean(cos_list)), fails


# ================================================================
# SHOOTOUT
# ================================================================
print("=" * 80)
print("  SHOOTOUT: Scaffold Trajectory Configurations")
print("=" * 80)
print()

scaffold_full = make_scaffold_attn(use_sign_only=False)
scaffold_sign = make_scaffold_attn(use_sign_only=True)

configs = {}

# Config A: Baseline (all real QK)
configs['A: All real QK + phi_softmax'] = {i: attn_real_qk_phi_bf16 for i in range(N_LAYERS)}

# Config B: All bias-aware (from 10b, 0/15 stacked)
configs['B: All bias-aware (10b)'] = {i: attn_bias_aware for i in range(N_LAYERS)}

# Config C: Anchor L0-3,27 real + bias-aware L4-26
cfg_c = {}
for i in range(4):
    cfg_c[i] = attn_real_qk_phi_bf16
cfg_c[27] = attn_real_qk_phi_bf16
for i in range(4, 27):
    cfg_c[i] = attn_bias_aware
configs['C: Anchor(0-3,27) + bias-aware(4-26)'] = cfg_c

# Config D: Anchor L0-3,27 real + scaffold+deviation L4-26
cfg_d = {}
for i in range(4):
    cfg_d[i] = attn_real_qk_phi_bf16
cfg_d[27] = attn_real_qk_phi_bf16
for i in range(4, 27):
    cfg_d[i] = scaffold_full
configs['D: Anchor(0-3,27) + scaffold+dev(4-26)'] = cfg_d

# Config E: Anchor L0-3,27 real + scaffold+sign(dev) L4-26
cfg_e = {}
for i in range(4):
    cfg_e[i] = attn_real_qk_phi_bf16
cfg_e[27] = attn_real_qk_phi_bf16
for i in range(4, 27):
    cfg_e[i] = scaffold_sign
configs['E: Anchor(0-3,27) + scaffold+sign(4-26)'] = cfg_e

# Config F: Only anchor L0,27 + scaffold rest
cfg_f = {}
cfg_f[0] = attn_real_qk_phi_bf16
cfg_f[27] = attn_real_qk_phi_bf16
for i in range(1, 27):
    cfg_f[i] = scaffold_full
configs['F: Anchor(0,27) + scaffold+dev(1-26)'] = cfg_f

# Config G: Anchor L0-3,27 + scaffold L4-26 (NO deviation, pure scaffold)
def make_pure_scaffold_attn():
    """Scaffold scores only — no deviation correction."""
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
        for h in routing:
            scaffold = scaffold_scores.get((layer_idx, h, seq_len))
            if scaffold is not None:
                scores = scaffold.clone()
            else:
                # Fallback
                tbl = head_tables[(layer_idx, h)]
                h_float = h_normed[0].float().cpu()
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
    return attn_fn

cfg_g = {}
for i in range(4):
    cfg_g[i] = attn_real_qk_phi_bf16
cfg_g[27] = attn_real_qk_phi_bf16
pure_scaffold = make_pure_scaffold_attn()
for i in range(4, 27):
    cfg_g[i] = pure_scaffold
configs['G: Anchor(0-3,27) + pure scaffold(4-26)'] = cfg_g

print(f"  {'Config':>55s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 75)

all_results = {}
for name, cfg in configs.items():
    n, t, c, fails = evaluate(name, cfg)
    pct = n / t * 100
    print(f"  {name:>55s}  {n:2d}/{t:2d}    {c:.4f}")
    if fails and n < 13:
        for fi, fp, fb, fg, m in fails[:3]:
            print(f"    FAIL: \"{fp}\" base='{fb}' got='{fg}' margin={m:.3f}")
    all_results[name] = {'n': n, 'total': t, 'cos': c, 'pct': pct}

print()

# ================================================================
# DIAGNOSTIC: How much does deviation matter vs scaffold?
# ================================================================
print("=" * 80)
print("  DIAGNOSTIC: Scaffold vs Deviation Contribution")
print("=" * 80)
print()

# For test prompts, measure the magnitude of deviation vs scaffold
dev_magnitudes = []
scaffold_magnitudes = []

for pi, prompt in enumerate(TEST_PROMPTS[:5]):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    layer_h = {}
    def cap_hook(li):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is not None:
                layer_h[li] = h[0].cpu().float()
            return output
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            cap_hook(li), with_kwargs=True))
    with torch.no_grad():
        model(ids, return_dict=True)
    for hk in hooks:
        hk.remove()

    for li in range(4, 27):
        if li not in layer_h:
            continue
        h_actual = layer_h[li]
        s = h_actual.shape[0]
        mean_h = mean_hidden_per_layer.get(li, {}).get(s, {}).get('mean')
        if mean_h is None:
            continue
        delta_h = h_actual - mean_h
        delta_norm = delta_h.norm().item()
        mean_norm = mean_h.norm().item()
        dev_magnitudes.append(delta_norm / max(mean_norm, 1e-10))

if dev_magnitudes:
    print(f"  Deviation / mean norm ratio (COMB layers):")
    print(f"    mean={np.mean(dev_magnitudes):.4f}  min={np.min(dev_magnitudes):.4f}  max={np.max(dev_magnitudes):.4f}")
    print(f"    → Deviation is {np.mean(dev_magnitudes):.1%} of mean trajectory")
print()


# Save
save_data = {
    'configs': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    'deviation_ratio': {
        'mean': float(np.mean(dev_magnitudes)) if dev_magnitudes else None,
        'min': float(np.min(dev_magnitudes)) if dev_magnitudes else None,
        'max': float(np.max(dev_magnitudes)) if dev_magnitudes else None,
    },
}
save_path = os.path.join(results_dir, 'phase10d_scaffold.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved to {save_path}")
print()
print("=" * 80)
print("  DONE — Phase 10d Scaffold Trajectory")
print("=" * 80)
