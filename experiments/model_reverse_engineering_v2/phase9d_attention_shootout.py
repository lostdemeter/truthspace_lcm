#!/usr/bin/env python3
"""
Phase 9d: Attention Replacement Shootout
=========================================

Test ALL candidate approaches for geometric attention replacement:

A. Hybrid: fixed heads → V[0], routing heads → full QK+RoPE+phi_softmax
   (saves 61% of QK compute, keeps position-aware scores for routing)

B. Skip L0: real attention at L0, d_k argmax for L1-27
   (L0 is catastrophic — does skipping it fix the cascade?)

C. Skip L0 + phi_softmax: real attention at L0, phi_softmax d_k for L1-27

D. Skip DRUM (L0-3): real attention for early layers, d_k for L4-27

E. Full phi_softmax QK: phi_softmax replaces softmax everywhere
   (mathematically identical — sanity check + establishes geometric pipeline)

F. Graduated: real attn L0, phi_softmax QK routing+fixed→V[0] for L1-27
   (combines best of A and B)

All use phi_softmax where applicable to stay in φ-basis.
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
print("  PHASE 9d: ATTENTION REPLACEMENT SHOOTOUT")
print("=" * 80)
print()

# Load phase9a classification
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
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",
)
model.eval()

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7; HIDDEN_DIM = 3584

print(f"  {N_LAYERS} layers, {NUM_HEADS} heads")
print()


# ================================================================
# phi_softmax
# ================================================================
def phi_softmax_torch(scores, dim=-1):
    """φ-basis softmax: φ^(x/ln(φ)) / Σ = e^x / Σ e^x. Exact."""
    scores_shifted = scores - scores.max(dim=dim, keepdim=True).values
    phi_powers = PHI ** (scores_shifted / LOG_PHI)
    return phi_powers / phi_powers.sum(dim=dim, keepdim=True)


# ================================================================
# Extract d_k for all routing heads
# ================================================================
print("Extracting d_k routing directions...")
d_k_vectors = {}

for layer_idx in range(N_LAYERS):
    routing = layer_classification[layer_idx]['routing']
    if not routing:
        continue
    d_k_vectors[layer_idx] = {}
    attn = model.model.layers[layer_idx].self_attn
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    W_q = {h: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for h in routing}
    needed_kv = set(h // HEADS_PER_KV for h in routing)
    W_k = {g: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for g in needed_kv}
    for s in range(0, HIDDEN_DIM, 512):
        e = min(s + 512, HIDDEN_DIM)
        c = identity[s:e].unsqueeze(0)
        with torch.no_grad():
            qo = attn.q_proj(c).float()
            ko = attn.k_proj(c).float()
        qr = qo[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = ko[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in routing: W_q[h][:, s:e] = qr[:, h, :].T
        for g in needed_kv: W_k[g][:, s:e] = kr[:, g, :].T
    for h in routing:
        g = h // HEADS_PER_KV
        M = W_q[h] @ W_k[g].T
        _, _, Vt = torch.linalg.svd(M)
        d_k_vectors[layer_idx][h] = (W_k[g].T @ Vt[0]).cpu()
    del W_q, W_k
    torch.cuda.empty_cache()

print("  Done.")
print()


# ================================================================
# RoPE helper (matches Qwen2's implementation)
# ================================================================
def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def get_rope_cache(seq_len, head_dim, device, dtype):
    """Compute RoPE cos/sin cache."""
    # Qwen2 uses default rope_theta=1000000
    rope_theta = 1000000.0
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos().to(dtype)
    sin_cache = emb.sin().to(dtype)
    return cos_cache.unsqueeze(0).unsqueeze(0), sin_cache.unsqueeze(0).unsqueeze(0)


# ================================================================
# Geometric attention variants
# ================================================================

def attn_dk_argmax(layer_idx, h_normed, attn_module):
    """Hard argmax d_k routing (Finding 84 baseline)."""
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
        dk = d_k_vectors[layer_idx][h].to(h_normed.device, dtype=torch.float32)
        scores = h_normed[0].float() @ dk
        for i in range(seq_len):
            sel = scores[:i+1].argmax()
            attn_out[0, i, h, :] = V_exp[0, sel, h, :]

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_dk_phi_softmax(layer_idx, h_normed, attn_module):
    """phi_softmax d_k routing."""
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
        dk = d_k_vectors[layer_idx][h].to(h_normed.device, dtype=torch.float32)
        scores = h_normed[0].float() @ dk
        for i in range(seq_len):
            w = phi_softmax_torch(scores[:i+1], dim=0)
            weighted_v = w.to(h_normed.dtype).unsqueeze(-1) * V_exp[0, :i+1, h, :]
            attn_out[0, i, h, :] = weighted_v.sum(dim=0)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_hybrid_qk_phi(layer_idx, h_normed, attn_module):
    """
    Hybrid: fixed heads → V[0], routing heads → full QK+RoPE+phi_softmax.
    Saves 61% of QK compute, keeps position-aware scores for routing.
    """
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']

    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).float()
        K = attn_module.k_proj(h_normed).float()
        V_full = attn_module.v_proj(h_normed)

    # Reshape
    Q = Q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)

    # Apply RoPE
    cos, sin = get_rope_cache(seq_len, HEAD_DIM, h_normed.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, cos, sin)
    K = apply_rotary_pos_emb(K, cos, sin)

    # Expand K for GQA
    K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)

    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)

    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]

    for h in routing:
        # Full QK scores for this head
        q_h = Q[0, h, :, :]  # (seq_len, HEAD_DIM)
        k_h = K_exp[0, h, :, :]  # (seq_len, HEAD_DIM)
        scores = q_h @ k_h.T / math.sqrt(HEAD_DIM)  # (seq_len, seq_len)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        # phi_softmax
        weights = phi_softmax_torch(scores, dim=-1)  # (seq_len, seq_len)

        # Weighted V
        v_h = V_exp[0, :, h, :].float()  # (seq_len, HEAD_DIM)
        out_h = weights @ v_h  # (seq_len, HEAD_DIM)
        attn_out[0, :, h, :] = out_h.to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_full_phi_softmax(layer_idx, h_normed, attn_module):
    """
    Full attention with phi_softmax replacing softmax.
    ALL heads use QK+RoPE+phi_softmax. Sanity check (should match baseline).
    """
    batch, seq_len, _ = h_normed.shape

    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).float()
        K = attn_module.k_proj(h_normed).float()
        V_full = attn_module.v_proj(h_normed)

    Q = Q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)

    cos, sin = get_rope_cache(seq_len, HEAD_DIM, h_normed.device, Q.dtype)
    Q = apply_rotary_pos_emb(Q, cos, sin)
    K = apply_rotary_pos_emb(K, cos, sin)
    K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)

    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)

    for h in range(NUM_HEADS):
        q_h = Q[0, h, :, :]
        k_h = K_exp[0, h, :, :]
        scores = q_h @ k_h.T / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax_torch(scores, dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# Runner: apply a strategy via hooks
# ================================================================

def run_with_hooks(input_ids, attn_fn_map):
    """
    Run model with geometric attention at specified layers.
    attn_fn_map: {layer_idx: attn_function} or 'all' for every layer.
    Layers not in map use standard attention.
    """
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
            make_hook(layer_idx, attn_fn), with_kwargs=True
        )
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
# Test prompts & baseline
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
baseline_ids = []
for p in TEST_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    baseline_ids.append(out.logits[0, -1, :].float().argmax().item())
print(f"  {len(TEST_PROMPTS)} baselines ready.")
print()


def evaluate_config(name, attn_fn_map):
    """Run all test prompts, return accuracy."""
    n_match = 0
    cos_list = []
    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits = run_with_hooks(ids, attn_fn_map)
        gl = logits[0, -1, :].float()
        gid = gl.argmax().item()
        if gid == baseline_ids[pi]:
            n_match += 1
        # Quick cos sim with baseline
        ids2 = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            bl = model(ids2, return_dict=True).logits[0, -1, :].float()
        cos = F.cosine_similarity(bl.cpu().unsqueeze(0), gl.cpu().unsqueeze(0)).item()
        cos_list.append(cos)
    return n_match, len(TEST_PROMPTS), float(np.mean(cos_list))


# ================================================================
# SHOOTOUT
# ================================================================
print("=" * 80)
print("  SHOOTOUT: Testing all approaches")
print("=" * 80)
print()
print(f"  {'Config':>60s}  {'Score':>7s}  {'Cos':>7s}  {'Layers replaced':>15s}")
print("  " + "-" * 95)

results = {}

# E. Full phi_softmax QK (sanity check — should match baseline)
config_E = {i: attn_full_phi_softmax for i in range(N_LAYERS)}
n, t, c = evaluate_config("E", config_E)
print(f"  {'E: Full QK + phi_softmax (sanity)':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'28/28':>15s}")
results['E_full_phi_softmax'] = {'accuracy': n/t, 'cos': c, 'n': n}

# A. Hybrid: fixed→V[0], routing→QK+RoPE+phi_softmax
config_A = {i: attn_hybrid_qk_phi for i in range(N_LAYERS)}
n, t, c = evaluate_config("A", config_A)
print(f"  {'A: Hybrid (fixed→V[0], route→QK+phi_soft)':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'28/28':>15s}")
results['A_hybrid_qk_phi'] = {'accuracy': n/t, 'cos': c, 'n': n}

# B. Skip L0, d_k argmax L1-27
config_B = {i: attn_dk_argmax for i in range(1, N_LAYERS)}
n, t, c = evaluate_config("B", config_B)
print(f"  {'B: Real L0, d_k argmax L1-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'27/28':>15s}")
results['B_skip_L0_argmax'] = {'accuracy': n/t, 'cos': c, 'n': n}

# C. Skip L0, phi_softmax d_k L1-27
config_C = {i: attn_dk_phi_softmax for i in range(1, N_LAYERS)}
n, t, c = evaluate_config("C", config_C)
print(f"  {'C: Real L0, phi_softmax d_k L1-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'27/28':>15s}")
results['C_skip_L0_phi_dk'] = {'accuracy': n/t, 'cos': c, 'n': n}

# D. Skip DRUM (L0-3), d_k argmax L4-27
config_D = {i: attn_dk_argmax for i in range(4, N_LAYERS)}
n, t, c = evaluate_config("D", config_D)
print(f"  {'D: Real L0-3, d_k argmax L4-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'24/28':>15s}")
results['D_skip_DRUM_argmax'] = {'accuracy': n/t, 'cos': c, 'n': n}

# D2. Skip DRUM (L0-3), phi_softmax d_k L4-27
config_D2 = {i: attn_dk_phi_softmax for i in range(4, N_LAYERS)}
n, t, c = evaluate_config("D2", config_D2)
print(f"  {'D2: Real L0-3, phi_softmax d_k L4-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'24/28':>15s}")
results['D2_skip_DRUM_phi_dk'] = {'accuracy': n/t, 'cos': c, 'n': n}

# F. Real L0, hybrid QK+phi for L1-27
config_F = {i: attn_hybrid_qk_phi for i in range(1, N_LAYERS)}
n, t, c = evaluate_config("F", config_F)
print(f"  {'F: Real L0, hybrid (fix→V[0], route→QK+phi) L1-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'27/28':>15s}")
results['F_real_L0_hybrid'] = {'accuracy': n/t, 'cos': c, 'n': n}

# G. Real L0, hybrid QK+phi L1-27, but routing heads also skip QK
# (most aggressive geometric: only V projection for all heads, route by d_k)
# This was already tested as phi_softmax d_k — skip

# H. Real L0-3, hybrid for L4-27
config_H = {i: attn_hybrid_qk_phi for i in range(4, N_LAYERS)}
n, t, c = evaluate_config("H", config_H)
print(f"  {'H: Real L0-3, hybrid L4-27':>60s}  {n:2d}/{t:2d}    {c:.4f}  {'24/28':>15s}")
results['H_real_DRUM_hybrid'] = {'accuracy': n/t, 'cos': c, 'n': n}

print()
print("=" * 80)
print("  RESULTS SUMMARY")
print("=" * 80)
print()

# Sort by accuracy then cos
sorted_results = sorted(results.items(), key=lambda x: (x[1]['accuracy'], x[1]['cos']), reverse=True)
for name, r in sorted_results:
    print(f"  {name:>30s}: {r['n']:2d}/15 ({r['accuracy']:.0%})  cos={r['cos']:.4f}")

# Save
save_path = os.path.join(results_dir, 'phase9d_shootout.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved to {save_path}")
