#!/usr/bin/env python3
"""
Phase 10a: QK Replacement Exploration
======================================

Three approaches to replacing the full QK matmul with geometric alternatives:

APPROACH 1 — RoPE-Aware Rank-1 Factorization
  Since MESH = σ₁ u₁ v₁ᵀ (rank-1, 368K:1), after RoPE the score factorizes:
    score(i,j) ≈ content_q(i) × content_k(j) × position_factor(j-i)
  Where position_factor(δ) = u₁ᵀ R(δ) v₁ is PRE-COMPUTABLE.
  Cost: O(seq × hidden_dim + seq²) vs O(seq² × head_dim).

APPROACH 2 — Content + Position Split
  Use d_k content score + geometric position bias:
    a) d_k + ALiBi-style linear decay
    b) d_k + log-φ decay
    c) d_k + learned position_factor from Approach 1

APPROACH 3 — Weighted V for Fixed Heads
  Instead of hard V[0], use exponential-decay weighted V mix:
    V_out = Σ w(i) V(i) where w(i) ∝ φ^(-λ·i)

Also: detailed diagnostics for each — per-layer agreement, score correlations,
position selection accuracy. We're in uncharted territory; observe everything.

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
print("  PHASE 10a: QK REPLACEMENT EXPLORATION")
print("  Three approaches to geometric attention without full QK matmul")
print("=" * 80)
print()

# ================================================================
# Setup
# ================================================================
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

print(f"  Model loaded: {N_LAYERS} layers, {NUM_HEADS} heads, head_dim={HEAD_DIM}")
print()


# ================================================================
# Helpers
# ================================================================
def phi_softmax_torch(scores, dim=-1):
    """φ-basis softmax: exact equivalent of standard softmax."""
    s = scores - scores.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)


def apply_rotary_pos_emb(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


def get_rope_cache(seq_len, device, dtype):
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM
    ))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def rope_rotate_vector(v, cos, sin):
    """Apply RoPE rotation to a single vector v (head_dim,)."""
    v1 = v[: len(v) // 2]
    v2 = v[len(v) // 2 :]
    rotated = torch.cat((-v2, v1))
    return v * cos + rotated * sin


# ================================================================
# Extract MESH SVD components for all routing heads
# ================================================================
print("Extracting MESH SVD components (u₁, v₁, σ₁, d_q, d_k) for all routing heads...")
mesh_data = {}  # {layer: {head: {u1, v1, sigma1, d_q, d_k}}}

for layer_idx in range(N_LAYERS):
    routing = layer_classification[layer_idx]['routing']
    if not routing:
        continue
    mesh_data[layer_idx] = {}
    attn = model.model.layers[layer_idx].self_attn

    # Extract W_q, W_k via identity probing
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    W_q_heads = {h: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for h in routing}
    needed_kv = set(h // HEADS_PER_KV for h in routing)
    W_k_groups = {g: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for g in needed_kv}

    for s in range(0, HIDDEN_DIM, 512):
        e = min(s + 512, HIDDEN_DIM)
        chunk = identity[s:e].unsqueeze(0)
        with torch.no_grad():
            qo = attn.q_proj(chunk).float()
            ko = attn.k_proj(chunk).float()
        qr = qo[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = ko[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in routing:
            W_q_heads[h][:, s:e] = qr[:, h, :].T
        for g in needed_kv:
            W_k_groups[g][:, s:e] = kr[:, g, :].T

    for h in routing:
        g = h // HEADS_PER_KV
        MESH = W_q_heads[h] @ W_k_groups[g].T  # (HEAD_DIM, HEAD_DIM)
        U, S, Vt = torch.linalg.svd(MESH)
        u1 = U[:, 0]      # (HEAD_DIM,) — dominant left singular vector
        v1 = Vt[0, :]     # (HEAD_DIM,) — dominant right singular vector
        sigma1 = S[0]     # dominant singular value
        ratio = (S[0] / S[1]).item() if S[1] > 0 else float('inf')

        # Hidden-space directions
        d_q = (W_q_heads[h].T @ u1)  # (HIDDEN_DIM,)
        d_k = (W_k_groups[g].T @ v1) # (HIDDEN_DIM,)

        mesh_data[layer_idx][h] = {
            'u1': u1.cpu(),
            'v1': v1.cpu(),
            'sigma1': sigma1.item(),
            'd_q': d_q.cpu(),
            'd_k': d_k.cpu(),
            'ratio': ratio,
        }

    del W_q_heads, W_k_groups
    torch.cuda.empty_cache()

n_routing_total = sum(len(v) for v in mesh_data.values())
print(f"  Extracted {n_routing_total} routing heads across {len(mesh_data)} layers")
print()


# ================================================================
# Pre-compute position_factor tables for Approach 1
# ================================================================
print("Pre-computing position_factor tables (u₁ᵀ R(δ) v₁ for all δ)...")
MAX_SEQ = 64  # sufficient for our test prompts

position_factors = {}  # {layer: {head: tensor of shape (MAX_SEQ,)}}

inv_freq = 1.0 / (ROPE_THETA ** (
    torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM
))

for layer_idx in mesh_data:
    position_factors[layer_idx] = {}
    for h in mesh_data[layer_idx]:
        md = mesh_data[layer_idx][h]
        u1 = md['u1']
        v1 = md['v1']
        sigma1 = md['sigma1']

        pf = torch.zeros(MAX_SEQ)
        for delta in range(MAX_SEQ):
            # R(delta) rotation applied to v1
            freqs = delta * inv_freq
            cos_d = torch.cat((freqs.cos(), freqs.cos()))
            sin_d = torch.cat((freqs.sin(), freqs.sin()))
            v1_rotated = rope_rotate_vector(v1, cos_d, sin_d)
            pf[delta] = u1 @ v1_rotated

        position_factors[layer_idx][h] = pf

# Quick diagnostic: how does position_factor decay?
sample_layer = list(position_factors.keys())[len(position_factors) // 2]
sample_head = list(position_factors[sample_layer].keys())[0]
sample_pf = position_factors[sample_layer][sample_head]
print(f"  Sample position_factor (L{sample_layer} H{sample_head}):")
print(f"    δ=0: {sample_pf[0]:.6f}  δ=1: {sample_pf[1]:.6f}  δ=2: {sample_pf[2]:.6f}")
print(f"    δ=5: {sample_pf[5]:.6f}  δ=10: {sample_pf[10]:.6f}  δ=20: {sample_pf[20]:.6f}")
pf_ratio = (sample_pf[1] / sample_pf[0]).item() if sample_pf[0] != 0 else 0
print(f"    Ratio pf[1]/pf[0] = {pf_ratio:.6f}")
print()


# ================================================================
# DIAGNOSTIC 1: Score correlation analysis
# How well does each approach's scores correlate with real QK scores?
# ================================================================
print("=" * 80)
print("  DIAGNOSTIC: Score Correlation with Real Attention")
print("=" * 80)
print()

DIAG_PROMPTS = [
    "The capital of France is",
    "Albert Einstein developed the theory of",
    "To be or not to",
    "The largest planet in our solar system is",
    "Once upon a time in a land",
]

score_correlations = defaultdict(lambda: defaultdict(list))

for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    # Hook to capture real attention scores
    real_scores_by_layer = {}

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

            layer_scores = {}
            for hd in range(NUM_HEADS):
                sc = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HEAD_DIM)).float()
                layer_scores[hd] = sc.cpu()

            # Also capture h_normed for our approaches
            real_scores_by_layer[layer_idx] = {
                'scores': layer_scores,
                'h': h[0].cpu().float(),
            }
            return output
        return hook_fn

    hooks = []
    for li in range(N_LAYERS):
        if li in mesh_data:
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(
                capture_hook(li), with_kwargs=True))
    with torch.no_grad():
        model(ids, return_dict=True)
    for hk in hooks:
        hk.remove()

    # Now compare approaches for each layer/head
    for li in real_scores_by_layer:
        if li not in mesh_data:
            continue
        h_states = real_scores_by_layer[li]['h']  # (seq_len, HIDDEN_DIM)
        s = h_states.shape[0]

        for hd in mesh_data[li]:
            real_sc = real_scores_by_layer[li]['scores'][hd]  # (s, s)
            md = mesh_data[li][hd]

            # --- Approach 1: Factorized scores ---
            content_q = h_states @ md['d_q']  # (s,)
            content_k = h_states @ md['d_k']  # (s,)
            pf = position_factors[li][hd]

            factored_scores = torch.zeros(s, s)
            for i in range(s):
                for j in range(i + 1):
                    delta = i - j
                    factored_scores[i, j] = content_q[i] * content_k[j] * pf[delta]

            # Scale to match real scores (the sigma and sqrt(d) factors)
            # Find best scale via regression on valid (non-masked) entries
            mask = torch.tril(torch.ones(s, s)).bool()
            real_vals = real_sc[mask]
            fact_vals = factored_scores[mask]

            if fact_vals.std() > 0:
                # Pearson correlation
                corr_1 = torch.corrcoef(torch.stack([real_vals, fact_vals]))[0, 1].item()
            else:
                corr_1 = 0.0

            # --- Approach 2a: d_k only (no position) ---
            dk_scores = torch.zeros(s, s)
            for i in range(s):
                for j in range(i + 1):
                    dk_scores[i, j] = content_k[j]  # content only, no position
            dk_vals = dk_scores[mask]
            if dk_vals.std() > 0:
                corr_2a = torch.corrcoef(torch.stack([real_vals, dk_vals]))[0, 1].item()
            else:
                corr_2a = 0.0

            # --- Approach 2b: d_k + ALiBi-style linear decay ---
            alibi_scores = torch.zeros(s, s)
            for i in range(s):
                for j in range(i + 1):
                    alibi_scores[i, j] = content_k[j] - 0.1 * (i - j)
            alibi_vals = alibi_scores[mask]
            if alibi_vals.std() > 0:
                corr_2b = torch.corrcoef(torch.stack([real_vals, alibi_vals]))[0, 1].item()
            else:
                corr_2b = 0.0

            # --- Approach 2c: d_k + log-φ decay ---
            phi_decay_scores = torch.zeros(s, s)
            for i in range(s):
                for j in range(i + 1):
                    phi_decay_scores[i, j] = content_k[j] * PHI ** (-(i - j) * 0.1)
            phi_vals = phi_decay_scores[mask]
            if phi_vals.std() > 0:
                corr_2c = torch.corrcoef(torch.stack([real_vals, phi_vals]))[0, 1].item()
            else:
                corr_2c = 0.0

            score_correlations['approach1_factored'][li].append(corr_1)
            score_correlations['approach2a_dk_only'][li].append(corr_2a)
            score_correlations['approach2b_dk_alibi'][li].append(corr_2b)
            score_correlations['approach2c_dk_phi_decay'][li].append(corr_2c)

# Print correlation summary
print(f"  {'Approach':>35s}  {'Mean corr':>9s}  {'Min':>7s}  {'Max':>7s}")
print("  " + "-" * 65)
for approach_name in ['approach1_factored', 'approach2a_dk_only',
                       'approach2b_dk_alibi', 'approach2c_dk_phi_decay']:
    all_corrs = []
    for li in score_correlations[approach_name]:
        all_corrs.extend(score_correlations[approach_name][li])
    if all_corrs:
        m = np.mean(all_corrs)
        mn = np.min(all_corrs)
        mx = np.max(all_corrs)
        print(f"  {approach_name:>35s}  {m:>9.4f}  {mn:>7.4f}  {mx:>7.4f}")

# Per-layer breakdown for approach 1
print()
print("  Approach 1 (factorized) per-layer mean correlation:")
for li in sorted(score_correlations['approach1_factored'].keys()):
    vals = score_correlations['approach1_factored'][li]
    print(f"    L{li:2d}: {np.mean(vals):.4f} (n={len(vals)} heads × {len(DIAG_PROMPTS)} prompts)")

print()


# ================================================================
# APPROACH 1: Full end-to-end test — RoPE-aware factorized attention
# ================================================================
print("=" * 80)
print("  APPROACH 1: RoPE-Aware Rank-1 Factorized Attention")
print("=" * 80)
print()

def attn_factored_rank1(layer_idx, h_normed, attn_module):
    """
    Replace QK scores with factorized rank-1 scores:
      score(i,j) = content_q(i) × content_k(j) × position_factor(i-j)
    Then use phi_softmax + standard V/O.
    """
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']

    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)

    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)

    # Fixed heads → V[0]
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]

    # Routing heads → factorized scores + phi_softmax
    h_float = h_normed[0].float().cpu()
    for h in routing:
        md = mesh_data[layer_idx][h]
        pf = position_factors[layer_idx][h]

        content_q = h_float @ md['d_q']  # (seq_len,)
        content_k = h_float @ md['d_k']  # (seq_len,)

        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                scores[i, j] = content_q[i] * content_k[j] * pf[i - j]

        # Scale to approximate real QK magnitude
        scale = md['sigma1'] / math.sqrt(HEAD_DIM)
        scores = scores * scale

        # Causal mask
        scores = scores.to(h_normed.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# APPROACH 2c: d_k content + position_factor from Approach 1
# (Separates content and position concerns)
# ================================================================
def attn_dk_with_position_factor(layer_idx, h_normed, attn_module):
    """
    d_k content score PLUS the geometric position factor.
    score(i,j) = content_k(j) × position_factor(i-j)
    (drops content_q — is the query direction even needed?)
    """
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
        md = mesh_data[layer_idx][h]
        pf = position_factors[layer_idx][h]
        content_k = h_float @ md['d_k']

        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                scores[i, j] = content_k[j] * pf[i - j]

        scores = scores.to(h_normed.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# APPROACH 3: Weighted V for fixed heads
# ================================================================
def attn_factored_with_weighted_v(layer_idx, h_normed, attn_module):
    """
    Approach 1 for routing heads + φ-decay weighted V for fixed heads.
    Fixed heads: V_out = Σ φ^(-λ·i) V(i) / Σ φ^(-λ·i)  (recency-weighted)
    """
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']

    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_exp = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)

    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)

    # Fixed heads: recency-weighted V (instead of hard V[0])
    for h in fixed:
        for i in range(seq_len):
            if i == 0:
                attn_out[0, i, h, :] = V_exp[0, 0, h, :]
            else:
                # Exponential recency: most weight on pos 0, some on recent
                positions = torch.arange(i + 1, device=h_normed.device, dtype=torch.float32)
                # Weight: high for pos 0, decaying for others
                w = torch.zeros(i + 1, device=h_normed.device)
                w[0] = 1.0  # strong pos-0 bias
                # Small uniform contribution from all positions
                w += 0.05
                w = w / w.sum()
                attn_out[0, i, h, :] = (w.unsqueeze(-1) * V_exp[0, :i+1, h, :].float()).sum(dim=0).to(h_normed.dtype)

    # Routing heads: factorized (same as Approach 1)
    h_float = h_normed[0].float().cpu()
    for h in routing:
        md = mesh_data[layer_idx][h]
        pf = position_factors[layer_idx][h]
        content_q = h_float @ md['d_q']
        content_k = h_float @ md['d_k']

        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                scores[i, j] = content_q[i] * content_k[j] * pf[i - j]

        scale = md['sigma1'] / math.sqrt(HEAD_DIM)
        scores = scores * scale
        scores = scores.to(h_normed.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        weights = phi_softmax_torch(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)

    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# Reference: full QK + phi_softmax (from F89, bf16 matched)
# ================================================================
def attn_full_qk_phi_bf16(layer_idx, h_normed, attn_module):
    """Full QK in bf16 + phi_softmax. The known-good baseline (59/60)."""
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


# ================================================================
# Runner (same hook infrastructure as phase9d)
# ================================================================
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
# Test prompts
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

# Baselines
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


def evaluate_config(name, attn_fn_map):
    n_match = 0
    cos_list = []
    fails = []
    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits = run_with_hooks(ids, attn_fn_map)
        gl = logits[0, -1, :].float()
        gid = gl.argmax().item()
        match = gid == baseline_tokens[pi]
        if match:
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
print("  SHOOTOUT: All Approaches (15 prompts)")
print("=" * 80)
print()

configs = {
    'Baseline: Full QK + phi_softmax (bf16)': {i: attn_full_qk_phi_bf16 for i in range(N_LAYERS)},
    'Approach 1: Factorized rank-1 (all layers)': {i: attn_factored_rank1 for i in range(N_LAYERS)},
    'Approach 2: d_k + position_factor (all layers)': {i: attn_dk_with_position_factor for i in range(N_LAYERS)},
    'Approach 3: Factored + weighted V (all layers)': {i: attn_factored_with_weighted_v for i in range(N_LAYERS)},
}

all_results = {}
print(f"  {'Config':>55s}  {'Score':>7s}  {'Cos':>7s}")
print("  " + "-" * 75)

for name, cfg in configs.items():
    n, t, c, fails = evaluate_config(name, cfg)
    pct = n / t * 100
    print(f"  {name:>55s}  {n:2d}/{t:2d}    {c:.4f}")
    if fails:
        for fi, fp, fb, fg, m in fails:
            print(f"    FAIL: \"{fp}\" base='{fb}' got='{fg}' margin={m:.3f}")
    all_results[name] = {'n': n, 'total': t, 'cos': c, 'pct': pct, 'fails': [(f[0], f[1]) for f in fails]}

print()

# ================================================================
# Per-layer ablation for best non-baseline approach
# ================================================================
print("=" * 80)
print("  PER-LAYER ABLATION: Factorized Rank-1 (one layer at a time)")
print("=" * 80)
print()

per_layer_results = {}
for li in range(N_LAYERS):
    cfg = {li: attn_factored_rank1}
    n, t, c, fails = evaluate_config(f"L{li}", cfg)
    status = "✓" if n == t else f"FAIL({n}/{t})"
    n_routing = len(layer_classification[li]['routing'])
    per_layer_results[li] = {'n': n, 'cos': c, 'routing': n_routing}
    if n < t:
        print(f"  L{li:2d}: {n:2d}/{t:2d}  cos={c:.4f}  routing_heads={n_routing:2d}  ← {status}")
    else:
        print(f"  L{li:2d}: {n:2d}/{t:2d}  cos={c:.4f}  routing_heads={n_routing:2d}")

n_perfect = sum(1 for v in per_layer_results.values() if v['n'] == 15)
print(f"\n  Perfect layers: {n_perfect}/28")
print()

# ================================================================
# OBSERVATION NOTES
# ================================================================
print("=" * 80)
print("  OBSERVATIONS & NOTES")
print("=" * 80)
print()

# 1. Position factor structure
print("  1. Position factor decay patterns:")
for li in sorted(list(position_factors.keys())[:5]):
    for h in sorted(list(position_factors[li].keys())[:2]):
        pf = position_factors[li][h]
        vals = [f"{pf[d]:.4f}" for d in [0, 1, 2, 5, 10]]
        print(f"     L{li:2d} H{h:2d}: δ=[0,1,2,5,10] → [{', '.join(vals)}]")
print()

# 2. Does content_q matter? (Approach 1 vs 2)
if 'Approach 1: Factorized rank-1 (all layers)' in all_results and \
   'Approach 2: d_k + position_factor (all layers)' in all_results:
    a1 = all_results['Approach 1: Factorized rank-1 (all layers)']
    a2 = all_results['Approach 2: d_k + position_factor (all layers)']
    print(f"  2. Does content_q matter?")
    print(f"     Approach 1 (with content_q): {a1['n']}/{a1['total']} cos={a1['cos']:.4f}")
    print(f"     Approach 2 (without content_q): {a2['n']}/{a2['total']} cos={a2['cos']:.4f}")
    if a1['n'] > a2['n']:
        print(f"     → YES, content_q adds {a1['n'] - a2['n']} correct predictions")
    elif a1['n'] == a2['n']:
        print(f"     → Same accuracy. content_q may not matter for top-1.")
    else:
        print(f"     → Surprisingly, content_q HURTS ({a2['n'] - a1['n']} more correct without it)")
    print()

# 3. Rank-1 ratio correlation with factored score quality
print(f"  3. Does higher rank-1 ratio → better factored scores?")
for li in sorted(score_correlations['approach1_factored'].keys())[:6]:
    corrs = score_correlations['approach1_factored'][li]
    heads = list(mesh_data[li].keys())
    ratios = [mesh_data[li][h]['ratio'] for h in heads]
    mean_ratio = np.mean(ratios)
    mean_corr = np.mean(corrs)
    print(f"     L{li:2d}: mean_ratio={mean_ratio:.0f}  mean_corr={mean_corr:.4f}")
print()

# Save results
save_path = os.path.join(results_dir, 'phase10a_qk_replacement.json')
save_data = {
    'shootout': {k: {kk: vv for kk, vv in v.items() if kk != 'fails'} for k, v in all_results.items()},
    'per_layer': per_layer_results,
    'score_correlations': {
        approach: {str(li): float(np.mean(vals)) for li, vals in layers.items()}
        for approach, layers in score_correlations.items()
    },
}
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Results saved to {save_path}")

print()
print("=" * 80)
print("  DONE — Phase 10a QK Replacement Exploration")
print("=" * 80)
