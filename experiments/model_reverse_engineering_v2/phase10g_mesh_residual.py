#!/usr/bin/env python3
"""
Phase 10g: MESH Residual Decomposition — Finding the True Null Space
=====================================================================

From 10f we learned:
  - The weight-weight term is 9.3% of cross-terms (NOT negligible)
  - Rank-1 MESH SVD has -0.70 correlation with ww residual (WRONG direction)
  - The rank-1 direction captures the BIAS structure, not the correction

The fix: decompose the MESH RESIDUAL after removing bias contribution.

The full MESH is: M(δ) = W_q^T R(δ) W_k   (HEAD_DIM × HEAD_DIM in hidden space)

But the score is: h(i)^T M(δ) h(j) / sqrt(d_k)

The bias-aware decomposition handles:
  score ≈ baseline(δ) + h(i)·c_q(δ) + c_k(δ)·h(j)

The residual is:
  ww_term = h(i)^T [W_q^T R(δ) W_k] h(j) / sqrt(d_k)

This is a bilinear form in hidden space. We want to find the dominant
directions of W_q^T R(δ) W_k in hidden space (not head space).

Approach:
  1. Compute M_hidden(δ) = W_q^T R(δ) W_k  (HIDDEN_DIM × HIDDEN_DIM — too big!)
     Actually: h^T W_q^T R W_k h = (W_q h)^T R (W_k h)
     So we need SVD of W_q^T (in hidden→head) and W_k (in hidden→head)
  
  2. Better: The bilinear form h_i^T A h_j where A = W_q^T R(δ) W_k / sqrt(dk)
     has rank = HEAD_DIM = 128 (the bottleneck).
     
     Factor: A = (W_q)^T R(δ) W_k = Q_proj^T R K_proj
     
     SVD of A gives: A = U S V^T where U, V are HIDDEN_DIM × 128
     The top-k directions of U and V give rank-k approximation:
     h_i^T A h_j ≈ Σ_{r=1}^{k} s_r (h_i · u_r)(h_j · v_r)
     
     This is k dot products per position + k products per pair.
     
  3. Key: since A has rank 128, the FULL ww term can be computed with
     128 dot products per position. But we want much fewer.

Tests:
  - SVD of A for various δ values
  - Singular value spectrum (how many directions matter?)
  - Sign agreement at rank 1, 2, 5, 10
  - End-to-end with low-rank ww correction
  - Stacked performance
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
print("  PHASE 10g: MESH RESIDUAL DECOMPOSITION")
print("  Finding correction directions in the null space of the bias")
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
# Extract everything + compute A(δ) matrices
# ================================================================
print("Extracting weights and computing hidden-space bilinear forms...")
inv_freq_cpu = 1.0 / (ROPE_THETA ** (
    torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

head_tables = {}
# Store W_q, W_k per routing head for bilinear form computation
head_weights = {}  # {(layer, head): {'W_q': ..., 'W_k': ..., 'b_q': ..., 'b_k': ...}}

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
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for h in routing:
        g = h // HEADS_PER_KV
        W_q_h = W_q_all[h]  # (HEAD_DIM, HIDDEN_DIM)
        W_k_g = W_k_all[g]  # (HEAD_DIM, HIDDEN_DIM)
        b_q_h = b_q_all[h]  # (HEAD_DIM,)
        b_k_g = b_k_all[g]  # (HEAD_DIM,)

        # Bias-aware tables
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
        head_weights[(layer_idx, h)] = {
            'W_q': W_q_h.clone(), 'W_k': W_k_g.clone(),
            'b_q': b_q_h.clone(), 'b_k': b_k_g.clone(),
        }

    del W_q_all, W_k_all
    torch.cuda.empty_cache()
    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx} done")

print(f"  {len(head_tables)} head tables + weight matrices")
print()


# ================================================================
# ANALYSIS 1: SVD of the hidden-space bilinear form A(δ)
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: Singular Value Spectrum of A(δ)")
print("  A(δ) = W_q^T R(δ) W_k / sqrt(d_k)")
print("=" * 80)
print()

# Pick representative heads and deltas
test_heads = []
for li in [0, 3, 7, 13, 20, 27]:
    routing = sorted(layer_classification[li]['routing'])
    if routing:
        test_heads.append((li, routing[0]))

test_deltas = [0, 1, 2, 3, 5, 10]

sv_spectra = {}

for li, h in test_heads:
    W_q = head_weights[(li, h)]['W_q']  # (HEAD_DIM, HIDDEN_DIM)
    W_k = head_weights[(li, h)]['W_k']  # (HEAD_DIM, HIDDEN_DIM)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for delta in test_deltas:
        # A(δ) = W_q^T R(δ) W_k / sqrt(d_k)
        # = (HIDDEN_DIM × HEAD_DIM) @ R(δ) @ (HEAD_DIM × HIDDEN_DIM)
        # = HIDDEN_DIM × HIDDEN_DIM — too big!
        #
        # But A = W_q^T R(δ) W_k has rank ≤ HEAD_DIM
        # Factor: A = (W_q^T) @ (R(δ) W_k)
        # SVD: compute thin SVD via W_q and R(δ)W_k
        #
        # Actually: h_i^T A h_j = (W_q h_i)^T R(δ) (W_k h_j) / sqrt(d_k)
        # Let q = W_q h_i (HEAD_DIM), k = W_k h_j (HEAD_DIM)
        # score_ww = q^T R(δ) k / sqrt(d_k)
        #
        # In hidden space: A_hidden = W_q^T R(δ) W_k * scale
        # This is (HIDDEN × HEAD) @ (HEAD × HEAD) @ (HEAD × HIDDEN) = HIDDEN × HIDDEN
        # Rank = HEAD_DIM = 128
        #
        # SVD via the thin form:
        # Let Q = W_q^T (HIDDEN × HEAD), K_rot = R(δ) W_k (HEAD × HIDDEN)
        # A = Q @ K_rot * scale
        #
        # Compute M = K_rot @ Q^T = (HEAD × HIDDEN) @ (HIDDEN × HEAD) = HEAD × HEAD
        # SVD(M) = U_m S V_m^T
        # Then: A = Q V_m S^{1/2} ... no, simpler:
        #
        # Direct SVD of the HEAD_DIM × HEAD_DIM matrix:
        # M_head = W_q @ W_k^T @ R(δ)^T — wait, let me think...
        #
        # h_i^T W_q^T R(δ) W_k h_j = (W_q h_i)^T R(δ) (W_k h_j)
        #
        # Let's compute the SVD in HEAD space:
        # R(δ) is HEAD_DIM × HEAD_DIM (rotation)
        # The product W_q @ W_q^T and W_k @ W_k^T are HEAD × HEAD
        #
        # For SVD of the HIDDEN-space form, factor as:
        # A = W_q^T (HEAD→HIDDEN) @ R(δ) (HEAD→HEAD) @ W_k (HIDDEN→HEAD)^T ... no
        # W_q is (HEAD, HIDDEN), so W_q^T is (HIDDEN, HEAD)
        # R(δ) is (HEAD, HEAD)
        # W_k is (HEAD, HIDDEN)
        # A = W_q^T @ R(δ) @ W_k = (HIDDEN, HEAD) @ (HEAD, HEAD) @ (HEAD, HIDDEN)
        #   = (HIDDEN, HIDDEN) with rank HEAD_DIM
        #
        # Thin SVD: compute (HEAD × HEAD) matrix M = R(δ) @ W_k @ W_q^T @ R(δ)^T ... no
        #
        # Easier: compute it as a product and take SVD of the thin factors
        # Left factor: L = W_q^T @ R(δ)  (HIDDEN × HEAD)
        # Right factor: R = W_k  (HEAD × HIDDEN)
        # A = L @ R  (rank HEAD_DIM)
        #
        # SVD of L @ R: compute R @ L^T = (HEAD × HEAD), SVD gives U_small, S, V_small
        # Then: U_big = L @ V_small @ S^{-1/2} ... or just compute directly

        W_k_rot = torch.zeros_like(W_k)
        for col in range(HIDDEN_DIM):
            W_k_rot[:, col] = rope_rotate_vector(W_k[:, col], delta, inv_freq_cpu)

        # M_head = W_q @ W_k_rot^T  (HEAD × HEAD)  — this is the MESH!
        # But we want the SVD of A_hidden = W_q^T @ W_k_rot * scale
        # which has rank HEAD_DIM
        #
        # Use: SVD(W_q^T @ W_k_rot) via the HEAD × HEAD inner product:
        # G = W_q @ W_q^T  (HEAD × HEAD) — Gram matrix
        # ... actually just compute the thin SVD directly

        # Left: W_q^T is (HIDDEN, HEAD)
        # Right: W_k_rot is (HEAD, HIDDEN)
        # Product: (HIDDEN, HIDDEN) rank HEAD_DIM
        #
        # Thin SVD via the (HEAD × HEAD) matrix:
        # M = W_k_rot @ W_q^T  → transpose of the usual MESH
        # Wait: (W_q^T @ W_k_rot)^T = W_k_rot^T @ W_q = MESH^T at this delta
        #
        # Let's just compute the MESH and SVD it
        MESH = (W_q @ W_k_rot.T) * scale  # (HEAD × HEAD)
        U, S, Vt = torch.linalg.svd(MESH)

        # The SVD of A_hidden = W_q^T MESH_row W_k ... actually:
        # h_i^T A h_j = (W_q h_i)^T R(δ) (W_k h_j) * scale
        # = q_i^T MESH_cols_of_R_Wk ... no
        #
        # Simplify: the bilinear form A in hidden space is:
        # A_hidden[a, b] = Σ_{mn} W_q[m,a] R(δ)[m,n] W_k[n,b] * scale
        # = (W_q^T)_row_a · (R(δ) W_k)_col_b * scale
        #
        # So A_hidden = W_q^T @ R(δ) @ W_k * scale  (HIDDEN × HIDDEN, rank HEAD_DIM)
        #
        # For the thin SVD: use the HEAD×HEAD MESH = W_q @ R(δ) @ W_k^T ... wait
        # MESH is usually W_q^T @ W_k in head space, which is HEAD × HEAD
        # But with RoPE: MESH(δ) = W_q R(δ) W_k^T ... ? No:
        # The standard MESH from our work: M = W_q.T @ W_k (finding 45)
        # With RoPE: score = q^T R(δ) k = h^T W_q^T R(δ) W_k h
        #
        # So MESH_rope(δ) = W_q^T R(δ) W_k in HEAD→HEAD would be wrong
        # Actually it's: HEAD×HIDDEN times HEAD×HEAD times HEAD×HIDDEN → need to be careful
        #
        # W_q is (HEAD, HIDDEN). W_q^T is (HIDDEN, HEAD).
        # R(δ) applied to the HEAD dim vectors.
        #
        # In HEAD space: MESH_head(δ) = R(δ)^{between q and k} but it operates on head vectors
        # score = (W_q h_i)^T R(δ) (W_k h_j) * scale
        #       = q_i^T R(δ) k_j * scale
        #
        # For the HIDDEN space bilinear: we want U, V in HIDDEN space such that
        # h_i^T A h_j ≈ Σ_r s_r (u_r · h_i)(v_r · h_j)
        #
        # Factor: A_hidden = W_q^T @ diag_in_head @ W_k (but with RoPE it's not diagonal)
        #
        # Just compute it directly for small HEAD_DIM:
        # The inner (HEAD × HEAD) matrix is B = R(δ) (applied element-wise to columns of W_k)
        # Then A_hidden = W_q^T @ B_applied @ W_k ... hmm
        #
        # Let me just compute the SVD of the MESH in head space and project to hidden space:
        # MESH = W_q @ W_k_rot.T (after RoPE applied to W_k columns for this delta)
        # Wait no: score = (W_q h)^T R(δ) (W_k h') = h^T W_q^T [stuff in head space] W_k h'
        # where [stuff] applies RoPE to the k vector

        # OK let me just be concrete:
        # q = W_q h_i  (HEAD_DIM)
        # k_rot = R(δ) W_k h_j  (HEAD_DIM) — RoPE applied to k
        # score = q^T k_rot * scale = (W_q h_i)^T R(δ) W_k h_j * scale
        #
        # Define: L = W_q  (HEAD × HIDDEN)
        #         R = R(δ)·W_k  (HEAD × HIDDEN) — W_k with RoPE on each output dim
        # score = h_i^T L^T R h_j * scale
        # A_hidden = L^T R * scale = (HIDDEN × HEAD)(HEAD × HIDDEN) = HIDDEN × HIDDEN
        #
        # Thin SVD: compute G = L L^T (HEAD × HEAD), and H = R R^T (HEAD × HEAD)
        # Or: SVD(L^T R) via SVD(R L^T) where R L^T is HEAD × HEAD
        #
        # P = R @ L^T = W_k_rot @ W_q^T  (HEAD × HEAD)  ... wait that's wrong dim
        # R is (HEAD, HIDDEN), L is (HEAD, HIDDEN)
        # R @ L^T = (HEAD, HIDDEN)(HIDDEN, HEAD) = HEAD × HEAD ✓
        #
        # No wait: A = L^T @ R (HIDDEN × HIDDEN)
        # SVD: A = U_A S_A V_A^T
        # Compute P = R @ L^T = R L^T (HEAD × HEAD)
        # This is not the same as A. But:
        # If L^T = U_L S_L V_L^T (thin: HIDDEN × HEAD)
        # and R = U_R S_R V_R^T (thin: HEAD × HIDDEN)
        # Then A = V_L S_L U_L^T U_R S_R V_R^T
        # ... complicated. Let me just compute the core HEAD×HEAD matrix and SVD it.

        # Core: C = W_q @ scale_factor @ W_k_rot^T
        # We already computed MESH = W_q @ W_k_rot.T * scale (HEAD × HEAD)
        # SVD(MESH) = U S Vt

        # Now: score_ww = (W_q h_i)^T R(δ)(W_k h_j) * scale
        #              = h_i^T W_q^T (stuff) h_j
        #
        # Using MESH SVD: MESH = U S Vt (HEAD × HEAD)
        # score = q_i^T k_rot_j * scale where q_i = W_q h_i, k_rot_j = W_k_rot h_j
        #       = Σ_r s_r (u_r · q_i)(v_r · k_rot_j)
        #       = Σ_r s_r (u_r · W_q h_i)(v_r · W_k_rot h_j)
        #       = Σ_r s_r (W_q^T u_r · h_i)(W_k_rot^T v_r · h_j)
        #
        # So the HIDDEN-space directions are:
        #   d_q_r = W_q^T u_r   (HIDDEN,)
        #   d_k_r = W_k_rot^T v_r  (HIDDEN,)
        # And: score = Σ_r s_r (d_q_r · h_i)(d_k_r · h_j)

        # Store for this head/delta
        sv_spectra[(li, h, delta)] = S.numpy()

        # Show spectrum
        if delta == 1:
            top10 = S[:10].numpy()
            total = S.sum().item()
            cum_pct = np.cumsum(S.numpy()) / total * 100
            print(f"  L{li:2d} h{h:2d} δ={delta}: top-10 σ = [{', '.join(f'{v:.3f}' for v in top10)}]")
            print(f"    cumulative %: [{', '.join(f'{v:.1f}' for v in cum_pct[:10])}]")
            print(f"    rank for 90%: {np.searchsorted(cum_pct, 90) + 1}")
            print(f"    rank for 95%: {np.searchsorted(cum_pct, 95) + 1}")
            print(f"    rank for 99%: {np.searchsorted(cum_pct, 99) + 1}")
            print()

print()


# ================================================================
# ANALYSIS 2: Low-rank ww approximation accuracy
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: Low-Rank WW Term Accuracy")
print("  How many SVD directions needed for correct ww sign?")
print("=" * 80)
print()

DIAG_PROMPTS = [
    "The capital of France is",
    "Albert Einstein developed the theory of",
    "To be or not to",
    "The largest planet in our solar system is",
    "The color of grass is",
]

# For a sample of heads, pre-compute hidden-space directions at various ranks
# and test sign agreement with actual ww term

# Pick one head per zone for detailed analysis
detail_heads = []
for li in [0, 7, 13, 20, 27]:
    routing = sorted(layer_classification[li]['routing'])
    if routing:
        detail_heads.append((li, routing[0]))

rank_sign_agreement = defaultdict(lambda: defaultdict(list))  # {(li,h): {rank: [agreements]}}
rank_correlations = defaultdict(lambda: defaultdict(list))

for pi, prompt in enumerate(DIAG_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    layer_h = {}
    real_scores_data = {}

    def capture_hook(li):
        def hook_fn(module, args, kwargs, output):
            h_in = args[0] if args else kwargs.get('hidden_states')
            if h_in is None: return output
            b, s, _ = h_in.shape
            with torch.no_grad():
                Q = module.q_proj(h_in).to(torch.bfloat16)
                K = module.k_proj(h_in).to(torch.bfloat16)
            Q = Q.reshape(b, s, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            K = K.reshape(b, s, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            cos, sin = get_rope_cache(s, h_in.device, torch.bfloat16)
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
            K_exp = K.repeat_interleave(HEADS_PER_KV, dim=1)
            scores = {}
            for hd in range(NUM_HEADS):
                scores[hd] = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HEAD_DIM)).float().cpu()
            real_scores_data[li] = scores
            layer_h[li] = h_in[0].cpu().float()
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

    for li, hd in detail_heads:
        if li not in layer_h:
            continue
        h_states = layer_h[li]
        s = h_states.shape[0]
        tbl = head_tables[(li, hd)]
        wts = head_weights[(li, hd)]
        W_q = wts['W_q']  # (HEAD_DIM, HIDDEN_DIM)
        W_k = wts['W_k']  # (HEAD_DIM, HIDDEN_DIM)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        real_sc = real_scores_data[li][hd]

        for i in range(s):
            for j in range(i + 1):
                delta = i - j
                # Actual ww term
                bl = tbl['baseline'][delta].item()
                cq = (h_states[i] @ tbl['c_q'][delta]).item()
                ck = (tbl['c_k'][delta] @ h_states[j]).item()
                ww_actual = real_sc[i, j].item() - (bl + cq + ck)

                # Compute low-rank approximations
                # MESH(δ) = W_q @ W_k_rot^T * scale
                W_k_rot = torch.zeros_like(W_k)
                for col in range(HIDDEN_DIM):
                    W_k_rot[:, col] = rope_rotate_vector(W_k[:, col], delta, inv_freq_cpu)

                MESH = (W_q @ W_k_rot.T) * scale  # HEAD × HEAD
                U, S_vals, Vt = torch.linalg.svd(MESH)

                # Project h_states through W_q and W_k_rot
                q_proj = W_q @ h_states[i]  # HEAD_DIM
                k_proj = W_k_rot @ h_states[j]  # HEAD_DIM

                # Low-rank: score_r = Σ_{m=0}^{r-1} s_m (u_m·q)(v_m·k)
                for rank in [1, 2, 3, 5, 10, 20, 50, 128]:
                    approx = 0.0
                    for m in range(min(rank, HEAD_DIM)):
                        approx += S_vals[m].item() * (U[:, m] @ q_proj).item() * (Vt[m] @ k_proj).item()

                    sign_agree = (1 if approx >= 0 else -1) == (1 if ww_actual >= 0 else -1)
                    rank_sign_agreement[(li, hd)][rank].append(1 if sign_agree else 0)

                    if rank in [1, 5, 10, 128]:
                        rank_correlations[(li, hd)][rank].append((ww_actual, approx))

print("  Sign agreement by rank:")
print(f"  {'Head':>12s}  {'R=1':>6s}  {'R=2':>6s}  {'R=3':>6s}  {'R=5':>6s}  {'R=10':>6s}  {'R=20':>6s}  {'R=50':>6s}  {'R=128':>6s}")
print("  " + "-" * 70)

for li, hd in detail_heads:
    vals = rank_sign_agreement[(li, hd)]
    parts = []
    for r in [1, 2, 3, 5, 10, 20, 50, 128]:
        v = np.mean(vals[r]) if vals[r] else 0
        parts.append(f"{v:.1%}")
    print(f"  L{li:2d} h{hd:2d}:   {'  '.join(parts)}")

# Correlation at key ranks
print()
print("  Correlation (ww_actual vs approx) by rank:")
print(f"  {'Head':>12s}  {'R=1':>8s}  {'R=5':>8s}  {'R=10':>8s}  {'R=128':>8s}")
print("  " + "-" * 50)
for li, hd in detail_heads:
    parts = []
    for r in [1, 5, 10, 128]:
        pairs = rank_correlations[(li, hd)][r]
        if len(pairs) > 2:
            actuals = [p[0] for p in pairs]
            approxs = [p[1] for p in pairs]
            c = np.corrcoef(actuals, approxs)[0, 1]
            parts.append(f"{c:.4f}")
        else:
            parts.append("n/a")
    print(f"  L{li:2d} h{hd:2d}:   {'    '.join(parts)}")

print()


# ================================================================
# END-TO-END: Bias-aware + low-rank ww correction
# ================================================================
print("=" * 80)
print("  END-TO-END: Bias + Low-Rank WW Correction")
print("=" * 80)
print()

# Pre-compute per-delta SVD directions for each routing head
# at rank K (we'll test K=5 based on spectrum analysis)
print("  Pre-computing low-rank ww directions...")

ww_directions = {}  # {(layer, head, delta): {'U_q': (K, HIDDEN), 'V_k': (K, HIDDEN), 'S': (K,)}}
TEST_RANKS = [5, 10, 20]
MAX_RANK = max(TEST_RANKS)

for layer_idx in range(N_LAYERS):
    routing = layer_classification[layer_idx]['routing']
    for h in routing:
        if (layer_idx, h) not in head_weights:
            continue
        wts = head_weights[(layer_idx, h)]
        W_q = wts['W_q']
        W_k = wts['W_k']
        scale = 1.0 / math.sqrt(HEAD_DIM)

        for delta in range(min(MAX_SEQ, 16)):  # Only precompute up to δ=15 for speed
            W_k_rot = torch.zeros_like(W_k)
            for col in range(HIDDEN_DIM):
                W_k_rot[:, col] = rope_rotate_vector(W_k[:, col], delta, inv_freq_cpu)

            MESH = (W_q @ W_k_rot.T) * scale  # HEAD × HEAD
            U, S_vals, Vt = torch.linalg.svd(MESH)

            # Hidden-space directions: d_q_r = W_q^T u_r, d_k_r = W_k_rot^T v_r
            U_q = torch.zeros(MAX_RANK, HIDDEN_DIM)
            V_k = torch.zeros(MAX_RANK, HIDDEN_DIM)
            for r in range(min(MAX_RANK, HEAD_DIM)):
                U_q[r] = W_q.T @ U[:, r]
                V_k[r] = W_k_rot.T @ Vt[r]

            ww_directions[(layer_idx, h, delta)] = {
                'U_q': U_q, 'V_k': V_k, 'S': S_vals[:MAX_RANK].clone()
            }

    if layer_idx % 7 == 0:
        print(f"    Layer {layer_idx} done")

print(f"    {len(ww_directions)} direction sets pre-computed")
print()


def make_bias_plus_ww_rank(rank):
    """Create attention function using bias-aware + rank-K ww correction."""
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

            # Pre-compute projections for ww directions
            ww_projs_q = {}  # {delta: tensor (rank,)}
            ww_projs_k = {}

            for i in range(seq_len):
                for j in range(i + 1):
                    delta = i - j
                    # Terms 1-3
                    bl = tbl['baseline'][delta].item()
                    cq = (h_float[i] @ tbl['c_q'][delta]).item()
                    ck = (tbl['c_k'][delta] @ h_float[j]).item()

                    # Term 4: low-rank ww
                    ww = 0.0
                    key = (layer_idx, h, delta)
                    if key in ww_directions:
                        dirs = ww_directions[key]
                        # Compute projections (cache per position)
                        if (i, delta) not in ww_projs_q:
                            ww_projs_q[(i, delta)] = h_float[i] @ dirs['U_q'][:rank].T
                        if (j, delta) not in ww_projs_k:
                            ww_projs_k[(j, delta)] = h_float[j] @ dirs['V_k'][:rank].T

                        pq = ww_projs_q[(i, delta)]
                        pk = ww_projs_k[(j, delta)]
                        ww = (pq * pk * dirs['S'][:rank]).sum().item()

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
    return attn_fn


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
n, t, c = evaluate("all_real", {i: attn_real_qk for i in range(N_LAYERS)})
print(f"  {'A: All real QK + phi_softmax':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# All layers with ww correction at various ranks (STACKED)
for rank in TEST_RANKS:
    attn_fn = make_bias_plus_ww_rank(rank)
    n, t, c = evaluate(f"ww_r{rank}", {i: attn_fn for i in range(N_LAYERS)})
    print(f"  {f'B: All bias + ww rank-{rank} (stacked)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Zone-aware anchoring + ww correction
anchor_layers = set(range(4)) | {7, 11, 15, 19, 23, 27}
for rank in TEST_RANKS:
    attn_fn = make_bias_plus_ww_rank(rank)
    cfg = {}
    for i in anchor_layers:
        cfg[i] = attn_real_qk
    for i in set(range(N_LAYERS)) - anchor_layers:
        cfg[i] = attn_fn
    n, t, c = evaluate(f"zone_ww_r{rank}", cfg)
    print(f"  {f'C: Zone anchored + ww rank-{rank}':>55s}  {n:2d}/{t:2d}    {c:.4f}")

print()

# Save
save_data = {
    'sv_spectra_sample': {},
    'sign_agreement': {},
    'rank_correlations': {},
}
for (li, h, delta), S in sv_spectra.items():
    save_data['sv_spectra_sample'][f'L{li}_h{h}_d{delta}'] = S[:20].tolist()
for (li, h), ranks in rank_sign_agreement.items():
    for r, vals in ranks.items():
        save_data['sign_agreement'][f'L{li}_h{h}_r{r}'] = float(np.mean(vals))
for (li, h), ranks in rank_correlations.items():
    for r, pairs in ranks.items():
        if len(pairs) > 2:
            c = np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1]
            save_data['rank_correlations'][f'L{li}_h{h}_r{r}'] = float(c)

save_path = os.path.join(results_dir, 'phase10g_mesh_residual.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved to {save_path}")
print()
print("=" * 80)
print("  DONE — Phase 10g MESH Residual Decomposition")
print("=" * 80)
