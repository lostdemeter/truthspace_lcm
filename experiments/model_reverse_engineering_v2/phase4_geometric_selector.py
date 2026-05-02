"""
Phase 4: Geometric Selector for Layer 23 Head 6

Finding 38 showed: head 6 alone = 6/6. The "irreducible" error is a routing
decision — which position to attend to.

This script tests whether that routing decision can be made geometrically,
without full Q·K^T matmuls, using:

1. MESH SVD of W_q^T · W_k for head 6 — what are the singular values?
   If Zipf α ≈ 1/φ, the routing lives in few dimensions.

2. Rank-k approximation of the attention scores — does rank-1 or rank-2
   make the same routing decision as the full head?

3. Geometric Selector — can we reduce the routing to:
   "project each position onto direction d, pick the max"?
   This is O(seq_len × k) instead of O(seq_len × head_dim).

4. φ-level selector — does the routing decision quantize to φ-levels?

If rank-1 works, head 6's routing becomes a single dot product per position —
trivial complexity, fully geometric.
"""

import sys
import numpy as np
import time
import gc

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI_CONST = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI_CONST)


def finish_forward(engine, hidden_start, start_layer):
    """Run remaining layers + final norm + LM head."""
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def get_top1(logits, tokenizer):
    idx = int(np.argmax(logits[0, -1, :]))
    tok = tokenizer.decode_token(idx)
    sorted_l = np.sort(logits[0, -1, :])[::-1]
    margin = sorted_l[0] - sorted_l[1]
    return idx, tok, margin


def run_head6_with_rank_k(engine, layer_idx, hidden, rank_k, U, S, Vt,
                          head_idx=6):
    """
    Run layer with only head_idx active, using rank-k approximation
    for the Q·K^T attention scores.

    Instead of: scores = Q @ K^T (full head_dim dot product)
    We use:     scores = (Q @ V_k) @ diag(S_k) @ (K @ U_k)^T

    Where U_k, S_k, V_k are the top-k components of the MESH SVD.
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    batch, seq_len, hidden_dim = hidden.shape

    normed = rms_norm(hidden, attn.norm_weight)

    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)

    Q = Q.reshape(batch, seq_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    V_val = V.reshape(batch, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

    Q = attn.rope.apply(Q)
    K = attn.rope.apply(K)

    K_expanded = np.repeat(K, attn.heads_per_kv, axis=1)
    V_expanded = np.repeat(V_val, attn.heads_per_kv, axis=1)

    # Full scores for reference
    full_scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * attn.scale

    if rank_k is None:
        # Use full scores for target head
        scores = full_scores.copy()
    else:
        # Rank-k approximation for target head only
        scores = np.zeros_like(full_scores)

        # For head_idx: use rank-k MESH approximation
        q_head = Q[0, head_idx, :, :]   # (seq_len, head_dim)
        k_head = K_expanded[0, head_idx, :, :]  # (seq_len, head_dim)

        # Rank-k: Q @ Vt[:k].T @ diag(S[:k]) @ U[:,:k].T @ K.T
        # = (Q @ Vt[:k].T) @ diag(S[:k]) @ (K @ U[:,:k]).T
        q_proj = q_head @ Vt[:rank_k, :].T          # (seq_len, k)
        k_proj = k_head @ U[:, :rank_k]              # (seq_len, k)
        approx_scores = (q_proj * S[:rank_k]) @ k_proj.T  # (seq_len, seq_len)
        scores[0, head_idx, :, :] = approx_scores * attn.scale

    # Apply causal mask
    if seq_len > 1:
        causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal_mask

    attn_weights = phi_softmax(scores, axis=-1)
    per_head_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)

    # Zero out all heads except head_idx
    for hi in range(attn.num_heads):
        if hi != head_idx:
            per_head_output[0, hi, :, :] = 0.0

    combined = per_head_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    attn_proj = phi_linear(attn.W_o, combined)

    post_attn = hidden + attn_proj

    # Real MLP
    mlp = layer.mlp
    normed_mlp = rms_norm(post_attn, mlp.norm_weight)
    gate = phi_linear(mlp.W_gate, normed_mlp)
    up = phi_linear(mlp.W_up, normed_mlp)
    mlp_hidden = phi_silu(gate) * up
    mlp_out = phi_linear(mlp.W_down, mlp_hidden)

    return post_attn + mlp_out


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23
    head_idx = 6
    layer = engine.layers[target_layer]
    attn = layer.attention

    test_prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]

    # =========================================================================
    #   Part A: MESH SVD for head 6
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: MESH SVD for layer 23, head 6")
    print("=" * 80)

    # Extract W_q and W_k for head 6
    # W_q shape: (num_heads * head_dim, hidden_dim) stored as phi-encoded
    # We need head 6's slice: rows [6*128 : 7*128]
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads

    # For GQA: head 6 uses KV group 6 // heads_per_kv
    kv_group = head_idx // heads_per_kv
    print(f"  Head {head_idx}: head_dim={head_dim}, KV group={kv_group}, "
          f"heads_per_kv={heads_per_kv}")

    # Extract the actual weight matrices by probing with identity
    # W_q for head 6: project e_i through W_q, take head 6's slice
    hidden_dim = engine.hidden_dim
    print(f"  Extracting W_q and W_k matrices for head 6...", flush=True)

    # Create identity matrix and project through Q and K
    # This gives us the actual weight matrices after φ-decoding
    identity = np.eye(hidden_dim, dtype=np.float32)

    # Process in chunks to avoid memory issues
    chunk_size = 512
    W_q_head = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    W_k_group = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]  # (1, chunk, hidden)

        q_out = phi_linear(attn.W_q, chunk, attn.b_q)  # (1, chunk, num_heads*head_dim)
        k_out = phi_linear(attn.W_k, chunk, attn.b_k)  # (1, chunk, num_kv_heads*head_dim)

        # Extract head 6's Q
        q_reshaped = q_out[0].reshape(-1, num_heads, head_dim)
        W_q_head[:, start:end] = q_reshaped[:, head_idx, :].T

        # Extract KV group's K
        k_reshaped = k_out[0].reshape(-1, num_kv_heads, head_dim)
        W_k_group[:, start:end] = k_reshaped[:, kv_group, :].T

        if start % 1024 == 0:
            print(f"    Extracted {end}/{hidden_dim}...", flush=True)

    # MESH matrix: M = W_q^T @ W_k (head_dim × head_dim after projecting through hidden)
    # Actually for attention: score = q @ k^T = (x @ W_q^T) @ (x @ W_k^T)^T
    # = x @ W_q^T @ W_k @ x^T
    # So MESH = W_q^T @ W_k  but we have W_q_head (head_dim, hidden) and W_k_group (head_dim, hidden)
    # score_ij = q_i @ k_j = (h_i @ W_q_head.T) @ (h_j @ W_k_group.T)^T
    #          = h_i @ (W_q_head.T @ W_k_group) @ h_j^T
    # So interaction matrix in hidden space: W_q_head.T @ W_k_group (hidden × hidden) — too big
    #
    # In head space: MESH = W_q_head @ W_k_group^T (head_dim × head_dim)
    # This is the matrix that transforms K-space into Q-space for scoring

    MESH = W_q_head @ W_k_group.T  # (head_dim, head_dim)
    print(f"\n  MESH matrix shape: {MESH.shape}")

    U, S, Vt = np.linalg.svd(MESH)
    print(f"  Singular values (top 20):")
    for i in range(min(20, len(S))):
        pct = S[i] / S[0] * 100
        cumvar = np.sum(S[:i+1]**2) / np.sum(S**2) * 100
        print(f"    S[{i:2d}] = {S[i]:10.4f}  ({pct:5.1f}% of S[0])  cumulative variance: {cumvar:5.1f}%")

    # Zipf fit
    ranks = np.arange(1, len(S) + 1)
    log_ranks = np.log(ranks[:20])
    log_svs = np.log(S[:20] + 1e-20)
    alpha = -np.polyfit(log_ranks, log_svs, 1)[0]
    print(f"\n  Zipf exponent α = {alpha:.4f}  (1/φ = {1/PHI_CONST:.4f})")

    kappa = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
    print(f"  Condition number κ = {kappa:.1f}")

    # How much variance does rank-1 capture?
    for k in [1, 2, 3, 5, 10]:
        var_k = np.sum(S[:k]**2) / np.sum(S**2) * 100
        print(f"  Rank-{k:2d}: {var_k:.1f}% of score variance")

    # =========================================================================
    #   Part B: Does rank-k routing make the same decision?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Rank-k routing decision comparison")
    print("=" * 80)
    print("  Does the rank-k approximation select the same position as full?")

    # Note: The SVD above is of the MESH in head-space (head_dim × head_dim).
    # But after RoPE, the Q and K are position-dependent, so we can't directly
    # use the MESH SVD for routing.
    #
    # Instead, we need the SVD of the EFFECTIVE interaction for each prompt,
    # which includes RoPE. Let's take a different approach:
    #
    # For each prompt, compute the full Q·K^T scores for head 6,
    # then compute the rank-k approximation and see if argmax matches.

    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                break
            h = lo(h)

        # Get full attention scores for head 6
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)

        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_expanded = np.repeat(K, heads_per_kv, axis=1)

        q_last = Q[0, head_idx, -1, :]   # (head_dim,) — last token's query
        K_head = K_expanded[0, head_idx, :, :]  # (seq_len, head_dim)

        full_scores = q_last @ K_head.T  # (seq_len,)

        # SVD of the per-prompt Q·K interaction (just for the last query)
        # The scores are: q_last @ K_head.T
        # This is a rank-1 outer product from q's perspective.
        # But we want to approximate the K_head matrix.

        # Actually, the MESH SVD tells us:
        # score = q @ M @ k  (in the pre-RoPE basis)
        # But RoPE rotates Q and K, so we need to work post-RoPE.
        #
        # Simpler approach: SVD of K_head (seq_len × head_dim)
        # Then project q onto the top-k right singular vectors.

        U_k, S_k, Vt_k = np.linalg.svd(K_head, full_matrices=False)

        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Tokens: {tokens}")
        print(f"  Full scores (last→all): ", end="")
        for pos in range(len(tokens)):
            marker = " ←" if pos == np.argmax(full_scores) else ""
            print(f" {tokens[pos]}={full_scores[pos]:.2f}{marker}", end="")
        print()

        full_argmax = np.argmax(full_scores)

        for k in [1, 2, 3, 5, 10]:
            if k > len(S_k):
                continue
            # Rank-k approx: project q and K onto top-k singular space
            q_proj = q_last @ Vt_k[:k, :].T  # (k,)
            K_proj = K_head @ Vt_k[:k, :].T   # (seq_len, k)
            approx_scores = K_proj @ q_proj    # (seq_len,)

            approx_argmax = np.argmax(approx_scores)
            match = "✓" if approx_argmax == full_argmax else "✗"
            cos_sim = np.dot(full_scores, approx_scores) / (
                np.linalg.norm(full_scores) * np.linalg.norm(approx_scores) + 1e-20)

            print(f"    Rank-{k:2d}: argmax={approx_argmax} ({tokens[approx_argmax]:>8s}) "
                  f"{match}  cos(scores)={cos_sim:.4f}")

    # =========================================================================
    #   Part C: Can we use a FIXED projection (not per-prompt SVD)?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: Fixed geometric selector — MESH top-k projection")
    print("=" * 80)
    print("  Using the MESH SVD (fixed, not per-prompt) as the selector direction.")
    print("  This is the true geometric selector: pre-computed, no per-prompt SVD.")

    # The MESH SVD gives us U, S, Vt of W_q_head @ W_k_group^T
    # Pre-RoPE: score = q_pre @ MESH @ k_pre^T
    # But RoPE modifies Q and K per-position, so let's test if the
    # pre-RoPE MESH directions still work post-RoPE.

    # Approach: use the right singular vectors of MESH as the "selector directions"
    # q projects onto U[:,:k] (query space), K projects onto Vt[:k,:].T (key space)

    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                break
            h = lo(h)

        normed = rms_norm(h, attn.norm_weight)
        Q_full = phi_linear(attn.W_q, normed, attn.b_q)
        K_full = phi_linear(attn.W_k, normed, attn.b_k)

        Q_full = Q_full.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K_full = K_full.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q_full = attn.rope.apply(Q_full)
        K_full = attn.rope.apply(K_full)
        K_exp = np.repeat(K_full, heads_per_kv, axis=1)

        q_last = Q_full[0, head_idx, -1, :]
        K_head = K_exp[0, head_idx, :, :]

        full_scores = q_last @ K_head.T
        full_argmax = np.argmax(full_scores)

        print(f"\n  \"{prompt}\"  full→{tokens[full_argmax]}")

        for k in [1, 2, 3, 5, 10, 20]:
            # Project using FIXED MESH singular vectors
            # Q projects onto left singular vectors (query direction)
            # K projects onto right singular vectors (key direction)
            q_proj = q_last @ U[:, :k]        # (k,) — project query
            K_proj = K_head @ Vt[:k, :].T      # (seq_len, k) — project keys
            approx_scores = K_proj @ (S[:k] * q_proj)  # weight by singular values

            approx_argmax = np.argmax(approx_scores)
            match = "✓" if approx_argmax == full_argmax else "✗"
            cos_sim = np.dot(full_scores, approx_scores) / (
                np.linalg.norm(full_scores) * np.linalg.norm(approx_scores) + 1e-20)

            print(f"    MESH rank-{k:2d}: → {tokens[approx_argmax]:>8s} {match}  "
                  f"cos={cos_sim:.4f}")

    # =========================================================================
    #   Part D: End-to-end test — rank-k selector → full pipeline → accuracy
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: End-to-end accuracy with rank-k geometric selector")
    print("=" * 80)
    print("  Replace head 6's full Q·K^T with rank-k MESH approx, test 6/6.")

    # We need to modify the attention computation so that head 6 uses
    # the rank-k approximated scores instead of full scores.

    for k in [1, 2, 3, 5, 10, 20, None]:
        label = f"rank-{k}" if k is not None else "full"
        n_pass = 0
        france_margin = None

        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)

            # Run with rank-k selector
            ablated_out = run_head6_with_rank_k(
                engine, target_layer, h, k, U, S, Vt, head_idx=head_idx)

            logits_full = finish_forward(engine, full_out, target_layer)
            logits_abl = finish_forward(engine, ablated_out, target_layer)

            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            abl_id, abl_tok, abl_margin = get_top1(logits_abl, tokenizer)

            if abl_id == full_id:
                n_pass += 1
            if 'France' in prompt:
                france_margin = abl_margin
                france_pass = abl_id == full_id

        fm_str = f"margin={france_margin:.3f}" if france_margin else ""
        fp_str = "✓" if france_pass else "✗"
        print(f"  {label:>8s}: {n_pass}/6  France={fp_str} {fm_str}")

    # =========================================================================
    #   Part E: φ-level analysis of selector scores
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: φ-level structure of selector scores")
    print("=" * 80)

    prompt = "The capital of France is"
    p_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode_token(i) for i in p_ids]
    h = engine.embedding(p_ids)[np.newaxis, :, :]
    for lo in engine.layers:
        if lo.layer_idx == target_layer:
            break
        h = lo(h)

    normed = rms_norm(h, attn.norm_weight)
    Q_full = phi_linear(attn.W_q, normed, attn.b_q)
    K_full = phi_linear(attn.W_k, normed, attn.b_k)

    Q_full = Q_full.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_full = K_full.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    Q_full = attn.rope.apply(Q_full)
    K_full = attn.rope.apply(K_full)
    K_exp = np.repeat(K_full, heads_per_kv, axis=1)

    q_last = Q_full[0, head_idx, -1, :]
    K_head = K_exp[0, head_idx, :, :]
    full_scores = q_last @ K_head.T

    print(f"  Tokens: {tokens}")
    print(f"\n  Full attention scores → φ-levels:")
    for pos in range(len(tokens)):
        s = full_scores[pos]
        if abs(s) > 1e-10:
            phi_level = np.log(abs(s)) / LOG_PHI
        else:
            phi_level = float('-inf')
        marker = " ← WINNER" if pos == np.argmax(full_scores) else ""
        print(f"    pos {pos} ({tokens[pos]:>10s}): score={s:8.3f}  "
              f"φ-level={phi_level:+.2f}{marker}")

    # Rank-1 selector scores
    q_proj_1 = q_last @ U[:, :1]
    K_proj_1 = K_head @ Vt[:1, :].T
    rank1_scores = (K_proj_1 @ (S[:1] * q_proj_1)).flatten()

    print(f"\n  Rank-1 selector scores → φ-levels:")
    for pos in range(len(tokens)):
        s = rank1_scores[pos]
        if abs(s) > 1e-10:
            phi_level = np.log(abs(s)) / LOG_PHI
        else:
            phi_level = float('-inf')
        marker = " ← WINNER" if pos == np.argmax(rank1_scores) else ""
        print(f"    pos {pos} ({tokens[pos]:>10s}): score={s:8.3f}  "
              f"φ-level={phi_level:+.2f}{marker}")

    # =========================================================================
    #   Part F: Compute savings summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part F: Compute savings summary")
    print("=" * 80)

    print(f"\n  Original layer 23 attention:")
    print(f"    28 heads × (Q + K + V + O) projections = full matmuls")
    print(f"    Q·K^T: 28 heads × seq_len × {head_dim} = {28 * head_dim} FLOPs/position")
    print(f"\n  With geometric selector (head 6 only, rank-k):")
    for k in [1, 2, 3, 5]:
        reduction = (28 * head_dim) / (1 * k)
        print(f"    Rank-{k}: 1 head × seq_len × {k} = {k} FLOPs/position "
              f"({reduction:.0f}× reduction in scoring)")
    print(f"\n  Note: Q/K/V/O projections for head 6 still needed ({head_dim} dims)")
    print(f"  But can potentially use MESH factored form for those too.")
    print(f"  Total attention compute for layer 23: 1/{28} of original = 3.6% of full attention")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
