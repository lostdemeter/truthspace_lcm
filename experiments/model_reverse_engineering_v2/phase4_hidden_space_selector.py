"""
Phase 4: Hidden-Space Geometric Selector for Layer 23 Head 6

Previous findings:
  - Head 6 alone = 6/6 (Finding 38)
  - Head 6's MESH is rank-1: S[0]=350K, S[1]=0.95 (Finding 39)
  - End-to-end rank-1 MESH approximation = 6/6 (but still needs Q/K/V matmuls)

This script eliminates even the Q/K/V projections by working in hidden space.

Since MESH = σ₁ u₁ v₁^T (rank-1), the pre-RoPE score is:
  score(q,k) = q^T @ MESH @ k = σ₁ (q · u₁)(k · v₁)

In hidden space (before Q/K projection):
  q = W_q @ h_normed,  k = W_k @ h_normed
  score = σ₁ (h_q_normed · W_q^T u₁)(h_k_normed · W_k^T v₁)
        = σ₁ (h_q_normed · d_q)(h_k_normed · d_k)

Where d_q = W_q^T @ u₁ and d_k = W_k^T @ v₁ are PRE-COMPUTABLE directions.

This reduces the routing to TWO dot products per position — no matmuls needed.

Tests:
  A. Pre-compute d_q, d_k from MESH SVD
  B. Hidden-space selector: does (h · d_k) pick the right position?
  C. End-to-end: use hidden-space routing to select a position, fetch V from
     that position, project through W_v and W_o — does this achieve 6/6?
  D. Simplest possible: just use (h · d_k) as the selector (ignore query)
  E. Full geometric pipeline: spectrometer for 14 layers + selector for layer 23
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

    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    kv_group = head_idx // heads_per_kv
    hidden_dim = engine.hidden_dim

    test_prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]

    # =========================================================================
    #   Part A: Extract W_q, W_k, W_v for head 6 and compute MESH SVD
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: Extract weight matrices and compute hidden-space directions")
    print("=" * 80)

    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512

    W_q_head = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    W_k_group = np.zeros((head_dim, hidden_dim), dtype=np.float32)
    W_v_group = np.zeros((head_dim, hidden_dim), dtype=np.float32)

    print(f"  Extracting Q/K/V weight matrices...", flush=True)
    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]

        q_out = phi_linear(attn.W_q, chunk, attn.b_q)
        k_out = phi_linear(attn.W_k, chunk, attn.b_k)
        v_out = phi_linear(attn.W_v, chunk, attn.b_v)

        q_reshaped = q_out[0].reshape(-1, num_heads, head_dim)
        k_reshaped = k_out[0].reshape(-1, num_kv_heads, head_dim)
        v_reshaped = v_out[0].reshape(-1, num_kv_heads, head_dim)

        W_q_head[:, start:end] = q_reshaped[:, head_idx, :].T
        W_k_group[:, start:end] = k_reshaped[:, kv_group, :].T
        W_v_group[:, start:end] = v_reshaped[:, kv_group, :].T

        if start % 1024 == 0:
            print(f"    {end}/{hidden_dim}...", flush=True)

    # Also extract W_o for head 6
    # W_o maps (num_heads * head_dim) → hidden_dim
    # We need the slice for head 6: columns [6*128 : 7*128]
    print(f"  Extracting W_o for head 6...", flush=True)
    identity_head = np.eye(num_heads * head_dim, dtype=np.float32)
    # Head 6's slice
    head6_input = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    W_o_head6 = np.zeros((hidden_dim, head_dim), dtype=np.float32)
    for d in range(head_dim):
        head6_input[0, 0, :] = 0.0
        head6_input[0, 0, head_idx * head_dim + d] = 1.0
        o_out = phi_linear(attn.W_o, head6_input)
        W_o_head6[:, d] = o_out[0, 0, :]

    # MESH SVD
    MESH = W_q_head @ W_k_group.T
    U, S, Vt = np.linalg.svd(MESH)

    print(f"\n  MESH S[0]={S[0]:.1f}, S[1]={S[1]:.4f}, ratio={S[0]/S[1]:.0f}:1")

    # Hidden-space directions
    u1 = U[:, 0]   # top left singular vector (query space)
    v1 = Vt[0, :]   # top right singular vector (key space)

    d_q = W_q_head.T @ u1   # (hidden_dim,) — query selector
    d_k = W_k_group.T @ v1  # (hidden_dim,) — key selector
    d_v = W_v_group.T @ v1  # (hidden_dim,) — value selector (project V along key direction)

    print(f"  ||d_q|| = {np.linalg.norm(d_q):.4f}")
    print(f"  ||d_k|| = {np.linalg.norm(d_k):.4f}")
    print(f"  ||d_v|| = {np.linalg.norm(d_v):.4f}")
    print(f"  cos(d_q, d_k) = {np.dot(d_q, d_k)/(np.linalg.norm(d_q)*np.linalg.norm(d_k)):.4f}")

    # =========================================================================
    #   Part B: Hidden-space selector — does (h · d_k) pick the right position?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Hidden-space key selector")
    print("=" * 80)
    print("  For each position, compute (rms_norm(h) · d_k). Does argmax match?")

    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == target_layer:
                break
            h = lo(h)

        # Full attention routing (ground truth)
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_exp = np.repeat(K, heads_per_kv, axis=1)
        full_scores = (Q[0, head_idx, -1, :] @ K_exp[0, head_idx, :, :].T)
        full_argmax = int(np.argmax(full_scores))

        # Hidden-space selector: just project normed hidden states onto d_k
        key_features = normed[0] @ d_k  # (seq_len,)
        selector_argmax = int(np.argmax(key_features))

        # Query-key selector: σ₁ × (h_q · d_q) × (h_k · d_k)
        q_feature = float(normed[0, -1, :] @ d_q)
        qk_scores = q_feature * key_features * S[0]
        qk_argmax = int(np.argmax(qk_scores))

        match_k = "✓" if selector_argmax == full_argmax else "✗"
        match_qk = "✓" if qk_argmax == full_argmax else "✗"

        print(f"\n  \"{prompt}\"")
        print(f"    Full attn: → {tokens[full_argmax]}")
        print(f"    Key-only:  → {tokens[selector_argmax]} {match_k}  features: ", end="")
        for pos in range(len(tokens)):
            marker = "←" if pos == selector_argmax else " "
            print(f" {tokens[pos]}={key_features[pos]:+.3f}{marker}", end="")
        print()
        print(f"    Q×K:       → {tokens[qk_argmax]} {match_qk}  (q_feat={q_feature:.3f})")

    # =========================================================================
    #   Part C: End-to-end with hidden-space selected position
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: End-to-end with hidden-space position selector")
    print("=" * 80)
    print("  Select position via (h · d_k), then compute V and output for that position.")

    def run_with_selector(engine, layer_idx, hidden, d_k_dir, W_v_grp, W_o_h6,
                          head_idx_=6):
        """
        Replace head 6's full attention with hidden-space selector:
        1. Select position = argmax(normed_h @ d_k)
        2. Compute V for selected position
        3. Project through W_o for head 6
        4. Add to residual, run MLP
        """
        layer_ = engine.layers[layer_idx]
        attn_ = layer_.attention
        batch, seq_len, hidden_dim_ = hidden.shape

        normed = rms_norm(hidden, attn_.norm_weight)

        # Selector: pick position based on key direction
        key_features = normed[0] @ d_k_dir  # (seq_len,)

        # Use softmax-like selection (soft routing)
        # Scale features to get attention-like weights
        # Temperature: match the scale of real attention scores
        selector_weights = phi_softmax(key_features[np.newaxis, :] * 10.0, axis=-1)
        # (1, seq_len)

        # Compute V for ALL positions (just the KV group, cheaper than full)
        V = phi_linear(attn_.W_v, normed, attn_.b_v)
        V = V.reshape(batch, seq_len, num_kv_heads, head_dim)
        V_group = V[0, :, kv_group, :]  # (seq_len, head_dim)

        # Weighted V: use selector weights
        weighted_V = selector_weights[0, :, np.newaxis] * V_group  # (seq_len, head_dim)
        head_output = np.sum(weighted_V, axis=0)  # (head_dim,)

        # Project through W_o for head 6
        attn_contribution = W_o_h6 @ head_output  # (hidden_dim,)

        post_attn = hidden.copy()
        # Add attention output to all positions (like real attention)
        post_attn[0, -1, :] += attn_contribution

        # Real MLP
        mlp = layer_.mlp
        normed_mlp = rms_norm(post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_hidden = phi_silu(gate) * up
        mlp_out = phi_linear(mlp.W_down, mlp_hidden)

        return post_attn + mlp_out

    def run_with_hard_selector(engine, layer_idx, hidden, d_k_dir, W_v_grp,
                               W_o_h6, head_idx_=6):
        """Hard selection: pick ONE position, use its V directly."""
        layer_ = engine.layers[layer_idx]
        attn_ = layer_.attention
        batch, seq_len, hidden_dim_ = hidden.shape

        normed = rms_norm(hidden, attn_.norm_weight)

        # Hard selector: argmax
        key_features = normed[0] @ d_k_dir
        selected_pos = int(np.argmax(key_features))

        # Compute V only for selected position
        V = phi_linear(attn_.W_v, normed[:, selected_pos:selected_pos+1, :],
                       attn_.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        v_selected = V[0, 0, kv_group, :]  # (head_dim,)

        # Project through W_o
        attn_contribution = W_o_h6 @ v_selected

        post_attn = hidden.copy()
        post_attn[0, -1, :] += attn_contribution

        # Real MLP
        mlp = layer_.mlp
        normed_mlp = rms_norm(post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_hidden = phi_silu(gate) * up
        mlp_out = phi_linear(mlp.W_down, mlp_hidden)

        return post_attn + mlp_out

    # Test different selector strategies
    strategies = [
        ("Soft selector (temp=10)", lambda e, l, h: run_with_selector(
            e, l, h, d_k, W_v_group, W_o_head6)),
        ("Hard selector (argmax)", lambda e, l, h: run_with_hard_selector(
            e, l, h, d_k, W_v_group, W_o_head6)),
    ]

    # Also test with different temperatures
    for temp in [1.0, 5.0, 10.0, 50.0, 100.0]:
        def make_fn(t):
            def fn(e, l, h):
                layer_ = e.layers[l]
                attn_ = layer_.attention
                batch, seq_len, hidden_dim_ = h.shape
                normed = rms_norm(h, attn_.norm_weight)
                key_features = normed[0] @ d_k
                weights = phi_softmax(key_features[np.newaxis, :] * t, axis=-1)
                V = phi_linear(attn_.W_v, normed, attn_.b_v)
                V = V.reshape(batch, seq_len, num_kv_heads, head_dim)
                V_group = V[0, :, kv_group, :]
                weighted_V = weights[0, :, np.newaxis] * V_group
                head_output = np.sum(weighted_V, axis=0)
                attn_contribution = W_o_head6 @ head_output
                post_attn = h.copy()
                post_attn[0, -1, :] += attn_contribution
                mlp = layer_.mlp
                normed_mlp = rms_norm(post_attn, mlp.norm_weight)
                gate = phi_linear(mlp.W_gate, normed_mlp)
                up = phi_linear(mlp.W_up, normed_mlp)
                mlp_hidden = phi_silu(gate) * up
                mlp_out = phi_linear(mlp.W_down, mlp_hidden)
                return post_attn + mlp_out
            return fn
        strategies.append((f"Soft selector (temp={temp})", make_fn(temp)))

    for strat_name, strat_fn in strategies:
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

            sel_out = strat_fn(engine, target_layer, h)

            logits_full = finish_forward(engine, full_out, target_layer)
            logits_sel = finish_forward(engine, sel_out, target_layer)

            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            sel_id, sel_tok, sel_margin = get_top1(logits_sel, tokenizer)

            if sel_id == full_id:
                n_pass += 1
            if 'France' in prompt:
                france_margin = sel_margin
                france_pass = sel_id == full_id

        fp = "✓" if france_pass else "✗"
        fm = f"margin={france_margin:.3f}" if france_margin else ""
        print(f"  {strat_name:>30s}: {n_pass}/6  France={fp} {fm}")

    # =========================================================================
    #   Part D: What about using the FULL hidden state at selected position?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Simplest possible — just use V at selected position")
    print("=" * 80)
    print("  No Q/K projection at all. Just: select pos → V proj → W_o → add to residual")

    # Also test: what if we skip V projection and just mix in the hidden state?
    def run_direct_mix(engine, layer_idx, hidden, d_k_dir, W_v_grp, W_o_h6,
                       alpha=1.0):
        """
        Simplest geometric selector:
        1. Select position = argmax(normed_h · d_k)
        2. V = W_v @ normed_h[selected_pos]
        3. attn_out = W_o_h6 @ V
        4. residual += alpha * attn_out
        5. MLP on residual
        """
        layer_ = engine.layers[layer_idx]
        attn_ = layer_.attention
        batch, seq_len, hidden_dim_ = hidden.shape

        normed = rms_norm(hidden, attn_.norm_weight)
        key_features = normed[0] @ d_k_dir
        selected_pos = int(np.argmax(key_features))

        # V projection at selected position only
        v_in = normed[:, selected_pos:selected_pos+1, :]
        V = phi_linear(attn_.W_v, v_in, attn_.b_v)
        V = V.reshape(1, 1, num_kv_heads, head_dim)
        v_selected = V[0, 0, kv_group, :]

        attn_contribution = alpha * (W_o_head6 @ v_selected)

        post_attn = hidden.copy()
        post_attn[0, -1, :] += attn_contribution

        mlp = layer_.mlp
        normed_mlp = rms_norm(post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_hidden = phi_silu(gate) * up
        mlp_out = phi_linear(mlp.W_down, mlp_hidden)

        return post_attn + mlp_out

    for alpha in [0.5, 1.0, 2.0, 5.0]:
        n_pass = 0
        france_margin = None
        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            tokens = [tokenizer.decode_token(i) for i in p_ids]
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for lo in engine.layers:
                if lo.layer_idx == target_layer:
                    full_out = lo(h.copy())
                    break
                h = lo(h)

            sel_out = run_direct_mix(engine, target_layer, h, d_k, W_v_group,
                                     W_o_head6, alpha=alpha)

            logits_full = finish_forward(engine, full_out, target_layer)
            logits_sel = finish_forward(engine, sel_out, target_layer)

            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            sel_id, sel_tok, sel_margin = get_top1(logits_sel, tokenizer)

            if sel_id == full_id:
                n_pass += 1
            if 'France' in prompt:
                france_margin = sel_margin
                france_pass = sel_id == full_id
                normed_h = rms_norm(h, attn.norm_weight)
                kf = normed_h[0] @ d_k
                sel_pos = int(np.argmax(kf))
                print(f"    France: selected pos={sel_pos} ({tokens[sel_pos]}), "
                      f"→ {sel_tok} (want {full_tok})")

        fp = "✓" if france_pass else "✗"
        fm = f"margin={france_margin:.3f}" if france_margin else ""
        print(f"  alpha={alpha:.1f}: {n_pass}/6  France={fp} {fm}")

    # =========================================================================
    #   Part E: Compute cost comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: Compute cost comparison")
    print("=" * 80)

    seq_len_example = 5
    print(f"\n  For seq_len={seq_len_example}:")
    print(f"\n  Full layer 23 attention (28 heads):")
    full_flops = (
        3 * hidden_dim * (num_heads * head_dim) +  # Q/K/V projections
        num_heads * seq_len_example * head_dim +     # Q·K^T scoring
        num_heads * seq_len_example * head_dim +     # attn @ V
        (num_heads * head_dim) * hidden_dim           # W_o projection
    )
    print(f"    Q/K/V projections: 3 × {hidden_dim} × {num_heads*head_dim} = {3*hidden_dim*(num_heads*head_dim):,}")
    print(f"    Scoring (Q·K^T): {num_heads} × {seq_len_example} × {head_dim} = {num_heads*seq_len_example*head_dim:,}")
    print(f"    Attention @ V: {num_heads} × {seq_len_example} × {head_dim} = {num_heads*seq_len_example*head_dim:,}")
    print(f"    W_o projection: {num_heads*head_dim} × {hidden_dim} = {(num_heads*head_dim)*hidden_dim:,}")
    print(f"    Total: ~{full_flops:,} FLOPs")

    print(f"\n  Hidden-space geometric selector (hard, 1 position):")
    sel_flops = (
        seq_len_example * hidden_dim +   # (h · d_k) for each position
        hidden_dim * head_dim +           # V projection for 1 position (1 KV group)
        hidden_dim * head_dim             # W_o projection for head 6
    )
    print(f"    Selector (h · d_k): {seq_len_example} × {hidden_dim} = {seq_len_example*hidden_dim:,}")
    print(f"    V projection (1 pos): {hidden_dim} × {head_dim} = {hidden_dim*head_dim:,}")
    print(f"    W_o projection (head 6): {hidden_dim} × {head_dim} = {hidden_dim*head_dim:,}")
    print(f"    Total: ~{sel_flops:,} FLOPs")
    print(f"    Reduction: {full_flops/sel_flops:.1f}×")

    print(f"\n  Even simpler (skip V/O, just selector decision):")
    bare_flops = seq_len_example * hidden_dim
    print(f"    Just selector: {seq_len_example} × {hidden_dim} = {bare_flops:,}")
    print(f"    Reduction: {full_flops/bare_flops:.0f}×")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
