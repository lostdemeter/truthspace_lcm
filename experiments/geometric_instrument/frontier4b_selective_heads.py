"""
Frontier 4b: Selective Head Caching
=====================================
F142 found: L0-L19 needs real Q/K, L20-L27 can be cached universally.
But F142 also showed some L0-L19 heads are already universal (cos=0.97).

Questions:
1. Per-head cross-structure cos map for ALL 28 layers × 28 heads
2. What fraction of L0-L19 heads can be cached universally?
3. If we cache universal heads + real Q/K for sensitive heads only,
   what's the accuracy? What's the compute savings?
4. Can early-layer attention be predicted from embeddings alone?
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


def get_full_attention(engine, h, li):
    """Extract full attention [nh, seq, seq]."""
    layer = engine.layers[li]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    return phi_softmax(scores, axis=-1)[0]


def main():
    print("=" * 80)
    print("  Frontier 4b: Selective Head Caching")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    nh = 28
    print(f" done in {time.time()-t0:.1f}s")

    # Diverse N=5 prompts — mix of structures
    prompts = [
        'The capital of France is',
        'The capital of Germany is',
        'The capital of Japan is',
        'The capital of Italy is',
        'The capital of Spain is',
        'The capital of Egypt is',
        'I really love eating pizza',
        'Please help me find this',
        'Once upon a time there',
        'How does the engine work',
    ]

    # Verify all N=5
    working = []
    for prompt in prompts:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working.append((prompt, tids))
    print(f"  Using {len(working)} N=5 prompts")

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: Full Head Sensitivity Map
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: Full Head Sensitivity Map (28 layers × 28 heads)")
    print("=" * 80)

    # Extract attention for all prompts at all layers
    all_attn = {}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        layers = {}
        for li in range(n_layers):
            layers[li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        all_attn[prompt] = layers
        gc.collect()
    print("  All attention extracted")

    # Separate same-structure vs different-structure
    capital_prompts = [p for p, _ in working if 'capital' in p]
    diverse_prompts = [p for p, _ in working if 'capital' not in p]
    all_prompt_names = [p for p, _ in working]

    # Compute per-head cross-structure cosine
    # For each (layer, head), compute mean cosine across ALL prompt pairs
    head_cos = np.zeros((n_layers, nh))
    for li in range(n_layers):
        for hi in range(nh):
            cos_vals = []
            for i in range(len(all_prompt_names)):
                for j in range(i + 1, len(all_prompt_names)):
                    a = all_attn[all_prompt_names[i]][li][hi].ravel()
                    b = all_attn[all_prompt_names[j]][li][hi].ravel()
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    cos_vals.append(cos)
            head_cos[li, hi] = np.mean(cos_vals)

    # Print summary map
    print("\n  Cross-structure cos by layer (min, mean, max across 28 heads):")
    print(f"  {'Layer':>6} | {'Min':>6} {'Mean':>6} {'Max':>6} | {'Heads≥0.95':>10} {'Heads≥0.99':>10} {'Heads<0.80':>10}")
    for li in range(n_layers):
        row = head_cos[li]
        n95 = int(np.sum(row >= 0.95))
        n99 = int(np.sum(row >= 0.99))
        n80 = int(np.sum(row < 0.80))
        print(f"  L{li:>4} | {row.min():.4f} {row.mean():.4f} {row.max():.4f} | "
              f"{n95:>10} {n99:>10} {n80:>10}")

    # Total cacheable heads at various thresholds
    for thresh in [0.95, 0.97, 0.99]:
        n_cacheable = int(np.sum(head_cos >= thresh))
        total = n_layers * nh
        pct = 100 * n_cacheable / total
        # In L0-L19 specifically
        n_early = int(np.sum(head_cos[:20] >= thresh))
        total_early = 20 * nh
        pct_early = 100 * n_early / total_early
        print(f"\n  Heads with cos ≥ {thresh}: {n_cacheable}/{total} ({pct:.1f}%) total, "
              f"{n_early}/{total_early} ({pct_early:.1f}%) in L0-L19")

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Which specific heads are sensitive?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: Most Sensitive Heads (cos < 0.90)")
    print("=" * 80)

    sensitive_heads = []
    for li in range(n_layers):
        for hi in range(nh):
            if head_cos[li, hi] < 0.90:
                sensitive_heads.append((li, hi, head_cos[li, hi]))

    sensitive_heads.sort(key=lambda x: x[2])
    print(f"\n  Total sensitive heads (cos < 0.90): {len(sensitive_heads)}")
    for li, hi, cos in sensitive_heads[:30]:
        print(f"    L{li:2d} H{hi:2d}: cos = {cos:.4f}")

    # Group by layer
    print(f"\n  Sensitive heads per layer:")
    for li in range(n_layers):
        sens = [(hi, c) for l, hi, c in sensitive_heads if l == li]
        if sens:
            heads_str = ", ".join(f"H{hi}({c:.2f})" for hi, c in sens)
            print(f"    L{li:2d}: {len(sens)} heads — {heads_str}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Selective Caching Test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Selective Head Caching — Accuracy Test")
    print("=" * 80)

    # Compute mean template across all prompts
    mean_template = {}
    for li in range(n_layers):
        templates = [all_attn[p][li] for p in all_prompt_names]
        mean_template[li] = np.mean(templates, axis=0)

    # Compute BOS sv0
    print("  Computing BOS sv0...", end="", flush=True)
    cal_tids = tokenizer.encode('The capital of France is')
    h_cal = engine.embedding(cal_tids)[np.newaxis, :, :]
    bos_mlp = {}
    for li in range(n_layers):
        layer = engine.layers[li]
        attn = layer.attention
        mlp = layer.mlp
        nhl, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nhl // nkv, attn.head_dim
        sl = h_cal.shape[1]
        normed = rms_norm(h_cal, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nhl, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h_cal + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        bos_mlp[li] = mlp_out[0, 0, :].copy()
        h_cal = h_pa + mlp_out

    def get_sv0_direction(engine, li):
        W_down = decode_weight(engine.layers[li].mlp.W_down)
        rng = np.random.RandomState(42)
        v = rng.randn(W_down.shape[1]).astype(np.float64)
        for _ in range(20):
            u = W_down.astype(np.float64) @ v
            u /= np.linalg.norm(u)
            v = W_down.astype(np.float64).T @ u
            v /= np.linalg.norm(v)
        return u.astype(np.float32)

    synth_sv0 = {}
    for li in range(n_layers):
        sv0 = get_sv0_direction(engine, li)
        if np.dot(sv0, bos_mlp[li]) < 0:
            sv0 = -sv0
        scale = float(np.dot(bos_mlp[li], sv0))
        synth_sv0[li] = scale * sv0
    print(" done")

    def run_selective_cache(engine, tids, mean_template, synth_sv0,
                            cos_threshold, head_cos_map, all_attn_for_prompt=None):
        """
        Run forward pass where:
        - Heads with cross-struct cos >= threshold use mean cached template
        - Heads below threshold use real Q/K (or prompt-specific cache if provided)
        - BOS MLP uses sv0
        """
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(len(engine.layers)):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp
            nhl, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nhl // nkv, attn.head_dim
            sl = h.shape[1]

            normed = rms_norm(h, attn.norm_weight)

            # V is always computed (needed for value)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            Ve = np.repeat(V, hpk, axis=1)

            # Build attention: selective per head
            w = np.zeros((1, nhl, sl, sl), dtype=np.float32)

            # Check if any heads need real Q/K at this layer
            needs_real = [hi for hi in range(nhl) if head_cos_map[li, hi] < cos_threshold]

            if needs_real:
                # Compute real Q/K
                Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nhl, hd).transpose(0, 2, 1, 3)
                K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
                Q, K = attn.rope.apply(Q), attn.rope.apply(K)
                Ke = np.repeat(K, hpk, axis=1)
                scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
                if sl > 1:
                    scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
                real_attn = phi_softmax(scores, axis=-1)

                for hi in needs_real:
                    w[0, hi] = real_attn[0, hi]

            # Cached heads use mean template
            for hi in range(nhl):
                if head_cos_map[li, hi] >= cos_threshold:
                    w[0, hi] = mean_template[li][hi]

            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
            h_pa = h + phi_linear(attn.W_o, ao)

            nm = rms_norm(h_pa, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
            mlp_out[0, 0, :] = synth_sv0[li]
            h = h_pa + mlp_out

        return h

    # Test at various thresholds
    facts = {
        'France':  ('The capital of France is', ' Paris'),
        'Japan':   ('The capital of Japan is', ' Tokyo'),
        'Germany': ('The capital of Germany is', ' Berlin'),
        'Italy':   ('The capital of Italy is', ' Rome'),
        'Spain':   ('The capital of Spain is', ' Madrid'),
        'Egypt':   ('The capital of Egypt is', ' Cairo'),
    }

    diverse_tests = [
        ('I really love eating pizza', ','),
        ('Please help me find this', ' limit'),
        ('Once upon a time there', ' was'),
        ('How does the engine work', ' in'),
    ]

    fnw = decode_weight(engine.final_norm_weight)

    for thresh in [0.99, 0.97, 0.95, 0.90, 0.85, 0.80]:
        n_cached = int(np.sum(head_cos >= thresh))
        n_real = n_layers * nh - n_cached
        n_layers_needing_qk = len(set(li for li in range(n_layers) for hi in range(nh) if head_cos[li, hi] < thresh))
        pct_cached = 100 * n_cached / (n_layers * nh)

        # Test on capitals
        cap_correct = 0
        for country, (prompt, answer) in facts.items():
            tids = tokenizer.encode(prompt)
            h = run_selective_cache(engine, tids, mean_template, synth_sv0, thresh, head_cos)
            normed = rms_norm(h[:, -1:, :], fnw)
            logits = engine.lm_head(normed)[0, 0, :]
            ans_tid = tokenizer.encode(answer)[0]
            rank = int(np.sum(logits > logits[ans_tid]))
            if rank == 0:
                cap_correct += 1

        # Test on diverse
        div_correct = 0
        for prompt, expected in diverse_tests:
            tids = tokenizer.encode(prompt)
            h = run_selective_cache(engine, tids, mean_template, synth_sv0, thresh, head_cos)
            normed = rms_norm(h[:, -1:, :], fnw)
            logits = engine.lm_head(normed)[0, 0, :]

            # Baseline
            h_real = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                h_real = engine.layers[li](h_real)
            normed_r = rms_norm(h_real[:, -1:, :], fnw)
            logits_r = engine.lm_head(normed_r)[0, 0, :]
            real_tok = int(np.argmax(logits_r))
            rank = int(np.sum(logits > logits[real_tok]))
            if rank == 0:
                div_correct += 1

        print(f"  cos≥{thresh}: {n_cached:3d}/{n_layers*nh} heads cached ({pct_cached:.0f}%), "
              f"{n_layers_needing_qk} layers need Q/K | "
              f"capitals={cap_correct}/6, diverse={div_correct}/4")

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Layer-0 Attention from Embeddings
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: Can L0 Attention Be Predicted from Embeddings?")
    print("=" * 80)

    # At L0, hidden state = embedding. Q and K are linear projections of embedding.
    # So L0 attention is a deterministic function of the token IDs.
    # If we can predict L0 attention from token IDs, we can skip L0 Q/K.

    # Key insight: at L0, h = embedding[token_id]. So Q = W_q @ emb + b_q.
    # The attention score for (q, k) depends on:
    #   score = (W_q @ emb[q] + b_q) . (W_k @ emb[k] + b_k) / sqrt(d)
    # with RoPE applied based on positions q, k.

    # This means L0 attention is fully determined by the token IDs.
    # We can pre-compute (W_q @ emb[tid]) for each token in vocabulary.

    print("\n  L0 attention = f(token_ids, positions)")
    print("  Pre-computing Q/K for each vocabulary token...")

    layer0 = engine.layers[0]
    attn0 = layer0.attention
    nhl, nkv = attn0.num_heads, attn0.num_kv_heads
    hpk, hd = nhl // nkv, attn0.head_dim

    # Get embedding matrix
    emb_matrix = decode_weight(engine.embedding_weight)  # [vocab, dim]
    vocab_size = emb_matrix.shape[0]
    print(f"  Vocabulary size: {vocab_size}")

    # For a small test: compute L0 attention from pre-computed token Q/K
    # Step 1: Pre-compute Q_per_token = (W_q @ emb[tid] + b_q) for each token
    # This is expensive for full vocab but we only need the tokens we use

    test_prompts_tids = {p: tokenizer.encode(p) for p, _ in working}
    unique_tids = set()
    for tids in test_prompts_tids.values():
        unique_tids.update(tids)
    unique_tids = sorted(unique_tids)
    print(f"  Unique tokens across test set: {len(unique_tids)}")

    # Pre-compute Q and K for each unique token
    # Q = W_q @ norm(emb[tid]) + b_q, but we need RMSNorm which depends on the FULL sequence
    # So we can't fully decouple tokens... UNLESS we can pre-compute per-token norms.

    # Actually at L0, the input is just the embedding. RMSNorm is applied to
    # each position independently: norm(emb[tid]) = emb[tid] / rms(emb[tid]).
    # So for L0, we CAN pre-compute per-token.

    norm_weight = attn0.norm_weight
    token_q = {}  # tid -> Q vector [nh, hd]
    token_k = {}  # tid -> K vector [nkv, hd]

    for tid in unique_tids:
        emb = emb_matrix[tid:tid+1, :]  # [1, dim]
        # RMSNorm
        normed = rms_norm(emb[np.newaxis, :, :], norm_weight)[0]  # [1, dim]
        q = (phi_linear(attn0.W_q, normed, attn0.b_q)
             .reshape(1, nhl, hd))  # [1, nh, hd]
        k = (phi_linear(attn0.W_k, normed, attn0.b_k)
             .reshape(1, nkv, hd))  # [1, nkv, hd]
        token_q[tid] = q[0]  # [nh, hd]
        token_k[tid] = k[0]  # [nkv, hd]

    print(f"  Pre-computed Q/K for {len(unique_tids)} tokens")

    # Now reconstruct L0 attention from pre-computed Q/K + RoPE
    print("\n  Reconstructing L0 attention from token Q/K cache:")
    for prompt, tids in list(test_prompts_tids.items())[:4]:
        sl = len(tids)
        # Assemble Q and K from per-token cache
        Q = np.zeros((1, nhl, sl, hd), dtype=np.float32)
        K = np.zeros((1, nkv, sl, hd), dtype=np.float32)
        for pos, tid in enumerate(tids):
            Q[0, :, pos, :] = token_q[tid]
            K[0, :, pos, :] = token_k[tid]

        # Apply RoPE
        Q, K = attn0.rope.apply(Q), attn0.rope.apply(K)
        Ke = np.repeat(K, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn0.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        recon_attn = phi_softmax(scores, axis=-1)[0]

        # Compare to real
        real_attn = all_attn[prompt][0]
        cos = float(np.sum(recon_attn * real_attn) /
                     (np.linalg.norm(recon_attn) * np.linalg.norm(real_attn) + 1e-12))
        max_diff = float(np.max(np.abs(recon_attn - real_attn)))
        short = prompt[:35].ljust(35)
        print(f"    {short} cos={cos:.6f}, max_diff={max_diff:.6f}")

    # Size of token Q/K cache
    per_token_bytes = (nhl * hd + nkv * hd) * 4  # float32
    full_vocab_bytes = vocab_size * per_token_bytes
    print(f"\n  Token Q/K cache size:")
    print(f"    Per token: {per_token_bytes} bytes ({per_token_bytes/1024:.1f} KB)")
    print(f"    Full vocab ({vocab_size}): {full_vocab_bytes/1024/1024:.1f} MB")
    print(f"    vs Q+K weight matrices: {2 * 3072 * 3072 * 4 / 1024/1024:.1f} MB")

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Extend to All Layers — Token-Level Q/K Cache
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 5: Can We Extend Token Q/K Cache Beyond L0?")
    print("=" * 80)

    # At L0, input = embedding (token-specific). ✓
    # At L1+, input = hidden state (sequence-specific). ✗
    # BUT: the hidden state at each position depends on attention routing.
    # If attention routing is structure-dependent, so is the hidden state.
    # This means L1+ Q/K cannot be pre-computed per-token.
    #
    # HOWEVER: What if we decompose the hidden state?
    # h[pos] = emb[pos] + Σ(attention_effects) + Σ(mlp_effects)
    # The attention effects are the "structure-dependent" part.
    # Could we separate structure from content?

    # Let's check: how much of the hidden state at L1+ is explained by
    # the embedding alone?
    print("\n  Hidden state explained by embedding at each layer:")
    for prompt, tids in list(test_prompts_tids.items())[:3]:
        h = engine.embedding(tids)[np.newaxis, :, :]
        emb_orig = h.copy()
        short = prompt[:30].ljust(30)

        for li in range(min(n_layers, 10)):
            # Cosine of current hidden state with original embedding
            cos_per_pos = []
            for pos in range(1, len(tids)):  # skip BOS
                a = h[0, pos, :]
                b = emb_orig[0, pos, :]
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                cos_per_pos.append(cos)

            if li in [0, 1, 2, 3, 5, 9]:
                mean_cos = np.mean(cos_per_pos)
                print(f"    {short} L{li}: mean cos(h, emb) = {mean_cos:.4f} "
                      f"[{', '.join(f'{c:.3f}' for c in cos_per_pos)}]")

            h = engine.layers[li](h)

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary: Selective Head Caching")
    print("=" * 80)
    print("""
  Key findings:
  1. Head sensitivity map reveals which heads are universal vs structure-dependent
  2. Selective caching at various thresholds shows accuracy/compute tradeoff
  3. L0 attention CAN be pre-computed from per-token Q/K cache
  4. L1+ hidden states diverge from embeddings — token-level cache only works at L0

  For a general solver:
  - L0: Token Q/K cache (no runtime Q/K needed)
  - L1-L19: Only sensitive heads need real Q/K
  - L20-L27: Universal mean template (no Q/K needed)
  - BOS at all layers: sv0 pump (no MLP needed)
""")


if __name__ == '__main__':
    main()
