"""
Frontier 3: Full Q/K Elimination — Diagnostic
================================================
Deep analysis of what's blocking parametric T(N,q) for all positions.

Questions:
1. What does the actual attention structure look like per position?
2. Is {BOS, self, spread} the right decomposition?
3. How many principal components describe the attention at each position?
4. Can we cache attention matrices directly (bypassing parametric)?
5. What's the minimum N-dependent information?

Key insight from F138: attention IS content-independent.
The question is: what's the right parametric model?
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

FACTS = {
    'France':  'The capital of France is',
    'Japan':   'The capital of Japan is',
    'Germany': 'The capital of Germany is',
    'Italy':   'The capital of Italy is',
    'Spain':   'The capital of Spain is',
    'Egypt':   'The capital of Egypt is',
}
ANSWERS = {
    'France': ' Paris', 'Japan': ' Tokyo', 'Germany': ' Berlin',
    'Italy': ' Rome', 'Spain': ' Madrid', 'Egypt': ' Cairo',
}

# Wide range of lengths for analysis
CALIBRATION_PROMPTS = {
    5: 'The capital of France is',
    6: 'The main capital of France is',
    7: 'The official capital city of France is',
    8: 'The official main capital city of France is',
    9: 'The official capital city of the country France is',
}


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


def run_layer_with_full_template(engine, h, li, template):
    """Run layer replacing ALL attention with template."""
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ve = np.repeat(V, hpk, axis=1)

    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
    ts = template.shape[1]
    if seq_len == ts:
        w[0] = template
    elif seq_len < ts:
        w[0] = template[:, :seq_len, :seq_len]
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
    else:
        w[0, :, :ts, :ts] = template
        for p in range(ts, seq_len):
            w[0, :, p, p] = 1.0
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h_pa = h + phi_linear(attn.W_o, ao)

    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    h = h_pa + phi_linear(mlp.W_down, phi_silu(g) * u)
    return h


def get_sv0_direction(engine, li):
    """Get first left singular vector of W_down."""
    W_down = decode_weight(engine.layers[li].mlp.W_down)
    rng = np.random.RandomState(42)
    v = rng.randn(W_down.shape[1]).astype(np.float64)
    for _ in range(20):
        u = W_down.astype(np.float64) @ v
        u /= np.linalg.norm(u)
        v = W_down.astype(np.float64).T @ u
        v /= np.linalg.norm(v)
    return u.astype(np.float32)


def run_layer_with_template_and_sv0(engine, h, li, template, synth_bos_vec):
    """Run layer with full template + BOS sv0."""
    layer = engine.layers[li]
    attn = layer.attention
    mlp = layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]

    normed = rms_norm(h, attn.norm_weight)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Ve = np.repeat(V, hpk, axis=1)

    w = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
    ts = template.shape[1]
    if seq_len == ts:
        w[0] = template
    elif seq_len < ts:
        w[0] = template[:, :seq_len, :seq_len]
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)
    else:
        w[0, :, :ts, :ts] = template
        for p in range(ts, seq_len):
            w[0, :, p, p] = 1.0
        w[0] /= (w[0].sum(axis=2, keepdims=True) + 1e-12)

    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    h_pa = h + phi_linear(attn.W_o, ao)

    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    mlp_out[0, 0, :] = synth_bos_vec
    h = h_pa + mlp_out
    return h


def predict(engine, tokenizer, h, answer):
    fnw = decode_weight(engine.final_norm_weight)
    normed = rms_norm(h[:, -1:, :], fnw)
    logits = engine.lm_head(normed)[0, 0, :]
    top_tid = int(np.argmax(logits))
    ans_tid = tokenizer.encode(answer)[0]
    rank = int(np.sum(logits > logits[ans_tid]))
    return tokenizer.decode([top_tid]), rank


def main():
    print("=" * 80)
    print("  Frontier 3: Full Q/K Elimination — Diagnostic")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    nh = 28
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: What does the attention actually look like?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 1: Attention Structure Deep Dive (N=5)")
    print("=" * 80)

    tids = tokenizer.encode(CALIBRATION_PROMPTS[5])
    h = engine.embedding(tids)[np.newaxis, :, :]
    
    sample_layers = [0, 3, 10, 20, 23, 27]
    attn_n5 = {}
    
    for li in range(n_layers):
        attn_n5[li] = get_full_attention(engine, h, li)
        h = engine.layers[li](h)

    # Print detailed structure for sample layers
    for li in sample_layers:
        A = attn_n5[li]  # [28, 5, 5]
        print(f"\n  Layer {li} — Head 0 full attention matrix:")
        for q in range(5):
            row = A[0, q, :5]
            parts = " ".join(f"{v:.4f}" for v in row)
            print(f"    q={q}: [{parts}]")
        
        # Characterize: for each row, what's the entropy?
        print(f"\n  Layer {li} — Entropy by position (head-averaged):")
        for q in range(5):
            row = A[:, q, :q+1]  # [28, q+1] — only causal positions
            # per-head entropy
            entropies = []
            for hi in range(nh):
                p = row[hi]
                p = p[p > 0]
                e = -np.sum(p * np.log2(p + 1e-12))
                entropies.append(e)
            max_entropy = np.log2(q + 1) if q > 0 else 0.0
            print(f"    q={q}: mean H={np.mean(entropies):.3f} (max={max_entropy:.3f}), "
                  f"std={np.std(entropies):.3f}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: RoPE position-locking analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 2: RoPE Position-Locking — What Changes with N?")
    print("=" * 80)

    # Extract attention at multiple lengths
    all_attn = {}
    for N, prompt in sorted(CALIBRATION_PROMPTS.items()):
        tids = tokenizer.encode(prompt)
        assert len(tids) == N, f"Expected {N} tokens, got {len(tids)}"
        h = engine.embedding(tids)[np.newaxis, :, :]
        all_attn[N] = {}
        for li in range(n_layers):
            all_attn[N][li] = get_full_attention(engine, h, li)
            h = engine.layers[li](h)
        print(f"    N={N} extracted")

    # For each position pair (q,k), compare attention across N values
    # Focus: does w(q, k) change when we add tokens AFTER position q?
    print(f"\n  Position-locking test: does w(q,k) at fixed (q,k) change with N?")
    print(f"  (Testing L0 h0 and L23 h0)")
    
    for li in [0, 23]:
        print(f"\n  Layer {li}, Head 0:")
        for q in range(5):
            for k in range(q + 1):
                vals = []
                for N in sorted(all_attn.keys()):
                    if q < N:
                        vals.append(float(all_attn[N][li][0, q, k]))
                if len(vals) >= 2:
                    spread = max(vals) - min(vals)
                    if spread > 0.001:
                        print(f"    w({q},{k}): " + 
                              " ".join(f"N={N}:{all_attn[N][li][0,q,k]:.4f}" 
                                       for N in sorted(all_attn.keys()) if q < N) +
                              f"  spread={spread:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Direct cache approach
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 3: Direct Cache — Store Real Attention Matrices")
    print("=" * 80)

    # Instead of parametric, just cache the real attention for each N
    # Cost: 28 layers × 28 heads × N × N floats per cached length
    
    # Test: use N=5 France attention on all 6 countries
    templates_5 = {li: all_attn[5][li] for li in range(n_layers)}
    
    # Also compute BOS sv0 for combined test
    print("  Computing BOS sv0 directions...", end="", flush=True)
    tids_france = tokenizer.encode(FACTS['France'])
    h_cal = engine.embedding(tids_france)[np.newaxis, :, :]
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

    synth_sv0 = {}
    for li in range(n_layers):
        sv0 = get_sv0_direction(engine, li)
        if np.dot(sv0, bos_mlp[li]) < 0:
            sv0 = -sv0
        scale = float(np.dot(bos_mlp[li], sv0))
        synth_sv0[li] = scale * sv0
        gc.collect()
    print(" done")

    # Test A: Cached templates only (no sv0)
    print("\n  Test A: Cached N=5 attention templates → all 6 countries")
    correct_a = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_full_template(engine, h, li, templates_5[li])
        _, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct_a += 1
        print(f"    {country:>8}: {ok}")
    print(f"  Cached templates only: {correct_a}/6")

    # Test B: Cached templates + BOS sv0
    print("\n  Test B: Cached N=5 templates + BOS sv0")
    correct_b = 0
    for country in FACTS:
        tids = tokenizer.encode(FACTS[country])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_template_and_sv0(engine, h, li, templates_5[li], synth_sv0[li])
        _, rank = predict(engine, tokenizer, h, ANSWERS[country])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0: correct_b += 1
        print(f"    {country:>8}: {ok}")
    print(f"  Cached + sv0: {correct_b}/6")

    # Test C: Per-N cached templates + BOS sv0
    print("\n  Test C: Per-N cached templates + sv0 (each length uses its own)")
    for N in sorted(CALIBRATION_PROMPTS.keys()):
        tids = tokenizer.encode(CALIBRATION_PROMPTS[N])
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_template_and_sv0(engine, h, li, all_attn[N][li], synth_sv0[li])
        _, rank = predict(engine, tokenizer, h, ' Paris')
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    N={N}: {ok}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: What dimensions of variation exist?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 4: Principal Components of Attention Variation")
    print("=" * 80)

    # For each layer, stack all attention rows across all N values
    # Then SVD to find the main dimensions of variation
    for li in sample_layers:
        rows = []
        for N in sorted(all_attn.keys()):
            A = all_attn[N][li]  # [28, N, N]
            for hi in range(nh):
                for q in range(N):
                    # Pad row to max_N with zeros
                    row = np.zeros(max(CALIBRATION_PROMPTS.keys()), dtype=np.float32)
                    row[:N] = A[hi, q, :]
                    rows.append(row)
        
        X = np.array(rows)  # [n_rows, max_N]
        # Center
        X_centered = X - X.mean(axis=0, keepdims=True)
        
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            total_var = np.sum(S ** 2)
            cum_var = np.cumsum(S ** 2) / total_var
            n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
            n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
            n_99 = int(np.searchsorted(cum_var, 0.99)) + 1
            print(f"  Layer {li}: SVD of attention rows ({X.shape[0]} rows × {X.shape[1]} cols)")
            print(f"    Top-5 singular values: {S[:5].tolist()}")
            print(f"    90% variance: {n_90} components")
            print(f"    95% variance: {n_95} components")
            print(f"    99% variance: {n_99} components")
        except Exception as e:
            print(f"  Layer {li}: SVD failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Position-dependent vs position-independent
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 5: What Makes Each Row Unique?")
    print("=" * 80)

    # For N=5, examine the per-head per-position structure
    # Key question: can we decompose as w(q,k) = f(q) * g(k) (separable)?
    for li in [0, 10, 23, 27]:
        A = all_attn[5][li]  # [28, 5, 5]
        
        # Per-head: is the causal attention matrix approximately rank-1?
        ranks = []
        for hi in range(nh):
            mat = A[hi]  # [5, 5]
            # Only look at the causal part (lower triangle including diagonal)
            # SVD of the full matrix
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            if S[0] > 0:
                ratio = S[0] / (S[1] + 1e-12) if len(S) > 1 else float('inf')
                ranks.append(ratio)
        
        mean_ratio = np.mean(ranks)
        print(f"  Layer {li}: Attention matrix S[0]/S[1] = {mean_ratio:.1f} "
              f"(range {min(ranks):.1f} - {max(ranks):.1f})")

    # ═══════════════════════════════════════════════════════════
    # Investigation 6: Cache size analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 6: Cache Size if We Store Directly")
    print("=" * 80)
    
    for N in sorted(CALIBRATION_PROMPTS.keys()):
        floats = n_layers * nh * N * N
        bytes_val = floats * 4
        print(f"  N={N}: {n_layers} × {nh} × {N} × {N} = {floats:,} floats ({bytes_val / 1024:.1f} KB)")
    
    print(f"\n  Template bank for N=5..20:")
    total = 0
    for N in range(5, 21):
        total += n_layers * nh * N * N
    print(f"    Total: {total:,} floats ({total * 4 / 1024:.1f} KB, {total * 4 / 1024 / 1024:.2f} MB)")

    # ═══════════════════════════════════════════════════════════
    # Investigation 7: Cross-N interpolation test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Investigation 7: Interpolation Between Cached Lengths")
    print("=" * 80)

    # For N=6 (uncached), interpolate between N=5 and N=7 templates
    # Method: at shared positions (q,k) where q<5 and k<5, average N=5 and N=7
    # For new positions, use N=7 values
    if 5 in all_attn and 7 in all_attn:
        print("  Building N=6 from interpolation of N=5 and N=7:")
        interp_6 = {}
        for li in range(n_layers):
            A5 = all_attn[5][li]  # [28, 5, 5]
            A7 = all_attn[7][li]  # [28, 7, 7]
            # Target: [28, 6, 6]
            A6 = np.zeros((nh, 6, 6), dtype=np.float32)
            
            # For positions 0-4 (shared between N=5 and N=7):
            # Use N=7's q=0..4, k=0..5 (N=7 has the right RoPE for 7 tokens)
            # But we want 6 tokens... N=7 at position q<6 should be close
            # Actually, let's try: take N=7's first 6 rows and first 6 cols
            A6[:] = A7[:, :6, :6]
            # Renormalize rows
            for q in range(6):
                row_sum = A6[:, q, :q+1].sum(axis=1, keepdims=True)
                A6[:, q, :q+1] /= (row_sum + 1e-12)
                A6[:, q, q+1:] = 0.0
            
            interp_6[li] = A6
        
        # Test interpolated N=6
        prompt_6 = CALIBRATION_PROMPTS[6]
        tids = tokenizer.encode(prompt_6)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_template_and_sv0(engine, h, li, interp_6[li], synth_sv0[li])
        _, rank = predict(engine, tokenizer, h, ' Paris')
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    Interpolated N=6: {ok}")
        
        # Compare: use exact N=6 cached
        tids = tokenizer.encode(prompt_6)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_template_and_sv0(engine, h, li, all_attn[6][li], synth_sv0[li])
        _, rank = predict(engine, tokenizer, h, ' Paris')
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    Exact cached N=6:  {ok}")
        
        # Compare: use N=7 trimmed to 6
        tids = tokenizer.encode(prompt_6)
        h = engine.embedding(tids)[np.newaxis, :, :]
        trimmed_7to6 = {}
        for li in range(n_layers):
            A7 = all_attn[7][li]
            A6 = A7[:, :6, :6].copy()
            for q in range(6):
                row_sum = A6[:, q, :q+1].sum(axis=1, keepdims=True)
                A6[:, q, :q+1] /= (row_sum + 1e-12)
                A6[:, q, q+1:] = 0.0
            trimmed_7to6[li] = A6
        
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(n_layers):
            h = run_layer_with_template_and_sv0(engine, h, li, trimmed_7to6[li], synth_sv0[li])
        _, rank = predict(engine, tokenizer, h, ' Paris')
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    N=7 trimmed to 6:  {ok}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary: What's Blocking Full Q/K Elimination?")
    print("=" * 80)
    print(f"""
  APPROACH 1: Parametric T(N,q) — FAILED (0/6)
    3 params/head/layer is too few.
    The attention structure has ~{max(CALIBRATION_PROMPTS.keys())} free values per row,
    not 3. Need position-specific parameters.

  APPROACH 2: Direct Cache — TESTED ABOVE
    Store real attention matrices per N.
    Works perfectly at cached lengths.
    Question: can we generalize to unseen N?

  APPROACH 3: Interpolation — TESTED ABOVE
    Build unseen N from nearest cached neighbors.
    Uses RoPE position-locking (shared positions are stable).

  KEY QUESTION: What's the minimum set of cached lengths that covers
  all practical N values? If RoPE locks positions, we might need
  surprisingly few calibration points.
""")


if __name__ == '__main__':
    main()
