"""
Explore the LM Head: Offload + Geometric Structure Analysis

The LM head is a (152064, 3584) matmul — 7.4s on local CPU.
Two questions:
  1. Can we offload it to gimli for immediate speedup?
  2. Can we navigate to argmax geometrically without 152K dot products?

The second question connects to the attention spigot insight:
  - Attention: don't compute all N² scores, navigate via φ-lattice → O(N)
  - LM head: don't compute all 152K logits, navigate via φ-structure → O(?)

Key observation: we only need argmax (or top-k). Computing ALL logits
is like computing full attention — most of the work is wasted.

Usage:
    python explore_lm_head.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from phi_geometric.inference.phi_integer import float_to_phi, phi_to_float
from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

# ──────────────────────────────────────────────────────────────────
# Part 1: Understand what the LM head actually does
# ──────────────────────────────────────────────────────────────────
def analyze_lm_head_structure(engine):
    """Analyze the geometric structure of the LM head weight matrix."""
    print("=" * 70)
    print("  Part 1: LM Head Structure Analysis")
    print("=" * 70)

    W = engine.lm_head.weight  # PhiEncoded (152064, 3584)
    print(f"  Shape: {W.shape}")
    print(f"  Vocab size: {W.shape[0]}")
    print(f"  Hidden dim: {W.shape[1]}")
    print(f"  Storage: signs={W.signs.nbytes/1e6:.0f} MB, "
          f"exps={W.exponents.nbytes/1e6:.0f} MB, "
          f"total={W.signs.nbytes/1e6 + W.exponents.nbytes/1e6:.0f} MB")

    # Sign structure
    print(f"\n  Sign distribution:")
    n_pos = np.sum(W.signs == 1)
    n_neg = np.sum(W.signs == -1)
    total = W.signs.size
    print(f"    +1: {n_pos/total*100:.1f}%  -1: {n_neg/total*100:.1f}%")

    # Exponent structure
    print(f"\n  Exponent statistics:")
    print(f"    Min: {W.exponents.min()}")
    print(f"    Max: {W.exponents.max()}")
    print(f"    Mean: {W.exponents.mean():.1f}")
    print(f"    Std: {W.exponents.std():.1f}")

    # Per-row (per-token) exponent norms
    row_mean_exp = W.exponents.mean(axis=1)
    print(f"\n  Per-token exponent mean:")
    print(f"    Min row mean: {row_mean_exp.min():.1f}")
    print(f"    Max row mean: {row_mean_exp.max():.1f}")
    print(f"    Std of row means: {row_mean_exp.std():.1f}")

    return W


def analyze_sign_patterns(W, engine, tokenizer):
    """Can we use sign patterns for fast navigation?"""
    print("\n" + "=" * 70)
    print("  Part 2: Sign-Based Navigation")
    print("=" * 70)

    # The key insight: in φ-encoding, the SIGN carries the direction
    # and the EXPONENT carries the magnitude. For argmax, the direction
    # (sign pattern) matters most.
    #
    # If we compute sign(h) · sign(W[i]) for each vocab token i,
    # we get a "direction match score" — tokens whose sign pattern
    # aligns with h will have high logits.

    # Test with actual prompts
    prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("The color of the sky is", "blue"),
        ("One plus one equals", "two"),
        ("The chemical symbol for gold is", "Au"),
    ]

    print(f"\n  Strategy: use sign(h) · sign(W) as a CHEAP filter")
    print(f"  Sign dot product: {W.shape[1]} XORs + popcount = O(d) per token")
    print(f"  Full dot product: {W.shape[1]} multiply-adds = O(d) per token")
    print(f"  But sign dot is ~10× faster (int8 vs float32+decode)")

    W_signs = W.signs  # (152064, 3584) int8
    W_decoded = W.decode_cached()

    for prompt, expected in prompts:
        tokens = tokenizer.encode(prompt)
        hidden = engine.embedding(tokens)
        hidden = hidden[np.newaxis, :, :]

        # Run through model (use float forward for speed)
        for layer in engine.layers:
            hidden = layer(hidden, pure=False)

        from phi_geometric.inference.phi_components import rms_norm
        hidden = rms_norm(hidden, engine.final_norm_weight)
        h = hidden[0, -1, :]  # Last position hidden state

        # Full logits (ground truth)
        t0 = time.perf_counter()
        full_logits = h @ W_decoded.T
        full_dt = (time.perf_counter() - t0) * 1000
        true_top = int(np.argmax(full_logits))
        true_tok = tokenizer.decode_token(true_top)

        # Sign-based score: sign(h) · sign(W[i]) = number of matching signs
        h_sign = np.sign(h).astype(np.int8)
        h_sign[h_sign == 0] = 1

        t0 = time.perf_counter()
        # This is a matmul of int8 × int8, much cheaper
        sign_scores = (h_sign[np.newaxis, :] * W_signs).sum(axis=1)
        sign_dt = (time.perf_counter() - t0) * 1000

        # Top-k by sign score
        sign_top_k = 100
        sign_candidates = np.argsort(sign_scores)[-sign_top_k:][::-1]

        # Now compute full logits only for candidates
        t0 = time.perf_counter()
        candidate_logits = h @ W_decoded[sign_candidates].T
        cand_dt = (time.perf_counter() - t0) * 1000

        best_candidate = sign_candidates[np.argmax(candidate_logits)]
        cand_tok = tokenizer.decode_token(best_candidate)

        # Where does the true answer rank in sign scores?
        true_sign_rank = int(np.sum(sign_scores > sign_scores[true_top]))

        match = "✓" if best_candidate == true_top else "✗"
        print(f"\n  {match} '{prompt}' → '{true_tok.strip()}'")
        print(f"    Full logits: {full_dt:.1f}ms")
        print(f"    Sign filter ({sign_top_k} candidates): {sign_dt:.1f}ms + {cand_dt:.1f}ms = {sign_dt+cand_dt:.1f}ms")
        print(f"    Speedup: {full_dt/(sign_dt+cand_dt):.1f}×")
        print(f"    True token sign-rank: {true_sign_rank} / {W.shape[0]}")
        if best_candidate != true_top:
            print(f"    Got '{cand_tok.strip()}' instead — true token at sign-rank {true_sign_rank}")


def analyze_exponent_navigation(W, engine, tokenizer):
    """Can we use exponent structure for hierarchical search?"""
    print("\n" + "=" * 70)
    print("  Part 3: φ-Level Navigation")
    print("=" * 70)

    # The exponent encodes magnitude on the φ-lattice.
    # High exponents = large values. The dot product h·w is dominated
    # by dimensions where BOTH h and w have high exponents AND matching signs.
    #
    # Hypothesis: we can identify the "important dimensions" of h
    # (where |h| is large, i.e., high exponents) and use only those
    # to filter candidates.

    W_decoded = W.decode_cached()
    W_signs = W.signs
    W_exps = W.exponents

    prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("The color of the sky is", "blue"),
    ]

    for n_dims in [64, 128, 256, 512]:
        correct = 0
        total = len(prompts)

        for prompt, expected in prompts:
            tokens = tokenizer.encode(prompt)
            hidden = engine.embedding(tokens)
            hidden = hidden[np.newaxis, :, :]
            for layer in engine.layers:
                hidden = layer(hidden, pure=False)
            from phi_geometric.inference.phi_components import rms_norm
            hidden = rms_norm(hidden, engine.final_norm_weight)
            h = hidden[0, -1, :]

            # Find top-k dimensions by |h|
            top_dims = np.argsort(np.abs(h))[-n_dims:]

            # Partial dot product using only these dimensions
            partial_logits = h[top_dims] @ W_decoded[:, top_dims].T
            partial_top = int(np.argmax(partial_logits))

            # Ground truth
            full_logits = h @ W_decoded.T
            true_top = int(np.argmax(full_logits))

            if partial_top == true_top:
                correct += 1

        print(f"  Top-{n_dims} dims: {correct}/{total} match full argmax")


def analyze_phi_clustering(W, tokenizer):
    """Analyze if vocab tokens cluster in φ-space."""
    print("\n" + "=" * 70)
    print("  Part 4: Vocabulary Clustering in φ-Space")
    print("=" * 70)

    W_signs = W.signs  # (152064, 3584)
    W_exps = W.exponents

    # Sign-based hash: first 64 sign dimensions → 64-bit signature
    # Tokens with similar sign patterns are in the same cluster
    sign_hash_dims = 64
    sign_signatures = W_signs[:, :sign_hash_dims]  # (152064, 64)

    # Convert to binary: +1 → 1, -1 → 0
    binary_sigs = ((sign_signatures + 1) // 2).astype(np.uint8)

    # Pack first 32 dims into uint32 for fast comparison
    packed = np.zeros(W.shape[0], dtype=np.uint64)
    for d in range(min(32, sign_hash_dims)):
        packed |= (binary_sigs[:, d].astype(np.uint64) << d)

    # How many unique hash values?
    unique_hashes = len(np.unique(packed))
    print(f"  Sign hash (32 dims): {unique_hashes} unique / {W.shape[0]} tokens")
    print(f"  Average bucket size: {W.shape[0] / unique_hashes:.1f}")

    # Check if semantically related tokens share buckets
    test_tokens = ["Paris", "France", "London", "Berlin", "Tokyo",
                   "Jupiter", "Saturn", "Mars", "blue", "red", "green",
                   "two", "three", "four", "Au", "Ag", "Cu"]

    print(f"\n  Token hash similarity (Hamming distance of sign patterns):")
    token_ids = {}
    for tok_str in test_tokens:
        tid = tokenizer.encode(tok_str)
        if len(tid) == 1:
            token_ids[tok_str] = tid[0]

    pairs_to_check = [
        ("Paris", "France"), ("Paris", "London"),
        ("Jupiter", "Saturn"), ("Jupiter", "blue"),
        ("two", "three"), ("two", "Au"),
        ("Au", "Ag"), ("Au", "blue"),
    ]

    for t1, t2 in pairs_to_check:
        if t1 in token_ids and t2 in token_ids:
            s1 = W_signs[token_ids[t1]]
            s2 = W_signs[token_ids[t2]]
            hamming = np.sum(s1 != s2)
            match_pct = (1 - hamming / W.shape[1]) * 100
            print(f"    {t1:10s} vs {t2:10s}: {match_pct:.1f}% sign match "
                  f"(hamming={hamming}/{W.shape[1]})")

    # Exponent-based clustering
    print(f"\n  Exponent norm distribution:")
    exp_norms = np.sqrt((W_exps.astype(np.float32) ** 2).mean(axis=1))
    print(f"    Min: {exp_norms.min():.1f}")
    print(f"    Max: {exp_norms.max():.1f}")
    print(f"    Mean: {exp_norms.mean():.1f}")
    print(f"    Std: {exp_norms.std():.1f}")

    # Do common tokens have different exponent norms than rare ones?
    # Token IDs roughly correlate with frequency in many tokenizers
    n_bins = 10
    bin_size = W.shape[0] // n_bins
    print(f"\n  Exponent norm by token ID range (rough frequency proxy):")
    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, W.shape[0])
        mean_norm = exp_norms[start:end].mean()
        print(f"    Tokens {start:6d}-{end:6d}: exp_norm={mean_norm:.1f}")


def timing_comparison(engine, tokenizer):
    """Time the various approaches."""
    print("\n" + "=" * 70)
    print("  Part 5: Timing Comparison")
    print("=" * 70)

    W = engine.lm_head.weight
    W_decoded = W.decode_cached()
    W_signs = W.signs

    # Get a hidden state
    tokens = tokenizer.encode("The capital of France is")
    hidden = engine.embedding(tokens)
    hidden = hidden[np.newaxis, :, :]
    for layer in engine.layers:
        hidden = layer(hidden, pure=False)
    from phi_geometric.inference.phi_components import rms_norm
    hidden = rms_norm(hidden, engine.final_norm_weight)
    h = hidden[0, -1, :]  # (3584,)

    # 1. Full matmul (cached float decode)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        logits = h @ W_decoded.T
        times.append(time.perf_counter() - t0)
    full_ms = np.median(times) * 1000
    true_top = int(np.argmax(logits))
    print(f"  Full matmul (float, cached): {full_ms:.1f} ms → token {true_top}")

    # 2. Full matmul (φ-decode + matmul, uncached)
    W.clear_cache()
    t0 = time.perf_counter()
    logits2 = phi_linear(W, h.reshape(1, 1, -1))
    uncached_ms = (time.perf_counter() - t0) * 1000
    print(f"  Full matmul (decode+matmul): {uncached_ms:.1f} ms")
    W.decode_cached()  # re-cache

    # 3. Sign filter → top-k matmul
    h_sign = np.sign(h).astype(np.int8)
    h_sign[h_sign == 0] = 1

    for k in [10, 50, 100, 500, 1000]:
        times = []
        correct = 0
        for _ in range(5):
            t0 = time.perf_counter()
            sign_scores = (h_sign[np.newaxis, :] * W_signs).sum(axis=1)
            candidates = np.argsort(sign_scores)[-k:][::-1]
            cand_logits = h @ W_decoded[candidates].T
            best = candidates[np.argmax(cand_logits)]
            times.append(time.perf_counter() - t0)
            if best == true_top:
                correct += 1
        ms = np.median(times) * 1000
        pct = correct / 5 * 100
        print(f"  Sign filter top-{k:4d}: {ms:.1f} ms  "
              f"accuracy={pct:.0f}%  speedup={full_ms/ms:.1f}×")

    # 4. Partial dimensions
    for n_dims in [128, 256, 512, 1024]:
        top_dims = np.argsort(np.abs(h))[-n_dims:]
        times = []
        correct = 0
        for _ in range(5):
            t0 = time.perf_counter()
            partial = h[top_dims] @ W_decoded[:, top_dims].T
            best = int(np.argmax(partial))
            times.append(time.perf_counter() - t0)
            if best == true_top:
                correct += 1
        ms = np.median(times) * 1000
        pct = correct / 5 * 100
        print(f"  Top-{n_dims:4d} dims:      {ms:.1f} ms  "
              f"accuracy={pct:.0f}%  speedup={full_ms/ms:.1f}×")

    # 5. Combined: partial dims as filter → full on candidates
    for n_filter_dims, k in [(256, 500), (512, 200), (128, 1000)]:
        top_dims = np.argsort(np.abs(h))[-n_filter_dims:]
        times = []
        correct = 0
        for _ in range(5):
            t0 = time.perf_counter()
            partial = h[top_dims] @ W_decoded[:, top_dims].T
            candidates = np.argsort(partial)[-k:][::-1]
            full_cand = h @ W_decoded[candidates].T
            best = candidates[np.argmax(full_cand)]
            times.append(time.perf_counter() - t0)
            if best == true_top:
                correct += 1
        ms = np.median(times) * 1000
        pct = correct / 5 * 100
        print(f"  Partial-{n_filter_dims}→top-{k:4d}: {ms:.1f} ms  "
              f"accuracy={pct:.0f}%  speedup={full_ms/ms:.1f}×")


def main():
    print("=" * 70)
    print("  LM Head Exploration: Offload + Geometric Navigation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"  {len(engine.layers)} layers, vocab={engine.vocab_size}")

    # Warm LM head cache for fair timing
    engine.lm_head.weight.decode_cached()

    W = analyze_lm_head_structure(engine)
    analyze_sign_patterns(W, engine, tokenizer)
    analyze_exponent_navigation(W, engine, tokenizer)
    analyze_phi_clustering(W, tokenizer)
    timing_comparison(engine, tokenizer)

    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)
    print("""
  The LM head is a 152064×3584 matmul. For next-token prediction,
  we only need argmax — not all 152K logits.

  IMMEDIATE WIN:
    Offload to gimli GPU via thin client MATMUL opcode.
    Expected: 7.4s → ~1s (matching layer matmul performance).

  GEOMETRIC NAVIGATION:
    The sign pattern of the hidden state provides a cheap filter.
    The top-k exponent dimensions provide dimensional reduction.
    Combined approaches can reduce the search space significantly.

  CONNECTION TO ATTENTION SPIGOT:
    Attention: navigate to relevant positions via φ-lattice
    LM head: navigate to relevant tokens via φ-sign structure
    Both avoid brute-force computation by using geometric structure.
""")


if __name__ == '__main__':
    main()
