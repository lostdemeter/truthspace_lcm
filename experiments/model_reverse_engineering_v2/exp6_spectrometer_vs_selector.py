#!/usr/bin/env python3
"""
Experiment 6: Spectrometer vs Selector — Head-to-Head Comparison
================================================================

Two geometric measurement instruments that decompose input into
components along specific axes:

  SPECTROMETER (ours):
    - Decomposes via SVD/φ-level analysis of weight matrices
    - 3584 dimensions, 166 φ-levels
    - α ≈ 1/φ  →  sees everything evenly
    - We designed it

  LAYER 1 SELECTOR BANK (model's):
    - Decomposes via rank-1 MESH projections
    - 28 learned directions
    - α ≈ 2/φ  →  selects specific things
    - The model designed it during training

Head-to-head tests:
  1. Subspace Agreement: Do they find the same axes?
  2. Variance Capture: How much of embedding space does each cover?
  3. Token Discrimination: Which clusters tokens better?
  4. Interchangeability: Can we swap one for the other?
  5. Complementarity: What does each see that the other misses?
  6. φ-Structure: Which decomposition reveals more φ-geometry?
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS


def efficient_mesh_svd(W_q_head, W_k_head):
    """Compute MESH SVD via QR + small SVD."""
    A = W_q_head.T.astype(np.float64)
    B = W_k_head.astype(np.float64)
    Q, R = np.linalg.qr(A)
    C = R @ B
    U_c, S, Vt = np.linalg.svd(C, full_matrices=False)
    U = Q @ U_c
    return U, S, Vt


def randomized_svd(M, n_components, n_oversamples=10, n_iter=4):
    """Randomized SVD for large matrices. Returns U, S, Vt."""
    m, n = M.shape
    k = n_components + n_oversamples
    rng = np.random.RandomState(42)

    # Random projection
    Omega = rng.randn(n, k).astype(np.float64)
    Y = M @ Omega

    # Power iteration for better accuracy
    for _ in range(n_iter):
        Y = M @ (M.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ M  # (k, n)
    U_b, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b

    # Truncate to n_components
    return U[:, :n_components], S[:n_components], Vt[:n_components]


def main():
    print()
    print("=" * 80)
    print("  Experiment 6: Spectrometer vs Selector — Head-to-Head")
    print("  Which geometric measurement instrument is stronger?")
    print("=" * 80)
    print()

    # ================================================================
    # Load data
    # ================================================================
    print("  Loading embeddings...")
    emb_phi = PhiEncoded.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    embeddings = emb_phi.decode()  # (152064, 3584)
    n_tokens, n_dims = embeddings.shape
    print(f"  Embeddings: {n_tokens} tokens × {n_dims} dims")

    print("  Loading Layer 1 weights...")
    layer_dir = os.path.join(MODEL_DIR, 'layer_01')
    q_phi = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    k_phi = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
    W_q = q_phi.decode().reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_k = k_phi.decode().reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

    # Extract Layer 1 selector directions (dominant SVD vectors)
    print("  Extracting 28 selector directions...")
    selector_V = []  # Key-side selectors (what tokens the head attends to)
    selector_U = []  # Query-side selectors
    selector_S = []
    for head_idx in range(NUM_HEADS):
        kv_idx = head_idx // HEADS_PER_KV
        U, S, Vt = efficient_mesh_svd(W_q[head_idx], W_k[kv_idx])
        selector_U.append(U[:, 0])
        selector_V.append(Vt[0])
        selector_S.append(S[0])
    selector_U = np.array(selector_U)  # (28, 3584)
    selector_V = np.array(selector_V)  # (28, 3584)
    selector_S = np.array(selector_S)  # (28,)

    # Compute spectrometer axes: top-28 SVD directions of embedding matrix
    print("  Computing top-28 SVD of embedding matrix (randomized)...")
    # Center embeddings for proper SVD
    emb_mean = embeddings.mean(axis=0, keepdims=True)
    emb_centered = (embeddings - emb_mean).astype(np.float64)
    U_emb, S_emb, Vt_emb = randomized_svd(emb_centered, n_components=28)
    # Vt_emb: (28, 3584) — the spectrometer's principal axes
    print(f"  Top-28 SVD computed. Variance captured: "
          f"{(S_emb**2).sum() / np.sum(emb_centered**2) * 100:.1f}%")
    print()

    # Also compute top-128 for deeper analysis
    print("  Computing top-128 SVD of embedding matrix...")
    _, S_emb_128, Vt_emb_128 = randomized_svd(emb_centered, n_components=128)
    total_var = np.sum(emb_centered ** 2)
    print(f"  Top-128 variance: {(S_emb_128**2).sum() / total_var * 100:.1f}%")
    print()

    # ================================================================
    # Test 1: Subspace Agreement
    # Do they find the same axes?
    # ================================================================
    print("─" * 80)
    print("  Test 1: SUBSPACE AGREEMENT")
    print("  Do the spectrometer and selector find the same directions?")
    print("─" * 80)
    print()

    # Normalize both sets
    spec_axes = Vt_emb / np.linalg.norm(Vt_emb, axis=1, keepdims=True)
    sel_V_hat = selector_V / np.linalg.norm(selector_V, axis=1, keepdims=True)
    sel_U_hat = selector_U / np.linalg.norm(selector_U, axis=1, keepdims=True)

    # Cross-similarity: how much of each selector is captured by the
    # spectrometer subspace (and vice versa)?
    # Project selector axes onto spectrometer subspace
    cross_sim = np.abs(sel_V_hat @ spec_axes.T)  # (28, 28)

    # For each selector, max alignment with any spectrometer axis
    max_alignment_per_selector = cross_sim.max(axis=1)
    # For each spectrometer axis, max alignment with any selector
    max_alignment_per_spec = cross_sim.max(axis=0)

    # Subspace overlap: project selectors onto spectrometer span
    # fraction of selector variance captured by spectrometer's 28-dim span
    selector_in_spec = sel_V_hat @ spec_axes.T  # (28, 28)
    frac_captured = np.sum(selector_in_spec ** 2, axis=1)  # fraction per selector

    print(f"  Per-selector alignment with spectrometer axes:")
    print(f"    Max |cos| per selector:  mean={max_alignment_per_selector.mean():.4f}  "
          f"min={max_alignment_per_selector.min():.4f}  "
          f"max={max_alignment_per_selector.max():.4f}")
    print(f"    Max |cos| per spec axis: mean={max_alignment_per_spec.mean():.4f}  "
          f"min={max_alignment_per_spec.min():.4f}  "
          f"max={max_alignment_per_spec.max():.4f}")
    print()

    print(f"  Fraction of each selector captured by spectrometer's 28-dim subspace:")
    print(f"    Mean: {frac_captured.mean():.4f}  ({frac_captured.mean()*100:.1f}%)")
    print(f"    Min:  {frac_captured.min():.4f}  ({frac_captured.min()*100:.1f}%)")
    print(f"    Max:  {frac_captured.max():.4f}  ({frac_captured.max()*100:.1f}%)")
    print()

    # Random baseline: 28 random directions in 3584-space
    rng = np.random.RandomState(42)
    random_dirs = rng.randn(28, HIDDEN_DIM)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_in_spec = random_dirs @ spec_axes.T
    random_frac = np.sum(random_in_spec ** 2, axis=1)
    expected_random = 28 / HIDDEN_DIM

    print(f"  Baselines:")
    print(f"    Random 28-in-3584:  {random_frac.mean():.4f} "
          f"(expected {expected_random:.4f})")
    print(f"    Selector in spec:   {frac_captured.mean():.4f}")
    print(f"    Ratio over random:  {frac_captured.mean()/expected_random:.1f}×")
    print()

    if frac_captured.mean() > 0.5:
        print("  → STRONG agreement: selectors live mostly in spectrometer subspace")
    elif frac_captured.mean() > 0.1:
        print("  → MODERATE agreement: partial overlap")
    else:
        print("  → WEAK agreement: selectors found different directions")
    print()

    # ================================================================
    # Test 2: Variance Capture
    # How much of the embedding space does each cover with 28 axes?
    # ================================================================
    print("─" * 80)
    print("  Test 2: VARIANCE CAPTURE")
    print("  How much of embedding space does each instrument cover?")
    print("─" * 80)
    print()

    # Spectrometer: variance captured by top-28 SVD axes
    spec_var = (S_emb ** 2).sum() / total_var

    # Selector: project embeddings onto 28 selector directions
    sel_projections_V = emb_centered @ sel_V_hat.T  # (152064, 28)
    sel_var_V = np.sum(sel_projections_V ** 2) / total_var

    sel_projections_U = emb_centered @ sel_U_hat.T  # (152064, 28)
    sel_var_U = np.sum(sel_projections_U ** 2) / total_var

    # Random baseline
    random_proj = emb_centered @ random_dirs.T
    random_var = np.sum(random_proj ** 2) / total_var

    print(f"  Variance captured by 28 axes:")
    print(f"    Spectrometer (SVD top-28): {spec_var*100:6.2f}%  ← UPPER BOUND")
    print(f"    Selector V (key-side):     {sel_var_V*100:6.2f}%")
    print(f"    Selector U (query-side):   {sel_var_U*100:6.2f}%")
    print(f"    Random 28 directions:      {random_var*100:6.2f}%  ← LOWER BOUND")
    print()

    efficiency_V = (sel_var_V - random_var) / (spec_var - random_var)
    efficiency_U = (sel_var_U - random_var) / (spec_var - random_var)
    print(f"  Efficiency (0%=random, 100%=optimal SVD):")
    print(f"    Selector V: {efficiency_V*100:.1f}%")
    print(f"    Selector U: {efficiency_U*100:.1f}%")
    print()

    # How many spectrometer axes does the selector match?
    cumvar = np.cumsum(S_emb_128 ** 2) / total_var
    sel_equivalent_rank_V = np.searchsorted(cumvar, sel_var_V) + 1
    sel_equivalent_rank_U = np.searchsorted(cumvar, sel_var_U) + 1
    print(f"  Selector V captures as much variance as top-{sel_equivalent_rank_V} SVD")
    print(f"  Selector U captures as much variance as top-{sel_equivalent_rank_U} SVD")
    print()

    # ================================================================
    # Test 3: Token Discrimination
    # Which clusters semantically similar tokens better?
    # ================================================================
    print("─" * 80)
    print("  Test 3: TOKEN DISCRIMINATION")
    print("  Which instrument better separates different token types?")
    print("─" * 80)
    print()

    # Load tokenizer for semantic labels
    try:
        import json
        for candidate in [
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
        ]:
            if os.path.exists(candidate):
                snapshots = os.listdir(candidate)
                if snapshots:
                    vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                    if os.path.exists(vocab_file):
                        with open(vocab_file, 'r') as f:
                            tokenizer_data = json.load(f)
                        vocab = tokenizer_data.get('model', {}).get('vocab', {})
                        token_names = [''] * n_tokens
                        for tok, idx in vocab.items():
                            if idx < len(token_names):
                                token_names[idx] = tok
                        break
        print(f"  Loaded tokenizer")
    except Exception:
        token_names = None
        print("  No tokenizer — using heuristic categories")

    # Define token categories for discrimination test
    categories = {}
    if token_names:
        for idx, name in enumerate(token_names):
            if not name:
                continue
            # Punctuation
            if name in ['.', ',', '!', '?', ':', ';', ')', '(', '"', "'",
                        '。', '、', '！', '？', '）', '（']:
                categories.setdefault('punctuation', []).append(idx)
            # Numbers
            elif name.strip('Ġ').isdigit():
                categories.setdefault('numbers', []).append(idx)
            # Whitespace/newlines
            elif name in ['Ċ', 'ĊĊ', 'ĉ', 'Ġ', '\n', '\t']:
                categories.setdefault('whitespace', []).append(idx)
            # Uppercase start (likely names/proper nouns)
            elif len(name) > 1 and name.startswith('Ġ') and name[1:2].isupper():
                categories.setdefault('capitalized', []).append(idx)
            # Programming
            elif any(c in name for c in ['def ', 'class ', 'import ', 'return ',
                                          'function', 'const ', 'var ']):
                categories.setdefault('code_keywords', []).append(idx)
            # Chinese characters
            elif any('\u4e00' <= c <= '\u9fff' for c in name):
                categories.setdefault('chinese', []).append(idx)

    print(f"  Categories: {', '.join(f'{k}({len(v)})' for k, v in categories.items())}")
    print()

    # For each category pair, measure separation in each feature space
    # Separation = (between-class distance) / (within-class distance)
    cat_names = sorted(categories.keys())

    # Build feature matrices
    # Spectrometer features: top-28 SVD projections
    spec_features = emb_centered @ spec_axes.T  # (n_tokens, 28)
    # Selector features: projections onto 28 selector V directions
    sel_features = emb_centered @ sel_V_hat.T    # (n_tokens, 28)

    print(f"  {'Category Pair':>35s}  {'Spec Sep':>9s}  {'Sel Sep':>9s}  {'Winner':>8s}")
    print("  " + "-" * 70)

    spec_wins = 0
    sel_wins = 0

    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            c1, c2 = cat_names[i], cat_names[j]
            idx1 = np.array(categories[c1][:500])  # Cap for speed
            idx2 = np.array(categories[c2][:500])

            if len(idx1) < 10 or len(idx2) < 10:
                continue

            # Spectrometer separation
            mean1_s = spec_features[idx1].mean(axis=0)
            mean2_s = spec_features[idx2].mean(axis=0)
            between_s = np.linalg.norm(mean1_s - mean2_s)
            within1_s = np.mean(np.linalg.norm(spec_features[idx1] - mean1_s, axis=1))
            within2_s = np.mean(np.linalg.norm(spec_features[idx2] - mean2_s, axis=1))
            within_s = (within1_s + within2_s) / 2
            sep_s = between_s / (within_s + 1e-10)

            # Selector separation
            mean1_l = sel_features[idx1].mean(axis=0)
            mean2_l = sel_features[idx2].mean(axis=0)
            between_l = np.linalg.norm(mean1_l - mean2_l)
            within1_l = np.mean(np.linalg.norm(sel_features[idx1] - mean1_l, axis=1))
            within2_l = np.mean(np.linalg.norm(sel_features[idx2] - mean2_l, axis=1))
            within_l = (within1_l + within2_l) / 2
            sep_l = between_l / (within_l + 1e-10)

            winner = "SPEC" if sep_s > sep_l else "SELECT"
            if sep_s > sep_l:
                spec_wins += 1
            else:
                sel_wins += 1

            pair_name = f"{c1} vs {c2}"
            print(f"  {pair_name:>35s}  {sep_s:9.4f}  {sep_l:9.4f}  {winner:>8s}")

    print()
    print(f"  Score: Spectrometer {spec_wins} — Selector {sel_wins}")
    print()

    # ================================================================
    # Test 4: Interchangeability
    # Replace selector axes with spectrometer axes — what happens?
    # ================================================================
    print("─" * 80)
    print("  Test 4: INTERCHANGEABILITY")
    print("  Can we swap spectrometer axes for selector axes?")
    print("─" * 80)
    print()

    # For a sample of tokens, compute attention scores using:
    # (a) Original Layer 1 selectors: score = emb @ MESH @ emb.T
    # (b) Replaced with spectrometer top-28: score = emb @ MESH_spec @ emb.T
    # where MESH_spec uses spectrometer axes instead of selector axes

    # Sample tokens
    np.random.seed(42)
    sample_idx = np.random.choice(n_tokens, 500, replace=False)
    sample_emb = emb_centered[sample_idx]  # (500, 3584)

    # Original Layer 1 attention: use rank-1 approximation
    # MESH_h ≈ σ_h × u_h ⊗ v_h
    # score_h = (q · u_h)(k · v_h)σ_h
    # Total multi-head score ≈ Σ_h σ_h × (q · u_h)(k · v_h)
    original_scores = np.zeros((500, 500))
    for h in range(NUM_HEADS):
        q_proj = sample_emb @ selector_U[h]  # (500,)
        k_proj = sample_emb @ selector_V[h]  # (500,)
        original_scores += selector_S[h] * np.outer(q_proj, k_proj)

    # Spectrometer replacement: use top-28 SVD axes as both U and V
    # This tests: "if we decompose along spectrometer axes instead of
    # selector axes, do we get similar attention patterns?"
    spec_scores = np.zeros((500, 500))
    for h in range(28):
        q_proj = sample_emb @ spec_axes[h]  # (500,)
        k_proj = sample_emb @ spec_axes[h]  # (500,) — symmetric
        spec_scores += S_emb[h] * np.outer(q_proj, k_proj)

    # Correlation between attention patterns
    orig_flat = original_scores.flatten()
    spec_flat = spec_scores.flatten()
    corr = np.corrcoef(orig_flat, spec_flat)[0, 1]

    # Also try: use spectrometer axes for V, keep selector U
    hybrid_scores = np.zeros((500, 500))
    for h in range(NUM_HEADS):
        q_proj = sample_emb @ selector_U[h]  # keep original query
        k_proj = sample_emb @ spec_axes[h % 28]  # replace key with spec axis
        hybrid_scores += selector_S[h] * np.outer(q_proj, k_proj)

    hybrid_flat = hybrid_scores.flatten()
    hybrid_corr = np.corrcoef(orig_flat, hybrid_flat)[0, 1]

    print(f"  Attention pattern correlation (500 tokens):")
    print(f"    Original selector → Spectrometer swap:  r = {corr:.4f}")
    print(f"    Hybrid (original Q, spec K):             r = {hybrid_corr:.4f}")
    print()

    if corr > 0.8:
        print("  → HIGH interchangeability: spectrometer axes produce similar attention")
    elif corr > 0.4:
        print("  → MODERATE interchangeability: partially overlapping but distinct")
    else:
        print("  → LOW interchangeability: different instruments, different views")
    print()

    # ================================================================
    # Test 5: Complementarity
    # What does each see that the other misses?
    # ================================================================
    print("─" * 80)
    print("  Test 5: COMPLEMENTARITY")
    print("  What does each see that the other misses?")
    print("─" * 80)
    print()

    # Combine both feature sets and measure information content
    combined_features = np.hstack([spec_features[sample_idx],
                                    sel_features[sample_idx]])  # (500, 56)

    # SVD of combined feature matrix
    _, S_combined, _ = np.linalg.svd(combined_features, full_matrices=False)
    _, S_spec_only, _ = np.linalg.svd(spec_features[sample_idx], full_matrices=False)
    _, S_sel_only, _ = np.linalg.svd(sel_features[sample_idx], full_matrices=False)

    # Effective dimensionality
    def effective_dim(S):
        p = S ** 2 / (S ** 2).sum()
        return np.exp(-np.sum(p * np.log(p + 1e-20)))

    eff_combined = effective_dim(S_combined)
    eff_spec = effective_dim(S_spec_only)
    eff_sel = effective_dim(S_sel_only)

    print(f"  Effective dimensionality:")
    print(f"    Spectrometer only:  {eff_spec:.1f} / 28")
    print(f"    Selector only:      {eff_sel:.1f} / 28")
    print(f"    Combined:           {eff_combined:.1f} / 56")
    print(f"    If independent:     {eff_spec + eff_sel:.1f}")
    print(f"    Overlap:            {eff_spec + eff_sel - eff_combined:.1f} shared dims")
    print()

    redundancy = 1.0 - (eff_combined / (eff_spec + eff_sel))
    unique_spec = eff_combined - eff_sel
    unique_sel = eff_combined - eff_spec

    print(f"  Information decomposition:")
    print(f"    Unique to spectrometer: {unique_spec:.1f} dims")
    print(f"    Unique to selector:     {unique_sel:.1f} dims")
    print(f"    Shared:                 {eff_spec + eff_sel - eff_combined:.1f} dims")
    print(f"    Redundancy:             {redundancy*100:.1f}%")
    print()

    if redundancy < 0.3:
        print("  → COMPLEMENTARY: each sees mostly different structure")
        print("  → Combining both would be significantly more powerful")
    elif redundancy < 0.6:
        print("  → PARTIALLY OVERLAPPING: some shared, some unique")
    else:
        print("  → LARGELY REDUNDANT: both see similar structure")
    print()

    # ================================================================
    # Test 6: φ-Structure
    # Which decomposition reveals more φ-geometry?
    # ================================================================
    print("─" * 80)
    print("  Test 6: φ-STRUCTURE")
    print("  Which decomposition reveals more φ-geometry?")
    print("─" * 80)
    print()

    # φ-encode the projections and check for level structure
    for name, features in [("Spectrometer", spec_features),
                            ("Selector", sel_features)]:
        # Take a sample
        sample = features[sample_idx]  # (500, 28)

        # φ-encode
        signs = np.sign(sample)
        signs[signs == 0] = 1
        mags = np.abs(sample) + 1e-20
        levels = np.round(PHI_GRID * np.log(mags) / LOG_PHI).astype(int)

        # How many unique levels?
        unique_levels = len(np.unique(levels))

        # Shannon entropy of level distribution
        level_counts = Counter(levels.flatten().tolist())
        total = levels.size
        probs = np.array(list(level_counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-20))

        # Zipf analysis of projections' singular values
        _, S_proj, _ = np.linalg.svd(sample, full_matrices=False)
        ranks = np.arange(1, len(S_proj) + 1)
        alpha = -np.polyfit(np.log(ranks), np.log(S_proj + 1e-20), 1)[0]

        phi_mark = " ← 1/φ!" if abs(alpha - 0.618) < 0.1 else ""

        # Sign balance
        pos_frac = (signs > 0).mean()

        print(f"  {name}:")
        print(f"    Unique φ-levels:  {unique_levels}")
        print(f"    Level entropy:    {entropy:.2f} bits")
        print(f"    Sign balance:     {pos_frac*100:.1f}% positive")
        print(f"    Projection SVD α: {alpha:.4f}{phi_mark}")
        print(f"    Projection SVD S: [{', '.join(f'{s:.4f}' for s in S_proj[:8])}  ...]")
        print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 80)
    print("  SUMMARY: Spectrometer vs Selector")
    print("=" * 80)
    print()

    print("  ┌──────────────────────┬──────────────┬──────────────┐")
    print("  │ Test                 │ Spectrometer │ Selector     │")
    print("  ├──────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Variance (28 axes)   │ {spec_var*100:10.1f}%% │ {sel_var_V*100:10.1f}%% │")
    print(f"  │ Effective dims       │ {eff_spec:10.1f}  │ {eff_sel:10.1f}  │")
    svr = f"SVD top-{sel_equivalent_rank_V}"
    print(f"  │ Equivalent SVD rank  │ top-28 (def) │ {svr:>12s} │")
    print(f"  │ Token discrimination │ {spec_wins:>10d}  │ {sel_wins:>10d}  │")
    print(f"  │ Interchangeability   │          {corr:.4f} correlation │")
    print(f"  │ Unique information   │ {unique_spec:10.1f}  │ {unique_sel:10.1f}  │")
    print(f"  │ Subspace overlap     │          {frac_captured.mean()*100:.1f}%% captured │")
    print(f"  │ Redundancy           │         {redundancy*100:.1f}%% overlap │")
    print("  └──────────────────────┴──────────────┴──────────────┘")
    print()

    # Determine winner per domain
    print("  VERDICTS:")
    print()

    print(f"  Coverage: {'SPECTROMETER' if spec_var > sel_var_V else 'SELECTOR'} "
          f"(SVD is optimal by definition, selector captures "
          f"{sel_var_V/spec_var*100:.0f}% of optimal)")
    print()

    if spec_wins > sel_wins:
        print(f"  Discrimination: SPECTROMETER ({spec_wins}-{sel_wins})")
        print(f"    → Better at separating token categories")
    else:
        print(f"  Discrimination: SELECTOR ({sel_wins}-{spec_wins})")
        print(f"    → Better at separating token categories")
    print()

    if corr > 0.5:
        print(f"  Interchangeability: YES (r={corr:.3f})")
        print(f"    → Could replace selector with spectrometer axes")
    else:
        print(f"  Interchangeability: LIMITED (r={corr:.3f})")
        print(f"    → Each captures different aspects")
    print()

    if redundancy < 0.4:
        print(f"  Complementarity: STRONG ({redundancy*100:.0f}% redundancy)")
        print(f"    → Combining both adds {eff_combined - max(eff_spec, eff_sel):.0f} "
              f"effective dimensions")
    else:
        print(f"  Complementarity: WEAK ({redundancy*100:.0f}% redundancy)")
        print(f"    → Largely see the same structure")
    print()

    print("  WHEN TO PREFER EACH:")
    print(f"    Spectrometer: Maximum coverage, variance capture, broad analysis")
    print(f"    Selector:     Task-specific discrimination, attention routing")
    print(f"    Combined:     Richest representation ({eff_combined:.0f} effective dims)")
    print()


if __name__ == '__main__':
    main()
