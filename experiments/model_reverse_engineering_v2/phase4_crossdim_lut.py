"""
Phase 4: Cross-Dimensional LUT Investigation for Layer 23

The spectrometer error at layer 23 is "irreducible" with per-dimension rules
because 1500+ dims contribute to the error, and the correction is prompt-dependent
(only 7.5% rank-1). This script investigates:

Part A: WHERE does the cross-dim error come from?
  - Decompose into: RMS norm contribution vs attention vs MLP
  - Measure how much of the error is due to each coupling point

Part B: WHAT cross-dim features predict the error?
  - RMS norm denominator (global scale)
  - Sign pattern hash (which dims are +/-)
  - Principal component projections of the input
  - φ-level distribution statistics

Part C: Can we build a LUT?
  - Discretize cross-dim features into bins
  - Map each bin to a correction vector
  - Test if the LUT fixes the "Paris" failure

The key insight from Design 151 (LUT-only compression): model weights live on
a 92-entry φ-lattice. The hypothesis here is that the CROSS-DIMENSIONAL STATE
also lives on a constrained manifold with a finite vocabulary.
"""

import sys
import json
import numpy as np
import time

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_spectrometer import SpectrometerLayer, load_all_rules

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
RULES_DIR = 'experiments/model_reverse_engineering_v2/results/phase4_rules_full'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


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


def extract_cross_dim_features(hidden_vec):
    """Extract cross-dimensional features from a single hidden state vector.
    
    These are features that per-dimension rules CANNOT see because they
    depend on relationships BETWEEN dimensions.
    
    Args:
        hidden_vec: (3584,) hidden state vector
        
    Returns:
        dict of feature name -> value
    """
    x = hidden_vec
    
    # 1. RMS norm denominator — the global scale factor
    rms = np.sqrt(np.mean(x ** 2))
    
    # 2. Sign statistics — what fraction are positive
    sign_frac = np.mean(x > 0)
    
    # 3. φ-level statistics
    mag = np.abs(x) + 1e-20
    levels = 64.0 * np.log(mag) / LOG_PHI
    level_mean = np.mean(levels)
    level_std = np.std(levels)
    level_median = np.median(levels)
    
    # 4. Magnitude distribution moments
    mag_kurtosis = np.mean((x - x.mean())**4) / (np.var(x)**2 + 1e-20)
    mag_skew = np.mean((x - x.mean())**3) / (np.std(x)**3 + 1e-20)
    
    # 5. Block correlations — do neighboring dims correlate?
    # (This is what RMS norm and matmuls create)
    x_shifted = np.roll(x, 1)
    neighbor_corr = np.corrcoef(x, x_shifted)[0, 1]
    
    # 6. Head-aligned features — the 28 heads see 128-dim blocks
    # Reshape to (28, 128) and measure inter-head variance
    if len(x) == 3584:
        heads = x.reshape(28, 128)
        head_norms = np.linalg.norm(heads, axis=1)
        head_norm_cv = np.std(head_norms) / (np.mean(head_norms) + 1e-20)
    else:
        head_norm_cv = 0.0
    
    return {
        'rms': rms,
        'sign_frac': sign_frac,
        'level_mean': level_mean,
        'level_std': level_std,
        'level_median': level_median,
        'kurtosis': mag_kurtosis,
        'skew': mag_skew,
        'neighbor_corr': neighbor_corr,
        'head_norm_cv': head_norm_cv,
    }


def main():
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    target_layer = 23
    rules = all_rules[target_layer]
    spec_layer = SpectrometerLayer(
        rules=rules, full_layer=engine.layers[target_layer],
        r2_threshold=0.7, mode='rules_only',
    )
    
    # =========================================================================
    #   Part A: WHERE does the cross-dim error come from?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part A: Decomposing cross-dimensional error sources")
    print("=" * 80)
    
    # For the "France" prompt, trace through layer 23 step by step
    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt)
    
    hidden = engine.embedding(ids)[np.newaxis, :, :]
    for layer in engine.layers:
        if layer.layer_idx == target_layer:
            break
        hidden = layer(hidden)
    
    layer = engine.layers[target_layer]
    
    # Step 1: Pre-attention RMS norm
    normed_attn = rms_norm(hidden, layer.attention.norm_weight)
    
    # The spectrometer applies rules directly to `hidden` → output
    # But the real layer does: RMSnorm → Q/K/V → attention → residual → RMSnorm → MLP → residual
    # The cross-dim coupling enters at EACH of these steps
    
    # Measure the RMS norm denominator per position
    print(f"\n  Prompt: \"{prompt}\" ({len(ids)} tokens)")
    print(f"\n  RMS norm denominators (attention) per position:")
    for t in range(len(ids)):
        x = hidden[0, t, :]
        rms_val = np.sqrt(np.mean(x ** 2))
        print(f"    pos {t}: RMS = {rms_val:.4f}")
    
    # The RMS norm creates cross-dim coupling because:
    # normed[d] = hidden[d] / sqrt(mean(hidden[:]^2))
    # The denominator depends on ALL dims, so normed[d] depends on ALL dims
    
    # Quantify: how much does the RMS denominator vary across prompts?
    cal_prompts = [
        '1 + 1 =', '2 + 2 =', 'The sky is', 'Water is made of',
        'The sun rises in the', 'Gravity makes things fall',
        'Once upon a time', 'She walked into the room and',
        'He said that he would', 'They decided to go to the',
        'The old man sat on the', 'After the rain stopped',
        'The quick brown fox', 'In machine learning',
        'Python is a programming', 'The function returns',
        'An algorithm that sorts', 'The largest planet is',
        'Albert Einstein developed the', 'Shakespeare wrote many',
        'The speed of light is', 'DNA stands for',
        'The Pacific Ocean is', 'I think that we should',
        'She said that she would', 'It is important to note that',
        'The reason for this is', 'According to the latest',
        'In 2024, the world', 'If you want to learn',
        'The best way to', 'One of the most important',
        'As a result of the', 'Between the two options',
        'Despite the challenges', 'For example, consider',
        'However, it is worth', 'In conclusion, the',
    ]
    
    print(f"\n  Collecting cross-dim features across {len(cal_prompts)} prompts...")
    
    # Collect data
    all_features = []
    all_errors = []
    all_rms_attn = []  # RMS norm denominator for attention
    all_rms_mlp = []   # RMS norm denominator for MLP
    all_hidden = []
    
    for pi, p in enumerate(cal_prompts):
        p_ids = tokenizer.encode(p)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                full_out = layer_obj(h.copy())
                spec_out = spec_layer(h.copy())
                break
            h = layer_obj(h)
        
        # Only look at last token (determines next-token prediction)
        t = len(p_ids) - 1
        h_vec = h[0, t, :]
        err_vec = (full_out - spec_out)[0, t, :]
        
        features = extract_cross_dim_features(h_vec)
        all_features.append(features)
        all_errors.append(err_vec)
        all_hidden.append(h_vec)
        
        # RMS norm values at this token position
        rms_attn = np.sqrt(np.mean(h_vec ** 2))
        all_rms_attn.append(rms_attn)
        
        # For MLP norm, we'd need the post-attention state
        # but let's approximate with the pre-layer state for now
        
        if pi % 10 == 0:
            print(f"    {pi}/{len(cal_prompts)}...")
    
    all_errors = np.array(all_errors)  # (N, 3584)
    all_hidden = np.array(all_hidden)  # (N, 3584)
    all_rms_attn = np.array(all_rms_attn)  # (N,)
    
    print(f"\n  Cross-dim feature statistics across {len(all_features)} last-tokens:")
    feature_names = list(all_features[0].keys())
    for fname in feature_names:
        vals = [f[fname] for f in all_features]
        print(f"    {fname:>15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
    
    # =========================================================================
    #   Part B: Which cross-dim features predict the error?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part B: Which cross-dim features predict the error?")
    print("=" * 80)
    
    err_norms = np.linalg.norm(all_errors, axis=1)  # (N,)
    
    print(f"\n  Error norm: mean={err_norms.mean():.2f}  std={err_norms.std():.2f}")
    
    # Correlation of each feature with error norm
    print(f"\n  Feature → error norm correlations:")
    for fname in feature_names:
        vals = np.array([f[fname] for f in all_features])
        if np.std(vals) < 1e-10:
            continue
        corr = np.corrcoef(vals, err_norms)[0, 1]
        print(f"    {fname:>15s}: r = {corr:+.4f}")
    
    # More importantly: can we predict the error DIRECTION?
    # SVD of errors
    U_e, S_e, Vt_e = np.linalg.svd(all_errors, full_matrices=False)
    total_e = (S_e ** 2).sum()
    
    print(f"\n  Error SVD:")
    for k in [1, 3, 5, 10, 20]:
        cap = (S_e[:k] ** 2).sum() / total_e * 100
        print(f"    Rank-{k:2d}: {cap:.1f}% of variance")
    
    # Project errors onto top SVD directions
    E_proj = all_errors @ Vt_e[:20].T  # (N, 20) — coefficients in error basis
    
    # Correlate each feature with each error direction
    print(f"\n  Feature → error SVD direction correlations (top 5 dirs):")
    for fname in feature_names:
        vals = np.array([f[fname] for f in all_features])
        if np.std(vals) < 1e-10:
            continue
        corrs = [np.corrcoef(vals, E_proj[:, k])[0, 1] for k in range(5)]
        corr_str = "  ".join([f"d{k}:{c:+.3f}" for k, c in enumerate(corrs)])
        print(f"    {fname:>15s}: {corr_str}")
    
    # =========================================================================
    #   Part B2: RMS norm denominator as the key cross-dim feature
    # =========================================================================
    print(f"\n  RMS norm analysis:")
    print(f"    RMS range: [{all_rms_attn.min():.4f}, {all_rms_attn.max():.4f}]")
    print(f"    RMS mean:  {all_rms_attn.mean():.4f}")
    print(f"    RMS std:   {all_rms_attn.std():.4f}")
    
    # Discretize RMS into φ-levels
    rms_levels = np.round(64.0 * np.log(all_rms_attn) / LOG_PHI).astype(int)
    unique_rms_levels = np.unique(rms_levels)
    print(f"    RMS φ-levels: {len(unique_rms_levels)} unique ({unique_rms_levels})")
    
    # =========================================================================
    #   Part B3: Principal components of the input as cross-dim features
    # =========================================================================
    print(f"\n  Input PCA analysis:")
    
    # Center the hidden states
    X_centered = all_hidden - all_hidden.mean(axis=0)
    U_x, S_x, Vt_x = np.linalg.svd(X_centered, full_matrices=False)
    
    total_x = (S_x ** 2).sum()
    for k in [1, 3, 5, 10, 20]:
        cap = (S_x[:k] ** 2).sum() / total_x * 100
        print(f"    Input rank-{k:2d}: {cap:.1f}% of variance")
    
    # Project inputs onto top PCs
    X_proj = X_centered @ Vt_x[:20].T  # (N, 20)
    
    # How well do input PCs predict error directions?
    print(f"\n  Input PC → error direction correlations:")
    for pc_k in range(5):
        corrs = [np.corrcoef(X_proj[:, pc_k], E_proj[:, e_k])[0, 1] for e_k in range(5)]
        corr_str = "  ".join([f"e{k}:{c:+.3f}" for k, c in enumerate(corrs)])
        print(f"    PC{pc_k}: {corr_str}")
    
    # Overall: can input PCs predict the full error?
    for n_pcs in [3, 5, 10, 20]:
        X_features = X_proj[:, :n_pcs]
        # Predict each error dimension
        W, _, _, _ = np.linalg.lstsq(X_features, all_errors, rcond=1e-3)
        E_pred = X_features @ W
        r2 = 1.0 - np.sum((all_errors - E_pred) ** 2) / np.sum(all_errors ** 2)
        print(f"    {n_pcs:2d} PCs → error R²: {r2:.4f}")
    
    # =========================================================================
    #   Part C: Build the LUT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C: LUT-based correction")
    print("=" * 80)
    
    # Strategy: discretize input into bins using key features, average the
    # error within each bin to get the correction vector
    
    # Strategy 1: RMS-level bins
    print(f"\n  Strategy 1: RMS φ-level bins")
    for n_bins in [2, 4, 8]:
        rms_quantiles = np.percentile(all_rms_attn, np.linspace(0, 100, n_bins + 1))
        bin_corrections = {}
        bin_counts = {}
        
        for i in range(len(all_rms_attn)):
            for b in range(n_bins):
                if all_rms_attn[i] <= rms_quantiles[b + 1] or b == n_bins - 1:
                    if b not in bin_corrections:
                        bin_corrections[b] = []
                    bin_corrections[b].append(all_errors[i])
                    bin_counts[b] = bin_counts.get(b, 0) + 1
                    break
        
        # Average correction per bin
        lut = {}
        for b in bin_corrections:
            lut[b] = np.mean(bin_corrections[b], axis=0)
        
        print(f"    {n_bins} bins: {[bin_counts.get(b, 0) for b in range(n_bins)]}")
    
    # Strategy 2: Input PC clustering
    print(f"\n  Strategy 2: Input PC clustering (k-means style)")
    
    def simple_kmeans(X, k, n_iter=20):
        """Simple k-means clustering without scipy."""
        n = X.shape[0]
        # Initialize with random data points
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=min(k, n), replace=False)
        centroids = X[idx].copy()
        
        for _ in range(n_iter):
            # Assign labels
            dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])  # (k, n)
            labels = np.argmin(dists, axis=0)
            # Update centroids
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = X[mask].mean(axis=0)
        return centroids, labels
    
    for n_clusters in [4, 8, 16, 32]:
        # Use top 5 PCs as clustering features
        features = X_proj[:, :5].astype(np.float64)
        
        # Simple k-means
        centroids, labels = simple_kmeans(features, n_clusters)
        
        # Build correction LUT
        lut = {}
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                lut[c] = all_errors[mask].mean(axis=0)
            else:
                lut[c] = np.zeros(3584)
        
        # Train error: apply LUT corrections
        E_corrected = np.zeros_like(all_errors)
        for i in range(len(all_errors)):
            E_corrected[i] = all_errors[i] - lut[labels[i]]
        
        train_r2 = 1.0 - np.sum(E_corrected ** 2) / np.sum(all_errors ** 2)
        cluster_sizes = [np.sum(labels == c) for c in range(n_clusters)]
        print(f"    {n_clusters:2d} clusters: train R²={train_r2:.4f}  "
              f"sizes={sorted(cluster_sizes, reverse=True)[:5]}...")
    
    # Strategy 3: φ-level sign pattern hash
    print(f"\n  Strategy 3: φ-level sign pattern features")
    
    # The sign pattern of the hidden state — which dims are +/-
    # This is a 3584-bit string, but we can hash it
    sign_hashes = []
    for h_vec in all_hidden:
        # Split into 28 head blocks of 128 dims
        heads = h_vec.reshape(28, 128)
        # For each head, compute sign-based hash (count of positive dims)
        head_pos_counts = (heads > 0).sum(axis=1)  # (28,) — 0 to 128 each
        # Quantize to ~8 levels
        quantized = np.clip(head_pos_counts // 16, 0, 7)
        sign_hashes.append(tuple(quantized))
    
    n_unique_hashes = len(set(sign_hashes))
    print(f"    Unique sign hashes: {n_unique_hashes} / {len(sign_hashes)}")
    
    # =========================================================================
    #   Part C2: Test the best LUT on the "France" prompt
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C2: Testing LUT corrections on held-out prompts")
    print("=" * 80)
    
    test_prompts = [
        'The capital of France is',
        'The largest ocean is the',
        'The color of grass is',
        'Barack Obama was the',
        'To be or not to',
        'Roses are red, violets are',
    ]
    
    # Build LUT from calibration: use PC-based clustering
    for n_clusters in [4, 8, 16, 32]:
        features_train = X_proj[:, :5].astype(np.float64)
        centroids, train_labels = simple_kmeans(features_train, n_clusters)
        
        # Build correction LUT
        lut = {}
        for c in range(n_clusters):
            mask = train_labels == c
            if mask.sum() > 0:
                lut[c] = all_errors[mask].mean(axis=0)
            else:
                lut[c] = np.zeros(3584)
        
        print(f"\n  LUT with {n_clusters} clusters (5 PCs):")
        
        n_pass = 0
        for prompt in test_prompts:
            p_ids = tokenizer.encode(prompt)
            h = engine.embedding(p_ids)[np.newaxis, :, :]
            for layer_obj in engine.layers:
                if layer_obj.layer_idx == target_layer:
                    full_out = layer_obj(h.copy())
                    spec_out = spec_layer(h.copy())
                    break
                h = layer_obj(h)
            
            # Extract test input features
            h_centered = h[0, -1, :] - all_hidden.mean(axis=0)
            test_proj = h_centered @ Vt_x[:5].T  # (5,)
            
            # Find nearest cluster
            dists = np.linalg.norm(centroids - test_proj[np.newaxis, :], axis=1)
            nearest = int(np.argmin(dists))
            
            # Apply LUT correction to ALL positions
            corrected = spec_out.copy()
            for t in range(corrected.shape[1]):
                corrected[0, t] += lut[nearest]
            
            logits_full = finish_forward(engine, full_out, target_layer)
            logits_corr = finish_forward(engine, corrected, target_layer)
            
            full_id, full_tok, _ = get_top1(logits_full, tokenizer)
            corr_id, corr_tok, corr_margin = get_top1(logits_corr, tokenizer)
            
            match = '✓' if corr_id == full_id else '✗'
            if corr_id == full_id: n_pass += 1
            
            extra = f" margin={corr_margin:.3f}" if 'France' in prompt else ""
            print(f"    {match} {prompt:>35s} → {corr_tok:>8s} (want {full_tok}){extra}")
        
        print(f"    Score: {n_pass}/{len(test_prompts)}")
    
    # =========================================================================
    #   Part C3: Most aggressive LUT — per-prompt nearest-neighbor
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part C3: Nearest-neighbor correction (upper bound)")
    print("=" * 80)
    print("  If we had infinite LUT entries, what's the best we could do?")
    
    # For each test prompt, find the nearest calibration prompt and use its correction
    n_pass = 0
    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                full_out = layer_obj(h.copy())
                spec_out = spec_layer(h.copy())
                break
            h = layer_obj(h)
        
        # Find nearest calibration hidden state (last token)
        test_h = h[0, -1, :]
        dists = np.linalg.norm(all_hidden - test_h[np.newaxis, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_prompt = cal_prompts[nearest_idx]
        nearest_dist = dists[nearest_idx]
        
        # Use that prompt's error as correction
        correction = all_errors[nearest_idx]
        
        corrected = spec_out.copy()
        for t in range(corrected.shape[1]):
            corrected[0, t] += correction
        
        logits_full = finish_forward(engine, full_out, target_layer)
        logits_corr = finish_forward(engine, corrected, target_layer)
        
        full_id, full_tok, _ = get_top1(logits_full, tokenizer)
        corr_id, corr_tok, corr_margin = get_top1(logits_corr, tokenizer)
        
        match = '✓' if corr_id == full_id else '✗'
        if corr_id == full_id: n_pass += 1
        
        extra = f" margin={corr_margin:.3f}" if 'France' in prompt else ""
        print(f"  {match} {prompt:>35s} → {corr_tok:>8s} (want {full_tok}){extra}")
        print(f"      nearest: \"{nearest_prompt}\"  dist={nearest_dist:.2f}")
    
    print(f"  Score: {n_pass}/{len(test_prompts)}")
    
    # =========================================================================
    #   Part D: The error IS the cross-dim state — can we predict it from
    #           a few key scalars?
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part D: Scalar predictors of the error")
    print("=" * 80)
    
    # Build a feature matrix from cross-dim scalars
    feature_matrix = np.column_stack([
        all_rms_attn,
        np.array([f['sign_frac'] for f in all_features]),
        np.array([f['level_mean'] for f in all_features]),
        np.array([f['level_std'] for f in all_features]),
        np.array([f['kurtosis'] for f in all_features]),
        np.array([f['skew'] for f in all_features]),
        np.array([f['neighbor_corr'] for f in all_features]),
        np.array([f['head_norm_cv'] for f in all_features]),
    ])
    feature_names_short = ['rms', 'sign_frac', 'level_mean', 'level_std', 
                           'kurtosis', 'skew', 'neighbor_corr', 'head_norm_cv']
    
    print(f"  Feature matrix: {feature_matrix.shape}")
    
    # Can these 8 scalars predict the 3584-dim error?
    W_feat, _, _, _ = np.linalg.lstsq(feature_matrix, all_errors, rcond=1e-3)
    E_pred_feat = feature_matrix @ W_feat
    r2_feat = 1.0 - np.sum((all_errors - E_pred_feat) ** 2) / np.sum(all_errors ** 2)
    print(f"  8 cross-dim scalars → error R²: {r2_feat:.4f}")
    
    # Add polynomial features (squares and interactions)
    from itertools import combinations_with_replacement
    poly_features = [feature_matrix]
    for i, j in combinations_with_replacement(range(feature_matrix.shape[1]), 2):
        poly_features.append((feature_matrix[:, i] * feature_matrix[:, j])[:, np.newaxis])
    
    poly_matrix = np.column_stack(poly_features)
    print(f"  Polynomial feature matrix: {poly_matrix.shape}")
    
    W_poly, _, _, _ = np.linalg.lstsq(poly_matrix, all_errors, rcond=1e-3)
    E_pred_poly = poly_matrix @ W_poly
    r2_poly = 1.0 - np.sum((all_errors - E_pred_poly) ** 2) / np.sum(all_errors ** 2)
    print(f"  Polynomial (deg 2) → error R²: {r2_poly:.4f}")
    
    # =========================================================================
    #   Part E: The RMS norm hypothesis
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Part E: The RMS norm hypothesis")
    print("=" * 80)
    print("  The spectrometer error comes from treating each dim independently,")
    print("  but RMS norm divides by sqrt(mean(x²)) which depends on ALL dims.")
    print("  If we correct the RMS norm effect, does the error collapse?")
    
    # The spectrometer applies: rule(x[d]) for each dim
    # The real layer applies: rule(x[d] / rms) * some_cross_dim_mixing
    # The first-order correction: error ≈ (x[d] / rms_actual - x[d] / rms_assumed) * slope
    
    # Compute what the RMS norm "sees" vs what the spectrometer "assumes"
    # The spectrometer doesn't normalize — it applies rules to raw hidden[d]
    # But the real layer normalizes hidden by its RMS before projections
    
    # This means the spectrometer error should scale with (1 - 1/rms_ratio)
    # where rms_ratio varies per prompt
    
    print(f"\n  RMS values across prompts:")
    print(f"    mean: {all_rms_attn.mean():.4f}")
    print(f"    std:  {all_rms_attn.std():.4f}")
    print(f"    CV:   {all_rms_attn.std()/all_rms_attn.mean():.4f}")
    
    # The coefficient of variation tells us how much the normalization varies
    # If CV is small, the normalization is nearly constant and per-dim rules
    # absorb it. If CV is large, the correction is prompt-dependent.
    
    # Correlation between RMS and error norm
    corr_rms_err = np.corrcoef(all_rms_attn, err_norms)[0, 1]
    print(f"    corr(RMS, ||error||): {corr_rms_err:.4f}")
    
    # Try: scale the correction by (RMS / mean_RMS) — a 1-parameter correction
    mean_rms = all_rms_attn.mean()
    mean_error = all_errors.mean(axis=0)
    
    print(f"\n  Testing RMS-scaled bias correction:")
    
    n_pass = 0
    for prompt in test_prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for layer_obj in engine.layers:
            if layer_obj.layer_idx == target_layer:
                full_out = layer_obj(h.copy())
                spec_out = spec_layer(h.copy())
                break
            h = layer_obj(h)
        
        test_rms = np.sqrt(np.mean(h[0, -1, :] ** 2))
        rms_scale = test_rms / mean_rms
        
        corrected = spec_out.copy()
        for t in range(corrected.shape[1]):
            t_rms = np.sqrt(np.mean(h[0, t, :] ** 2))
            corrected[0, t] += mean_error * (t_rms / mean_rms)
        
        logits_full = finish_forward(engine, full_out, target_layer)
        logits_corr = finish_forward(engine, corrected, target_layer)
        
        full_id, full_tok, _ = get_top1(logits_full, tokenizer)
        corr_id, corr_tok, corr_margin = get_top1(logits_corr, tokenizer)
        
        match = '✓' if corr_id == full_id else '✗'
        if corr_id == full_id: n_pass += 1
        
        extra = f" margin={corr_margin:.3f}" if 'France' in prompt else ""
        print(f"    {match} {prompt:>35s} → {corr_tok:>8s} (want {full_tok}){extra}")
        print(f"        RMS={test_rms:.4f}, scale={rms_scale:.4f}")
    
    print(f"  Score: {n_pass}/{len(test_prompts)}")
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
