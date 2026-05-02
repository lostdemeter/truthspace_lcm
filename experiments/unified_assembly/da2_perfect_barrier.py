#!/usr/bin/env python3
"""
Perfect Barrier Analysis: What Prevents φ from Reaching 100%?

We know:
- φ-decoder achieves ~0.88 correlation
- φ + error = DA2 (trivially perfect at α=1)
- Error has 0.60 correlation with depth

Question: What's in the error that φ CAN'T capture?

Hypotheses:
1. Non-linear relationships (φ is linear)
2. Higher-order interactions between dimensions
3. Spatial/local information (φ treats patches independently)
4. Information in discarded dimensions (bottom 334)
5. Something fundamental about the φ-basis

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, sobel, laplace
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def load_da2():
    """Load DA2 model."""
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    
    return model, processor


def extract_structure(model, processor, rgb: np.ndarray):
    """Extract DA2's backbone structure."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
        
        full_output = model(inputs['pixel_values'])
        da2_depth = full_output.predicted_depth.squeeze().numpy()
    
    return structure, _normalize(da2_depth)


def collect_data(model, processor, n_images: int = 25):
    """Collect patch-level data."""
    print("\n  Collecting data...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
        structure = structure[1:]
        N, C = structure.shape
        
        depth_h, depth_w = da2_depth.shape
        H_s, W_s = depth_h // 14, depth_w // 14
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        for y in range(H_s):
            for x in range(W_s):
                all_features.append(struct_spatial[y, x])
                all_depths.append(depth_small[y, x])
    
    return np.array(all_features), np.array(all_depths)


def test_hypothesis_1_nonlinearity(features: np.ndarray, depths: np.ndarray):
    """
    Hypothesis 1: φ is linear, but depth encoding is non-linear.
    Test: Does adding polynomial features improve correlation?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Non-linearity")
    print("=" * 70)
    
    # Get top 20 dimensions
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_dims = [c[0] for c in correlations[:20]]
    top_features = features[:, top_dims]
    
    # Linear baseline
    lr = Ridge(alpha=1.0)
    lr.fit(top_features, depths)
    linear_pred = lr.predict(top_features)
    linear_corr = np.corrcoef(linear_pred, depths)[0, 1]
    
    print(f"\n  Linear (20 dims): {linear_corr:.4f}")
    
    # Polynomial degree 2
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    features_poly2 = poly2.fit_transform(top_features)
    
    lr2 = Ridge(alpha=1.0)
    lr2.fit(features_poly2, depths)
    poly2_pred = lr2.predict(features_poly2)
    poly2_corr = np.corrcoef(poly2_pred, depths)[0, 1]
    
    print(f"  Polynomial deg 2: {poly2_corr:.4f} ({features_poly2.shape[1]} features)")
    
    # Polynomial degree 3
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    features_poly3 = poly3.fit_transform(top_features[:, :10])  # Use fewer to avoid explosion
    
    lr3 = Ridge(alpha=1.0)
    lr3.fit(features_poly3, depths)
    poly3_pred = lr3.predict(features_poly3)
    poly3_corr = np.corrcoef(poly3_pred, depths)[0, 1]
    
    print(f"  Polynomial deg 3: {poly3_corr:.4f} ({features_poly3.shape[1]} features)")
    
    improvement = poly2_corr - linear_corr
    print(f"\n  Non-linearity improvement: {improvement:+.4f}")
    
    if improvement > 0.05:
        print(f"  → SIGNIFICANT non-linearity exists!")
    else:
        print(f"  → Non-linearity is NOT the barrier")
    
    return linear_corr, poly2_corr, poly3_corr


def test_hypothesis_2_interactions(features: np.ndarray, depths: np.ndarray):
    """
    Hypothesis 2: Dimension interactions matter.
    Test: Do pairwise products improve correlation?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Dimension Interactions")
    print("=" * 70)
    
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_dims = [c[0] for c in correlations[:20]]
    top_features = features[:, top_dims]
    
    # Linear baseline
    lr = Ridge(alpha=1.0)
    lr.fit(top_features, depths)
    linear_pred = lr.predict(top_features)
    linear_corr = np.corrcoef(linear_pred, depths)[0, 1]
    
    # Add pairwise interactions only (not squares)
    interactions = []
    for i in range(20):
        for j in range(i+1, 20):
            interactions.append(top_features[:, i] * top_features[:, j])
    
    interactions = np.array(interactions).T
    features_interact = np.hstack([top_features, interactions])
    
    lr_interact = Ridge(alpha=1.0)
    lr_interact.fit(features_interact, depths)
    interact_pred = lr_interact.predict(features_interact)
    interact_corr = np.corrcoef(interact_pred, depths)[0, 1]
    
    print(f"\n  Linear (20 dims): {linear_corr:.4f}")
    print(f"  + Interactions: {interact_corr:.4f} ({features_interact.shape[1]} features)")
    
    improvement = interact_corr - linear_corr
    print(f"\n  Interaction improvement: {improvement:+.4f}")
    
    if improvement > 0.05:
        print(f"  → SIGNIFICANT interactions exist!")
    else:
        print(f"  → Interactions are NOT the barrier")
    
    return linear_corr, interact_corr


def test_hypothesis_3_discarded_dims(features: np.ndarray, depths: np.ndarray):
    """
    Hypothesis 3: Information in discarded dimensions.
    Test: Does using ALL 384 dims significantly improve?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Discarded Dimensions")
    print("=" * 70)
    
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Test different numbers of dimensions
    results = []
    for n_dims in [50, 100, 200, 384]:
        top_dims = [c[0] for c in correlations[:n_dims]]
        top_features = features[:, top_dims]
        
        lr = Ridge(alpha=1.0)
        lr.fit(top_features, depths)
        pred = lr.predict(top_features)
        corr = np.corrcoef(pred, depths)[0, 1]
        
        results.append((n_dims, corr))
        print(f"  {n_dims:3d} dims: {corr:.4f}")
    
    improvement = results[-1][1] - results[0][1]
    print(f"\n  Improvement from 50→384 dims: {improvement:+.4f}")
    
    if improvement > 0.05:
        print(f"  → Discarded dims contain SIGNIFICANT information!")
    else:
        print(f"  → Discarded dims are NOT the barrier")
    
    return results


def test_hypothesis_4_theoretical_max(features: np.ndarray, depths: np.ndarray):
    """
    Hypothesis 4: What's the theoretical maximum?
    Test: Unconstrained linear regression on all features.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Theoretical Maximum")
    print("=" * 70)
    
    # Full linear regression
    lr = Ridge(alpha=0.1)  # Light regularization
    lr.fit(features, depths)
    pred = lr.predict(features)
    
    linear_corr = np.corrcoef(pred, depths)[0, 1]
    r_squared = 1 - np.sum((depths - pred)**2) / np.sum((depths - depths.mean())**2)
    
    print(f"\n  Full linear regression (384 dims):")
    print(f"    Correlation: {linear_corr:.4f}")
    print(f"    R²: {r_squared:.4f}")
    
    # Residual analysis
    residual = depths - pred
    residual_std = residual.std()
    
    print(f"\n  Residual analysis:")
    print(f"    Residual std: {residual_std:.4f}")
    print(f"    Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    # Is residual correlated with any feature?
    max_res_corr = 0
    max_res_dim = 0
    for dim in range(features.shape[1]):
        corr = np.abs(np.corrcoef(features[:, dim], residual)[0, 1])
        if corr > max_res_corr:
            max_res_corr = corr
            max_res_dim = dim
    
    print(f"    Max residual-feature correlation: {max_res_corr:.4f} (dim {max_res_dim})")
    
    if max_res_corr < 0.1:
        print(f"\n  → Residual is UNCORRELATED with features")
        print(f"  → Linear model has extracted ALL available information!")
        print(f"  → Theoretical max ≈ {linear_corr:.4f}")
    
    return linear_corr, r_squared, residual_std


def test_hypothesis_5_phi_vs_optimal(features: np.ndarray, depths: np.ndarray):
    """
    Hypothesis 5: Is φ-scaling suboptimal?
    Test: Compare φ-weights vs optimal linear weights.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: φ-Scaling vs Optimal")
    print("=" * 70)
    
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_dims = [c[0] for c in correlations[:50]]
    top_corrs = np.array([c[1] for c in correlations[:50]])
    top_features = features[:, top_dims]
    
    # φ-scaled weights
    phi_scales = np.array([PHI ** (-i/10) for i in range(50)])
    phi_weights = phi_scales * np.sign(top_corrs)
    phi_weights = phi_weights / np.abs(phi_weights).sum()
    
    phi_pred = top_features @ phi_weights
    phi_pred = _normalize(phi_pred)
    phi_corr = np.corrcoef(phi_pred, depths)[0, 1]
    
    # Optimal linear weights
    lr = Ridge(alpha=1.0)
    lr.fit(top_features, depths)
    optimal_pred = lr.predict(top_features)
    optimal_corr = np.corrcoef(optimal_pred, depths)[0, 1]
    
    print(f"\n  φ-scaled weights (50 dims): {phi_corr:.4f}")
    print(f"  Optimal linear weights: {optimal_corr:.4f}")
    
    gap = optimal_corr - phi_corr
    print(f"\n  φ vs optimal gap: {gap:.4f}")
    
    if gap > 0.05:
        print(f"  → φ-scaling is SUBOPTIMAL by {gap:.1%}")
    else:
        print(f"  → φ-scaling is NEAR-OPTIMAL")
    
    # Analyze weight differences
    optimal_weights = lr.coef_
    optimal_weights_norm = optimal_weights / np.abs(optimal_weights).sum()
    
    weight_corr = np.corrcoef(phi_weights, optimal_weights_norm)[0, 1]
    print(f"\n  φ-weights vs optimal-weights correlation: {weight_corr:.4f}")
    
    return phi_corr, optimal_corr, weight_corr


def summarize_barriers(results: dict):
    """Summarize what's preventing perfect replication."""
    print("\n" + "=" * 70)
    print("SUMMARY: What Prevents Perfect Replication?")
    print("=" * 70)
    
    theoretical_max = results['theoretical_max']
    phi_corr = results['phi_corr']
    optimal_corr = results['optimal_corr']
    
    gap_to_perfect = 1.0 - theoretical_max
    gap_phi_to_optimal = optimal_corr - phi_corr
    gap_optimal_to_max = theoretical_max - optimal_corr
    
    print(f"\n  Correlation breakdown:")
    print(f"    φ-decoder (50 dims):      {phi_corr:.4f}")
    print(f"    Optimal linear (50 dims): {optimal_corr:.4f}")
    print(f"    Theoretical max (384):    {theoretical_max:.4f}")
    print(f"    Perfect:                  1.0000")
    
    print(f"\n  Gap analysis:")
    print(f"    φ → Optimal:    {gap_phi_to_optimal:.4f} ({100*gap_phi_to_optimal:.1f}%)")
    print(f"    Optimal → Max:  {gap_optimal_to_max:.4f} ({100*gap_optimal_to_max:.1f}%)")
    print(f"    Max → Perfect:  {gap_to_perfect:.4f} ({100*gap_to_perfect:.1f}%)")
    
    print(f"\n  CONCLUSIONS:")
    
    if gap_to_perfect < 0.02:
        print(f"    ✓ Theoretical max is {theoretical_max:.4f} - nearly perfect!")
        print(f"    → The backbone CONTAINS almost all depth information")
        print(f"    → The remaining {100*gap_to_perfect:.1f}% is in DA2's neck/head")
    
    if gap_phi_to_optimal < 0.02:
        print(f"    ✓ φ-scaling is near-optimal (gap = {gap_phi_to_optimal:.4f})")
    else:
        print(f"    ! φ-scaling loses {100*gap_phi_to_optimal:.1f}% vs optimal")
    
    if gap_optimal_to_max > 0.02:
        print(f"    ! Using only 50 dims loses {100*gap_optimal_to_max:.1f}%")
        print(f"    → More dimensions would help")


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect data
    features, depths = collect_data(model, processor, n_images=25)
    print(f"  Collected {len(features)} patches")
    
    # Test hypotheses
    linear_corr, poly2_corr, poly3_corr = test_hypothesis_1_nonlinearity(features, depths)
    linear_corr2, interact_corr = test_hypothesis_2_interactions(features, depths)
    dim_results = test_hypothesis_3_discarded_dims(features, depths)
    theoretical_max, r_squared, residual_std = test_hypothesis_4_theoretical_max(features, depths)
    phi_corr, optimal_corr, weight_corr = test_hypothesis_5_phi_vs_optimal(features, depths)
    
    # Summarize
    results = {
        'theoretical_max': theoretical_max,
        'phi_corr': phi_corr,
        'optimal_corr': optimal_corr,
        'poly2_corr': poly2_corr,
        'interact_corr': interact_corr
    }
    
    summarize_barriers(results)
