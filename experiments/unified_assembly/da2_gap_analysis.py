#!/usr/bin/env python3
"""
DA2 Gap Analysis: What Prevents Perfect Replication?

Current best: 0.91 correlation
Target: 1.0 (perfect match to DA2)

This script analyzes the remaining 9% gap to understand:
1. Is it the φ-exponent precision?
2. Is it missing dimensions?
3. Is it non-linear effects in DA2's decoder?
4. Is it spatial/local effects we're missing?

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, sobel, gaussian_filter
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
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


def collect_data(model, processor, n_images: int = 30):
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
        H_s = depth_h // 14
        W_s = depth_w // 14
        
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
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{n_images}")
    
    return np.array(all_features), np.array(all_depths)


def analyze_gap_sources(features: np.ndarray, depths: np.ndarray):
    """Analyze different potential sources of the gap."""
    print("\n" + "=" * 70)
    print("GAP ANALYSIS: What Prevents Perfect Replication?")
    print("=" * 70)
    
    results = {}
    
    # 1. BASELINE: Unconstrained linear regression (theoretical maximum)
    print("\n  1. THEORETICAL MAXIMUM (unconstrained linear)")
    lr = LinearRegression()
    lr.fit(features, depths)
    lr_pred = lr.predict(features)
    lr_corr = np.corrcoef(lr_pred, depths)[0, 1]
    lr_r2 = r2_score(depths, lr_pred)
    print(f"     Linear regression: Corr={lr_corr:.4f}, R²={lr_r2:.4f}")
    results['linear_max'] = lr_corr
    
    # 2. φ-DECODER with 50 dims (our current best)
    print("\n  2. φ-DECODER (50 dims, optimized exponents)")
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims_50 = [c[0] for c in correlations[:50]]
    corrs_50 = [c[1] for c in correlations[:50]]
    selected_50 = features[:, dims_50]
    
    def objective(exponents, sel_features, corrs):
        n = len(corrs)
        weights = np.array([np.sign(corrs[i]) * (PHI ** exponents[i]) for i in range(n)])
        weights = weights / np.abs(weights).sum()
        pred = sel_features @ weights
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return -np.corrcoef(pred, depths)[0, 1]
    
    init_exp = np.array([np.log(abs(c) + 0.1) / np.log(PHI) for c in corrs_50])
    result = minimize(objective, init_exp, args=(selected_50, corrs_50),
                     method='L-BFGS-B', bounds=[(-3, 3)] * 50)
    phi_50_corr = -result.fun
    print(f"     φ-decoder (50 dims): Corr={phi_50_corr:.4f}")
    results['phi_50'] = phi_50_corr
    
    # 3. φ-DECODER with ALL 384 dims
    print("\n  3. φ-DECODER (ALL 384 dims)")
    all_dims = [c[0] for c in correlations]
    all_corrs = [c[1] for c in correlations]
    
    # Use top 100 for optimization (384 is too many parameters)
    dims_100 = [c[0] for c in correlations[:100]]
    corrs_100 = [c[1] for c in correlations[:100]]
    selected_100 = features[:, dims_100]
    
    init_exp_100 = np.array([np.log(abs(c) + 0.1) / np.log(PHI) for c in corrs_100])
    result_100 = minimize(objective, init_exp_100, args=(selected_100, corrs_100),
                         method='L-BFGS-B', bounds=[(-3, 3)] * 100)
    phi_100_corr = -result_100.fun
    print(f"     φ-decoder (100 dims): Corr={phi_100_corr:.4f}")
    results['phi_100'] = phi_100_corr
    
    # 4. LINEAR on top 50 dims (is φ-scaling helping or hurting?)
    print("\n  4. LINEAR DECODER (50 dims, learned weights)")
    lr_50 = Ridge(alpha=1.0)
    lr_50.fit(selected_50, depths)
    lr_50_pred = lr_50.predict(selected_50)
    lr_50_corr = np.corrcoef(lr_50_pred, depths)[0, 1]
    print(f"     Linear (50 dims): Corr={lr_50_corr:.4f}")
    results['linear_50'] = lr_50_corr
    
    # 5. Check for NON-LINEAR effects
    print("\n  5. NON-LINEAR EFFECTS")
    # Add squared terms for top 20 dims
    top_20 = features[:, dims_50[:20]]
    top_20_sq = top_20 ** 2
    features_nonlin = np.hstack([top_20, top_20_sq])
    
    lr_nonlin = Ridge(alpha=1.0)
    lr_nonlin.fit(features_nonlin, depths)
    nonlin_pred = lr_nonlin.predict(features_nonlin)
    nonlin_corr = np.corrcoef(nonlin_pred, depths)[0, 1]
    print(f"     Linear + squared (20 dims): Corr={nonlin_corr:.4f}")
    results['nonlinear'] = nonlin_corr
    
    # 6. Check for INTERACTION effects
    print("\n  6. INTERACTION EFFECTS")
    # Add pairwise products for top 10 dims
    top_10 = features[:, dims_50[:10]]
    interactions = []
    for i in range(10):
        for j in range(i+1, 10):
            interactions.append(top_10[:, i] * top_10[:, j])
    interactions = np.array(interactions).T
    features_interact = np.hstack([top_10, interactions])
    
    lr_interact = Ridge(alpha=1.0)
    lr_interact.fit(features_interact, depths)
    interact_pred = lr_interact.predict(features_interact)
    interact_corr = np.corrcoef(interact_pred, depths)[0, 1]
    print(f"     Linear + interactions (10 dims): Corr={interact_corr:.4f}")
    results['interactions'] = interact_corr
    
    return results


def analyze_residual_structure(features: np.ndarray, depths: np.ndarray):
    """Analyze what's in the residual that we're missing."""
    print("\n" + "=" * 70)
    print("RESIDUAL STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Build best φ-decoder
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims = [c[0] for c in correlations[:50]]
    corrs = [c[1] for c in correlations[:50]]
    selected = features[:, dims]
    
    def objective(exponents):
        weights = np.array([np.sign(corrs[i]) * (PHI ** exponents[i]) for i in range(50)])
        weights = weights / np.abs(weights).sum()
        pred = selected @ weights
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return -np.corrcoef(pred, depths)[0, 1]
    
    init_exp = np.array([np.log(abs(c) + 0.1) / np.log(PHI) for c in corrs])
    result = minimize(objective, init_exp, method='L-BFGS-B', bounds=[(-3, 3)] * 50)
    
    # Compute residual
    final_exp = result.x
    weights = np.array([np.sign(corrs[i]) * (PHI ** final_exp[i]) for i in range(50)])
    weights = weights / np.abs(weights).sum()
    phi_pred = selected @ weights
    phi_pred = (phi_pred - phi_pred.min()) / (phi_pred.max() - phi_pred.min() + 1e-10)
    
    residual = depths - phi_pred
    
    print(f"\n  Residual statistics:")
    print(f"    Mean: {residual.mean():.4f}")
    print(f"    Std: {residual.std():.4f}")
    print(f"    Skewness: {np.mean((residual - residual.mean())**3) / residual.std()**3:.4f}")
    print(f"    Kurtosis: {np.mean((residual - residual.mean())**4) / residual.std()**4:.4f}")
    
    # Is residual correlated with depth itself? (non-linearity)
    res_depth_corr = np.corrcoef(residual, depths)[0, 1]
    print(f"\n  Residual-Depth correlation: {res_depth_corr:.4f}")
    if abs(res_depth_corr) > 0.1:
        print(f"    → Suggests non-linear depth relationship")
    
    # Is residual correlated with prediction? (systematic bias)
    res_pred_corr = np.corrcoef(residual, phi_pred)[0, 1]
    print(f"  Residual-Prediction correlation: {res_pred_corr:.4f}")
    if abs(res_pred_corr) > 0.1:
        print(f"    → Suggests systematic over/under-estimation")
    
    # Check if residual has spatial structure (would need neighbor info)
    print(f"\n  Residual variance by depth range:")
    for low, high in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        mask = (depths >= low) & (depths < high)
        if mask.sum() > 100:
            res_std = residual[mask].std()
            print(f"    Depth [{low:.2f}-{high:.2f}]: std={res_std:.4f}, n={mask.sum()}")
    
    return residual


def summarize_findings(results: dict):
    """Summarize what's causing the gap."""
    print("\n" + "=" * 70)
    print("SUMMARY: Sources of the Gap")
    print("=" * 70)
    
    linear_max = results['linear_max']
    phi_50 = results['phi_50']
    phi_100 = results['phi_100']
    linear_50 = results['linear_50']
    nonlinear = results['nonlinear']
    interactions = results['interactions']
    
    print(f"\n  Performance comparison:")
    print(f"    Theoretical max (linear, all dims): {linear_max:.4f}")
    print(f"    φ-decoder (100 dims):               {phi_100:.4f}")
    print(f"    φ-decoder (50 dims):                {phi_50:.4f}")
    print(f"    Linear (50 dims):                   {linear_50:.4f}")
    print(f"    Linear + squared:                   {nonlinear:.4f}")
    print(f"    Linear + interactions:              {interactions:.4f}")
    
    print(f"\n  Gap breakdown:")
    
    # Gap from φ-scaling vs linear
    phi_vs_linear = linear_50 - phi_50
    print(f"    φ-scaling penalty: {phi_vs_linear:+.4f}")
    if phi_vs_linear > 0.01:
        print(f"      → φ-exponents don't perfectly match optimal linear weights")
    
    # Gap from dimension count
    dim_gap = phi_100 - phi_50
    print(f"    More dimensions (+50): {dim_gap:+.4f}")
    
    # Gap from non-linearity
    nonlin_gap = nonlinear - linear_50
    print(f"    Non-linear effects: {nonlin_gap:+.4f}")
    if nonlin_gap > 0.01:
        print(f"      → DA2's decoder has some non-linear components")
    
    # Gap from interactions
    interact_gap = interactions - linear_50
    print(f"    Interaction effects: {interact_gap:+.4f}")
    
    # Remaining unexplained gap
    best_achievable = max(nonlinear, interactions, phi_100)
    remaining = linear_max - best_achievable
    print(f"\n    Remaining gap to theoretical max: {remaining:.4f}")
    
    print(f"\n  CONCLUSIONS:")
    if phi_vs_linear < 0.01:
        print(f"    ✓ φ-scaling matches linear almost perfectly")
    else:
        print(f"    ✗ φ-scaling has {phi_vs_linear:.1%} penalty vs optimal linear")
    
    if nonlin_gap > 0.02:
        print(f"    ! Non-linear effects contribute {nonlin_gap:.1%}")
    
    if remaining < 0.02:
        print(f"    ✓ We're within {remaining:.1%} of theoretical maximum")
    else:
        print(f"    ? {remaining:.1%} gap unexplained (likely DA2's neck/head)")


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect data
    features, depths = collect_data(model, processor, n_images=30)
    print(f"  Collected {len(features)} patches")
    
    # Analyze gap sources
    results = analyze_gap_sources(features, depths)
    
    # Analyze residual structure
    residual = analyze_residual_structure(features, depths)
    
    # Summarize
    summarize_findings(results)
