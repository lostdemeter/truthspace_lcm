#!/usr/bin/env python3
"""
φ-Pattern Analysis of Correction Weights

Hypothesis: If the correction weights also follow φ-patterns, then the entire
DA2 decoder is φ-geometric with higher-order terms.

We analyze:
1. Do correction weights cluster around φ-related values?
2. Can we express correction as φ^n terms?
3. Is DA2 = Σ φ^n × dim_i (a φ-polynomial)?

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # 0.618
PHI_SQ = PHI ** 2  # 2.618
PHI_CUBE = PHI ** 3  # 4.236

OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")

# φ-related values to check against
PHI_VALUES = {
    'φ^-3': PHI ** -3,   # 0.236
    'φ^-2': PHI ** -2,   # 0.382
    'φ^-1': PHI ** -1,   # 0.618
    'φ^0': 1.0,          # 1.000
    'φ^0.5': PHI ** 0.5, # 1.272
    'φ^1': PHI,          # 1.618
    'φ^1.5': PHI ** 1.5, # 2.058
    'φ^2': PHI ** 2,     # 2.618
    'φ^2.5': PHI ** 2.5, # 3.330
    'φ^3': PHI ** 3,     # 4.236
}


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


def build_phi_decoder(features: np.ndarray, depths: np.ndarray, n_dims: int = 50):
    """Build optimized φ-decoder and return weights + exponents."""
    
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims = [c[0] for c in correlations[:n_dims]]
    corrs = [c[1] for c in correlations[:n_dims]]
    
    selected_features = features[:, dims]
    
    def objective(exponents):
        weights = np.array([
            np.sign(corrs[i]) * (PHI ** exponents[i])
            for i in range(n_dims)
        ])
        weights = weights / np.abs(weights).sum()
        
        pred = selected_features @ weights
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        
        corr = np.corrcoef(pred, depths)[0, 1]
        return -corr
    
    initial_exponents = np.array([
        np.log(abs(c) + 0.1) / np.log(PHI) for c in corrs
    ])
    
    bounds = [(-3, 3) for _ in range(n_dims)]
    
    result = minimize(
        objective,
        initial_exponents,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    weights = np.zeros(384)
    for i, dim in enumerate(dims):
        weights[dim] = np.sign(corrs[i]) * (PHI ** result.x[i])
    weights = weights / np.abs(weights).sum()
    
    return weights, dims, corrs, result.x


def analyze_correction_phi_patterns(features: np.ndarray, depths: np.ndarray,
                                     phi_weights: np.ndarray, phi_dims: list,
                                     phi_exponents: np.ndarray):
    """Analyze if correction weights follow φ-patterns."""
    print("\n" + "=" * 70)
    print("ANALYZING φ-PATTERNS IN CORRECTION WEIGHTS")
    print("=" * 70)
    
    # Compute φ-decoder prediction
    phi_pred = features @ phi_weights
    phi_pred = _normalize(phi_pred)
    
    # Compute residual
    residual = depths - phi_pred
    
    print(f"\n  Residual std: {residual.std():.4f}")
    
    # Find dimensions correlated with residual
    residual_corrs = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], residual)
        residual_corrs.append((dim, corr))
    
    residual_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Learn linear correction weights
    top_residual_dims = [c[0] for c in residual_corrs[:30]]
    top_residual_corr_values = [c[1] for c in residual_corrs[:30]]
    
    selected_features = features[:, top_residual_dims]
    
    model = Ridge(alpha=1.0)
    model.fit(selected_features, residual)
    
    correction_weights = model.coef_
    
    print(f"\n  Correction weights (top 30 dims):")
    print(f"    Range: [{correction_weights.min():.4f}, {correction_weights.max():.4f}]")
    print(f"    Mean abs: {np.abs(correction_weights).mean():.4f}")
    
    # Analyze if correction weights are φ-related
    print(f"\n  Checking if correction weights follow φ-patterns...")
    
    # Normalize weights to compare magnitudes
    abs_weights = np.abs(correction_weights)
    max_weight = abs_weights.max()
    normalized_weights = abs_weights / max_weight
    
    # Check each weight against φ-values
    phi_matches = []
    
    for i, w in enumerate(normalized_weights):
        best_match = None
        best_diff = float('inf')
        
        for name, phi_val in PHI_VALUES.items():
            # Check both w and 1/w against φ-values
            diff1 = abs(w - phi_val)
            diff2 = abs(w - 1/phi_val) if phi_val > 0 else float('inf')
            
            if diff1 < best_diff:
                best_diff = diff1
                best_match = (name, phi_val, diff1)
            if diff2 < best_diff:
                best_diff = diff2
                best_match = (f"1/{name}", 1/phi_val, diff2)
        
        phi_matches.append({
            'dim': top_residual_dims[i],
            'weight': correction_weights[i],
            'normalized': w,
            'best_phi': best_match[0],
            'phi_value': best_match[1],
            'diff': best_match[2]
        })
    
    # Count how many are close to φ-values
    close_threshold = 0.1
    close_matches = [m for m in phi_matches if m['diff'] < close_threshold]
    
    print(f"\n  Weights close to φ-values (threshold={close_threshold}):")
    print(f"    {len(close_matches)} / {len(phi_matches)} ({100*len(close_matches)/len(phi_matches):.1f}%)")
    
    print(f"\n  Top 10 correction weights and their φ-matches:")
    sorted_matches = sorted(phi_matches, key=lambda x: abs(x['weight']), reverse=True)
    for m in sorted_matches[:10]:
        sign = '+' if m['weight'] > 0 else '-'
        print(f"    Dim {m['dim']:3d}: {sign}{abs(m['weight']):.4f} ≈ {m['best_phi']} ({m['phi_value']:.3f}), diff={m['diff']:.4f}")
    
    return phi_matches, correction_weights, top_residual_dims


def fit_phi_polynomial(features: np.ndarray, depths: np.ndarray, n_dims: int = 50):
    """
    Fit depth as a φ-polynomial: depth = Σ c_i × φ^(e_i) × dim_i
    
    where e_i is constrained to be a multiple of 0.5 (φ-related).
    """
    print("\n" + "=" * 70)
    print("FITTING φ-POLYNOMIAL DECODER")
    print("=" * 70)
    
    # Find top depth-correlated dimensions
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims = [c[0] for c in correlations[:n_dims]]
    corrs = [c[1] for c in correlations[:n_dims]]
    
    selected_features = features[:, dims]
    
    # Constrain exponents to φ-related values (multiples of 0.5)
    phi_exponent_options = np.arange(-3, 3.5, 0.5)  # -3, -2.5, -2, ..., 3
    
    print(f"\n  Fitting with constrained φ-exponents: {list(phi_exponent_options)}")
    
    def objective(params):
        # params[i] is index into phi_exponent_options
        exponent_indices = np.round(params).astype(int)
        exponent_indices = np.clip(exponent_indices, 0, len(phi_exponent_options) - 1)
        exponents = phi_exponent_options[exponent_indices]
        
        weights = np.array([
            np.sign(corrs[i]) * (PHI ** exponents[i])
            for i in range(n_dims)
        ])
        weights = weights / np.abs(weights).sum()
        
        pred = selected_features @ weights
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        
        corr = np.corrcoef(pred, depths)[0, 1]
        return -corr
    
    # Start with middle index (exponent = 0)
    initial_indices = np.ones(n_dims) * (len(phi_exponent_options) // 2)
    
    bounds = [(0, len(phi_exponent_options) - 1) for _ in range(n_dims)]
    
    result = minimize(
        objective,
        initial_indices,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )
    
    # Get final exponents
    final_indices = np.round(result.x).astype(int)
    final_indices = np.clip(final_indices, 0, len(phi_exponent_options) - 1)
    final_exponents = phi_exponent_options[final_indices]
    
    final_corr = -result.fun
    
    print(f"\n  φ-polynomial correlation: {final_corr:.4f}")
    
    # Count exponent distribution
    print(f"\n  Exponent distribution:")
    unique, counts = np.unique(final_exponents, return_counts=True)
    for exp, count in zip(unique, counts):
        print(f"    φ^{exp:.1f}: {count} dimensions")
    
    # Build final weights
    weights = np.zeros(384)
    for i, dim in enumerate(dims):
        weights[dim] = np.sign(corrs[i]) * (PHI ** final_exponents[i])
    weights = weights / np.abs(weights).sum()
    
    return weights, dims, final_exponents, final_corr


def test_phi_polynomial(model, processor, weights: np.ndarray, n_test: int = 12):
    """Test φ-polynomial decoder."""
    print("\n" + "=" * 70)
    print("TESTING φ-POLYNOMIAL DECODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[35:35+n_test]
    outlier_ids = ["000000002587", "000000003501"]
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    results = []
    
    for img_id in test_ids:
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
        
        pred_depth = np.tensordot(struct_spatial, weights, axes=([2], [0]))
        pred_depth = _normalize(pred_depth)
        
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        pred_resized = _normalize(zoom(pred_depth, (zoom_h, zoom_w), order=3))
        
        corr = np.corrcoef(pred_resized.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'pred_depth': pred_resized,
            'corr': corr,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        print(f"  {img_id}: Corr={corr:.3f}{marker}")
    
    return results


def create_visualization(phi_matches: list, poly_exponents: np.ndarray, results: list):
    """Visualize φ-pattern analysis."""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('φ-Pattern Analysis: Is the Entire Decoder φ-Geometric?',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Correction weights vs φ-values
    ax1 = fig.add_subplot(gs[0, 0])
    normalized_weights = [m['normalized'] for m in phi_matches]
    phi_diffs = [m['diff'] for m in phi_matches]
    ax1.scatter(normalized_weights, phi_diffs, alpha=0.6, c='steelblue')
    ax1.axhline(y=0.1, color='red', linestyle='--', label='Close threshold')
    ax1.set_xlabel('Normalized Weight')
    ax1.set_ylabel('Distance to Nearest φ-value')
    ax1.set_title('Correction Weights vs φ-Values')
    ax1.legend()
    
    # Plot 2: φ-polynomial exponent distribution
    ax2 = fig.add_subplot(gs[0, 1])
    unique, counts = np.unique(poly_exponents, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
    ax2.bar([f'φ^{e:.1f}' for e in unique], counts, color=colors)
    ax2.set_xlabel('Exponent')
    ax2.set_ylabel('Count')
    ax2.set_title('φ-Polynomial Exponent Distribution')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: φ-value reference
    ax3 = fig.add_subplot(gs[0, 2])
    phi_names = list(PHI_VALUES.keys())
    phi_vals = list(PHI_VALUES.values())
    ax3.barh(phi_names, phi_vals, color='gold')
    ax3.set_xlabel('Value')
    ax3.set_title('φ-Related Values Reference')
    ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4-6: Sample results
    results_sorted = sorted(results, key=lambda x: -x['is_outlier'])[:6]
    
    for i, r in enumerate(results_sorted):
        row = 1 + i // 3
        col = i % 3
        
        ax = fig.add_subplot(gs[row, col])
        
        # Side by side: DA2 | φ-polynomial
        h, w = r['da2_depth'].shape
        comparison = np.zeros((h, w * 2))
        comparison[:, :w] = r['da2_depth']
        comparison[:, w:] = r['pred_depth']
        
        ax.imshow(comparison, cmap='magma')
        ax.axvline(x=w, color='white', linewidth=2)
        
        title = f"{'OUTLIER ' if r['is_outlier'] else ''}{r['img_id'][-4:]}: {r['corr']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    output_file = OUTPUT_PATH / "da2_phi_correction_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect data
    features, depths = collect_data(model, processor, n_images=30)
    print(f"  Collected {len(features)} patches")
    
    # Build φ-decoder
    print("\n  Building φ-decoder...")
    phi_weights, phi_dims, phi_corrs, phi_exponents = build_phi_decoder(features, depths, n_dims=50)
    
    # Analyze correction φ-patterns
    phi_matches, correction_weights, correction_dims = analyze_correction_phi_patterns(
        features, depths, phi_weights, phi_dims, phi_exponents
    )
    
    # Fit φ-polynomial (constrained exponents)
    poly_weights, poly_dims, poly_exponents, poly_corr = fit_phi_polynomial(
        features, depths, n_dims=50
    )
    
    # Test φ-polynomial
    results = test_phi_polynomial(model, processor, poly_weights, n_test=10)
    
    # Visualize
    viz_file = create_visualization(phi_matches, poly_exponents, results)
    
    # Summary
    avg_corr = np.mean([r['corr'] for r in results])
    outlier_results = [r for r in results if r['is_outlier']]
    
    close_matches = [m for m in phi_matches if m['diff'] < 0.1]
    pct_phi = 100 * len(close_matches) / len(phi_matches)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Correction weights near φ-values: {pct_phi:.1f}%")
    print(f"\n  φ-polynomial decoder:")
    print(f"    Training correlation: {poly_corr:.4f}")
    print(f"    Test correlation: {avg_corr:.4f}")
    
    if outlier_results:
        outlier_corr = np.mean([r['corr'] for r in outlier_results])
        print(f"    Outlier correlation: {outlier_corr:.4f}")
    
    if pct_phi > 50:
        print(f"\n  CONCLUSION: Correction weights ARE φ-related ({pct_phi:.1f}%)")
        print(f"              The entire decoder is φ-geometric!")
    else:
        print(f"\n  CONCLUSION: Correction weights are partially φ-related ({pct_phi:.1f}%)")
        print(f"              Some non-φ structure exists in corrections.")
