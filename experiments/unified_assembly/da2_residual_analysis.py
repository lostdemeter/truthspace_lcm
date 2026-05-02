#!/usr/bin/env python3
"""
DA2 Residual Analysis: Is Learning Just Error Correction?

Hypothesis: The learned decoder = φ-decoder + error correction

If true:
1. The residual (DA2 - φ_decoder) should be small and structured
2. Learning the residual should be easier than learning depth directly
3. φ-decoder + learned_residual should match DA2 exactly

This would prove that φ-geometry captures the fundamental structure,
and learning only captures the "noise" or corrections.

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
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
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


def build_phi_decoder(features: np.ndarray, depths: np.ndarray, n_dims: int = 50):
    """Build optimized φ-decoder (from previous work)."""
    
    # Find top depth-correlated dimensions
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims = [c[0] for c in correlations[:n_dims]]
    corrs = [c[1] for c in correlations[:n_dims]]
    
    selected_features = features[:, dims]
    
    # Optimize φ-exponents
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
    
    bounds = [(-2, 2) for _ in range(n_dims)]
    
    result = minimize(
        objective,
        initial_exponents,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    # Build weight vector
    weights = np.zeros(384)
    for i, dim in enumerate(dims):
        weights[dim] = np.sign(corrs[i]) * (PHI ** result.x[i])
    weights = weights / np.abs(weights).sum()
    
    return weights, dims, result.x


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


def analyze_residual(features: np.ndarray, depths: np.ndarray, phi_weights: np.ndarray):
    """Analyze the residual between φ-decoder and DA2."""
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    
    # φ-decoder prediction
    phi_pred = features @ phi_weights
    phi_pred = _normalize(phi_pred)
    
    # Residual
    residual = depths - phi_pred
    
    print(f"\n  φ-decoder correlation: {np.corrcoef(phi_pred, depths)[0,1]:.4f}")
    print(f"\n  Residual statistics:")
    print(f"    Mean: {residual.mean():.4f}")
    print(f"    Std: {residual.std():.4f}")
    print(f"    Range: [{residual.min():.4f}, {residual.max():.4f}]")
    print(f"    |Residual| mean: {np.abs(residual).mean():.4f}")
    
    # Is residual correlated with any dimensions?
    print(f"\n  Checking if residual correlates with structure dimensions...")
    
    residual_corrs = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], residual)
        residual_corrs.append((dim, corr))
    
    residual_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n  Top dimensions correlated with residual:")
    for dim, corr in residual_corrs[:10]:
        print(f"    Dim {dim}: {corr:.4f}")
    
    # Is residual structured or noise?
    top_residual_dims = [c[0] for c in residual_corrs[:20]]
    top_residual_corrs = [c[1] for c in residual_corrs[:20]]
    
    max_residual_corr = max(abs(c) for c in top_residual_corrs)
    
    if max_residual_corr < 0.1:
        print(f"\n  FINDING: Residual is mostly NOISE (max corr = {max_residual_corr:.4f})")
        print(f"           The φ-decoder captures almost all the structure!")
    else:
        print(f"\n  FINDING: Residual has STRUCTURE (max corr = {max_residual_corr:.4f})")
        print(f"           Some dimensions encode corrections to φ-decoder.")
    
    return residual, residual_corrs


def learn_residual_correction(features: np.ndarray, residual: np.ndarray, 
                               residual_corrs: list, n_dims: int = 20):
    """Learn a correction for the residual."""
    print("\n" + "=" * 70)
    print("LEARNING RESIDUAL CORRECTION")
    print("=" * 70)
    
    # Use top residual-correlated dimensions
    top_dims = [c[0] for c in residual_corrs[:n_dims]]
    selected_features = features[:, top_dims]
    
    # Simple linear regression on residual
    model = Ridge(alpha=1.0)
    model.fit(selected_features, residual)
    
    residual_pred = model.predict(selected_features)
    
    corr = np.corrcoef(residual_pred, residual)[0, 1]
    mae = np.mean(np.abs(residual_pred - residual))
    
    print(f"\n  Residual prediction:")
    print(f"    Correlation: {corr:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    Residual std: {residual.std():.4f}")
    print(f"    Prediction std: {residual_pred.std():.4f}")
    
    # Build correction weights
    correction_weights = np.zeros(384)
    for i, dim in enumerate(top_dims):
        correction_weights[dim] = model.coef_[i]
    
    return correction_weights, model.intercept_


def test_combined_decoder(model, processor, phi_weights: np.ndarray, 
                          correction_weights: np.ndarray, correction_bias: float,
                          n_test: int = 12):
    """Test φ-decoder + residual correction."""
    print("\n" + "=" * 70)
    print("TESTING COMBINED DECODER (φ + correction)")
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
        
        # φ-decoder only
        phi_depth = np.tensordot(struct_spatial, phi_weights, axes=([2], [0]))
        phi_depth = _normalize(phi_depth)
        
        # Correction
        correction = np.tensordot(struct_spatial, correction_weights, axes=([2], [0]))
        correction = correction + correction_bias
        
        # Combined
        combined_depth = phi_depth + correction
        combined_depth = _normalize(combined_depth)
        
        # Upscale
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        phi_resized = _normalize(zoom(phi_depth, (zoom_h, zoom_w), order=3))
        combined_resized = _normalize(zoom(combined_depth, (zoom_h, zoom_w), order=3))
        
        # Metrics
        phi_corr = np.corrcoef(phi_resized.flatten(), da2_depth.flatten())[0, 1]
        combined_corr = np.corrcoef(combined_resized.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'phi_depth': phi_resized,
            'combined_depth': combined_resized,
            'phi_corr': phi_corr,
            'combined_corr': combined_corr,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        improvement = combined_corr - phi_corr
        print(f"  {img_id}: φ={phi_corr:.3f}, combined={combined_corr:.3f} ({improvement:+.3f}){marker}")
    
    return results


def create_visualization(results: list, residual: np.ndarray, residual_corrs: list):
    """Visualize residual analysis."""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Residual Analysis: Is Learning Just Error Correction?',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Residual distribution
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.hist(residual, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Residual (DA2 - φ-decoder)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Residual Distribution\nMean={residual.mean():.4f}, Std={residual.std():.4f}')
    
    # Plot 2: Top residual-correlated dimensions
    ax2 = fig.add_subplot(gs[0, 2:4])
    top_dims = [c[0] for c in residual_corrs[:20]]
    top_corrs = [c[1] for c in residual_corrs[:20]]
    colors = ['green' if c > 0 else 'red' for c in top_corrs]
    ax2.barh(range(len(top_dims)), top_corrs, color=colors)
    ax2.set_yticks(range(len(top_dims)))
    ax2.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=8)
    ax2.set_xlabel('Correlation with Residual')
    ax2.set_title('Dimensions Correlated with Residual')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    # Plot 3-6: Sample results
    results_sorted = sorted(results, key=lambda x: -x['is_outlier'])[:4]
    
    for i, r in enumerate(results_sorted):
        row = 1 + i // 2
        col_start = (i % 2) * 2
        
        # Original + DA2
        ax = fig.add_subplot(gs[row, col_start])
        ax.imshow(r['rgb'])
        ax.set_title(f"{'OUTLIER ' if r['is_outlier'] else ''}{r['img_id'][-4:]}", fontsize=9)
        ax.axis('off')
        
        # Comparison
        ax = fig.add_subplot(gs[row, col_start + 1])
        
        # Create comparison: left=φ, right=combined
        h, w = r['da2_depth'].shape
        comparison = np.zeros((h, w * 2))
        comparison[:, :w] = r['phi_depth']
        comparison[:, w:] = r['combined_depth']
        
        ax.imshow(comparison, cmap='magma')
        ax.axvline(x=w, color='white', linewidth=2)
        ax.set_title(f'φ={r["phi_corr"]:.3f} | +corr={r["combined_corr"]:.3f}', fontsize=9)
        ax.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    avg_phi = np.mean([r['phi_corr'] for r in results])
    avg_combined = np.mean([r['combined_corr'] for r in results])
    improvement = avg_combined - avg_phi
    
    max_residual_corr = max(abs(c[1]) for c in residual_corrs[:10])
    
    if max_residual_corr < 0.15:
        finding = "RESIDUAL IS MOSTLY NOISE - φ-decoder captures the structure!"
        color = 'lightgreen'
    else:
        finding = f"RESIDUAL HAS STRUCTURE (max corr = {max_residual_corr:.3f})"
        color = 'lightyellow'
    
    summary = f"""
    RESIDUAL ANALYSIS RESULTS
    
    Residual = DA2_depth - φ_decoder_depth
    Residual mean: {residual.mean():.4f} | Residual std: {residual.std():.4f}
    
    Max dimension correlation with residual: {max_residual_corr:.4f}
    
    φ-decoder alone: {avg_phi:.4f}
    φ + learned correction: {avg_combined:.4f} ({improvement:+.4f})
    
    FINDING: {finding}
    """
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color))
    
    output_file = OUTPUT_PATH / "da2_residual_analysis.png"
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
    phi_weights, phi_dims, phi_exponents = build_phi_decoder(features, depths, n_dims=50)
    
    # Analyze residual
    residual, residual_corrs = analyze_residual(features, depths, phi_weights)
    
    # Learn residual correction
    correction_weights, correction_bias = learn_residual_correction(
        features, residual, residual_corrs, n_dims=20
    )
    
    # Test combined decoder
    results = test_combined_decoder(
        model, processor, phi_weights, correction_weights, correction_bias, n_test=10
    )
    
    # Visualize
    viz_file = create_visualization(results, residual, residual_corrs)
    
    # Summary
    avg_phi = np.mean([r['phi_corr'] for r in results])
    avg_combined = np.mean([r['combined_corr'] for r in results])
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  φ-decoder: {avg_phi:.4f}")
    print(f"  φ + correction: {avg_combined:.4f}")
    print(f"  Improvement: {avg_combined - avg_phi:+.4f}")
    
    max_residual_corr = max(abs(c[1]) for c in residual_corrs[:10])
    
    if max_residual_corr < 0.15:
        print(f"\n  CONCLUSION: The residual is mostly noise!")
        print(f"              φ-decoder captures the fundamental structure.")
        print(f"              Learning just captures minor corrections.")
    else:
        print(f"\n  CONCLUSION: The residual has some structure (max corr = {max_residual_corr:.4f})")
        print(f"              Some dimensions encode corrections to φ-geometry.")
