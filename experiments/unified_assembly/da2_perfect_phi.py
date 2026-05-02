#!/usr/bin/env python3
"""
Perfect φ-Decoder: Attempting Near-Perfect DA2 Replication

Combining all improvements:
1. All 384 dimensions (not just top 50)
2. Calibration curve to fix systematic bias
3. Better upsampling to match DA2's spatial refinement

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, gaussian_filter
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
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


def collect_training_data(model, processor, n_images: int = 30):
    """Collect patch-level data for training."""
    print("\n  Collecting training data...")
    
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


def build_full_phi_decoder(features: np.ndarray, depths: np.ndarray, n_dims: int = 100):
    """Build φ-decoder using more dimensions."""
    print(f"\n  Building φ-decoder with {n_dims} dimensions...")
    
    # Find correlations for all dimensions
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
    
    bounds = [(-3, 3) for _ in range(n_dims)]
    
    result = minimize(
        objective,
        initial_exponents,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )
    
    # Build weight vector for all 384 dims
    weights = np.zeros(384)
    for i, dim in enumerate(dims):
        weights[dim] = np.sign(corrs[i]) * (PHI ** result.x[i])
    weights = weights / np.abs(weights).sum()
    
    train_corr = -result.fun
    print(f"    Training correlation: {train_corr:.4f}")
    
    return weights, dims, result.x, train_corr


def build_calibration_curve(features: np.ndarray, depths: np.ndarray, weights: np.ndarray):
    """Build calibration curve to fix systematic bias."""
    print("\n  Building calibration curve...")
    
    # Get predictions
    pred = features @ weights
    pred = _normalize(pred)
    
    # Bin predictions and compute mean actual depth for each bin
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    calibrated_values = []
    valid_centers = []
    
    for i in range(n_bins):
        mask = (pred >= bin_edges[i]) & (pred < bin_edges[i+1])
        if mask.sum() > 50:
            mean_actual = depths[mask].mean()
            calibrated_values.append(mean_actual)
            valid_centers.append(bin_centers[i])
    
    # Fit spline for smooth calibration
    if len(valid_centers) > 3:
        spline = UnivariateSpline(valid_centers, calibrated_values, s=0.01)
    else:
        spline = None
    
    # Test calibration
    if spline:
        calibrated_pred = spline(np.clip(pred, valid_centers[0], valid_centers[-1]))
        calibrated_pred = _normalize(calibrated_pred)
        calib_corr = np.corrcoef(calibrated_pred, depths)[0, 1]
        print(f"    Calibrated correlation: {calib_corr:.4f}")
    else:
        calib_corr = 0
    
    return spline, valid_centers


def decode_single_image(model, processor, rgb: np.ndarray, weights: np.ndarray, 
                        calibration_spline=None, valid_centers=None):
    """Decode a single image with full pipeline."""
    
    structure, da2_depth = extract_structure(model, processor, rgb)
    
    # Skip CLS token
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
    
    # φ-decode
    phi_depth = np.tensordot(struct_spatial, weights, axes=([2], [0]))
    phi_depth = _normalize(phi_depth)
    
    # Apply calibration
    if calibration_spline and valid_centers:
        phi_calibrated = calibration_spline(
            np.clip(phi_depth, valid_centers[0], valid_centers[-1])
        )
        phi_calibrated = _normalize(phi_calibrated)
    else:
        phi_calibrated = phi_depth
    
    # Upscale with bicubic + slight smoothing (mimics DA2's neck)
    zoom_h = depth_h / H_s
    zoom_w = depth_w / W_s
    
    # Method 1: Simple bicubic
    phi_upscaled = zoom(phi_calibrated, (zoom_h, zoom_w), order=3)
    phi_upscaled = _normalize(phi_upscaled)
    
    # Method 2: Bicubic + edge-aware smoothing
    phi_smooth = gaussian_filter(phi_upscaled, sigma=1.0)
    
    # Blend: keep edges from upscaled, smooth from gaussian
    edge_weight = 0.7
    phi_final = edge_weight * phi_upscaled + (1 - edge_weight) * phi_smooth
    phi_final = _normalize(phi_final)
    
    return {
        'phi_raw': _normalize(zoom(phi_depth, (zoom_h, zoom_w), order=3)),
        'phi_calibrated': phi_upscaled,
        'phi_final': phi_final,
        'da2_depth': da2_depth,
        'patch_depth': phi_depth,
        'H_s': H_s,
        'W_s': W_s
    }


def visualize_comparison(model, processor, weights, calibration_spline, valid_centers):
    """Visualize comparison on multiple images."""
    print("\n" + "=" * 70)
    print("VISUALIZING COMPARISON TO DA2")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Select diverse test images
    test_ids = available_ids[40:46]  # Regular images
    outlier_ids = ["000000002587", "000000003501"]  # Known outliers
    
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    results = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        decoded = decode_single_image(
            model, processor, rgb, weights, calibration_spline, valid_centers
        )
        
        # Compute metrics
        da2 = decoded['da2_depth']
        phi_raw = decoded['phi_raw']
        phi_calib = decoded['phi_calibrated']
        phi_final = decoded['phi_final']
        
        corr_raw = np.corrcoef(phi_raw.flatten(), da2.flatten())[0, 1]
        corr_calib = np.corrcoef(phi_calib.flatten(), da2.flatten())[0, 1]
        corr_final = np.corrcoef(phi_final.flatten(), da2.flatten())[0, 1]
        
        # Resize RGB for display
        depth_h, depth_w = da2.shape
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2': da2,
            'phi_raw': phi_raw,
            'phi_calib': phi_calib,
            'phi_final': phi_final,
            'corr_raw': corr_raw,
            'corr_calib': corr_calib,
            'corr_final': corr_final,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        print(f"  {img_id}: raw={corr_raw:.3f}, calib={corr_calib:.3f}, final={corr_final:.3f}{marker}")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 4 * len(results) + 1))
    fig.suptitle('Perfect φ-Decoder: Comparison to DA2\n'
                 '100 dims + calibration + smoothing',
                 fontsize=14, fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(len(results) + 1, 6, figure=fig, hspace=0.15, wspace=0.05,
                          height_ratios=[1] * len(results) + [0.2])
    
    headers = ['Original', 'DA2 Depth', 'φ Raw', 'φ Calibrated', 'φ Final', 'Difference']
    
    for row, r in enumerate(results):
        # Original
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(r['rgb'])
        if row == 0:
            ax.set_title(headers[0], fontsize=10)
        label = f"{'OUTLIER ' if r['is_outlier'] else ''}{r['img_id'][-4:]}"
        ax.set_ylabel(label, fontsize=8, color='red' if r['is_outlier'] else 'black')
        ax.axis('off')
        
        # DA2 Depth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(r['da2'], cmap='magma')
        if row == 0:
            ax.set_title(headers[1], fontsize=10)
        ax.axis('off')
        
        # φ Raw
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(r['phi_raw'], cmap='magma')
        title = f'{headers[2]} ({r["corr_raw"]:.3f})' if row == 0 else f'{r["corr_raw"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # φ Calibrated
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(r['phi_calib'], cmap='magma')
        title = f'{headers[3]} ({r["corr_calib"]:.3f})' if row == 0 else f'{r["corr_calib"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # φ Final
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(r['phi_final'], cmap='magma')
        title = f'{headers[4]} ({r["corr_final"]:.3f})' if row == 0 else f'{r["corr_final"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Difference
        ax = fig.add_subplot(gs[row, 5])
        diff = r['phi_final'] - r['da2']
        ax.imshow(diff, cmap='RdBu', vmin=-0.2, vmax=0.2)
        if row == 0:
            ax.set_title(headers[5], fontsize=10)
        ax.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[len(results), :])
    ax_summary.axis('off')
    
    avg_raw = np.mean([r['corr_raw'] for r in results])
    avg_calib = np.mean([r['corr_calib'] for r in results])
    avg_final = np.mean([r['corr_final'] for r in results])
    
    outlier_results = [r for r in results if r['is_outlier']]
    if outlier_results:
        outlier_final = np.mean([r['corr_final'] for r in outlier_results])
    else:
        outlier_final = 0
    
    summary = f"""
    PERFECT φ-DECODER RESULTS
    
    φ Raw (100 dims):        {avg_raw:.4f}
    φ + Calibration:         {avg_calib:.4f}
    φ + Calib + Smoothing:   {avg_final:.4f}
    Outliers:                {outlier_final:.4f}
    
    Theoretical max: 0.9865 | Current: {avg_final:.4f} | Gap: {0.9865 - avg_final:.4f}
    """
    color = 'lightgreen' if avg_final > 0.94 else 'lightyellow'
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color))
    
    output_file = OUTPUT_PATH / "da2_perfect_phi.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect training data
    features, depths = collect_training_data(model, processor, n_images=30)
    print(f"  Collected {len(features)} patches")
    
    # Build full φ-decoder with 100 dimensions
    weights, dims, exponents, train_corr = build_full_phi_decoder(
        features, depths, n_dims=100
    )
    
    # Build calibration curve
    calibration_spline, valid_centers = build_calibration_curve(features, depths, weights)
    
    # Visualize comparison
    results = visualize_comparison(model, processor, weights, calibration_spline, valid_centers)
    
    # Summary
    avg_final = np.mean([r['corr_final'] for r in results])
    outlier_results = [r for r in results if r['is_outlier']]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Average correlation: {avg_final:.4f}")
    print(f"  Theoretical max: 0.9865")
    print(f"  Gap: {0.9865 - avg_final:.4f}")
    
    if outlier_results:
        outlier_avg = np.mean([r['corr_final'] for r in outlier_results])
        print(f"\n  Outlier average: {outlier_avg:.4f}")
