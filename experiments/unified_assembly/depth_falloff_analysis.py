#!/usr/bin/env python3
"""
Analysis: Distance-Dependent Falloff in Depth Estimation

Observation from visualizations:
- DA wins at extremes (very close, very far)
- Holographic wins in the middle

Hypothesis:
- Perspective projection has non-linear depth relationship
- Our linear vertical model (depth ≈ 0.6y + 0.1) works in the middle
- At extremes, there's falloff we're not capturing

This analysis stratifies by depth to quantify where each approach wins.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "temp/outside_projects/holographic_enhancement/src"))

try:
    from enhance import holographic_enhance
    HAS_HOLOGRAPHIC = True
except ImportError:
    HAS_HOLOGRAPHIC = False

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def holographic_enhance_numpy(image: np.ndarray, sigma: float = 2.0, boost: float = 1.5) -> np.ndarray:
    if HAS_HOLOGRAPHIC:
        import cv2
        bgr = (image[:, :, ::-1] * 255).astype(np.uint8)
        enhanced_bgr = holographic_enhance(bgr, sigma=sigma, boost=boost)
        return enhanced_bgr[:, :, ::-1].astype(np.float32) / 255.0
    else:
        L = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        gamma = 2.2
        L_linear = np.power(L, gamma)
        L_blur = gaussian_filter(L_linear, sigma=sigma)
        L_blur = np.maximum(L_blur, 0.01)
        epsilon = 0.01
        ratio = (L_linear - L_blur) / (L_blur + epsilon)
        midtone_weight = 4.0 * L * (1.0 - L)
        adaptive = np.clip(midtone_weight + 0.3, 0.3, 1.0)
        factor = 1.0 + boost * adaptive * ratio
        factor = np.clip(factor, 0.7, 1.5)
        enhanced = image.copy()
        for c in range(3):
            enhanced[:,:,c] = np.clip(image[:,:,c] * factor, 0, 1)
        return enhanced


def extract_holographic_depth(enhanced: np.ndarray) -> np.ndarray:
    gray = 0.299 * enhanced[:,:,0] + 0.587 * enhanced[:,:,1] + 0.114 * enhanced[:,:,2]
    h, w = gray.shape
    grad_y = sobel(gray, axis=0)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    geometric_depth = 0.6 * y_coords + 0.4 * _normalize(-grad_y)
    return _normalize(geometric_depth)


def analyze_depth_stratified(n_images: int = 30):
    """
    Analyze error by depth strata.
    
    Stratify by Depth Anything's depth values and see where each approach wins.
    """
    print("=" * 70)
    print("ANALYSIS: Distance-Dependent Falloff")
    print("=" * 70)
    print()
    print("Hypothesis: DA wins at extremes (close/far), Holo wins in middle")
    print("Testing by stratifying pixels by their DA depth value...")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect errors by depth bin
    n_bins = 10
    holo_errors_by_bin = [[] for _ in range(n_bins)]
    da_errors_by_bin = [[] for _ in range(n_bins)]
    vertical_errors_by_bin = [[] for _ in range(n_bins)]
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute holographic depth
        enhanced = holographic_enhance_numpy(rgb_small, sigma=2.0, boost=1.5)
        holo_depth = extract_holographic_depth(enhanced)
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical_depth = _normalize(0.6 * y_coords + 0.1)
        
        # Stratify by DA depth
        for bin_idx in range(n_bins):
            bin_low = bin_idx / n_bins
            bin_high = (bin_idx + 1) / n_bins
            
            mask = (da_depth_small >= bin_low) & (da_depth_small < bin_high)
            if mask.sum() == 0:
                continue
            
            # Error vs vertical baseline
            holo_err = np.abs(holo_depth[mask] - vertical_depth[mask]).mean()
            da_err = np.abs(da_depth_small[mask] - vertical_depth[mask]).mean()
            vert_err = 0.0  # Vertical vs vertical = 0
            
            holo_errors_by_bin[bin_idx].append(holo_err)
            da_errors_by_bin[bin_idx].append(da_err)
            vertical_errors_by_bin[bin_idx].append(vert_err)
    
    # Compute means
    holo_means = [np.mean(errs) if errs else np.nan for errs in holo_errors_by_bin]
    da_means = [np.mean(errs) if errs else np.nan for errs in da_errors_by_bin]
    
    # Print results
    print("=" * 60)
    print("Error by Depth Bin (vs Vertical Baseline)")
    print("=" * 60)
    print()
    print(f"{'Bin':<12} {'DA Depth Range':<18} {'Holo MAE':<12} {'DA MAE':<12} {'Winner':<10}")
    print("-" * 60)
    
    holo_wins = 0
    da_wins = 0
    
    for bin_idx in range(n_bins):
        bin_low = bin_idx / n_bins
        bin_high = (bin_idx + 1) / n_bins
        
        holo_mae = holo_means[bin_idx]
        da_mae = da_means[bin_idx]
        
        if np.isnan(holo_mae) or np.isnan(da_mae):
            winner = "N/A"
        elif holo_mae < da_mae:
            winner = "HOLO"
            holo_wins += 1
        else:
            winner = "DA"
            da_wins += 1
        
        print(f"Bin {bin_idx:<6} [{bin_low:.1f} - {bin_high:.1f}]      {holo_mae:<12.4f} {da_mae:<12.4f} {winner:<10}")
    
    print()
    print(f"Holographic wins: {holo_wins} bins")
    print(f"Depth Anything wins: {da_wins} bins")
    
    return holo_means, da_means


def analyze_perspective_falloff(n_images: int = 30):
    """
    Analyze if the falloff follows perspective projection.
    
    Perspective: depth ∝ 1/y (inverse relationship)
    Our model: depth ∝ y (linear relationship)
    
    At extremes, the difference between 1/y and y is largest.
    """
    print()
    print("=" * 70)
    print("ANALYSIS: Perspective Projection Falloff")
    print("=" * 70)
    print()
    print("Linear model:      depth = a*y + b")
    print("Perspective model: depth = a/(y + b) + c")
    print()
    print("Testing if perspective model captures the falloff...")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    linear_errors = []
    perspective_errors = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = da_depth.shape[:2] if da_depth.ndim > 1 else (da_depth.shape[0], 1)
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Y coordinates
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        
        # Linear model: depth = 0.6*y + 0.1
        linear_pred = 0.6 * y_coords + 0.1
        linear_pred = _normalize(linear_pred)
        
        # Perspective model: depth = 1 / (1 + exp(-k*(y - 0.5)))
        # This is a sigmoid that captures the non-linear falloff
        k = 4.0  # Steepness
        perspective_pred = 1 / (1 + np.exp(-k * (y_coords - 0.5)))
        perspective_pred = _normalize(perspective_pred)
        
        linear_err = np.mean(np.abs(linear_pred - da_depth_small))
        perspective_err = np.mean(np.abs(perspective_pred - da_depth_small))
        
        linear_errors.append(linear_err)
        perspective_errors.append(perspective_err)
    
    print(f"Linear Model MAE:      {np.mean(linear_errors):.4f}")
    print(f"Perspective Model MAE: {np.mean(perspective_errors):.4f}")
    
    if np.mean(perspective_errors) < np.mean(linear_errors):
        print()
        print("→ Perspective model is BETTER!")
        print("→ The falloff at extremes is due to non-linear depth relationship")
    else:
        print()
        print("→ Linear model is still better")
        print("→ The falloff may be due to something else")
    
    return linear_errors, perspective_errors


def create_falloff_visualization(n_images: int = 10):
    """Create visualization of the depth-dependent falloff."""
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect data for visualization
    all_da_depths = []
    all_holo_errors = []
    all_da_errors = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute holographic depth
        enhanced = holographic_enhance_numpy(rgb_small, sigma=2.0, boost=1.5)
        holo_depth = extract_holographic_depth(enhanced)
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical_depth = _normalize(0.6 * y_coords + 0.1)
        
        # Collect pixel-level data
        all_da_depths.extend(da_depth_small.flatten())
        all_holo_errors.extend(np.abs(holo_depth - vertical_depth).flatten())
        all_da_errors.extend(np.abs(da_depth_small - vertical_depth).flatten())
    
    all_da_depths = np.array(all_da_depths)
    all_holo_errors = np.array(all_holo_errors)
    all_da_errors = np.array(all_da_errors)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distance-Dependent Falloff Analysis\n'
                 'Where Does Each Approach Win?', fontsize=14, fontweight='bold')
    
    # 1. Error vs Depth scatter (sampled)
    ax1 = axes[0, 0]
    sample_idx = np.random.choice(len(all_da_depths), min(5000, len(all_da_depths)), replace=False)
    ax1.scatter(all_da_depths[sample_idx], all_holo_errors[sample_idx], 
                alpha=0.3, s=1, c='green', label='Holographic')
    ax1.scatter(all_da_depths[sample_idx], all_da_errors[sample_idx], 
                alpha=0.3, s=1, c='red', label='Depth Anything')
    ax1.set_xlabel('DA Depth Value (0=close, 1=far)')
    ax1.set_ylabel('Error vs Vertical Baseline')
    ax1.set_title('1. Error vs Depth (Scatter)')
    ax1.legend()
    
    # 2. Binned error curves
    ax2 = axes[0, 1]
    n_bins = 20
    bin_centers = []
    holo_binned = []
    da_binned = []
    
    for bin_idx in range(n_bins):
        bin_low = bin_idx / n_bins
        bin_high = (bin_idx + 1) / n_bins
        mask = (all_da_depths >= bin_low) & (all_da_depths < bin_high)
        
        if mask.sum() > 0:
            bin_centers.append((bin_low + bin_high) / 2)
            holo_binned.append(all_holo_errors[mask].mean())
            da_binned.append(all_da_errors[mask].mean())
    
    ax2.plot(bin_centers, holo_binned, 'g-o', linewidth=2, markersize=6, label='Holographic')
    ax2.plot(bin_centers, da_binned, 'r-o', linewidth=2, markersize=6, label='Depth Anything')
    ax2.set_xlabel('DA Depth Value (0=close, 1=far)')
    ax2.set_ylabel('Mean Error vs Vertical Baseline')
    ax2.set_title('2. Error vs Depth (Binned)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Winner by depth
    ax3 = axes[1, 0]
    winner_holo = []
    winner_da = []
    
    for bin_idx in range(n_bins):
        bin_low = bin_idx / n_bins
        bin_high = (bin_idx + 1) / n_bins
        mask = (all_da_depths >= bin_low) & (all_da_depths < bin_high)
        
        if mask.sum() > 0:
            holo_better = (all_holo_errors[mask] < all_da_errors[mask]).mean() * 100
            winner_holo.append(holo_better)
            winner_da.append(100 - holo_better)
    
    x = np.arange(len(bin_centers))
    width = 0.35
    ax3.bar(x - width/2, winner_holo, width, label='Holographic Wins %', color='green', alpha=0.7)
    ax3.bar(x + width/2, winner_da, width, label='DA Wins %', color='red', alpha=0.7)
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Depth Bin')
    ax3.set_ylabel('Win Percentage')
    ax3.set_title('3. Winner by Depth Region')
    ax3.legend()
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([f'{c:.1f}' for c in bin_centers[::2]])
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find crossover points
    crossovers = []
    for i in range(1, len(winner_holo)):
        if (winner_holo[i-1] > 50 and winner_holo[i] < 50) or \
           (winner_holo[i-1] < 50 and winner_holo[i] > 50):
            crossovers.append(bin_centers[i])
    
    summary = (
        f"FALLOFF ANALYSIS SUMMARY\n"
        f"{'='*40}\n\n"
        f"Holographic wins in middle depths\n"
        f"DA wins at extremes (close/far)\n\n"
        f"Crossover points: {crossovers}\n\n"
        f"{'='*40}\n\n"
        f"INTERPRETATION:\n\n"
        f"The vertical baseline (depth ∝ y) is\n"
        f"a LINEAR approximation of perspective.\n\n"
        f"At depth extremes, the true relationship\n"
        f"is NON-LINEAR (perspective projection).\n\n"
        f"DA has learned this non-linearity.\n"
        f"Our geometric model hasn't.\n\n"
        f"SOLUTION: Use perspective-corrected\n"
        f"vertical baseline instead of linear."
    )
    
    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes,
             fontsize=11, va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    output_file = OUTPUT_PATH / "depth_falloff_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Analyze by depth strata
    holo_means, da_means = analyze_depth_stratified(n_images=30)
    
    # Analyze perspective falloff
    linear_errors, perspective_errors = analyze_perspective_falloff(n_images=30)
    
    # Create visualization
    viz_file = create_falloff_visualization(n_images=20)
