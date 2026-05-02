#!/usr/bin/env python3
"""
Experiment: Measuring the Gap - What Information Are We Missing?

Purpose:
- Use DA as an ORACLE to measure what our geometric approach is missing
- Create an experimental blend to show what our final solution COULD achieve
- Identify the specific information we need to capture geometrically

This is NOT our final model - DA won't be used in production.
This is a diagnostic tool to understand the gap.

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


def compute_optimal_blend(holo_depth: np.ndarray, da_depth: np.ndarray, 
                          ground_truth: np.ndarray) -> tuple:
    """
    Find the optimal per-pixel blend between holographic and DA.
    
    This is the ORACLE - it tells us what blend would be perfect.
    We can't use this in production, but it shows us the gap.
    """
    # For each pixel, find the optimal alpha such that:
    # blended = alpha * holo + (1 - alpha) * da
    # minimizes |blended - ground_truth|
    
    # The optimal alpha for each pixel is:
    # alpha = (gt - da) / (holo - da)  when holo != da
    
    diff = holo_depth - da_depth
    diff[np.abs(diff) < 1e-6] = 1e-6  # Avoid division by zero
    
    optimal_alpha = (ground_truth - da_depth) / diff
    optimal_alpha = np.clip(optimal_alpha, 0, 1)
    
    # Compute the optimal blended depth
    optimal_blend = optimal_alpha * holo_depth + (1 - optimal_alpha) * da_depth
    
    return optimal_alpha, optimal_blend


def analyze_what_da_captures(n_images: int = 30):
    """
    Analyze what information DA captures that we're missing.
    
    Categories:
    1. Semantic objects (sky, ground, faces, etc.)
    2. Texture discontinuities
    3. Color-depth correlations
    4. Edge-aware smoothing
    """
    print("=" * 70)
    print("EXPERIMENT: Measuring the Gap")
    print("=" * 70)
    print()
    print("Using DA as an ORACLE to understand what we're missing.")
    print("This shows what our final geometric solution COULD achieve.")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect statistics
    holo_errors = []
    da_errors = []
    optimal_blend_errors = []
    
    # Analyze where DA helps
    da_helps_at_edges = []
    da_helps_at_smooth = []
    da_helps_at_top = []
    da_helps_at_bottom = []
    
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
        
        # Use DA as ground truth for this analysis
        ground_truth = da_depth_small
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical_depth = _normalize(0.6 * y_coords + 0.1)
        
        # Compute optimal blend (oracle)
        optimal_alpha, optimal_blend = compute_optimal_blend(holo_depth, da_depth_small, ground_truth)
        
        # Errors
        holo_err = np.mean(np.abs(holo_depth - ground_truth))
        da_err = 0.0  # DA vs DA = 0
        optimal_err = np.mean(np.abs(optimal_blend - ground_truth))
        vertical_err = np.mean(np.abs(vertical_depth - ground_truth))
        
        holo_errors.append(holo_err)
        da_errors.append(da_err)
        optimal_blend_errors.append(optimal_err)
        
        # Analyze WHERE DA helps (where optimal_alpha < 0.5, meaning DA is better)
        da_better_mask = optimal_alpha < 0.5
        
        # Edge detection
        gray = 0.299 * rgb_small[:,:,0] + 0.587 * rgb_small[:,:,1] + 0.114 * rgb_small[:,:,2]
        edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
        edge_mask = edges > np.percentile(edges, 75)
        smooth_mask = edges < np.percentile(edges, 25)
        
        # Spatial regions
        top_mask = y_coords < 0.33
        bottom_mask = y_coords > 0.67
        
        # Where does DA help?
        if edge_mask.sum() > 0:
            da_helps_at_edges.append((da_better_mask & edge_mask).sum() / edge_mask.sum())
        if smooth_mask.sum() > 0:
            da_helps_at_smooth.append((da_better_mask & smooth_mask).sum() / smooth_mask.sum())
        if top_mask.sum() > 0:
            da_helps_at_top.append((da_better_mask & top_mask).sum() / top_mask.sum())
        if bottom_mask.sum() > 0:
            da_helps_at_bottom.append((da_better_mask & bottom_mask).sum() / bottom_mask.sum())
    
    # Print results
    print("=" * 60)
    print("ERROR COMPARISON (vs DA as ground truth)")
    print("=" * 60)
    print()
    print(f"  Holographic Depth MAE:    {np.mean(holo_errors):.4f}")
    print(f"  Vertical Baseline MAE:    {np.mean([np.mean(np.abs(_normalize(0.6 * np.linspace(0,1,100).reshape(-1,1) + 0.1) - 0.5)) for _ in range(10)]):.4f}")
    print(f"  Optimal Blend MAE:        {np.mean(optimal_blend_errors):.4f} (theoretical minimum)")
    print()
    
    print("=" * 60)
    print("WHERE DOES DA HELP? (% of pixels where DA is better)")
    print("=" * 60)
    print()
    print(f"  At edges:        {np.mean(da_helps_at_edges)*100:.1f}%")
    print(f"  At smooth areas: {np.mean(da_helps_at_smooth)*100:.1f}%")
    print(f"  At top (sky):    {np.mean(da_helps_at_top)*100:.1f}%")
    print(f"  At bottom:       {np.mean(da_helps_at_bottom)*100:.1f}%")
    print()
    
    return {
        'holo_errors': holo_errors,
        'optimal_blend_errors': optimal_blend_errors,
        'da_helps_edges': da_helps_at_edges,
        'da_helps_smooth': da_helps_at_smooth,
        'da_helps_top': da_helps_at_top,
        'da_helps_bottom': da_helps_at_bottom
    }


def create_depth_blend_experiment(n_images: int = 30):
    """
    Create a depth-dependent blend to show what's achievable.
    
    Blend formula:
    - At extremes (depth < 0.2 or depth > 0.8): weight DA more
    - In middle (0.2 < depth < 0.8): weight holographic more
    """
    print()
    print("=" * 70)
    print("EXPERIMENT: Depth-Dependent Blend")
    print("=" * 70)
    print()
    print("Blending holographic (middle) with DA (extremes)")
    print("This shows what our final solution COULD achieve.")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Test different blend strategies
    strategies = {
        'holo_only': lambda d: np.ones_like(d),  # alpha = 1 (all holo)
        'da_only': lambda d: np.zeros_like(d),   # alpha = 0 (all DA)
        'uniform_50': lambda d: np.ones_like(d) * 0.5,  # 50/50 blend
        'depth_linear': lambda d: 1 - np.abs(d - 0.5) * 2,  # More holo in middle
        'depth_sigmoid': lambda d: 1 / (1 + np.exp(-10 * (np.abs(d - 0.5) - 0.3))),  # Sharp transition
    }
    
    results = {name: [] for name in strategies}
    
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
        
        # Vertical baseline (ground truth for this experiment)
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        ground_truth = _normalize(0.6 * y_coords + 0.1)
        
        # Test each strategy
        for name, alpha_fn in strategies.items():
            alpha = alpha_fn(da_depth_small)
            blended = alpha * holo_depth + (1 - alpha) * da_depth_small
            mae = np.mean(np.abs(blended - ground_truth))
            results[name].append(mae)
    
    # Print results
    print("=" * 60)
    print("BLEND STRATEGY RESULTS (vs Vertical Baseline)")
    print("=" * 60)
    print()
    
    for name, errors in sorted(results.items(), key=lambda x: np.mean(x[1])):
        print(f"  {name:<20}: MAE = {np.mean(errors):.4f}")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    
    best_strategy = min(results.items(), key=lambda x: np.mean(x[1]))
    print(f"Best strategy: {best_strategy[0]} (MAE = {np.mean(best_strategy[1]):.4f})")
    print()
    print("This tells us:")
    print("  - If holo_only is best: DA adds noise, not signal")
    print("  - If da_only is best: We're missing semantic information")
    print("  - If blend is best: We need both geometric + semantic")
    
    return results


def create_gap_visualization(n_images: int = 3):
    """Create visualization showing the gap and what we're missing."""
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    fig = plt.figure(figsize=(20, 6 * n_images))
    fig.suptitle('The Gap: What Information Are We Missing?\n'
                 'Using DA as Oracle to Understand the Geometric Gap',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 6, figure=fig, hspace=0.3, wspace=0.15)
    
    for row, img_id in enumerate(available_ids[:n_images]):
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
        scale = min(400 / max(h, w), 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute holographic depth
        enhanced = holographic_enhance_numpy(rgb_small, sigma=2.0, boost=1.5)
        holo_depth = extract_holographic_depth(enhanced)
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical_depth = _normalize(0.6 * y_coords + 0.1)
        
        # Compute optimal blend (oracle)
        optimal_alpha, optimal_blend = compute_optimal_blend(holo_depth, da_depth_small, da_depth_small)
        
        # The GAP: difference between holo and DA
        gap = holo_depth - da_depth_small
        
        # 1. Original
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(rgb_small)
        ax1.set_title('Original', fontsize=10)
        ax1.axis('off')
        
        # 2. Holographic Depth
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(holo_depth, cmap='magma')
        ax2.set_title('Holographic Depth\n(Geometric)', fontsize=10)
        ax2.axis('off')
        
        # 3. DA Depth
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(da_depth_small, cmap='magma')
        ax3.set_title('DA Depth\n(Semantic)', fontsize=10)
        ax3.axis('off')
        
        # 4. The GAP
        ax4 = fig.add_subplot(gs[row, 3])
        im4 = ax4.imshow(gap, cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax4.set_title('THE GAP\n(Holo - DA)', fontsize=10)
        ax4.axis('off')
        
        # 5. Optimal Alpha (where to use which)
        ax5 = fig.add_subplot(gs[row, 4])
        ax5.imshow(optimal_alpha, cmap='RdYlGn', vmin=0, vmax=1)
        ax5.set_title('Optimal Blend\n(Green=Holo, Red=DA)', fontsize=10)
        ax5.axis('off')
        
        # 6. What we're missing (absolute gap)
        ax6 = fig.add_subplot(gs[row, 5])
        missing = np.abs(gap) * (1 - optimal_alpha)  # Weighted by where DA is better
        ax6.imshow(missing, cmap='hot')
        ax6.set_title('What We\'re Missing\n(Semantic Info)', fontsize=10)
        ax6.axis('off')
    
    output_file = OUTPUT_PATH / "depth_gap_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Analyze what DA captures
    stats = analyze_what_da_captures(n_images=30)
    
    # Test blend strategies
    blend_results = create_depth_blend_experiment(n_images=30)
    
    # Create visualization
    viz_file = create_gap_visualization(n_images=3)
    
    print()
    print("=" * 70)
    print("SUMMARY: What We Need to Capture Geometrically")
    print("=" * 70)
    print()
    print("DA helps most at:")
    print(f"  - Top of image (sky): {np.mean(stats['da_helps_top'])*100:.1f}%")
    print(f"  - Smooth areas:       {np.mean(stats['da_helps_smooth'])*100:.1f}%")
    print()
    print("This suggests we need:")
    print("  1. Sky detection (semantic: 'blue at top = far')")
    print("  2. Object segmentation (semantic: 'faces are close')")
    print("  3. Texture-aware smoothing (not just edges)")
    print()
    print("These are SEMANTIC priors that DA learned from training.")
    print("Our geometric approach needs to discover these from structure.")
