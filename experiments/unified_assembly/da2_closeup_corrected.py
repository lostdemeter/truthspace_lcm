#!/usr/bin/env python3
"""
Close-up Corrected Transcoder for φ-DA2

We discovered that DA2 uses specific dimensions to detect and handle close-ups:
- Dimensions 73, 162, 54: Flip sign for close-ups (negative→positive)
- Dimension 138: Spikes to ~2.0 for close-ups (close-up detector)

This experiment:
1. Uses these discriminative dimensions to detect close-up images
2. Applies different geometric strategy based on detection
3. Weights the transcoder differently for close-ups

Goal: Fix the outlier images (banana, food bowl) while maintaining
good performance on structured scenes.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.ndimage import zoom, sobel, gaussian_filter
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")

# Discriminative dimensions from our analysis
CLOSEUP_DETECTOR_DIMS = [73, 162, 54, 192, 239]
CLOSEUP_SPIKE_DIM = 138  # This dimension spikes to ~2.0 for close-ups


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


def extract_da2_structure(model, processor, rgb: np.ndarray):
    """Extract DA2's backbone structure."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1]
        
        full_output = model(inputs['pixel_values'])
        da2_depth = full_output.predicted_depth.squeeze().numpy()
    
    return structure.squeeze().numpy(), _normalize(da2_depth)


def detect_closeup(structure: np.ndarray) -> tuple:
    """
    Detect if image is a close-up using discriminative dimensions.
    
    Returns:
    - is_closeup: bool
    - closeup_score: float (0-1, higher = more close-up like)
    """
    # Skip CLS token, get mean over patches
    if len(structure.shape) == 2:
        features = structure[1:].mean(axis=0)
    else:
        features = structure.mean()
    
    # Check discriminative dimensions
    # For close-ups, these dimensions tend to be positive (they flip from negative)
    detector_values = features[CLOSEUP_DETECTOR_DIMS]
    
    # Close-up score based on how positive these dimensions are
    # Good images have negative values, close-ups have positive
    closeup_score = np.mean(detector_values > 0)
    
    # Also check the spike dimension
    spike_value = features[CLOSEUP_SPIKE_DIM]
    spike_score = min(spike_value / 2.0, 1.0) if spike_value > 0 else 0
    
    # Combined score
    combined_score = 0.6 * closeup_score + 0.4 * spike_score
    
    is_closeup = combined_score > 0.5
    
    return is_closeup, combined_score


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Edge detection."""
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    return _normalize(np.sqrt(grad_x**2 + grad_y**2))


def extract_multiscale_edges(gray: np.ndarray, n_scales: int = 3) -> list:
    """Multi-scale edges."""
    edge_maps = []
    for i in range(n_scales):
        sigma = PHI ** i
        smoothed = gaussian_filter(gray, sigma=sigma)
        grad_x = sobel(smoothed, axis=1)
        grad_y = sobel(smoothed, axis=0)
        edge_maps.append(_normalize(np.sqrt(grad_x**2 + grad_y**2)))
    return edge_maps


def extract_closeup_features(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Extract features specifically useful for close-up images.
    
    For close-ups, depth often correlates with:
    - Distance from center (objects in center are closer)
    - Color saturation (saturated = closer, in focus)
    - Local texture (textured = closer)
    """
    h, w = gray.shape
    
    # Distance from center (inverted: center = close = high value)
    y, x = np.mgrid[0:h, 0:w]
    y_norm = (y - h/2) / (h/2)
    x_norm = (x - w/2) / (w/2)
    center_dist = np.sqrt(y_norm**2 + x_norm**2)
    center_proximity = 1.0 - _normalize(center_dist)
    
    # Color saturation
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
    saturation = _normalize(saturation)
    
    # Local texture (variance in small window)
    local_var = gaussian_filter(gray**2, sigma=3) - gaussian_filter(gray, sigma=3)**2
    texture = _normalize(np.sqrt(np.maximum(local_var, 0)))
    
    # Luminance gradient (for close-ups, often uniform)
    lum_grad = np.abs(np.gradient(gray, axis=0)) + np.abs(np.gradient(gray, axis=1))
    lum_grad = _normalize(lum_grad)
    
    return np.stack([center_proximity, saturation, texture, lum_grad], axis=-1)


def learn_adaptive_transcoder(model, processor, n_train: int = 30):
    """
    Learn a transcoder that adapts based on close-up detection.
    
    Uses discriminative dimensions to weight features differently.
    """
    print("\n" + "=" * 70)
    print("LEARNING CLOSE-UP ADAPTIVE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    all_closeup_scores = []
    
    pixels_per_image = 400
    n_edge_scales = 3
    
    print(f"\nCollecting samples from {n_train} images...")
    print(f"  Using discriminative dimensions: {CLOSEUP_DETECTOR_DIMS}")
    print(f"  Spike dimension: {CLOSEUP_SPIKE_DIM}")
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Get DA2 structure
        structure, da2_depth = extract_da2_structure(model, processor, rgb)
        
        # Detect close-up
        is_closeup, closeup_score = detect_closeup(structure)
        
        # Skip CLS token
        structure = structure[1:]
        N, C = structure.shape
        
        # Get spatial dimensions
        depth_h, depth_w = da2_depth.shape
        patch_size = 14
        H_s = depth_h // patch_size
        W_s = depth_w // patch_size
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        # Resize
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        gray_small = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Extract features
        edge_maps = extract_multiscale_edges(gray_small, n_edge_scales)
        closeup_feats = extract_closeup_features(rgb_small, gray_small)
        
        # Adaptive vertical gradient (reduced for close-ups)
        vertical = np.linspace(0, 1, H_s).reshape(-1, 1)
        vertical = np.tile(vertical, (1, W_s))
        vertical_weight = 1.0 - closeup_score  # Reduce for close-ups
        vertical_weighted = vertical * vertical_weight
        
        # Sample positions
        np.random.seed(i)
        for _ in range(pixels_per_image):
            y = np.random.randint(0, H_s)
            x = np.random.randint(0, W_s)
            
            # DA2 structure (384-dim)
            da2_feat = struct_spatial[y, x]
            
            # Discriminative dimensions (explicitly include them with higher weight)
            disc_feat = da2_feat[CLOSEUP_DETECTOR_DIMS] * PHI  # Boost discriminative dims
            spike_feat = np.array([da2_feat[CLOSEUP_SPIKE_DIM] * PHI])
            
            # Multi-scale edges (3-dim)
            edge_feat = np.array([
                edge_maps[0][y, x] * PHI**0,
                edge_maps[1][y, x] * PHI**0.5,
                edge_maps[2][y, x] * PHI**1,
            ])
            
            # Close-up specific features (4-dim)
            closeup_feat = closeup_feats[y, x] * closeup_score  # Weight by closeup score
            
            # Adaptive vertical (1-dim)
            vert_feat = np.array([vertical_weighted[y, x]])
            
            # Closeup score as feature (1-dim)
            score_feat = np.array([closeup_score])
            
            # Concatenate all
            combined = np.concatenate([
                da2_feat,           # 384
                disc_feat,          # 5 (discriminative dims, boosted)
                spike_feat,         # 1 (spike dim, boosted)
                edge_feat,          # 3
                closeup_feat,       # 4
                vert_feat,          # 1
                score_feat,         # 1
            ])
            
            all_features.append(combined)
            all_depths.append(depth_small[y, x])
            all_closeup_scores.append(closeup_score)
        
        if (i + 1) % 5 == 0:
            n_closeup = sum(1 for s in all_closeup_scores[-pixels_per_image:] if s > 0.5)
            print(f"  Processed {i+1}/{n_train} (last batch: {n_closeup}/{pixels_per_image} close-up)")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    all_closeup_scores = np.array(all_closeup_scores)
    
    print(f"\n  Collected {len(all_features)} samples")
    print(f"  Feature dim: {all_features.shape[1]}")
    print(f"  Close-up samples: {(all_closeup_scores > 0.5).sum()} ({(all_closeup_scores > 0.5).mean()*100:.1f}%)")
    
    # PCA
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    n_components = 50
    pca_components = Vt[:n_components]
    pca_features = features_centered @ pca_components.T
    
    cumvar = np.cumsum(S[:n_components]**2) / (S**2).sum()
    print(f"  Top {n_components} components explain {cumvar[-1]*100:.1f}% variance")
    
    # Learn linear mapping
    X = np.column_stack([pca_features, np.ones(len(pca_features))])
    weights, _, _, _ = np.linalg.lstsq(X, all_depths, rcond=None)
    
    # Evaluate
    pred = X @ weights
    mae = np.mean(np.abs(pred - all_depths))
    corr = np.corrcoef(pred, all_depths)[0, 1]
    
    print(f"\n  Training MAE: {mae:.4f}")
    print(f"  Training Correlation: {corr:.4f}")
    
    return {
        'feature_mean': feature_mean,
        'pca_components': pca_components,
        'weights': weights,
        'n_components': n_components,
        'n_edge_scales': n_edge_scales,
        'mae': mae,
        'corr': corr
    }


def test_adaptive_transcoder(model, processor, transcoder: dict, n_test: int = 12):
    """Test the adaptive transcoder."""
    print("\n" + "=" * 70)
    print("TESTING CLOSE-UP ADAPTIVE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Include known outliers in test set
    test_ids = available_ids[30:30+n_test]
    
    # Add our known outliers
    outlier_ids = ["000000002587", "000000003501"]  # Banana, food bowl
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    n_edge_scales = transcoder['n_edge_scales']
    
    results = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Get DA2 structure
        structure, da2_depth = extract_da2_structure(model, processor, rgb)
        
        # Detect close-up
        is_closeup, closeup_score = detect_closeup(structure)
        
        structure = structure[1:]
        N, C = structure.shape
        
        depth_h, depth_w = da2_depth.shape
        patch_size = 14
        H_s = depth_h // patch_size
        W_s = depth_w // patch_size
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        # Resize
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        gray_small = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Extract features
        edge_maps = extract_multiscale_edges(gray_small, n_edge_scales)
        closeup_feats = extract_closeup_features(rgb_small, gray_small)
        
        # Adaptive vertical
        vertical = np.linspace(0, 1, H_s).reshape(-1, 1)
        vertical = np.tile(vertical, (1, W_s))
        vertical_weight = 1.0 - closeup_score
        vertical_weighted = vertical * vertical_weight
        
        # Build feature maps
        disc_map = struct_spatial[:, :, CLOSEUP_DETECTOR_DIMS] * PHI
        spike_map = struct_spatial[:, :, CLOSEUP_SPIKE_DIM:CLOSEUP_SPIKE_DIM+1] * PHI
        
        edge_stack = np.stack([
            edge_maps[0] * PHI**0,
            edge_maps[1] * PHI**0.5,
            edge_maps[2] * PHI**1,
        ], axis=-1)
        
        closeup_feat_weighted = closeup_feats * closeup_score
        vert_stack = vertical_weighted[:, :, np.newaxis]
        score_stack = np.full((H_s, W_s, 1), closeup_score)
        
        # Combine
        combined = np.concatenate([
            struct_spatial,
            disc_map,
            spike_map,
            edge_stack,
            closeup_feat_weighted,
            vert_stack,
            score_stack,
        ], axis=-1)
        
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Apply transcoder
        features_centered = combined_flat - transcoder['feature_mean']
        pca_features = features_centered @ transcoder['pca_components'].T
        
        X = np.column_stack([pca_features, np.ones(len(pca_features))])
        pred_flat = X @ transcoder['weights']
        
        pred_depth = pred_flat.reshape(H_s, W_s)
        pred_depth = _normalize(pred_depth)
        
        # Upscale
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        pred_resized = zoom(pred_depth, (zoom_h, zoom_w), order=3)
        pred_resized = _normalize(pred_resized)
        
        # Metrics
        mae = np.mean(np.abs(pred_resized - da2_depth))
        corr = np.corrcoef(pred_resized.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_known_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'pred_depth': pred_resized,
            'mae': mae,
            'corr': corr,
            'closeup_score': closeup_score,
            'is_closeup': is_closeup,
            'is_known_outlier': is_known_outlier
        })
        
        marker = " [OUTLIER]" if is_known_outlier else ""
        closeup_marker = " (close-up)" if is_closeup else ""
        print(f"  {img_id}: MAE={mae:.3f}, Corr={corr:.3f}, closeup={closeup_score:.2f}{closeup_marker}{marker}")
    
    return results


def create_visualization(results: list, transcoder: dict):
    """Visualize results with close-up detection info."""
    
    # Sort: known outliers first, then by closeup score
    results_sorted = sorted(results, key=lambda x: (-x['is_known_outlier'], -x['closeup_score']))
    
    n_images = min(len(results_sorted), 12)
    
    fig = plt.figure(figsize=(18, 3.5 * n_images + 1))
    fig.suptitle('Close-up Adaptive Transcoder\n'
                 f'Using discriminative dims {CLOSEUP_DETECTOR_DIMS} + spike dim {CLOSEUP_SPIKE_DIM}',
                 fontsize=14, fontweight='bold', y=0.99)
    
    gs = gridspec.GridSpec(n_images + 1, 5, figure=fig, hspace=0.2, wspace=0.1,
                          height_ratios=[1] * n_images + [0.25])
    
    for row, r in enumerate(results_sorted[:n_images]):
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(r['rgb'])
        title = 'Original' if row == 0 else ''
        if r['is_known_outlier']:
            title = f"OUTLIER: {r['img_id'][-4:]}"
        ax1.set_title(title, fontsize=9, color='red' if r['is_known_outlier'] else 'black')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(r['da2_depth'], cmap='magma')
        ax2.set_title('DA2 Depth' if row == 0 else '', fontsize=9)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(r['pred_depth'], cmap='magma')
        title = f'Adaptive (Corr: {r["corr"]:.3f})' if row == 0 else f'Corr: {r["corr"]:.3f}'
        ax3.set_title(title, fontsize=9)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        diff = r['pred_depth'] - r['da2_depth']
        ax4.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax4.set_title('Difference' if row == 0 else '', fontsize=9)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[row, 4])
        ax5.axis('off')
        info = f"Close-up: {r['closeup_score']:.2f}\n{'YES' if r['is_closeup'] else 'NO'}"
        color = 'orange' if r['is_closeup'] else 'lightblue'
        ax5.text(0.5, 0.5, info, transform=ax5.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color))
    
    # Summary
    ax_summary = fig.add_subplot(gs[n_images, :])
    ax_summary.axis('off')
    
    avg_mae = np.mean([r['mae'] for r in results])
    avg_corr = np.mean([r['corr'] for r in results])
    
    outlier_results = [r for r in results if r['is_known_outlier']]
    if outlier_results:
        outlier_corr = np.mean([r['corr'] for r in outlier_results])
    else:
        outlier_corr = 0
    
    baseline_outlier_corr = 0.32  # Previous average for outliers
    improvement = (outlier_corr - baseline_outlier_corr) / abs(baseline_outlier_corr) * 100 if baseline_outlier_corr != 0 else 0
    
    summary = f"""
    CLOSE-UP ADAPTIVE TRANSCODER: Uses discriminative dimensions to detect and adapt
    Overall: MAE={avg_mae:.4f}, Corr={avg_corr:.4f} | Outliers: Corr={outlier_corr:.4f} (vs baseline {baseline_outlier_corr:.2f}, {improvement:+.0f}%)
    """
    color = 'lightgreen' if improvement > 0 else 'lightyellow'
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color))
    
    output_file = OUTPUT_PATH / "da2_closeup_corrected.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Learn adaptive transcoder
    transcoder = learn_adaptive_transcoder(model, processor, n_train=30)
    
    # Test
    results = test_adaptive_transcoder(model, processor, transcoder, n_test=10)
    
    # Visualize
    viz_file = create_visualization(results, transcoder)
    
    # Summary
    avg_corr = np.mean([r['corr'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    outlier_results = [r for r in results if r['is_known_outlier']]
    if outlier_results:
        outlier_corr = np.mean([r['corr'] for r in outlier_results])
        print(f"\n  OUTLIER PERFORMANCE: {outlier_corr:.4f}")
    
    print("\n" + "=" * 70)
    print("CLOSE-UP ADAPTIVE RESULTS")
    print("=" * 70)
    print(f"\n  Average MAE: {avg_mae:.4f}")
    print(f"  Average Correlation: {avg_corr:.4f}")
    print()
    print("  Key innovations:")
    print(f"    - Close-up detection using dims {CLOSEUP_DETECTOR_DIMS}")
    print(f"    - Spike dimension {CLOSEUP_SPIKE_DIM} for close-up intensity")
    print("    - Adaptive vertical gradient (reduced for close-ups)")
    print("    - Close-up specific features (center proximity, saturation, texture)")
