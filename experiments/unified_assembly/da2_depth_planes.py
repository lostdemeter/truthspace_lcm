#!/usr/bin/env python3
"""
Depth Plane Segmentation for φ-DA2

Key insight from user: DA2 slices depth planes like peering through fog.
Each "layer" of fog is a depth plane where similar things cluster together.

This is fundamentally different from vertical gradient:
- Vertical gradient: y-position → depth (fails on close-ups)
- Depth planes: color/texture similarity → same depth layer

We implement:
1. Color clustering in φ-space to find depth planes
2. Adaptive vertical prior (detect when it applies)
3. Reduced reliance on vertical gradient
4. Plane-based depth assignment

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.ndimage import zoom, sobel, gaussian_filter, label
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
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


# =============================================================================
# DEPTH PLANE EXTRACTION
# =============================================================================

def extract_color_planes(rgb: np.ndarray, n_planes: int = 5) -> tuple:
    """
    Segment image into color-based depth planes.
    
    Similar colors often belong to the same object/depth layer.
    Returns plane labels and plane features.
    """
    h, w, c = rgb.shape
    
    # Flatten for clustering
    pixels = rgb.reshape(-1, 3)
    
    # Add spatial information (weighted less than color)
    y_coords = np.repeat(np.arange(h), w).reshape(-1, 1) / h
    x_coords = np.tile(np.arange(w), h).reshape(-1, 1) / w
    
    # Combine: color (weight 1) + position (weight 0.3)
    features = np.hstack([
        pixels,
        y_coords * 0.3,
        x_coords * 0.3
    ])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_planes, random_state=42, n_init=3)
    labels = kmeans.fit_predict(features)
    
    # Reshape to image
    plane_map = labels.reshape(h, w)
    
    # Get mean color and position for each plane
    plane_features = []
    for i in range(n_planes):
        mask = (plane_map == i)
        if mask.sum() > 0:
            mean_color = rgb[mask].mean(axis=0)
            mean_y = y_coords.reshape(h, w)[mask].mean()
            mean_x = x_coords.reshape(h, w)[mask].mean()
            area = mask.sum() / (h * w)
            plane_features.append([*mean_color, mean_y, mean_x, area])
        else:
            plane_features.append([0, 0, 0, 0.5, 0.5, 0])
    
    return plane_map, np.array(plane_features)


def extract_plane_depth_features(rgb: np.ndarray, plane_map: np.ndarray, 
                                  plane_features: np.ndarray) -> np.ndarray:
    """
    Extract per-pixel features based on which depth plane it belongs to.
    
    Each pixel gets:
    - Its plane's mean color (3)
    - Its plane's mean position (2)
    - Its plane's area (1)
    - Distance from plane center (1)
    """
    h, w, _ = rgb.shape
    n_planes = len(plane_features)
    
    # Create feature map
    features = np.zeros((h, w, 7))
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_norm = y_grid / h
    x_norm = x_grid / w
    
    for i in range(n_planes):
        mask = (plane_map == i)
        if mask.sum() > 0:
            pf = plane_features[i]
            features[mask, 0:3] = pf[0:3]  # Mean color
            features[mask, 3] = pf[3]       # Mean y
            features[mask, 4] = pf[4]       # Mean x
            features[mask, 5] = pf[5]       # Area
            
            # Distance from plane center
            dist = np.sqrt((y_norm - pf[3])**2 + (x_norm - pf[4])**2)
            features[mask, 6] = dist[mask]
    
    return features


def detect_vertical_scene(gray: np.ndarray, edge_map: np.ndarray) -> float:
    """
    Detect if the scene follows vertical depth structure.
    
    Returns a weight [0, 1] for how much to trust vertical gradient.
    - Outdoor landscapes with sky: high weight
    - Close-up objects: low weight
    - Indoor scenes: medium weight
    """
    h, w = gray.shape
    
    # Check if top is brighter (sky-like)
    top_brightness = gray[:h//4].mean()
    bottom_brightness = gray[3*h//4:].mean()
    brightness_gradient = top_brightness - bottom_brightness
    
    # Check edge distribution
    top_edges = edge_map[:h//3].mean()
    bottom_edges = edge_map[2*h//3:].mean()
    edge_ratio = bottom_edges / (top_edges + 1e-6)
    
    # Check if there's a clear horizon (horizontal edge band)
    horizontal_profile = edge_map.mean(axis=1)
    horizon_strength = np.max(horizontal_profile) / (np.mean(horizontal_profile) + 1e-6)
    
    # Combine signals
    # High brightness gradient + more edges at bottom + horizon = outdoor scene
    vertical_score = 0.0
    
    if brightness_gradient > 0.1:  # Top brighter than bottom
        vertical_score += 0.3
    
    if edge_ratio > 1.5:  # More edges at bottom
        vertical_score += 0.3
    
    if horizon_strength > 2.0:  # Clear horizon
        vertical_score += 0.4
    
    return min(vertical_score, 1.0)


def extract_adaptive_vertical(gray: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    """
    Adaptive vertical gradient that's weighted by scene type.
    """
    h, w = gray.shape
    
    # Detect scene type
    vertical_weight = detect_vertical_scene(gray, edge_map)
    
    # Base vertical gradient
    gradient = np.linspace(0, 1, h).reshape(-1, 1)
    gradient = np.tile(gradient, (1, w))
    
    # Weight by scene type
    return gradient * vertical_weight


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Edge detection."""
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    return _normalize(np.sqrt(grad_x**2 + grad_y**2))


def extract_multiscale_edges(gray: np.ndarray, n_scales: int = 3) -> list:
    """Multi-scale edges at φ-scaled sigmas."""
    edge_maps = []
    for i in range(n_scales):
        sigma = PHI ** i
        smoothed = gaussian_filter(gray, sigma=sigma)
        grad_x = sobel(smoothed, axis=1)
        grad_y = sobel(smoothed, axis=0)
        edge_maps.append(_normalize(np.sqrt(grad_x**2 + grad_y**2)))
    return edge_maps


# =============================================================================
# DA2 STRUCTURE
# =============================================================================

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


def learn_plane_transcoder(model, processor, n_train: int = 25):
    """
    Learn transcoder with depth plane features.
    """
    print("\n" + "=" * 70)
    print("LEARNING DEPTH PLANE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    pixels_per_image = 400
    n_planes = 5
    n_edge_scales = 3
    
    print(f"\nCollecting samples from {n_train} images...")
    print(f"  Depth planes: {n_planes}")
    print(f"  Edge scales: {n_edge_scales}")
    print(f"  Adaptive vertical gradient: enabled")
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Get DA2 structure
        structure, da2_depth = extract_da2_structure(model, processor, rgb)
        
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
        
        # Resize to patch size
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        gray_small = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Extract depth plane features
        plane_map, plane_features = extract_color_planes(rgb_small, n_planes)
        plane_feat_map = extract_plane_depth_features(rgb_small, plane_map, plane_features)
        
        # Extract edge features
        edge_maps = extract_multiscale_edges(gray_small, n_edge_scales)
        
        # Adaptive vertical gradient
        base_edges = extract_edges(gray_small)
        adaptive_vertical = extract_adaptive_vertical(gray_small, base_edges)
        
        # Sample positions
        np.random.seed(i)
        for _ in range(pixels_per_image):
            y = np.random.randint(0, H_s)
            x = np.random.randint(0, W_s)
            
            # DA2 structure
            da2_feat = struct_spatial[y, x]  # 384-dim
            
            # Plane features (7-dim)
            plane_feat = plane_feat_map[y, x]
            
            # Multi-scale edges with φ-weights (3-dim)
            edge_feat = np.array([
                edge_maps[0][y, x] * PHI**0,
                edge_maps[1][y, x] * PHI**0.5,
                edge_maps[2][y, x] * PHI**1,
            ])
            
            # Adaptive vertical (1-dim) - reduced weight
            vert_feat = np.array([adaptive_vertical[y, x] * PHI**(-0.5)])  # Reduced weight!
            
            # Concatenate
            combined = np.concatenate([da2_feat, plane_feat, edge_feat, vert_feat])
            
            all_features.append(combined)
            all_depths.append(depth_small[y, x])
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    
    n_geo = 7 + n_edge_scales + 1
    print(f"\n  Collected {len(all_features)} samples")
    print(f"  Feature dim: {all_features.shape[1]} (384 DA2 + {n_geo} geometric)")
    
    # PCA
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    n_components = 40
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
        'n_planes': n_planes,
        'n_edge_scales': n_edge_scales,
        'mae': mae,
        'corr': corr
    }


def test_plane_transcoder(model, processor, transcoder: dict, n_test: int = 10):
    """Test the plane-based transcoder."""
    print("\n" + "=" * 70)
    print("TESTING DEPTH PLANE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[30:30+n_test]
    
    n_planes = transcoder['n_planes']
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
        plane_map, plane_features = extract_color_planes(rgb_small, n_planes)
        plane_feat_map = extract_plane_depth_features(rgb_small, plane_map, plane_features)
        
        edge_maps = extract_multiscale_edges(gray_small, n_edge_scales)
        base_edges = extract_edges(gray_small)
        adaptive_vertical = extract_adaptive_vertical(gray_small, base_edges)
        
        # Stack features
        edge_stack = np.stack([
            edge_maps[0] * PHI**0,
            edge_maps[1] * PHI**0.5,
            edge_maps[2] * PHI**1,
        ], axis=-1)
        
        vert_stack = adaptive_vertical[:, :, np.newaxis] * PHI**(-0.5)
        
        # Combine
        combined = np.concatenate([struct_spatial, plane_feat_map, edge_stack, vert_stack], axis=-1)
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
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'pred_depth': pred_resized,
            'mae': mae,
            'corr': corr
        })
        
        print(f"  {img_id}: MAE={mae:.3f}, Corr={corr:.3f}")
    
    return results


def create_visualization(results: list, transcoder: dict):
    """Visualize results."""
    
    n_images = len(results)
    
    fig = plt.figure(figsize=(16, 3.5 * n_images + 1))
    fig.suptitle('Depth Plane Transcoder\n'
                 f'Color planes: {transcoder["n_planes"]} | Adaptive vertical | Reduced gradient weight',
                 fontsize=14, fontweight='bold', y=0.99)
    
    gs = gridspec.GridSpec(n_images + 1, 4, figure=fig, hspace=0.2, wspace=0.1,
                          height_ratios=[1] * n_images + [0.25])
    
    for row, r in enumerate(results):
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(r['rgb'])
        ax1.set_title('Original' if row == 0 else '', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(r['da2_depth'], cmap='magma')
        ax2.set_title('DA2 Depth' if row == 0 else '', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(r['pred_depth'], cmap='magma')
        title = f'Plane-Based (Corr: {r["corr"]:.3f})' if row == 0 else f'Corr: {r["corr"]:.3f}'
        ax3.set_title(title, fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        diff = r['pred_depth'] - r['da2_depth']
        ax4.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax4.set_title('Difference' if row == 0 else '', fontsize=10)
        ax4.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[n_images, :])
    ax_summary.axis('off')
    
    avg_mae = np.mean([r['mae'] for r in results])
    avg_corr = np.mean([r['corr'] for r in results])
    
    baseline_corr = 0.816
    improvement = (avg_corr - baseline_corr) / baseline_corr * 100
    
    summary = f"""
    DEPTH PLANE TRANSCODER: Color clustering + Adaptive vertical + Reduced gradient weight
    Test MAE: {avg_mae:.4f}  |  Test Corr: {avg_corr:.4f}  |  vs Multi-Scale Baseline: {improvement:+.1f}%
    """
    color = 'lightgreen' if improvement > 0 else 'lightyellow'
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color))
    
    output_file = OUTPUT_PATH / "da2_depth_planes.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Learn plane-based transcoder
    transcoder = learn_plane_transcoder(model, processor, n_train=25)
    
    # Test
    results = test_plane_transcoder(model, processor, transcoder, n_test=10)
    
    # Visualize
    viz_file = create_visualization(results, transcoder)
    
    # Summary
    avg_corr = np.mean([r['corr'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    print("\n" + "=" * 70)
    print("DEPTH PLANE RESULTS")
    print("=" * 70)
    print(f"\n  Average MAE: {avg_mae:.4f}")
    print(f"  Average Correlation: {avg_corr:.4f}")
    print()
    print("  Key changes:")
    print("    - Color clustering for depth planes (5 planes)")
    print("    - Adaptive vertical detection (sky/ground vs close-up)")
    print("    - Reduced vertical gradient weight (φ^-0.5)")
    print("    - Plane features: mean color, position, area, distance")
