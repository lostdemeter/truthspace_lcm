#!/usr/bin/env python3
"""
Pure φ-Reconstruction: No Vertical Gradient Bias

The previous model was contaminated by explicit vertical gradient.
Let's remove that and see what pure φ-features can capture.

Key changes:
1. No explicit vertical gradient in output
2. Use per-pixel color features
3. Let the φ-structure emerge naturally

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
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
    
    print("Loading Depth Anything V2...")
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    
    return model, processor


def get_da2_depth(model, processor, image: np.ndarray):
    """Get depth map from DA2."""
    import torch
    
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs['pixel_values'])
        depth = outputs.predicted_depth
    
    depth_np = depth.squeeze().numpy()
    return _normalize(depth_np)


def extract_phi_features_per_pixel(rgb: np.ndarray, scale: int = 8):
    """
    Extract φ-scaled features at each pixel location.
    
    Uses multi-scale color analysis with φ-scaling.
    """
    h, w = rgb.shape[:2]
    
    # Multi-scale features at φ-scaled radii
    scales = [int(scale * PHI**i) for i in range(4)]  # φ^0, φ^1, φ^2, φ^3
    
    features = []
    
    for s in scales:
        if s < 1:
            s = 1
        # Gaussian blur at this scale
        blurred = np.stack([
            gaussian_filter(rgb[:,:,c], sigma=s) for c in range(3)
        ], axis=-1)
        features.append(blurred)
    
    # Stack all scales
    all_features = np.concatenate(features, axis=-1)  # [H, W, 12]
    
    # Add local contrast (difference between scales)
    contrast = []
    for i in range(len(features) - 1):
        diff = features[i] - features[i+1]
        contrast.append(diff)
    
    if contrast:
        contrast_features = np.concatenate(contrast, axis=-1)
        all_features = np.concatenate([all_features, contrast_features], axis=-1)
    
    return all_features


def train_phi_pixel_model(model, processor, n_train: int = 30):
    """
    Train a per-pixel φ-model.
    
    Learn mapping from φ-scaled color features to depth.
    """
    print(f"\nTraining pure φ-model on {n_train} images...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect training data (sample pixels)
    all_features = []
    all_depths = []
    
    pixels_per_image = 500  # Sample pixels per image
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Get DA2 depth
        da2_depth = get_da2_depth(model, processor, rgb)
        
        # Resize RGB to match depth
        h_d, w_d = da2_depth.shape
        rgb_resized = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w_d, h_d))) / 255.0
        
        # Extract φ-features
        phi_features = extract_phi_features_per_pixel(rgb_resized, scale=4)
        
        # Sample random pixels
        np.random.seed(i)
        y_samples = np.random.randint(0, h_d, pixels_per_image)
        x_samples = np.random.randint(0, w_d, pixels_per_image)
        
        for y, x in zip(y_samples, x_samples):
            all_features.append(phi_features[y, x])
            all_depths.append(da2_depth[y, x])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    
    print(f"  Collected {len(all_features)} pixel samples")
    print(f"  Feature dimension: {all_features.shape[1]}")
    
    # PCA to reduce dimensionality
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    # Keep components that explain 95% variance
    cumvar = np.cumsum(S**2) / (S**2).sum()
    n_components = np.searchsorted(cumvar, 0.95) + 1
    n_components = min(n_components, 10)  # Cap at 10
    
    print(f"  Using {n_components} PCA components (95% variance)")
    
    pca_components = Vt[:n_components]
    pca_features = features_centered @ pca_components.T
    
    # φ-scale the components
    phi_features = pca_features.copy()
    for j in range(n_components):
        phi_features[:, j] *= PHI ** (j / 2)  # Gentler scaling
    
    # Learn linear mapping
    X = np.column_stack([phi_features, np.ones(len(phi_features))])
    coeffs, _, _, _ = np.linalg.lstsq(X, all_depths, rcond=None)
    
    # Evaluate
    pred = X @ coeffs
    mae = np.mean(np.abs(pred - all_depths))
    corr = np.corrcoef(pred, all_depths)[0, 1]
    
    print(f"  Training MAE: {mae:.4f}")
    print(f"  Training Correlation: {corr:.4f}")
    
    return {
        'feature_mean': feature_mean,
        'pca_components': pca_components,
        'coeffs': coeffs,
        'n_components': n_components,
        'scale': 4
    }


def phi_predict_depth_pure(phi_model: dict, rgb: np.ndarray):
    """
    Predict depth using pure φ-features (no vertical gradient).
    """
    h, w = rgb.shape[:2]
    
    # Extract φ-features
    phi_features = extract_phi_features_per_pixel(rgb, scale=phi_model['scale'])
    
    # Flatten for processing
    features_flat = phi_features.reshape(-1, phi_features.shape[-1])
    
    # Project to PCA space
    features_centered = features_flat - phi_model['feature_mean']
    pca_features = features_centered @ phi_model['pca_components'].T
    
    # φ-scale
    n_components = phi_model['n_components']
    for j in range(n_components):
        pca_features[:, j] *= PHI ** (j / 2)
    
    # Predict
    X = np.column_stack([pca_features, np.ones(len(pca_features))])
    depth_flat = X @ phi_model['coeffs']
    
    # Reshape
    depth_map = depth_flat.reshape(h, w)
    
    return _normalize(depth_map)


def create_pure_comparison(n_images: int = 4):
    """Create comparison with pure φ-model (no vertical gradient)."""
    
    model, processor = load_da2()
    phi_model = train_phi_pixel_model(model, processor, n_train=30)
    
    print("\nGenerating pure φ-model comparison...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[30:30+n_images]
    
    fig = plt.figure(figsize=(16, 4 * n_images))
    fig.suptitle('Pure φ-Reconstruction (No Vertical Gradient)\n'
                 'Letting φ-Features Emerge Naturally',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 4, figure=fig, hspace=0.3, wspace=0.15)
    
    total_phi_mae = []
    
    for row, img_id in enumerate(test_ids):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Get DA2 depth
        da2_depth = get_da2_depth(model, processor, rgb)
        
        # Resize RGB to match
        h_d, w_d = da2_depth.shape
        rgb_resized = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w_d, h_d))) / 255.0
        
        # Get pure φ-model depth
        phi_depth = phi_predict_depth_pure(phi_model, rgb_resized)
        
        # Compute error
        phi_mae = np.mean(np.abs(phi_depth - da2_depth))
        total_phi_mae.append(phi_mae)
        
        # Plot
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(rgb_resized)
        ax1.set_title('Original Image' if row == 0 else '', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(da2_depth, cmap='magma')
        ax2.set_title('DA2 Depth' if row == 0 else '', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(phi_depth, cmap='magma')
        ax3.set_title(f'Pure φ-Model\n(MAE: {phi_mae:.3f})' if row == 0 else f'MAE: {phi_mae:.3f}', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        diff = phi_depth - da2_depth
        ax4.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax4.set_title('φ - DA2 Difference' if row == 0 else '', fontsize=10)
        ax4.axis('off')
    
    output_file = OUTPUT_PATH / "phi_pure_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    print(f"Average φ-model MAE: {np.mean(total_phi_mae):.4f}")
    
    return output_file


if __name__ == "__main__":
    viz_file = create_pure_comparison(n_images=4)
    
    print("\n" + "=" * 70)
    print("PURE φ-MODEL COMPARISON COMPLETE")
    print("=" * 70)
    print()
    print("This version uses:")
    print("  - Multi-scale φ-features (φ^0, φ^1, φ^2, φ^3 radii)")
    print("  - Per-pixel prediction")
    print("  - NO explicit vertical gradient")
    print()
    print("Look for object boundaries emerging from pure color structure!")
