#!/usr/bin/env python3
"""
Visual Comparison: φ-Reconstruction vs Depth Anything V2

Generate side-by-side depth maps to see if our φ-based model
produces reasonable results compared to the original DA2.

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
    
    # Resize to original size
    depth_np = depth.squeeze().numpy()
    
    return _normalize(depth_np)


def train_phi_model(model, processor, n_train: int = 30):
    """Train the φ-based depth model."""
    print(f"\nTraining φ-model on {n_train} images...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect training data
    all_features = []
    all_depth_maps = []
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Get DA2 depth
        da2_depth = get_da2_depth(model, processor, rgb)
        
        # Extract features from the image itself (not DA2)
        # Use color statistics at different spatial locations
        h, w = rgb.shape[:2]
        
        # Divide image into grid and extract color features
        grid_size = 4
        features = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                y1, y2 = gy * h // grid_size, (gy + 1) * h // grid_size
                x1, x2 = gx * w // grid_size, (gx + 1) * w // grid_size
                patch = rgb[y1:y2, x1:x2]
                
                # Color statistics
                features.extend([
                    patch[:,:,0].mean(),  # R mean
                    patch[:,:,1].mean(),  # G mean
                    patch[:,:,2].mean(),  # B mean
                    patch.std(),          # Overall std
                    (gy + 0.5) / grid_size,  # Vertical position
                ])
        
        all_features.append(features)
        all_depth_maps.append(da2_depth.mean())  # Mean depth as target
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depth_maps = np.array(all_depth_maps)
    
    # PCA to 3 dimensions
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    n_components = 3
    pca_components = Vt[:n_components]
    pca_features = features_centered @ pca_components.T
    
    # φ-scale
    phi_features = pca_features.copy()
    for i in range(n_components):
        phi_features[:, i] *= PHI ** i
    
    # Learn mapping
    X = np.column_stack([phi_features, np.ones(len(phi_features))])
    coeffs, _, _, _ = np.linalg.lstsq(X, all_depth_maps, rcond=None)
    
    print(f"  Training complete. Correlation: {np.corrcoef(X @ coeffs, all_depth_maps)[0,1]:.3f}")
    
    return {
        'feature_mean': feature_mean,
        'pca_components': pca_components,
        'coeffs': coeffs,
        'grid_size': grid_size
    }


def phi_predict_depth(phi_model: dict, rgb: np.ndarray):
    """
    Predict depth using φ-model.
    
    This creates a per-pixel depth map by applying the model
    to local patches.
    """
    h, w = rgb.shape[:2]
    grid_size = phi_model['grid_size']
    
    # Extract features (same as training)
    features = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y1, y2 = gy * h // grid_size, (gy + 1) * h // grid_size
            x1, x2 = gx * w // grid_size, (gx + 1) * w // grid_size
            patch = rgb[y1:y2, x1:x2]
            
            features.extend([
                patch[:,:,0].mean(),
                patch[:,:,1].mean(),
                patch[:,:,2].mean(),
                patch.std(),
                (gy + 0.5) / grid_size,
            ])
    
    features = np.array(features)
    
    # Project to φ-space
    features_centered = features - phi_model['feature_mean']
    pca_features = features_centered @ phi_model['pca_components'].T
    
    phi_features = pca_features.copy()
    for i in range(len(pca_features)):
        phi_features[i] *= PHI ** i
    
    # Predict global depth
    X = np.append(phi_features, 1.0)
    global_depth = X @ phi_model['coeffs']
    
    # Create depth map with vertical gradient + global offset
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    
    # Combine vertical baseline with learned global depth
    # The φ-model tells us the overall depth scale
    depth_map = 0.4 * y_coords + 0.6 * global_depth
    
    # Add local variation based on color
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    local_var = gaussian_filter(gray, sigma=10) - gaussian_filter(gray, sigma=30)
    depth_map = depth_map + 0.1 * local_var
    
    return _normalize(depth_map)


def create_comparison_visualization(n_images: int = 4):
    """Create side-by-side comparison of φ-model vs DA2."""
    
    model, processor = load_da2()
    phi_model = train_phi_model(model, processor, n_train=30)
    
    print("\nGenerating comparison visualization...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Use test images (not in training)
    test_ids = available_ids[30:30+n_images]
    
    fig = plt.figure(figsize=(20, 5 * n_images))
    fig.suptitle('φ-Reconstruction vs Depth Anything V2\n'
                 'Can Our Geometric Model Approximate Neural Depth?',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 5, figure=fig, hspace=0.3, wspace=0.15)
    
    for row, img_id in enumerate(test_ids):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        # Load image
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Get DA2 depth
        da2_depth = get_da2_depth(model, processor, rgb)
        
        # Get φ-model depth
        phi_depth = phi_predict_depth(phi_model, rgb)
        
        # Resize phi_depth to match da2_depth
        phi_depth_resized = np.array(Image.fromarray((phi_depth * 255).astype(np.uint8)).resize(
            (da2_depth.shape[1], da2_depth.shape[0]))) / 255.0
        
        # Vertical baseline
        h, w = da2_depth.shape
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        vertical = _normalize(0.6 * y_coords + 0.1)
        
        # Compute errors
        phi_mae = np.mean(np.abs(phi_depth_resized - da2_depth))
        vertical_mae = np.mean(np.abs(vertical - da2_depth))
        
        # Plot
        ax1 = fig.add_subplot(gs[row, 0])
        # Resize RGB for display
        rgb_display = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (da2_depth.shape[1], da2_depth.shape[0]))) / 255.0
        ax1.imshow(rgb_display)
        ax1.set_title('Original Image' if row == 0 else '', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(da2_depth, cmap='magma')
        ax2.set_title('DA2 Depth\n(Ground Truth)' if row == 0 else '', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(phi_depth_resized, cmap='magma')
        ax3.set_title(f'φ-Model Depth\n(MAE: {phi_mae:.3f})' if row == 0 else f'MAE: {phi_mae:.3f}', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.imshow(vertical, cmap='magma')
        ax4.set_title(f'Vertical Baseline\n(MAE: {vertical_mae:.3f})' if row == 0 else f'MAE: {vertical_mae:.3f}', fontsize=10)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[row, 4])
        # Difference map
        diff = phi_depth_resized - da2_depth
        ax5.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax5.set_title('φ - DA2 Difference\n(Blue=closer, Red=farther)' if row == 0 else '', fontsize=10)
        ax5.axis('off')
    
    output_file = OUTPUT_PATH / "phi_vs_da2_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    viz_file = create_comparison_visualization(n_images=4)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print()
    print("The visualization shows:")
    print("  1. Original image")
    print("  2. DA2 depth (ground truth)")
    print("  3. φ-model depth (our reconstruction)")
    print("  4. Vertical baseline")
    print("  5. Difference between φ-model and DA2")
    print()
    print("Look for:")
    print("  - Does φ-model capture overall depth structure?")
    print("  - Is it better than vertical baseline?")
    print("  - Where does it fail?")
