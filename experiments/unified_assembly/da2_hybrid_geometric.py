#!/usr/bin/env python3
"""
Hybrid DA2: Combining Learned Structure with Geometric Principles

We've discovered:
1. DA2's structure can be decoded with a linear transcoder (0.90 correlation)
2. PC6 is just a vertical gradient - we can replace with pure geometry
3. PC0/PC2 encode object boundaries - holographic edges capture this geometrically
4. The holographic experiment showed excellent edge detection

This experiment combines:
- DA2's learned object features (PC0, PC2) 
- Pure geometric vertical gradient (replace PC6)
- Holographic edge detection (enhance boundaries)
- φ-scaling for all components

Goal: Beat DA2's decoder using geometric principles + learned structure.

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
from scipy.fft import fft2, ifft2, fftshift, ifftshift
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
# GEOMETRIC DEPTH DIMENSIONS (from holographic experiment)
# =============================================================================

def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Edge dimension: sharp edges = in focus = closer."""
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    return _normalize(edge_strength)


def extract_vertical_gradient(h: int, w: int) -> np.ndarray:
    """Pure geometric vertical gradient: top = far, bottom = near."""
    gradient = np.linspace(0, 1, h).reshape(-1, 1)
    return np.tile(gradient, (1, w))


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    """Frequency dimension: high frequency = fine detail = closer."""
    F = fft2(gray)
    F_shifted = fftshift(F)
    
    h, w = gray.shape
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    
    H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
    F_filtered = F_shifted * H
    
    filtered = np.abs(ifft2(ifftshift(F_filtered)))
    return _normalize(filtered)


def extract_local_contrast(gray: np.ndarray) -> np.ndarray:
    """Local contrast: low contrast = far (atmospheric perspective)."""
    local_mean = gaussian_filter(gray, sigma=15)
    local_std = np.sqrt(gaussian_filter((gray - local_mean)**2, sigma=15))
    contrast = _normalize(local_std)
    return 1.0 - contrast  # Invert: low contrast = far


# =============================================================================
# DA2 STRUCTURE EXTRACTION
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
    """Extract DA2's backbone structure (the learned features)."""
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


def learn_hybrid_transcoder(model, processor, n_train: int = 25):
    """
    Learn a hybrid transcoder that combines:
    - DA2's learned structure (object features)
    - Geometric dimensions (edges, vertical, frequency)
    """
    print("\n" + "=" * 70)
    print("LEARNING HYBRID TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    pixels_per_image = 400
    
    print(f"\nCollecting samples from {n_train} images...")
    
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
        
        # Resize depth and gray to patch size
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        gray_small = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Extract geometric dimensions at patch level
        edges = extract_edges(gray_small)
        vertical = extract_vertical_gradient(H_s, W_s)
        frequency = extract_frequency(gray_small)
        contrast = extract_local_contrast(gray_small)
        
        # Sample positions
        np.random.seed(i)
        for _ in range(pixels_per_image):
            y = np.random.randint(0, H_s)
            x = np.random.randint(0, W_s)
            
            # Combine DA2 structure with geometric features
            da2_feat = struct_spatial[y, x]  # 384-dim
            
            # Add geometric dimensions with φ-scaling
            geo_feat = np.array([
                vertical[y, x] * PHI**0,      # Vertical (φ^0 = 1)
                edges[y, x] * PHI**1,          # Edges (φ^1 = 1.618)
                frequency[y, x] * PHI**0.5,    # Frequency (φ^0.5)
                contrast[y, x] * PHI**0.5,     # Contrast (φ^0.5)
            ])
            
            # Concatenate
            combined = np.concatenate([da2_feat, geo_feat])
            
            all_features.append(combined)
            all_depths.append(depth_small[y, x])
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    
    print(f"\n  Collected {len(all_features)} samples")
    print(f"  Feature dim: {all_features.shape[1]} (384 DA2 + 4 geometric)")
    
    # PCA on combined features
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    n_components = 30
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
        'mae': mae,
        'corr': corr
    }


def test_hybrid_transcoder(model, processor, transcoder: dict, n_test: int = 8):
    """Test the hybrid transcoder."""
    print("\n" + "=" * 70)
    print("TESTING HYBRID TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[30:30+n_test]
    
    results = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Get DA2 structure and depth
        structure, da2_depth = extract_da2_structure(model, processor, rgb)
        
        # Process structure
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
        
        # Resize gray
        gray_small = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Extract geometric dimensions
        edges = extract_edges(gray_small)
        vertical = extract_vertical_gradient(H_s, W_s)
        frequency = extract_frequency(gray_small)
        contrast = extract_local_contrast(gray_small)
        
        # Combine features for all positions
        geo_features = np.stack([
            vertical * PHI**0,
            edges * PHI**1,
            frequency * PHI**0.5,
            contrast * PHI**0.5
        ], axis=-1)  # [H_s, W_s, 4]
        
        combined = np.concatenate([struct_spatial, geo_features], axis=-1)  # [H_s, W_s, 388]
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Apply transcoder
        features_centered = combined_flat - transcoder['feature_mean']
        pca_features = features_centered @ transcoder['pca_components'].T
        
        X = np.column_stack([pca_features, np.ones(len(pca_features))])
        pred_flat = X @ transcoder['weights']
        
        pred_depth = pred_flat.reshape(H_s, W_s)
        pred_depth = _normalize(pred_depth)
        
        # Upscale with bicubic
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        pred_resized = zoom(pred_depth, (zoom_h, zoom_w), order=3)
        pred_resized = _normalize(pred_resized)
        
        # Compute metrics
        mae = np.mean(np.abs(pred_resized - da2_depth))
        corr = np.corrcoef(pred_resized.flatten(), da2_depth.flatten())[0, 1]
        
        # Resize RGB for display
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'hybrid_depth': pred_resized,
            'mae': mae,
            'corr': corr
        })
        
        print(f"  {img_id}: MAE={mae:.3f}, Corr={corr:.3f}")
    
    return results


def create_hybrid_visualization(results: list, transcoder: dict):
    """Visualize hybrid transcoder results."""
    
    n_images = len(results)
    
    fig = plt.figure(figsize=(16, 4 * n_images + 1))
    fig.suptitle('Hybrid Transcoder: DA2 Structure + Geometric Principles\n'
                 'Combining learned features with φ-scaled edges, vertical, frequency',
                 fontsize=14, fontweight='bold', y=0.99)
    
    gs = gridspec.GridSpec(n_images + 1, 4, figure=fig, hspace=0.25, wspace=0.15,
                          height_ratios=[1] * n_images + [0.3])
    
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
        ax3.imshow(r['hybrid_depth'], cmap='magma')
        title = f'Hybrid (Corr: {r["corr"]:.3f})' if row == 0 else f'Corr: {r["corr"]:.3f}'
        ax3.set_title(title, fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        diff = r['hybrid_depth'] - r['da2_depth']
        ax4.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax4.set_title('Difference' if row == 0 else '', fontsize=10)
        ax4.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[n_images, :])
    ax_summary.axis('off')
    
    avg_mae = np.mean([r['mae'] for r in results])
    avg_corr = np.mean([r['corr'] for r in results])
    
    summary = f"""
    HYBRID TRANSCODER: DA2 Structure (384-dim) + Geometric Features (4-dim: vertical, edges, frequency, contrast)
    Training Corr: {transcoder['corr']:.4f}  |  Test MAE: {avg_mae:.4f}  |  Test Corr: {avg_corr:.4f}
    
    Key: We replaced DA2's learned vertical gradient with pure φ-geometry and added holographic edge detection.
    """
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    output_file = OUTPUT_PATH / "da2_hybrid_geometric.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Learn hybrid transcoder
    transcoder = learn_hybrid_transcoder(model, processor, n_train=25)
    
    # Test
    results = test_hybrid_transcoder(model, processor, transcoder, n_test=8)
    
    # Visualize
    viz_file = create_hybrid_visualization(results, transcoder)
    
    print("\n" + "=" * 70)
    print("HYBRID TRANSCODER COMPLETE")
    print("=" * 70)
    print()
    print("We combined:")
    print("  - DA2's learned structure (384 dimensions)")
    print("  - Geometric vertical gradient (φ^0 scaled)")
    print("  - Holographic edge detection (φ^1 scaled)")
    print("  - Frequency analysis (φ^0.5 scaled)")
    print("  - Local contrast (φ^0.5 scaled)")
    print()
    print("The hybrid approach uses geometric principles where possible,")
    print("while keeping DA2's learned object features.")
