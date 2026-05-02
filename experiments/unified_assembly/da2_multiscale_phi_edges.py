#!/usr/bin/env python3
"""
Multi-Scale φ-Edge Detection for Improved DA2

Enhancement over the basic hybrid transcoder:
- Edge detection at multiple φ-scaled scales
- Each scale captures different depth information:
  - Fine edges (φ⁰) → close objects, fine detail
  - Medium edges (φ¹) → object boundaries
  - Coarse edges (φ²) → large structures, far objects
  - Very coarse (φ³) → scene-level structure

The φ-scaling ensures self-similar behavior across scales.

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
# MULTI-SCALE φ-EDGE DETECTION
# =============================================================================

def extract_multiscale_phi_edges(gray: np.ndarray, n_scales: int = 4) -> list:
    """
    Extract edges at multiple φ-scaled scales.
    
    Scale 0: σ = 1 (fine detail, close objects)
    Scale 1: σ = φ (medium detail)
    Scale 2: σ = φ² (coarse detail)
    Scale 3: σ = φ³ (very coarse, scene structure)
    
    Returns list of edge maps, one per scale.
    """
    edge_maps = []
    
    for i in range(n_scales):
        sigma = PHI ** i
        
        # Smooth at this scale
        smoothed = gaussian_filter(gray, sigma=sigma)
        
        # Compute edges
        grad_x = sobel(smoothed, axis=1)
        grad_y = sobel(smoothed, axis=0)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        edge_maps.append(_normalize(edge_strength))
    
    return edge_maps


def extract_multiscale_frequency(gray: np.ndarray, n_scales: int = 3) -> list:
    """
    Extract frequency content at multiple φ-scaled bands.
    
    Band 0: High frequency (fine texture, close)
    Band 1: Medium frequency
    Band 2: Low frequency (coarse structure, far)
    """
    F = fft2(gray)
    F_shifted = fftshift(F)
    
    h, w = gray.shape
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)
    max_r = np.sqrt((w//2)**2 + (h//2)**2)
    
    freq_maps = []
    
    for i in range(n_scales):
        # φ-scaled frequency bands
        r_low = max_r / (PHI ** (i + 1))
        r_high = max_r / (PHI ** i) if i > 0 else max_r
        
        # Bandpass filter
        H = ((R >= r_low) & (R < r_high)).astype(float)
        H = gaussian_filter(H, sigma=2)  # Smooth edges
        
        F_filtered = F_shifted * H
        filtered = np.abs(ifft2(ifftshift(F_filtered)))
        
        freq_maps.append(_normalize(filtered))
    
    return freq_maps


def extract_vertical_gradient(h: int, w: int) -> np.ndarray:
    """Pure geometric vertical gradient."""
    gradient = np.linspace(0, 1, h).reshape(-1, 1)
    return np.tile(gradient, (1, w))


def extract_local_contrast(gray: np.ndarray) -> np.ndarray:
    """Local contrast: low contrast = far."""
    local_mean = gaussian_filter(gray, sigma=15)
    local_std = np.sqrt(gaussian_filter((gray - local_mean)**2, sigma=15))
    contrast = _normalize(local_std)
    return 1.0 - contrast


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


def learn_multiscale_transcoder(model, processor, n_train: int = 25):
    """
    Learn transcoder with multi-scale φ-edge features.
    """
    print("\n" + "=" * 70)
    print("LEARNING MULTI-SCALE φ-EDGE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    pixels_per_image = 400
    n_edge_scales = 4
    n_freq_scales = 3
    
    print(f"\nCollecting samples from {n_train} images...")
    print(f"  Edge scales: {n_edge_scales} (σ = φ⁰, φ¹, φ², φ³)")
    print(f"  Frequency scales: {n_freq_scales}")
    
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
        
        # Extract multi-scale geometric features
        edge_maps = extract_multiscale_phi_edges(gray_small, n_edge_scales)
        freq_maps = extract_multiscale_frequency(gray_small, n_freq_scales)
        vertical = extract_vertical_gradient(H_s, W_s)
        contrast = extract_local_contrast(gray_small)
        
        # Sample positions
        np.random.seed(i)
        for _ in range(pixels_per_image):
            y = np.random.randint(0, H_s)
            x = np.random.randint(0, W_s)
            
            # DA2 structure features
            da2_feat = struct_spatial[y, x]  # 384-dim
            
            # Multi-scale edge features with φ-weights
            edge_feat = np.array([
                edge_maps[0][y, x] * PHI**0,    # Fine edges (close)
                edge_maps[1][y, x] * PHI**0.5,  # Medium edges
                edge_maps[2][y, x] * PHI**1,    # Coarse edges
                edge_maps[3][y, x] * PHI**1.5,  # Very coarse (far)
            ])
            
            # Multi-scale frequency features
            freq_feat = np.array([
                freq_maps[0][y, x] * PHI**0,    # High freq (close)
                freq_maps[1][y, x] * PHI**0.5,  # Medium freq
                freq_maps[2][y, x] * PHI**1,    # Low freq (far)
            ])
            
            # Other geometric features
            other_feat = np.array([
                vertical[y, x] * PHI**0,
                contrast[y, x] * PHI**0.5,
            ])
            
            # Concatenate all
            combined = np.concatenate([da2_feat, edge_feat, freq_feat, other_feat])
            
            all_features.append(combined)
            all_depths.append(depth_small[y, x])
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    
    n_geo_features = n_edge_scales + n_freq_scales + 2
    print(f"\n  Collected {len(all_features)} samples")
    print(f"  Feature dim: {all_features.shape[1]} (384 DA2 + {n_geo_features} geometric)")
    
    # PCA on combined features
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    n_components = 35  # Slightly more to capture multi-scale info
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
        'n_freq_scales': n_freq_scales,
        'mae': mae,
        'corr': corr
    }


def test_multiscale_transcoder(model, processor, transcoder: dict, n_test: int = 10):
    """Test the multi-scale transcoder."""
    print("\n" + "=" * 70)
    print("TESTING MULTI-SCALE φ-EDGE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[30:30+n_test]
    
    n_edge_scales = transcoder['n_edge_scales']
    n_freq_scales = transcoder['n_freq_scales']
    
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
        
        # Extract multi-scale features
        edge_maps = extract_multiscale_phi_edges(gray_small, n_edge_scales)
        freq_maps = extract_multiscale_frequency(gray_small, n_freq_scales)
        vertical = extract_vertical_gradient(H_s, W_s)
        contrast = extract_local_contrast(gray_small)
        
        # Stack edge features with φ-weights
        edge_stack = np.stack([
            edge_maps[0] * PHI**0,
            edge_maps[1] * PHI**0.5,
            edge_maps[2] * PHI**1,
            edge_maps[3] * PHI**1.5,
        ], axis=-1)
        
        # Stack frequency features
        freq_stack = np.stack([
            freq_maps[0] * PHI**0,
            freq_maps[1] * PHI**0.5,
            freq_maps[2] * PHI**1,
        ], axis=-1)
        
        # Stack other features
        other_stack = np.stack([
            vertical * PHI**0,
            contrast * PHI**0.5,
        ], axis=-1)
        
        # Combine all
        combined = np.concatenate([struct_spatial, edge_stack, freq_stack, other_stack], axis=-1)
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
            'pred_depth': pred_resized,
            'mae': mae,
            'corr': corr
        })
        
        print(f"  {img_id}: MAE={mae:.3f}, Corr={corr:.3f}")
    
    return results


def create_comparison_visualization(results: list, transcoder: dict):
    """Visualize multi-scale transcoder results."""
    
    n_images = len(results)
    
    fig = plt.figure(figsize=(16, 3.5 * n_images + 1))
    fig.suptitle('Multi-Scale φ-Edge Transcoder\n'
                 f'Edge scales: φ⁰, φ⁰·⁵, φ¹, φ¹·⁵ | Freq scales: 3 | Components: {transcoder["n_components"]}',
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
        title = f'Multi-Scale φ (Corr: {r["corr"]:.3f})' if row == 0 else f'Corr: {r["corr"]:.3f}'
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
    
    # Compare to baseline
    baseline_corr = 0.80  # From previous hybrid experiment
    improvement = (avg_corr - baseline_corr) / baseline_corr * 100
    
    summary = f"""
    MULTI-SCALE φ-EDGE TRANSCODER: 384 DA2 + 4 edge scales + 3 freq scales + 2 other = 393 features
    Test MAE: {avg_mae:.4f}  |  Test Corr: {avg_corr:.4f}  |  vs Baseline: {improvement:+.1f}%
    """
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightyellow'))
    
    output_file = OUTPUT_PATH / "da2_multiscale_phi_edges.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Learn multi-scale transcoder
    transcoder = learn_multiscale_transcoder(model, processor, n_train=25)
    
    # Test
    results = test_multiscale_transcoder(model, processor, transcoder, n_test=10)
    
    # Visualize
    viz_file = create_comparison_visualization(results, transcoder)
    
    # Summary
    avg_corr = np.mean([r['corr'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    print("\n" + "=" * 70)
    print("MULTI-SCALE φ-EDGE RESULTS")
    print("=" * 70)
    print(f"\n  Average MAE: {avg_mae:.4f}")
    print(f"  Average Correlation: {avg_corr:.4f}")
    print()
    print("  Multi-scale features:")
    print("    - 4 edge scales: σ = 1, φ, φ², φ³")
    print("    - 3 frequency bands: high, medium, low")
    print("    - Vertical gradient + local contrast")
    print()
    print("  Each scale captures different depth information:")
    print("    - Fine edges → close objects")
    print("    - Coarse edges → far structures")
