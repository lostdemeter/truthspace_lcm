#!/usr/bin/env python3
"""
DA2 Single Image Colorizer - Can we 100% recolorize ONE image?

The hypothesis: If we train the decoder on a SINGLE image,
we should be able to perfectly reconstruct its colors.

If we can't, something is fundamentally missing.
If we can, then the issue is generalization across images.

Also check: Do close-up dimensions (73, 162, 54, 138 from Doc 122)
affect color encoding?

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import pearsonr
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = -0.147 * rgb[..., 0] - 0.289 * rgb[..., 1] + 0.436 * rgb[..., 2]
    v = 0.615 * rgb[..., 0] - 0.515 * rgb[..., 1] - 0.100 * rgb[..., 2]
    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


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
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
    
    return structure


def single_image_colorize_test(model, processor, rgb: np.ndarray, n_dims: int = 384):
    """
    Test: Can we perfectly recolorize a SINGLE image?
    
    Train decoder on this image's patches, test on same image.
    Use ALL 384 dimensions to maximize information.
    """
    if rgb.max() > 1:
        rgb = rgb.astype(np.float32) / 255.0
    
    # Extract structure
    structure = extract_da2_structure(model, processor, rgb)
    structure = structure[1:]  # Skip CLS
    
    N, C = structure.shape
    H, W = rgb.shape[:2]
    
    # Get spatial dimensions
    H_s = int(np.sqrt(N * H / W))
    W_s = N // H_s
    
    if H_s * W_s != N:
        for h in range(1, int(np.sqrt(N)) + 10):
            if N % h == 0:
                w = N // h
                if abs(w/h - W/H) < 0.5:
                    H_s, W_s = h, w
                    break
    
    struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
    
    # Get ground truth color at patch resolution
    rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
    yuv_small = rgb_to_yuv(rgb_small * 255)
    
    # Collect all patches from THIS image
    features = []
    u_vals = []
    v_vals = []
    
    for y in range(H_s):
        for x in range(W_s):
            features.append(struct_spatial[y, x])
            u_vals.append(yuv_small[y, x, 1])
            v_vals.append(yuv_small[y, x, 2])
    
    features = np.array(features)
    u_vals = np.array(u_vals)
    v_vals = np.array(v_vals)
    
    print(f"   Single image: {H_s}x{W_s} = {len(features)} patches")
    
    # Train decoder on ALL dimensions
    # Use linear regression: U = features @ u_weights
    u_weights = np.linalg.lstsq(features, u_vals, rcond=None)[0]
    v_weights = np.linalg.lstsq(features, v_vals, rcond=None)[0]
    
    # Predict on same image
    u_pred = features @ u_weights
    v_pred = features @ v_weights
    
    # Compute correlation and error
    corr_u = np.corrcoef(u_vals, u_pred)[0, 1]
    corr_v = np.corrcoef(v_vals, v_pred)[0, 1]
    
    rmse_u = np.sqrt(np.mean((u_vals - u_pred)**2))
    rmse_v = np.sqrt(np.mean((v_vals - v_pred)**2))
    
    print(f"   Single-image decoder:")
    print(f"     U: corr={corr_u:.4f}, RMSE={rmse_u:.6f}")
    print(f"     V: corr={corr_v:.4f}, RMSE={rmse_v:.6f}")
    
    # Reconstruct the image
    u_map = u_pred.reshape(H_s, W_s)
    v_map = v_pred.reshape(H_s, W_s)
    
    # Upsample
    u_full = zoom(u_map, (H / H_s, W / W_s), order=3)[:H, :W]
    v_full = zoom(v_map, (H / H_s, W / W_s), order=3)[:H, :W]
    
    # Get luminance from original
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    yuv_recon = np.stack([gray, u_full, v_full], axis=-1)
    rgb_recon = yuv_to_rgb(yuv_recon)
    
    # Compute MAE
    mae = np.abs(rgb_recon.astype(float) - (rgb * 255).astype(float)).mean()
    
    print(f"     Reconstruction MAE: {mae:.2f}")
    
    return rgb_recon, u_weights, v_weights, corr_u, corr_v, mae


def analyze_per_image_dimensions(model, processor, images: List[np.ndarray]):
    """
    Analyze: Do different images use different dimensions for color?
    
    For each image, find which dimensions correlate most with U and V.
    See if there's consistency or if each image is different.
    """
    print("\n   Analyzing per-image color dimensions...")
    
    all_u_top_dims = []
    all_v_top_dims = []
    
    for i, rgb in enumerate(images):
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0
        
        structure = extract_da2_structure(model, processor, rgb)
        structure = structure[1:]
        
        N, C = structure.shape
        H, W = rgb.shape[:2]
        
        H_s = int(np.sqrt(N * H / W))
        W_s = N // H_s
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - W/H) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        yuv_small = rgb_to_yuv(rgb_small * 255)
        
        # Compute correlations for this image
        u_corrs = np.zeros(384)
        v_corrs = np.zeros(384)
        
        features_flat = struct_spatial.reshape(-1, 384)
        u_flat = yuv_small[:,:,1].flatten()
        v_flat = yuv_small[:,:,2].flatten()
        
        for dim in range(384):
            c_u, _ = pearsonr(features_flat[:, dim], u_flat)
            c_v, _ = pearsonr(features_flat[:, dim], v_flat)
            u_corrs[dim] = c_u if not np.isnan(c_u) else 0
            v_corrs[dim] = c_v if not np.isnan(c_v) else 0
        
        # Top 10 dimensions for this image
        u_top = np.argsort(np.abs(u_corrs))[::-1][:10]
        v_top = np.argsort(np.abs(v_corrs))[::-1][:10]
        
        all_u_top_dims.append(set(u_top))
        all_v_top_dims.append(set(v_top))
        
        print(f"     Image {i}: U top dims = {list(u_top[:5])}, V top dims = {list(v_top[:5])}")
    
    # Check overlap between images
    if len(all_u_top_dims) > 1:
        u_common = all_u_top_dims[0]
        v_common = all_v_top_dims[0]
        for i in range(1, len(all_u_top_dims)):
            u_common = u_common.intersection(all_u_top_dims[i])
            v_common = v_common.intersection(all_v_top_dims[i])
        
        print(f"\n   Common U dimensions across all images: {u_common}")
        print(f"   Common V dimensions across all images: {v_common}")
    
    return all_u_top_dims, all_v_top_dims


def load_coco_images(n_images: int, start_idx: int = 0) -> List[Tuple[str, np.ndarray]]:
    image_files = sorted(COCO_PATH.glob("*.jpg"))
    images = []
    for img_path in image_files[start_idx:start_idx + n_images]:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append((img_path.stem, img))
        except:
            pass
    return images


def run_single_image_test():
    """Test single-image colorization."""
    print("=" * 70)
    print("DA2 SINGLE IMAGE COLORIZER")
    print("Can we 100% recolorize ONE image?")
    print("=" * 70)
    
    print("\n0. LOADING DA2")
    print("-" * 50)
    model, processor = load_da2()
    
    test_data = load_coco_images(5, start_idx=200)
    
    print("\n1. SINGLE IMAGE RECONSTRUCTION")
    print("-" * 50)
    print("   Training decoder on each image, testing on SAME image")
    
    results = []
    for name, img in test_data:
        print(f"\n   === Image: {name} ===")
        rgb_recon, u_w, v_w, corr_u, corr_v, mae = single_image_colorize_test(
            model, processor, img, n_dims=384
        )
        results.append((name, img, rgb_recon, corr_u, corr_v, mae))
    
    print("\n2. PER-IMAGE DIMENSION ANALYSIS")
    print("-" * 50)
    images = [img for _, img in test_data]
    u_dims, v_dims = analyze_per_image_dimensions(model, processor, images)
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    
    for i, (name, original, recon, corr_u, corr_v, mae) in enumerate(results):
        gray = (0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]).astype(np.uint8)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(recon)
        axes[i, 2].set_title(f'Single-img recon ({mae:.1f})' if i == 0 else f'{mae:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(recon.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=30)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    avg_mae = np.mean([r[5] for r in results])
    avg_corr_u = np.mean([r[3] for r in results])
    avg_corr_v = np.mean([r[4] for r in results])
    
    fig.suptitle(f'Single-Image Reconstruction: Avg MAE={avg_mae:.1f}, U corr={avg_corr_u:.3f}, V corr={avg_corr_v:.3f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "da2_single_image_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'da2_single_image_test.png'}")
    
    return results


if __name__ == "__main__":
    results = run_single_image_test()
    
    print("\n" + "=" * 70)
    print("SINGLE IMAGE TEST SUMMARY")
    print("=" * 70)
    
    avg_mae = np.mean([r[5] for r in results])
    avg_corr_u = np.mean([r[3] for r in results])
    avg_corr_v = np.mean([r[4] for r in results])
    
    print(f"""
   Question: Can we 100% recolorize a SINGLE image?
   
   Results (training and testing on SAME image):
   - Average MAE: {avg_mae:.2f}
   - Average U correlation: {avg_corr_u:.4f}
   - Average V correlation: {avg_corr_v:.4f}
   
   Per-image results:
""")
    for name, _, _, corr_u, corr_v, mae in results:
        print(f"     {name}: MAE={mae:.2f}, U={corr_u:.4f}, V={corr_v:.4f}")
    
    if avg_corr_u > 0.99 and avg_corr_v > 0.99:
        print(f"""
   CONCLUSION: YES! We can nearly perfectly recolorize single images.
   The issue is GENERALIZATION across images, not the encoding itself.
""")
    else:
        print(f"""
   CONCLUSION: Even single-image reconstruction isn't perfect.
   Something is fundamentally limited in how DA2 encodes color.
""")
