#!/usr/bin/env python3
"""
True Recreation Gap Analysis

Question: How far are we from PERFECT recreation?

We need to measure:
1. Theoretical minimum: What's the best possible MAE given the constraints?
2. Single-image ceiling: Best we can do when training on same image
3. Cross-image gap: How much we lose from generalization
4. Resolution gap: How much we lose from patch-level resolution
5. Information gap: Is the color information even IN the features?

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
LN_PHI = np.log(PHI)

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
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    return model, processor


def extract_da2_structure(model, processor, rgb: np.ndarray):
    import torch
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        backbone_output = model.backbone(inputs['pixel_values'], output_hidden_states=True)
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
    return structure


def measure_resolution_gap(rgb: np.ndarray, H_s: int, W_s: int) -> float:
    """
    Measure error from downsampling and upsampling color.
    
    This is the MINIMUM error we can achieve given patch resolution.
    """
    H, W = rgb.shape[:2]
    
    # Downsample to patch resolution
    rgb_small = np.array(Image.fromarray(rgb).resize((W_s, H_s)))
    
    # Upsample back
    rgb_recon = np.array(Image.fromarray(rgb_small).resize((W, H), Image.BICUBIC))
    
    mae = np.abs(rgb_recon.astype(float) - rgb.astype(float)).mean()
    return mae, rgb_recon


def measure_yuv_resolution_gap(rgb: np.ndarray, H_s: int, W_s: int) -> float:
    """
    Measure error from downsampling/upsampling in YUV space.
    
    Keep full-res Y, only downsample U and V.
    """
    H, W = rgb.shape[:2]
    
    yuv = rgb_to_yuv(rgb)
    y_full = yuv[:, :, 0]
    
    # Downsample U and V
    u_small = np.array(Image.fromarray((yuv[:,:,1] * 255 + 128).astype(np.uint8)).resize((W_s, H_s)))
    v_small = np.array(Image.fromarray((yuv[:,:,2] * 255 + 128).astype(np.uint8)).resize((W_s, H_s)))
    
    # Upsample
    u_full = np.array(Image.fromarray(u_small).resize((W, H), Image.BICUBIC)).astype(float)
    v_full = np.array(Image.fromarray(v_small).resize((W, H), Image.BICUBIC)).astype(float)
    
    u_full = (u_full - 128) / 255
    v_full = (v_full - 128) / 255
    
    yuv_recon = np.stack([y_full, u_full, v_full], axis=-1)
    rgb_recon = yuv_to_rgb(yuv_recon)
    
    mae = np.abs(rgb_recon.astype(float) - rgb.astype(float)).mean()
    return mae, rgb_recon


def measure_single_image_ceiling(model, processor, rgb: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Train decoder on THIS image, test on THIS image.
    
    This is the ceiling for what's possible with linear regression.
    """
    if rgb.max() > 1:
        rgb_norm = rgb.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb
    
    structure = extract_da2_structure(model, processor, rgb_norm)
    structure = structure[1:]
    
    N, C = structure.shape
    H, W = rgb_norm.shape[:2]
    
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
    
    rgb_small = np.array(Image.fromarray((rgb_norm * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
    yuv_small = rgb_to_yuv(rgb_small * 255)
    
    # Collect all patches
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
    
    # Train on ALL 384 dimensions
    u_weights = np.linalg.lstsq(features, u_vals, rcond=None)[0]
    v_weights = np.linalg.lstsq(features, v_vals, rcond=None)[0]
    
    # Predict
    u_pred = features @ u_weights
    v_pred = features @ v_weights
    
    # Reshape
    u_map = u_pred.reshape(H_s, W_s)
    v_map = v_pred.reshape(H_s, W_s)
    
    # Upsample
    u_full = zoom(u_map, (H / H_s, W / W_s), order=3)[:H, :W]
    v_full = zoom(v_map, (H / H_s, W / W_s), order=3)[:H, :W]
    
    gray = 0.299 * rgb_norm[:,:,0] + 0.587 * rgb_norm[:,:,1] + 0.114 * rgb_norm[:,:,2]
    
    yuv_recon = np.stack([gray, u_full, v_full], axis=-1)
    rgb_recon = yuv_to_rgb(yuv_recon)
    
    mae = np.abs(rgb_recon.astype(float) - (rgb_norm * 255).astype(float)).mean()
    
    # Also compute correlation
    corr_u = np.corrcoef(u_vals, u_pred)[0, 1]
    corr_v = np.corrcoef(v_vals, v_pred)[0, 1]
    
    return mae, rgb_recon, corr_u, corr_v, H_s, W_s


def measure_information_content(model, processor, rgb: np.ndarray) -> dict:
    """
    How much color information is actually in the features?
    
    Compute R² for predicting U and V from features.
    """
    if rgb.max() > 1:
        rgb_norm = rgb.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb
    
    structure = extract_da2_structure(model, processor, rgb_norm)
    structure = structure[1:]
    
    N, C = structure.shape
    H, W = rgb_norm.shape[:2]
    
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
    
    rgb_small = np.array(Image.fromarray((rgb_norm * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
    yuv_small = rgb_to_yuv(rgb_small * 255)
    
    features = struct_spatial.reshape(-1, C)
    u_vals = yuv_small[:,:,1].flatten()
    v_vals = yuv_small[:,:,2].flatten()
    
    # R² = 1 - SS_res / SS_tot
    u_weights = np.linalg.lstsq(features, u_vals, rcond=None)[0]
    v_weights = np.linalg.lstsq(features, v_vals, rcond=None)[0]
    
    u_pred = features @ u_weights
    v_pred = features @ v_weights
    
    ss_res_u = np.sum((u_vals - u_pred) ** 2)
    ss_tot_u = np.sum((u_vals - u_vals.mean()) ** 2)
    r2_u = 1 - ss_res_u / ss_tot_u
    
    ss_res_v = np.sum((v_vals - v_pred) ** 2)
    ss_tot_v = np.sum((v_vals - v_vals.mean()) ** 2)
    r2_v = 1 - ss_res_v / ss_tot_v
    
    return {
        'r2_u': r2_u,
        'r2_v': r2_v,
        'u_variance': np.var(u_vals),
        'v_variance': np.var(v_vals),
        'u_residual_variance': np.var(u_vals - u_pred),
        'v_residual_variance': np.var(v_vals - v_pred)
    }


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


def run_gap_analysis():
    """Analyze the gap between our results and true recreation."""
    print("=" * 70)
    print("TRUE RECREATION GAP ANALYSIS")
    print("How far are we from perfect?")
    print("=" * 70)
    
    print("\n0. LOADING DA2")
    print("-" * 50)
    model, processor = load_da2()
    
    test_data = load_coco_images(10, start_idx=200)
    
    print("\n1. MEASURING GAPS")
    print("-" * 50)
    
    results = []
    
    for name, rgb in test_data:
        print(f"\n   === {name} ===")
        
        # Get single-image ceiling first (to get H_s, W_s)
        mae_ceiling, rgb_ceiling, corr_u, corr_v, H_s, W_s = measure_single_image_ceiling(
            model, processor, rgb
        )
        
        # Resolution gap (pure downsampling/upsampling)
        mae_res_rgb, rgb_res = measure_resolution_gap(rgb, H_s, W_s)
        mae_res_yuv, rgb_res_yuv = measure_yuv_resolution_gap(rgb, H_s, W_s)
        
        # Information content
        info = measure_information_content(model, processor, rgb)
        
        print(f"     Patch resolution: {H_s}x{W_s}")
        print(f"     Resolution gap (RGB downsample): MAE = {mae_res_rgb:.2f}")
        print(f"     Resolution gap (YUV, Y full-res): MAE = {mae_res_yuv:.2f}")
        print(f"     Single-image ceiling (384 dims): MAE = {mae_ceiling:.2f}")
        print(f"     Correlation: U={corr_u:.4f}, V={corr_v:.4f}")
        print(f"     R²: U={info['r2_u']:.4f}, V={info['r2_v']:.4f}")
        
        results.append({
            'name': name,
            'rgb': rgb,
            'H_s': H_s,
            'W_s': W_s,
            'mae_res_rgb': mae_res_rgb,
            'mae_res_yuv': mae_res_yuv,
            'mae_ceiling': mae_ceiling,
            'corr_u': corr_u,
            'corr_v': corr_v,
            'r2_u': info['r2_u'],
            'r2_v': info['r2_v'],
            'rgb_ceiling': rgb_ceiling,
            'rgb_res_yuv': rgb_res_yuv
        })
    
    # Summary statistics
    print("\n2. SUMMARY")
    print("-" * 50)
    
    avg_res_rgb = np.mean([r['mae_res_rgb'] for r in results])
    avg_res_yuv = np.mean([r['mae_res_yuv'] for r in results])
    avg_ceiling = np.mean([r['mae_ceiling'] for r in results])
    avg_r2_u = np.mean([r['r2_u'] for r in results])
    avg_r2_v = np.mean([r['r2_v'] for r in results])
    
    print(f"""
   Average across {len(results)} images:
   
   THEORETICAL LIMITS:
   - Resolution gap (RGB): {avg_res_rgb:.2f} (minimum from downsampling)
   - Resolution gap (YUV): {avg_res_yuv:.2f} (minimum with full-res Y)
   
   SINGLE-IMAGE CEILING:
   - Linear regression (384 dims): {avg_ceiling:.2f}
   - R² for U: {avg_r2_u:.4f} ({avg_r2_u*100:.1f}% of variance explained)
   - R² for V: {avg_r2_v:.4f} ({avg_r2_v*100:.1f}% of variance explained)
   
   GAP ANALYSIS:
   - Resolution accounts for: {avg_res_yuv:.2f} MAE
   - Linear regression adds: {avg_ceiling - avg_res_yuv:.2f} MAE
   - Total single-image: {avg_ceiling:.2f} MAE
   
   WHAT WE'VE BEEN ACHIEVING:
   - Cross-image (50 dims): ~12-14 MAE
   - Gap from ceiling: {12 - avg_ceiling:.2f} MAE (generalization loss)
""")
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    n_show = min(5, len(results))
    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
    
    for i in range(n_show):
        r = results[i]
        
        axes[i, 0].imshow(r['rgb'])
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        gray = (0.299 * r['rgb'][:,:,0] + 0.587 * r['rgb'][:,:,1] + 0.114 * r['rgb'][:,:,2]).astype(np.uint8)
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(r['rgb_res_yuv'])
        axes[i, 2].set_title(f'Resolution limit ({r["mae_res_yuv"]:.1f})' if i == 0 else f'{r["mae_res_yuv"]:.1f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(r['rgb_ceiling'])
        axes[i, 3].set_title(f'Single-img ceiling ({r["mae_ceiling"]:.1f})' if i == 0 else f'{r["mae_ceiling"]:.1f}')
        axes[i, 3].axis('off')
        
        diff = np.abs(r['rgb_ceiling'].astype(float) - r['rgb'].astype(float)).mean(axis=2)
        axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=30)
        axes[i, 4].set_title('Ceiling error' if i == 0 else '')
        axes[i, 4].axis('off')
    
    fig.suptitle(f'Gap Analysis: Resolution limit={avg_res_yuv:.1f}, Single-image ceiling={avg_ceiling:.1f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "true_recreation_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'true_recreation_gap.png'}")
    
    return results


if __name__ == "__main__":
    results = run_gap_analysis()
    
    print("\n" + "=" * 70)
    print("GAP ANALYSIS CONCLUSION")
    print("=" * 70)
    
    avg_res_yuv = np.mean([r['mae_res_yuv'] for r in results])
    avg_ceiling = np.mean([r['mae_ceiling'] for r in results])
    avg_r2_u = np.mean([r['r2_u'] for r in results])
    avg_r2_v = np.mean([r['r2_v'] for r in results])
    
    print(f"""
   THE GAPS:
   
   1. RESOLUTION GAP: {avg_res_yuv:.2f} MAE
      - This is UNAVOIDABLE given patch-level resolution
      - Even with perfect color, we lose detail
   
   2. INFORMATION GAP: {(1-avg_r2_u)*100:.1f}% U, {(1-avg_r2_v)*100:.1f}% V unexplained
      - DA2 features don't capture ALL color information
      - Some color variance is simply not in the features
   
   3. SINGLE-IMAGE CEILING: {avg_ceiling:.2f} MAE
      - Best possible with linear regression on 384 dims
      - This is our TARGET for single-image
   
   4. GENERALIZATION GAP: ~{12 - avg_ceiling:.1f} MAE
      - Loss from training on multiple images
      - Each image uses different dimensions
   
   IMPLICATION:
   We're not far from the ceiling! The main gaps are:
   - Resolution (unavoidable)
   - Information content (DA2 wasn't trained for color)
   - Generalization (each image is different)
   
   To improve further, we need:
   - Higher resolution features (not patch-level)
   - A backbone trained to preserve color
   - Or: Accept that ~{avg_ceiling:.0f} MAE is the practical limit
""")
