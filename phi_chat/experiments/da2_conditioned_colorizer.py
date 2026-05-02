#!/usr/bin/env python3
"""
DA2 Conditioned Colorizer - Use image-type dimensions to condition color decoding

Key finding: Each image uses DIFFERENT dimensions for color encoding.
The close-up dimensions (73, 162, 54, 138) vary significantly between images.

Hypothesis: Color encoding is CONDITIONED on image type.
We need to include these conditioning dimensions in our decoder.

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

# Image-type dimensions from Doc 122
CLOSEUP_DIMS = [73, 162, 54, 138]
# Center distance dimension
CENTER_DIST_DIM = 262


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


def get_image_type_features(structure: np.ndarray) -> np.ndarray:
    """
    Extract image-type features from structure.
    These are global features that describe the image type.
    """
    # Skip CLS token
    patches = structure[1:]
    
    # Global mean of close-up dimensions
    closeup_features = patches[:, CLOSEUP_DIMS].mean(axis=0)
    
    # Center distance dimension mean
    center_dist = patches[:, CENTER_DIST_DIM].mean()
    
    # Overall structure statistics
    global_mean = patches.mean(axis=0)
    global_std = patches.std(axis=0)
    
    return closeup_features, center_dist, global_mean, global_std


def build_conditioned_decoder(model, processor, images: List[np.ndarray], n_dims: int = 100):
    """
    Build a decoder that's conditioned on image-type features.
    
    Instead of just: U = features @ weights
    We use: U = features @ weights + image_type @ type_weights
    
    This allows the decoder to adapt based on image type.
    """
    print("   Building conditioned decoder...")
    
    all_features = []
    all_u = []
    all_v = []
    all_image_types = []
    
    for i, rgb in enumerate(images):
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0
        
        structure = extract_da2_structure(model, processor, rgb)
        
        # Get image-type features
        closeup_feats, center_dist, global_mean, global_std = get_image_type_features(structure)
        image_type = np.concatenate([closeup_feats, [center_dist]])
        
        structure = structure[1:]  # Skip CLS
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
        
        for y in range(H_s):
            for x in range(W_s):
                # Concatenate patch features with image-type features
                patch_feat = struct_spatial[y, x]
                combined_feat = np.concatenate([patch_feat, image_type])
                
                all_features.append(combined_feat)
                all_u.append(yuv_small[y, x, 1])
                all_v.append(yuv_small[y, x, 2])
        
        if (i + 1) % 5 == 0:
            print(f"     Processed {i+1}/{len(images)}")
    
    features = np.array(all_features)
    u_vals = np.array(all_u)
    v_vals = np.array(all_v)
    
    print(f"   Collected {len(features)} patches, {features.shape[1]} features each")
    print(f"   (384 patch dims + {features.shape[1] - 384} image-type dims)")
    
    # Find top correlated dimensions
    u_corrs = np.zeros(features.shape[1])
    v_corrs = np.zeros(features.shape[1])
    
    for dim in range(features.shape[1]):
        c_u, _ = pearsonr(features[:, dim], u_vals)
        c_v, _ = pearsonr(features[:, dim], v_vals)
        u_corrs[dim] = c_u if not np.isnan(c_u) else 0
        v_corrs[dim] = c_v if not np.isnan(c_v) else 0
    
    # Use top n_dims
    u_top = np.argsort(np.abs(u_corrs))[::-1][:n_dims]
    v_top = np.argsort(np.abs(v_corrs))[::-1][:n_dims]
    
    # Check if image-type dims are in top
    n_patch_dims = 384
    u_type_in_top = sum(1 for d in u_top if d >= n_patch_dims)
    v_type_in_top = sum(1 for d in v_top if d >= n_patch_dims)
    print(f"   Image-type dims in top {n_dims}: U={u_type_in_top}, V={v_type_in_top}")
    
    # Linear regression on top dims
    X_u = features[:, u_top]
    X_v = features[:, v_top]
    
    u_coeffs = np.linalg.lstsq(X_u, u_vals, rcond=None)[0]
    v_coeffs = np.linalg.lstsq(X_v, v_vals, rcond=None)[0]
    
    # Test
    u_pred = X_u @ u_coeffs
    v_pred = X_v @ v_coeffs
    
    corr_u = np.corrcoef(u_vals, u_pred)[0, 1]
    corr_v = np.corrcoef(v_vals, v_pred)[0, 1]
    
    print(f"   Training correlation: U={corr_u:.4f}, V={corr_v:.4f}")
    
    return u_top, v_top, u_coeffs, v_coeffs, corr_u, corr_v


def colorize_conditioned(model, processor, rgb: np.ndarray, 
                         u_top, v_top, u_coeffs, v_coeffs):
    """Colorize using conditioned decoder."""
    if rgb.max() > 1:
        rgb_norm = rgb.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb
    
    structure = extract_da2_structure(model, processor, rgb_norm)
    
    # Get image-type features
    closeup_feats, center_dist, _, _ = get_image_type_features(structure)
    image_type = np.concatenate([closeup_feats, [center_dist]])
    
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
    
    u_map = np.zeros((H_s, W_s))
    v_map = np.zeros((H_s, W_s))
    
    for y in range(H_s):
        for x in range(W_s):
            patch_feat = struct_spatial[y, x]
            combined_feat = np.concatenate([patch_feat, image_type])
            
            u_map[y, x] = np.dot(combined_feat[u_top], u_coeffs)
            v_map[y, x] = np.dot(combined_feat[v_top], v_coeffs)
    
    # Smooth and amplify
    u_map = gaussian_filter(u_map, sigma=0.5) * 1.3
    v_map = gaussian_filter(v_map, sigma=0.5) * 1.3
    
    # Upsample
    u_full = zoom(u_map, (H / H_s, W / W_s), order=3)[:H, :W]
    v_full = zoom(v_map, (H / H_s, W / W_s), order=3)[:H, :W]
    
    gray = 0.299 * rgb_norm[:,:,0] + 0.587 * rgb_norm[:,:,1] + 0.114 * rgb_norm[:,:,2]
    
    yuv = np.stack([gray, u_full, v_full], axis=-1)
    return yuv_to_rgb(yuv)


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


def run_conditioned_test():
    """Test conditioned colorizer."""
    print("=" * 70)
    print("DA2 CONDITIONED COLORIZER")
    print("Using image-type dimensions to condition color decoding")
    print("=" * 70)
    
    print("\n0. LOADING DA2")
    print("-" * 50)
    model, processor = load_da2()
    
    train_data = load_coco_images(30, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. BUILDING CONDITIONED DECODER")
    print("-" * 50)
    u_top, v_top, u_coeffs, v_coeffs, train_corr_u, train_corr_v = build_conditioned_decoder(
        model, processor, train_images, n_dims=100
    )
    
    print("\n2. TESTING")
    print("-" * 50)
    
    results = []
    for name, img in test_data:
        colorized = colorize_conditioned(model, processor, img, u_top, v_top, u_coeffs, v_coeffs)
        mae = np.abs(colorized.astype(float) - img.astype(float)).mean()
        results.append((name, img, colorized, mae))
        print(f"   {name}: MAE = {mae:.2f}")
    
    avg_mae = np.mean([r[3] for r in results])
    print(f"\n   Average MAE: {avg_mae:.2f}")
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    
    for i, (name, original, colorized, mae) in enumerate(results):
        gray = (0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]).astype(np.uint8)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Conditioned ({mae:.1f})' if i == 0 else f'{mae:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=30)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Conditioned Colorizer: Avg MAE={avg_mae:.1f}, Train corr U={train_corr_u:.3f}, V={train_corr_v:.3f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "da2_conditioned_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'da2_conditioned_test.png'}")
    
    return results, avg_mae


if __name__ == "__main__":
    results, avg_mae = run_conditioned_test()
    
    print("\n" + "=" * 70)
    print("CONDITIONED COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   Key insight: Color encoding is CONDITIONED on image type.
   
   We added image-type features:
   - Close-up dimensions: 73, 162, 54, 138
   - Center distance dimension: 262
   
   Results:
   - Average test MAE: {avg_mae:.2f}
   
   Per-image:
""")
    for name, _, _, mae in results:
        print(f"     {name}: MAE={mae:.2f}")
