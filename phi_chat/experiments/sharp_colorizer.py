#!/usr/bin/env python3
"""
Sharp Colorizer - Preserve Original Detail While Adding Color

The insight: We have the COLORS right, but the image is blurry because
we're applying flat colors to patches.

Solution: Use the grayscale as LUMINANCE, and only add CHROMINANCE from
the drum. This preserves all the original detail.

Process:
1. Predict color per patch (as before)
2. Convert predicted color to chrominance (UV in YUV space)
3. Keep original grayscale as luminance (Y)
4. Combine: sharp Y + predicted UV = sharp colorized image

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')
from phi_space import PhiSpace

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to YUV color space."""
    rgb = rgb.astype(np.float32) / 255.0
    
    # Standard YUV conversion matrix
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = -0.147 * rgb[..., 0] - 0.289 * rgb[..., 1] + 0.436 * rgb[..., 2]
    v = 0.615 * rgb[..., 0] - 0.515 * rgb[..., 1] - 0.100 * rgb[..., 2]
    
    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Convert YUV to RGB color space."""
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


class SharpColorizer:
    """
    Colorizer that preserves original image sharpness.
    
    Key insight: Predict COLOR (chrominance) per patch, but keep
    original DETAIL (luminance) from grayscale.
    """
    
    def __init__(self, n_feature_dims: int = 16, patch_size: int = 16):
        self.n_dims = n_feature_dims
        self.patch_size = patch_size
        self.drum = PhiSpace(dims=n_feature_dims)
        self.n_images = 0
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract geometric features from grayscale patch."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        luminance = patch.mean()
        contrast = patch.std()
        
        if h > 1 and w > 1:
            texture_h = np.abs(np.diff(patch, axis=1)).mean()
            texture_v = np.abs(np.diff(patch, axis=0)).mean()
        else:
            texture_h = texture_v = 0.0
        
        if h > 2 and w > 2:
            center = patch[1:-1, 1:-1]
            neighbors = (patch[:-2, 1:-1] + patch[2:, 1:-1] + 
                        patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
            edge_density = np.abs(center - neighbors).mean()
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            gradient_mag = np.sqrt(gx**2 + gy**2).mean()
            gradient_dir = np.arctan2(gy.mean(), gx.mean()) / np.pi
        else:
            edge_density = gradient_mag = gradient_dir = 0.0
        
        total_var = texture_h + texture_v
        smoothness = 1.0 / (1.0 + total_var * 10)
        
        local_max = patch.max()
        local_min = patch.min()
        
        if h >= 4 and w >= 4:
            coarse = patch.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            texture_coarse = coarse.std()
            texture_fine = patch.reshape(h//2, 2, w//2, 2).std(axis=(1, 3)).mean()
        else:
            texture_coarse = texture_fine = contrast
        
        uniformity = 1.0 / (1.0 + contrast * 5)
        
        hist, _ = np.histogram(patch.flatten(), bins=8, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy_approx = -np.sum(hist * np.log(hist + 1e-10)) / np.log(8)
        
        return np.array([
            luminance, contrast, texture_h, texture_v,
            y_pos, x_pos, edge_density, smoothness,
            gradient_mag, gradient_dir, local_max, local_min,
            texture_coarse, texture_fine, uniformity, entropy_approx
        ], dtype=np.float32)
    
    def learn_from_image(self, color_image: np.ndarray, sample_rate: float = 0.15) -> int:
        """Learn color mappings from a color image."""
        H, W = color_image.shape[:2]
        
        grayscale = (0.299 * color_image[:,:,0] + 
                     0.587 * color_image[:,:,1] + 
                     0.114 * color_image[:,:,2]).astype(np.uint8)
        
        # Convert to YUV to store chrominance
        yuv = rgb_to_yuv(color_image)
        
        patches_learned = 0
        
        for y in range(0, H - self.patch_size, self.patch_size):
            for x in range(0, W - self.patch_size, self.patch_size):
                if np.random.random() > sample_rate:
                    continue
                
                gray_patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                features = self.extract_features(gray_patch, y_pos, x_pos)
                
                # Store mean chrominance (U, V) - NOT luminance
                mean_u = yuv_patch[:, :, 1].mean()
                mean_v = yuv_patch[:, :, 2].mean()
                
                # Also store RGB for reference
                mean_rgb = color_image[y:y+self.patch_size, x:x+self.patch_size].mean(axis=(0, 1))
                
                point_id = f"p_{self.n_images}_{patches_learned}"
                self.drum.add(point_id, features, metadata={
                    'u': float(mean_u),
                    'v': float(mean_v),
                    'rgb': tuple(mean_rgb.astype(int))
                })
                patches_learned += 1
        
        self.n_images += 1
        return patches_learned
    
    def predict_chrominance(self, gray_patch: np.ndarray, 
                            y_pos: float, x_pos: float, 
                            k: int = 5) -> Tuple[float, float]:
        """Predict U, V chrominance for a patch."""
        features = self.extract_features(gray_patch, y_pos, x_pos)
        nearest = self.drum.query(features, k=k)
        
        if not nearest:
            return 0.0, 0.0  # Neutral (grayscale)
        
        total_weight = 0
        weighted_u = 0
        weighted_v = 0
        
        for point_id, distance in nearest:
            point = self.drum[point_id]
            u = point.metadata['u']
            v = point.metadata['v']
            
            weight = 1.0 / (distance**2 + 0.001)
            weighted_u += weight * u
            weighted_v += weight * v
            total_weight += weight
        
        return weighted_u / total_weight, weighted_v / total_weight
    
    def colorize_sharp(self, grayscale: np.ndarray) -> np.ndarray:
        """
        Colorize while preserving original sharpness.
        
        Process:
        1. Create low-res chrominance map from patch predictions
        2. Upsample chrominance to full resolution (smooth)
        3. Combine with original grayscale as luminance
        """
        H, W = grayscale.shape
        
        # Compute chrominance at patch resolution
        n_patches_y = H // self.patch_size
        n_patches_x = W // self.patch_size
        
        u_map = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)
        v_map = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)
        
        for py in range(n_patches_y):
            for px in range(n_patches_x):
                y = py * self.patch_size
                x = px * self.patch_size
                
                patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                u, v = self.predict_chrominance(patch, y_pos, x_pos)
                u_map[py, px] = u
                v_map[py, px] = v
        
        # Upsample chrominance to full resolution using bilinear interpolation
        # This gives smooth color transitions while keeping sharp luminance
        scale_y = H / n_patches_y
        scale_x = W / n_patches_x
        u_full = zoom(u_map, (scale_y, scale_x), order=1)
        v_full = zoom(v_map, (scale_y, scale_x), order=1)
        
        # Ensure exact size match
        u_full = u_full[:H, :W]
        v_full = v_full[:H, :W]
        
        # Pad if needed
        if u_full.shape[0] < H or u_full.shape[1] < W:
            u_padded = np.zeros((H, W), dtype=np.float32)
            v_padded = np.zeros((H, W), dtype=np.float32)
            u_padded[:u_full.shape[0], :u_full.shape[1]] = u_full
            v_padded[:v_full.shape[0], :v_full.shape[1]] = v_full
            u_full, v_full = u_padded, v_padded
        
        # Combine: original grayscale as Y, predicted as U/V
        y_channel = grayscale.astype(np.float32) / 255.0
        
        yuv = np.stack([y_channel, u_full, v_full], axis=-1)
        rgb = yuv_to_rgb(yuv)
        
        return rgb


def load_coco_images(n_images: int, start_idx: int = 0) -> List[Tuple[str, np.ndarray]]:
    """Load COCO images."""
    image_files = sorted(COCO_PATH.glob("*.jpg"))
    images = []
    for img_path in image_files[start_idx:start_idx + n_images]:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append((img_path.stem, img))
        except:
            pass
    return images


def run_sharp_test():
    """Test the sharp colorizer."""
    print("=" * 70)
    print("SHARP COLORIZATION TEST")
    print("=" * 70)
    
    colorizer = SharpColorizer(n_feature_dims=16, patch_size=16)
    
    # Train
    print("\n1. TRAINING (150 images)")
    print("-" * 50)
    train_images = load_coco_images(150, start_idx=0)
    
    for i, (name, img) in enumerate(train_images):
        colorizer.learn_from_image(img, sample_rate=0.12)
        if (i + 1) % 50 == 0:
            print(f"   Trained on {i+1}/{len(train_images)} images")
    
    print(f"   Drum size: {len(colorizer.drum)} points")
    colorizer.drum.build_index()
    
    # Test
    print("\n2. TESTING (5 images)")
    print("-" * 50)
    test_images = load_coco_images(5, start_idx=200)
    
    results = []
    for name, img in test_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        
        colorized = colorizer.colorize_sharp(gray)
        
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        print(f"   {name}: MAE = {error:.2f}")
        
        results.append((name, img, gray, colorized, error))
    
    avg_error = np.mean([r[4] for r in results])
    print(f"   Average MAE: {avg_error:.2f}")
    
    # Visualize
    print("\n3. CREATING VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    
    for i, (name, original, gray, colorized, error) in enumerate(results):
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Sharp Colorized (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Sharp Colorization: Preserves Original Detail (Avg MAE={avg_error:.1f})',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "sharp_colorization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_sharp_test()
    
    print("\n" + "=" * 70)
    print("SHARP COLORIZATION SUMMARY")
    print("=" * 70)
    print("""
   The key insight: Separate LUMINANCE (detail) from CHROMINANCE (color).
   
   Process:
   1. Predict chrominance (U, V) per patch from drum
   2. Upsample chrominance smoothly to full resolution
   3. Keep original grayscale as luminance (Y)
   4. Combine: Y (sharp) + UV (smooth) = sharp colorized image
   
   Result: Colors from the drum, sharpness from the original.
""")
