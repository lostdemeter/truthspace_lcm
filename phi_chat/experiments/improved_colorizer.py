#!/usr/bin/env python3
"""
Improved Geometric Colorizer

Improvements over the baseline:
1. Better features (texture descriptors, multi-scale)
2. Overlapping patches with Gaussian blending
3. More training data (200 images)
4. Larger feature space (16 dimensions)

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, sobel
from typing import List, Tuple, Dict
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')
from phi_space import PhiSpace

PHI = (1 + np.sqrt(5)) / 2

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


class ImprovedColorizer:
    """
    Improved geometric colorizer with better features and blending.
    """
    
    def __init__(self, n_feature_dims: int = 16, patch_size: int = 16):
        self.n_dims = n_feature_dims
        self.patch_size = patch_size
        self.drum = PhiSpace(
            dims=n_feature_dims,
            dim_names=[
                'luminance', 'contrast', 'texture_h', 'texture_v',
                'y_position', 'x_position', 'edge_density', 'smoothness',
                'gradient_mag', 'gradient_dir', 'local_max', 'local_min',
                'texture_coarse', 'texture_fine', 'uniformity', 'entropy_approx'
            ]
        )
        self.n_images = 0
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float,
                         context: np.ndarray = None) -> np.ndarray:
        """
        Extract rich geometric features from grayscale patch.
        
        16 dimensions capturing luminance, texture, edges, and context.
        """
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        # Basic features (0-7)
        luminance = patch.mean()
        contrast = patch.std()
        
        if h > 1 and w > 1:
            texture_h = np.abs(np.diff(patch, axis=1)).mean()
            texture_v = np.abs(np.diff(patch, axis=0)).mean()
        else:
            texture_h = texture_v = 0.0
        
        y_position = y_pos
        x_position = x_pos
        
        if h > 2 and w > 2:
            center = patch[1:-1, 1:-1]
            neighbors = (patch[:-2, 1:-1] + patch[2:, 1:-1] + 
                        patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
            edge_density = np.abs(center - neighbors).mean()
        else:
            edge_density = 0.0
        
        total_var = texture_h + texture_v
        smoothness = 1.0 / (1.0 + total_var * 10)
        
        # Advanced features (8-15)
        
        # Gradient magnitude and direction
        if h > 2 and w > 2:
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            gradient_mag = np.sqrt(gx**2 + gy**2).mean()
            # Dominant gradient direction (simplified)
            gradient_dir = np.arctan2(gy.mean(), gx.mean()) / np.pi  # Normalize to [-1, 1]
        else:
            gradient_mag = 0.0
            gradient_dir = 0.0
        
        # Local extrema
        local_max = patch.max()
        local_min = patch.min()
        
        # Multi-scale texture
        if h >= 4 and w >= 4:
            # Coarse texture (2x2 blocks)
            coarse = patch.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            texture_coarse = coarse.std()
            # Fine texture (pixel level variance in 2x2 blocks)
            texture_fine = patch.reshape(h//2, 2, w//2, 2).std(axis=(1, 3)).mean()
        else:
            texture_coarse = contrast
            texture_fine = contrast
        
        # Uniformity (how uniform is the patch)
        uniformity = 1.0 / (1.0 + contrast * 5)
        
        # Entropy approximation (histogram spread)
        hist, _ = np.histogram(patch.flatten(), bins=8, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy_approx = -np.sum(hist * np.log(hist + 1e-10)) / np.log(8)  # Normalize to [0, 1]
        
        return np.array([
            luminance, contrast, texture_h, texture_v,
            y_position, x_position, edge_density, smoothness,
            gradient_mag, gradient_dir, local_max, local_min,
            texture_coarse, texture_fine, uniformity, entropy_approx
        ], dtype=np.float32)
    
    def learn_from_image(self, color_image: np.ndarray, sample_rate: float = 0.2) -> int:
        """Learn from a single color image with multi-scale patches."""
        H, W = color_image.shape[:2]
        
        grayscale = (0.299 * color_image[:,:,0] + 
                     0.587 * color_image[:,:,1] + 
                     0.114 * color_image[:,:,2]).astype(np.uint8)
        
        patches_learned = 0
        
        # Sample at multiple scales
        for scale in [1.0, 0.5]:
            ps = int(self.patch_size * scale)
            if ps < 4:
                continue
                
            for y in range(0, H - ps, ps):
                for x in range(0, W - ps, ps):
                    if np.random.random() > sample_rate:
                        continue
                    
                    gray_patch = grayscale[y:y+ps, x:x+ps]
                    color_patch = color_image[y:y+ps, x:x+ps]
                    
                    # Resize to standard patch size for feature extraction
                    if scale != 1.0:
                        gray_patch = np.array(Image.fromarray(gray_patch).resize(
                            (self.patch_size, self.patch_size), Image.BILINEAR))
                    
                    y_pos = (y + ps/2) / H
                    x_pos = (x + ps/2) / W
                    
                    features = self.extract_features(gray_patch, y_pos, x_pos)
                    mean_color = color_patch.mean(axis=(0, 1)).astype(np.uint8)
                    
                    # Also store color variance for confidence
                    color_var = color_patch.std(axis=(0, 1)).mean()
                    
                    point_id = f"p_{self.n_images}_{patches_learned}"
                    self.drum.add(point_id, features, metadata={
                        'rgb': tuple(mean_color),
                        'color_var': float(color_var),
                        'scale': scale
                    })
                    patches_learned += 1
        
        self.n_images += 1
        return patches_learned
    
    def colorize_patch(self, gray_patch: np.ndarray, 
                       y_pos: float, x_pos: float, k: int = 7) -> Tuple[np.ndarray, float]:
        """
        Colorize a single patch with confidence score.
        
        Returns (color, confidence)
        """
        features = self.extract_features(gray_patch, y_pos, x_pos)
        nearest = self.drum.query(features, k=k)
        
        if not nearest:
            return np.array([128, 128, 128], dtype=np.uint8), 0.0
        
        total_weight = 0
        weighted_color = np.zeros(3, dtype=np.float32)
        
        for point_id, distance in nearest:
            point = self.drum[point_id]
            rgb = np.array(point.metadata['rgb'], dtype=np.float32)
            
            # Weight by inverse distance squared (sharper falloff)
            weight = 1.0 / (distance**2 + 0.001)
            
            # Reduce weight for high-variance patches (less confident)
            color_var = point.metadata.get('color_var', 50)
            confidence_factor = 1.0 / (1.0 + color_var / 50)
            weight *= confidence_factor
            
            weighted_color += weight * rgb
            total_weight += weight
        
        color = (weighted_color / total_weight).astype(np.uint8)
        
        # Confidence based on distance to nearest
        confidence = 1.0 / (1.0 + nearest[0][1])
        
        return color, confidence
    
    def colorize(self, grayscale: np.ndarray, overlap: float = 0.5) -> np.ndarray:
        """
        Colorize with overlapping patches and Gaussian blending.
        
        Args:
            grayscale: HxW grayscale image
            overlap: Overlap fraction (0.5 = 50% overlap)
        
        Returns:
            HxWx3 RGB image
        """
        H, W = grayscale.shape
        output = np.zeros((H, W, 3), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)
        
        step = int(self.patch_size * (1 - overlap))
        step = max(step, 1)
        
        # Create Gaussian weight window
        y_weights = np.exp(-((np.arange(self.patch_size) - self.patch_size/2)**2) / (self.patch_size/2)**2)
        x_weights = np.exp(-((np.arange(self.patch_size) - self.patch_size/2)**2) / (self.patch_size/2)**2)
        gaussian_window = np.outer(y_weights, x_weights)
        
        for y in range(0, H - self.patch_size + 1, step):
            for x in range(0, W - self.patch_size + 1, step):
                patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                color, confidence = self.colorize_patch(patch, y_pos, x_pos)
                
                # Apply Gaussian-weighted color
                weight = gaussian_window * confidence
                
                output[y:y+self.patch_size, x:x+self.patch_size] += (
                    weight[:, :, np.newaxis] * color
                )
                weights[y:y+self.patch_size, x:x+self.patch_size] += weight
        
        # Handle edges that might not be covered
        weights = np.maximum(weights, 1e-6)
        output = output / weights[:, :, np.newaxis]
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def colorize_with_luminance_preservation(self, grayscale: np.ndarray) -> np.ndarray:
        """
        Colorize and preserve original luminance.
        
        This often improves results by keeping the grayscale structure
        and only adding chroma.
        """
        # Get colorized result
        colorized = self.colorize(grayscale)
        
        # Convert to LAB-like space (simplified)
        # Keep original luminance, use colorized for chroma
        colorized_gray = (0.299 * colorized[:,:,0] + 
                         0.587 * colorized[:,:,1] + 
                         0.114 * colorized[:,:,2])
        
        # Scale colorized to match original luminance
        scale = (grayscale.astype(np.float32) + 1) / (colorized_gray + 1)
        scale = np.clip(scale, 0.5, 2.0)  # Limit scaling
        
        result = colorized.astype(np.float32) * scale[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)


def load_coco_images(n_images: int, start_idx: int = 0) -> List[Tuple[str, np.ndarray]]:
    """Load COCO images."""
    image_files = sorted(COCO_PATH.glob("*.jpg"))
    images = []
    
    for img_path in image_files[start_idx:start_idx + n_images]:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append((img_path.stem, img))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return images


def run_improved_test(n_train: int = 200, n_test: int = 8, sample_rate: float = 0.12):
    """Run the improved colorization test."""
    print("=" * 70)
    print("IMPROVED GEOMETRIC COLORIZATION")
    print("=" * 70)
    
    colorizer = ImprovedColorizer(n_feature_dims=16, patch_size=16)
    
    # Load training images
    print(f"\n1. LOADING {n_train} TRAINING IMAGES")
    print("-" * 50)
    train_images = load_coco_images(n_train, start_idx=0)
    print(f"   Loaded {len(train_images)} images")
    
    # Bootstrap the drum
    print(f"\n2. BOOTSTRAPPING DRUM")
    print("-" * 50)
    print(f"   Features: 16 dimensions (vs 8 in baseline)")
    print(f"   Multi-scale: Yes (1.0x and 0.5x)")
    print(f"   Sample rate: {sample_rate}")
    
    total_patches = 0
    for i, (name, img) in enumerate(train_images):
        patches = colorizer.learn_from_image(img, sample_rate=sample_rate)
        total_patches += patches
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(train_images)}, total patches: {total_patches}")
    
    print(f"\n   Final drum size: {len(colorizer.drum)} points")
    print(f"   From {len(train_images)} images")
    
    # Build spatial index for fast queries
    print(f"   Building KD-tree index...")
    colorizer.drum.build_index()
    print(f"   Index built!")
    
    # Load test images
    print(f"\n3. LOADING {n_test} TEST IMAGES")
    print("-" * 50)
    test_images = load_coco_images(n_test, start_idx=n_train + 200)
    print(f"   Loaded {len(test_images)} test images")
    
    # Colorize test images
    print(f"\n4. COLORIZING TEST IMAGES")
    print("-" * 50)
    print(f"   Overlap: 50% with Gaussian blending")
    print(f"   Luminance preservation: Yes")
    
    results = []
    for name, color_img in test_images:
        grayscale = (0.299 * color_img[:,:,0] + 
                     0.587 * color_img[:,:,1] + 
                     0.114 * color_img[:,:,2]).astype(np.uint8)
        
        # Standard colorization
        colorized = colorizer.colorize(grayscale, overlap=0.5)
        
        # With luminance preservation
        colorized_lum = colorizer.colorize_with_luminance_preservation(grayscale)
        
        # Compute errors
        error_std = np.abs(colorized.astype(float) - color_img.astype(float)).mean()
        error_lum = np.abs(colorized_lum.astype(float) - color_img.astype(float)).mean()
        
        results.append({
            'name': name,
            'original': color_img,
            'grayscale': grayscale,
            'colorized': colorized,
            'colorized_lum': colorized_lum,
            'error_std': error_std,
            'error_lum': error_lum
        })
        
        print(f"   {name}: MAE_std={error_std:.2f}, MAE_lum={error_lum:.2f}")
    
    # Create visualization
    print(f"\n5. CREATING VISUALIZATION")
    print("-" * 50)
    
    fig = plt.figure(figsize=(20, 4 * len(results)))
    gs = gridspec.GridSpec(len(results), 5, figure=fig, hspace=0.3, wspace=0.1)
    
    for i, r in enumerate(results):
        # Original
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(r['original'])
        ax.set_title('Original' if i == 0 else '')
        ax.axis('off')
        
        # Grayscale
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(r['grayscale'], cmap='gray')
        ax.set_title('Grayscale' if i == 0 else '')
        ax.axis('off')
        
        # Standard colorized
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(r['colorized'])
        ax.set_title(f'Colorized (MAE={r["error_std"]:.1f})' if i == 0 else f'MAE={r["error_std"]:.1f}')
        ax.axis('off')
        
        # Luminance preserved
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(r['colorized_lum'])
        ax.set_title(f'+ Lum Preserve (MAE={r["error_lum"]:.1f})' if i == 0 else f'MAE={r["error_lum"]:.1f}')
        ax.axis('off')
        
        # Error map
        ax = fig.add_subplot(gs[i, 4])
        diff = np.abs(r['colorized_lum'].astype(float) - r['original'].astype(float)).mean(axis=2)
        ax.imshow(diff, cmap='hot', vmin=0, vmax=80)
        ax.set_title('Error' if i == 0 else '')
        ax.axis('off')
    
    fig.suptitle(f'Improved Geometric Colorization: {n_train} images, {len(colorizer.drum)} points, 16D features',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "improved_colorization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved to: {output_file}")
    
    # Summary
    print(f"\n6. SUMMARY")
    print("-" * 50)
    avg_std = np.mean([r['error_std'] for r in results])
    avg_lum = np.mean([r['error_lum'] for r in results])
    print(f"   Training images: {n_train}")
    print(f"   Drum size: {len(colorizer.drum)} points")
    print(f"   Average MAE (standard): {avg_std:.2f}")
    print(f"   Average MAE (lum preserve): {avg_lum:.2f}")
    
    # Compare to baseline
    print(f"\n   Improvements over baseline:")
    print(f"   - 4x more training images (200 vs 50)")
    print(f"   - 2x more features (16D vs 8D)")
    print(f"   - Multi-scale patches")
    print(f"   - Gaussian blending (no blocky artifacts)")
    print(f"   - Luminance preservation option")
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_improved_test(
        n_train=200,
        n_test=8,
        sample_rate=0.12
    )
    
    print("\n" + "=" * 70)
    print("IMPROVED TEST COMPLETE")
    print("=" * 70)
