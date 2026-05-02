#!/usr/bin/env python3
"""
Real Colorization Test - Using COCO val2017 Images

Test the drum bootstrapping approach with real photographs.

Process:
1. Load N color images from COCO val2017
2. Bootstrap the drum from these images
3. Test colorization on held-out images
4. Compare original vs colorized

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')
from phi_space import PhiSpace

PHI = (1 + np.sqrt(5)) / 2

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


class RealDrumBootstrapper:
    """Bootstrapper optimized for real images."""
    
    def __init__(self, n_feature_dims: int = 8, patch_size: int = 16):
        self.n_dims = n_feature_dims
        self.patch_size = patch_size
        self.drum = PhiSpace(
            dims=n_feature_dims,
            dim_names=[
                'luminance', 'contrast', 'texture_h', 'texture_v',
                'y_position', 'x_position', 'edge_density', 'smoothness'
            ]
        )
        self.n_images = 0
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract geometric features from grayscale patch."""
        patch = gray_patch.astype(np.float32) / 255.0
        
        luminance = patch.mean()
        contrast = patch.std()
        
        if patch.shape[0] > 1 and patch.shape[1] > 1:
            texture_h = np.abs(np.diff(patch, axis=1)).mean()
            texture_v = np.abs(np.diff(patch, axis=0)).mean()
        else:
            texture_h = texture_v = 0.0
        
        if patch.shape[0] > 2 and patch.shape[1] > 2:
            center = patch[1:-1, 1:-1]
            neighbors = (patch[:-2, 1:-1] + patch[2:, 1:-1] + 
                        patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
            edge_density = np.abs(center - neighbors).mean()
        else:
            edge_density = 0.0
        
        total_var = texture_h + texture_v
        smoothness = 1.0 / (1.0 + total_var * 10)
        
        return np.array([
            luminance, contrast, texture_h, texture_v,
            y_pos, x_pos, edge_density, smoothness
        ], dtype=np.float32)
    
    def learn_from_image(self, color_image: np.ndarray, sample_rate: float = 0.2) -> int:
        """Learn from a single color image."""
        H, W = color_image.shape[:2]
        
        grayscale = (0.299 * color_image[:,:,0] + 
                     0.587 * color_image[:,:,1] + 
                     0.114 * color_image[:,:,2]).astype(np.uint8)
        
        patches_learned = 0
        
        for y in range(0, H - self.patch_size, self.patch_size):
            for x in range(0, W - self.patch_size, self.patch_size):
                if np.random.random() > sample_rate:
                    continue
                
                gray_patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                color_patch = color_image[y:y+self.patch_size, x:x+self.patch_size]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                features = self.extract_features(gray_patch, y_pos, x_pos)
                mean_color = color_patch.mean(axis=(0, 1)).astype(np.uint8)
                
                point_id = f"p_{self.n_images}_{patches_learned}"
                self.drum.add(point_id, features, metadata={'rgb': tuple(mean_color)})
                patches_learned += 1
        
        self.n_images += 1
        return patches_learned
    
    def colorize_patch(self, gray_patch: np.ndarray, 
                       y_pos: float, x_pos: float, k: int = 5) -> np.ndarray:
        """Colorize a single patch."""
        features = self.extract_features(gray_patch, y_pos, x_pos)
        nearest = self.drum.query(features, k=k)
        
        if not nearest:
            return np.array([128, 128, 128], dtype=np.uint8)
        
        total_weight = 0
        weighted_color = np.zeros(3, dtype=np.float32)
        
        for point_id, distance in nearest:
            point = self.drum[point_id]
            rgb = np.array(point.metadata['rgb'], dtype=np.float32)
            weight = 1.0 / (distance + 0.01)
            weighted_color += weight * rgb
            total_weight += weight
        
        return (weighted_color / total_weight).astype(np.uint8)
    
    def colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize a grayscale image."""
        H, W = grayscale.shape
        output = np.zeros((H, W, 3), dtype=np.uint8)
        
        for y in range(0, H, self.patch_size):
            for x in range(0, W, self.patch_size):
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                
                patch = grayscale[y:y_end, x:x_end]
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                color = self.colorize_patch(patch, y_pos, x_pos)
                output[y:y_end, x:x_end] = color
        
        return output


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


def run_real_test(n_train: int = 50, n_test: int = 5, sample_rate: float = 0.15):
    """Run the real colorization test."""
    print("=" * 70)
    print("REAL COLORIZATION TEST WITH COCO IMAGES")
    print("=" * 70)
    
    # Create bootstrapper
    bootstrapper = RealDrumBootstrapper(patch_size=16)
    
    # Load training images
    print(f"\n1. LOADING {n_train} TRAINING IMAGES")
    print("-" * 50)
    train_images = load_coco_images(n_train, start_idx=0)
    print(f"   Loaded {len(train_images)} images")
    
    # Bootstrap the drum
    print(f"\n2. BOOTSTRAPPING DRUM (sample_rate={sample_rate})")
    print("-" * 50)
    
    total_patches = 0
    for i, (name, img) in enumerate(train_images):
        patches = bootstrapper.learn_from_image(img, sample_rate=sample_rate)
        total_patches += patches
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(train_images)}, total patches: {total_patches}")
    
    print(f"\n   Final drum size: {len(bootstrapper.drum)} points")
    print(f"   From {len(train_images)} images")
    
    # Load test images (different from training)
    print(f"\n3. LOADING {n_test} TEST IMAGES")
    print("-" * 50)
    test_images = load_coco_images(n_test, start_idx=n_train + 100)  # Skip ahead
    print(f"   Loaded {len(test_images)} test images")
    
    # Colorize test images
    print(f"\n4. COLORIZING TEST IMAGES")
    print("-" * 50)
    
    results = []
    for name, color_img in test_images:
        # Convert to grayscale
        grayscale = (0.299 * color_img[:,:,0] + 
                     0.587 * color_img[:,:,1] + 
                     0.114 * color_img[:,:,2]).astype(np.uint8)
        
        # Colorize
        colorized = bootstrapper.colorize(grayscale)
        
        # Compute error
        error = np.abs(colorized.astype(float) - color_img.astype(float)).mean()
        
        results.append({
            'name': name,
            'original': color_img,
            'grayscale': grayscale,
            'colorized': colorized,
            'error': error
        })
        
        print(f"   {name}: MAE = {error:.2f}")
    
    # Create visualization
    print(f"\n5. CREATING VISUALIZATION")
    print("-" * 50)
    
    fig = plt.figure(figsize=(16, 4 * len(results)))
    gs = gridspec.GridSpec(len(results), 4, figure=fig, hspace=0.3, wspace=0.1)
    
    for i, r in enumerate(results):
        # Original
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(r['original'])
        ax.set_title('Original' if i == 0 else '')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(r['name'][-4:], fontsize=10)
        
        # Grayscale
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(r['grayscale'], cmap='gray')
        ax.set_title('Grayscale' if i == 0 else '')
        ax.axis('off')
        
        # Colorized
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(r['colorized'])
        ax.set_title('Colorized (Ours)' if i == 0 else '')
        ax.axis('off')
        
        # Difference
        ax = fig.add_subplot(gs[i, 3])
        diff = np.abs(r['colorized'].astype(float) - r['original'].astype(float)).mean(axis=2)
        ax.imshow(diff, cmap='hot', vmin=0, vmax=100)
        ax.set_title(f'Error (MAE={r["error"]:.1f})' if i == 0 else f'MAE={r["error"]:.1f}')
        ax.axis('off')
    
    # Add summary
    fig.suptitle(f'Geometric Colorization: {n_train} training images, {len(bootstrapper.drum)} drum points',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "real_colorization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved to: {output_file}")
    
    # Summary statistics
    print(f"\n6. SUMMARY")
    print("-" * 50)
    avg_error = np.mean([r['error'] for r in results])
    print(f"   Training images: {n_train}")
    print(f"   Drum size: {len(bootstrapper.drum)} points")
    print(f"   Average MAE: {avg_error:.2f}")
    print(f"\n   Note: This is a NAIVE baseline with simple features.")
    print(f"   Better features would significantly improve results.")
    
    return bootstrapper, results


if __name__ == "__main__":
    bootstrapper, results = run_real_test(
        n_train=50,      # Use 50 images for training
        n_test=5,        # Test on 5 held-out images
        sample_rate=0.15  # Sample 15% of patches
    )
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"""
   We trained on {bootstrapper.n_images} real COCO images.
   The drum contains {len(bootstrapper.drum)} learned color points.
   
   This is a proof-of-concept showing:
   1. Real images can bootstrap the drum
   2. Colorization works via geometric nearest-neighbor
   3. No neural network training required
   
   To improve:
   - Better features (texture, edges, context)
   - More training images
   - Attractor/repeller dynamics to organize drum
   - Multi-scale patches
""")
