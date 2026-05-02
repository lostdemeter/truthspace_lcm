#!/usr/bin/env python3
"""
Focused Colorizer - Finding the Right Dimensional Focus

The insight: Maybe we have too many dimensions, or the wrong weighting.
Like a camera lens, we need to find the right "focus" - the dimensions
that matter most for the universal path.

Approaches:
1. Vary number of dimensions (4, 8, 16, 32)
2. Use SVD to find principal dimensions
3. φ-scale the dimension weights (like LOD)
4. Learn which dimensions to "focus on"

The hypothesis: The universal path exists in a LOWER dimensional
subspace. By finding that subspace, we find the path.

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from typing import List, Tuple, Dict, Optional
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


class FocusedColorizer:
    """
    Colorizer with adjustable dimensional focus.
    
    Key ideas:
    1. Extract many features (32 raw)
    2. Use SVD to find principal components
    3. Weight dimensions by φ^(-level) for focus control
    4. Vary the "focus level" to find optimal path
    """
    
    def __init__(self, patch_size: int = 16, focus_dims: int = 8):
        self.patch_size = patch_size
        self.focus_dims = focus_dims  # How many dimensions to use
        
        # Raw feature dimension (before focusing)
        self.raw_dims = 32
        
        # SVD components for dimensionality reduction
        self.svd_components = None  # V matrix from SVD
        self.svd_mean = None
        self.singular_values = None
        
        # φ-scaled focus weights
        self.focus_weights = None
        
        # Training data
        self.training_features = []
        self.training_u = []
        self.training_v = []
        
        # Path clusters in focused space
        self.path_clusters = None
        self.n_clusters = 0
        
        self.n_images = 0
    
    def extract_raw_features(self, gray_patch: np.ndarray, 
                              y_pos: float, x_pos: float) -> np.ndarray:
        """Extract 32 raw features before focusing."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        features = []
        
        # Basic statistics (4)
        features.append(patch.mean())  # luminance
        features.append(patch.std())   # contrast
        features.append(patch.min())   # min
        features.append(patch.max())   # max
        
        # Position (2)
        features.append(y_pos)
        features.append(x_pos)
        
        # Texture in different directions (4)
        if h > 1 and w > 1:
            features.append(np.abs(np.diff(patch, axis=1)).mean())  # horizontal
            features.append(np.abs(np.diff(patch, axis=0)).mean())  # vertical
            features.append(np.abs(np.diff(patch, axis=1)).std())   # h variance
            features.append(np.abs(np.diff(patch, axis=0)).std())   # v variance
        else:
            features.extend([0, 0, 0, 0])
        
        # Gradients (4)
        if h > 2 and w > 2:
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            features.append(np.sqrt(gx**2 + gy**2).mean())  # magnitude
            features.append(np.arctan2(gy.mean(), gx.mean()) / np.pi)  # direction
            features.append(gx.std())
            features.append(gy.std())
        else:
            features.extend([0, 0, 0, 0])
        
        # Multi-scale texture (6)
        if h >= 4 and w >= 4:
            # Coarse scale
            coarse = patch.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            features.append(coarse.mean())
            features.append(coarse.std())
            features.append(np.abs(np.diff(coarse, axis=1)).mean())
            
            # Fine scale variance
            fine_var = patch.reshape(h//2, 2, w//2, 2).std(axis=(1, 3))
            features.append(fine_var.mean())
            features.append(fine_var.std())
            features.append(fine_var.max())
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Histogram features (8)
        hist, _ = np.histogram(patch.flatten(), bins=8, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        features.extend(hist.tolist())
        
        # Edge features (4)
        if h > 2 and w > 2:
            center = patch[1:-1, 1:-1]
            neighbors = (patch[:-2, 1:-1] + patch[2:, 1:-1] + 
                        patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
            edge = np.abs(center - neighbors)
            features.append(edge.mean())
            features.append(edge.std())
            features.append(edge.max())
            features.append((edge > 0.1).mean())  # edge density
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features[:self.raw_dims], dtype=np.float32)
    
    def collect_training_data(self, color_image: np.ndarray, sample_rate: float = 0.15) -> int:
        """Collect raw features and colors for SVD and training."""
        H, W = color_image.shape[:2]
        
        grayscale = (0.299 * color_image[:,:,0] + 
                     0.587 * color_image[:,:,1] + 
                     0.114 * color_image[:,:,2]).astype(np.uint8)
        
        yuv = rgb_to_yuv(color_image)
        collected = 0
        
        for y in range(0, H - self.patch_size, self.patch_size):
            for x in range(0, W - self.patch_size, self.patch_size):
                if np.random.random() > sample_rate:
                    continue
                
                gray_patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                features = self.extract_raw_features(gray_patch, y_pos, x_pos)
                mean_u = yuv_patch[:, :, 1].mean()
                mean_v = yuv_patch[:, :, 2].mean()
                
                self.training_features.append(features)
                self.training_u.append(mean_u)
                self.training_v.append(mean_v)
                collected += 1
        
        self.n_images += 1
        return collected
    
    def learn_focus(self):
        """
        Learn the dimensional focus using SVD.
        
        This finds the principal components of the feature space
        and weights them by φ^(-level) for focus control.
        """
        if len(self.training_features) < 100:
            return
        
        X = np.array(self.training_features)
        u = np.array(self.training_u)
        v = np.array(self.training_v)
        
        # Center the data
        self.svd_mean = X.mean(axis=0)
        X_centered = X - self.svd_mean
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.svd_components = Vt[:self.focus_dims]  # Top k components
        self.singular_values = S[:self.focus_dims]
        
        # φ-scaled focus weights
        # Higher singular values get more weight, scaled by φ
        self.focus_weights = np.zeros(self.focus_dims)
        for i in range(self.focus_dims):
            # Weight = S[i] * φ^(-i/2) - combines importance with φ-scaling
            self.focus_weights[i] = self.singular_values[i] * (PHI ** (-i / 2))
        
        # Normalize
        self.focus_weights /= self.focus_weights.sum()
        
        # Compute variance explained
        total_var = (S ** 2).sum()
        explained_var = (self.singular_values ** 2).sum() / total_var
        
        print(f"   SVD focus learned:")
        print(f"     Raw dimensions: {self.raw_dims}")
        print(f"     Focused dimensions: {self.focus_dims}")
        print(f"     Variance explained: {explained_var:.1%}")
        print(f"     Top singular values: {self.singular_values[:5].round(2)}")
        print(f"     φ-scaled weights: {self.focus_weights[:5].round(3)}")
        
        # Now find correlations between focused features and color
        X_focused = self.focus_features(X)
        
        u_corr = np.zeros(self.focus_dims)
        v_corr = np.zeros(self.focus_dims)
        for d in range(self.focus_dims):
            if X_focused[:, d].std() > 1e-6:
                u_corr[d] = np.corrcoef(X_focused[:, d], u)[0, 1]
                v_corr[d] = np.corrcoef(X_focused[:, d], v)[0, 1]
        
        u_corr = np.nan_to_num(u_corr)
        v_corr = np.nan_to_num(v_corr)
        
        print(f"\n   Color correlations in focused space:")
        print(f"     U correlations: {u_corr[:5].round(3)}")
        print(f"     V correlations: {v_corr[:5].round(3)}")
        
        # Compute R² for linear prediction in focused space
        X_bias = np.hstack([X_focused, np.ones((X_focused.shape[0], 1))])
        
        weights_u = np.linalg.lstsq(X_bias, u, rcond=None)[0]
        weights_v = np.linalg.lstsq(X_bias, v, rcond=None)[0]
        
        u_pred = X_bias @ weights_u
        v_pred = X_bias @ weights_v
        
        r2_u = 1 - np.sum((u - u_pred)**2) / (np.sum((u - u.mean())**2) + 1e-10)
        r2_v = 1 - np.sum((v - v_pred)**2) / (np.sum((v - v.mean())**2) + 1e-10)
        
        print(f"\n   Linear R² in focused space:")
        print(f"     R² for U: {r2_u:.3f}")
        print(f"     R² for V: {r2_v:.3f}")
        
        return explained_var, r2_u, r2_v
    
    def focus_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Project raw features into focused space."""
        if self.svd_components is None:
            return raw_features[:, :self.focus_dims]
        
        centered = raw_features - self.svd_mean
        focused = centered @ self.svd_components.T
        
        # Apply φ-scaled weights
        focused = focused * self.focus_weights
        
        return focused
    
    def find_path_clusters(self, n_clusters: int = 20):
        """Find clusters in focused space."""
        if len(self.training_features) < n_clusters:
            return
        
        X = np.array(self.training_features)
        u = np.array(self.training_u)
        v = np.array(self.training_v)
        
        X_focused = self.focus_features(X)
        
        # Joint space: focused features + color
        joint = np.hstack([X_focused, u.reshape(-1, 1), v.reshape(-1, 1)])
        
        # K-means clustering
        indices = np.random.choice(len(joint), n_clusters, replace=False)
        centers = joint[indices].copy()
        
        for iteration in range(20):
            assignments = np.argmin(
                np.linalg.norm(joint[:, np.newaxis] - centers, axis=2), 
                axis=1
            )
            
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers[k] = joint[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]
            
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers
        
        self.path_clusters = centers
        self.n_clusters = n_clusters
        
        print(f"\n   Found {n_clusters} clusters in focused space")
    
    def predict_color(self, raw_features: np.ndarray, k: int = 5) -> Tuple[float, float]:
        """Predict color using focused features and clusters."""
        if self.path_clusters is None:
            return 0.0, 0.0
        
        focused = self.focus_features(raw_features.reshape(1, -1))[0]
        
        cluster_features = self.path_clusters[:, :-2]
        distances = np.linalg.norm(cluster_features - focused, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        
        total_weight = 0
        weighted_u = 0
        weighted_v = 0
        
        for idx in nearest_idx:
            dist = distances[idx]
            weight = 1.0 / (dist**2 + 0.001)
            
            u = self.path_clusters[idx, -2]
            v = self.path_clusters[idx, -1]
            
            weighted_u += weight * u
            weighted_v += weight * v
            total_weight += weight
        
        return weighted_u / total_weight, weighted_v / total_weight
    
    def colorize_sharp(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize using focused features."""
        H, W = grayscale.shape
        
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
                
                features = self.extract_raw_features(patch, y_pos, x_pos)
                u, v = self.predict_color(features)
                u_map[py, px] = u
                v_map[py, px] = v
        
        scale_y = H / n_patches_y
        scale_x = W / n_patches_x
        u_full = zoom(u_map, (scale_y, scale_x), order=1)
        v_full = zoom(v_map, (scale_y, scale_x), order=1)
        
        u_full = u_full[:H, :W]
        v_full = v_full[:H, :W]
        
        if u_full.shape[0] < H or u_full.shape[1] < W:
            u_padded = np.zeros((H, W), dtype=np.float32)
            v_padded = np.zeros((H, W), dtype=np.float32)
            u_padded[:u_full.shape[0], :u_full.shape[1]] = u_full
            v_padded[:v_full.shape[0], :v_full.shape[1]] = v_full
            u_full, v_full = u_padded, v_padded
        
        y_channel = grayscale.astype(np.float32) / 255.0
        yuv = np.stack([y_channel, u_full, v_full], axis=-1)
        
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


def test_focus_level(focus_dims: int, train_images, test_images, new_images):
    """Test a specific focus level."""
    colorizer = FocusedColorizer(patch_size=16, focus_dims=focus_dims)
    
    # Collect training data
    for name, img in train_images:
        colorizer.collect_training_data(img, sample_rate=0.12)
    
    # Learn focus
    explained_var, r2_u, r2_v = colorizer.learn_focus()
    
    # Find clusters
    colorizer.find_path_clusters(n_clusters=25)
    
    # Test on test images
    test_errors = []
    for name, img in test_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize_sharp(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        test_errors.append(error)
    
    # Test on new images (generalization)
    gen_errors = []
    for name, img in new_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize_sharp(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        gen_errors.append(error)
    
    return {
        'focus_dims': focus_dims,
        'explained_var': explained_var,
        'r2_u': r2_u,
        'r2_v': r2_v,
        'test_mae': np.mean(test_errors),
        'gen_mae': np.mean(gen_errors),
        'colorizer': colorizer
    }


def run_focus_experiment():
    """Test different focus levels to find optimal dimensionality."""
    print("=" * 70)
    print("FOCUSED COLORIZER - Finding Optimal Dimensionality")
    print("=" * 70)
    
    # Load images
    print("\n1. LOADING IMAGES")
    print("-" * 50)
    train_images = load_coco_images(100, start_idx=0)
    test_images = load_coco_images(5, start_idx=200)
    new_images = load_coco_images(5, start_idx=300)
    print(f"   Train: {len(train_images)}, Test: {len(test_images)}, New: {len(new_images)}")
    
    # Test different focus levels
    print("\n2. TESTING FOCUS LEVELS")
    print("-" * 50)
    
    focus_levels = [4, 6, 8, 12, 16, 24]
    results = []
    
    for focus_dims in focus_levels:
        print(f"\n   === Focus dims: {focus_dims} ===")
        result = test_focus_level(focus_dims, train_images, test_images, new_images)
        results.append(result)
        print(f"   Test MAE: {result['test_mae']:.2f}, Gen MAE: {result['gen_mae']:.2f}")
    
    # Find best
    print("\n3. RESULTS SUMMARY")
    print("-" * 50)
    print(f"   {'Dims':>6} {'Var%':>8} {'R²_U':>8} {'R²_V':>8} {'Test':>8} {'Gen':>8} {'Gap':>8}")
    print("   " + "-" * 56)
    
    best_gen = min(results, key=lambda r: r['gen_mae'])
    
    for r in results:
        gap = r['gen_mae'] - r['test_mae']
        marker = " *" if r == best_gen else ""
        print(f"   {r['focus_dims']:>6} {r['explained_var']:>7.1%} {r['r2_u']:>8.3f} "
              f"{r['r2_v']:>8.3f} {r['test_mae']:>8.2f} {r['gen_mae']:>8.2f} {gap:>+8.2f}{marker}")
    
    print(f"\n   Best generalization: {best_gen['focus_dims']} dimensions (MAE={best_gen['gen_mae']:.2f})")
    
    # Visualize best result
    print("\n4. VISUALIZING BEST RESULT")
    print("-" * 50)
    
    colorizer = best_gen['colorizer']
    
    vis_results = []
    for name, img in test_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize_sharp(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        vis_results.append((name, img, gray, colorized, error))
    
    fig, axes = plt.subplots(len(vis_results), 4, figsize=(16, 4 * len(vis_results)))
    
    for i, (name, original, gray, colorized, error) in enumerate(vis_results):
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Focused (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Focused Colorizer: {best_gen["focus_dims"]}D, '
                 f'Test={best_gen["test_mae"]:.1f}, Gen={best_gen["gen_mae"]:.1f}',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "focused_colorization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return results, best_gen


if __name__ == "__main__":
    results, best = run_focus_experiment()
    
    print("\n" + "=" * 70)
    print("FOCUS EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"""
   The insight: Find the RIGHT dimensional focus for the universal path.
   
   Key findings:
   - Best focus: {best['focus_dims']} dimensions
   - Variance explained: {best['explained_var']:.1%}
   - R² for color: U={best['r2_u']:.3f}, V={best['r2_v']:.3f}
   - Test MAE: {best['test_mae']:.2f}
   - Generalization MAE: {best['gen_mae']:.2f}
   
   The focus level affects:
   - Too few dims: Not enough information
   - Too many dims: Overfitting to training data
   - Just right: Best generalization
   
   Like a camera lens:
   - Out of focus: blurry (wrong dims)
   - In focus: sharp (right dims)
   
   The universal path exists in a specific dimensional subspace.
""")
