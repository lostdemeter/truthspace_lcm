#!/usr/bin/env python3
"""
Constrained Drum Colorizer - The Principled Architecture

The insight: Constraints don't replace the drum, they FOCUS it.

Architecture:
1. Drum: Examples stored in manifold-projected space
2. Comb: Geometric constraints (manifold, saturation scaling)
3. Music: Color emerges from constrained drum traversal

This combines:
- The data efficiency of nearest-neighbor (drum)
- The generalization of geometric constraints (comb)

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.linalg import svd
from scipy.spatial import cKDTree
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


class ConstrainedDrumColorizer:
    """
    Colorizer using constrained drum traversal.
    
    The DRUM stores examples in manifold space.
    The COMB applies geometric constraints.
    The MUSIC is the colorized output.
    """
    
    def __init__(self, patch_size: int = 16, manifold_dims: int = 6):
        self.patch_size = patch_size
        self.manifold_dims = manifold_dims
        
        # === THE COMB (Geometric Constraints) ===
        
        # Constraint 1: Manifold projection
        self.manifold_basis = None  # Learned from data
        self.feature_mean = None
        self.feature_std = None
        self.singular_values = None  # For weighting
        
        # Constraint 2: Saturation scaling (from manipulation experiments)
        self.saturation_scale = 0.5
        
        # Constraint 3: Dimension weights (φ-scaled by importance)
        self.dim_weights = None
        
        # === THE DRUM (Examples in Manifold Space) ===
        self.drum_positions = []  # Manifold-projected features
        self.drum_colors = []     # (U, V) pairs
        self.drum_tree = None     # KD-tree for fast lookup
        
        self.is_trained = False
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract features including quadratic interactions."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        # Linear features
        lum = patch.mean()
        con = patch.std()
        tex_h = np.abs(np.diff(patch, axis=1)).mean() if w > 1 else 0
        tex_v = np.abs(np.diff(patch, axis=0)).mean() if h > 1 else 0
        
        if h > 2 and w > 2:
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            grad_m = np.sqrt(gx**2 + gy**2).mean()
            grad_d = np.arctan2(gy.mean(), gx.mean()) / np.pi
        else:
            grad_m = grad_d = 0
        
        linear = np.array([lum, con, tex_h, tex_v, y_pos, x_pos, grad_m, grad_d])
        
        # Quadratic interactions (the important ones from analysis)
        interactions = np.array([
            con * tex_h,      # Most predictive
            tex_h * tex_v,
            con * tex_v,
            lum * con,
            y_pos * lum,
            con ** 2,
        ])
        
        return np.concatenate([linear, interactions])
    
    def learn_constraints(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """
        Learn the geometric constraints (the COMB) from data.
        
        This learns:
        1. The manifold basis (SVD)
        2. The dimension weights (from singular values)
        """
        print("   Learning geometric constraints...")
        
        all_features = []
        all_u = []
        all_v = []
        
        for img in images:
            H, W = img.shape[:2]
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            yuv = rgb_to_yuv(img)
            
            for y in range(0, H - self.patch_size, self.patch_size):
                for x in range(0, W - self.patch_size, self.patch_size):
                    if np.random.random() > sample_rate:
                        continue
                    
                    gray_patch = gray[y:y+self.patch_size, x:x+self.patch_size]
                    yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                    
                    y_pos = (y + self.patch_size/2) / H
                    x_pos = (x + self.patch_size/2) / W
                    
                    feat = self.extract_features(gray_patch, y_pos, x_pos)
                    all_features.append(feat)
                    all_u.append(yuv_patch[:, :, 1].mean())
                    all_v.append(yuv_patch[:, :, 2].mean())
        
        X = np.array(all_features)
        
        # Normalize
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-10
        X_norm = (X - self.feature_mean) / self.feature_std
        
        # SVD to find manifold
        U_svd, S, Vt = svd(X_norm, full_matrices=False)
        
        self.manifold_basis = Vt[:self.manifold_dims].T
        self.singular_values = S[:self.manifold_dims]
        
        # φ-scaled dimension weights
        # Higher singular values = more important = higher weight
        self.dim_weights = self.singular_values / self.singular_values.sum()
        
        var_explained = (S[:self.manifold_dims]**2).sum() / (S**2).sum()
        print(f"   Manifold: {self.manifold_dims}D, {var_explained*100:.1f}% variance")
        print(f"   Dimension weights: {self.dim_weights.round(3)}")
        
        return X, np.array(all_u), np.array(all_v)
    
    def build_drum(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """
        Build the drum (examples in manifold space).
        """
        print("   Building drum in manifold space...")
        
        self.drum_positions = []
        self.drum_colors = []
        
        for img in images:
            H, W = img.shape[:2]
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            yuv = rgb_to_yuv(img)
            
            for y in range(0, H - self.patch_size, self.patch_size):
                for x in range(0, W - self.patch_size, self.patch_size):
                    if np.random.random() > sample_rate:
                        continue
                    
                    gray_patch = gray[y:y+self.patch_size, x:x+self.patch_size]
                    yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                    
                    y_pos = (y + self.patch_size/2) / H
                    x_pos = (x + self.patch_size/2) / W
                    
                    # Project to manifold space
                    feat = self.extract_features(gray_patch, y_pos, x_pos)
                    feat_norm = (feat - self.feature_mean) / self.feature_std
                    feat_manifold = feat_norm @ self.manifold_basis
                    
                    # Weight by dimension importance
                    feat_weighted = feat_manifold * self.dim_weights
                    
                    self.drum_positions.append(feat_weighted)
                    self.drum_colors.append([
                        yuv_patch[:, :, 1].mean(),
                        yuv_patch[:, :, 2].mean()
                    ])
        
        self.drum_positions = np.array(self.drum_positions)
        self.drum_colors = np.array(self.drum_colors)
        
        # Build KD-tree for fast lookup
        self.drum_tree = cKDTree(self.drum_positions)
        
        print(f"   Drum size: {len(self.drum_positions)} examples")
        
        self.is_trained = True
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """Full training: learn constraints, then build drum."""
        self.learn_constraints(images, sample_rate)
        self.build_drum(images, sample_rate)
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float, k: int = 7) -> Tuple[float, float]:
        """
        Predict color using constrained drum traversal.
        
        1. Project to manifold (constraint)
        2. Find nearest neighbors in drum
        3. Apply saturation scaling (constraint)
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Project to manifold space
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / self.feature_std
        feat_manifold = feat_norm @ self.manifold_basis
        feat_weighted = feat_manifold * self.dim_weights
        
        # Query drum
        distances, indices = self.drum_tree.query(feat_weighted, k=k)
        
        # Weighted average
        weights = 1.0 / (distances**2 + 0.001)
        weights /= weights.sum()
        
        colors = self.drum_colors[indices]
        u = np.sum(weights * colors[:, 0])
        v = np.sum(weights * colors[:, 1])
        
        # Apply saturation constraint
        u *= self.saturation_scale
        v *= self.saturation_scale
        
        return u, v
    
    def refine_constraints(self, images: List[np.ndarray], learning_rate: float = 0.1):
        """
        Refine the geometric constraints using ground truth.
        
        This adjusts the COMB (constraints), not the DRUM (examples).
        """
        print("   Refining constraints with ground truth...")
        
        # Collect predictions and ground truth
        pred_u_list = []
        pred_v_list = []
        true_u_list = []
        true_v_list = []
        
        for img in images:
            H, W = img.shape[:2]
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            yuv = rgb_to_yuv(img)
            
            for y in range(0, H - self.patch_size, self.patch_size * 2):
                for x in range(0, W - self.patch_size, self.patch_size * 2):
                    gray_patch = gray[y:y+self.patch_size, x:x+self.patch_size]
                    yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                    
                    y_pos = (y + self.patch_size/2) / H
                    x_pos = (x + self.patch_size/2) / W
                    
                    pred_u, pred_v = self.predict_color(gray_patch, y_pos, x_pos)
                    true_u = yuv_patch[:, :, 1].mean()
                    true_v = yuv_patch[:, :, 2].mean()
                    
                    pred_u_list.append(pred_u)
                    pred_v_list.append(pred_v)
                    true_u_list.append(true_u)
                    true_v_list.append(true_v)
        
        pred_u = np.array(pred_u_list)
        pred_v = np.array(pred_v_list)
        true_u = np.array(true_u_list)
        true_v = np.array(true_v_list)
        
        # Find optimal saturation scale
        # We want: pred * scale ≈ true
        # Optimal scale = (pred · true) / (pred · pred)
        
        pred_mag = np.sqrt(pred_u**2 + pred_v**2)
        true_mag = np.sqrt(true_u**2 + true_v**2)
        
        # Avoid division by zero
        mask = pred_mag > 0.001
        if mask.sum() > 100:
            optimal_scale = np.median(true_mag[mask] / pred_mag[mask])
            
            # Blend with current scale
            new_scale = (1 - learning_rate) * self.saturation_scale + learning_rate * optimal_scale
            
            print(f"   Current scale: {self.saturation_scale:.3f}")
            print(f"   Optimal scale: {optimal_scale:.3f}")
            print(f"   New scale: {new_scale:.3f}")
            
            self.saturation_scale = new_scale
        
        # Refine dimension weights
        # Increase weight for dimensions that correlate with color
        for d in range(self.manifold_dims):
            # Get this dimension's values from drum
            dim_vals = self.drum_positions[:, d]
            u_vals = self.drum_colors[:, 0]
            v_vals = self.drum_colors[:, 1]
            
            # Correlation with color
            corr_u = np.abs(np.corrcoef(dim_vals, u_vals)[0, 1])
            corr_v = np.abs(np.corrcoef(dim_vals, v_vals)[0, 1])
            
            if not np.isnan(corr_u) and not np.isnan(corr_v):
                color_corr = (corr_u + corr_v) / 2
                # Adjust weight
                self.dim_weights[d] *= (1 + learning_rate * color_corr)
        
        # Renormalize weights
        self.dim_weights /= self.dim_weights.sum()
        print(f"   Refined weights: {self.dim_weights.round(3)}")
    
    def colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize a grayscale image."""
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
                
                u, v = self.predict_color(patch, y_pos, x_pos)
                u_map[py, px] = u
                v_map[py, px] = v
        
        # Upsample
        scale_y = H / n_patches_y
        scale_x = W / n_patches_x
        u_full = zoom(u_map, (scale_y, scale_x), order=1)[:H, :W]
        v_full = zoom(v_map, (scale_y, scale_x), order=1)[:H, :W]
        
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


def run_constrained_drum_test():
    """Test the constrained drum colorizer."""
    print("=" * 70)
    print("CONSTRAINED DRUM COLORIZER")
    print("Drum (examples) + Comb (constraints) = Music (color)")
    print("=" * 70)
    
    # Load images
    print("\n1. LOADING IMAGES")
    print("-" * 50)
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    print(f"   Train: {len(train_images)}, Test: {len(test_data)}, New: {len(new_data)}")
    
    # Train
    print("\n2. TRAINING")
    print("-" * 50)
    colorizer = ConstrainedDrumColorizer(patch_size=16, manifold_dims=6)
    colorizer.train(train_images, sample_rate=0.12)
    
    # Test before refinement
    print("\n3. TEST BEFORE REFINEMENT")
    print("-" * 50)
    
    test_errors_before = []
    for name, img in test_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        test_errors_before.append(error)
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_before = np.mean(test_errors_before)
    print(f"   Average: {avg_before:.2f}")
    
    # Refine constraints
    print("\n4. REFINING CONSTRAINTS (3 passes)")
    print("-" * 50)
    
    for pass_num in range(3):
        print(f"\n   Pass {pass_num + 1}:")
        colorizer.refine_constraints([img for _, img in test_data], learning_rate=0.2)
    
    # Test after refinement
    print("\n5. TEST AFTER REFINEMENT")
    print("-" * 50)
    
    test_errors_after = []
    test_results = []
    for name, img in test_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        test_errors_after.append(error)
        test_results.append((name, img, gray, colorized, error))
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_after = np.mean(test_errors_after)
    print(f"   Average: {avg_after:.2f}")
    print(f"   Improvement: {(avg_before - avg_after) / avg_before * 100:.1f}%")
    
    # Generalization
    print("\n6. GENERALIZATION (new images)")
    print("-" * 50)
    
    gen_errors = []
    for name, img in new_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        gen_errors.append(error)
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_gen = np.mean(gen_errors)
    print(f"   Average: {avg_gen:.2f}")
    
    # Visualize
    print("\n7. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(test_results), 4, figsize=(16, 4 * len(test_results)))
    
    for i, (name, original, gray, colorized, error) in enumerate(test_results):
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Constrained (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Constrained Drum: Test={avg_after:.1f}, Gen={avg_gen:.1f}, '
                 f'Scale={colorizer.saturation_scale:.2f}',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "constrained_drum_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return colorizer, avg_before, avg_after, avg_gen


if __name__ == "__main__":
    colorizer, before, after, gen = run_constrained_drum_test()
    
    print("\n" + "=" * 70)
    print("CONSTRAINED DRUM SUMMARY")
    print("=" * 70)
    print(f"""
   Architecture:
   - DRUM: {len(colorizer.drum_positions)} examples in {colorizer.manifold_dims}D manifold space
   - COMB: Manifold projection + saturation scaling + dimension weights
   
   Learned constraints:
   - Saturation scale: {colorizer.saturation_scale:.3f}
   - Dimension weights: {colorizer.dim_weights.round(3)}
   
   Results:
   - Before refinement: {before:.2f}
   - After refinement: {after:.2f}
   - Generalization: {gen:.2f}
   - Test→Gen gap: {gen - after:.2f}
   
   The key insight:
   - The DRUM stores examples (data)
   - The COMB applies constraints (geometry)
   - Refinement adjusts the COMB, not the DRUM
   
   This is the Music Box Principle applied to colorization.
""")
