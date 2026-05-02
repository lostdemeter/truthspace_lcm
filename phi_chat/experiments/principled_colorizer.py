#!/usr/bin/env python3
"""
Principled Colorizer - Based on Transformation Analysis

Key findings from analysis:
1. Linear R² is only ~0.03 → need non-linear
2. Quadratic R² is ~0.07 → texture interactions matter
3. Top terms: con×tex_h, tex_h×tex_v, con², tex_v²
4. Intrinsic dimensionality: 5-6 dims

This colorizer uses:
1. Quadratic texture interaction features
2. SVD to find the low-dimensional manifold
3. The manifold structure to predict color

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.linalg import svd
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


class PrincipledColorizer:
    """
    Colorizer based on the geometric structure discovered in analysis.
    
    Key insight: Color depends on INTERACTIONS between texture features,
    not just the features themselves.
    
    The transformation lives on a low-dimensional manifold.
    """
    
    def __init__(self, patch_size: int = 16, manifold_dims: int = 6):
        self.patch_size = patch_size
        self.manifold_dims = manifold_dims
        
        # Linear features: 8
        # Quadratic features: 8 + 28 = 36 (squares + pairs)
        # Total: 44 features before manifold projection
        
        # Manifold projection (learned from data)
        self.manifold_basis = None  # V from SVD
        self.feature_mean = None
        self.feature_std = None
        
        # Color prediction weights (on manifold)
        self.W_u = None
        self.W_v = None
        
        # Saturation scale (from manipulation experiments)
        self.saturation_scale = 0.5  # We found 0.5x improves accuracy
        
        self.is_trained = False
    
    def extract_linear_features(self, gray_patch: np.ndarray, 
                                 y_pos: float, x_pos: float) -> np.ndarray:
        """Extract 8 linear features."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        luminance = patch.mean()
        contrast = patch.std()
        
        texture_h = np.abs(np.diff(patch, axis=1)).mean() if w > 1 else 0
        texture_v = np.abs(np.diff(patch, axis=0)).mean() if h > 1 else 0
        
        if h > 2 and w > 2:
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            gradient_mag = np.sqrt(gx**2 + gy**2).mean()
            gradient_dir = np.arctan2(gy.mean(), gx.mean()) / np.pi
        else:
            gradient_mag = gradient_dir = 0
        
        return np.array([
            luminance, contrast, texture_h, texture_v,
            y_pos, x_pos, gradient_mag, gradient_dir
        ], dtype=np.float32)
    
    def extract_quadratic_features(self, linear: np.ndarray) -> np.ndarray:
        """
        Extract quadratic interaction features.
        
        Based on analysis, the key interactions are:
        - con × tex_h
        - tex_h × tex_v
        - con²
        - tex_v²
        """
        lum, con, tex_h, tex_v, y_pos, x_pos, grad_m, grad_d = linear
        
        # Squares (8)
        squares = linear ** 2
        
        # Key interactions identified in analysis (6)
        interactions = np.array([
            con * tex_h,      # Most important for U
            tex_h * tex_v,    # Second most important
            con * tex_v,      # Important for V
            lum * con,        # Luminance-contrast interaction
            y_pos * lum,      # Position-luminance (sky vs ground)
            tex_h * grad_m,   # Texture-gradient interaction
        ], dtype=np.float32)
        
        return np.concatenate([linear, squares, interactions])
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract full feature vector."""
        linear = self.extract_linear_features(gray_patch, y_pos, x_pos)
        return self.extract_quadratic_features(linear)
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """
        Train the colorizer by learning the manifold structure.
        
        Steps:
        1. Collect (features, U, V) from all images
        2. Normalize features
        3. SVD to find manifold
        4. Learn linear mapping on manifold
        """
        print("   Collecting training data...")
        
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
        u = np.array(all_u)
        v = np.array(all_v)
        
        print(f"   Collected {len(X)} samples, {X.shape[1]} features")
        
        # Normalize
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-10
        X_norm = (X - self.feature_mean) / self.feature_std
        
        # SVD to find manifold
        print("   Finding manifold structure...")
        U_svd, S, Vt = svd(X_norm, full_matrices=False)
        
        # Keep top manifold_dims
        self.manifold_basis = Vt[:self.manifold_dims].T  # Shape: (n_features, manifold_dims)
        
        var_explained = (S[:self.manifold_dims]**2).sum() / (S**2).sum()
        print(f"   Manifold: {self.manifold_dims} dims, {var_explained*100:.1f}% variance")
        
        # Project to manifold
        X_manifold = X_norm @ self.manifold_basis
        
        # Learn linear mapping on manifold
        print("   Learning color mapping on manifold...")
        X_bias = np.hstack([X_manifold, np.ones((len(X_manifold), 1))])
        
        self.W_u = np.linalg.lstsq(X_bias, u, rcond=None)[0]
        self.W_v = np.linalg.lstsq(X_bias, v, rcond=None)[0]
        
        # Compute R²
        u_pred = X_bias @ self.W_u
        v_pred = X_bias @ self.W_v
        
        r2_u = 1 - np.sum((u - u_pred)**2) / np.sum((u - u.mean())**2)
        r2_v = 1 - np.sum((v - v_pred)**2) / np.sum((v - v.mean())**2)
        
        print(f"   Manifold R²: U={r2_u:.4f}, V={r2_v:.4f}")
        
        self.is_trained = True
        
        return r2_u, r2_v
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float) -> Tuple[float, float]:
        """Predict color using manifold projection."""
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract features
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        
        # Normalize
        feat_norm = (feat - self.feature_mean) / self.feature_std
        
        # Project to manifold
        feat_manifold = feat_norm @ self.manifold_basis
        
        # Predict color
        feat_bias = np.concatenate([feat_manifold, [1.0]])
        u = np.dot(feat_bias, self.W_u) * self.saturation_scale
        v = np.dot(feat_bias, self.W_v) * self.saturation_scale
        
        return u, v
    
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
        
        # Upsample chrominance smoothly
        scale_y = H / n_patches_y
        scale_x = W / n_patches_x
        u_full = zoom(u_map, (scale_y, scale_x), order=1)[:H, :W]
        v_full = zoom(v_map, (scale_y, scale_x), order=1)[:H, :W]
        
        # Pad if needed
        if u_full.shape[0] < H or u_full.shape[1] < W:
            u_padded = np.zeros((H, W), dtype=np.float32)
            v_padded = np.zeros((H, W), dtype=np.float32)
            u_padded[:u_full.shape[0], :u_full.shape[1]] = u_full
            v_padded[:v_full.shape[0], :v_full.shape[1]] = v_full
            u_full, v_full = u_padded, v_padded
        
        # Combine with original luminance
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


def run_principled_test():
    """Test the principled colorizer."""
    print("=" * 70)
    print("PRINCIPLED COLORIZER")
    print("Based on Transformation Analysis Findings")
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
    colorizer = PrincipledColorizer(patch_size=16, manifold_dims=6)
    r2_u, r2_v = colorizer.train(train_images, sample_rate=0.12)
    
    # Test
    print("\n3. TESTING")
    print("-" * 50)
    
    test_errors = []
    test_results = []
    for name, img in test_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        test_errors.append(error)
        test_results.append((name, img, gray, colorized, error))
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_test = np.mean(test_errors)
    print(f"   Average test MAE: {avg_test:.2f}")
    
    # Generalization
    print("\n4. GENERALIZATION (new images)")
    print("-" * 50)
    
    gen_errors = []
    for name, img in new_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        gen_errors.append(error)
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_gen = np.mean(gen_errors)
    print(f"   Average generalization MAE: {avg_gen:.2f}")
    
    # Visualize
    print("\n5. VISUALIZATION")
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
        axes[i, 2].set_title(f'Principled (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Principled Colorizer: {colorizer.manifold_dims}D manifold, '
                 f'Test={avg_test:.1f}, Gen={avg_gen:.1f}',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "principled_colorization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return colorizer, avg_test, avg_gen


if __name__ == "__main__":
    colorizer, test_mae, gen_mae = run_principled_test()
    
    print("\n" + "=" * 70)
    print("PRINCIPLED COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   Based on transformation analysis:
   
   Key features:
   - Quadratic texture interactions (con×tex_h, tex_h×tex_v, etc.)
   - 6-dimensional manifold projection
   - 0.5x saturation scaling
   
   Results:
   - Test MAE: {test_mae:.2f}
   - Generalization MAE: {gen_mae:.2f}
   - Gap: {gen_mae - test_mae:.2f}
   
   This is a PRINCIPLED approach:
   - Features chosen based on analysis (not trial and error)
   - Manifold dims chosen based on variance analysis
   - Saturation scale chosen based on manipulation experiments
   
   The transformation is:
   1. Extract quadratic texture features
   2. Project to 6D manifold
   3. Linear mapping to color
   4. Scale by 0.5x
""")
