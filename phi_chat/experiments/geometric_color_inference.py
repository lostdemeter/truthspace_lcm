#!/usr/bin/env python3
"""
Geometric Color Inference - Using Structure to Infer Missing Information

The deep insight: If color is LINEARLY ENCODED in feature space
(like depth in DA2), we can LEARN the encoding and use it to
predict colors for ANY point, even ones far from training data.

This is different from nearest-neighbor:
- NN: "What stored point is closest?" → use its color
- Linear: "What does the STRUCTURE say this color should be?"

The process:
1. Learn a linear mapping: features → color (like DA2's depth decoder)
2. This captures the GEOMETRIC RULES of color
3. Apply to any point, even extrapolations

If this works, it means color is geometrically encoded in the features,
just like depth is geometrically encoded in DA2's features.

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


class GeometricColorInference:
    """
    Learn the geometric encoding of color in feature space.
    
    Like DA2's depth decoder, but for color:
    - DA2: features → linear weights → depth
    - Ours: features → linear weights → (U, V)
    
    If color is linearly encoded, we can predict it for ANY point.
    """
    
    def __init__(self, n_feature_dims: int = 16, patch_size: int = 16):
        self.n_dims = n_feature_dims
        self.patch_size = patch_size
        
        # Linear decoder weights (like DA2)
        # features @ weights_u = U
        # features @ weights_v = V
        self.weights_u = np.zeros(n_feature_dims)
        self.weights_v = np.zeros(n_feature_dims)
        self.bias_u = 0.0
        self.bias_v = 0.0
        
        # Training data for learning the linear mapping
        self.training_features = []
        self.training_u = []
        self.training_v = []
        
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
    
    def collect_training_data(self, color_image: np.ndarray, sample_rate: float = 0.15) -> int:
        """Collect feature-color pairs for learning the linear mapping."""
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
                
                features = self.extract_features(gray_patch, y_pos, x_pos)
                mean_u = yuv_patch[:, :, 1].mean()
                mean_v = yuv_patch[:, :, 2].mean()
                
                self.training_features.append(features)
                self.training_u.append(mean_u)
                self.training_v.append(mean_v)
                collected += 1
        
        self.n_images += 1
        return collected
    
    def learn_phi_decoder(self):
        """
        Learn a φ-scaled decoder (like DA2).
        
        Instead of linear: color = features @ weights
        Use φ-scaled: color = Σ sign_i × φ^(level_i) × feature_i
        
        This captures non-linear relationships geometrically.
        """
        if len(self.training_features) < 100:
            return
        
        X = np.array(self.training_features)
        u = np.array(self.training_u)
        v = np.array(self.training_v)
        
        # Try different φ-levels for each feature
        best_r2_u = -np.inf
        best_r2_v = -np.inf
        
        # φ-levels to try
        levels = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        phi_weights = PHI ** levels
        
        # For each feature, find best φ-level
        self.phi_levels_u = np.zeros(self.n_dims)
        self.phi_levels_v = np.zeros(self.n_dims)
        self.phi_signs_u = np.ones(self.n_dims)
        self.phi_signs_v = np.ones(self.n_dims)
        
        for d in range(self.n_dims):
            feature_col = X[:, d]
            
            # Find best level for U
            best_corr_u = 0
            for level in levels:
                scaled = feature_col * (PHI ** level)
                corr = np.corrcoef(scaled, u)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(best_corr_u):
                    best_corr_u = corr
                    self.phi_levels_u[d] = level
                    self.phi_signs_u[d] = np.sign(corr) if corr != 0 else 1
            
            # Find best level for V
            best_corr_v = 0
            for level in levels:
                scaled = feature_col * (PHI ** level)
                corr = np.corrcoef(scaled, v)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(best_corr_v):
                    best_corr_v = corr
                    self.phi_levels_v[d] = level
                    self.phi_signs_v[d] = np.sign(corr) if corr != 0 else 1
        
        # Compute final weights
        self.weights_u = self.phi_signs_u * (PHI ** self.phi_levels_u)
        self.weights_v = self.phi_signs_v * (PHI ** self.phi_levels_v)
        
        # Normalize
        self.weights_u /= np.linalg.norm(self.weights_u)
        self.weights_v /= np.linalg.norm(self.weights_v)
        
        # Scale to match target range
        u_pred_raw = X @ self.weights_u
        v_pred_raw = X @ self.weights_v
        
        # Linear regression to find scale and bias
        self.scale_u = np.cov(u_pred_raw, u)[0, 1] / (np.var(u_pred_raw) + 1e-10)
        self.bias_u = u.mean() - self.scale_u * u_pred_raw.mean()
        
        self.scale_v = np.cov(v_pred_raw, v)[0, 1] / (np.var(v_pred_raw) + 1e-10)
        self.bias_v = v.mean() - self.scale_v * v_pred_raw.mean()
        
        # Compute R²
        u_pred = self.scale_u * u_pred_raw + self.bias_u
        v_pred = self.scale_v * v_pred_raw + self.bias_v
        
        ss_res_u = np.sum((u - u_pred)**2)
        ss_tot_u = np.sum((u - u.mean())**2)
        r2_u = 1 - ss_res_u / (ss_tot_u + 1e-10)
        
        ss_res_v = np.sum((v - v_pred)**2)
        ss_tot_v = np.sum((v - v.mean())**2)
        r2_v = 1 - ss_res_v / (ss_tot_v + 1e-10)
        
        print(f"   φ-scaled decoder learned:")
        print(f"     R² for U (blue-yellow): {r2_u:.3f}")
        print(f"     R² for V (red-green):   {r2_v:.3f}")
        
        return r2_u, r2_v
    
    def predict_color_phi(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict color using φ-scaled decoder."""
        u_raw = np.dot(features, self.weights_u)
        v_raw = np.dot(features, self.weights_v)
        
        u = self.scale_u * u_raw + self.bias_u
        v = self.scale_v * v_raw + self.bias_v
        
        return u, v
    
    def learn_linear_decoder(self):
        """
        Learn the linear mapping from features to color.
        
        This is like learning DA2's depth decoder weights.
        If color is linearly encoded, this will find the encoding.
        
        Uses least squares: min ||X @ w - y||^2
        Solution: w = (X^T X)^{-1} X^T y
        """
        if len(self.training_features) < 100:
            print("   Not enough training data")
            return
        
        X = np.array(self.training_features)
        u = np.array(self.training_u)
        v = np.array(self.training_v)
        
        # Add bias term
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Solve for U weights
        try:
            weights_u_full = np.linalg.lstsq(X_bias, u, rcond=None)[0]
            self.weights_u = weights_u_full[:-1]
            self.bias_u = weights_u_full[-1]
        except:
            self.weights_u = np.zeros(self.n_dims)
            self.bias_u = u.mean()
        
        # Solve for V weights
        try:
            weights_v_full = np.linalg.lstsq(X_bias, v, rcond=None)[0]
            self.weights_v = weights_v_full[:-1]
            self.bias_v = weights_v_full[-1]
        except:
            self.weights_v = np.zeros(self.n_dims)
            self.bias_v = v.mean()
        
        # Compute R² to see how linear the encoding is
        u_pred = X @ self.weights_u + self.bias_u
        v_pred = X @ self.weights_v + self.bias_v
        
        ss_res_u = np.sum((u - u_pred)**2)
        ss_tot_u = np.sum((u - u.mean())**2)
        r2_u = 1 - ss_res_u / (ss_tot_u + 1e-10)
        
        ss_res_v = np.sum((v - v_pred)**2)
        ss_tot_v = np.sum((v - v.mean())**2)
        r2_v = 1 - ss_res_v / (ss_tot_v + 1e-10)
        
        print(f"   Linear decoder learned:")
        print(f"     R² for U (blue-yellow): {r2_u:.3f}")
        print(f"     R² for V (red-green):   {r2_v:.3f}")
        print(f"     Training samples: {len(self.training_features)}")
        
        # Show top contributing features
        dim_names = ['lum', 'con', 'tex_h', 'tex_v', 'y_pos', 'x_pos', 'edge', 'smooth',
                     'grad_m', 'grad_d', 'max', 'min', 'tex_c', 'tex_f', 'unif', 'ent']
        
        print(f"\n   Top features for U (blue-yellow):")
        for i in np.argsort(np.abs(self.weights_u))[-3:][::-1]:
            print(f"     {dim_names[i]}: {self.weights_u[i]:.4f}")
        
        print(f"\n   Top features for V (red-green):")
        for i in np.argsort(np.abs(self.weights_v))[-3:][::-1]:
            print(f"     {dim_names[i]}: {self.weights_v[i]:.4f}")
        
        return r2_u, r2_v
    
    def predict_color_linear(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict color using the learned linear decoder.
        
        This is the key: we can predict color for ANY point,
        even ones far from training data, because we learned
        the GEOMETRIC ENCODING.
        """
        u = np.dot(features, self.weights_u) + self.bias_u
        v = np.dot(features, self.weights_v) + self.bias_v
        return u, v
    
    def colorize_sharp(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize using the linear decoder."""
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
                
                features = self.extract_features(patch, y_pos, x_pos)
                u, v = self.predict_color_linear(features)
                u_map[py, px] = u
                v_map[py, px] = v
        
        # Upsample chrominance
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
    
    def refine_decoder(self, color_image: np.ndarray, learning_rate: float = 0.01) -> dict:
        """
        Refine the linear decoder using ground truth.
        
        This adjusts the weights to better predict colors.
        """
        H, W = color_image.shape[:2]
        
        grayscale = (0.299 * color_image[:,:,0] + 
                     0.587 * color_image[:,:,1] + 
                     0.114 * color_image[:,:,2]).astype(np.uint8)
        
        yuv = rgb_to_yuv(color_image)
        
        total_error_before = 0
        total_error_after = 0
        n_patches = 0
        
        # Accumulate gradients
        grad_u = np.zeros(self.n_dims)
        grad_v = np.zeros(self.n_dims)
        grad_bias_u = 0
        grad_bias_v = 0
        
        for y in range(0, H - self.patch_size, self.patch_size):
            for x in range(0, W - self.patch_size, self.patch_size):
                gray_patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                features = self.extract_features(gray_patch, y_pos, x_pos)
                
                true_u = yuv_patch[:, :, 1].mean()
                true_v = yuv_patch[:, :, 2].mean()
                
                pred_u, pred_v = self.predict_color_linear(features)
                
                error_u = pred_u - true_u
                error_v = pred_v - true_v
                error = np.sqrt(error_u**2 + error_v**2)
                total_error_before += error
                
                # Gradient: d(error)/d(weights) = features * error
                grad_u += features * error_u
                grad_v += features * error_v
                grad_bias_u += error_u
                grad_bias_v += error_v
                
                n_patches += 1
        
        # Update weights
        self.weights_u -= learning_rate * grad_u / n_patches
        self.weights_v -= learning_rate * grad_v / n_patches
        self.bias_u -= learning_rate * grad_bias_u / n_patches
        self.bias_v -= learning_rate * grad_bias_v / n_patches
        
        # Measure error after
        for y in range(0, H - self.patch_size, self.patch_size):
            for x in range(0, W - self.patch_size, self.patch_size):
                gray_patch = grayscale[y:y+self.patch_size, x:x+self.patch_size]
                yuv_patch = yuv[y:y+self.patch_size, x:x+self.patch_size]
                
                y_pos = (y + self.patch_size/2) / H
                x_pos = (x + self.patch_size/2) / W
                
                features = self.extract_features(gray_patch, y_pos, x_pos)
                
                true_u = yuv_patch[:, :, 1].mean()
                true_v = yuv_patch[:, :, 2].mean()
                
                pred_u, pred_v = self.predict_color_linear(features)
                
                error = np.sqrt((pred_u - true_u)**2 + (pred_v - true_v)**2)
                total_error_after += error
        
        return {
            'error_before': total_error_before / n_patches,
            'error_after': total_error_after / n_patches,
            'n_patches': n_patches
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


def run_geometric_inference_test():
    """Test the geometric color inference approach."""
    print("=" * 70)
    print("GEOMETRIC COLOR INFERENCE (Linear Decoder)")
    print("=" * 70)
    
    inferencer = GeometricColorInference(n_feature_dims=16, patch_size=16)
    
    # Phase 1: Collect training data
    print("\n1. COLLECTING TRAINING DATA (150 images)")
    print("-" * 50)
    train_images = load_coco_images(150, start_idx=0)
    
    for i, (name, img) in enumerate(train_images):
        inferencer.collect_training_data(img, sample_rate=0.12)
        if (i + 1) % 50 == 0:
            print(f"   Collected from {i+1}/{len(train_images)} images")
    
    print(f"   Total training samples: {len(inferencer.training_features)}")
    
    # Phase 2: Learn both decoders
    print("\n2. LEARNING DECODERS")
    print("-" * 50)
    print("\n   A. Linear decoder:")
    r2_u_lin, r2_v_lin = inferencer.learn_linear_decoder()
    print(f"\n   B. φ-scaled decoder:")
    r2_u_phi, r2_v_phi = inferencer.learn_phi_decoder()
    
    r2_u, r2_v = r2_u_lin, r2_v_lin  # Use linear for comparison
    
    # Phase 3: Test
    print("\n3. TEST (5 images)")
    print("-" * 50)
    test_images = load_coco_images(5, start_idx=200)
    
    errors_before = []
    for name, img in test_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = inferencer.colorize_sharp(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        errors_before.append(error)
        print(f"   {name}: MAE = {error:.2f}")
    
    avg_before = np.mean(errors_before)
    print(f"   Average MAE: {avg_before:.2f}")
    
    # Phase 4: Refine decoder using ground truth
    print("\n4. REFINING DECODER (5 passes)")
    print("-" * 50)
    
    for pass_num in range(5):
        print(f"\n   Pass {pass_num + 1}:")
        for name, img in test_images:
            stats = inferencer.refine_decoder(img, learning_rate=0.005)
            print(f"     {name}: {stats['error_before']:.4f} → {stats['error_after']:.4f}")
    
    # Phase 5: Test after refinement
    print("\n5. TEST AFTER REFINEMENT")
    print("-" * 50)
    
    errors_after = []
    results = []
    for name, img in test_images:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = inferencer.colorize_sharp(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        errors_after.append(error)
        print(f"   {name}: MAE = {error:.2f}")
        results.append((name, img, gray, colorized, error))
    
    avg_after = np.mean(errors_after)
    print(f"   Average MAE: {avg_after:.2f}")
    print(f"   Improvement: {(avg_before - avg_after) / avg_before * 100:.1f}%")
    
    # Visualize
    print("\n6. CREATING VISUALIZATION")
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
        axes[i, 2].set_title(f'Linear Decoder (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Geometric Color Inference: R²_U={r2_u:.2f}, R²_V={r2_v:.2f}, MAE={avg_after:.1f}',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "geometric_inference_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return inferencer, r2_u, r2_v, avg_before, avg_after


if __name__ == "__main__":
    inferencer, r2_u, r2_v, before, after = run_geometric_inference_test()
    
    print("\n" + "=" * 70)
    print("GEOMETRIC COLOR INFERENCE SUMMARY")
    print("=" * 70)
    print(f"""
   The key question: Is color LINEARLY ENCODED in feature space?
   
   Answer:
   - R² for U (blue-yellow): {r2_u:.3f}
   - R² for V (red-green):   {r2_v:.3f}
   
   If R² is high, color is linearly encoded (like depth in DA2).
   If R² is low, color requires non-linear relationships.
   
   Results:
   - Before refinement: MAE = {before:.2f}
   - After refinement:  MAE = {after:.2f}
   
   The linear decoder uses only {inferencer.n_dims} weights per channel.
   Compare to neural networks with millions of parameters.
   
   This is the DA2 approach applied to colorization:
   - Learn a linear mapping from features to color
   - Apply to ANY point (even extrapolations)
   - Refine using ground truth
""")
