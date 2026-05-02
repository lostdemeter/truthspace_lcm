#!/usr/bin/env python3
"""
DA2-Inspired Colorizer - Using Dimension Separation

Key insight from DA2 reverse engineering (Doc 122):
- DA2 has DEDICATED dimensions for different features
- Luminance: 15 dimensions, top dim 323 (0.72 correlation)
- Color (R,G,B): Separate dedicated dimensions
- Depth: 101 dimensions
- Position: 85 (X) + 38 (Y) dimensions

For colorization, we should:
1. Separate features into dedicated "dimensions"
2. Have dedicated U-dimensions and V-dimensions
3. Use φ-scaled weights like DA2's decoder

The DA2 decoder formula (Doc 125):
    depth = Σ sign(corr_i) × φ^(exp_i) × dim_i

For color:
    U = Σ sign(corr_i) × φ^(exp_i) × feature_i
    V = Σ sign(corr_j) × φ^(exp_j) × feature_j

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom, gaussian_filter
from scipy.stats import pearsonr
from typing import List, Tuple, Dict
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
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


class DA2InspiredColorizer:
    """
    Colorizer inspired by DA2's dimension separation architecture.
    
    Like DA2:
    - Extract many features (like DA2's 384 dimensions)
    - Find which features correlate with U and V
    - Use φ-scaled weights for decoding
    
    The key insight: SEPARATE the features, then COMBINE with φ-weights.
    """
    
    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
        
        # Feature extraction produces many "dimensions"
        self.n_features = 32  # Like DA2's head features
        
        # Learned correlations (like DA2's dimension mapping)
        self.u_correlations = None  # Which features correlate with U
        self.v_correlations = None  # Which features correlate with V
        
        # φ-scaled weights (like DA2's decoder)
        self.u_weights = None
        self.v_weights = None
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        self.is_trained = False
    
    def extract_rich_features(self, gray_patch: np.ndarray, 
                               y_pos: float, x_pos: float) -> np.ndarray:
        """
        Extract rich features like DA2's backbone.
        
        DA2 has 384 dimensions encoding:
        - Position (X, Y)
        - Luminance
        - Edges
        - Local structure
        
        We create similar features from grayscale.
        """
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        features = []
        
        # 1. Basic statistics (4 features)
        features.append(patch.mean())           # Luminance
        features.append(patch.std())            # Contrast
        features.append(patch.min())            # Darkest
        features.append(patch.max())            # Brightest
        
        # 2. Position (4 features)
        features.append(y_pos)                  # Vertical position
        features.append(x_pos)                  # Horizontal position
        features.append(y_pos * x_pos)          # Position interaction
        features.append(np.sqrt(y_pos**2 + x_pos**2))  # Distance from origin
        
        # 3. Texture (8 features)
        if h > 1 and w > 1:
            # Horizontal and vertical gradients
            grad_h = np.abs(np.diff(patch, axis=1)).mean()
            grad_v = np.abs(np.diff(patch, axis=0)).mean()
            features.append(grad_h)
            features.append(grad_v)
            features.append(grad_h * grad_v)    # Texture interaction
            features.append(grad_h / (grad_v + 0.01))  # Texture ratio
            
            # Sobel edges
            if h > 2 and w > 2:
                gx = sobel(patch, axis=1)
                gy = sobel(patch, axis=0)
                edge_mag = np.sqrt(gx**2 + gy**2).mean()
                edge_dir = np.arctan2(gy.mean(), gx.mean()) / np.pi
                features.append(edge_mag)
                features.append(edge_dir)
                features.append(np.abs(gx).mean())
                features.append(np.abs(gy).mean())
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0] * 8)
        
        # 4. Local structure (8 features)
        # Quadrant means
        mid_h, mid_w = h // 2, w // 2
        if mid_h > 0 and mid_w > 0:
            q1 = patch[:mid_h, :mid_w].mean()
            q2 = patch[:mid_h, mid_w:].mean()
            q3 = patch[mid_h:, :mid_w].mean()
            q4 = patch[mid_h:, mid_w:].mean()
        else:
            q1 = q2 = q3 = q4 = patch.mean()
        
        features.append(q1)
        features.append(q2)
        features.append(q3)
        features.append(q4)
        
        # Quadrant contrasts
        features.append(q1 - q4)  # Diagonal contrast
        features.append(q2 - q3)  # Anti-diagonal contrast
        features.append((q1 + q4) - (q2 + q3))  # Cross contrast
        features.append(max(q1, q2, q3, q4) - min(q1, q2, q3, q4))  # Max contrast
        
        # 5. Luminance-position interactions (8 features)
        lum = patch.mean()
        features.append(lum * y_pos)            # Sky tends to be bright at top
        features.append(lum * (1 - y_pos))      # Ground tends to be dark at bottom
        features.append(lum * x_pos)
        features.append(lum * (1 - x_pos))
        features.append(patch.std() * y_pos)   # Texture-position
        features.append(patch.std() * (1 - y_pos))
        features.append(lum ** 2)               # Quadratic luminance
        features.append(np.sqrt(lum + 0.01))    # Square root luminance
        
        return np.array(features[:self.n_features], dtype=np.float32)
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """
        Train like DA2's dimension mapping:
        1. Collect features and colors
        2. Find correlations (which features predict U, which predict V)
        3. Compute φ-scaled weights
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
                    
                    feat = self.extract_rich_features(gray_patch, y_pos, x_pos)
                    u = yuv_patch[:,:,1].mean()
                    v = yuv_patch[:,:,2].mean()
                    
                    all_features.append(feat)
                    all_u.append(u)
                    all_v.append(v)
        
        features = np.array(all_features)
        u_vals = np.array(all_u)
        v_vals = np.array(all_v)
        
        print(f"   Collected {len(features)} samples, {self.n_features} features")
        
        # Normalize features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-10
        X = (features - self.feature_mean) / self.feature_std
        
        # Step 1: Find correlations (like DA2 dimension mapping)
        print("   Computing feature correlations...")
        
        self.u_correlations = np.zeros(self.n_features)
        self.v_correlations = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            corr_u, _ = pearsonr(X[:, i], u_vals)
            corr_v, _ = pearsonr(X[:, i], v_vals)
            self.u_correlations[i] = corr_u if not np.isnan(corr_u) else 0
            self.v_correlations[i] = corr_v if not np.isnan(corr_v) else 0
        
        # Report top correlations
        u_sorted = np.argsort(np.abs(self.u_correlations))[::-1]
        v_sorted = np.argsort(np.abs(self.v_correlations))[::-1]
        
        print("\n   Top U-correlated features:")
        for i in u_sorted[:5]:
            print(f"     Feature {i}: corr = {self.u_correlations[i]:.4f}")
        
        print("\n   Top V-correlated features:")
        for i in v_sorted[:5]:
            print(f"     Feature {i}: corr = {self.v_correlations[i]:.4f}")
        
        # Step 2: Compute φ-scaled weights (like DA2 decoder)
        print("\n   Computing φ-scaled weights...")
        
        # Use correlation as base, scale by φ^(-rank)
        self.u_weights = np.zeros(self.n_features)
        self.v_weights = np.zeros(self.n_features)
        
        for rank, i in enumerate(u_sorted):
            sign = np.sign(self.u_correlations[i])
            magnitude = abs(self.u_correlations[i])
            phi_scale = PHI ** (-rank / 10)  # φ-decay by rank
            self.u_weights[i] = sign * magnitude * phi_scale
        
        for rank, i in enumerate(v_sorted):
            sign = np.sign(self.v_correlations[i])
            magnitude = abs(self.v_correlations[i])
            phi_scale = PHI ** (-rank / 10)
            self.v_weights[i] = sign * magnitude * phi_scale
        
        # Normalize weights
        self.u_weights /= np.abs(self.u_weights).sum() + 1e-10
        self.v_weights /= np.abs(self.v_weights).sum() + 1e-10
        
        # Compute R² to see how well this works
        u_pred = X @ self.u_weights
        v_pred = X @ self.v_weights
        
        # Scale predictions to match target range
        u_scale = np.std(u_vals) / (np.std(u_pred) + 1e-10)
        v_scale = np.std(v_vals) / (np.std(v_pred) + 1e-10)
        
        self.u_weights *= u_scale
        self.v_weights *= v_scale
        
        # Recompute predictions
        u_pred = X @ self.u_weights
        v_pred = X @ self.v_weights
        
        r2_u = 1 - np.sum((u_vals - u_pred)**2) / np.sum((u_vals - u_vals.mean())**2)
        r2_v = 1 - np.sum((v_vals - v_pred)**2) / np.sum((v_vals - v_vals.mean())**2)
        
        print(f"\n   φ-decoder R²: U={r2_u:.4f}, V={r2_v:.4f}")
        
        self.is_trained = True
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float) -> Tuple[float, float]:
        """
        Predict color using φ-scaled weights.
        
        Like DA2: depth = Σ weight_i × feature_i
        For us:   U = Σ u_weight_i × feature_i
                  V = Σ v_weight_i × feature_i
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract features
        feat = self.extract_rich_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / self.feature_std
        
        # φ-weighted sum
        u = np.dot(feat_norm, self.u_weights)
        v = np.dot(feat_norm, self.v_weights)
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize using DA2-style φ-decoder."""
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


def run_da2_inspired_test():
    """Test the DA2-inspired colorizer."""
    print("=" * 70)
    print("DA2-INSPIRED COLORIZER")
    print("Using dimension separation and φ-scaled weights")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. TRAINING")
    print("-" * 50)
    
    colorizer = DA2InspiredColorizer(patch_size=16)
    colorizer.train(train_images, sample_rate=0.12)
    
    print("\n2. TESTING")
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
    
    test_mae = np.mean(test_errors)
    print(f"   Average test MAE: {test_mae:.2f}")
    
    print("\n3. GENERALIZATION")
    print("-" * 50)
    
    gen_errors = []
    for name, img in new_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        gen_errors.append(error)
        print(f"   {name}: MAE = {error:.2f}")
    
    gen_mae = np.mean(gen_errors)
    print(f"   Average generalization MAE: {gen_mae:.2f}")
    
    # Visualize
    print("\n4. VISUALIZATION")
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
        axes[i, 2].set_title(f'DA2-inspired ({error:.1f})' if i == 0 else f'{error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'DA2-Inspired Colorizer: {colorizer.n_features} features, φ-scaled weights\n'
                 f'Test={test_mae:.1f}, Gen={gen_mae:.1f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "da2_inspired_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'da2_inspired_test.png'}")
    
    return colorizer, test_mae, gen_mae


if __name__ == "__main__":
    colorizer, test_mae, gen_mae = run_da2_inspired_test()
    
    print("\n" + "=" * 70)
    print("DA2-INSPIRED COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   Inspired by DA2 reverse engineering (Docs 122, 125):
   
   DA2's architecture:
   - 384 dimensions encoding different features
   - Dedicated dimensions for luminance, color, position, edges
   - φ-scaled weights for decoding: depth = Σ sign × φ^exp × dim
   
   Our approach:
   - {colorizer.n_features} features (like DA2's dimensions)
   - Separate correlations for U and V
   - φ-scaled weights: U = Σ u_weight × feature
   
   Results:
   - Test MAE: {test_mae:.2f}
   - Generalization MAE: {gen_mae:.2f}
   
   Key insight from DA2:
   - SEPARATE features into dedicated dimensions
   - Find CORRELATIONS with target
   - Use φ-SCALED weights for decoding
   
   This is the same principle that achieved 99.98% accuracy on DA2!
""")
