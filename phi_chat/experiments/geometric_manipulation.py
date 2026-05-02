#!/usr/bin/env python3
"""
Geometric Manipulation Experiments

Since we control the model's geometry, we can directly manipulate it
and observe the effects on colorization.

Experiments:
1. SHIFT: Move all clusters in a direction → see color change
2. SCALE: Expand/contract clusters → see saturation change
3. ROTATE: Rotate in focused space → see hue shift
4. WARP: Apply φ-scaled warping → see what emerges

This is like having knobs on a synthesizer - we can turn them
and hear what happens.

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from typing import List, Tuple, Dict
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


class ManipulableColorizer:
    """
    A colorizer where we can directly manipulate the geometry.
    """
    
    def __init__(self, patch_size: int = 16, focus_dims: int = 6):
        self.patch_size = patch_size
        self.focus_dims = focus_dims
        self.raw_dims = 32
        
        self.svd_components = None
        self.svd_mean = None
        self.singular_values = None
        self.focus_weights = None
        
        self.training_features = []
        self.training_u = []
        self.training_v = []
        
        self.path_clusters = None
        self.original_clusters = None  # Keep original for reset
        self.n_clusters = 0
        
        self.n_images = 0
    
    def extract_raw_features(self, gray_patch: np.ndarray, 
                              y_pos: float, x_pos: float) -> np.ndarray:
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        features = []
        features.append(patch.mean())
        features.append(patch.std())
        features.append(patch.min())
        features.append(patch.max())
        features.append(y_pos)
        features.append(x_pos)
        
        if h > 1 and w > 1:
            features.append(np.abs(np.diff(patch, axis=1)).mean())
            features.append(np.abs(np.diff(patch, axis=0)).mean())
            features.append(np.abs(np.diff(patch, axis=1)).std())
            features.append(np.abs(np.diff(patch, axis=0)).std())
        else:
            features.extend([0, 0, 0, 0])
        
        if h > 2 and w > 2:
            gy = sobel(patch, axis=0)
            gx = sobel(patch, axis=1)
            features.append(np.sqrt(gx**2 + gy**2).mean())
            features.append(np.arctan2(gy.mean(), gx.mean()) / np.pi)
            features.append(gx.std())
            features.append(gy.std())
        else:
            features.extend([0, 0, 0, 0])
        
        if h >= 4 and w >= 4:
            coarse = patch.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            features.append(coarse.mean())
            features.append(coarse.std())
            features.append(np.abs(np.diff(coarse, axis=1)).mean())
            fine_var = patch.reshape(h//2, 2, w//2, 2).std(axis=(1, 3))
            features.append(fine_var.mean())
            features.append(fine_var.std())
            features.append(fine_var.max())
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        hist, _ = np.histogram(patch.flatten(), bins=8, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        features.extend(hist.tolist())
        
        if h > 2 and w > 2:
            center = patch[1:-1, 1:-1]
            neighbors = (patch[:-2, 1:-1] + patch[2:, 1:-1] + 
                        patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
            edge = np.abs(center - neighbors)
            features.append(edge.mean())
            features.append(edge.std())
            features.append(edge.max())
            features.append((edge > 0.1).mean())
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features[:self.raw_dims], dtype=np.float32)
    
    def collect_training_data(self, color_image: np.ndarray, sample_rate: float = 0.15) -> int:
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
                self.training_features.append(features)
                self.training_u.append(yuv_patch[:, :, 1].mean())
                self.training_v.append(yuv_patch[:, :, 2].mean())
                collected += 1
        
        self.n_images += 1
        return collected
    
    def learn_focus(self):
        X = np.array(self.training_features)
        self.svd_mean = X.mean(axis=0)
        X_centered = X - self.svd_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.svd_components = Vt[:self.focus_dims]
        self.singular_values = S[:self.focus_dims]
        self.focus_weights = np.array([PHI ** (-i / 2) for i in range(self.focus_dims)])
        self.focus_weights /= self.focus_weights.sum()
    
    def focus_features(self, raw_features: np.ndarray) -> np.ndarray:
        centered = raw_features - self.svd_mean
        focused = centered @ self.svd_components.T
        return focused * self.focus_weights
    
    def find_path_clusters(self, n_clusters: int = 25):
        X = np.array(self.training_features)
        u = np.array(self.training_u)
        v = np.array(self.training_v)
        X_focused = self.focus_features(X)
        joint = np.hstack([X_focused, u.reshape(-1, 1), v.reshape(-1, 1)])
        
        indices = np.random.choice(len(joint), n_clusters, replace=False)
        centers = joint[indices].copy()
        
        for _ in range(20):
            assignments = np.argmin(
                np.linalg.norm(joint[:, np.newaxis] - centers, axis=2), axis=1)
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
        self.original_clusters = centers.copy()
        self.n_clusters = n_clusters
    
    def reset_clusters(self):
        """Reset to original clusters."""
        self.path_clusters = self.original_clusters.copy()
    
    # ========== GEOMETRIC MANIPULATIONS ==========
    
    def shift_color(self, delta_u: float, delta_v: float):
        """Shift all cluster colors by a fixed amount."""
        self.path_clusters[:, -2] += delta_u
        self.path_clusters[:, -1] += delta_v
    
    def scale_color(self, scale: float):
        """Scale color saturation (multiply U and V)."""
        self.path_clusters[:, -2] *= scale
        self.path_clusters[:, -1] *= scale
    
    def rotate_color(self, angle: float):
        """Rotate in UV color space (hue shift)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        u = self.path_clusters[:, -2].copy()
        v = self.path_clusters[:, -1].copy()
        self.path_clusters[:, -2] = cos_a * u - sin_a * v
        self.path_clusters[:, -1] = sin_a * u + cos_a * v
    
    def warp_by_feature(self, feature_idx: int, strength: float):
        """Warp colors based on a feature dimension."""
        feature_vals = self.path_clusters[:, feature_idx]
        feature_normalized = (feature_vals - feature_vals.mean()) / (feature_vals.std() + 1e-10)
        self.path_clusters[:, -2] += strength * feature_normalized
        self.path_clusters[:, -1] += strength * feature_normalized * PHI
    
    def phi_scale_clusters(self, level: float):
        """Apply φ-scaling to cluster positions (not colors)."""
        features = self.path_clusters[:, :-2]
        center = features.mean(axis=0)
        features_centered = features - center
        scaled = features_centered * (PHI ** level)
        self.path_clusters[:, :-2] = scaled + center
    
    def swap_dimensions(self, dim1: int, dim2: int):
        """Swap two feature dimensions to see effect."""
        self.path_clusters[:, [dim1, dim2]] = self.path_clusters[:, [dim2, dim1]]
    
    # ========== COLORIZATION ==========
    
    def predict_color(self, raw_features: np.ndarray, k: int = 5) -> Tuple[float, float]:
        focused = self.focus_features(raw_features.reshape(1, -1))[0]
        cluster_features = self.path_clusters[:, :-2]
        distances = np.linalg.norm(cluster_features - focused, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        
        total_weight = 0
        weighted_u = 0
        weighted_v = 0
        
        for idx in nearest_idx:
            weight = 1.0 / (distances[idx]**2 + 0.001)
            weighted_u += weight * self.path_clusters[idx, -2]
            weighted_v += weight * self.path_clusters[idx, -1]
            total_weight += weight
        
        return weighted_u / total_weight, weighted_v / total_weight
    
    def colorize_sharp(self, grayscale: np.ndarray) -> np.ndarray:
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


def run_manipulation_experiments():
    """Run geometric manipulation experiments."""
    print("=" * 70)
    print("GEOMETRIC MANIPULATION EXPERIMENTS")
    print("=" * 70)
    
    # Setup
    print("\n1. SETUP")
    print("-" * 50)
    colorizer = ManipulableColorizer(patch_size=16, focus_dims=6)
    
    train_images = load_coco_images(100, start_idx=0)
    for name, img in train_images:
        colorizer.collect_training_data(img, sample_rate=0.12)
    
    colorizer.learn_focus()
    colorizer.find_path_clusters(n_clusters=25)
    print(f"   Trained on {len(train_images)} images")
    print(f"   {len(colorizer.training_features)} patches, {colorizer.n_clusters} clusters")
    
    # Test image
    test_images = load_coco_images(1, start_idx=200)
    test_name, test_img = test_images[0]
    test_gray = (0.299 * test_img[:,:,0] + 0.587 * test_img[:,:,1] + 0.114 * test_img[:,:,2]).astype(np.uint8)
    
    # Baseline
    colorizer.reset_clusters()
    baseline = colorizer.colorize_sharp(test_gray)
    baseline_mae = np.abs(baseline.astype(float) - test_img.astype(float)).mean()
    
    print(f"\n   Baseline MAE: {baseline_mae:.2f}")
    
    # Experiments
    print("\n2. MANIPULATION EXPERIMENTS")
    print("-" * 50)
    
    experiments = []
    
    # Experiment 1: Color shift
    shifts = [
        ("Shift +U (bluer)", 0.05, 0),
        ("Shift -U (yellower)", -0.05, 0),
        ("Shift +V (warmer)", 0, 0.05),
        ("Shift -V (cooler)", 0, -0.05),
    ]
    
    for name, du, dv in shifts:
        colorizer.reset_clusters()
        colorizer.shift_color(du, dv)
        result = colorizer.colorize_sharp(test_gray)
        mae = np.abs(result.astype(float) - test_img.astype(float)).mean()
        experiments.append((name, result, mae))
        print(f"   {name}: MAE = {mae:.2f} (Δ = {mae - baseline_mae:+.2f})")
    
    # Experiment 2: Saturation scaling
    scales = [
        ("Scale 0.5x (desaturate)", 0.5),
        ("Scale 1.5x (saturate)", 1.5),
        ("Scale 2.0x (vivid)", 2.0),
    ]
    
    for name, scale in scales:
        colorizer.reset_clusters()
        colorizer.scale_color(scale)
        result = colorizer.colorize_sharp(test_gray)
        mae = np.abs(result.astype(float) - test_img.astype(float)).mean()
        experiments.append((name, result, mae))
        print(f"   {name}: MAE = {mae:.2f} (Δ = {mae - baseline_mae:+.2f})")
    
    # Experiment 3: Hue rotation
    rotations = [
        ("Rotate +30°", np.pi/6),
        ("Rotate +90°", np.pi/2),
        ("Rotate +180°", np.pi),
    ]
    
    for name, angle in rotations:
        colorizer.reset_clusters()
        colorizer.rotate_color(angle)
        result = colorizer.colorize_sharp(test_gray)
        mae = np.abs(result.astype(float) - test_img.astype(float)).mean()
        experiments.append((name, result, mae))
        print(f"   {name}: MAE = {mae:.2f} (Δ = {mae - baseline_mae:+.2f})")
    
    # Experiment 4: Feature warping
    warps = [
        ("Warp by dim 0 (luminance)", 0, 0.02),
        ("Warp by dim 1", 1, 0.02),
        ("Warp by dim 2", 2, 0.02),
    ]
    
    for name, dim, strength in warps:
        colorizer.reset_clusters()
        colorizer.warp_by_feature(dim, strength)
        result = colorizer.colorize_sharp(test_gray)
        mae = np.abs(result.astype(float) - test_img.astype(float)).mean()
        experiments.append((name, result, mae))
        print(f"   {name}: MAE = {mae:.2f} (Δ = {mae - baseline_mae:+.2f})")
    
    # Experiment 5: φ-scaling
    phi_scales = [
        ("φ-scale +0.5 (expand)", 0.5),
        ("φ-scale -0.5 (contract)", -0.5),
        ("φ-scale +1.0 (expand more)", 1.0),
    ]
    
    for name, level in phi_scales:
        colorizer.reset_clusters()
        colorizer.phi_scale_clusters(level)
        result = colorizer.colorize_sharp(test_gray)
        mae = np.abs(result.astype(float) - test_img.astype(float)).mean()
        experiments.append((name, result, mae))
        print(f"   {name}: MAE = {mae:.2f} (Δ = {mae - baseline_mae:+.2f})")
    
    # Find best manipulation
    best = min(experiments, key=lambda x: x[2])
    print(f"\n   Best manipulation: {best[0]} (MAE = {best[2]:.2f})")
    
    # Visualize
    print("\n3. CREATING VISUALIZATION")
    print("-" * 50)
    
    # Select interesting experiments to show
    show_experiments = [
        ("Baseline", baseline, baseline_mae),
        experiments[2],  # Shift +V (warmer)
        experiments[3],  # Shift -V (cooler)
        experiments[5],  # Scale 1.5x
        experiments[8],  # Rotate +90°
        experiments[11], # Warp by dim 0
    ]
    
    n_show = len(show_experiments)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    
    # Top row: manipulated results
    for i, (name, result, mae) in enumerate(show_experiments):
        axes[0, i].imshow(result)
        axes[0, i].set_title(f'{name}\nMAE={mae:.1f}', fontsize=9)
        axes[0, i].axis('off')
    
    # Bottom row: difference from original
    for i, (name, result, mae) in enumerate(show_experiments):
        diff = np.abs(result.astype(float) - test_img.astype(float)).mean(axis=2)
        axes[1, i].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[1, i].set_title('Error map', fontsize=9)
        axes[1, i].axis('off')
    
    fig.suptitle('Geometric Manipulations: Effects on Colorization', fontsize=12, fontweight='bold')
    
    output_file = OUTPUT_PATH / "geometric_manipulation_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    # Also show original for reference
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    axes2[0].imshow(test_img)
    axes2[0].set_title('Original')
    axes2[0].axis('off')
    axes2[1].imshow(test_gray, cmap='gray')
    axes2[1].set_title('Grayscale')
    axes2[1].axis('off')
    axes2[2].imshow(baseline)
    axes2[2].set_title(f'Baseline (MAE={baseline_mae:.1f})')
    axes2[2].axis('off')
    
    output_file2 = OUTPUT_PATH / "geometric_manipulation_reference.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved reference to: {output_file2}")
    
    return colorizer, experiments, baseline_mae


def search_optimal_transformation():
    """Search for the optimal geometric transformation."""
    print("\n" + "=" * 70)
    print("SEARCHING FOR OPTIMAL TRANSFORMATION")
    print("=" * 70)
    
    # Setup
    colorizer = ManipulableColorizer(patch_size=16, focus_dims=6)
    train_images = load_coco_images(100, start_idx=0)
    for name, img in train_images:
        colorizer.collect_training_data(img, sample_rate=0.12)
    colorizer.learn_focus()
    colorizer.find_path_clusters(n_clusters=25)
    
    # Test images
    test_images = load_coco_images(5, start_idx=200)
    
    def evaluate(colorizer, test_images):
        errors = []
        for name, img in test_images:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            result = colorizer.colorize_sharp(gray)
            errors.append(np.abs(result.astype(float) - img.astype(float)).mean())
        return np.mean(errors)
    
    # Baseline
    colorizer.reset_clusters()
    baseline_mae = evaluate(colorizer, test_images)
    print(f"\n   Baseline MAE: {baseline_mae:.2f}")
    
    # Grid search over scale and shift
    print("\n   Searching scale × shift combinations...")
    
    best_mae = baseline_mae
    best_params = (1.0, 0, 0)
    
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    shifts_v = [-0.03, -0.02, -0.01, 0, 0.01]
    
    for scale in scales:
        for shift_v in shifts_v:
            colorizer.reset_clusters()
            colorizer.scale_color(scale)
            colorizer.shift_color(0, shift_v)
            mae = evaluate(colorizer, test_images)
            
            if mae < best_mae:
                best_mae = mae
                best_params = (scale, 0, shift_v)
                print(f"   NEW BEST: scale={scale}, shift_v={shift_v} → MAE={mae:.2f}")
    
    print(f"\n   Best transformation:")
    print(f"     Scale: {best_params[0]}")
    print(f"     Shift V: {best_params[2]}")
    print(f"     MAE: {best_mae:.2f} (improvement: {(baseline_mae - best_mae) / baseline_mae * 100:.1f}%)")
    
    # Test generalization
    print("\n   Testing generalization on NEW images...")
    new_images = load_coco_images(5, start_idx=300)
    
    colorizer.reset_clusters()
    gen_baseline = evaluate(colorizer, new_images)
    
    colorizer.reset_clusters()
    colorizer.scale_color(best_params[0])
    colorizer.shift_color(0, best_params[2])
    gen_optimized = evaluate(colorizer, new_images)
    
    print(f"     Baseline on new: {gen_baseline:.2f}")
    print(f"     Optimized on new: {gen_optimized:.2f}")
    print(f"     Improvement: {(gen_baseline - gen_optimized) / gen_baseline * 100:.1f}%")
    
    return best_params, best_mae


if __name__ == "__main__":
    colorizer, experiments, baseline = run_manipulation_experiments()
    
    # Also search for optimal
    best_params, best_mae = search_optimal_transformation()
    
    print("\n" + "=" * 70)
    print("MANIPULATION EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"""
   We can directly manipulate the geometry and observe effects:
   
   COLOR SHIFTS:
   - +U makes image bluer
   - -U makes image yellower  
   - +V makes image warmer (red/orange)
   - -V makes image cooler (blue/green)
   
   SATURATION:
   - Scale < 1 desaturates
   - Scale > 1 increases saturation
   
   HUE ROTATION:
   - Rotating in UV space shifts all hues
   - 180° inverts colors
   
   FEATURE WARPING:
   - Linking color to feature dimensions
   - Creates systematic color variations
   
   φ-SCALING:
   - Expands/contracts cluster spacing
   - Affects how colors blend
   
   This is the power of interpretable geometry:
   We can DIRECTLY CONTROL the transformation.
""")
