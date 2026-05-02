#!/usr/bin/env python3
"""
Modal Colorizer - Choosing Modes, Not Averaging

The real insight: Dull colors come from AVERAGING multiple color modes.
The phase transition isn't about scaling - it's about CHOOSING.

Example: A gray patch could be:
- Blue sky (mode 1)
- Green grass (mode 2)  
- Brown earth (mode 3)

Averaging gives muddy gray-brown. We should CHOOSE one mode.

The phase transition is:
- Low confidence → average (safe, dull)
- High confidence → commit to a mode (bold, saturated)

Like quantum measurement: superposition → collapse to eigenstate.

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI

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


class ModalColorizer:
    """
    Colorizer that chooses color MODES instead of averaging.
    
    Key insight: For each grayscale feature, there may be multiple
    valid color modes. Instead of averaging (→ dull), we should
    choose the most likely mode (→ saturated).
    
    The phase transition is confidence-based:
    - High confidence in one mode → commit to it
    - Low confidence → fall back to weighted average
    """
    
    def __init__(self, patch_size: int = 16, n_modes: int = 8):
        self.patch_size = patch_size
        self.n_modes = n_modes
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Color modes (cluster centers in UV space)
        self.mode_centers = None  # Shape: (n_modes, 2)
        self.mode_counts = None   # How many samples in each mode
        
        # Feature-to-mode mapping
        # For each feature region, which modes are likely?
        self.feature_to_joint = None
        self.joint_to_mode_weights = None  # Shape: (n_joints, n_modes)
        
        # Drum with mode labels
        self.drum_positions = []
        self.drum_colors = []
        self.drum_modes = []
        
        self.is_trained = False
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract features."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        lum = patch.mean()
        con = patch.std()
        tex_h = np.abs(np.diff(patch, axis=1)).mean() if w > 1 else 0
        tex_v = np.abs(np.diff(patch, axis=0)).mean() if h > 1 else 0
        con_tex = con * (tex_h + tex_v)
        pos_lum = y_pos * lum
        
        return np.array([lum, con, tex_h, tex_v, con_tex, pos_lum])
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """Train by finding color modes and feature-to-mode mapping."""
        print("   Collecting training data...")
        
        all_features = []
        all_colors = []
        
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
                    u = yuv_patch[:,:,1].mean()
                    v = yuv_patch[:,:,2].mean()
                    
                    all_features.append(feat)
                    all_colors.append([u, v])
        
        features = np.array(all_features)
        colors = np.array(all_colors)
        
        print(f"   Collected {len(features)} samples")
        
        # Step 1: Find color modes using k-means
        print(f"   Finding {self.n_modes} color modes...")
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_modes, random_state=42, n_init=10)
        mode_labels = kmeans.fit_predict(colors)
        self.mode_centers = kmeans.cluster_centers_
        
        # Count samples per mode
        self.mode_counts = np.bincount(mode_labels, minlength=self.n_modes)
        
        print("   Mode centers (U, V) and counts:")
        for i, (center, count) in enumerate(zip(self.mode_centers, self.mode_counts)):
            sat = np.sqrt(center[0]**2 + center[1]**2)
            print(f"     Mode {i}: ({center[0]:+.3f}, {center[1]:+.3f}), sat={sat:.3f}, n={count}")
        
        # Step 2: Normalize features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-10
        X = (features - self.feature_mean) / self.feature_std
        
        # Step 3: Feature-to-joint mapping
        U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
        n_joints = 6
        self.feature_to_joint = Vt[:n_joints]
        joints = X @ self.feature_to_joint.T
        
        # Step 4: Learn joint-to-mode weights
        # For each joint configuration, what's the probability of each mode?
        print("   Learning joint-to-mode mapping...")
        
        # Simple approach: for each mode, learn a linear classifier
        joints_bias = np.hstack([joints, np.ones((len(joints), 1))])
        
        # One-hot encode modes
        mode_onehot = np.zeros((len(mode_labels), self.n_modes))
        mode_onehot[np.arange(len(mode_labels)), mode_labels] = 1
        
        # Least squares to get weights
        self.joint_to_mode_weights = np.linalg.lstsq(joints_bias, mode_onehot, rcond=None)[0]
        
        # Store drum
        self.drum_positions = joints
        self.drum_colors = colors
        self.drum_modes = mode_labels
        
        self.is_trained = True
        
        # Compute accuracy
        pred_mode_probs = joints_bias @ self.joint_to_mode_weights
        pred_modes = np.argmax(pred_mode_probs, axis=1)
        accuracy = (pred_modes == mode_labels).mean()
        print(f"   Mode prediction accuracy: {accuracy*100:.1f}%")
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      mode_selection: str = 'max') -> Tuple[float, float]:
        """
        Predict color by choosing a mode.
        
        mode_selection:
        - 'max': Choose the most likely mode (bold)
        - 'weighted': Weighted average of modes (safe but dull)
        - 'sample': Sample from mode distribution (stochastic)
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract and normalize
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Map to joints
        joints = self.feature_to_joint @ feat_norm
        joints_bias = np.concatenate([joints, [1.0]])
        
        # Get mode probabilities
        mode_logits = joints_bias @ self.joint_to_mode_weights
        
        # Softmax
        mode_logits = mode_logits - mode_logits.max()  # Stability
        mode_probs = np.exp(mode_logits) / np.exp(mode_logits).sum()
        
        if mode_selection == 'max':
            # Choose the most likely mode
            best_mode = np.argmax(mode_probs)
            u, v = self.mode_centers[best_mode]
            
        elif mode_selection == 'weighted':
            # Weighted average of mode centers
            u = np.sum(mode_probs * self.mode_centers[:, 0])
            v = np.sum(mode_probs * self.mode_centers[:, 1])
            
        elif mode_selection == 'sample':
            # Sample from distribution
            sampled_mode = np.random.choice(self.n_modes, p=mode_probs)
            u, v = self.mode_centers[sampled_mode]
            
        elif mode_selection == 'confident':
            # Use max if confident, weighted if uncertain
            confidence = mode_probs.max()
            threshold = 1.0 / self.n_modes * 2  # 2x uniform
            
            if confidence > threshold:
                best_mode = np.argmax(mode_probs)
                u, v = self.mode_centers[best_mode]
            else:
                u = np.sum(mode_probs * self.mode_centers[:, 0])
                v = np.sum(mode_probs * self.mode_centers[:, 1])
        
        else:
            raise ValueError(f"Unknown mode_selection: {mode_selection}")
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray, mode_selection: str = 'max') -> np.ndarray:
        """Colorize using mode selection."""
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
                
                u, v = self.predict_color(patch, y_pos, x_pos, mode_selection)
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


def run_modal_test():
    """Test modal colorizer."""
    print("=" * 70)
    print("MODAL COLORIZER")
    print("Choosing modes instead of averaging")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. TRAINING")
    print("-" * 50)
    colorizer = ModalColorizer(patch_size=16, n_modes=8)
    colorizer.train(train_images, sample_rate=0.12)
    
    print("\n2. TESTING DIFFERENT MODE SELECTIONS")
    print("-" * 50)
    
    results = []
    for mode_sel in ['weighted', 'max', 'confident']:
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, mode_selection=mode_sel)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
        
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, mode_selection=mode_sel)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        test_mae = np.mean(test_errors)
        gen_mae = np.mean(gen_errors)
        
        print(f"   {mode_sel:>10}: Test={test_mae:.2f}, Gen={gen_mae:.2f}")
        
        results.append({
            'mode_sel': mode_sel,
            'test_mae': test_mae,
            'gen_mae': gen_mae
        })
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i, (name, img) in enumerate(test_data[:3]):
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        
        col_weighted = colorizer.colorize(gray, mode_selection='weighted')
        col_max = colorizer.colorize(gray, mode_selection='max')
        
        err_w = np.abs(col_weighted.astype(float) - img.astype(float)).mean()
        err_m = np.abs(col_max.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(col_weighted)
        axes[i, 2].set_title(f'Weighted ({err_w:.1f})' if i == 0 else f'{err_w:.1f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(col_max)
        axes[i, 3].set_title(f'Max mode ({err_m:.1f})' if i == 0 else f'{err_m:.1f}')
        axes[i, 3].axis('off')
        
        diff = np.abs(col_max.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 4].set_title('Error' if i == 0 else '')
        axes[i, 4].axis('off')
    
    fig.suptitle(f'Modal Colorizer: {colorizer.n_modes} modes',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "modal_colorizer_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'modal_colorizer_test.png'}")
    
    # Visualize mode centers
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, (center, count) in enumerate(zip(colorizer.mode_centers, colorizer.mode_counts)):
        size = 100 + count / 10
        ax.scatter(center[0], center[1], s=size, label=f'Mode {i} (n={count})')
        ax.annotate(f'{i}', (center[0], center[1]), fontsize=12, ha='center', va='center')
    
    ax.set_xlabel('U (blue-yellow)')
    ax.set_ylabel('V (red-green)')
    ax.set_title('Color Mode Centers')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.savefig(OUTPUT_PATH / "modal_colorizer_modes.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_modal_test()
    
    print("\n" + "=" * 70)
    print("MODAL COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   The insight: Dull colors come from AVERAGING multiple modes.
   
   Instead of averaging, we CHOOSE:
   - 'weighted': Average of mode centers (safe, dull)
   - 'max': Choose most likely mode (bold, saturated)
   - 'confident': Max if confident, weighted if uncertain
   
   Results:
""")
    for r in results:
        print(f"     {r['mode_sel']:>10}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
    
    print(f"""
   The phase transition is about COMMITMENT:
   - Low confidence → hedge bets → average → dull
   - High confidence → commit → choose mode → saturated
   
   Like quantum measurement:
   - Superposition of color modes
   - Measurement collapses to one eigenstate
   - The "measurement" is our confidence threshold
""")
