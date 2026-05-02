#!/usr/bin/env python3
"""
Phase Transition Colorizer - Finding the Critical Point

The insight: Dull colors come from linear averaging.
Real color has PHASE TRANSITIONS:
- Below threshold: neutral (gray)
- Above threshold: saturated color "activates"

Like the 137/30 barrier in attention, there's a critical point
where color behavior changes qualitatively.

We need to:
1. Find the phase transition threshold
2. Apply non-linear activation at that threshold
3. Colors should "snap" to saturated or neutral, not blend

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.stats import gaussian_kde
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


def analyze_color_distribution(images: List[np.ndarray], sample_rate: float = 0.1):
    """
    Analyze the distribution of color saturation to find phase transitions.
    """
    print("   Analyzing color distribution...")
    
    all_saturation = []
    all_u = []
    all_v = []
    
    for img in images:
        yuv = rgb_to_yuv(img)
        u = yuv[:, :, 1].flatten()
        v = yuv[:, :, 2].flatten()
        
        # Sample
        n_samples = int(len(u) * sample_rate)
        indices = np.random.choice(len(u), n_samples, replace=False)
        
        all_u.extend(u[indices])
        all_v.extend(v[indices])
        
        # Saturation = sqrt(u² + v²)
        sat = np.sqrt(u[indices]**2 + v[indices]**2)
        all_saturation.extend(sat)
    
    saturation = np.array(all_saturation)
    u_vals = np.array(all_u)
    v_vals = np.array(all_v)
    
    print(f"   Collected {len(saturation)} samples")
    print(f"   Saturation range: [{saturation.min():.4f}, {saturation.max():.4f}]")
    print(f"   Saturation mean: {saturation.mean():.4f}")
    print(f"   Saturation std: {saturation.std():.4f}")
    
    # Find the distribution
    # Look for bimodality (two peaks = phase transition)
    
    # Histogram
    hist, bin_edges = np.histogram(saturation, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=0.5, distance=10)
    
    print(f"\n   Distribution peaks at saturation:")
    for p in peaks:
        print(f"     {bin_centers[p]:.4f} (density: {hist[p]:.2f})")
    
    # Find the valley between peaks (phase transition point)
    if len(peaks) >= 2:
        valley_region = hist[peaks[0]:peaks[1]]
        valley_idx = peaks[0] + np.argmin(valley_region)
        transition_point = bin_centers[valley_idx]
        print(f"\n   Phase transition at: {transition_point:.4f}")
    else:
        # Use percentile-based threshold
        transition_point = np.percentile(saturation, 50)
        print(f"\n   No clear bimodality. Using median: {transition_point:.4f}")
    
    # Check for φ-relationship
    phi_threshold = saturation.mean() * INV_PHI
    print(f"   φ-scaled threshold (mean/φ): {phi_threshold:.4f}")
    
    return {
        'saturation': saturation,
        'u': u_vals,
        'v': v_vals,
        'hist': hist,
        'bin_centers': bin_centers,
        'peaks': peaks,
        'transition_point': transition_point,
        'phi_threshold': phi_threshold
    }


class PhaseTransitionColorizer:
    """
    Colorizer that applies phase transition to color saturation.
    
    Instead of linear blending (which gives dull colors),
    we apply a non-linear activation that "snaps" colors
    to either saturated or neutral.
    """
    
    def __init__(self, patch_size: int = 16, n_joints: int = 6):
        self.patch_size = patch_size
        self.n_joints = n_joints
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Kinematic model
        self.feature_to_joint = None
        self.jacobian = None
        self.bias = None
        
        # Phase transition parameters
        self.transition_threshold = 0.05  # Will be learned
        self.activation_sharpness = 10.0  # How sharp the transition is
        
        # Drum for lookup
        self.drum_positions = []
        self.drum_colors = []
        
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
    
    def phase_activation(self, saturation: float) -> float:
        """
        Apply phase transition activation.
        
        Below threshold: suppress (multiply by small factor)
        Above threshold: amplify (multiply by large factor)
        
        Uses sigmoid-like transition for smoothness.
        """
        # Sigmoid centered at threshold
        x = (saturation - self.transition_threshold) * self.activation_sharpness
        activation = 1.0 / (1.0 + np.exp(-x))
        
        # Scale: below threshold → 0.3x, above threshold → 1.5x
        scale = 0.3 + 1.2 * activation
        
        return scale
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """Train and find phase transition."""
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
        
        # Find phase transition in color distribution
        saturation = np.sqrt(colors[:, 0]**2 + colors[:, 1]**2)
        
        # Use percentile-based threshold (bimodal detection)
        # The transition is where the derivative of the CDF changes most
        sorted_sat = np.sort(saturation)
        cdf = np.arange(len(sorted_sat)) / len(sorted_sat)
        
        # Find inflection point (where second derivative is maximum)
        # This is the phase transition
        window = len(sorted_sat) // 20
        derivatives = []
        for i in range(window, len(sorted_sat) - window):
            d1 = (cdf[i] - cdf[i-window]) / (sorted_sat[i] - sorted_sat[i-window] + 1e-10)
            d2 = (cdf[i+window] - cdf[i]) / (sorted_sat[i+window] - sorted_sat[i] + 1e-10)
            derivatives.append(abs(d2 - d1))
        
        if derivatives:
            inflection_idx = window + np.argmax(derivatives)
            self.transition_threshold = sorted_sat[inflection_idx]
        else:
            self.transition_threshold = np.median(saturation)
        
        print(f"   Phase transition threshold: {self.transition_threshold:.4f}")
        
        # Also try φ-scaled threshold
        phi_threshold = saturation.mean() * INV_PHI
        print(f"   φ-scaled threshold: {phi_threshold:.4f}")
        
        # Normalize features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-10
        X = (features - self.feature_mean) / self.feature_std
        
        # Feature-to-joint mapping
        U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.feature_to_joint = Vt[:self.n_joints]
        
        # Compute joints
        joints = X @ self.feature_to_joint.T
        
        # Learn Jacobian
        joints_bias = np.hstack([joints, np.ones((len(joints), 1))])
        solution = np.linalg.lstsq(joints_bias, colors, rcond=None)[0]
        self.jacobian = solution[:-1].T
        self.bias = solution[-1]
        
        # Store drum
        self.drum_positions = joints
        self.drum_colors = colors
        
        self.is_trained = True
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      use_phase: bool = True) -> Tuple[float, float]:
        """Predict color with optional phase transition."""
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract and normalize
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Map to joints
        joints = self.feature_to_joint @ feat_norm
        
        # Forward kinematics
        color = self.jacobian @ joints + self.bias
        u, v = color[0], color[1]
        
        if use_phase:
            # Apply phase transition
            saturation = np.sqrt(u**2 + v**2)
            scale = self.phase_activation(saturation)
            u *= scale
            v *= scale
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray, use_phase: bool = True) -> np.ndarray:
        """Colorize with phase transition."""
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
                
                u, v = self.predict_color(patch, y_pos, x_pos, use_phase=use_phase)
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


def run_phase_transition_test():
    """Test phase transition colorizer."""
    print("=" * 70)
    print("PHASE TRANSITION COLORIZER")
    print("Finding the critical point where color activates")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    # Analyze color distribution
    print("\n1. ANALYZING COLOR DISTRIBUTION")
    print("-" * 50)
    analysis = analyze_color_distribution(train_images[:50])
    
    # Visualize distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Saturation histogram
    axes[0].hist(analysis['saturation'], bins=100, density=True, alpha=0.7)
    axes[0].axvline(analysis['transition_point'], color='r', linestyle='--', label=f'Transition: {analysis["transition_point"]:.3f}')
    axes[0].axvline(analysis['phi_threshold'], color='g', linestyle='--', label=f'φ-threshold: {analysis["phi_threshold"]:.3f}')
    axes[0].set_xlabel('Saturation')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Color Saturation Distribution')
    axes[0].legend()
    
    # U-V scatter
    sample_idx = np.random.choice(len(analysis['u']), min(5000, len(analysis['u'])), replace=False)
    axes[1].scatter(analysis['u'][sample_idx], analysis['v'][sample_idx], alpha=0.1, s=1)
    axes[1].set_xlabel('U (blue-yellow)')
    axes[1].set_ylabel('V (red-green)')
    axes[1].set_title('Color Distribution in UV Space')
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_ylim(-0.5, 0.5)
    
    # Add threshold circle
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(analysis['transition_point'] * np.cos(theta), 
                 analysis['transition_point'] * np.sin(theta), 
                 'r--', label='Transition')
    
    # Activation function
    sat_range = np.linspace(0, 0.3, 100)
    colorizer_temp = PhaseTransitionColorizer()
    colorizer_temp.transition_threshold = analysis['transition_point']
    activations = [colorizer_temp.phase_activation(s) for s in sat_range]
    axes[2].plot(sat_range, activations)
    axes[2].axvline(analysis['transition_point'], color='r', linestyle='--')
    axes[2].set_xlabel('Saturation')
    axes[2].set_ylabel('Scale Factor')
    axes[2].set_title('Phase Transition Activation')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "phase_transition_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Train colorizer
    print("\n2. TRAINING")
    print("-" * 50)
    colorizer = PhaseTransitionColorizer(patch_size=16, n_joints=6)
    colorizer.train(train_images, sample_rate=0.12)
    
    # Test with different sharpness values
    print("\n3. TESTING DIFFERENT SHARPNESS VALUES")
    print("-" * 50)
    
    results = []
    for sharpness in [0, 5, 10, 20, 50]:
        colorizer.activation_sharpness = sharpness
        use_phase = sharpness > 0
        
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_phase=use_phase)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
        
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_phase=use_phase)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        test_mae = np.mean(test_errors)
        gen_mae = np.mean(gen_errors)
        
        label = "No phase" if sharpness == 0 else f"Sharpness={sharpness}"
        print(f"   {label}: Test={test_mae:.2f}, Gen={gen_mae:.2f}")
        
        results.append({
            'sharpness': sharpness,
            'test_mae': test_mae,
            'gen_mae': gen_mae
        })
    
    # Visualize comparison
    print("\n4. VISUALIZATION")
    print("-" * 50)
    
    colorizer.activation_sharpness = 10
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i, (name, img) in enumerate(test_data[:3]):
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        
        col_no_phase = colorizer.colorize(gray, use_phase=False)
        col_with_phase = colorizer.colorize(gray, use_phase=True)
        
        err_no = np.abs(col_no_phase.astype(float) - img.astype(float)).mean()
        err_yes = np.abs(col_with_phase.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(col_no_phase)
        axes[i, 2].set_title(f'No phase ({err_no:.1f})' if i == 0 else f'{err_no:.1f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(col_with_phase)
        axes[i, 3].set_title(f'With phase ({err_yes:.1f})' if i == 0 else f'{err_yes:.1f}')
        axes[i, 3].axis('off')
        
        diff = np.abs(col_with_phase.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 4].set_title('Error' if i == 0 else '')
        axes[i, 4].axis('off')
    
    fig.suptitle(f'Phase Transition: threshold={colorizer.transition_threshold:.3f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "phase_transition_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'phase_transition_test.png'}")
    
    return colorizer, analysis, results


if __name__ == "__main__":
    colorizer, analysis, results = run_phase_transition_test()
    
    print("\n" + "=" * 70)
    print("PHASE TRANSITION SUMMARY")
    print("=" * 70)
    print(f"""
   The insight: Linear averaging gives dull colors.
   Real color has PHASE TRANSITIONS.
   
   Found transition threshold: {colorizer.transition_threshold:.4f}
   φ-scaled threshold: {analysis['phi_threshold']:.4f}
   
   Results by sharpness:
""")
    for r in results:
        label = "No phase" if r['sharpness'] == 0 else f"Sharpness={r['sharpness']}"
        print(f"     {label:>15}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
    
    print(f"""
   The phase transition:
   - Below threshold: suppress color (×0.3)
   - Above threshold: amplify color (×1.5)
   - Transition is smooth (sigmoid)
   
   This is like the 137/30 barrier:
   - A critical point where behavior changes qualitatively
   - Not just quantitative scaling, but phase change
""")
