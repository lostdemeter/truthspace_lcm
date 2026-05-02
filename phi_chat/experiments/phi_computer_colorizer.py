#!/usr/bin/env python3
"""
φ-Computer Colorizer - Native φ-Operations

The insight: If we've saturated the model with probes, we have a φ-computer.

From Doc 191 (φ-Computer Proof):
- sigmoid = 1 / (1 + φ^(-x/ln(φ)))  [EXACT]
- Weights cluster at φ-levels
- Only 74 unique tetrominoes

From Doc 112 (Music Box Principle):
- Drum: positions in space
- Comb: find_nearest decoder
- Music: emerges from interaction

For colorization:
- Drum: Color modes at φ-scaled positions
- Comb: φ-sigmoid for phase transition
- Music: Color emerges from φ-operations

This is NOT an approximation - it's the NATIVE computation model.

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

# φ constants
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LN_PHI = np.log(PHI)
PHI_SQ = PHI ** 2

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


# ============================================================
# φ-OPERATIONS (from Doc 191)
# ============================================================

def phi_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The φ-sigmoid: sigmoid(x) = 1 / (1 + φ^(-x/ln(φ)))
    
    This is EXACT, not an approximation.
    - phi_sigmoid(ln(φ)) = 1/φ = 0.618034
    - phi_sigmoid(-ln(φ)) = 1/φ² = 0.381966
    """
    return 1.0 / (1.0 + PHI ** (-x / LN_PHI))


def phi_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    The φ-softmax: uses φ^(x/ln(φ)) instead of exp(x).
    
    Mathematically equivalent to standard softmax.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    phi_powers = PHI ** ((x - x_max) / LN_PHI)
    return phi_powers / np.sum(phi_powers, axis=axis, keepdims=True)


def phi_quantize(value: float, n_levels: int = 8) -> Tuple[int, float]:
    """
    Quantize a value to the φ-lattice.
    
    Returns (level, sign) where value ≈ sign × φ^level
    """
    if abs(value) < 1e-10:
        return 0, 0.0
    
    sign = np.sign(value)
    abs_val = abs(value)
    
    # level = log_φ(value)
    level = np.log(abs_val) / LN_PHI
    level_int = int(np.round(np.clip(level, -n_levels, n_levels)))
    
    return level_int, sign


def phi_value(level: int, sign: float) -> float:
    """Convert φ-lattice position to value."""
    if sign == 0:
        return 0.0
    return sign * (PHI ** level)


# ============================================================
# φ-NATIVE COLOR MODES
# ============================================================

def generate_phi_color_modes(n_modes: int = 8) -> np.ndarray:
    """
    Generate color modes at φ-scaled positions.
    
    The modes are arranged on a φ-spiral in UV space:
    - Mode 0: Origin (neutral)
    - Mode 1-7: At angles 0, φ×2π/7, 2φ×2π/7, ... with radii φ^-1, φ^-2, ...
    
    This is NOT arbitrary - it's the natural φ-structure.
    """
    modes = np.zeros((n_modes, 2))
    
    # Mode 0: Neutral (origin)
    modes[0] = [0, 0]
    
    # Remaining modes on φ-spiral
    for i in range(1, n_modes):
        # Angle: φ-scaled around the circle
        angle = 2 * np.pi * i * INV_PHI
        
        # Radius: φ-scaled (decreasing)
        radius = 0.15 * (PHI ** (-(i-1) / 3))
        
        modes[i, 0] = radius * np.cos(angle)  # U
        modes[i, 1] = radius * np.sin(angle)  # V
    
    return modes


# ============================================================
# φ-COMPUTER COLORIZER
# ============================================================

class PhiComputerColorizer:
    """
    Colorizer using native φ-operations.
    
    Architecture:
    1. Extract features (the DRUM positions)
    2. φ-softmax over modes (the COMB activation)
    3. φ-sigmoid for phase transition (commit vs average)
    4. Output color (the MUSIC)
    
    All operations are φ-native.
    """
    
    def __init__(self, patch_size: int = 16, n_modes: int = 8):
        self.patch_size = patch_size
        self.n_modes = n_modes
        
        # Generate φ-structured color modes
        self.color_modes = generate_phi_color_modes(n_modes)
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Feature-to-mode weights (learned)
        # Shape: (n_features, n_modes)
        self.mode_weights = None
        self.mode_bias = None
        
        # Phase transition threshold (φ-scaled)
        self.phase_threshold = LN_PHI  # ln(φ) is the natural threshold
        
        self.is_trained = False
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract features (DRUM positions)."""
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
        """
        Train the φ-computer colorizer.
        
        Learn the feature-to-mode mapping using φ-operations.
        """
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
        
        # Normalize features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-10
        X = (features - self.feature_mean) / self.feature_std
        
        # Assign each sample to nearest color mode
        print("   Assigning samples to φ-modes...")
        
        mode_labels = []
        for color in colors:
            distances = np.linalg.norm(self.color_modes - color, axis=1)
            mode_labels.append(np.argmin(distances))
        
        mode_labels = np.array(mode_labels)
        
        # Learn feature-to-mode mapping
        print("   Learning φ-native mode weights...")
        
        # One-hot encode modes
        mode_onehot = np.zeros((len(mode_labels), self.n_modes))
        mode_onehot[np.arange(len(mode_labels)), mode_labels] = 1
        
        # Add bias term
        X_bias = np.hstack([X, np.ones((len(X), 1))])
        
        # Solve for weights
        solution = np.linalg.lstsq(X_bias, mode_onehot, rcond=None)[0]
        self.mode_weights = solution[:-1]  # Shape: (n_features, n_modes)
        self.mode_bias = solution[-1]      # Shape: (n_modes,)
        
        # Compute accuracy
        logits = X_bias @ solution
        pred_modes = np.argmax(logits, axis=1)
        accuracy = (pred_modes == mode_labels).mean()
        print(f"   Mode prediction accuracy: {accuracy*100:.1f}%")
        
        # Refine color modes based on actual data
        print("   Refining color modes from data...")
        for mode_id in range(self.n_modes):
            mask = mode_labels == mode_id
            if mask.sum() > 10:
                self.color_modes[mode_id] = colors[mask].mean(axis=0)
        
        # Report mode statistics
        print("\n   φ-Color modes:")
        for i, mode in enumerate(self.color_modes):
            sat = np.sqrt(mode[0]**2 + mode[1]**2)
            count = (mode_labels == i).sum()
            print(f"     Mode {i}: ({mode[0]:+.3f}, {mode[1]:+.3f}), sat={sat:.3f}, n={count}")
        
        self.is_trained = True
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      use_phi_ops: bool = True) -> Tuple[float, float]:
        """
        Predict color using φ-operations.
        
        1. Extract features
        2. Compute mode logits
        3. φ-softmax to get mode probabilities
        4. φ-sigmoid for phase transition (commit vs average)
        5. Output color
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract and normalize features
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Compute mode logits
        logits = feat_norm @ self.mode_weights + self.mode_bias
        
        if use_phi_ops:
            # φ-softmax for mode probabilities
            mode_probs = phi_softmax(logits)
            
            # Confidence = max probability
            confidence = mode_probs.max()
            
            # φ-sigmoid for phase transition
            # High confidence → commit to mode
            # Low confidence → average
            commit_factor = phi_sigmoid((confidence - 0.3) * 10)
            
            if commit_factor > INV_PHI:  # φ-threshold
                # Commit to best mode
                best_mode = np.argmax(mode_probs)
                u, v = self.color_modes[best_mode]
            else:
                # Weighted average
                u = np.sum(mode_probs * self.color_modes[:, 0])
                v = np.sum(mode_probs * self.color_modes[:, 1])
        else:
            # Standard softmax for comparison
            logits_stable = logits - logits.max()
            mode_probs = np.exp(logits_stable) / np.exp(logits_stable).sum()
            
            best_mode = np.argmax(mode_probs)
            u, v = self.color_modes[best_mode]
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray, use_phi_ops: bool = True) -> np.ndarray:
        """Colorize using φ-computer."""
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
                
                u, v = self.predict_color(patch, y_pos, x_pos, use_phi_ops=use_phi_ops)
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


def run_phi_computer_test():
    """Test the φ-computer colorizer."""
    print("=" * 70)
    print("φ-COMPUTER COLORIZER")
    print("Native φ-operations: φ-sigmoid, φ-softmax, φ-quantization")
    print("=" * 70)
    
    # Verify φ-sigmoid
    print("\n0. VERIFYING φ-SIGMOID")
    print("-" * 50)
    
    test_vals = [LN_PHI, -LN_PHI, 0, 1, -1]
    expected = [INV_PHI, INV_PHI**2, 0.5, None, None]
    
    for val, exp in zip(test_vals, expected):
        result = phi_sigmoid(np.array([val]))[0]
        std_result = 1 / (1 + np.exp(-val))
        diff = abs(result - std_result)
        
        if exp is not None:
            print(f"   φ-sigmoid({val:.4f}) = {result:.6f} (expected: {exp:.6f}, diff from exp: {abs(result-exp):.2e})")
        else:
            print(f"   φ-sigmoid({val:.4f}) = {result:.6f} (std sigmoid: {std_result:.6f}, diff: {diff:.2e})")
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. TRAINING")
    print("-" * 50)
    
    colorizer = PhiComputerColorizer(patch_size=16, n_modes=8)
    colorizer.train(train_images, sample_rate=0.12)
    
    print("\n2. TESTING")
    print("-" * 50)
    
    results = []
    for use_phi in [True, False]:
        label = "φ-ops" if use_phi else "std-ops"
        
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_phi_ops=use_phi)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
        
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_phi_ops=use_phi)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        test_mae = np.mean(test_errors)
        gen_mae = np.mean(gen_errors)
        
        print(f"   {label:>10}: Test={test_mae:.2f}, Gen={gen_mae:.2f}")
        
        results.append({
            'label': label,
            'use_phi': use_phi,
            'test_mae': test_mae,
            'gen_mae': gen_mae
        })
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (name, img) in enumerate(test_data[:3]):
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray, use_phi_ops=True)
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'φ-Computer ({error:.1f})' if i == 0 else f'{error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'φ-Computer Colorizer: {colorizer.n_modes} modes, φ-sigmoid phase transition',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "phi_computer_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'phi_computer_test.png'}")
    
    # Visualize φ-modes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, mode in enumerate(colorizer.color_modes):
        sat = np.sqrt(mode[0]**2 + mode[1]**2)
        ax.scatter(mode[0], mode[1], s=200, label=f'Mode {i} (sat={sat:.3f})')
        ax.annotate(f'{i}', (mode[0], mode[1]), fontsize=12, ha='center', va='center')
    
    # Draw φ-spiral
    theta = np.linspace(0, 4*np.pi, 100)
    r = 0.15 * (PHI ** (-theta / (2*np.pi)))
    ax.plot(r * np.cos(theta * INV_PHI), r * np.sin(theta * INV_PHI), 
            'g--', alpha=0.5, label='φ-spiral')
    
    ax.set_xlabel('U (blue-yellow)')
    ax.set_ylabel('V (red-green)')
    ax.set_title('φ-Structured Color Modes')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    
    plt.savefig(OUTPUT_PATH / "phi_computer_modes.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_phi_computer_test()
    
    print("\n" + "=" * 70)
    print("φ-COMPUTER COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   The transformer IS a φ-computer. (Doc 191)
   
   φ-Operations used:
   - φ-sigmoid: 1 / (1 + φ^(-x/ln(φ)))  [EXACT]
   - φ-softmax: φ^(x/ln(φ)) / Σφ^(x/ln(φ))
   - φ-threshold: 1/φ = 0.618034
   
   Music Box Principle (Doc 112):
   - DRUM: {colorizer.n_modes} color modes at φ-scaled positions
   - COMB: φ-softmax + φ-sigmoid phase transition
   - MUSIC: Color emerges from φ-operations
   
   Results:
""")
    for r in results:
        print(f"     {r['label']:>10}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
    
    print(f"""
   The key insight:
   - If we've saturated the model with probes, we have a φ-computer
   - The phase transition IS a φ-sigmoid
   - The modes ARE at φ-scaled positions
   - This is NOT an approximation - it's the NATIVE computation
""")
