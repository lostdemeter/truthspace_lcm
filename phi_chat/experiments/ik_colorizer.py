#!/usr/bin/env python3
"""
IK-Inspired Colorizer - Inverse Kinematics for Color

The analogy:
- End effector = target color (U, V)
- Joint angles = φ-dimension values  
- Joint limits = φ-grid quantization
- Kinematic chain = grayscale → φ-space → color

Key IK principles applied:
1. Don't search all configurations - SOLVE for the one that reaches target
2. Constraints (joint limits / φ-grid) reduce solution space
3. The Jacobian tells us how joints affect end position
4. Redundant DOF can be exploited for secondary objectives

For colorization:
- Learn the "Jacobian" - how each φ-dimension affects color
- Given grayscale features, solve for the φ-configuration
- The φ-grid constrains valid solutions
- Redundancy allows for style/preference

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LOG_PHI = np.log(PHI)

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


class IKColorizer:
    """
    Inverse Kinematics approach to colorization.
    
    The "kinematic chain":
    grayscale features → φ-joints → end effector (color)
    
    We learn:
    1. The forward kinematics: φ-joints → color (the Jacobian)
    2. The inverse: given features, solve for φ-joints that produce valid color
    """
    
    def __init__(self, patch_size: int = 16, n_joints: int = 4, n_phi_levels: int = 8):
        self.patch_size = patch_size
        self.n_joints = n_joints  # Number of φ-dimensions (DOF)
        self.n_phi_levels = n_phi_levels
        
        # Joint limits (φ-grid positions)
        self.joint_limits = self._compute_joint_limits()
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # The "Jacobian" - how joints affect color
        # J[i, j] = ∂color_i / ∂joint_j
        self.jacobian_u = None  # Shape: (n_joints,)
        self.jacobian_v = None  # Shape: (n_joints,)
        
        # Bias terms
        self.bias_u = 0.0
        self.bias_v = 0.0
        
        # Feature-to-joint mapping (how features determine joint angles)
        self.feature_to_joint = None  # Shape: (n_joints, n_features)
        
        self.is_trained = False
    
    def _compute_joint_limits(self) -> np.ndarray:
        """Compute valid φ-grid positions (joint limits)."""
        levels = np.arange(-self.n_phi_levels, self.n_phi_levels + 1)
        positions = []
        for level in levels:
            positions.append(PHI ** level)
            positions.append(-PHI ** level)
        positions.append(0.0)
        return np.unique(np.array(positions))
    
    def _snap_to_joint_limit(self, value: float) -> float:
        """Snap a value to the nearest valid joint position."""
        idx = np.argmin(np.abs(self.joint_limits - value))
        return self.joint_limits[idx]
    
    def extract_features(self, gray_patch: np.ndarray, 
                         y_pos: float, x_pos: float) -> np.ndarray:
        """Extract features from grayscale patch."""
        patch = gray_patch.astype(np.float32) / 255.0
        h, w = patch.shape
        
        lum = patch.mean()
        con = patch.std()
        tex_h = np.abs(np.diff(patch, axis=1)).mean() if w > 1 else 0
        tex_v = np.abs(np.diff(patch, axis=0)).mean() if h > 1 else 0
        con_tex = con * (tex_h + tex_v)
        pos_lum = y_pos * lum
        
        return np.array([lum, con, tex_h, tex_v, con_tex, pos_lum])
    
    def features_to_joints(self, features: np.ndarray) -> np.ndarray:
        """
        Map features to joint angles.
        
        This is like computing the "desired" joint configuration
        from the task space (grayscale features).
        """
        if self.feature_to_joint is None:
            return features[:self.n_joints]
        
        # Normalize features
        feat_norm = (features - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Linear mapping to joint space
        joints = self.feature_to_joint @ feat_norm
        
        # Snap to valid joint positions (φ-grid)
        joints_snapped = np.array([self._snap_to_joint_limit(j) for j in joints])
        
        return joints_snapped
    
    def forward_kinematics(self, joints: np.ndarray) -> Tuple[float, float]:
        """
        Compute color from joint angles.
        
        color = J @ joints + bias
        
        This is the "forward kinematics" - given joint angles, where is the end effector?
        """
        if self.jacobian_u is None:
            return 0.0, 0.0
        
        u = np.dot(self.jacobian_u, joints) + self.bias_u
        v = np.dot(self.jacobian_v, joints) + self.bias_v
        
        return u, v
    
    def inverse_kinematics(self, features: np.ndarray, 
                           target_u: Optional[float] = None,
                           target_v: Optional[float] = None) -> np.ndarray:
        """
        Solve for joint angles that produce target color.
        
        If no target given, use the learned feature-to-joint mapping.
        If target given, solve the IK problem.
        """
        # Start from feature-based estimate
        joints_init = self.features_to_joints(features)
        
        if target_u is None or target_v is None:
            return joints_init
        
        # Solve IK: find joints that minimize |FK(joints) - target|²
        # Subject to: joints on φ-grid
        
        def objective(joints):
            u, v = self.forward_kinematics(joints)
            return (u - target_u)**2 + (v - target_v)**2
        
        # Simple gradient descent with snapping
        joints = joints_init.copy()
        lr = 0.1
        
        for _ in range(10):
            u, v = self.forward_kinematics(joints)
            
            # Gradient: ∂loss/∂joints = 2 * (FK - target) * J
            grad_u = 2 * (u - target_u) * self.jacobian_u
            grad_v = 2 * (v - target_v) * self.jacobian_v
            grad = grad_u + grad_v
            
            # Update
            joints = joints - lr * grad
            
            # Snap to φ-grid
            joints = np.array([self._snap_to_joint_limit(j) for j in joints])
        
        return joints
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """
        Learn the kinematic model.
        
        1. Collect (features, color) pairs
        2. Learn feature-to-joint mapping
        3. Learn Jacobian (joint-to-color mapping)
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
        
        # Learn feature-to-joint mapping using SVD
        # We want to find a low-rank mapping that captures the essential structure
        print("   Learning feature-to-joint mapping...")
        
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Use top n_joints components
        self.feature_to_joint = Vt[:self.n_joints]  # Shape: (n_joints, n_features)
        
        # Compute joint positions for all samples
        joints = X @ self.feature_to_joint.T  # Shape: (n_samples, n_joints)
        
        # Snap to φ-grid
        joints_snapped = np.zeros_like(joints)
        for i in range(len(joints)):
            for j in range(self.n_joints):
                joints_snapped[i, j] = self._snap_to_joint_limit(joints[i, j])
        
        # Learn Jacobian: color = J @ joints + bias
        print("   Learning Jacobian (joint → color mapping)...")
        
        # Add bias term
        joints_bias = np.hstack([joints_snapped, np.ones((len(joints_snapped), 1))])
        
        # Solve for U channel
        solution_u = np.linalg.lstsq(joints_bias, colors[:, 0], rcond=None)[0]
        self.jacobian_u = solution_u[:-1]
        self.bias_u = solution_u[-1]
        
        # Solve for V channel
        solution_v = np.linalg.lstsq(joints_bias, colors[:, 1], rcond=None)[0]
        self.jacobian_v = solution_v[:-1]
        self.bias_v = solution_v[-1]
        
        # Compute R²
        u_pred = joints_bias @ solution_u
        v_pred = joints_bias @ solution_v
        
        r2_u = 1 - np.sum((colors[:, 0] - u_pred)**2) / np.sum((colors[:, 0] - colors[:, 0].mean())**2)
        r2_v = 1 - np.sum((colors[:, 1] - v_pred)**2) / np.sum((colors[:, 1] - colors[:, 1].mean())**2)
        
        print(f"   Jacobian R²: U={r2_u:.4f}, V={r2_v:.4f}")
        print(f"   Jacobian U: {self.jacobian_u.round(4)}")
        print(f"   Jacobian V: {self.jacobian_v.round(4)}")
        
        self.is_trained = True
        
        return r2_u, r2_v
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float) -> Tuple[float, float]:
        """
        Predict color using the kinematic chain.
        
        features → joints → color
        """
        if not self.is_trained:
            return 0.0, 0.0
        
        # Extract features
        features = self.extract_features(gray_patch, y_pos, x_pos)
        
        # Map to joint angles
        joints = self.features_to_joints(features)
        
        # Forward kinematics to get color
        u, v = self.forward_kinematics(joints)
        
        return u, v
    
    def refine_with_target(self, gray_patch: np.ndarray,
                           y_pos: float, x_pos: float,
                           target_u: float, target_v: float) -> Tuple[float, float]:
        """
        Refine prediction using IK to reach target color.
        
        This is like using IK to correct the end effector position.
        """
        features = self.extract_features(gray_patch, y_pos, x_pos)
        
        # Solve IK for target
        joints = self.inverse_kinematics(features, target_u, target_v)
        
        # Forward kinematics with refined joints
        u, v = self.forward_kinematics(joints)
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize using the kinematic chain."""
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


def run_ik_test():
    """Test the IK colorizer."""
    print("=" * 70)
    print("IK COLORIZER")
    print("Inverse Kinematics for Color")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    # Test different joint configurations
    print("\n1. TESTING CONFIGURATIONS")
    print("-" * 50)
    
    results = []
    
    for n_joints in [2, 3, 4, 5, 6]:
        print(f"\n   === {n_joints} joints (DOF) ===")
        
        colorizer = IKColorizer(patch_size=16, n_joints=n_joints, n_phi_levels=8)
        r2_u, r2_v = colorizer.train(train_images, sample_rate=0.12)
        
        # Test
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
        
        # Generalization
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        test_mae = np.mean(test_errors)
        gen_mae = np.mean(gen_errors)
        
        print(f"   Test: {test_mae:.2f}, Gen: {gen_mae:.2f}")
        
        results.append({
            'n_joints': n_joints,
            'r2_u': r2_u,
            'r2_v': r2_v,
            'test_mae': test_mae,
            'gen_mae': gen_mae,
            'colorizer': colorizer
        })
    
    # Find best
    best = min(results, key=lambda r: r['gen_mae'])
    
    print("\n2. RESULTS SUMMARY")
    print("-" * 50)
    print(f"   {'Joints':>8} {'R²_U':>8} {'R²_V':>8} {'Test':>8} {'Gen':>8}")
    for r in results:
        marker = " *" if r == best else ""
        print(f"   {r['n_joints']:>8} {r['r2_u']:>8.4f} {r['r2_v']:>8.4f} {r['test_mae']:>8.2f} {r['gen_mae']:>8.2f}{marker}")
    
    # Visualize best
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    colorizer = best['colorizer']
    
    vis_results = []
    for name, img in test_data:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray)
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
        axes[i, 2].set_title(f'IK (MAE={error:.1f})' if i == 0 else f'MAE={error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'IK Colorizer: {best["n_joints"]} joints, R²=({best["r2_u"]:.3f}, {best["r2_v"]:.3f}), Gen={best["gen_mae"]:.1f}',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "ik_colorizer_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return results, best


if __name__ == "__main__":
    results, best = run_ik_test()
    
    print("\n" + "=" * 70)
    print("IK COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   The IK analogy:
   - End effector = color (U, V)
   - Joint angles = φ-dimension values
   - Joint limits = φ-grid quantization
   - Jacobian = how joints affect color
   
   Best configuration:
   - Joints (DOF): {best['n_joints']}
   - Jacobian R²: U={best['r2_u']:.4f}, V={best['r2_v']:.4f}
   
   Results:
   - Test MAE: {best['test_mae']:.2f}
   - Generalization MAE: {best['gen_mae']:.2f}
   
   Key insight:
   - The Jacobian tells us how each φ-dimension affects color
   - Joint limits (φ-grid) constrain valid configurations
   - We SOLVE for the configuration, not search for it
""")
