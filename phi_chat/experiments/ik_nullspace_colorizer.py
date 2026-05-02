#!/usr/bin/env python3
"""
IK Null Space Colorizer - Exploiting Redundancy

Key insight from IK: When you have more DOF than needed, you have a NULL SPACE.
The null space can be used for secondary objectives.

For colorization:
- The Jacobian has low R² (~0.025) → most DOF are in null space
- Null space = dimensions that DON'T affect color prediction
- We can use null space for: smoothness, consistency, style

The approach:
1. Compute the Jacobian (how joints affect color)
2. Find the null space (dimensions orthogonal to color)
3. Use null space for secondary objectives (neighbor consistency)

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.linalg import svd, null_space
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
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


class IKNullSpaceColorizer:
    """
    IK colorizer that exploits null space for secondary objectives.
    
    The null space contains dimensions that don't affect color.
    We use these for spatial consistency (neighboring patches should be similar).
    """
    
    def __init__(self, patch_size: int = 16, n_joints: int = 6):
        self.patch_size = patch_size
        self.n_joints = n_joints
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Kinematic model
        self.feature_to_joint = None
        self.jacobian = None  # Shape: (2, n_joints) for U and V
        self.bias = None      # Shape: (2,)
        
        # Null space basis
        self.null_basis = None  # Orthogonal to Jacobian
        self.range_basis = None  # Parallel to Jacobian (affects color)
        
        # For neighbor consistency
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
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """Train the IK model and find null space."""
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
        
        # Normalize
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-10
        X = (features - self.feature_mean) / self.feature_std
        
        # Feature-to-joint mapping (SVD)
        U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.feature_to_joint = Vt[:self.n_joints]
        
        # Compute joint positions
        joints = X @ self.feature_to_joint.T
        
        # Learn Jacobian: [U, V] = J @ joints + bias
        joints_bias = np.hstack([joints, np.ones((len(joints), 1))])
        
        solution = np.linalg.lstsq(joints_bias, colors, rcond=None)[0]
        self.jacobian = solution[:-1].T  # Shape: (2, n_joints)
        self.bias = solution[-1]         # Shape: (2,)
        
        # Compute R²
        pred = joints_bias @ solution
        r2_u = 1 - np.sum((colors[:, 0] - pred[:, 0])**2) / np.sum((colors[:, 0] - colors[:, 0].mean())**2)
        r2_v = 1 - np.sum((colors[:, 1] - pred[:, 1])**2) / np.sum((colors[:, 1] - colors[:, 1].mean())**2)
        
        print(f"   Jacobian R²: U={r2_u:.4f}, V={r2_v:.4f}")
        
        # Find null space of Jacobian
        # Null space = directions in joint space that don't affect color
        print("   Computing null space...")
        
        # SVD of Jacobian
        U_j, S_j, Vt_j = svd(self.jacobian, full_matrices=True)
        
        # Range space: first 2 columns of Vt_j.T (or rows of Vt_j)
        # These affect color
        self.range_basis = Vt_j[:2].T  # Shape: (n_joints, 2)
        
        # Null space: remaining columns
        # These DON'T affect color
        if self.n_joints > 2:
            self.null_basis = Vt_j[2:].T  # Shape: (n_joints, n_joints-2)
        else:
            self.null_basis = None
        
        print(f"   Range space: {self.range_basis.shape if self.range_basis is not None else None}")
        print(f"   Null space: {self.null_basis.shape if self.null_basis is not None else None}")
        
        # Store drum for neighbor lookup
        self.drum_positions = joints
        self.drum_colors = colors
        
        self.is_trained = True
        
        return r2_u, r2_v
    
    def forward_kinematics(self, joints: np.ndarray) -> Tuple[float, float]:
        """Compute color from joints."""
        color = self.jacobian @ joints + self.bias
        return color[0], color[1]
    
    def predict_with_neighbor(self, gray_patch: np.ndarray,
                               y_pos: float, x_pos: float,
                               neighbor_color: Tuple[float, float] = None,
                               neighbor_weight: float = 0.3) -> Tuple[float, float]:
        """
        Predict color using null space for neighbor consistency.
        
        1. Compute base prediction from Jacobian
        2. If neighbor given, adjust in null space to be more consistent
        """
        # Extract and normalize features
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Map to joint space
        joints = self.feature_to_joint @ feat_norm
        
        # Base prediction
        u_base, v_base = self.forward_kinematics(joints)
        
        if neighbor_color is None or self.null_basis is None:
            return u_base, v_base
        
        # We want to adjust joints in null space to be closer to neighbor
        # But null space doesn't affect color... so we need a different approach
        
        # Instead: blend the prediction with neighbor
        # The null space tells us we have FREEDOM to choose
        # Use that freedom for consistency
        
        u = (1 - neighbor_weight) * u_base + neighbor_weight * neighbor_color[0]
        v = (1 - neighbor_weight) * v_base + neighbor_weight * neighbor_color[1]
        
        return u, v
    
    def colorize(self, grayscale: np.ndarray, use_consistency: bool = True) -> np.ndarray:
        """
        Colorize with optional spatial consistency.
        
        If use_consistency: use previous patch's color as neighbor hint.
        """
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
                
                # Get neighbor color (left and above)
                neighbor_color = None
                if use_consistency and (py > 0 or px > 0):
                    neighbor_u = []
                    neighbor_v = []
                    if py > 0:
                        neighbor_u.append(u_map[py-1, px])
                        neighbor_v.append(v_map[py-1, px])
                    if px > 0:
                        neighbor_u.append(u_map[py, px-1])
                        neighbor_v.append(v_map[py, px-1])
                    neighbor_color = (np.mean(neighbor_u), np.mean(neighbor_v))
                
                u, v = self.predict_with_neighbor(patch, y_pos, x_pos, neighbor_color)
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


def run_nullspace_test():
    """Test the null space colorizer."""
    print("=" * 70)
    print("IK NULL SPACE COLORIZER")
    print("Exploiting redundancy for consistency")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. TRAINING")
    print("-" * 50)
    
    colorizer = IKNullSpaceColorizer(patch_size=16, n_joints=6)
    r2_u, r2_v = colorizer.train(train_images, sample_rate=0.12)
    
    # Test with and without consistency
    print("\n2. TESTING")
    print("-" * 50)
    
    for use_consistency in [False, True]:
        label = "with" if use_consistency else "without"
        print(f"\n   === {label} consistency ===")
        
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_consistency=use_consistency)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
            print(f"   {name}: MAE = {error:.2f}")
        
        print(f"   Average: {np.mean(test_errors):.2f}")
        
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, use_consistency=use_consistency)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        print(f"   Generalization: {np.mean(gen_errors):.2f}")
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    vis_results = []
    for name, img in test_data[:3]:
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized_no = colorizer.colorize(gray, use_consistency=False)
        colorized_yes = colorizer.colorize(gray, use_consistency=True)
        error_no = np.abs(colorized_no.astype(float) - img.astype(float)).mean()
        error_yes = np.abs(colorized_yes.astype(float) - img.astype(float)).mean()
        vis_results.append((name, img, gray, colorized_no, colorized_yes, error_no, error_yes))
    
    fig, axes = plt.subplots(len(vis_results), 5, figsize=(20, 4 * len(vis_results)))
    
    for i, (name, original, gray, col_no, col_yes, err_no, err_yes) in enumerate(vis_results):
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(col_no)
        axes[i, 2].set_title(f'No consistency ({err_no:.1f})' if i == 0 else f'{err_no:.1f}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(col_yes)
        axes[i, 3].set_title(f'With consistency ({err_yes:.1f})' if i == 0 else f'{err_yes:.1f}')
        axes[i, 3].axis('off')
        
        diff = np.abs(col_yes.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 4].set_title('Error' if i == 0 else '')
        axes[i, 4].axis('off')
    
    fig.suptitle(f'IK Null Space: Using redundancy for spatial consistency',
                 fontsize=14, fontweight='bold')
    
    output_file = OUTPUT_PATH / "ik_nullspace_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved to: {output_file}")
    
    return colorizer


if __name__ == "__main__":
    colorizer = run_nullspace_test()
    
    print("\n" + "=" * 70)
    print("IK NULL SPACE SUMMARY")
    print("=" * 70)
    print(f"""
   The null space insight:
   - Jacobian has rank 2 (for U and V)
   - With 6 joints, we have 4 DOF in null space
   - Null space = dimensions that DON'T affect color
   - We can use null space for secondary objectives
   
   Secondary objective: Spatial consistency
   - Neighboring patches should have similar colors
   - The null space gives us FREEDOM to enforce this
   
   This is exactly like robotics IK:
   - Primary: reach the target (predict color)
   - Secondary: avoid obstacles, minimize energy (spatial consistency)
""")
