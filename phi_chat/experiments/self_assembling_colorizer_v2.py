#!/usr/bin/env python3
"""
Self-Assembling Colorizer v2 - Preserving Diversity

Problem with v1: Everything collapsed to neutral colors.
The attraction was too strong, merging saturated modes into the center.

Solution: 
1. Separate feature-space and color-space dynamics
2. Repel anchors with DIFFERENT colors (preserve diversity)
3. Only attract anchors with SIMILAR colors
4. Use color saturation as a "charge" - saturated colors repel neutral

This is like electrostatics:
- Same charge repels
- Opposite charge attracts
- But we want: same COLOR attracts, different COLOR repels

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from typing import List, Tuple, Optional
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


class ColorAnchor:
    """Color anchor with separate feature and color positions."""
    
    def __init__(self, feature_pos: np.ndarray, color: np.ndarray):
        self.feature_pos = feature_pos.copy()
        self.color = color.copy()
        self.mass = 1.0
        self.samples = [(feature_pos.copy(), color.copy())]  # Keep history
    
    @property
    def saturation(self) -> float:
        return np.sqrt(self.color[0]**2 + self.color[1]**2)
    
    def add_sample(self, feature_pos: np.ndarray, color: np.ndarray):
        """Add a sample to this anchor."""
        self.samples.append((feature_pos.copy(), color.copy()))
        
        # Update running average
        n = len(self.samples)
        self.feature_pos = (self.feature_pos * (n-1) + feature_pos) / n
        self.color = (self.color * (n-1) + color) / n
        self.mass = n


class SelfAssemblingColorizerV2:
    """
    Self-assembling colorizer that preserves color diversity.
    
    Key changes from v1:
    1. Match on BOTH feature similarity AND color similarity
    2. Saturated colors are "protected" from merging with neutral
    3. The phase transition is built into the dynamics
    """
    
    def __init__(self, patch_size: int = 16,
                 feature_threshold: float = 0.5,
                 color_threshold: float = 0.05,
                 saturation_protection: float = 0.03):
        
        self.patch_size = patch_size
        self.feature_threshold = feature_threshold
        self.color_threshold = color_threshold
        self.saturation_protection = saturation_protection
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Self-assembled anchors
        self.anchors: List[ColorAnchor] = []
        
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
    
    def find_matching_anchor(self, feature_pos: np.ndarray, color: np.ndarray) -> Optional[int]:
        """
        Find an anchor that matches BOTH feature position AND color.
        
        This is the key change: we don't just match on features,
        we also require color similarity.
        """
        if not self.anchors:
            return None
        
        sample_sat = np.sqrt(color[0]**2 + color[1]**2)
        
        best_idx = None
        best_score = float('inf')
        
        for i, anchor in enumerate(self.anchors):
            # Feature distance
            feat_dist = np.linalg.norm(feature_pos - anchor.feature_pos)
            
            # Color distance
            color_dist = np.linalg.norm(color - anchor.color)
            
            # Saturation difference
            sat_diff = abs(sample_sat - anchor.saturation)
            
            # Only consider if features are close enough
            if feat_dist > self.feature_threshold:
                continue
            
            # Protect saturated colors from merging with neutral
            if sample_sat > self.saturation_protection and anchor.saturation < self.saturation_protection:
                continue
            if anchor.saturation > self.saturation_protection and sample_sat < self.saturation_protection:
                continue
            
            # Only consider if colors are similar enough
            if color_dist > self.color_threshold:
                continue
            
            # Score: prefer closer in both feature and color space
            score = feat_dist + color_dist * 5  # Weight color more
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15):
        """Train by self-assembly with diversity preservation."""
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
        features_norm = (features - self.feature_mean) / self.feature_std
        
        # Self-assembly
        print("   Self-assembling structure...")
        
        n_merged = 0
        n_spawned = 0
        
        # Shuffle for randomness
        indices = np.random.permutation(len(features_norm))
        
        for i, idx in enumerate(indices):
            feat = features_norm[idx]
            color = colors[idx]
            
            # Find matching anchor
            match_idx = self.find_matching_anchor(feat, color)
            
            if match_idx is not None:
                # Merge with existing anchor
                self.anchors[match_idx].add_sample(feat, color)
                n_merged += 1
            else:
                # Spawn new anchor
                self.anchors.append(ColorAnchor(feat, color))
                n_spawned += 1
            
            if (i + 1) % 5000 == 0:
                print(f"     Processed {i+1}/{len(indices)}, anchors: {len(self.anchors)}")
        
        print(f"   Merged: {n_merged}, Spawned: {n_spawned}")
        print(f"   Total anchors: {len(self.anchors)}")
        
        # Prune small anchors
        min_mass = 3
        self.anchors = [a for a in self.anchors if a.mass >= min_mass]
        print(f"   After pruning (mass >= {min_mass}): {len(self.anchors)}")
        
        self.is_trained = True
        
        # Statistics
        masses = [a.mass for a in self.anchors]
        sats = [a.saturation for a in self.anchors]
        
        print(f"\n   Anchor statistics:")
        print(f"     Mass range: [{min(masses):.0f}, {max(masses):.0f}]")
        print(f"     Saturation range: [{min(sats):.3f}, {max(sats):.3f}]")
        
        # Count by saturation
        n_neutral = sum(1 for s in sats if s < 0.03)
        n_moderate = sum(1 for s in sats if 0.03 <= s < 0.1)
        n_saturated = sum(1 for s in sats if s >= 0.1)
        print(f"     Neutral (sat<0.03): {n_neutral}")
        print(f"     Moderate (0.03-0.1): {n_moderate}")
        print(f"     Saturated (sat>=0.1): {n_saturated}")
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      mode: str = 'nearest') -> Tuple[float, float]:
        """Predict color."""
        if not self.is_trained or not self.anchors:
            return 0.0, 0.0
        
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Get distances
        distances = np.array([np.linalg.norm(feat_norm - a.feature_pos) for a in self.anchors])
        
        if mode == 'nearest':
            nearest_idx = np.argmin(distances)
            return tuple(self.anchors[nearest_idx].color)
        
        elif mode == 'weighted':
            weights = np.array([a.mass / (d**2 + 0.01) for a, d in zip(self.anchors, distances)])
            weights /= weights.sum()
            
            colors = np.array([a.color for a in self.anchors])
            u = np.sum(weights * colors[:, 0])
            v = np.sum(weights * colors[:, 1])
            return u, v
        
        elif mode == 'max_sat':
            # Among nearby anchors, choose the most saturated
            k = min(10, len(self.anchors))
            top_k_idx = np.argsort(distances)[:k]
            
            # Weight by inverse distance
            weights = 1.0 / (distances[top_k_idx]**2 + 0.01)
            
            # Boost saturated anchors
            for i, idx in enumerate(top_k_idx):
                sat = self.anchors[idx].saturation
                if sat > 0.05:
                    weights[i] *= (1 + sat * 10)  # Boost saturated
            
            weights /= weights.sum()
            
            colors = np.array([self.anchors[i].color for i in top_k_idx])
            u = np.sum(weights * colors[:, 0])
            v = np.sum(weights * colors[:, 1])
            return u, v
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def colorize(self, grayscale: np.ndarray, mode: str = 'nearest') -> np.ndarray:
        """Colorize."""
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
                
                u, v = self.predict_color(patch, y_pos, x_pos, mode=mode)
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


def run_v2_test():
    """Test v2 self-assembling colorizer."""
    print("=" * 70)
    print("SELF-ASSEMBLING COLORIZER V2")
    print("Preserving color diversity")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. SELF-ASSEMBLY")
    print("-" * 50)
    
    colorizer = SelfAssemblingColorizerV2(
        patch_size=16,
        feature_threshold=0.6,
        color_threshold=0.04,
        saturation_protection=0.03
    )
    colorizer.train(train_images, sample_rate=0.12)
    
    print("\n2. TESTING")
    print("-" * 50)
    
    results = []
    for mode in ['nearest', 'weighted', 'max_sat']:
        test_errors = []
        for name, img in test_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, mode=mode)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            test_errors.append(error)
        
        gen_errors = []
        for name, img in new_data:
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
            colorized = colorizer.colorize(gray, mode=mode)
            error = np.abs(colorized.astype(float) - img.astype(float)).mean()
            gen_errors.append(error)
        
        test_mae = np.mean(test_errors)
        gen_mae = np.mean(gen_errors)
        
        print(f"   {mode:>10}: Test={test_mae:.2f}, Gen={gen_mae:.2f}")
        
        results.append({
            'mode': mode,
            'test_mae': test_mae,
            'gen_mae': gen_mae
        })
    
    # Visualize
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (name, img) in enumerate(test_data[:3]):
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        colorized = colorizer.colorize(gray, mode='max_sat')
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Self-assembled v2 ({error:.1f})' if i == 0 else f'{error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Self-Assembling v2: {len(colorizer.anchors)} anchors with diversity preservation',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "self_assembling_v2_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'self_assembling_v2_test.png'}")
    
    # Visualize anchor structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = np.array([a.color for a in colorizer.anchors])
    masses = np.array([a.mass for a in colorizer.anchors])
    sats = np.array([a.saturation for a in colorizer.anchors])
    
    scatter = ax.scatter(colors[:, 0], colors[:, 1], s=masses, c=sats, cmap='plasma', alpha=0.7)
    ax.set_xlabel('U (blue-yellow)')
    ax.set_ylabel('V (red-green)')
    ax.set_title(f'Emerged Color Anchors ({len(colorizer.anchors)}) - Color = Saturation')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, label='Saturation')
    
    plt.savefig(OUTPUT_PATH / "self_assembling_v2_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_v2_test()
    
    print("\n" + "=" * 70)
    print("SELF-ASSEMBLING V2 SUMMARY")
    print("=" * 70)
    
    sats = [a.saturation for a in colorizer.anchors]
    n_neutral = sum(1 for s in sats if s < 0.03)
    n_moderate = sum(1 for s in sats if 0.03 <= s < 0.1)
    n_saturated = sum(1 for s in sats if s >= 0.1)
    
    print(f"""
   Structure emerged with DIVERSITY:
   - {len(colorizer.anchors)} total anchors
   - {n_neutral} neutral (sat < 0.03)
   - {n_moderate} moderate (0.03 - 0.1)
   - {n_saturated} saturated (sat >= 0.1)
   
   Key change: Saturated colors are PROTECTED from merging with neutral.
   This preserves the phase transition structure.
   
   Results:
""")
    for r in results:
        print(f"     {r['mode']:>10}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
