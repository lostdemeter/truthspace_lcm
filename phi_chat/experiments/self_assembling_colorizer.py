#!/usr/bin/env python3
"""
Self-Assembling Colorizer - Structure Emerges from Data

The principle: Don't design the structure top-down.
Let it EMERGE through attractor/repeller dynamics.

Like the vocabulary self-organization:
- Similar things ATTRACT (converge to same position)
- Dissimilar things REPEL (diverge to different positions)

For colorization:
- Start with random color anchors
- Let them self-organize based on feature-color relationships
- The structure that emerges IS the colorizer

No pre-defined number of modes.
No fixed architecture.
The data shapes the structure.

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.spatial.distance import cdist
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
    """
    A color anchor point that can move based on attractor/repeller forces.
    
    Each anchor has:
    - position: where it sits in feature space
    - color: the (U, V) color it represents
    - mass: how many samples it has absorbed (affects movement)
    """
    
    def __init__(self, position: np.ndarray, color: np.ndarray):
        self.position = position.copy()
        self.color = color.copy()
        self.mass = 1.0
        self.velocity = np.zeros_like(position)
        self.color_velocity = np.zeros_like(color)
    
    def absorb(self, other_position: np.ndarray, other_color: np.ndarray, weight: float = 1.0):
        """Absorb another point, updating position and color."""
        total_mass = self.mass + weight
        self.position = (self.mass * self.position + weight * other_position) / total_mass
        self.color = (self.mass * self.color + weight * other_color) / total_mass
        self.mass = total_mass


class SelfAssemblingColorizer:
    """
    Colorizer that self-assembles its structure from data.
    
    Process:
    1. Start with a few seed anchors (or none)
    2. For each data point:
       - Find nearby anchors
       - If close enough: attract (merge)
       - If far enough: repel (push apart)
       - If no anchor nearby: create new anchor
    3. The structure emerges from these dynamics
    
    Parameters:
    - attraction_threshold: distance below which points attract
    - repulsion_threshold: distance below which points repel
    - spawn_threshold: distance above which a new anchor is created
    """
    
    def __init__(self, patch_size: int = 16,
                 attraction_threshold: float = 0.5,
                 repulsion_threshold: float = 0.1,
                 spawn_threshold: float = 1.0,
                 attraction_strength: float = 0.1,
                 repulsion_strength: float = 0.01):
        
        self.patch_size = patch_size
        self.attraction_threshold = attraction_threshold
        self.repulsion_threshold = repulsion_threshold
        self.spawn_threshold = spawn_threshold
        self.attraction_strength = attraction_strength
        self.repulsion_strength = repulsion_strength
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # The self-assembled structure
        self.anchors: List[ColorAnchor] = []
        
        # Statistics
        self.n_spawned = 0
        self.n_merged = 0
        self.n_repelled = 0
        
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
    
    def find_nearest_anchor(self, position: np.ndarray) -> Tuple[Optional[int], float]:
        """Find the nearest anchor to a position."""
        if not self.anchors:
            return None, float('inf')
        
        distances = [np.linalg.norm(position - a.position) for a in self.anchors]
        nearest_idx = np.argmin(distances)
        return nearest_idx, distances[nearest_idx]
    
    def process_sample(self, position: np.ndarray, color: np.ndarray):
        """
        Process a single sample through attractor/repeller dynamics.
        
        This is where the self-assembly happens.
        """
        nearest_idx, distance = self.find_nearest_anchor(position)
        
        if nearest_idx is None or distance > self.spawn_threshold:
            # No nearby anchor - spawn a new one
            self.anchors.append(ColorAnchor(position, color))
            self.n_spawned += 1
            
        elif distance < self.attraction_threshold:
            # Close enough - attract (merge)
            anchor = self.anchors[nearest_idx]
            
            # Move anchor toward sample
            direction = position - anchor.position
            anchor.velocity += self.attraction_strength * direction / (anchor.mass + 1)
            
            # Update color toward sample
            color_direction = color - anchor.color
            anchor.color_velocity += self.attraction_strength * color_direction / (anchor.mass + 1)
            
            # Increase mass
            anchor.mass += 0.1
            self.n_merged += 1
            
        elif distance < self.repulsion_threshold:
            # Too close - repel
            anchor = self.anchors[nearest_idx]
            direction = anchor.position - position
            anchor.velocity += self.repulsion_strength * direction / (distance + 0.01)
            self.n_repelled += 1
    
    def apply_velocities(self, damping: float = 0.9):
        """Apply accumulated velocities to anchors."""
        for anchor in self.anchors:
            anchor.position += anchor.velocity
            anchor.color += anchor.color_velocity
            anchor.velocity *= damping
            anchor.color_velocity *= damping
    
    def prune_anchors(self, min_mass: float = 1.0):
        """Remove anchors with too little mass."""
        self.anchors = [a for a in self.anchors if a.mass >= min_mass]
    
    def merge_close_anchors(self, threshold: float = 0.2):
        """Merge anchors that are too close together."""
        if len(self.anchors) < 2:
            return
        
        merged = []
        used = set()
        
        for i, anchor_i in enumerate(self.anchors):
            if i in used:
                continue
            
            # Find all anchors close to this one
            to_merge = [anchor_i]
            for j, anchor_j in enumerate(self.anchors):
                if j <= i or j in used:
                    continue
                
                dist = np.linalg.norm(anchor_i.position - anchor_j.position)
                if dist < threshold:
                    to_merge.append(anchor_j)
                    used.add(j)
            
            # Merge them
            if len(to_merge) > 1:
                total_mass = sum(a.mass for a in to_merge)
                new_pos = sum(a.mass * a.position for a in to_merge) / total_mass
                new_color = sum(a.mass * a.color for a in to_merge) / total_mass
                new_anchor = ColorAnchor(new_pos, new_color)
                new_anchor.mass = total_mass
                merged.append(new_anchor)
            else:
                merged.append(anchor_i)
            
            used.add(i)
        
        self.anchors = merged
    
    def train(self, images: List[np.ndarray], sample_rate: float = 0.15, n_epochs: int = 3):
        """
        Train by letting the structure self-assemble.
        
        Multiple epochs allow the structure to stabilize.
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
        features_norm = (features - self.feature_mean) / self.feature_std
        
        # Self-assembly epochs
        for epoch in range(n_epochs):
            print(f"\n   Epoch {epoch + 1}/{n_epochs}")
            
            # Reset statistics
            self.n_spawned = 0
            self.n_merged = 0
            self.n_repelled = 0
            
            # Shuffle data
            indices = np.random.permutation(len(features_norm))
            
            # Process each sample
            for i, idx in enumerate(indices):
                self.process_sample(features_norm[idx], colors[idx])
                
                # Apply velocities periodically
                if (i + 1) % 1000 == 0:
                    self.apply_velocities()
                    
                    # Progress
                    if (i + 1) % 5000 == 0:
                        print(f"     Processed {i+1}/{len(indices)}, anchors: {len(self.anchors)}")
            
            # End of epoch cleanup
            self.apply_velocities()
            self.merge_close_anchors(threshold=0.3)
            self.prune_anchors(min_mass=2.0)
            
            print(f"     Spawned: {self.n_spawned}, Merged: {self.n_merged}, Repelled: {self.n_repelled}")
            print(f"     Final anchors: {len(self.anchors)}")
        
        self.is_trained = True
        
        # Report anchor statistics
        print("\n   Anchor statistics:")
        masses = [a.mass for a in self.anchors]
        print(f"     Total anchors: {len(self.anchors)}")
        print(f"     Mass range: [{min(masses):.1f}, {max(masses):.1f}]")
        print(f"     Total mass: {sum(masses):.1f}")
        
        # Show top anchors by mass
        sorted_anchors = sorted(self.anchors, key=lambda a: a.mass, reverse=True)
        print("\n   Top anchors by mass:")
        for i, anchor in enumerate(sorted_anchors[:10]):
            sat = np.sqrt(anchor.color[0]**2 + anchor.color[1]**2)
            print(f"     {i}: mass={anchor.mass:.1f}, color=({anchor.color[0]:+.3f}, {anchor.color[1]:+.3f}), sat={sat:.3f}")
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      mode: str = 'nearest') -> Tuple[float, float]:
        """
        Predict color using self-assembled structure.
        
        Modes:
        - 'nearest': Use nearest anchor's color
        - 'weighted': Weight by distance to nearby anchors
        - 'top_k': Use top k nearest anchors
        """
        if not self.is_trained or not self.anchors:
            return 0.0, 0.0
        
        # Extract and normalize
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        # Get distances to all anchors
        distances = np.array([np.linalg.norm(feat_norm - a.position) for a in self.anchors])
        
        if mode == 'nearest':
            nearest_idx = np.argmin(distances)
            return tuple(self.anchors[nearest_idx].color)
        
        elif mode == 'weighted':
            # Weight by inverse distance squared, scaled by mass
            weights = np.array([a.mass / (d**2 + 0.01) for a, d in zip(self.anchors, distances)])
            weights /= weights.sum()
            
            colors = np.array([a.color for a in self.anchors])
            u = np.sum(weights * colors[:, 0])
            v = np.sum(weights * colors[:, 1])
            return u, v
        
        elif mode == 'top_k':
            k = min(5, len(self.anchors))
            top_k_idx = np.argsort(distances)[:k]
            
            weights = np.array([self.anchors[i].mass / (distances[i]**2 + 0.01) for i in top_k_idx])
            weights /= weights.sum()
            
            colors = np.array([self.anchors[i].color for i in top_k_idx])
            u = np.sum(weights * colors[:, 0])
            v = np.sum(weights * colors[:, 1])
            return u, v
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def colorize(self, grayscale: np.ndarray, mode: str = 'nearest') -> np.ndarray:
        """Colorize using self-assembled structure."""
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


def run_self_assembly_test():
    """Test the self-assembling colorizer."""
    print("=" * 70)
    print("SELF-ASSEMBLING COLORIZER")
    print("Structure emerges from attractor/repeller dynamics")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. SELF-ASSEMBLY")
    print("-" * 50)
    
    colorizer = SelfAssemblingColorizer(
        patch_size=16,
        attraction_threshold=0.8,
        repulsion_threshold=0.1,
        spawn_threshold=1.5,
        attraction_strength=0.05,
        repulsion_strength=0.01
    )
    colorizer.train(train_images, sample_rate=0.12, n_epochs=2)
    
    print("\n2. TESTING")
    print("-" * 50)
    
    results = []
    for mode in ['nearest', 'weighted', 'top_k']:
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
        colorized = colorizer.colorize(gray, mode='nearest')
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Self-assembled ({error:.1f})' if i == 0 else f'{error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Self-Assembling Colorizer: {len(colorizer.anchors)} anchors emerged',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "self_assembling_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'self_assembling_test.png'}")
    
    # Visualize anchor structure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color space
    colors = np.array([a.color for a in colorizer.anchors])
    masses = np.array([a.mass for a in colorizer.anchors])
    
    scatter = axes[0].scatter(colors[:, 0], colors[:, 1], s=masses*2, c=masses, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('U (blue-yellow)')
    axes[0].set_ylabel('V (red-green)')
    axes[0].set_title(f'Emerged Color Anchors ({len(colorizer.anchors)})')
    axes[0].set_xlim(-0.3, 0.3)
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=axes[0], label='Mass')
    
    # Mass distribution
    axes[1].hist(masses, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Anchor Mass')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Mass Distribution')
    axes[1].axvline(np.median(masses), color='r', linestyle='--', label=f'Median: {np.median(masses):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "self_assembling_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved structure to: {OUTPUT_PATH / 'self_assembling_structure.png'}")
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_self_assembly_test()
    
    print("\n" + "=" * 70)
    print("SELF-ASSEMBLING COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   The structure EMERGED from data:
   - Started with 0 anchors
   - {len(colorizer.anchors)} anchors self-assembled
   - No pre-defined number of modes
   - No fixed architecture
   
   Dynamics:
   - Similar features ATTRACT → merge into anchors
   - Dissimilar features REPEL → stay separate
   - New patterns SPAWN → new anchors
   
   Results:
""")
    for r in results:
        print(f"     {r['mode']:>10}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
    
    print(f"""
   The key insight:
   - Don't design the structure
   - Let it EMERGE from the data
   - The structure IS the knowledge
""")
