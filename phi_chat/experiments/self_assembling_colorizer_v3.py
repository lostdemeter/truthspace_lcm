#!/usr/bin/env python3
"""
Self-Assembling Colorizer v3 - Emergent Phase Structure

Key insight: The structure should emerge with NATURAL phase boundaries.

v1: Too much merging → everything collapsed to neutral
v2: Too little merging → 1382 anchors (too many)
v3: Let the PHASE STRUCTURE emerge naturally

The approach:
1. First pass: Collect all samples
2. Cluster by COLOR first (find natural color modes)
3. Within each color mode, cluster by FEATURES
4. The structure emerges from this two-level clustering

This is like:
- First: What colors exist? (phase discovery)
- Then: What features map to each color? (mapping discovery)

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, zoom
from scipy.cluster.hierarchy import fcluster, linkage
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


class EmergentPhaseColorizer:
    """
    Colorizer where the phase structure emerges from data.
    
    Two-level emergence:
    1. Color phases emerge from clustering colors
    2. Feature mappings emerge within each phase
    
    No pre-defined number of phases.
    The data determines the structure.
    """
    
    def __init__(self, patch_size: int = 16,
                 color_distance_threshold: float = 0.05,
                 feature_distance_threshold: float = 0.8):
        
        self.patch_size = patch_size
        self.color_distance_threshold = color_distance_threshold
        self.feature_distance_threshold = feature_distance_threshold
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Emergent structure
        self.color_phases = []  # List of (color_center, feature_clusters)
        # Each feature_cluster is (feature_center, count)
        
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
        """Train by discovering emergent phase structure."""
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
        
        # Step 1: Discover color phases through hierarchical clustering
        print("   Discovering color phases...")
        
        # Subsample for clustering (full dataset too large)
        n_subsample = min(5000, len(colors))
        subsample_idx = np.random.choice(len(colors), n_subsample, replace=False)
        colors_sub = colors[subsample_idx]
        
        # Hierarchical clustering on colors
        Z = linkage(colors_sub, method='ward')
        
        # Cut at threshold to get clusters
        color_labels = fcluster(Z, t=self.color_distance_threshold, criterion='distance')
        n_color_phases = len(np.unique(color_labels))
        
        print(f"   Found {n_color_phases} color phases")
        
        # Compute color phase centers
        color_phase_centers = []
        for phase_id in range(1, n_color_phases + 1):
            mask = color_labels == phase_id
            center = colors_sub[mask].mean(axis=0)
            count = mask.sum()
            color_phase_centers.append((center, count))
        
        # Sort by count (most common first)
        color_phase_centers.sort(key=lambda x: -x[1])
        
        print("   Color phase centers:")
        for i, (center, count) in enumerate(color_phase_centers[:10]):
            sat = np.sqrt(center[0]**2 + center[1]**2)
            print(f"     Phase {i}: ({center[0]:+.3f}, {center[1]:+.3f}), sat={sat:.3f}, n={count}")
        
        # Step 2: Assign all samples to nearest color phase
        print("   Assigning samples to phases...")
        
        phase_centers = np.array([c[0] for c in color_phase_centers])
        
        # For each sample, find nearest color phase
        sample_phases = []
        for color in colors:
            distances = np.linalg.norm(phase_centers - color, axis=1)
            sample_phases.append(np.argmin(distances))
        
        sample_phases = np.array(sample_phases)
        
        # Step 3: Within each phase, cluster features
        print("   Clustering features within each phase...")
        
        self.color_phases = []
        
        for phase_id in range(len(color_phase_centers)):
            phase_mask = sample_phases == phase_id
            phase_features = features_norm[phase_mask]
            phase_colors = colors[phase_mask]
            
            if len(phase_features) < 10:
                continue
            
            # Subsample for clustering
            n_sub = min(1000, len(phase_features))
            sub_idx = np.random.choice(len(phase_features), n_sub, replace=False)
            
            # Cluster features within this phase
            Z_feat = linkage(phase_features[sub_idx], method='ward')
            feat_labels = fcluster(Z_feat, t=self.feature_distance_threshold, criterion='distance')
            
            # Compute feature cluster centers
            feature_clusters = []
            for cluster_id in np.unique(feat_labels):
                mask = feat_labels == cluster_id
                feat_center = phase_features[sub_idx][mask].mean(axis=0)
                color_center = phase_colors[sub_idx][mask].mean(axis=0)
                count = mask.sum()
                feature_clusters.append({
                    'feature_center': feat_center,
                    'color': color_center,
                    'count': count
                })
            
            self.color_phases.append({
                'color_center': color_phase_centers[phase_id][0],
                'feature_clusters': feature_clusters,
                'total_count': len(phase_features)
            })
        
        # Statistics
        total_clusters = sum(len(p['feature_clusters']) for p in self.color_phases)
        print(f"\n   Emergent structure:")
        print(f"     Color phases: {len(self.color_phases)}")
        print(f"     Total feature clusters: {total_clusters}")
        
        self.is_trained = True
    
    def predict_color(self, gray_patch: np.ndarray, 
                      y_pos: float, x_pos: float,
                      mode: str = 'nearest_cluster') -> Tuple[float, float]:
        """Predict color using emergent structure."""
        if not self.is_trained:
            return 0.0, 0.0
        
        feat = self.extract_features(gray_patch, y_pos, x_pos)
        feat_norm = (feat - self.feature_mean) / (self.feature_std + 1e-10)
        
        if mode == 'nearest_cluster':
            # Find nearest feature cluster across all phases
            best_color = None
            best_dist = float('inf')
            
            for phase in self.color_phases:
                for cluster in phase['feature_clusters']:
                    dist = np.linalg.norm(feat_norm - cluster['feature_center'])
                    if dist < best_dist:
                        best_dist = dist
                        best_color = cluster['color']
            
            if best_color is not None:
                return tuple(best_color)
            return 0.0, 0.0
        
        elif mode == 'weighted_phase':
            # Weight by distance to phase color centers
            total_u = 0.0
            total_v = 0.0
            total_weight = 0.0
            
            for phase in self.color_phases:
                # Find nearest cluster in this phase
                best_dist = float('inf')
                best_cluster = None
                
                for cluster in phase['feature_clusters']:
                    dist = np.linalg.norm(feat_norm - cluster['feature_center'])
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cluster
                
                if best_cluster is not None:
                    weight = best_cluster['count'] / (best_dist**2 + 0.01)
                    total_u += weight * best_cluster['color'][0]
                    total_v += weight * best_cluster['color'][1]
                    total_weight += weight
            
            if total_weight > 0:
                return total_u / total_weight, total_v / total_weight
            return 0.0, 0.0
        
        elif mode == 'max_phase':
            # Find the phase with the nearest feature cluster, use that phase's color
            best_phase_color = None
            best_dist = float('inf')
            
            for phase in self.color_phases:
                for cluster in phase['feature_clusters']:
                    dist = np.linalg.norm(feat_norm - cluster['feature_center'])
                    if dist < best_dist:
                        best_dist = dist
                        best_phase_color = phase['color_center']
            
            if best_phase_color is not None:
                return tuple(best_phase_color)
            return 0.0, 0.0
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def colorize(self, grayscale: np.ndarray, mode: str = 'nearest_cluster') -> np.ndarray:
        """Colorize using emergent structure."""
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


def run_emergent_phase_test():
    """Test emergent phase colorizer."""
    print("=" * 70)
    print("EMERGENT PHASE COLORIZER")
    print("Structure emerges from two-level clustering")
    print("=" * 70)
    
    train_data = load_coco_images(150, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    new_data = load_coco_images(5, start_idx=300)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. DISCOVERING EMERGENT STRUCTURE")
    print("-" * 50)
    
    colorizer = EmergentPhaseColorizer(
        patch_size=16,
        color_distance_threshold=0.08,
        feature_distance_threshold=1.0
    )
    colorizer.train(train_images, sample_rate=0.12)
    
    print("\n2. TESTING")
    print("-" * 50)
    
    results = []
    for mode in ['nearest_cluster', 'weighted_phase', 'max_phase']:
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
        
        print(f"   {mode:>18}: Test={test_mae:.2f}, Gen={gen_mae:.2f}")
        
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
        colorized = colorizer.colorize(gray, mode='nearest_cluster')
        error = np.abs(colorized.astype(float) - img.astype(float)).mean()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'Emergent ({error:.1f})' if i == 0 else f'{error:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - img.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=60)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    n_phases = len(colorizer.color_phases)
    n_clusters = sum(len(p['feature_clusters']) for p in colorizer.color_phases)
    
    fig.suptitle(f'Emergent Phase Colorizer: {n_phases} phases, {n_clusters} clusters',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "emergent_phase_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'emergent_phase_test.png'}")
    
    # Visualize phase structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for phase in colorizer.color_phases:
        color = phase['color_center']
        sat = np.sqrt(color[0]**2 + color[1]**2)
        size = phase['total_count'] / 10
        
        ax.scatter(color[0], color[1], s=size, alpha=0.7,
                   c=[sat], cmap='plasma', vmin=0, vmax=0.3)
    
    ax.set_xlabel('U (blue-yellow)')
    ax.set_ylabel('V (red-green)')
    ax.set_title(f'Emergent Color Phases ({n_phases})')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.savefig(OUTPUT_PATH / "emergent_phase_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return colorizer, results


if __name__ == "__main__":
    colorizer, results = run_emergent_phase_test()
    
    n_phases = len(colorizer.color_phases)
    n_clusters = sum(len(p['feature_clusters']) for p in colorizer.color_phases)
    
    print("\n" + "=" * 70)
    print("EMERGENT PHASE COLORIZER SUMMARY")
    print("=" * 70)
    print(f"""
   Two-level emergent structure:
   1. Color phases emerged: {n_phases}
   2. Feature clusters within phases: {n_clusters}
   
   No pre-defined architecture.
   The data determined the structure.
   
   Results:
""")
    for r in results:
        print(f"     {r['mode']:>18}: Test={r['test_mae']:.2f}, Gen={r['gen_mae']:.2f}")
    
    print(f"""
   The key insight:
   - First discover WHAT colors exist (phases)
   - Then discover WHICH features map to each color
   - The structure IS the knowledge
""")
