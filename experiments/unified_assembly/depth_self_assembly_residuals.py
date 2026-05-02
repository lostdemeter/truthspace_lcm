#!/usr/bin/env python3
"""
Self-Assembly of Depth Assignments from Geometric Residuals

The Hypothesis:
- Vertical baseline gives us geometric truth (depth ≈ 0.6y + 0.1)
- The RESIDUAL from vertical is where semantic priors live
- If we cluster regions by geometric signature and look at their residuals,
  we might discover that certain signatures consistently have positive/negative residuals
- This would be SELF-ASSEMBLING semantic priors without training labels!

The Algorithm:
1. For each pixel, extract geometric signature (color, texture, position)
2. Compute residual = DA_depth - vertical_baseline
3. Build similarity matrix between pixels based on geometric signature
4. Eigendecompose to find clusters
5. For each cluster, compute mean residual
6. This mean residual IS the depth correction for that geometric signature
7. Apply corrections to vertical baseline

This mirrors TruthSpace self-assembly:
- Extract PAIRS (pixel pairs with similar signatures)
- Build SIMILARITY MATRIX
- Discover DIMENSIONS via eigendecomposition
- Output POSITION (depth correction)

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


class GeometricSignatureExtractor:
    """
    Extract geometric signatures from image regions.
    
    A geometric signature captures:
    - Color (R, G, B)
    - Position (y coordinate)
    - Texture (gradient magnitude)
    - Smoothness (local variance)
    - Blue dominance (sky-like)
    """
    
    def __init__(self):
        pass
    
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """
        Extract geometric signature for each pixel.
        
        Returns: (H, W, D) array where D is signature dimension
        """
        h, w = rgb.shape[:2]
        
        # Color features
        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]
        
        # Position
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
        
        # Texture (gradient magnitude)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        texture = np.sqrt(grad_x**2 + grad_y**2)
        texture = gaussian_filter(texture, sigma=2)
        
        # Smoothness (inverse of local variance)
        local_mean = uniform_filter(gray, size=8)
        local_var = uniform_filter(gray**2, size=8) - local_mean**2
        smoothness = 1 / (local_var + 0.01)
        smoothness = _normalize(smoothness)
        
        # Blue dominance
        total = R + G + B + 0.01
        blue_dom = B / total
        
        # Stack into signature
        signature = np.stack([
            R, G, B,                    # Color (3)
            y_coords,                   # Vertical position (1)
            _normalize(texture),        # Texture (1)
            smoothness,                 # Smoothness (1)
            blue_dom,                   # Blue dominance (1)
        ], axis=2)
        
        return signature


class ResidualSelfAssembler:
    """
    Self-assemble depth corrections from geometric residuals.
    
    This is the core of the experiment:
    1. Cluster pixels by geometric signature
    2. For each cluster, compute mean residual from vertical baseline
    3. The mean residual IS the depth correction for that signature
    """
    
    def __init__(self, n_clusters: int = 16):
        self.n_clusters = n_clusters
        self.extractor = GeometricSignatureExtractor()
        self.cluster_corrections = None
        self.cluster_centers = None
    
    def _simple_kmeans(self, features: np.ndarray, n_clusters: int, n_iter: int = 10) -> tuple:
        """Simple k-means without sklearn."""
        n_samples, n_features = features.shape
        
        # Initialize centers randomly
        idx = np.random.choice(n_samples, n_clusters, replace=False)
        centers = features[idx].copy()
        
        for _ in range(n_iter):
            # Assign to nearest center
            dists = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                dists[:, k] = np.sum((features - centers[k])**2, axis=1)
            labels = np.argmin(dists, axis=1)
            
            # Update centers
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centers[k] = features[mask].mean(axis=0)
        
        return labels, centers
    
    def fit(self, images: list, da_depths: list):
        """
        Learn depth corrections from geometric residuals.
        
        For each geometric signature cluster, learn the mean residual
        from the vertical baseline.
        """
        print("  Extracting geometric signatures...")
        
        all_signatures = []
        all_residuals = []
        
        for rgb, da_depth in zip(images, da_depths):
            h, w = rgb.shape[:2]
            
            # Extract signature
            signature = self.extractor.extract(rgb)
            
            # Compute vertical baseline
            y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
            vertical = _normalize(0.6 * y_coords + 0.1)
            
            # Compute residual
            residual = da_depth - vertical
            
            # Flatten and collect
            all_signatures.append(signature.reshape(-1, signature.shape[2]))
            all_residuals.append(residual.flatten())
        
        # Stack all data
        all_signatures = np.vstack(all_signatures)
        all_residuals = np.concatenate(all_residuals)
        
        print(f"  Total pixels: {len(all_residuals)}")
        
        # Subsample for clustering (memory efficiency)
        n_samples = min(50000, len(all_residuals))
        idx = np.random.choice(len(all_residuals), n_samples, replace=False)
        signatures_sample = all_signatures[idx]
        residuals_sample = all_residuals[idx]
        
        # Normalize signatures for clustering
        sig_mean = signatures_sample.mean(axis=0)
        sig_std = signatures_sample.std(axis=0) + 0.001
        signatures_norm = (signatures_sample - sig_mean) / sig_std
        
        print(f"  Clustering into {self.n_clusters} groups...")
        
        # Cluster by geometric signature
        labels, centers = self._simple_kmeans(signatures_norm, self.n_clusters)
        
        # For each cluster, compute mean residual
        self.cluster_corrections = np.zeros(self.n_clusters)
        self.cluster_counts = np.zeros(self.n_clusters)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                self.cluster_corrections[k] = residuals_sample[mask].mean()
                self.cluster_counts[k] = mask.sum()
        
        # Store normalization parameters and centers
        self.sig_mean = sig_mean
        self.sig_std = sig_std
        self.cluster_centers = centers
        
        print("  Learned corrections:")
        for k in range(self.n_clusters):
            print(f"    Cluster {k}: correction = {self.cluster_corrections[k]:+.3f} "
                  f"(n={int(self.cluster_counts[k])})")
    
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """
        Predict depth using self-assembled corrections.
        
        1. Extract geometric signature
        2. Assign each pixel to nearest cluster
        3. Apply cluster's correction to vertical baseline
        """
        h, w = rgb.shape[:2]
        
        # Extract signature
        signature = self.extractor.extract(rgb)
        sig_flat = signature.reshape(-1, signature.shape[2])
        
        # Normalize
        sig_norm = (sig_flat - self.sig_mean) / self.sig_std
        
        # Assign to nearest cluster
        dists = np.zeros((len(sig_norm), self.n_clusters))
        for k in range(self.n_clusters):
            dists[:, k] = np.sum((sig_norm - self.cluster_centers[k])**2, axis=1)
        labels = np.argmin(dists, axis=1)
        
        # Get corrections
        corrections = self.cluster_corrections[labels].reshape(h, w)
        
        # Apply to vertical baseline
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        vertical = _normalize(0.6 * y_coords + 0.1)
        
        depth = vertical + corrections
        
        return _normalize(depth)
    
    def analyze_clusters(self):
        """Analyze what each cluster represents."""
        print("\n  Cluster Analysis:")
        print("  " + "-" * 60)
        
        # Feature names
        feature_names = ['R', 'G', 'B', 'Y_pos', 'Texture', 'Smooth', 'Blue_dom']
        
        for k in range(self.n_clusters):
            center = self.cluster_centers[k]
            correction = self.cluster_corrections[k]
            
            # Denormalize center
            center_denorm = center * self.sig_std + self.sig_mean
            
            # Find dominant features
            dominant = np.argsort(np.abs(center))[-3:][::-1]
            
            print(f"\n  Cluster {k}: correction = {correction:+.3f}")
            print(f"    Dominant features: ", end="")
            for d in dominant:
                sign = "+" if center[d] > 0 else "-"
                print(f"{sign}{feature_names[d]} ", end="")
            print()
            
            # Interpret
            if correction > 0.05:
                print(f"    → This signature means FARTHER than vertical predicts")
            elif correction < -0.05:
                print(f"    → This signature means CLOSER than vertical predicts")
            else:
                print(f"    → This signature matches vertical baseline")


def run_self_assembly_experiment(n_train: int = 20, n_test: int = 10):
    """
    Test if self-assembly can discover semantic depth priors.
    """
    print("=" * 70)
    print("EXPERIMENT: Self-Assembly of Depth Assignments")
    print("=" * 70)
    print()
    print("Hypothesis: Semantic priors can EMERGE from geometric residuals")
    print("Method: Cluster by signature, learn mean residual per cluster")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Load training data
    print("Loading training data...")
    train_images = []
    train_depths = []
    
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        train_images.append(rgb_small)
        train_depths.append(da_depth_small)
    
    print(f"  Loaded {len(train_images)} training images")
    
    # Create and train assembler
    print("\nTraining self-assembler...")
    assembler = ResidualSelfAssembler(n_clusters=16)
    assembler.fit(train_images, train_depths)
    
    # Analyze what emerged
    assembler.analyze_clusters()
    
    # Test
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    vertical_errors = []
    assembled_errors = []
    
    for i, img_id in enumerate(available_ids[n_train:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical = _normalize(0.6 * y_coords + 0.1)
        
        # Self-assembled prediction
        assembled = assembler.predict(rgb_small)
        
        # Errors
        vertical_err = np.mean(np.abs(vertical - da_depth_small))
        assembled_err = np.mean(np.abs(assembled - da_depth_small))
        
        vertical_errors.append(vertical_err)
        assembled_errors.append(assembled_err)
        
        if i < 3:
            print(f"\n  Test {i+1}: {img_id}")
            print(f"    Vertical MAE:   {vertical_err:.4f}")
            print(f"    Assembled MAE:  {assembled_err:.4f}")
            if assembled_err < vertical_err:
                print(f"    → Self-assembly IMPROVED by {(vertical_err - assembled_err) / vertical_err * 100:.1f}%")
            else:
                print(f"    → Self-assembly did not improve")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    mean_vertical = np.mean(vertical_errors)
    mean_assembled = np.mean(assembled_errors)
    
    print(f"\n  Vertical Baseline MAE:    {mean_vertical:.4f}")
    print(f"  Self-Assembled MAE:       {mean_assembled:.4f}")
    
    if mean_assembled < mean_vertical:
        improvement = (mean_vertical - mean_assembled) / mean_vertical * 100
        print(f"\n  IMPROVEMENT: {improvement:.1f}%")
        print("\n  → Semantic priors CAN emerge from geometric residuals!")
        print("  → Self-assembly discovered depth corrections without labels!")
    else:
        print(f"\n  No improvement over vertical baseline.")
        print("\n  → Semantic priors may require more sophisticated self-assembly")
        print("  → Or they may be fundamentally non-geometric")
    
    return assembler, vertical_errors, assembled_errors


def create_self_assembly_visualization(assembler, n_images: int = 3):
    """Visualize the self-assembled depth corrections."""
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    fig = plt.figure(figsize=(20, 6 * n_images))
    fig.suptitle('Self-Assembly of Depth Assignments from Geometric Residuals\n'
                 'Can Semantic Priors Emerge Without Labels?',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 6, figure=fig, hspace=0.3, wspace=0.15)
    
    for row, img_id in enumerate(available_ids[20:20+n_images]):  # Use test images
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = min(400 / max(h, w), 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute predictions
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical = _normalize(0.6 * y_coords + 0.1)
        assembled = assembler.predict(rgb_small)
        
        # Corrections applied
        corrections = assembled - vertical
        
        # Errors
        vertical_err = np.mean(np.abs(vertical - da_depth_small))
        assembled_err = np.mean(np.abs(assembled - da_depth_small))
        
        # Plot
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(rgb_small)
        ax1.set_title('Original', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(da_depth_small, cmap='magma')
        ax2.set_title('DA Depth\n(Target)', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(vertical, cmap='magma')
        ax3.set_title(f'Vertical Baseline\n(MAE: {vertical_err:.3f})', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.imshow(corrections, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax4.set_title('Self-Assembled\nCorrections', fontsize=10)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[row, 4])
        ax5.imshow(assembled, cmap='magma')
        ax5.set_title(f'Assembled Depth\n(MAE: {assembled_err:.3f})', fontsize=10)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[row, 5])
        # Show improvement
        vertical_error_map = np.abs(vertical - da_depth_small)
        assembled_error_map = np.abs(assembled - da_depth_small)
        improvement = vertical_error_map - assembled_error_map
        ax6.imshow(improvement, cmap='RdYlGn', vmin=-0.2, vmax=0.2)
        ax6.set_title('Improvement\n(Green=Better)', fontsize=10)
        ax6.axis('off')
    
    output_file = OUTPUT_PATH / "self_assembly_depth_corrections.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Run experiment
    assembler, vertical_errors, assembled_errors = run_self_assembly_experiment(
        n_train=20, n_test=10
    )
    
    # Create visualization
    viz_file = create_self_assembly_visualization(assembler, n_images=3)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("This experiment tests whether semantic depth priors can EMERGE")
    print("from self-assembly of geometric residuals, without training labels.")
    print()
    print("The key insight: if certain geometric signatures consistently")
    print("have positive/negative residuals from vertical baseline,")
    print("then the structure IS encoding its own navigation rules.")
