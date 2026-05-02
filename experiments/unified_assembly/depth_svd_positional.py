#!/usr/bin/env python3
"""
Experiment: Combining Positional Features with SVD Similarity

Two complementary approaches:
1. POSITIONAL: Local geometric cues (edges, vertical gradient, frequency)
   - Captures "where in the image" and "what local structure"
   
2. SVD SIMILARITY: Global relational structure
   - Patches that look similar should have similar depth
   - SVD finds latent dimensions in the similarity matrix
   - This is the holographic pattern space approach

Combined: positional tells us local cues, SVD tells us global relationships.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.linalg import svd
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


# =============================================================================
# POSITIONAL FEATURES (from skeleton model)
# =============================================================================

def extract_vertical_gradient(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    return np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))


def extract_edges(gray: np.ndarray) -> np.ndarray:
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    return _normalize(np.sqrt(grad_x**2 + grad_y**2))


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    F = fft2(gray)
    F_shifted = fftshift(F)
    h, w = gray.shape
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
    F_filtered = F_shifted * H
    filtered = np.abs(ifft2(ifftshift(F_filtered)))
    return _normalize(filtered)


def extract_saliency(gray: np.ndarray) -> np.ndarray:
    F = fft2(gray)
    amplitude = np.abs(F)
    log_amplitude = np.log(amplitude + 1e-10)
    log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
    residual = log_amplitude - log_amplitude_smoothed
    phase = np.angle(F)
    F_residual = np.exp(residual + 1j * phase)
    saliency = np.abs(ifft2(F_residual)) ** 2
    saliency = gaussian_filter(saliency, sigma=5.0)
    return _normalize(saliency)


# =============================================================================
# SVD SIMILARITY APPROACH
# =============================================================================

class PatchSVDDepth:
    """
    Use SVD to learn a latent space from RGB-depth patch pairs.
    
    Training:
    1. Collect RGB patches and corresponding depth patches from training images
    2. SVD on RGB patches → find basis vectors
    3. Learn linear mapping: RGB latent position → mean depth of patch
    
    Inference:
    1. Extract RGB patches from new image
    2. Project into learned latent space
    3. Use learned mapping to predict depth per patch
    """
    
    def __init__(self, patch_size: int = 16, n_components: int = 16):
        self.patch_size = patch_size
        self.n_components = n_components
        
        # Learned from training
        self.rgb_mean = None
        self.rgb_basis = None  # V from SVD
        self.depth_weights = None  # Linear map: latent → depth
        self.is_trained = False
    
    def extract_patches_with_depth(self, image: np.ndarray, depth: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Extract RGB and depth patches."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        gray = _normalize(gray)
        
        # Resize depth to match
        if depth.shape != gray.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((gray.shape[1], gray.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        h, w = gray.shape
        ph, pw = self.patch_size, self.patch_size
        nh, nw = h // ph, w // pw
        
        rgb_patches = []
        depth_values = []
        
        for i in range(nh):
            for j in range(nw):
                rgb_patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw].flatten()
                depth_patch = depth[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                
                # Add position features to RGB patch
                pos_y = i / nh
                pos_x = j / nw
                rgb_patch_with_pos = np.concatenate([rgb_patch, [pos_y, pos_x]])
                
                rgb_patches.append(rgb_patch_with_pos)
                depth_values.append(np.mean(depth_patch))  # Mean depth of patch
        
        return np.array(rgb_patches), np.array(depth_values), (nh, nw)
    
    def extract_patches(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract RGB patches only (for inference)."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        gray = _normalize(gray)
        
        h, w = gray.shape
        ph, pw = self.patch_size, self.patch_size
        nh, nw = h // ph, w // pw
        
        rgb_patches = []
        
        for i in range(nh):
            for j in range(nw):
                rgb_patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw].flatten()
                pos_y = i / nh
                pos_x = j / nw
                rgb_patch_with_pos = np.concatenate([rgb_patch, [pos_y, pos_x]])
                rgb_patches.append(rgb_patch_with_pos)
        
        return np.array(rgb_patches), (nh, nw)
    
    def fit(self, images: List[np.ndarray], depths: List[np.ndarray]):
        """Learn latent space and depth mapping from training data."""
        print(f"  Learning SVD latent space from {len(images)} images...")
        
        # Collect all patches
        all_rgb_patches = []
        all_depth_values = []
        
        for img, depth in zip(images, depths):
            rgb_patches, depth_vals, _ = self.extract_patches_with_depth(img, depth)
            all_rgb_patches.append(rgb_patches)
            all_depth_values.append(depth_vals)
        
        all_rgb_patches = np.vstack(all_rgb_patches)
        all_depth_values = np.concatenate(all_depth_values)
        
        print(f"    Collected {len(all_rgb_patches)} patches")
        
        # Center the data
        self.rgb_mean = np.mean(all_rgb_patches, axis=0)
        centered = all_rgb_patches - self.rgb_mean
        
        # SVD to find basis
        U, sigma, Vt = svd(centered, full_matrices=False)
        
        # Keep top components
        self.rgb_basis = Vt[:self.n_components, :].T  # Shape: (features, components)
        
        print(f"    SVD: kept {self.n_components} components")
        print(f"    Variance explained: {100 * np.sum(sigma[:self.n_components]**2) / np.sum(sigma**2):.1f}%")
        
        # Project patches into latent space
        latent = centered @ self.rgb_basis
        
        # Learn linear mapping: latent → depth
        # Using least squares: depth = latent @ weights
        # weights = (latent.T @ latent)^-1 @ latent.T @ depth
        self.depth_weights = np.linalg.lstsq(latent, all_depth_values, rcond=None)[0]
        
        # Compute training error
        pred_depths = latent @ self.depth_weights
        train_mae = np.mean(np.abs(pred_depths - all_depth_values))
        print(f"    Patch-level train MAE: {train_mae:.4f}")
        
        self.is_trained = True
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using learned SVD mapping."""
        if not self.is_trained:
            # Fallback: just use vertical gradient
            h, w = image.shape[:2]
            return np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        
        rgb_patches, (nh, nw) = self.extract_patches(image)
        
        # Center and project
        centered = rgb_patches - self.rgb_mean
        latent = centered @ self.rgb_basis
        
        # Predict depth per patch
        depth_values = latent @ self.depth_weights
        depth_values = np.clip(depth_values, 0, 1)
        
        # Reshape to grid
        depth_grid = depth_values.reshape(nh, nw)
        
        # Upsample
        h, w = image.shape[:2]
        pil = Image.fromarray((depth_grid * 255).astype(np.uint8))
        pil = pil.resize((w, h), Image.BILINEAR)
        depth = np.array(pil).astype(np.float32) / 255.0
        
        depth = gaussian_filter(depth, sigma=3.0)
        return _normalize(depth)


# =============================================================================
# COMBINED MODEL: Positional + SVD
# =============================================================================

class CombinedDepthModel:
    """
    Combine positional features with SVD similarity.
    
    Positional: local geometric cues
    SVD: global relational structure
    
    The combination should capture both local and global depth information.
    """
    
    def __init__(self, patch_size: int = 16, n_components: int = 8):
        self.svd_model = PatchSVDDepth(patch_size, n_components)
        
        # Weights for combining approaches
        self.positional_weight = PHI / (1 + PHI)  # ~0.618
        self.svd_weight = 1 / (1 + PHI)           # ~0.382
        
        # Positional dimension weights
        self.dim_weights = {
            'vertical_gradient': PHI**2,
            'edges': PHI**1,
            'frequency': PHI**0,
            'saliency': PHI**(-1),
        }
        
        # Learnable adjustments
        self.adjustments = {}
    
    def extract_positional_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract positional features."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        gray = _normalize(gray)
        
        return {
            'vertical_gradient': extract_vertical_gradient(gray),
            'edges': extract_edges(gray),
            'frequency': extract_frequency(gray),
            'saliency': extract_saliency(gray),
        }
    
    def compose_positional_depth(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compose depth from positional features."""
        total = None
        total_weight = 0
        
        for name, weight in self.dim_weights.items():
            adj = self.adjustments.get(name, 1.0)
            effective_weight = weight * adj
            
            if effective_weight <= 0:
                continue
            
            feat = features[name]
            if total is None:
                total = effective_weight * feat
            else:
                total = total + effective_weight * feat
            total_weight += effective_weight
        
        if total is None:
            return np.zeros_like(list(features.values())[0])
        
        depth = total / total_weight
        return _normalize(depth)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth using combined approach.
        
        depth = w_pos * positional_depth + w_svd * svd_depth
        """
        # Positional prediction
        pos_features = self.extract_positional_features(image)
        pos_depth = self.compose_positional_depth(pos_features)
        
        # SVD prediction
        svd_depth = self.svd_model.predict(image)
        
        # Resize SVD depth to match positional
        if svd_depth.shape != pos_depth.shape:
            pil = Image.fromarray((svd_depth * 255).astype(np.uint8))
            pil = pil.resize((pos_depth.shape[1], pos_depth.shape[0]), Image.BILINEAR)
            svd_depth = np.array(pil).astype(np.float32) / 255.0
        
        # Combine
        combined = (self.positional_weight * pos_depth + 
                    self.svd_weight * svd_depth)
        
        combined = gaussian_filter(combined, sigma=2.0)
        
        return _normalize(combined)
    
    def fit(self, images: List[np.ndarray], depths: List[np.ndarray],
            n_iterations: int = 10):
        """Learn optimal weights for combining approaches."""
        print(f"Fitting combined model on {len(images)} images...")
        
        # Pre-compute predictions
        all_pos_depths = []
        all_svd_depths = []
        all_targets = []
        
        for img, depth in zip(images, depths):
            pos_features = self.extract_positional_features(img)
            pos_depth = self.compose_positional_depth(pos_features)
            svd_depth = self.svd_model.predict(img)
            
            # Resize to match
            target_shape = pos_depth.shape
            if svd_depth.shape != target_shape:
                pil = Image.fromarray((svd_depth * 255).astype(np.uint8))
                pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                svd_depth = np.array(pil).astype(np.float32) / 255.0
            
            if depth.shape != target_shape:
                pil = Image.fromarray((depth * 255).astype(np.uint8))
                pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                depth = np.array(pil).astype(np.float32) / 255.0
            
            all_pos_depths.append(pos_depth)
            all_svd_depths.append(svd_depth)
            all_targets.append(depth)
        
        # Compute individual MAEs
        pos_mae = np.mean([np.mean(np.abs(p - t)) for p, t in zip(all_pos_depths, all_targets)])
        svd_mae = np.mean([np.mean(np.abs(s - t)) for s, t in zip(all_svd_depths, all_targets)])
        
        print(f"  Positional-only MAE: {pos_mae:.4f}")
        print(f"  SVD-only MAE: {svd_mae:.4f}")
        
        # Search for optimal combination weights
        best_mae = float('inf')
        best_pos_w = self.positional_weight
        best_svd_w = self.svd_weight
        
        for pos_w in np.linspace(0.0, 1.0, 21):
            svd_w = 1.0 - pos_w
            
            total_mae = 0
            for pos_d, svd_d, target in zip(all_pos_depths, all_svd_depths, all_targets):
                combined = pos_w * pos_d + svd_w * svd_d
                combined = _normalize(combined)
                total_mae += np.mean(np.abs(combined - target))
            
            avg_mae = total_mae / len(all_targets)
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_pos_w = pos_w
                best_svd_w = svd_w
        
        self.positional_weight = best_pos_w
        self.svd_weight = best_svd_w
        
        print(f"\nOptimal weights:")
        print(f"  Positional: {self.positional_weight:.3f}")
        print(f"  SVD: {self.svd_weight:.3f}")
        print(f"  Combined MAE: {best_mae:.4f}")
        
        return best_mae


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_combined_experiment(n_train: int = 50, n_test: int = 10):
    """
    Compare:
    1. Positional only
    2. SVD only
    3. Combined (positional + SVD)
    """
    print("=" * 70)
    print("EXPERIMENT: Positional + SVD Similarity Combined")
    print("=" * 70)
    print()
    print("Hypothesis: Combining local positional cues with global SVD")
    print("similarity structure should improve depth estimation.")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_images = []
    train_depths = []
    test_images = []
    test_depths = []
    
    print("Loading data...")
    for i, img_id in enumerate(available_ids[:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        if i < n_train:
            train_images.append(rgb)
            train_depths.append(depth)
        else:
            test_images.append(rgb)
            test_depths.append(depth)
    
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")
    
    # Train SVD model first
    print("\n" + "=" * 60)
    print("TRAINING SVD MODEL")
    print("=" * 60)
    
    svd_model = PatchSVDDepth(patch_size=16, n_components=16)
    svd_model.fit(train_images, train_depths)
    
    # Train combined model
    print("\n" + "=" * 60)
    print("TRAINING COMBINED MODEL")
    print("=" * 60)
    
    model = CombinedDepthModel(patch_size=16, n_components=16)
    model.svd_model = svd_model  # Use pre-trained SVD
    train_mae = model.fit(train_images, train_depths, n_iterations=10)
    
    # Test
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    
    pos_errors = []
    svd_errors = []
    combined_errors = []
    
    for img, depth in zip(test_images, test_depths):
        # Positional only
        pos_features = model.extract_positional_features(img)
        pos_depth = model.compose_positional_depth(pos_features)
        
        # SVD only
        svd_depth = model.svd_model.predict(img)
        
        # Combined
        combined_depth = model.predict(img)
        
        # Resize targets
        target_shape = pos_depth.shape
        if depth.shape != target_shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        if svd_depth.shape != target_shape:
            pil = Image.fromarray((svd_depth * 255).astype(np.uint8))
            pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            svd_depth = np.array(pil).astype(np.float32) / 255.0
        
        pos_errors.append(np.mean(np.abs(pos_depth - depth)))
        svd_errors.append(np.mean(np.abs(svd_depth - depth)))
        combined_errors.append(np.mean(np.abs(combined_depth - depth)))
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Positional Only:  Test MAE = {np.mean(pos_errors):.4f}")
    print(f"  SVD Only:         Test MAE = {np.mean(svd_errors):.4f}")
    print(f"  Combined:         Test MAE = {np.mean(combined_errors):.4f}")
    print()
    
    improvement = np.mean(pos_errors) - np.mean(combined_errors)
    print(f"Improvement from adding SVD: {improvement:.4f}")
    print(f"  ({100 * improvement / np.mean(pos_errors):.1f}% better)")
    
    return model


def run_svd_on_features_experiment(n_train: int = 50, n_test: int = 10):
    """
    Alternative: Use SVD on POSITIONAL FEATURES to find optimal basis.
    
    Instead of SVD on raw patches, apply SVD to the stacked positional
    features to find the latent dimensions that best predict depth.
    """
    print("=" * 70)
    print("EXPERIMENT: SVD on Positional Features")
    print("=" * 70)
    print()
    print("Approach: Stack positional features, SVD to find optimal basis,")
    print("then learn linear mapping to depth.")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
    test_data = []
    
    print("Loading and extracting features...")
    for i, img_id in enumerate(available_ids[:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Extract positional features
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        
        features = {
            'vertical': extract_vertical_gradient(gray),
            'edges': extract_edges(gray),
            'frequency': extract_frequency(gray),
            'saliency': extract_saliency(gray),
            'luminance': gray,
        }
        
        # Resize depth
        target_shape = gray.shape
        if depth.shape != target_shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        if i < n_train:
            train_data.append((features, depth))
        else:
            test_data.append((features, depth))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Stack features into matrix: each pixel is a sample
    print("\nBuilding feature matrix...")
    
    all_features = []
    all_depths = []
    
    for features, depth in train_data:
        h, w = depth.shape
        n_pixels = h * w
        
        # Stack features: (n_pixels, n_features)
        feature_matrix = np.column_stack([
            features['vertical'].flatten(),
            features['edges'].flatten(),
            features['frequency'].flatten(),
            features['saliency'].flatten(),
            features['luminance'].flatten(),
        ])
        
        all_features.append(feature_matrix)
        all_depths.append(depth.flatten())
    
    X = np.vstack(all_features)
    y = np.concatenate(all_depths)
    
    print(f"  Feature matrix: {X.shape}")
    
    # Center features
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # SVD on features
    print("\nComputing SVD on features...")
    U, sigma, Vt = svd(X_centered, full_matrices=False)
    
    print(f"  Singular values: {sigma[:5]}")
    print(f"  Variance explained by top 3: {100 * np.sum(sigma[:3]**2) / np.sum(sigma**2):.1f}%")
    
    # The columns of V are the principal directions
    # These tell us how to combine features optimally
    print("\nPrincipal directions (how to combine features):")
    feature_names = ['vertical', 'edges', 'frequency', 'saliency', 'luminance']
    for i in range(min(3, len(sigma))):
        print(f"  PC{i+1}: ", end="")
        for j, name in enumerate(feature_names):
            print(f"{name}={Vt[i,j]:+.3f} ", end="")
        print()
    
    # Project features onto principal components
    n_components = 3
    X_proj = X_centered @ Vt[:n_components, :].T
    
    # Learn linear mapping: projected features → depth
    weights = np.linalg.lstsq(X_proj, y, rcond=None)[0]
    
    print(f"\nLearned weights for PCs: {weights}")
    
    # Compute train MAE
    y_pred_train = X_proj @ weights
    train_mae = np.mean(np.abs(y_pred_train - y))
    print(f"Train MAE: {train_mae:.4f}")
    
    # Test
    print("\nTesting...")
    test_errors_svd = []
    test_errors_pos = []
    
    # Positional baseline weights (from skeleton)
    pos_weights = np.array([PHI**2, PHI**1, PHI**0, PHI**(-1), 0.0])
    pos_weights = pos_weights / np.sum(pos_weights)
    
    for features, depth in test_data:
        h, w = depth.shape
        
        # Stack features
        feature_matrix = np.column_stack([
            features['vertical'].flatten(),
            features['edges'].flatten(),
            features['frequency'].flatten(),
            features['saliency'].flatten(),
            features['luminance'].flatten(),
        ])
        
        # SVD prediction
        X_test_centered = feature_matrix - X_mean
        X_test_proj = X_test_centered @ Vt[:n_components, :].T
        y_pred_svd = X_test_proj @ weights
        y_pred_svd = np.clip(y_pred_svd, 0, 1)
        pred_svd = y_pred_svd.reshape(h, w)
        pred_svd = gaussian_filter(pred_svd, sigma=2.0)
        pred_svd = _normalize(pred_svd)
        
        # Positional baseline
        y_pred_pos = feature_matrix @ pos_weights
        pred_pos = y_pred_pos.reshape(h, w)
        pred_pos = gaussian_filter(pred_pos, sigma=2.0)
        pred_pos = _normalize(pred_pos)
        
        test_errors_svd.append(np.mean(np.abs(pred_svd - depth)))
        test_errors_pos.append(np.mean(np.abs(pred_pos - depth)))
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Positional (φ-weights):  Test MAE = {np.mean(test_errors_pos):.4f}")
    print(f"  SVD-optimal weights:     Test MAE = {np.mean(test_errors_svd):.4f}")
    
    improvement = np.mean(test_errors_pos) - np.mean(test_errors_svd)
    print(f"\nImprovement: {improvement:.4f} ({100*improvement/np.mean(test_errors_pos):.1f}%)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PART 1: Raw patch SVD + Positional")
    print("="*70)
    model = run_combined_experiment(n_train=50, n_test=10)
    
    print("\n" + "="*70)
    print("PART 2: SVD on Positional Features")
    print("="*70)
    run_svd_on_features_experiment(n_train=50, n_test=10)
