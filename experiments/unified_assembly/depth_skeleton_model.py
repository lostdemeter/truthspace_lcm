#!/usr/bin/env python3
"""
Experiment: Geometric Skeleton as Structural Prior

Hypothesis: Our φ-holographic geometry provides the SKELETON that a 
subsequent model can use to achieve better depth estimation.

The skeleton provides:
1. Relevant feature dimensions (edges, vertical gradient, frequency, saliency)
2. φ-weighted composition (how dimensions combine)
3. Phase relationships (where to add vs subtract)

A simple learner on top of this skeleton should outperform:
- The skeleton alone (our current ~0.19 MAE)
- A learner on raw pixels (no geometric prior)

This tests whether our geometry captures genuine structural information.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.fft import fft2, ifft2, fftshift
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
# GEOMETRIC FEATURE EXTRACTORS (The Skeleton)
# =============================================================================

def extract_vertical_gradient(gray: np.ndarray) -> np.ndarray:
    """Structural prior: lower in frame = typically closer."""
    h, w = gray.shape
    return np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Structural prior: depth changes at boundaries."""
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    return _normalize(np.sqrt(grad_x**2 + grad_y**2))


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    """Structural prior: fine texture = typically closer."""
    F = fft2(gray)
    F_shifted = fftshift(F)
    h, w = gray.shape
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
    F_filtered = F_shifted * H
    from scipy.fft import ifftshift
    filtered = np.abs(ifft2(ifftshift(F_filtered)))
    return _normalize(filtered)


def extract_saliency(gray: np.ndarray) -> np.ndarray:
    """Structural prior: salient regions have distinct depth."""
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
# SKELETON EXTRACTOR
# =============================================================================

class GeometricSkeleton:
    """
    Extracts the geometric skeleton from an image.
    
    The skeleton consists of:
    1. Feature dimensions with φ-weights
    2. The composed depth prior
    3. Per-dimension features for downstream learning
    """
    
    def __init__(self):
        # φ-weighted dimensions (learned from previous experiments)
        self.dimensions = {
            'vertical_gradient': PHI**2,  # ~2.618
            'edges': PHI**1.5,             # ~2.058
            'frequency': PHI**0,           # 1.0
            'saliency': PHI**(-1),         # ~0.618
        }
        
        # Phase corrections (learned)
        self.phase_corrections = {
            'edges': -0.4,  # Destructive - reduce edge contribution
        }
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all skeleton features from an image."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        gray = _normalize(gray)
        
        features = {
            'vertical_gradient': extract_vertical_gradient(gray),
            'edges': extract_edges(gray),
            'frequency': extract_frequency(gray),
            'saliency': extract_saliency(gray),
            'luminance': gray,  # Raw luminance as additional feature
        }
        
        return features
    
    def compose_prior(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compose the geometric depth prior using φ-weights.
        
        This is what our skeleton "believes" depth should be,
        based purely on geometric structure.
        """
        total = None
        total_weight = 0
        
        for name, weight in self.dimensions.items():
            if name not in features:
                continue
            
            feat = features[name]
            
            # Apply phase correction if exists
            if name in self.phase_corrections:
                alpha = self.phase_corrections[name]
                # Phase correction modifies contribution
                effective_weight = weight * (1 + alpha)
            else:
                effective_weight = weight
            
            if effective_weight <= 0:
                continue
            
            if total is None:
                total = effective_weight * feat
            else:
                total = total + effective_weight * feat
            total_weight += effective_weight
        
        if total is None:
            return np.zeros_like(list(features.values())[0])
        
        prior = total / total_weight
        prior = gaussian_filter(prior, sigma=2.0)
        return _normalize(prior)
    
    def get_skeleton_representation(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get the full skeleton representation:
        - The composed prior (what geometry predicts)
        - Individual features (for downstream learning)
        """
        features = self.extract_features(image)
        prior = self.compose_prior(features)
        return prior, features


# =============================================================================
# SIMPLE LEARNER ON TOP OF SKELETON
# =============================================================================

class SkeletonRefinementModel:
    """
    A simple model that learns to refine the geometric skeleton.
    
    Instead of learning depth from scratch, it learns:
    depth = skeleton_prior + learned_correction(skeleton_features)
    
    This tests whether the skeleton provides useful structure.
    """
    
    def __init__(self):
        self.skeleton = GeometricSkeleton()
        
        # Learnable parameters: per-feature correction weights
        # These are learned from data
        self.correction_weights = {}
        self.bias = 0.0
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using skeleton + learned corrections."""
        prior, features = self.skeleton.get_skeleton_representation(image)
        
        # Start with geometric prior
        depth = prior.copy()
        
        # Add learned corrections
        for name, weight in self.correction_weights.items():
            if name in features:
                depth = depth + weight * features[name]
        
        depth = depth + self.bias
        return np.clip(depth, 0, 1)
    
    def fit(self, images: List[np.ndarray], depths: List[np.ndarray], 
            n_iterations: int = 10):
        """
        Learn correction weights from training data.
        
        Uses coordinate descent to find optimal weights.
        """
        print(f"Fitting on {len(images)} images...")
        
        # Pre-extract skeleton representations
        all_priors = []
        all_features = []
        all_targets = []
        
        for img, depth in zip(images, depths):
            prior, features = self.skeleton.get_skeleton_representation(img)
            
            # Resize depth to match
            if depth.shape != prior.shape:
                pil = Image.fromarray((depth * 255).astype(np.uint8))
                pil = pil.resize((prior.shape[1], prior.shape[0]), Image.BILINEAR)
                depth = np.array(pil).astype(np.float32) / 255.0
            
            all_priors.append(prior)
            all_features.append(features)
            all_targets.append(depth)
        
        # Initialize correction weights
        feature_names = list(all_features[0].keys())
        for name in feature_names:
            self.correction_weights[name] = 0.0
        
        # Compute baseline MAE (skeleton only)
        baseline_mae = self._compute_mae(all_priors, all_targets)
        print(f"  Skeleton-only MAE: {baseline_mae:.4f}")
        
        # Coordinate descent
        search_values = np.linspace(-0.3, 0.3, 13)
        
        best_mae = baseline_mae
        
        for iteration in range(n_iterations):
            improved = False
            
            for name in feature_names:
                best_weight = self.correction_weights[name]
                
                for w in search_values:
                    self.correction_weights[name] = w
                    
                    # Compute predictions
                    preds = []
                    for i, prior in enumerate(all_priors):
                        pred = prior.copy()
                        for n, weight in self.correction_weights.items():
                            if n in all_features[i]:
                                pred = pred + weight * all_features[i][n]
                        pred = np.clip(pred, 0, 1)
                        preds.append(pred)
                    
                    mae = self._compute_mae(preds, all_targets)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_weight = w
                        improved = True
                
                self.correction_weights[name] = best_weight
            
            if not improved:
                break
            
            print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
        
        # Report learned weights
        print(f"\nLearned correction weights:")
        for name, weight in sorted(self.correction_weights.items(), 
                                   key=lambda x: abs(x[1]), reverse=True):
            if abs(weight) > 0.01:
                print(f"  {name}: {weight:+.3f}")
        
        return best_mae
    
    def _compute_mae(self, predictions: List[np.ndarray], 
                     targets: List[np.ndarray]) -> float:
        total = 0
        for pred, target in zip(predictions, targets):
            total += np.mean(np.abs(pred - target))
        return total / len(predictions)


# =============================================================================
# BASELINE: LEARNING WITHOUT SKELETON
# =============================================================================

class NoSkeletonModel:
    """
    Baseline: Learn depth directly from raw features without geometric prior.
    
    Uses NORMALIZED weighted sum (same as skeleton) to be a fair comparison.
    The difference is: skeleton has φ-weights baked in, this learns from scratch.
    """
    
    def __init__(self):
        self.weights = {}
    
    def extract_raw_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract raw features (same as skeleton but no composition)."""
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
            'luminance': gray,
        }
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        features = self.extract_raw_features(image)
        
        # Normalized weighted sum (same structure as skeleton)
        total = None
        total_weight = 0
        
        for name, weight in self.weights.items():
            if name in features and weight > 0:
                if total is None:
                    total = weight * features[name]
                else:
                    total = total + weight * features[name]
                total_weight += weight
        
        if total is None or total_weight == 0:
            return np.ones_like(list(features.values())[0]) * 0.5
        
        depth = total / total_weight
        depth = gaussian_filter(depth, sigma=2.0)
        return _normalize(depth)
    
    def fit(self, images: List[np.ndarray], depths: List[np.ndarray],
            n_iterations: int = 10, use_phi_init: bool = False):
        """Learn weights from scratch (no geometric prior)."""
        print(f"Fitting NO-SKELETON model on {len(images)} images...")
        print(f"  Using φ-initialization: {use_phi_init}")
        
        # Pre-extract features
        all_features = []
        all_targets = []
        
        for img, depth in zip(images, depths):
            features = self.extract_raw_features(img)
            target_shape = list(features.values())[0].shape
            
            if depth.shape != target_shape:
                pil = Image.fromarray((depth * 255).astype(np.uint8))
                pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                depth = np.array(pil).astype(np.float32) / 255.0
            
            all_features.append(features)
            all_targets.append(depth)
        
        # Initialize
        feature_names = list(all_features[0].keys())
        
        if use_phi_init:
            # Initialize with φ-weights (same as skeleton)
            phi_weights = {
                'vertical_gradient': PHI**2,
                'edges': PHI**1.5,
                'frequency': PHI**0,
                'saliency': PHI**(-1),
                'luminance': 0.1,
            }
            for name in feature_names:
                self.weights[name] = phi_weights.get(name, 0.1)
        else:
            # Random positive initialization
            for name in feature_names:
                self.weights[name] = 1.0  # Equal weights
        
        # Coordinate descent - search positive weights only (normalized sum)
        search_values = np.linspace(0.0, 5.0, 21)
        
        best_mae = float('inf')
        
        for iteration in range(n_iterations):
            improved = False
            
            # Optimize weights
            for name in feature_names:
                best_weight = self.weights[name]
                
                for w in search_values:
                    self.weights[name] = w
                    preds = self._predict_all(all_features)
                    mae = self._compute_mae(preds, all_targets)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_weight = w
                        improved = True
                
                self.weights[name] = best_weight
            
            if not improved:
                break
            
            print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
        
        print(f"\nLearned weights (no skeleton):")
        for name, weight in sorted(self.weights.items(),
                                   key=lambda x: abs(x[1]), reverse=True):
            if abs(weight) > 0.01:
                print(f"  {name}: {weight:+.3f}")
        
        return best_mae
    
    def _predict_all(self, all_features: List[Dict]) -> List[np.ndarray]:
        preds = []
        for features in all_features:
            # Normalized weighted sum (same as skeleton)
            total = None
            total_weight = 0
            
            for name, weight in self.weights.items():
                if name in features and weight > 0:
                    if total is None:
                        total = weight * features[name]
                    else:
                        total = total + weight * features[name]
                    total_weight += weight
            
            if total is None or total_weight == 0:
                depth = np.ones_like(list(features.values())[0]) * 0.5
            else:
                depth = total / total_weight
                depth = gaussian_filter(depth, sigma=2.0)
                depth = _normalize(depth)
            
            preds.append(depth)
        return preds
    
    def _compute_mae(self, predictions: List[np.ndarray],
                     targets: List[np.ndarray]) -> float:
        total = 0
        for pred, target in zip(predictions, targets):
            total += np.mean(np.abs(pred - target))
        return total / len(predictions)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_skeleton_experiment(n_train: int = 50, n_test: int = 10):
    """
    Compare:
    1. Skeleton only (no learning)
    2. Skeleton + learned corrections
    3. No skeleton (learn from scratch)
    """
    print("=" * 70)
    print("EXPERIMENT: Geometric Skeleton as Structural Prior")
    print("=" * 70)
    print()
    print("Hypothesis: The skeleton provides structural information that")
    print("makes subsequent learning more effective.")
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
    
    # Model 1: Skeleton only
    print("\n" + "=" * 60)
    print("MODEL 1: Skeleton Only (No Learning)")
    print("=" * 60)
    
    skeleton = GeometricSkeleton()
    skeleton_preds = []
    skeleton_targets = []
    
    for img, depth in zip(test_images, test_depths):
        prior, _ = skeleton.get_skeleton_representation(img)
        
        if depth.shape != prior.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((prior.shape[1], prior.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        skeleton_preds.append(prior)
        skeleton_targets.append(depth)
    
    skeleton_mae = sum(np.mean(np.abs(p - t)) for p, t in 
                       zip(skeleton_preds, skeleton_targets)) / len(skeleton_preds)
    print(f"Test MAE: {skeleton_mae:.4f}")
    
    # Model 2: Skeleton + Learned Corrections
    print("\n" + "=" * 60)
    print("MODEL 2: Skeleton + Learned Corrections")
    print("=" * 60)
    
    skeleton_model = SkeletonRefinementModel()
    skeleton_model.fit(train_images, train_depths, n_iterations=10)
    
    refined_preds = []
    for img, depth in zip(test_images, test_depths):
        pred = skeleton_model.predict(img)
        
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        refined_preds.append((pred, depth))
    
    refined_mae = sum(np.mean(np.abs(p - t)) for p, t in refined_preds) / len(refined_preds)
    print(f"Test MAE: {refined_mae:.4f}")
    
    # Model 3: No Skeleton (Learn from Scratch)
    print("\n" + "=" * 60)
    print("MODEL 3: No Skeleton (Learn from Scratch, zero init)")
    print("=" * 60)
    
    no_skeleton_model = NoSkeletonModel()
    no_skeleton_model.fit(train_images, train_depths, n_iterations=10, use_phi_init=False)
    
    no_skeleton_preds = []
    for img, depth in zip(test_images, test_depths):
        pred = no_skeleton_model.predict(img)
        
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        no_skeleton_preds.append((pred, depth))
    
    no_skeleton_mae = sum(np.mean(np.abs(p - t)) for p, t in no_skeleton_preds) / len(no_skeleton_preds)
    print(f"Test MAE: {no_skeleton_mae:.4f}")
    
    # Model 4: No Skeleton but WITH φ-initialization
    print("\n" + "=" * 60)
    print("MODEL 4: No Skeleton (φ-initialized weights)")
    print("=" * 60)
    
    phi_init_model = NoSkeletonModel()
    phi_init_model.fit(train_images, train_depths, n_iterations=10, use_phi_init=True)
    
    phi_init_preds = []
    for img, depth in zip(test_images, test_depths):
        pred = phi_init_model.predict(img)
        
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        phi_init_preds.append((pred, depth))
    
    phi_init_mae = sum(np.mean(np.abs(p - t)) for p, t in phi_init_preds) / len(phi_init_preds)
    print(f"Test MAE: {phi_init_mae:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  1. Skeleton Only:              MAE = {skeleton_mae:.4f}")
    print(f"  2. Skeleton + Corrections:     MAE = {refined_mae:.4f}")
    print(f"  3. No Skeleton (zero init):    MAE = {no_skeleton_mae:.4f}")
    print(f"  4. No Skeleton (φ-init):       MAE = {phi_init_mae:.4f}")
    print()
    
    print("Analysis:")
    print(f"  φ-initialization alone:     {no_skeleton_mae - phi_init_mae:.4f} improvement")
    print(f"  Skeleton composition:       {phi_init_mae - skeleton_mae:.4f} improvement")
    print(f"  Learned corrections:        {skeleton_mae - refined_mae:.4f} improvement")
    print()
    
    print("What the skeleton provides:")
    print(f"  - φ-weights as starting point (vs zero): {no_skeleton_mae - phi_init_mae:.4f}")
    print(f"  - Proper composition structure:          {phi_init_mae - skeleton_mae:.4f}")
    print(f"  - Foundation for refinement:             {skeleton_mae - refined_mae:.4f}")
    
    return skeleton_model, no_skeleton_model


if __name__ == "__main__":
    skeleton_model, no_skeleton_model = run_skeleton_experiment(n_train=50, n_test=10)
