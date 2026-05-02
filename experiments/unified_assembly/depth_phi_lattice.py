#!/usr/bin/env python3
"""
Experiment: φ-Lattice Coordinates for Depth Estimation

Key insight from Design 099: Use ABSOLUTE φ-lattice coordinates instead of
relative/learned weights.

For depth estimation:
- Each pixel has a position on the φ-lattice
- Dimensions have SEMANTIC meaning for depth:
  - vertical_position: φ^k where k encodes height in frame
  - edge_strength: φ^k where k encodes boundary importance  
  - texture_frequency: φ^k where k encodes detail level
  - saliency: φ^k where k encodes attention importance

φ-Zipf Duality:
- Low frequency features (rare) → HIGH importance → high φ-level
- High frequency features (common) → LOW importance → low φ-level
- This naturally weights depth cues by their informativeness

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.fft import fft2, ifft2, fftshift, ifftshift
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
# φ-LATTICE COORDINATE SYSTEM FOR DEPTH
# =============================================================================

# Semantic dimensions for depth estimation
DEPTH_DIMENSIONS = {
    0: {
        'name': 'vertical_position',
        'description': 'Height in frame (bottom=close, top=far)',
        'levels': {
            # φ^k levels for vertical position
            3: 'very_bottom',   # φ³ ≈ 4.24 - closest
            2: 'lower',         # φ² ≈ 2.62
            1: 'middle',        # φ¹ ≈ 1.62
            0: 'upper',         # φ⁰ = 1.0
            -1: 'top',          # φ⁻¹ ≈ 0.62 - farthest
        }
    },
    1: {
        'name': 'edge_strength',
        'description': 'Boundary strength (strong edges = depth discontinuity)',
        'levels': {
            2: 'strong_edge',   # φ² - major depth boundary
            1: 'moderate_edge', # φ¹ - minor boundary
            0: 'weak_edge',     # φ⁰ - texture edge
            -1: 'no_edge',      # φ⁻¹ - smooth region
        }
    },
    2: {
        'name': 'texture_frequency',
        'description': 'Detail level (fine texture = close)',
        'levels': {
            2: 'fine_detail',   # φ² - very close
            1: 'medium_detail', # φ¹ - moderate distance
            0: 'coarse',        # φ⁰ - far
            -1: 'smooth',       # φ⁻¹ - very far / sky
        }
    },
    3: {
        'name': 'saliency',
        'description': 'Visual attention (salient = distinct depth)',
        'levels': {
            2: 'highly_salient', # φ² - foreground object
            1: 'salient',        # φ¹ - notable feature
            0: 'neutral',        # φ⁰ - background
            -1: 'ignored',       # φ⁻¹ - sky/uniform
        }
    },
}


def quantize_to_phi_level(value: float, min_level: int = -3, max_level: int = 3) -> int:
    """
    Quantize a [0,1] value to the nearest φ-level.
    
    This is the key operation: continuous features → discrete φ-lattice.
    """
    if value <= 0:
        return min_level
    
    # Find k such that φ^k is closest to value (scaled)
    # value in [0,1] maps to φ^k in [φ^min, φ^max]
    
    # Scale value to φ range
    phi_min = PHI ** min_level
    phi_max = PHI ** max_level
    scaled = phi_min + value * (phi_max - phi_min)
    
    # Find nearest φ^k
    k = round(np.log(scaled) / np.log(PHI))
    k = max(min_level, min(max_level, k))
    
    return k


def phi_level_to_value(k: int) -> float:
    """Convert φ-level back to value."""
    return PHI ** k


class PhiLatticeDepth:
    """
    Depth estimation using absolute φ-lattice coordinates.
    
    Each pixel is assigned a position on the 4D φ-lattice:
    [vertical_level, edge_level, texture_level, saliency_level]
    
    Depth is computed by combining these levels with φ-Zipf weighting.
    """
    
    def __init__(self):
        # φ-Zipf weights: importance inversely proportional to frequency
        # Vertical position is most informative (rare to have depth info)
        # Edges are next (boundaries are informative)
        # Texture and saliency are supplementary
        self.dimension_weights = {
            'vertical_position': PHI**2,   # Most important
            'edge_strength': PHI**1,       # Important
            'texture_frequency': PHI**0,   # Moderate
            'saliency': PHI**(-1),         # Supplementary
        }
        
        # Training data
        self.training_data = []
        
        # Learned level-to-depth mapping (optional refinement)
        self.level_depth_map = None
    
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        return image.copy()
    
    def _resize(self, arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        if arr.shape == shape:
            return arr
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0
    
    def extract_phi_levels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract φ-level for each dimension at each pixel.
        
        Returns integer arrays where each value is a φ-level (k in φ^k).
        """
        gray = _normalize(self._to_gray(image))
        h, w = gray.shape
        
        levels = {}
        
        # Dimension 0: Vertical position
        # Bottom of frame = high level (close), top = low level (far)
        vertical = np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
        levels['vertical_position'] = np.vectorize(
            lambda v: quantize_to_phi_level(v, -1, 3)
        )(vertical)
        
        # Dimension 1: Edge strength
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        edges = _normalize(np.sqrt(grad_x**2 + grad_y**2))
        levels['edge_strength'] = np.vectorize(
            lambda v: quantize_to_phi_level(v, -1, 2)
        )(edges)
        
        # Dimension 2: Texture frequency
        F = fft2(gray)
        F_shifted = fftshift(F)
        u = np.arange(w) - w // 2
        v = np.arange(h) - h // 2
        U, V = np.meshgrid(u, v)
        H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
        F_filtered = F_shifted * H
        frequency = _normalize(np.abs(ifft2(ifftshift(F_filtered))))
        levels['texture_frequency'] = np.vectorize(
            lambda v: quantize_to_phi_level(v, -1, 2)
        )(frequency)
        
        # Dimension 3: Saliency
        amplitude = np.abs(F)
        log_amplitude = np.log(amplitude + 1e-10)
        log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
        residual = log_amplitude - log_amplitude_smoothed
        phase = np.angle(F)
        F_residual = np.exp(residual + 1j * phase)
        saliency = _normalize(gaussian_filter(np.abs(ifft2(F_residual))**2, sigma=5.0))
        levels['saliency'] = np.vectorize(
            lambda v: quantize_to_phi_level(v, -1, 2)
        )(saliency)
        
        return levels
    
    def levels_to_depth(self, levels: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert φ-levels to depth using φ-Zipf weighted combination.
        
        depth = Σ (weight_d × φ^level_d) / Σ weight_d
        
        This is the key: each dimension contributes φ^level, weighted by importance.
        """
        total = None
        total_weight = 0
        
        for dim_name, weight in self.dimension_weights.items():
            level_array = levels[dim_name]
            
            # Convert levels to φ values
            phi_values = np.vectorize(phi_level_to_value)(level_array)
            
            if total is None:
                total = weight * phi_values
            else:
                total = total + weight * phi_values
            total_weight += weight
        
        depth = total / total_weight
        depth = gaussian_filter(depth, sigma=2.0)
        
        return _normalize(depth)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using φ-lattice coordinates."""
        levels = self.extract_phi_levels(image)
        return self.levels_to_depth(levels)
    
    def add_training_image(self, rgb: np.ndarray, depth: np.ndarray, image_id: str):
        self.training_data.append((rgb, depth, image_id))
    
    def learn_level_mapping(self):
        """
        Learn optimal mapping from φ-levels to depth.
        
        Instead of using φ^k directly, learn what depth each level combination
        should map to based on training data.
        """
        if not self.training_data:
            return
        
        print(f"Learning level-to-depth mapping from {len(self.training_data)} images...")
        
        # Collect (level_tuple, depth) pairs
        level_depth_pairs = {}
        
        for rgb, true_depth, _ in self.training_data:
            levels = self.extract_phi_levels(rgb)
            h, w = list(levels.values())[0].shape
            
            true_resized = self._resize(true_depth, (h, w))
            
            for i in range(h):
                for j in range(w):
                    level_tuple = (
                        levels['vertical_position'][i, j],
                        levels['edge_strength'][i, j],
                        levels['texture_frequency'][i, j],
                        levels['saliency'][i, j],
                    )
                    
                    if level_tuple not in level_depth_pairs:
                        level_depth_pairs[level_tuple] = []
                    level_depth_pairs[level_tuple].append(true_resized[i, j])
        
        # Average depth for each level combination
        self.level_depth_map = {}
        for level_tuple, depths in level_depth_pairs.items():
            self.level_depth_map[level_tuple] = np.mean(depths)
        
        print(f"  Learned mapping for {len(self.level_depth_map)} level combinations")
        
        # Show some examples
        print("\n  Sample level → depth mappings:")
        sorted_levels = sorted(self.level_depth_map.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for levels, depth in sorted_levels:
            print(f"    {levels} → depth={depth:.3f}")
    
    def predict_with_learned_mapping(self, image: np.ndarray) -> np.ndarray:
        """Predict using learned level-to-depth mapping."""
        if self.level_depth_map is None:
            return self.predict(image)
        
        levels = self.extract_phi_levels(image)
        h, w = list(levels.values())[0].shape
        
        depth = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                level_tuple = (
                    levels['vertical_position'][i, j],
                    levels['edge_strength'][i, j],
                    levels['texture_frequency'][i, j],
                    levels['saliency'][i, j],
                )
                
                if level_tuple in self.level_depth_map:
                    depth[i, j] = self.level_depth_map[level_tuple]
                else:
                    # Fallback: use φ-weighted combination
                    phi_sum = 0
                    weight_sum = 0
                    for dim_idx, (dim_name, weight) in enumerate(self.dimension_weights.items()):
                        phi_sum += weight * PHI ** level_tuple[dim_idx]
                        weight_sum += weight
                    depth[i, j] = phi_sum / weight_sum
        
        depth = gaussian_filter(depth, sigma=2.0)
        return _normalize(depth)
    
    def compute_mae(self, use_learned: bool = False) -> float:
        """Compute MAE on training data."""
        if not self.training_data:
            return float('inf')
        
        total_mae = 0
        for rgb, true_depth, _ in self.training_data:
            if use_learned:
                pred = self.predict_with_learned_mapping(rgb)
            else:
                pred = self.predict(rgb)
            
            true_resized = self._resize(true_depth, pred.shape)
            total_mae += np.mean(np.abs(pred - true_resized))
        
        return total_mae / len(self.training_data)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_phi_lattice_experiment(n_train: int = 50, n_test: int = 10):
    """
    Compare:
    1. φ-lattice with default φ^k mapping
    2. φ-lattice with learned level-to-depth mapping
    3. Previous best (positional with phase corrections)
    """
    print("=" * 70)
    print("EXPERIMENT: φ-Lattice Coordinates for Depth")
    print("=" * 70)
    print()
    print("Key insight: Use ABSOLUTE φ-lattice positions instead of learned weights")
    print()
    print("Dimensions:")
    for dim_id, dim_info in DEPTH_DIMENSIONS.items():
        print(f"  {dim_id}: {dim_info['name']} - {dim_info['description']}")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    model = PhiLatticeDepth()
    test_data = []
    
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
            model.add_training_image(rgb, depth, img_id)
        else:
            test_data.append((rgb, depth, img_id))
    
    print(f"  Train: {len(model.training_data)}, Test: {len(test_data)}")
    
    # Test 1: Default φ^k mapping
    print("\n" + "=" * 60)
    print("TEST 1: Default φ^k Mapping")
    print("=" * 60)
    
    train_mae_default = model.compute_mae(use_learned=False)
    print(f"  Train MAE: {train_mae_default:.4f}")
    
    test_errors_default = []
    for rgb, depth, _ in test_data:
        pred = model.predict(rgb)
        true_resized = model._resize(depth, pred.shape)
        test_errors_default.append(np.mean(np.abs(pred - true_resized)))
    
    print(f"  Test MAE: {np.mean(test_errors_default):.4f}")
    
    # Test 2: Learned level-to-depth mapping
    print("\n" + "=" * 60)
    print("TEST 2: Learned Level-to-Depth Mapping")
    print("=" * 60)
    
    model.learn_level_mapping()
    
    train_mae_learned = model.compute_mae(use_learned=True)
    print(f"  Train MAE: {train_mae_learned:.4f}")
    
    test_errors_learned = []
    for rgb, depth, _ in test_data:
        pred = model.predict_with_learned_mapping(rgb)
        true_resized = model._resize(depth, pred.shape)
        test_errors_learned.append(np.mean(np.abs(pred - true_resized)))
    
    print(f"  Test MAE: {np.mean(test_errors_learned):.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  φ-Lattice (default φ^k):     Test MAE = {np.mean(test_errors_default):.4f}")
    print(f"  φ-Lattice (learned mapping): Test MAE = {np.mean(test_errors_learned):.4f}")
    print()
    print("Previous best (positional + phase): Test MAE ≈ 0.199")
    print()
    
    improvement = 0.199 - np.mean(test_errors_learned)
    if improvement > 0:
        print(f"φ-Lattice improvement: {improvement:.4f} ({100*improvement/0.199:.1f}% better)")
    else:
        print(f"φ-Lattice vs previous: {-improvement:.4f} worse")
    
    return model


def run_phi_zipf_experiment(n_train: int = 50, n_test: int = 10):
    """
    φ-Zipf Duality: Weight features by their RARITY (informativeness).
    
    Key insight from Design 039:
    - Low frequency (rare) → HIGH importance → φ^high
    - High frequency (common) → LOW importance → φ^low
    
    For depth features:
    - Vertical gradient: Always present → LOW weight (common)
    - Strong edges: Rare → HIGH weight (informative)
    - Fine texture: Moderate rarity → MODERATE weight
    - High saliency: Rare → HIGH weight
    
    This inverts our previous weighting!
    """
    print("=" * 70)
    print("EXPERIMENT: φ-Zipf Duality Weighting")
    print("=" * 70)
    print()
    print("Insight: Weight features by RARITY (informativeness), not presence")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
    test_data = []
    
    print("Loading data and computing feature statistics...")
    
    # First pass: compute feature statistics to determine rarity
    all_features = {
        'vertical': [],
        'edges': [],
        'frequency': [],
        'saliency': [],
    }
    
    for i, img_id in enumerate(available_ids[:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Extract features
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        h, w = gray.shape
        
        # Vertical gradient
        vertical = np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
        
        # Edges
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        edges = _normalize(np.sqrt(grad_x**2 + grad_y**2))
        
        # Frequency
        F = fft2(gray)
        F_shifted = fftshift(F)
        u = np.arange(w) - w // 2
        v = np.arange(h) - h // 2
        U, V = np.meshgrid(u, v)
        H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
        F_filtered = F_shifted * H
        frequency = _normalize(np.abs(ifft2(ifftshift(F_filtered))))
        
        # Saliency
        amplitude = np.abs(F)
        log_amplitude = np.log(amplitude + 1e-10)
        log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
        residual = log_amplitude - log_amplitude_smoothed
        phase = np.angle(F)
        F_residual = np.exp(residual + 1j * phase)
        saliency = _normalize(gaussian_filter(np.abs(ifft2(F_residual))**2, sigma=5.0))
        
        features = {
            'vertical': vertical,
            'edges': edges,
            'frequency': frequency,
            'saliency': saliency,
        }
        
        if i < n_train:
            train_data.append((features, depth, gray.shape))
            for name in all_features:
                all_features[name].append(features[name].mean())
        else:
            test_data.append((features, depth, gray.shape))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Compute feature statistics
    print("\nFeature statistics (mean ± std):")
    feature_stats = {}
    for name, values in all_features.items():
        mean = np.mean(values)
        std = np.std(values)
        feature_stats[name] = (mean, std)
        print(f"  {name}: {mean:.3f} ± {std:.3f}")
    
    # φ-Zipf weighting: weight by INVERSE of mean (rarer = higher weight)
    # But also consider variance (more variable = more informative)
    print("\nφ-Zipf weights (rarity-based):")
    zipf_weights = {}
    for name, (mean, std) in feature_stats.items():
        # Rarity score: lower mean = rarer = higher weight
        # Variability bonus: higher std = more informative
        rarity = 1.0 / (mean + 0.1)  # Avoid division by zero
        variability = 1.0 + std
        
        # Map to φ-level
        raw_weight = rarity * variability
        zipf_weights[name] = raw_weight
    
    # Normalize to sum to 1
    total = sum(zipf_weights.values())
    for name in zipf_weights:
        zipf_weights[name] /= total
        print(f"  {name}: {zipf_weights[name]:.3f}")
    
    # Compare with our previous φ-weights
    print("\nPrevious φ-weights (for comparison):")
    prev_weights = {
        'vertical': PHI**2 / (PHI**2 + PHI + 1 + PHI**(-1)),
        'edges': PHI / (PHI**2 + PHI + 1 + PHI**(-1)),
        'frequency': 1.0 / (PHI**2 + PHI + 1 + PHI**(-1)),
        'saliency': PHI**(-1) / (PHI**2 + PHI + 1 + PHI**(-1)),
    }
    for name, w in prev_weights.items():
        print(f"  {name}: {w:.3f}")
    
    # Test both weightings
    def predict_with_weights(features, weights, shape):
        total = None
        for name, w in weights.items():
            feat = features[name]
            if feat.shape != shape:
                pil = Image.fromarray((feat * 255).astype(np.uint8))
                pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
                feat = np.array(pil).astype(np.float32) / 255.0
            
            if total is None:
                total = w * feat
            else:
                total = total + w * feat
        
        total = gaussian_filter(total, sigma=2.0)
        return _normalize(total)
    
    # Test on training data
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Previous weights
    train_errors_prev = []
    for features, depth, shape in train_data:
        pred = predict_with_weights(features, prev_weights, shape)
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        train_errors_prev.append(np.mean(np.abs(pred - depth)))
    
    # Zipf weights
    train_errors_zipf = []
    for features, depth, shape in train_data:
        pred = predict_with_weights(features, zipf_weights, shape)
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        train_errors_zipf.append(np.mean(np.abs(pred - depth)))
    
    print(f"\nTrain MAE:")
    print(f"  Previous φ-weights: {np.mean(train_errors_prev):.4f}")
    print(f"  φ-Zipf weights:     {np.mean(train_errors_zipf):.4f}")
    
    # Test
    test_errors_prev = []
    test_errors_zipf = []
    
    for features, depth, shape in test_data:
        pred_prev = predict_with_weights(features, prev_weights, shape)
        pred_zipf = predict_with_weights(features, zipf_weights, shape)
        
        if depth.shape != pred_prev.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred_prev.shape[1], pred_prev.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        test_errors_prev.append(np.mean(np.abs(pred_prev - depth)))
        test_errors_zipf.append(np.mean(np.abs(pred_zipf - depth)))
    
    print(f"\nTest MAE:")
    print(f"  Previous φ-weights: {np.mean(test_errors_prev):.4f}")
    print(f"  φ-Zipf weights:     {np.mean(test_errors_zipf):.4f}")
    
    improvement = np.mean(test_errors_prev) - np.mean(test_errors_zipf)
    if improvement > 0:
        print(f"\nφ-Zipf improvement: {improvement:.4f} ({100*improvement/np.mean(test_errors_prev):.1f}% better)")
    else:
        print(f"\nφ-Zipf vs previous: {-improvement:.4f} worse")
    
    return zipf_weights


def run_phi_zipf_correlation_experiment(n_train: int = 50, n_test: int = 10):
    """
    Correct φ-Zipf interpretation: Weight by CORRELATION with depth.
    
    The insight: φ-Zipf duality is about INFORMATION content, not rarity.
    - Features that correlate strongly with depth → HIGH weight
    - Features that correlate weakly with depth → LOW weight
    
    This is the "importance" interpretation of Zipf's law:
    - Rare words carry more information (in language)
    - Rare depth correlations carry more information (in vision)
    
    But "rare" here means "rare to find a feature that predicts depth well",
    not "rare to find the feature in the image".
    """
    print("=" * 70)
    print("EXPERIMENT: φ-Zipf by Depth Correlation")
    print("=" * 70)
    print()
    print("Insight: Weight features by their CORRELATION with depth")
    print("(This is the correct interpretation of φ-Zipf for prediction)")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
    test_data = []
    
    print("Loading data and computing correlations...")
    
    # Collect feature-depth correlations
    correlations = {
        'vertical': [],
        'edges': [],
        'frequency': [],
        'saliency': [],
    }
    
    for i, img_id in enumerate(available_ids[:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Extract features
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        h, w = gray.shape
        
        # Resize depth to match
        if depth.shape != (h, w):
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((w, h), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        # Vertical gradient
        vertical = np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
        
        # Edges
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        edges = _normalize(np.sqrt(grad_x**2 + grad_y**2))
        
        # Frequency
        F = fft2(gray)
        F_shifted = fftshift(F)
        u = np.arange(w) - w // 2
        v = np.arange(h) - h // 2
        U, V = np.meshgrid(u, v)
        H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
        F_filtered = F_shifted * H
        frequency = _normalize(np.abs(ifft2(ifftshift(F_filtered))))
        
        # Saliency
        amplitude = np.abs(F)
        log_amplitude = np.log(amplitude + 1e-10)
        log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
        residual = log_amplitude - log_amplitude_smoothed
        phase = np.angle(F)
        F_residual = np.exp(residual + 1j * phase)
        saliency = _normalize(gaussian_filter(np.abs(ifft2(F_residual))**2, sigma=5.0))
        
        features = {
            'vertical': vertical,
            'edges': edges,
            'frequency': frequency,
            'saliency': saliency,
        }
        
        if i < n_train:
            train_data.append((features, depth, (h, w)))
            
            # Compute correlation with depth for each feature
            depth_flat = depth.flatten()
            for name, feat in features.items():
                feat_flat = feat.flatten()
                corr = np.corrcoef(feat_flat, depth_flat)[0, 1]
                correlations[name].append(abs(corr))  # Use absolute correlation
        else:
            test_data.append((features, depth, (h, w)))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Compute mean correlations
    print("\nFeature-Depth Correlations (mean |r|):")
    mean_corrs = {}
    for name, corrs in correlations.items():
        mean_corr = np.mean(corrs)
        mean_corrs[name] = mean_corr
        print(f"  {name}: {mean_corr:.3f}")
    
    # φ-Zipf weights: proportional to correlation (high correlation = high weight)
    print("\nφ-Zipf weights (correlation-based):")
    
    # Map correlations to φ-levels
    # Highest correlation → φ^2, lowest → φ^(-1)
    sorted_features = sorted(mean_corrs.items(), key=lambda x: x[1], reverse=True)
    phi_levels = [2, 1, 0, -1]  # Assign levels by rank
    
    zipf_weights = {}
    for (name, corr), level in zip(sorted_features, phi_levels):
        weight = PHI ** level
        zipf_weights[name] = weight
        print(f"  {name}: corr={corr:.3f} → φ^{level} = {weight:.3f}")
    
    # Normalize
    total = sum(zipf_weights.values())
    for name in zipf_weights:
        zipf_weights[name] /= total
    
    print("\nNormalized weights:")
    for name, w in zipf_weights.items():
        print(f"  {name}: {w:.3f}")
    
    # Compare with previous
    print("\nPrevious φ-weights:")
    prev_weights = {
        'vertical': PHI**2 / (PHI**2 + PHI + 1 + PHI**(-1)),
        'edges': PHI / (PHI**2 + PHI + 1 + PHI**(-1)),
        'frequency': 1.0 / (PHI**2 + PHI + 1 + PHI**(-1)),
        'saliency': PHI**(-1) / (PHI**2 + PHI + 1 + PHI**(-1)),
    }
    for name, w in prev_weights.items():
        print(f"  {name}: {w:.3f}")
    
    # Test
    def predict_with_weights(features, weights, shape):
        total = None
        for name, w in weights.items():
            feat = features[name]
            if feat.shape != shape:
                pil = Image.fromarray((feat * 255).astype(np.uint8))
                pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
                feat = np.array(pil).astype(np.float32) / 255.0
            
            if total is None:
                total = w * feat
            else:
                total = total + w * feat
        
        total = gaussian_filter(total, sigma=2.0)
        return _normalize(total)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Test errors
    test_errors_prev = []
    test_errors_zipf = []
    
    for features, depth, shape in test_data:
        pred_prev = predict_with_weights(features, prev_weights, shape)
        pred_zipf = predict_with_weights(features, zipf_weights, shape)
        
        if depth.shape != shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        test_errors_prev.append(np.mean(np.abs(pred_prev - depth)))
        test_errors_zipf.append(np.mean(np.abs(pred_zipf - depth)))
    
    print(f"\nTest MAE:")
    print(f"  Previous φ-weights:      {np.mean(test_errors_prev):.4f}")
    print(f"  φ-Zipf (correlation):    {np.mean(test_errors_zipf):.4f}")
    
    # Check if correlation-based weights match our intuition
    print("\n" + "=" * 60)
    print("ANALYSIS: Do correlations match φ-hierarchy?")
    print("=" * 60)
    
    # Our hypothesis: vertical > edges > frequency > saliency
    # Does the data support this?
    print("\nExpected order (by φ-level): vertical > edges > frequency > saliency")
    print(f"Actual order (by correlation): {' > '.join([n for n, _ in sorted_features])}")
    
    # Check if they match
    expected_order = ['vertical', 'edges', 'frequency', 'saliency']
    actual_order = [n for n, _ in sorted_features]
    
    if expected_order == actual_order:
        print("\n✓ Correlations MATCH the φ-hierarchy!")
        print("  This validates the geometric prior.")
    else:
        print("\n✗ Correlations differ from φ-hierarchy")
        print("  The data suggests a different ordering.")
    
    return zipf_weights, mean_corrs


def run_phi_lattice_with_corrections(n_train: int = 50, n_test: int = 10):
    """
    Combine φ-lattice absolute coordinates with learned corrections.
    
    Key insight: The φ-lattice provides the SKELETON (absolute structure),
    and corrections refine it (like phase in the holographic model).
    
    This connects to Design 099:
    - Positions are ABSOLUTE (φ^k on semantic dimensions)
    - Similarity is used for NAVIGATION (corrections)
    - The lattice IS the coordinate system
    """
    print("=" * 70)
    print("EXPERIMENT: φ-Lattice + Learned Corrections")
    print("=" * 70)
    print()
    print("Approach:")
    print("  1. Use correlation-ranked φ-weights (data-validated lattice)")
    print("  2. Learn signed corrections from residuals")
    print("  3. Compare to pure geometric and pure learned")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
    test_data = []
    
    # Correlation-based weights (from previous experiment)
    # vertical > saliency > edges > frequency
    phi_weights = {
        'vertical': PHI**2,
        'saliency': PHI**1,
        'edges': PHI**0,
        'frequency': PHI**(-1),
    }
    total = sum(phi_weights.values())
    phi_weights = {k: v/total for k, v in phi_weights.items()}
    
    print("φ-Lattice weights (correlation-ranked):")
    for name, w in phi_weights.items():
        print(f"  {name}: {w:.3f}")
    
    print("\nLoading data...")
    for i, img_id in enumerate(available_ids[:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Extract features
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        h, w = gray.shape
        
        # Resize depth
        if depth.shape != (h, w):
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((w, h), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        # Vertical gradient (inverted: bottom=1, top=0 for "close at bottom")
        vertical = np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
        
        # Edges
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        edges = _normalize(np.sqrt(grad_x**2 + grad_y**2))
        
        # Frequency
        F = fft2(gray)
        F_shifted = fftshift(F)
        u = np.arange(w) - w // 2
        v = np.arange(h) - h // 2
        U, V = np.meshgrid(u, v)
        H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
        F_filtered = F_shifted * H
        frequency = _normalize(np.abs(ifft2(ifftshift(F_filtered))))
        
        # Saliency
        amplitude = np.abs(F)
        log_amplitude = np.log(amplitude + 1e-10)
        log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
        residual = log_amplitude - log_amplitude_smoothed
        phase = np.angle(F)
        F_residual = np.exp(residual + 1j * phase)
        saliency = _normalize(gaussian_filter(np.abs(ifft2(F_residual))**2, sigma=5.0))
        
        features = {
            'vertical': vertical,
            'edges': edges,
            'frequency': frequency,
            'saliency': saliency,
            'luminance': gray,  # For corrections
        }
        
        if i < n_train:
            train_data.append((features, depth))
        else:
            test_data.append((features, depth))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Step 1: Compute φ-lattice predictions
    print("\n" + "=" * 60)
    print("Step 1: φ-Lattice Baseline")
    print("=" * 60)
    
    def predict_phi_lattice(features):
        total = None
        for name, w in phi_weights.items():
            if total is None:
                total = w * features[name]
            else:
                total = total + w * features[name]
        total = gaussian_filter(total, sigma=2.0)
        return _normalize(total)
    
    train_residuals = []
    train_errors_base = []
    
    for features, depth in train_data:
        pred = predict_phi_lattice(features)
        residual = depth - pred  # SIGNED residual
        train_residuals.append(residual)
        train_errors_base.append(np.mean(np.abs(pred - depth)))
    
    print(f"  Train MAE: {np.mean(train_errors_base):.4f}")
    
    # Step 2: Learn correction from residuals
    print("\n" + "=" * 60)
    print("Step 2: Learning Signed Corrections")
    print("=" * 60)
    
    # Find which feature best predicts residual
    print("\n  Correlation of features with residual:")
    best_corr = 0
    best_feature = None
    
    for fname in ['luminance', 'edges', 'vertical', 'frequency', 'saliency']:
        correlations = []
        for i, (features, _) in enumerate(train_data):
            feature = features[fname]
            residual = train_residuals[i]
            corr = np.corrcoef(feature.flatten(), residual.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_corr = np.mean(correlations)
        print(f"    {fname}: r = {avg_corr:.4f}")
        
        if abs(avg_corr) > abs(best_corr):
            best_corr = avg_corr
            best_feature = fname
    
    print(f"\n  Best predictor: {best_feature} (r = {best_corr:.4f})")
    
    # Find optimal correction coefficient
    best_alpha = 0
    best_mae = np.mean(train_errors_base)
    
    for alpha in np.linspace(-0.5, 0.5, 21):
        total_mae = 0
        for i, (features, depth) in enumerate(train_data):
            pred = predict_phi_lattice(features)
            correction = alpha * features[best_feature]
            pred_corrected = np.clip(pred + correction, 0, 1)
            total_mae += np.mean(np.abs(pred_corrected - depth))
        
        avg_mae = total_mae / len(train_data)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_alpha = alpha
    
    print(f"\n  Optimal α = {best_alpha:.3f}")
    print(f"  Corrected Train MAE: {best_mae:.4f}")
    
    # Step 3: Test
    print("\n" + "=" * 60)
    print("Step 3: Testing")
    print("=" * 60)
    
    test_errors_base = []
    test_errors_corrected = []
    
    for features, depth in test_data:
        pred = predict_phi_lattice(features)
        test_errors_base.append(np.mean(np.abs(pred - depth)))
        
        pred_corrected = np.clip(pred + best_alpha * features[best_feature], 0, 1)
        test_errors_corrected.append(np.mean(np.abs(pred_corrected - depth)))
    
    print(f"\n  φ-Lattice Test MAE:     {np.mean(test_errors_base):.4f}")
    print(f"  + Correction Test MAE:  {np.mean(test_errors_corrected):.4f}")
    
    improvement = np.mean(test_errors_base) - np.mean(test_errors_corrected)
    print(f"\n  Improvement: {improvement:.4f} ({100*improvement/np.mean(test_errors_base):.1f}%)")
    
    # Compare to previous best
    print("\n" + "=" * 60)
    print("COMPARISON TO PREVIOUS BEST")
    print("=" * 60)
    print(f"\n  Previous best (phase holographic): ~0.199")
    print(f"  φ-Lattice + Correction:            {np.mean(test_errors_corrected):.4f}")
    
    diff = np.mean(test_errors_corrected) - 0.199
    if diff < 0:
        print(f"\n  ✓ NEW BEST! Improvement: {-diff:.4f}")
    else:
        print(f"\n  Still {diff:.4f} behind previous best")
    
    return phi_weights, best_feature, best_alpha


if __name__ == "__main__":
    # Skip the slower experiments, run the key one
    print("\n" + "="*70)
    print("φ-LATTICE DEPTH ESTIMATION EXPERIMENTS")
    print("="*70)
    
    print("\n" + "="*70)
    print("EXPERIMENT: φ-Lattice + Corrections (Main)")
    print("="*70)
    weights, correction_feature, alpha = run_phi_lattice_with_corrections(n_train=50, n_test=10)
