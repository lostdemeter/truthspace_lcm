#!/usr/bin/env python3
"""
Experiment: φ-Holographic Depth Estimation

Adapts holographic depth principles to the φ-based self-assembly model.

Key insight: Depth cues are DIMENSIONS in φ-space, not statistical relationships.
Each cue (luminance, edges, frequency, saliency) is a dimension that can be
φ-weighted and composed.

The transformation RGB → Depth becomes traversal through these dimensions,
with weights emerging from the φ-geometry.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


# =============================================================================
# DEPTH DIMENSIONS - Each is a geometric dimension in φ-space
# =============================================================================

@dataclass
class DepthDimension:
    """A depth cue as a geometric dimension."""
    name: str
    weight: float  # φ-based weight
    extractor: callable
    
    def extract(self, gray: np.ndarray) -> np.ndarray:
        return self.extractor(gray)


def extract_luminance(gray: np.ndarray) -> np.ndarray:
    """Luminance dimension: brighter = closer."""
    return gray.copy()


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Edge dimension: sharp edges = in focus = closer."""
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    return _normalize(edge_strength)


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    """Frequency dimension: high frequency = fine detail = closer."""
    F = fft2(gray)
    F_shifted = fftshift(F)
    
    h, w = gray.shape
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    
    # High-pass filter
    H = np.sqrt(U**2 + V**2) / np.sqrt((w//2)**2 + (h//2)**2)
    F_filtered = F_shifted * H
    
    filtered = np.abs(ifft2(ifftshift(F_filtered)))
    return _normalize(filtered)


def extract_saliency(gray: np.ndarray) -> np.ndarray:
    """Saliency dimension: spectral residual = perceptually important."""
    F = fft2(gray)
    amplitude = np.abs(F)
    log_amplitude = np.log(amplitude + 1e-10)
    
    # Spectral residual
    log_amplitude_smoothed = gaussian_filter(log_amplitude, sigma=3.0)
    residual = log_amplitude - log_amplitude_smoothed
    
    phase = np.angle(F)
    F_residual = np.exp(residual + 1j * phase)
    
    saliency = np.abs(ifft2(F_residual)) ** 2
    saliency = gaussian_filter(saliency, sigma=5.0)
    return _normalize(saliency)


def extract_center_bias(gray: np.ndarray) -> np.ndarray:
    """Center bias dimension: compositional prior (subjects centered)."""
    h, w = gray.shape
    x_c, y_c = w / 2, h / 2
    sigma_c = min(h, w) / 3.0
    
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    return np.exp(-((X - x_c)**2 + (Y - y_c)**2) / (2 * sigma_c**2))


def extract_vertical_gradient(gray: np.ndarray) -> np.ndarray:
    """Vertical gradient dimension: top = far, bottom = near (ground plane)."""
    h, w = gray.shape
    gradient = np.linspace(0, 1, h).reshape(-1, 1)
    return np.tile(gradient, (1, w))


def extract_local_blur(gray: np.ndarray) -> np.ndarray:
    """
    Local blur dimension: blurry regions = far (depth of field).
    
    Measures local sharpness - sharp regions are in focus (closer).
    We INVERT this so blurry = high value = far.
    """
    # Laplacian measures sharpness (second derivative)
    from scipy.ndimage import laplace
    laplacian = np.abs(laplace(gray))
    
    # Local variance of Laplacian (more robust sharpness measure)
    window_size = 15
    
    def local_var(values):
        return np.var(values)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sharpness = generic_filter(laplacian, local_var, size=window_size)
    
    # Normalize and INVERT: blurry (low sharpness) = high depth
    sharpness = _normalize(sharpness)
    blur = 1.0 - sharpness  # Invert: blur = far
    
    return blur


def extract_local_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Local contrast dimension: low contrast = far (atmospheric perspective).
    
    Distant objects have reduced contrast due to haze/atmosphere.
    We INVERT this so low contrast = high value = far.
    """
    window_size = 21
    
    def local_range(values):
        return np.max(values) - np.min(values)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contrast = generic_filter(gray, local_range, size=window_size)
    
    # Normalize and INVERT: low contrast = high depth
    contrast = _normalize(contrast)
    low_contrast = 1.0 - contrast  # Invert: low contrast = far
    
    return low_contrast


def extract_texture_density(gray: np.ndarray) -> np.ndarray:
    """
    Texture density dimension: sparse texture = far.
    
    Distant objects have less visible texture detail.
    Uses local entropy as texture measure.
    """
    from scipy.ndimage import uniform_filter
    
    # Local entropy approximation via local variance
    window_size = 15
    
    local_mean = uniform_filter(gray, size=window_size)
    local_sq_mean = uniform_filter(gray**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    
    # High variance = high texture = near
    texture = _normalize(np.sqrt(local_var))
    sparse_texture = 1.0 - texture  # Invert: sparse = far
    
    return sparse_texture


def extract_color_saturation(image: np.ndarray) -> np.ndarray:
    """
    Color saturation dimension: desaturated = far (atmospheric perspective).
    
    Distant objects appear more desaturated/gray due to atmosphere.
    Requires RGB input.
    """
    if image.ndim != 3:
        # Can't compute saturation from grayscale
        return np.zeros(image.shape[:2])
    
    # Simple saturation: (max - min) / max
    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)
    
    saturation = np.zeros_like(max_rgb)
    mask = max_rgb > 0
    saturation[mask] = (max_rgb[mask] - min_rgb[mask]) / max_rgb[mask]
    
    # Invert: low saturation = far
    desaturated = 1.0 - _normalize(saturation)
    
    return desaturated


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr


# =============================================================================
# φ-HOLOGRAPHIC DEPTH ESTIMATOR
# =============================================================================

class PhiHolographicDepth:
    """
    Depth estimation using φ-weighted holographic dimensions.
    
    Each depth cue is a dimension. The final depth is a φ-weighted
    composition of these dimensions, with weights that can be learned
    or set based on the golden ratio hierarchy.
    """
    
    def __init__(self, should_learn_weights: bool = True):
        self.should_learn_weights = should_learn_weights
        
        # Initialize dimensions with φ-based weights
        # DETAIL dimensions (capture texture/edges - what we had before)
        # DISTANCE dimensions (capture actual depth - new)
        self.dimensions = [
            # Detail dimensions (lower weight now)
            DepthDimension("luminance", PHI**0, extract_luminance),
            DepthDimension("edges", PHI**(-1), extract_edges),
            DepthDimension("frequency", PHI**(-2), extract_frequency),
            DepthDimension("saliency", PHI**(-2), extract_saliency),
            DepthDimension("center_bias", PHI**(-1), extract_center_bias),
            # Distance dimensions (higher weight - these capture actual depth)
            DepthDimension("vertical_gradient", PHI**2, extract_vertical_gradient),
            DepthDimension("local_blur", PHI**2, extract_local_blur),
            DepthDimension("local_contrast", PHI**1, extract_local_contrast),
            DepthDimension("texture_density", PHI**1, extract_texture_density),
        ]
        
        # Color saturation handled separately (needs RGB)
        self.use_color_saturation = True
        
        # Learned weight adjustments (initialized to 1.0)
        self.weight_adjustments = {d.name: 1.0 for d in self.dimensions}
        
        # Training data for weight learning
        self.training_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def extract_dimensions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all depth dimensions from an image."""
        # Convert to grayscale
        if image.ndim == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image.copy()
        
        gray = _normalize(gray)
        
        dims = {d.name: d.extract(gray) for d in self.dimensions}
        
        # Add color saturation if RGB available
        if self.use_color_saturation and image.ndim == 3:
            dims["color_saturation"] = extract_color_saturation(image)
            if "color_saturation" not in self.weight_adjustments:
                self.weight_adjustments["color_saturation"] = 1.0
        
        return dims
    
    def compose_depth(self, dimensions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compose depth from dimensions using φ-weighted fusion.
        
        This is the key geometric operation: traversal through dimension space.
        """
        total_weight = 0
        depth = None
        
        for dim in self.dimensions:
            if dim.name not in dimensions:
                continue
            w = dim.weight * self.weight_adjustments[dim.name]
            total_weight += w
            
            if depth is None:
                depth = w * dimensions[dim.name]
            else:
                depth = depth + w * dimensions[dim.name]
        
        # Add color saturation if present
        if "color_saturation" in dimensions:
            sat_weight = PHI**1 * self.weight_adjustments.get("color_saturation", 1.0)
            total_weight += sat_weight
            if depth is None:
                depth = sat_weight * dimensions["color_saturation"]
            else:
                depth = depth + sat_weight * dimensions["color_saturation"]
        
        depth = depth / total_weight
        
        # Final smoothing
        depth = gaussian_filter(depth, sigma=2.0)
        
        return _normalize(depth)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth from RGB image."""
        dimensions = self.extract_dimensions(image)
        return self.compose_depth(dimensions)
    
    def add_training_pair(self, rgb: np.ndarray, depth: np.ndarray):
        """Add a training pair for weight learning."""
        self.training_pairs.append((rgb, depth))
    
    def learn_weights(self, n_iterations: int = 10):
        """
        Learn optimal weight adjustments from training data.
        
        Uses gradient-free optimization: adjust weights to minimize MAE.
        This is still geometric - we're finding the optimal position
        in weight-space through traversal.
        """
        if not self.training_pairs:
            return
        
        print(f"Learning weights from {len(self.training_pairs)} pairs...")
        
        # Extract dimensions for all training images
        all_dimensions = []
        all_targets = []
        
        for rgb, depth in self.training_pairs:
            dims = self.extract_dimensions(rgb)
            # Resize depth to match
            h, w = list(dims.values())[0].shape
            depth_resized = np.array(Image.fromarray(
                (depth * 255).astype(np.uint8)
            ).resize((w, h))).astype(np.float32) / 255.0
            
            all_dimensions.append(dims)
            all_targets.append(depth_resized)
        
        # Optimize weights using coordinate descent with finer grid
        best_mae = float('inf')
        best_adjustments = self.weight_adjustments.copy()
        
        # More aggressive search range
        search_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        
        for iteration in range(n_iterations):
            for dim_name in list(self.weight_adjustments.keys()):
                # Try different adjustment values
                for adj in search_values:
                    self.weight_adjustments[dim_name] = adj
                    
                    # Compute MAE
                    total_mae = 0
                    for dims, target in zip(all_dimensions, all_targets):
                        pred = self.compose_depth(dims)
                        total_mae += np.mean(np.abs(pred - target))
                    
                    avg_mae = total_mae / len(all_targets)
                    
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_adjustments = self.weight_adjustments.copy()
                
                # Restore best
                self.weight_adjustments = best_adjustments.copy()
            
            print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
        
        print(f"Final weights:")
        for dim in self.dimensions:
            final_w = dim.weight * self.weight_adjustments[dim.name]
            print(f"  {dim.name}: {final_w:.3f}")
    
    def get_dimension_contributions(self, image: np.ndarray) -> Dict[str, float]:
        """Get the contribution of each dimension to the final depth."""
        dimensions = self.extract_dimensions(image)
        
        contributions = {}
        total_weight = sum(d.weight * self.weight_adjustments[d.name] for d in self.dimensions)
        
        for dim in self.dimensions:
            w = dim.weight * self.weight_adjustments[dim.name]
            contributions[dim.name] = w / total_weight
        
        return contributions


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_phi_holographic_experiment(n_train: int = 50, n_test: int = 10):
    """Run the φ-holographic depth experiment."""
    print("=" * 70)
    print("EXPERIMENT: φ-Holographic Depth Estimation")
    print("=" * 70)
    print()
    print("Key insight: Depth cues are DIMENSIONS in φ-space.")
    print("Each cue (luminance, edges, frequency, saliency) is a dimension")
    print("that can be φ-weighted and composed geometrically.")
    print()
    
    # Get available images
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Initialize
    estimator = PhiHolographicDepth(should_learn_weights=True)
    
    # Load training data
    print(f"Loading {n_train} training images...")
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        estimator.add_training_pair(rgb, depth)
    
    print(f"  Loaded {len(estimator.training_pairs)} pairs")
    
    # Learn weights
    print()
    print("=" * 60)
    print("LEARNING φ-WEIGHTS")
    print("=" * 60)
    estimator.learn_weights(n_iterations=5)
    
    # Test
    print()
    print("=" * 60)
    print("TESTING")
    print("=" * 60)
    
    test_ids = available_ids[n_train:n_train + n_test]
    errors = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        pred_depth = estimator.predict(rgb)
        
        # Resize true depth to match prediction
        h, w = pred_depth.shape
        true_resized = np.array(Image.fromarray(
            (true_depth * 255).astype(np.uint8)
        ).resize((w, h))).astype(np.float32) / 255.0
        
        mae = np.mean(np.abs(pred_depth - true_resized))
        errors.append(mae)
        print(f"  {img_id}.jpg: MAE = {mae:.4f}")
    
    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    mean_mae = np.mean(errors) if errors else 0
    print(f"  Mean Absolute Error: {mean_mae:.4f}")
    print()
    
    # Show dimension contributions
    print("Dimension contributions (φ-weighted):")
    sample_rgb = np.array(Image.open(COCO_VAL_PATH / f"{test_ids[0]}.jpg").convert("RGB")).astype(np.float32) / 255.0
    contributions = estimator.get_dimension_contributions(sample_rgb)
    for name, contrib in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"  {name}: {contrib:.1%}")
    
    print()
    if mean_mae < 0.15:
        print("✓ SUCCESS: φ-holographic depth works well!")
    elif mean_mae < 0.25:
        print("◐ PARTIAL: Reasonable depth structure captured")
    else:
        print("✗ LIMITED: Needs refinement")
    
    return estimator


if __name__ == "__main__":
    estimator = run_phi_holographic_experiment(n_train=50, n_test=10)
