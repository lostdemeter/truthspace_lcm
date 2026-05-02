#!/usr/bin/env python3
"""
Experiment: Dynamic Dimension Discovery with Back-Propagation

The key insight: If there's structure in the residual error, that's a MISSING DIMENSION.

Process:
1. Predict with current dimensions
2. Compute residual (true - predicted)
3. Analyze residual for patterns (via SVD/PCA)
4. If pattern is strong, extract it as a new dimension
5. Back-propagate: re-evaluate ALL previous images with new dimension
6. Re-learn weights with expanded dimension set
7. Repeat until residuals are structureless (just noise)

This is the self-assembly loop applied to dimension discovery:
INGEST → DETECT (patterns in residual) → DISCOVER (new dimension) → 
REBALANCE (re-learn weights) → VERIFY (check improvement)

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


# =============================================================================
# DIMENSION REPRESENTATION
# =============================================================================

@dataclass
class Dimension:
    """A depth dimension - can be predefined or discovered."""
    name: str
    weight: float
    extractor: Callable[[np.ndarray], np.ndarray]
    discovered: bool = False  # True if this was discovered from residuals
    discovery_iteration: int = 0


@dataclass 
class DiscoveredPattern:
    """A pattern discovered from residual analysis."""
    pattern: np.ndarray  # The pattern template
    strength: float  # How strong/consistent this pattern is
    explained_variance: float  # How much of the residual it explains


# =============================================================================
# PREDEFINED DIMENSION EXTRACTORS
# =============================================================================

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def extract_vertical_gradient(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    gradient = np.linspace(0, 1, h).reshape(-1, 1)
    return np.tile(gradient, (1, w))


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
# DYNAMIC DIMENSION DISCOVERY
# =============================================================================

class DynamicDepthModel:
    """
    Depth model with dynamic dimension discovery.
    
    Discovers new dimensions from residual patterns and back-propagates
    them to improve predictions on all images.
    """
    
    def __init__(self):
        # Start with minimal predefined dimensions
        self.dimensions: List[Dimension] = [
            Dimension("vertical_gradient", PHI**2, extract_vertical_gradient),
            Dimension("edges", PHI**1, extract_edges),
            Dimension("frequency", PHI**0, extract_frequency),
            Dimension("saliency", PHI**(-1), extract_saliency),
        ]
        
        # Weight adjustments (learned)
        self.weight_adjustments: Dict[str, float] = {d.name: 1.0 for d in self.dimensions}
        
        # Training data - kept for back-propagation
        self.training_data: List[Tuple[np.ndarray, np.ndarray, str]] = []  # (rgb, depth, id)
        
        # Discovered patterns (templates for new dimensions)
        self.discovered_patterns: List[DiscoveredPattern] = []
        
        # History for tracking improvement
        self.mae_history: List[float] = []
        self.dimension_history: List[int] = []
    
    def add_training_image(self, rgb: np.ndarray, depth: np.ndarray, image_id: str):
        """Add a training image."""
        self.training_data.append((rgb, depth, image_id))
    
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return image.copy()
    
    def _resize_to_match(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize array to match target shape."""
        if arr.shape == target_shape:
            return arr
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0
    
    def extract_dimensions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all dimensions from an image."""
        gray = _normalize(self._to_gray(image))
        return {d.name: d.extractor(gray) for d in self.dimensions}
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth from image."""
        dims = self.extract_dimensions(image)
        
        total_weight = 0
        depth = None
        
        for dim in self.dimensions:
            w = dim.weight * self.weight_adjustments.get(dim.name, 1.0)
            if w <= 0:
                continue
            total_weight += w
            
            dim_val = dims[dim.name]
            if depth is None:
                depth = w * dim_val
            else:
                # Resize if needed
                if dim_val.shape != depth.shape:
                    dim_val = self._resize_to_match(dim_val, depth.shape)
                depth = depth + w * dim_val
        
        if total_weight > 0:
            depth = depth / total_weight
        else:
            depth = np.zeros_like(list(dims.values())[0])
        
        return _normalize(gaussian_filter(depth, sigma=2.0))
    
    def compute_residuals(self) -> List[Tuple[np.ndarray, str]]:
        """Compute residuals for all training images."""
        residuals = []
        
        for rgb, true_depth, image_id in self.training_data:
            pred = self.predict(rgb)
            
            # Resize true depth to match prediction
            true_resized = self._resize_to_match(true_depth, pred.shape)
            
            residual = true_resized - pred
            residuals.append((residual, image_id))
        
        return residuals
    
    def analyze_residuals(self, residuals: List[Tuple[np.ndarray, str]], 
                          min_variance_explained: float = 0.1) -> Optional[Tuple[DiscoveredPattern, str]]:
        """
        Analyze residuals to find patterns that can be learned.
        
        Two approaches:
        1. Find input features that correlate with residuals (input-dependent)
        2. Learn a per-image correction coefficient (how much to adjust)
        
        Key insight: The residual pattern might be MULTIPLICATIVE with existing features.
        E.g., "edges predict depth, but we need MORE edge contribution in bright areas"
        """
        if len(residuals) < 5:
            return None
        
        target_shape = residuals[0][0].shape
        
        # Expanded candidate features including INTERACTIONS
        # Interactions capture "edges matter MORE in bright areas" etc.
        candidate_features = [
            'luminance',
            'inv_luminance',
            'local_variance',
            'horizontal_gradient',
            'radial_distance',
            'inv_radial',
            'diagonal',
            # Interactions (multiplicative)
            'luminance_x_edges',  # edges weighted by brightness
            'vertical_x_luminance',  # vertical gradient weighted by brightness
            'edges_x_variance',  # edges weighted by texture
        ]
        
        best_correlation = 0
        best_feature_name = None
        
        for feature_name in candidate_features:
            correlations = []
            
            for i, (residual, _) in enumerate(residuals):
                if residual.shape != target_shape:
                    residual = self._resize_to_match(residual, target_shape)
                
                rgb, depth, _ = self.training_data[i]
                gray = _normalize(self._to_gray(rgb))
                gray = self._resize_to_match(gray, target_shape)
                h, w = gray.shape
                
                # Compute feature
                if feature_name == 'luminance':
                    feature = gray
                elif feature_name == 'inv_luminance':
                    feature = 1.0 - gray
                elif feature_name == 'local_variance':
                    from scipy.ndimage import uniform_filter
                    local_mean = uniform_filter(gray, size=15)
                    local_sq_mean = uniform_filter(gray**2, size=15)
                    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
                    feature = _normalize(np.sqrt(local_var))
                elif feature_name == 'horizontal_gradient':
                    feature = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
                elif feature_name == 'radial_distance':
                    y, x = np.ogrid[:h, :w]
                    feature = _normalize(np.sqrt((x - w/2)**2 + (y - h/2)**2))
                elif feature_name == 'inv_radial':
                    y, x = np.ogrid[:h, :w]
                    feature = 1.0 - _normalize(np.sqrt((x - w/2)**2 + (y - h/2)**2))
                elif feature_name == 'quadrant_tl':
                    feature = np.zeros((h, w))
                    feature[:h//2, :w//2] = 1.0
                    feature = gaussian_filter(feature, sigma=h//8)
                elif feature_name == 'quadrant_br':
                    feature = np.zeros((h, w))
                    feature[h//2:, w//2:] = 1.0
                    feature = gaussian_filter(feature, sigma=h//8)
                elif feature_name == 'diagonal':
                    y, x = np.ogrid[:h, :w]
                    feature = _normalize((x / w + y / h) / 2)
                elif feature_name == 'luminance_x_edges':
                    edges = _normalize(np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2))
                    feature = _normalize(gray * edges)
                elif feature_name == 'vertical_x_luminance':
                    vert = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
                    feature = _normalize(vert * gray)
                elif feature_name == 'edges_x_variance':
                    from scipy.ndimage import uniform_filter
                    edges = _normalize(np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2))
                    local_mean = uniform_filter(gray, size=15)
                    local_sq_mean = uniform_filter(gray**2, size=15)
                    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
                    variance = _normalize(np.sqrt(local_var))
                    feature = _normalize(edges * variance)
                else:
                    continue
                
                # Correlation
                corr = np.corrcoef(feature.flatten(), residual.flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)  # Keep sign!
            
            # Use absolute mean but track sign
            if correlations:
                avg_corr = np.mean(correlations)
                if abs(avg_corr) > abs(best_correlation):
                    best_correlation = avg_corr
                    best_feature_name = feature_name
        
        print(f"  Best correlated feature: {best_feature_name} (r={best_correlation:.3f})")
        
        if abs(best_correlation) < 0.1:
            print(f"  No strong input-residual correlation found")
            return None
        
        # Return with sign info (negative correlation means we need to SUBTRACT)
        return DiscoveredPattern(
            pattern=np.array([[best_correlation]]),
            strength=abs(best_correlation),
            explained_variance=best_correlation**2
        ), (best_feature_name, best_correlation > 0)
    
    def create_dimension_from_feature(self, feature_name: str, iteration: int, 
                                       is_positive: bool = True) -> Dimension:
        """
        Create a new dimension from a discovered feature type.
        
        The feature_name tells us which input feature correlates with residuals.
        is_positive tells us if the correlation is positive (add) or negative (subtract).
        """
        from scipy.ndimage import uniform_filter
        
        def make_extractor(fname, positive):
            def extractor(gray: np.ndarray) -> np.ndarray:
                h, w = gray.shape
                
                if fname == 'luminance':
                    feature = gray
                elif fname == 'inv_luminance':
                    feature = 1.0 - gray
                elif fname == 'local_variance':
                    local_mean = uniform_filter(gray, size=15)
                    local_sq_mean = uniform_filter(gray**2, size=15)
                    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
                    feature = _normalize(np.sqrt(local_var))
                elif fname == 'horizontal_gradient':
                    feature = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
                elif fname == 'radial_distance':
                    y, x = np.ogrid[:h, :w]
                    feature = _normalize(np.sqrt((x - w/2)**2 + (y - h/2)**2))
                elif fname == 'inv_radial':
                    y, x = np.ogrid[:h, :w]
                    feature = 1.0 - _normalize(np.sqrt((x - w/2)**2 + (y - h/2)**2))
                elif fname == 'quadrant_tl':
                    feature = np.zeros((h, w))
                    feature[:h//2, :w//2] = 1.0
                    feature = gaussian_filter(feature, sigma=max(1, h//8))
                elif fname == 'quadrant_br':
                    feature = np.zeros((h, w))
                    feature[h//2:, w//2:] = 1.0
                    feature = gaussian_filter(feature, sigma=max(1, h//8))
                elif fname == 'diagonal':
                    y, x = np.ogrid[:h, :w]
                    feature = _normalize((x / w + y / h) / 2)
                elif fname == 'luminance_x_edges':
                    edges = _normalize(np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2))
                    feature = _normalize(gray * edges)
                elif fname == 'vertical_x_luminance':
                    vert = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
                    feature = _normalize(vert * gray)
                elif fname == 'edges_x_variance':
                    edges = _normalize(np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2))
                    local_mean = uniform_filter(gray, size=15)
                    local_sq_mean = uniform_filter(gray**2, size=15)
                    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
                    variance = _normalize(np.sqrt(local_var))
                    feature = _normalize(edges * variance)
                else:
                    feature = np.zeros_like(gray)
                
                # If negative correlation, invert the feature
                if not positive:
                    feature = 1.0 - feature
                
                return feature
            return extractor
        
        dim_name = f"discovered_{feature_name}_{iteration}"
        
        return Dimension(
            name=dim_name,
            weight=PHI**1,
            extractor=make_extractor(feature_name, is_positive),
            discovered=True,
            discovery_iteration=iteration
        )
    
    def learn_weights(self, n_iterations: int = 5):
        """Learn optimal weights for all dimensions."""
        if not self.training_data:
            return
        
        # Pre-extract dimensions for all images
        all_dims = []
        all_targets = []
        
        for rgb, depth, _ in self.training_data:
            dims = self.extract_dimensions(rgb)
            target_shape = list(dims.values())[0].shape
            depth_resized = self._resize_to_match(depth, target_shape)
            all_dims.append(dims)
            all_targets.append(depth_resized)
        
        best_mae = float('inf')
        best_adjustments = self.weight_adjustments.copy()
        
        search_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        
        for iteration in range(n_iterations):
            for dim_name in list(self.weight_adjustments.keys()):
                for adj in search_values:
                    self.weight_adjustments[dim_name] = adj
                    
                    total_mae = 0
                    for dims, target in zip(all_dims, all_targets):
                        # Inline prediction to avoid re-extracting
                        total_weight = 0
                        depth = None
                        for dim in self.dimensions:
                            w = dim.weight * self.weight_adjustments.get(dim.name, 1.0)
                            if w <= 0:
                                continue
                            total_weight += w
                            dim_val = dims.get(dim.name)
                            if dim_val is None:
                                continue
                            if depth is None:
                                depth = w * dim_val
                            else:
                                if dim_val.shape != depth.shape:
                                    dim_val = self._resize_to_match(dim_val, depth.shape)
                                depth = depth + w * dim_val
                        
                        if total_weight > 0 and depth is not None:
                            depth = depth / total_weight
                            depth = gaussian_filter(depth, sigma=2.0)
                            total_mae += np.mean(np.abs(_normalize(depth) - target))
                    
                    avg_mae = total_mae / len(all_targets) if all_targets else float('inf')
                    
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_adjustments = self.weight_adjustments.copy()
                
                self.weight_adjustments = best_adjustments.copy()
        
        return best_mae
    
    def compute_mae(self) -> float:
        """Compute current MAE on training data."""
        if not self.training_data:
            return float('inf')
        
        total_mae = 0
        for rgb, depth, _ in self.training_data:
            pred = self.predict(rgb)
            true_resized = self._resize_to_match(depth, pred.shape)
            total_mae += np.mean(np.abs(pred - true_resized))
        
        return total_mae / len(self.training_data)
    
    def discover_and_backpropagate(self, max_iterations: int = 5, 
                                    min_improvement: float = 0.01):
        """
        Main loop: discover dimensions and back-propagate.
        
        1. Compute residuals
        2. Find patterns in residuals
        3. Create new dimension from pattern
        4. Re-learn all weights (back-propagation)
        5. Check improvement
        6. Repeat until no improvement
        """
        print("=" * 60)
        print("DYNAMIC DIMENSION DISCOVERY")
        print("=" * 60)
        
        # Initial weight learning
        print("\nInitial weight learning...")
        initial_mae = self.learn_weights()
        print(f"  Initial MAE: {initial_mae:.4f}")
        
        self.mae_history.append(initial_mae)
        self.dimension_history.append(len(self.dimensions))
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Compute residuals
            print("Computing residuals...")
            residuals = self.compute_residuals()
            
            # Analyze for patterns
            print("Analyzing residual patterns...")
            result = self.analyze_residuals(residuals)
            
            if result is None:
                print("No significant pattern found. Stopping.")
                break
            
            pattern, (feature_name, is_positive) = result
            
            # Create new dimension from the discovered feature
            print(f"Creating new dimension: {feature_name} (positive={is_positive})")
            new_dim = self.create_dimension_from_feature(feature_name, iteration, is_positive)
            self.dimensions.append(new_dim)
            self.weight_adjustments[new_dim.name] = 1.0
            
            print(f"  Added dimension: {new_dim.name}")
            print(f"  Total dimensions: {len(self.dimensions)}")
            
            # Back-propagate: re-learn ALL weights with new dimension
            print("Back-propagating (re-learning weights)...")
            new_mae = self.learn_weights()
            
            improvement = self.mae_history[-1] - new_mae
            print(f"  New MAE: {new_mae:.4f} (improvement: {improvement:.4f})")
            
            self.mae_history.append(new_mae)
            self.dimension_history.append(len(self.dimensions))
            
            # Check if improvement is significant
            if improvement < min_improvement:
                print(f"Improvement below threshold ({min_improvement}). Stopping.")
                # Remove the dimension that didn't help
                self.dimensions.pop()
                del self.weight_adjustments[new_dim.name]
                break
        
        print("\n" + "=" * 60)
        print("DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"Final dimensions: {len(self.dimensions)}")
        print(f"Final MAE: {self.mae_history[-1]:.4f}")
        print(f"Total improvement: {self.mae_history[0] - self.mae_history[-1]:.4f}")
        
        print("\nDimension weights:")
        for dim in self.dimensions:
            w = dim.weight * self.weight_adjustments.get(dim.name, 1.0)
            discovered = " (discovered)" if dim.discovered else ""
            print(f"  {dim.name}: {w:.3f}{discovered}")


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_dynamic_discovery_experiment(n_images: int = 50):
    """Run the dynamic dimension discovery experiment."""
    print("=" * 70)
    print("EXPERIMENT: Dynamic Dimension Discovery with Back-Propagation")
    print("=" * 70)
    print()
    print("Process:")
    print("1. Start with minimal dimensions")
    print("2. Predict → compute residuals → find patterns")
    print("3. Create new dimension from pattern")
    print("4. Back-propagate: re-learn ALL weights")
    print("5. Repeat until no improvement")
    print()
    
    # Get available images
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Initialize model
    model = DynamicDepthModel()
    
    # Load training data
    print(f"Loading {n_images} images...")
    for img_id in available_ids[:n_images]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        model.add_training_image(rgb, depth, img_id)
    
    print(f"  Loaded {len(model.training_data)} images")
    
    # Run discovery
    model.discover_and_backpropagate(max_iterations=5, min_improvement=0.005)
    
    return model


if __name__ == "__main__":
    model = run_dynamic_discovery_experiment(n_images=50)
