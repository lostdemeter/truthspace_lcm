#!/usr/bin/env python3
"""
Experiment: Dimensional Upcasting for Depth Estimation

Inspired by Dimensional Downcasting for Riemann Zeta Zeros.

The key insight from downcasting:
- N_smooth(t_n) ≈ n - 0.5 is the SELECTION CRITERION
- Multiple candidates exist, but only one satisfies the criterion
- Refinement to machine precision follows

For depth UPCASTING:
- 2D image → 3D depth (upcast)
- Multiple depth candidates at each pixel
- Selection criterion: intersection of pivot constraints
- Refinement via Gaussian splatting

The analogy:
    Downcasting: ∞D → 1D via N_smooth selection
    Upcasting:   2D → 3D via pivot intersection selection

Key question: What is our "N_smooth ≈ n - 0.5" equivalent?

Hypothesis: The "smooth depth function" is the weighted combination
of pivot constraints. When all pivots AGREE, we have the correct depth.

D_smooth(x,y) = Σ w_i × D_i(x,y)

At the correct depth:
    D_smooth ≈ D_true when all D_i agree (low variance)

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import brentq, minimize_scalar
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
# DEPTH PREDICTORS (analogous to RamanujanPredictor)
# =============================================================================

class VerticalPredictor:
    """
    Fast O(1) predictor using vertical position.
    
    Analogous to RamanujanPredictor for zeta zeros.
    
    Accuracy: ~0.182 MAE (the "quantum barrier" for depth)
    """
    
    def __init__(self):
        # Learned from data: depth ≈ 0.6 * y + 0.1
        self.slope = 0.599
        self.intercept = 0.095
    
    def predict(self, h: int, w: int) -> np.ndarray:
        """Predict depth from vertical position alone."""
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        return self.slope * y_coords + self.intercept


class GeometricPredictor:
    """
    Geometric predictor using multiple constraints.
    
    Analogous to GeometricPredictor for zeta zeros.
    """
    
    def __init__(self):
        self.vertical = VerticalPredictor()
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth using geometric constraints."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Vertical base
        d_vertical = self.vertical.predict(h, w)
        
        # Edge correction (edges often indicate depth discontinuities)
        edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
        d_edge = _normalize(edges) * 0.1
        
        return d_vertical + d_edge


# =============================================================================
# PIVOT CONSTRAINTS (analogous to N_smooth)
# =============================================================================

def compute_pivot_depths(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute depth estimates from each pivot.
    
    Each pivot gives a different "view" of the depth.
    The true depth is where they all agree.
    """
    if image.ndim == 3:
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    pivots = {}
    
    # Pivot 1: Vertical (camera orientation)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    pivots['vertical'] = 0.599 * y_coords + 0.095
    
    # Pivot 2: Shading (light direction)
    grad_y = sobel(gray, axis=0)
    pivots['shading'] = _normalize(-grad_y) * 0.5 + 0.25  # Darker at top = farther
    
    # Pivot 3: Texture (surface orientation)
    local_var = gaussian_filter(gray**2, sigma=3) - gaussian_filter(gray, sigma=3)**2
    pivots['texture'] = _normalize(np.sqrt(np.maximum(local_var, 0))) * 0.3 + 0.35
    
    # Pivot 4: Color (chromatic depth)
    if image.ndim == 3:
        rb_diff = image[:,:,0] - image[:,:,2]
        pivots['color'] = _normalize(rb_diff) * 0.2 + 0.4
    else:
        pivots['color'] = np.ones_like(gray) * 0.5
    
    return pivots


def compute_D_smooth(pivots: Dict[str, np.ndarray], 
                     weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Compute the smooth depth function (analogous to N_smooth).
    
    D_smooth = Σ w_i × D_i
    
    This is our selection criterion. At the correct depth,
    D_smooth should equal the true depth.
    """
    if weights is None:
        # Default weights based on confidence
        weights = {
            'vertical': 0.6,   # Most reliable
            'shading': 0.2,
            'texture': 0.1,
            'color': 0.1
        }
    
    D_smooth = np.zeros_like(list(pivots.values())[0])
    total_weight = 0
    
    for name, depth in pivots.items():
        w = weights.get(name, 0.1)
        D_smooth += w * depth
        total_weight += w
    
    return D_smooth / total_weight


def compute_pivot_variance(pivots: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute variance across pivots.
    
    Low variance = pivots agree = high confidence
    High variance = pivots disagree = need refinement
    
    This is analogous to how N_smooth ≈ n - 0.5 identifies the correct zero.
    """
    depths = np.stack(list(pivots.values()), axis=0)
    return np.var(depths, axis=0)


# =============================================================================
# DIMENSIONAL UPCASTER
# =============================================================================

class DimensionalUpcaster:
    """
    Dimensional Upcasting solver for depth estimation.
    
    Analogous to DimensionalDowncaster for zeta zeros.
    
    Algorithm:
        1. Initial guess from VerticalPredictor (O(1))
        2. Compute pivot depths (multiple candidates)
        3. Select using D_smooth (where pivots agree)
        4. Refine using Gaussian splatting
    
    The key insight: D_smooth identifies correct depth like
    N_smooth identifies correct zero.
    """
    
    def __init__(self):
        self.predictor = VerticalPredictor()
        self.stats = {'pixels_solved': 0, 'refinements': 0}
    
    def _find_bracket(self, pivots: Dict[str, np.ndarray], 
                      x: int, y: int) -> Tuple[float, float]:
        """
        Find a bracket [a, b] containing the correct depth at (x, y).
        
        Uses pivot variance to identify regions of agreement.
        """
        # Get all pivot values at this pixel
        values = [pivots[name][y, x] for name in pivots]
        
        # Bracket is [min, max] of pivot values
        a = min(values)
        b = max(values)
        
        # Expand slightly for safety
        margin = 0.05
        return (max(0, a - margin), min(1, b + margin))
    
    def _D_smooth_at(self, pivots: Dict[str, np.ndarray], 
                     x: int, y: int, d: float) -> float:
        """
        Evaluate how well depth d matches the smooth function at (x, y).
        
        Returns the "error" - how far d is from the pivot consensus.
        """
        # Compute weighted distance from each pivot
        weights = {'vertical': 0.6, 'shading': 0.2, 'texture': 0.1, 'color': 0.1}
        
        error = 0
        for name, depth_map in pivots.items():
            w = weights.get(name, 0.1)
            error += w * (d - depth_map[y, x])**2
        
        return np.sqrt(error)
    
    def _refine_pixel(self, pivots: Dict[str, np.ndarray], 
                      x: int, y: int, tol: float = 0.01) -> float:
        """
        Refine depth at a single pixel using pivot consensus.
        
        Analogous to bisection + Brent's method for zeta zeros.
        """
        a, b = self._find_bracket(pivots, x, y)
        
        # Find depth that minimizes pivot disagreement
        def objective(d):
            return self._D_smooth_at(pivots, x, y, d)
        
        result = minimize_scalar(objective, bounds=(a, b), method='bounded')
        self.stats['refinements'] += 1
        
        return result.x
    
    def solve(self, image: np.ndarray) -> np.ndarray:
        """
        Solve for depth using dimensional upcasting.
        
        This is the main entry point.
        """
        h, w = image.shape[:2]
        
        # Step 1: Initial guess
        d_initial = self.predictor.predict(h, w)
        
        # Step 2: Compute pivot depths
        pivots = compute_pivot_depths(image)
        
        # Step 3: Compute D_smooth (weighted consensus)
        D_smooth = compute_D_smooth(pivots)
        
        # Step 4: Compute variance (confidence)
        variance = compute_pivot_variance(pivots)
        
        # Step 5: Refine high-variance regions
        # For efficiency, only refine where pivots disagree significantly
        high_var_mask = variance > np.percentile(variance, 90)
        
        depth = D_smooth.copy()
        
        # Refine high-variance pixels
        high_var_y, high_var_x = np.where(high_var_mask)
        for y, x in zip(high_var_y[:100], high_var_x[:100]):  # Limit for speed
            depth[y, x] = self._refine_pixel(pivots, x, y)
        
        # Step 6: Gaussian smoothing (splatting)
        depth = gaussian_filter(depth, sigma=2.0)
        
        self.stats['pixels_solved'] += h * w
        return _normalize(depth)
    
    def verify(self, image: np.ndarray, true_depth: np.ndarray) -> Dict:
        """
        Solve and verify against ground truth.
        """
        pred_depth = self.solve(image)
        mae = np.mean(np.abs(pred_depth - true_depth))
        
        # Compute pivot agreement
        pivots = compute_pivot_depths(image)
        variance = compute_pivot_variance(pivots)
        
        return {
            'mae': mae,
            'mean_variance': variance.mean(),
            'max_variance': variance.max(),
            'refinements': self.stats['refinements']
        }


# =============================================================================
# THE KEY INSIGHT: SELECTION CRITERION
# =============================================================================

def analyze_selection_criterion(image: np.ndarray, true_depth: np.ndarray):
    """
    Analyze what makes a good selection criterion for depth.
    
    In zeta zeros: N_smooth(t_n) ≈ n - 0.5
    
    For depth: What is the analogous relationship?
    
    Hypothesis: At correct depth, pivot variance is minimized.
    """
    pivots = compute_pivot_depths(image)
    D_smooth = compute_D_smooth(pivots)
    variance = compute_pivot_variance(pivots)
    
    # At each pixel, check if low variance correlates with correct depth
    error = np.abs(D_smooth - true_depth)
    
    # Correlation between variance and error
    # If negative: low variance → low error (good!)
    correlation = np.corrcoef(variance.flatten(), error.flatten())[0, 1]
    
    return {
        'variance_error_correlation': correlation,
        'mean_error_low_var': error[variance < np.median(variance)].mean(),
        'mean_error_high_var': error[variance >= np.median(variance)].mean(),
    }


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_upcasting_experiment(n_train: int = 20, n_test: int = 10):
    """
    Test dimensional upcasting for depth estimation.
    """
    print("=" * 70)
    print("EXPERIMENT: Dimensional Upcasting")
    print("=" * 70)
    print()
    print("Analogous to Dimensional Downcasting for Riemann Zeta Zeros:")
    print()
    print("  Downcasting: ∞D → 1D via N_smooth ≈ n - 0.5")
    print("  Upcasting:   2D → 3D via pivot consensus")
    print()
    print("Key insight: At correct depth, pivots AGREE (low variance)")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    upcaster = DimensionalUpcaster()
    
    # Analyze selection criterion
    print("=" * 60)
    print("Analyzing Selection Criterion")
    print("=" * 60)
    
    correlations = []
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        depth_small = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        analysis = analyze_selection_criterion(rgb_small, depth_small)
        correlations.append(analysis['variance_error_correlation'])
        
        if i < 3:
            print(f"\n  Image {i+1}: {img_id}")
            print(f"    Variance-Error Correlation: {analysis['variance_error_correlation']:.3f}")
            print(f"    Mean Error (low var):  {analysis['mean_error_low_var']:.4f}")
            print(f"    Mean Error (high var): {analysis['mean_error_high_var']:.4f}")
    
    avg_corr = np.mean(correlations)
    print(f"\n  Average Variance-Error Correlation: {avg_corr:.3f}")
    
    if avg_corr > 0:
        print("  → POSITIVE: High variance correlates with high error")
        print("  → Selection criterion: MINIMIZE variance")
    else:
        print("  → NEGATIVE: Low variance correlates with high error")
        print("  → Need different selection criterion")
    
    # Test
    print("\n" + "=" * 60)
    print("Testing Dimensional Upcasting")
    print("=" * 60)
    
    test_errors = []
    
    for i, img_id in enumerate(available_ids[n_train:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        true_depth_small = np.array(Image.fromarray((true_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        result = upcaster.verify(rgb_small, true_depth_small)
        test_errors.append(result['mae'])
        
        if i < 3:
            print(f"\n  Test {i+1}: {img_id}")
            print(f"    MAE: {result['mae']:.4f}")
            print(f"    Mean Variance: {result['mean_variance']:.4f}")
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Dimensional Upcasting Test MAE: {np.mean(test_errors):.4f}")
    print(f"  Self-Assembling Pivots:         0.212")
    print(f"  Vertical alone:                 0.182")
    print(f"  Statistical best:               0.199")
    
    # The key insight
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print()
    print("In Dimensional Downcasting:")
    print("  N_smooth(t_n) ≈ n - 0.5 identifies the correct zero")
    print()
    print("In Dimensional Upcasting:")
    print("  Low pivot variance identifies correct depth regions")
    print()
    print("The selection criterion is the KEY to both methods.")
    print("Without it, you have multiple candidates and no way to choose.")
    
    return upcaster


def analyze_pivot_reliability(n_images: int = 30):
    """
    Analyze which pivot is most reliable in which regions.
    
    The key insight from dimensional downcasting is that N_smooth ≈ n - 0.5
    is a SPECIFIC relationship, not just "agreement".
    
    For depth, we need to find: which pivot is reliable WHERE?
    """
    print("=" * 70)
    print("ANALYZING PIVOT RELIABILITY BY REGION")
    print("=" * 70)
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect errors by region for each pivot
    # Regions: top, middle, bottom (vertical)
    # Also: edges vs smooth areas
    
    pivot_errors = {
        'vertical': {'top': [], 'middle': [], 'bottom': [], 'edges': [], 'smooth': []},
        'shading': {'top': [], 'middle': [], 'bottom': [], 'edges': [], 'smooth': []},
        'texture': {'top': [], 'middle': [], 'bottom': [], 'edges': [], 'smooth': []},
        'color': {'top': [], 'middle': [], 'bottom': [], 'edges': [], 'smooth': []},
    }
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        true_depth_small = np.array(Image.fromarray((true_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute pivot depths
        pivots = compute_pivot_depths(rgb_small)
        
        # Define regions
        top_mask = np.zeros((new_h, new_w), dtype=bool)
        top_mask[:new_h//3, :] = True
        
        middle_mask = np.zeros((new_h, new_w), dtype=bool)
        middle_mask[new_h//3:2*new_h//3, :] = True
        
        bottom_mask = np.zeros((new_h, new_w), dtype=bool)
        bottom_mask[2*new_h//3:, :] = True
        
        # Edge detection
        gray = 0.299 * rgb_small[:,:,0] + 0.587 * rgb_small[:,:,1] + 0.114 * rgb_small[:,:,2]
        edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
        edge_mask = edges > np.percentile(edges, 70)
        smooth_mask = ~edge_mask
        
        # Compute errors by region for each pivot
        for pivot_name, pivot_depth in pivots.items():
            error = np.abs(pivot_depth - true_depth_small)
            
            pivot_errors[pivot_name]['top'].append(error[top_mask].mean())
            pivot_errors[pivot_name]['middle'].append(error[middle_mask].mean())
            pivot_errors[pivot_name]['bottom'].append(error[bottom_mask].mean())
            pivot_errors[pivot_name]['edges'].append(error[edge_mask].mean())
            pivot_errors[pivot_name]['smooth'].append(error[smooth_mask].mean())
    
    # Print results
    print("\nMean Absolute Error by Region:")
    print("-" * 70)
    print(f"{'Pivot':<12} {'Top':<10} {'Middle':<10} {'Bottom':<10} {'Edges':<10} {'Smooth':<10}")
    print("-" * 70)
    
    best_by_region = {}
    
    for pivot_name in pivot_errors:
        top_mae = np.mean(pivot_errors[pivot_name]['top'])
        mid_mae = np.mean(pivot_errors[pivot_name]['middle'])
        bot_mae = np.mean(pivot_errors[pivot_name]['bottom'])
        edge_mae = np.mean(pivot_errors[pivot_name]['edges'])
        smooth_mae = np.mean(pivot_errors[pivot_name]['smooth'])
        
        print(f"{pivot_name:<12} {top_mae:<10.4f} {mid_mae:<10.4f} {bot_mae:<10.4f} {edge_mae:<10.4f} {smooth_mae:<10.4f}")
    
    # Find best pivot for each region
    print("\n" + "-" * 70)
    print("Best Pivot by Region:")
    
    for region in ['top', 'middle', 'bottom', 'edges', 'smooth']:
        best_pivot = min(pivot_errors.keys(), 
                        key=lambda p: np.mean(pivot_errors[p][region]))
        best_mae = np.mean(pivot_errors[best_pivot][region])
        print(f"  {region:<10}: {best_pivot} (MAE: {best_mae:.4f})")
        best_by_region[region] = best_pivot
    
    return best_by_region, pivot_errors


def run_region_weighted_experiment(n_train: int = 30, n_test: int = 10):
    """
    Use region-specific pivot weighting based on reliability analysis.
    
    This is the "N_smooth ≈ n - 0.5" equivalent for depth:
    Use the BEST pivot for each region, not a uniform combination.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Region-Weighted Pivot Selection")
    print("=" * 70)
    print()
    print("Instead of uniform weighting, use the BEST pivot for each region.")
    print("This is analogous to N_smooth selecting the correct zero.")
    print()
    
    # First, analyze which pivot is best where
    best_by_region, _ = analyze_pivot_reliability(n_train)
    
    print("\n" + "=" * 60)
    print("Testing Region-Weighted Selection")
    print("=" * 60)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_errors = []
    
    for i, img_id in enumerate(available_ids[n_train:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        true_depth_small = np.array(Image.fromarray((true_depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compute pivot depths
        pivots = compute_pivot_depths(rgb_small)
        
        # Create region masks
        top_mask = np.zeros((new_h, new_w), dtype=bool)
        top_mask[:new_h//3, :] = True
        
        middle_mask = np.zeros((new_h, new_w), dtype=bool)
        middle_mask[new_h//3:2*new_h//3, :] = True
        
        bottom_mask = np.zeros((new_h, new_w), dtype=bool)
        bottom_mask[2*new_h//3:, :] = True
        
        # Edge detection
        gray = 0.299 * rgb_small[:,:,0] + 0.587 * rgb_small[:,:,1] + 0.114 * rgb_small[:,:,2]
        edges = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
        edge_mask = edges > np.percentile(edges, 70)
        
        # Build depth map using best pivot for each region
        pred_depth = np.zeros((new_h, new_w))
        
        # Vertical regions
        pred_depth[top_mask] = pivots[best_by_region['top']][top_mask]
        pred_depth[middle_mask] = pivots[best_by_region['middle']][middle_mask]
        pred_depth[bottom_mask] = pivots[best_by_region['bottom']][bottom_mask]
        
        # Override with edge-specific pivot at edges
        pred_depth[edge_mask] = pivots[best_by_region['edges']][edge_mask]
        
        # Smooth
        pred_depth = gaussian_filter(pred_depth, sigma=2.0)
        pred_depth = _normalize(pred_depth)
        
        mae = np.mean(np.abs(pred_depth - true_depth_small))
        test_errors.append(mae)
        
        if i < 3:
            print(f"\n  Test {i+1}: {img_id}")
            print(f"    MAE: {mae:.4f}")
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Region-Weighted Selection MAE: {np.mean(test_errors):.4f}")
    print(f"  Dimensional Upcasting:         0.212")
    print(f"  Self-Assembling Pivots:        0.212")
    print(f"  Vertical alone:                0.182")
    print(f"  Statistical best:              0.199")
    
    return best_by_region


if __name__ == "__main__":
    # First run the basic upcasting experiment
    upcaster = run_upcasting_experiment(n_train=20, n_test=10)
    
    # Then run the region-weighted experiment
    best_by_region = run_region_weighted_experiment(n_train=30, n_test=10)
