#!/usr/bin/env python3
"""
Experiment: Phase-Aware Holographic Depth Estimation

Key insight: If magnitude captures HOW MUCH of a depth cue, 
phase captures HOW IT COMBINES with other cues.

- Same phase → constructive interference (reinforce)
- Opposite phase → destructive interference (cancel)
- Phase difference → partial combination

This is Feynman's path integral applied to depth:
Each depth cue is a "path" with magnitude and phase.
The final depth is the interference pattern.

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
# COMPLEX-VALUED DEPTH DIMENSIONS
# =============================================================================

@dataclass
class ComplexDimension:
    """A depth dimension with magnitude and phase."""
    name: str
    weight: float  # φ-based weight (magnitude)
    phase: float   # Phase in radians [0, 2π)
    extractor: callable
    
    def extract_complex(self, gray: np.ndarray) -> np.ndarray:
        """Extract dimension as complex values: magnitude × e^(iφ)"""
        magnitude = self.extractor(gray)
        # Phase can vary spatially or be constant
        return magnitude * np.exp(1j * self.phase)


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


def extract_luminance(gray: np.ndarray) -> np.ndarray:
    return gray.copy()


def extract_local_phase(gray: np.ndarray) -> np.ndarray:
    """
    Extract LOCAL phase from the image using Fourier analysis.
    
    This captures the "twist" at each location - how the local
    structure relates to the global structure.
    """
    F = fft2(gray)
    # Local phase is the angle of the Fourier transform
    phase = np.angle(F)
    # Shift to get spatial phase variation
    phase_shifted = np.angle(ifft2(fftshift(F)))
    return phase_shifted


def extract_edge_orientation(gray: np.ndarray) -> np.ndarray:
    """
    Extract edge orientation as phase.
    
    Horizontal edges → phase 0
    Vertical edges → phase π/2
    Diagonal edges → phase π/4 or 3π/4
    """
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    orientation = np.arctan2(grad_y, grad_x)  # [-π, π]
    return orientation


# =============================================================================
# PHASE-AWARE DEPTH ESTIMATOR
# =============================================================================

class PhaseHolographicDepth:
    """
    Depth estimation using complex-valued (magnitude + phase) dimensions.
    
    Phase determines how dimensions interfere:
    - φ = 0: adds to depth (foreground cue)
    - φ = π: subtracts from depth (background cue)
    - φ = π/2: orthogonal (independent cue)
    
    Key insight: Phase can be SPATIALLY VARYING, derived from the image.
    This allows local interference patterns that adapt to image content.
    """
    
    def __init__(self, use_spatial_phase: bool = False):
        self.use_spatial_phase = use_spatial_phase
        
        # Initialize dimensions with magnitude weights AND phases
        # Phases are learned to optimize interference patterns
        self.dimensions = [
            ComplexDimension("vertical_gradient", PHI**2, 0.0, extract_vertical_gradient),
            ComplexDimension("edges", PHI**1, 0.0, extract_edges),
            ComplexDimension("frequency", PHI**0, 0.0, extract_frequency),
            ComplexDimension("saliency", PHI**(-1), 0.0, extract_saliency),
            ComplexDimension("luminance", PHI**0, 0.0, extract_luminance),
        ]
        
        # Learnable parameters
        self.weight_adjustments = {d.name: 1.0 for d in self.dimensions}
        self.phase_adjustments = {d.name: 0.0 for d in self.dimensions}
        
        # Training data
        self.training_data: List[Tuple[np.ndarray, np.ndarray, str]] = []
    
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return image.copy()
    
    def _resize_to_match(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if arr.shape == target_shape:
            return arr
        pil = Image.fromarray((np.abs(arr) * 255).astype(np.uint8))
        pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0
    
    def extract_complex_dimensions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all dimensions as complex values."""
        gray = _normalize(self._to_gray(image))
        
        # Get spatially-varying phase from image if enabled
        if self.use_spatial_phase:
            # Edge orientation provides local phase
            edge_phase = extract_edge_orientation(gray)
            # Local Fourier phase
            local_phase = extract_local_phase(gray)
        
        result = {}
        for dim in self.dimensions:
            magnitude = dim.extractor(gray)
            
            if self.use_spatial_phase:
                # Phase varies spatially based on image content
                # Different dimensions use different phase sources
                if dim.name == "edges":
                    # Edges use their own orientation as phase
                    phase = edge_phase + self.phase_adjustments[dim.name]
                elif dim.name == "frequency":
                    # Frequency uses local Fourier phase
                    phase = local_phase + self.phase_adjustments[dim.name]
                else:
                    # Others use global learned phase
                    phase = dim.phase + self.phase_adjustments[dim.name]
            else:
                # Global phase only
                phase = dim.phase + self.phase_adjustments[dim.name]
            
            result[dim.name] = magnitude * np.exp(1j * phase)
        
        return result
    
    def compose_depth_complex(self, dimensions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compose depth using complex interference.
        
        depth = |Σ weight_i × complex_i|
        
        Dimensions with same phase reinforce.
        Dimensions with opposite phase cancel.
        """
        total = None
        
        for dim in self.dimensions:
            w = dim.weight * self.weight_adjustments[dim.name]
            if w <= 0:
                continue
            
            complex_val = dimensions[dim.name]
            
            if total is None:
                total = w * complex_val
            else:
                if complex_val.shape != total.shape:
                    # Resize magnitude, preserve phase
                    mag = self._resize_to_match(np.abs(complex_val), total.shape)
                    phase = self.phase_adjustments[dim.name]
                    complex_val = mag * np.exp(1j * phase)
                total = total + w * complex_val
        
        # Final depth is the MAGNITUDE of the interference
        depth = np.abs(total)
        depth = gaussian_filter(depth, sigma=2.0)
        
        return _normalize(depth)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        dims = self.extract_complex_dimensions(image)
        return self.compose_depth_complex(dims)
    
    def add_training_image(self, rgb: np.ndarray, depth: np.ndarray, image_id: str):
        self.training_data.append((rgb, depth, image_id))
    
    def learn_phases(self, n_iterations: int = 5):
        """
        Learn optimal phases for each dimension.
        
        This is the key: find phases that make dimensions interfere
        constructively where depth is high, destructively where low.
        """
        if not self.training_data:
            return
        
        print(f"Learning phases from {len(self.training_data)} images...")
        
        # Pre-extract dimensions
        all_dims = []
        all_targets = []
        
        for rgb, depth, _ in self.training_data:
            dims = {}
            gray = _normalize(self._to_gray(rgb))
            for dim in self.dimensions:
                dims[dim.name] = dim.extractor(gray)  # Just magnitude for now
            
            target_shape = list(dims.values())[0].shape
            depth_resized = self._resize_to_match(depth, target_shape)
            
            all_dims.append(dims)
            all_targets.append(depth_resized)
        
        # Search for optimal phases
        best_mae = float('inf')
        best_phases = self.phase_adjustments.copy()
        best_weights = self.weight_adjustments.copy()
        
        # Phase search values (in radians)
        phase_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        weight_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        
        for iteration in range(n_iterations):
            # Optimize weights first
            for dim_name in self.weight_adjustments:
                for w in weight_values:
                    self.weight_adjustments[dim_name] = w
                    
                    mae = self._compute_mae(all_dims, all_targets)
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = self.weight_adjustments.copy()
                        best_phases = self.phase_adjustments.copy()
                
                self.weight_adjustments = best_weights.copy()
            
            # Then optimize phases
            for dim_name in self.phase_adjustments:
                for phase in phase_values:
                    self.phase_adjustments[dim_name] = phase
                    
                    mae = self._compute_mae(all_dims, all_targets)
                    if mae < best_mae:
                        best_mae = mae
                        best_phases = self.phase_adjustments.copy()
                        best_weights = self.weight_adjustments.copy()
                
                self.phase_adjustments = best_phases.copy()
            
            print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
        
        self.phase_adjustments = best_phases
        self.weight_adjustments = best_weights
        
        print(f"\nLearned parameters:")
        for dim in self.dimensions:
            w = dim.weight * self.weight_adjustments[dim.name]
            p = self.phase_adjustments[dim.name]
            print(f"  {dim.name}: weight={w:.3f}, phase={p:.2f}rad ({np.degrees(p):.0f}°)")
        
        return best_mae
    
    def _compute_mae(self, all_dims: List[Dict], all_targets: List[np.ndarray]) -> float:
        """Compute MAE with current parameters."""
        total_mae = 0
        
        for dims_mag, target in zip(all_dims, all_targets):
            # Convert magnitudes to complex with current phases
            dims_complex = {}
            for dim in self.dimensions:
                phase = dim.phase + self.phase_adjustments[dim.name]
                dims_complex[dim.name] = dims_mag[dim.name] * np.exp(1j * phase)
            
            pred = self.compose_depth_complex(dims_complex)
            
            if pred.shape != target.shape:
                pred = self._resize_to_match(pred, target.shape)
            
            total_mae += np.mean(np.abs(pred - target))
        
        return total_mae / len(all_targets)
    
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


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_phase_experiment(n_train: int = 50, n_test: int = 10, use_spatial_phase: bool = False):
    """Run the phase-aware holographic depth experiment."""
    print("=" * 70)
    print("EXPERIMENT: Phase-Aware Holographic Depth")
    print("=" * 70)
    print()
    print("Key insight: Phase determines HOW dimensions combine")
    print("  - Same phase (0°) → constructive interference (reinforce)")
    print("  - Opposite phase (180°) → destructive interference (cancel)")
    print("  - Orthogonal phase (90°) → independent contribution")
    print()
    print(f"Spatial phase: {use_spatial_phase}")
    print()
    
    # Get available images
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Initialize
    model = PhaseHolographicDepth(use_spatial_phase=use_spatial_phase)
    
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
        
        model.add_training_image(rgb, depth, img_id)
    
    print(f"  Loaded {len(model.training_data)} images")
    
    # Learn phases
    print()
    print("=" * 60)
    print("LEARNING PHASES")
    print("=" * 60)
    model.learn_phases(n_iterations=5)
    
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
        
        pred_depth = model.predict(rgb)
        true_resized = model._resize_to_match(true_depth, pred_depth.shape)
        
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
    
    # Analyze phase relationships
    print("Phase analysis:")
    for dim in model.dimensions:
        p = model.phase_adjustments[dim.name]
        if abs(p) < 0.1:
            role = "foreground cue (adds to depth)"
        elif abs(p - np.pi) < 0.1:
            role = "background cue (subtracts from depth)"
        elif abs(p - np.pi/2) < 0.1 or abs(p - 3*np.pi/2) < 0.1:
            role = "orthogonal (independent)"
        else:
            role = f"partial interference"
        print(f"  {dim.name}: {np.degrees(p):.0f}° → {role}")
    
    return model


def run_learned_correction_experiment(n_train: int = 50, n_test: int = 10):
    """
    Learn a SIGNED correction term from residuals.
    
    Key insight from user: "filling things in past this point is phase shifts"
    
    Interpretation: The magnitude-only model captures the "base" depth.
    The residual has STRUCTURE that can be learned as a signed correction.
    
    Phase here means: the correction can be POSITIVE or NEGATIVE at each pixel,
    determined by learning which input features predict the sign of the residual.
    """
    print("=" * 70)
    print("EXPERIMENT: Learned Signed Correction (Phase as Sign)")
    print("=" * 70)
    print()
    print("Approach:")
    print("  1. Train magnitude-only baseline")
    print("  2. Compute residuals (true - predicted)")
    print("  3. Learn which features predict residual SIGN")
    print("  4. Add signed correction term")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Step 1: Train magnitude-only model
    print("Step 1: Training magnitude-only baseline...")
    model = PhaseHolographicDepth(use_spatial_phase=False)
    
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        if not img_path.exists():
            continue
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        model.add_training_image(rgb, depth, img_id)
    
    baseline_mae = model.learn_phases(n_iterations=3)
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    
    # Step 2: Analyze residual structure
    print("\nStep 2: Analyzing residual structure...")
    
    # Collect residuals and their relationship to input features
    all_residuals = []
    all_features = []
    
    for rgb, true_depth, img_id in model.training_data:
        pred = model.predict(rgb)
        true_resized = model._resize_to_match(true_depth, pred.shape)
        residual = true_resized - pred  # SIGNED residual
        
        gray = _normalize(model._to_gray(rgb))
        gray = model._resize_to_match(gray, pred.shape)
        
        # Extract features that might predict residual sign
        features = {
            'luminance': gray,
            'edges': extract_edges(gray),
            'vertical': extract_vertical_gradient(gray),
            'frequency': extract_frequency(gray),
        }
        
        all_residuals.append(residual)
        all_features.append(features)
    
    # Find which feature best predicts residual sign
    print("\n  Correlation of features with residual:")
    best_corr = 0
    best_feature = None
    
    for fname in ['luminance', 'edges', 'vertical', 'frequency']:
        correlations = []
        for i, residual in enumerate(all_residuals):
            feature = all_features[i][fname]
            # Correlation with SIGNED residual
            corr = np.corrcoef(feature.flatten(), residual.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_corr = np.mean(correlations)
        print(f"    {fname}: r = {avg_corr:.4f}")
        
        if abs(avg_corr) > abs(best_corr):
            best_corr = avg_corr
            best_feature = fname
    
    print(f"\n  Best predictor: {best_feature} (r = {best_corr:.4f})")
    
    # Step 3: Learn correction coefficient
    print("\nStep 3: Learning signed correction...")
    
    # The correction is: pred_corrected = pred + α * feature * sign(best_corr)
    # Find optimal α
    best_alpha = 0
    best_mae = baseline_mae
    
    for alpha in np.linspace(-0.5, 0.5, 21):
        total_mae = 0
        for i, (rgb, true_depth, _) in enumerate(model.training_data):
            pred = model.predict(rgb)
            true_resized = model._resize_to_match(true_depth, pred.shape)
            
            # Apply correction
            correction = alpha * all_features[i][best_feature]
            pred_corrected = np.clip(pred + correction, 0, 1)
            
            total_mae += np.mean(np.abs(pred_corrected - true_resized))
        
        avg_mae = total_mae / len(model.training_data)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_alpha = alpha
    
    print(f"  Optimal α = {best_alpha:.3f}")
    print(f"  Corrected MAE: {best_mae:.4f}")
    print(f"  Improvement: {baseline_mae - best_mae:.4f}")
    
    # Step 4: Test with correction
    print("\nStep 4: Testing with signed correction...")
    test_ids = available_ids[n_train:n_train + n_test]
    errors_baseline = []
    errors_corrected = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        pred = model.predict(rgb)
        true_resized = model._resize_to_match(true_depth, pred.shape)
        
        # Baseline error
        mae_base = np.mean(np.abs(pred - true_resized))
        errors_baseline.append(mae_base)
        
        # Corrected error
        gray = _normalize(model._to_gray(rgb))
        gray = model._resize_to_match(gray, pred.shape)
        
        if best_feature == 'luminance':
            correction_feature = gray
        elif best_feature == 'edges':
            correction_feature = extract_edges(gray)
        elif best_feature == 'vertical':
            correction_feature = extract_vertical_gradient(gray)
        else:
            correction_feature = extract_frequency(gray)
        
        pred_corrected = np.clip(pred + best_alpha * correction_feature, 0, 1)
        mae_corr = np.mean(np.abs(pred_corrected - true_resized))
        errors_corrected.append(mae_corr)
    
    print(f"\n  Baseline Test MAE: {np.mean(errors_baseline):.4f}")
    print(f"  Corrected Test MAE: {np.mean(errors_corrected):.4f}")
    print(f"  Test Improvement: {np.mean(errors_baseline) - np.mean(errors_corrected):.4f}")
    
    return model, best_feature, best_alpha


def run_iterative_correction_experiment(n_train: int = 50, n_test: int = 10):
    """
    Apply MULTIPLE signed corrections iteratively.
    
    Each iteration:
    1. Compute residual from current prediction
    2. Find feature that best predicts residual
    3. Add correction term
    4. Repeat until no improvement
    
    This is like discovering multiple "phase dimensions" that each
    contribute a signed correction to the base prediction.
    """
    print("=" * 70)
    print("EXPERIMENT: Iterative Signed Corrections")
    print("=" * 70)
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Load data
    print("Loading data...")
    model = PhaseHolographicDepth(use_spatial_phase=False)
    
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        if not img_path.exists():
            continue
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        model.add_training_image(rgb, depth, img_id)
    
    # Train baseline
    print("\nTraining baseline...")
    baseline_mae = model.learn_phases(n_iterations=3)
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    
    # Pre-extract all features
    all_features = []
    all_true = []
    
    for rgb, true_depth, _ in model.training_data:
        pred = model.predict(rgb)
        gray = _normalize(model._to_gray(rgb))
        gray = model._resize_to_match(gray, pred.shape)
        true_resized = model._resize_to_match(true_depth, pred.shape)
        
        features = {
            'luminance': gray,
            'inv_luminance': 1.0 - gray,
            'edges': extract_edges(gray),
            'vertical': extract_vertical_gradient(gray),
            'frequency': extract_frequency(gray),
            'saliency': extract_saliency(gray),
        }
        all_features.append(features)
        all_true.append(true_resized)
    
    # Iterative correction
    corrections = []  # List of (feature_name, alpha)
    current_predictions = [model.predict(rgb) for rgb, _, _ in model.training_data]
    
    for iteration in range(5):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Compute residuals
        residuals = []
        for i, pred in enumerate(current_predictions):
            residual = all_true[i] - pred
            residuals.append(residual)
        
        # Find best predictor
        best_corr = 0
        best_feature = None
        
        for fname in all_features[0].keys():
            # Skip already used features
            if any(f == fname for f, _ in corrections):
                continue
            
            correlations = []
            for i, residual in enumerate(residuals):
                feature = all_features[i][fname]
                corr = np.corrcoef(feature.flatten(), residual.flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            avg_corr = np.mean(correlations) if correlations else 0
            if abs(avg_corr) > abs(best_corr):
                best_corr = avg_corr
                best_feature = fname
        
        if best_feature is None or abs(best_corr) < 0.05:
            print("  No significant predictor found. Stopping.")
            break
        
        print(f"  Best predictor: {best_feature} (r = {best_corr:.4f})")
        
        # Find optimal alpha
        best_alpha = 0
        best_mae = float('inf')
        
        for alpha in np.linspace(-0.5, 0.5, 21):
            total_mae = 0
            for i, pred in enumerate(current_predictions):
                correction = alpha * all_features[i][best_feature]
                pred_corrected = np.clip(pred + correction, 0, 1)
                total_mae += np.mean(np.abs(pred_corrected - all_true[i]))
            
            avg_mae = total_mae / len(current_predictions)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_alpha = alpha
        
        # Check if improvement is significant
        current_mae = sum(np.mean(np.abs(p - t)) for p, t in zip(current_predictions, all_true)) / len(current_predictions)
        
        if best_mae >= current_mae - 0.001:
            print(f"  No improvement (current: {current_mae:.4f}, best: {best_mae:.4f}). Stopping.")
            break
        
        print(f"  α = {best_alpha:.3f}, MAE: {best_mae:.4f} (was {current_mae:.4f})")
        
        # Apply correction
        corrections.append((best_feature, best_alpha))
        for i in range(len(current_predictions)):
            correction = best_alpha * all_features[i][best_feature]
            current_predictions[i] = np.clip(current_predictions[i] + correction, 0, 1)
    
    # Summary
    print("\n" + "=" * 60)
    print("LEARNED CORRECTIONS (Phase Shifts)")
    print("=" * 60)
    for fname, alpha in corrections:
        sign = "+" if alpha > 0 else "-"
        print(f"  {fname}: α = {alpha:+.3f} ({sign} constructive)")
    
    final_mae = sum(np.mean(np.abs(p - t)) for p, t in zip(current_predictions, all_true)) / len(current_predictions)
    print(f"\nFinal Train MAE: {final_mae:.4f}")
    print(f"Improvement from baseline: {baseline_mae - final_mae:.4f}")
    
    # Test
    print("\nTesting...")
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
        
        pred = model.predict(rgb)
        gray = _normalize(model._to_gray(rgb))
        gray = model._resize_to_match(gray, pred.shape)
        true_resized = model._resize_to_match(true_depth, pred.shape)
        
        # Apply all corrections
        for fname, alpha in corrections:
            if fname == 'luminance':
                feature = gray
            elif fname == 'inv_luminance':
                feature = 1.0 - gray
            elif fname == 'edges':
                feature = extract_edges(gray)
            elif fname == 'vertical':
                feature = extract_vertical_gradient(gray)
            elif fname == 'frequency':
                feature = extract_frequency(gray)
            elif fname == 'saliency':
                feature = extract_saliency(gray)
            else:
                continue
            
            pred = np.clip(pred + alpha * feature, 0, 1)
        
        mae = np.mean(np.abs(pred - true_resized))
        errors.append(mae)
    
    print(f"Test MAE: {np.mean(errors):.4f}")
    
    return model, corrections


if __name__ == "__main__":
    model, corrections = run_iterative_correction_experiment(n_train=50, n_test=10)
