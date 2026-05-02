#!/usr/bin/env python3
"""
Experiment: Physical Mechanisms for Depth Estimation

Key insight from user: We've been operating in OUTPUT SPACE (correlating 
features with depth), not modeling the INTERNAL PHYSICAL MECHANISMS that
generate depth perception.

Physical mechanisms that encode depth:

1. LIGHT SOURCES (multiple, positioned)
   - Shadows: occluded from light source
   - Shading: I ∝ cos(θ) for Lambertian surfaces
   - Multiple lights create multiple shadow/shading patterns

2. POLARIZATION
   - Specular = polarized (encodes surface normal)
   - Diffuse = unpolarized
   - Phase of polarization → surface orientation → depth

3. LENS/OPTICS
   - Defocus blur: objects at different depths have different blur
   - Chromatic aberration: R/G/B focus at different depths
   - Vignetting: light falloff encodes radial distance

4. PERSPECTIVE GEOMETRY
   - Vanishing points
   - Texture gradient (density increases with distance)
   - Relative size

The φ-Zipf duality applies to these MECHANISMS, not the output features:
- Rare lighting conditions → HIGH information about depth
- Common lighting → LOW information (ambiguous)

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
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
# PHYSICAL MECHANISM 1: LIGHT SOURCE ANALYSIS
# =============================================================================

def estimate_light_direction(gray: np.ndarray) -> Tuple[float, float]:
    """
    Estimate dominant light direction from shading gradients.
    
    For Lambertian surfaces: I ∝ cos(θ) where θ is angle to light.
    The gradient of intensity points toward the light source.
    """
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    
    # Weight by gradient magnitude (strong gradients are more reliable)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    weights = magnitude / (magnitude.sum() + 1e-10)
    
    # Weighted average gradient direction
    avg_grad_x = np.sum(grad_x * weights)
    avg_grad_y = np.sum(grad_y * weights)
    
    # Light direction is opposite to gradient (bright toward light)
    light_x = -avg_grad_x
    light_y = -avg_grad_y
    
    # Normalize
    norm = np.sqrt(light_x**2 + light_y**2) + 1e-10
    return light_x / norm, light_y / norm


def extract_shading_depth(gray: np.ndarray, light_dir: Tuple[float, float]) -> np.ndarray:
    """
    Extract depth from shading assuming Lambertian surface.
    
    If we know light direction L and observe intensity I,
    and assume surface normal N varies smoothly,
    then depth Z relates to N which relates to I.
    
    Simplified: surfaces facing the light are "closer" in perceptual depth.
    """
    lx, ly = light_dir
    
    # Compute how much each pixel "faces" the light
    # Using gradient as proxy for surface normal
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    
    # Dot product with light direction
    facing_light = lx * grad_x + ly * grad_y
    
    # Normalize
    return _normalize(facing_light)


def extract_shadow_depth(gray: np.ndarray) -> np.ndarray:
    """
    Extract depth from shadows.
    
    Shadows indicate:
    1. The shadowed region is BEHIND something
    2. The shadow-casting object is in FRONT
    
    Dark regions with sharp boundaries are likely shadows.
    """
    # Find dark regions
    dark_mask = gray < 0.3
    
    # Find edges of dark regions (shadow boundaries)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    # Shadow boundaries: dark region + strong edge
    shadow_boundary = dark_mask.astype(float) * edges
    
    # Shadows are typically at greater depth (behind objects)
    # But shadow-casting edges indicate depth discontinuity
    shadow_depth = 1.0 - gray  # Dark = far (in shadow)
    shadow_depth = shadow_depth * (1 + shadow_boundary)  # Enhance at boundaries
    
    return _normalize(shadow_depth)


# =============================================================================
# PHYSICAL MECHANISM 2: DEFOCUS/BLUR ANALYSIS
# =============================================================================

def extract_defocus_depth(gray: np.ndarray) -> np.ndarray:
    """
    Extract depth from defocus blur.
    
    Objects at the focal plane are sharp.
    Objects away from focal plane are blurred.
    
    Measure local sharpness as proxy for depth.
    """
    # Compute local variance (sharpness measure)
    local_mean = uniform_filter(gray, size=5)
    local_sq_mean = uniform_filter(gray**2, size=5)
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    
    # High variance = sharp = at focal plane
    # Low variance = blurred = away from focal plane
    sharpness = np.sqrt(local_var)
    
    # Typically, photographers focus on foreground subjects
    # So sharp = close, blurred = far
    # But this is scene-dependent
    
    return _normalize(sharpness)


def extract_chromatic_aberration(rgb: np.ndarray) -> np.ndarray:
    """
    Extract depth from chromatic aberration.
    
    Different wavelengths focus at different depths.
    R focuses farther, B focuses closer (typically).
    
    The difference between R and B channels encodes depth.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros(rgb.shape[:2])
    
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    
    # Chromatic difference
    # Objects at different depths have different R-B relationships
    # due to lens chromatic aberration
    chromatic_diff = r - b
    
    # Smooth to reduce noise
    chromatic_diff = gaussian_filter(chromatic_diff, sigma=2.0)
    
    return _normalize(chromatic_diff)


# =============================================================================
# PHYSICAL MECHANISM 3: PERSPECTIVE GEOMETRY
# =============================================================================

def extract_texture_gradient(gray: np.ndarray) -> np.ndarray:
    """
    Extract depth from texture gradient.
    
    Texture density increases with distance (perspective).
    Fine texture = close, coarse texture = far.
    
    Measure local frequency content.
    """
    # Compute local frequency using windowed FFT
    h, w = gray.shape
    window_size = 32
    stride = 8
    
    freq_map = np.zeros((h, w))
    count_map = np.zeros((h, w))
    
    for i in range(0, h - window_size, stride):
        for j in range(0, w - window_size, stride):
            window = gray[i:i+window_size, j:j+window_size]
            
            # FFT
            F = fft2(window)
            magnitude = np.abs(F)
            
            # Compute mean frequency (weighted by magnitude)
            u = np.arange(window_size) - window_size // 2
            v = np.arange(window_size) - window_size // 2
            U, V = np.meshgrid(u, v)
            freq = np.sqrt(U**2 + V**2)
            
            mean_freq = np.sum(freq * fftshift(magnitude)) / (np.sum(magnitude) + 1e-10)
            
            # Assign to center of window
            ci, cj = i + window_size // 2, j + window_size // 2
            freq_map[ci, cj] = mean_freq
            count_map[ci, cj] = 1
    
    # Interpolate
    freq_map = gaussian_filter(freq_map, sigma=stride)
    
    # High frequency = fine texture = close
    return _normalize(freq_map)


def extract_vanishing_point_depth(gray: np.ndarray) -> np.ndarray:
    """
    Extract depth from vanishing point geometry.
    
    Lines converge toward vanishing points at infinity.
    Distance from vanishing point indicates depth.
    
    Simplified: assume single vanishing point near image center-top.
    """
    h, w = gray.shape
    
    # Assume vanishing point at top-center (common for outdoor scenes)
    vp_y, vp_x = 0, w // 2
    
    # Distance from vanishing point
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    
    dist_from_vp = np.sqrt((y_coords - vp_y)**2 + (x_coords - vp_x)**2)
    
    # Closer to VP = farther in depth
    # Farther from VP = closer in depth
    depth = dist_from_vp / dist_from_vp.max()
    
    return _normalize(depth)


# =============================================================================
# PHYSICAL MECHANISM 4: POLARIZATION (SIMULATED)
# =============================================================================

def extract_polarization_proxy(rgb: np.ndarray) -> np.ndarray:
    """
    Extract polarization-like information from RGB.
    
    We don't have actual polarization data, but we can approximate:
    - Specular highlights are polarized
    - Diffuse regions are unpolarized
    
    The ratio of specular to diffuse encodes surface orientation.
    """
    if rgb.ndim != 3:
        return np.zeros(rgb.shape[:2])
    
    # Convert to HSV-like representation
    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    
    # Saturation: low saturation = specular (white highlights)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
    
    # Specular regions have low saturation and high value
    specular_mask = (saturation < 0.3) & (max_rgb > 0.7)
    
    # Specular highlights often indicate surfaces facing the camera
    # (Fresnel effect: more specular at grazing angles)
    polarization_proxy = 1.0 - saturation  # High where specular
    
    return _normalize(polarization_proxy)


# =============================================================================
# COMBINED PHYSICAL MODEL
# =============================================================================

class PhysicalDepthModel:
    """
    Depth estimation based on physical mechanisms.
    
    Instead of correlating output features with depth,
    we model the physical processes that generate depth cues.
    """
    
    def __init__(self):
        # φ-Zipf weights based on information content of each mechanism
        # Rare/informative mechanisms get higher weight
        self.mechanism_weights = {
            'shading': PHI**1,        # Common but informative
            'shadow': PHI**0,         # Less common, very informative
            'defocus': PHI**(-1),     # Depends on camera settings
            'chromatic': PHI**(-2),   # Subtle, lens-dependent
            'texture': PHI**1,        # Common, informative
            'vanishing': PHI**2,      # Strong geometric prior
            'polarization': PHI**(-1), # We only have proxy
        }
        
        # Normalize
        total = sum(self.mechanism_weights.values())
        self.mechanism_weights = {k: v/total for k, v in self.mechanism_weights.items()}
        
        # Learned adjustments
        self.weight_adjustments = {k: 1.0 for k in self.mechanism_weights}
        self.phase_adjustments = {k: 0.0 for k in self.mechanism_weights}
    
    def extract_mechanisms(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract depth cues from each physical mechanism."""
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        
        # Light source analysis
        light_dir = estimate_light_direction(gray)
        
        mechanisms = {
            'shading': extract_shading_depth(gray, light_dir),
            'shadow': extract_shadow_depth(gray),
            'defocus': extract_defocus_depth(gray),
            'chromatic': extract_chromatic_aberration(rgb) if rgb.ndim == 3 else np.zeros_like(gray),
            'texture': extract_texture_gradient(gray),
            'vanishing': extract_vanishing_point_depth(gray),
            'polarization': extract_polarization_proxy(rgb) if rgb.ndim == 3 else np.zeros_like(gray),
        }
        
        return mechanisms
    
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Predict depth using physical mechanisms with complex interference."""
        mechanisms = self.extract_mechanisms(rgb)
        
        # Combine using complex interference (like phase holographic)
        total = None
        target_shape = list(mechanisms.values())[0].shape
        
        for name, depth_cue in mechanisms.items():
            w = self.mechanism_weights[name] * self.weight_adjustments[name]
            if w <= 0:
                continue
            
            phase = self.phase_adjustments[name]
            
            # Resize if needed
            if depth_cue.shape != target_shape:
                from PIL import Image as PILImage
                pil = PILImage.fromarray((depth_cue * 255).astype(np.uint8))
                pil = pil.resize((target_shape[1], target_shape[0]), PILImage.BILINEAR)
                depth_cue = np.array(pil).astype(np.float32) / 255.0
            
            # Complex representation
            complex_cue = depth_cue * np.exp(1j * phase)
            
            if total is None:
                total = w * complex_cue
            else:
                total = total + w * complex_cue
        
        # Final depth is magnitude of interference
        depth = np.abs(total)
        depth = gaussian_filter(depth, sigma=2.0)
        
        return _normalize(depth)
    
    def learn_parameters(self, train_data: List[Tuple[np.ndarray, np.ndarray]], 
                         n_iterations: int = 3):
        """Learn optimal weights and phases from training data."""
        print(f"Learning parameters from {len(train_data)} images...")
        
        # Pre-extract mechanisms
        all_mechanisms = []
        all_targets = []
        
        for rgb, depth in train_data:
            mechs = self.extract_mechanisms(rgb)
            target_shape = list(mechs.values())[0].shape
            
            if depth.shape != target_shape:
                pil = Image.fromarray((depth * 255).astype(np.uint8))
                pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                depth = np.array(pil).astype(np.float32) / 255.0
            
            all_mechanisms.append(mechs)
            all_targets.append(depth)
        
        # Coordinate descent
        weight_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        phase_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        
        best_mae = float('inf')
        best_weights = self.weight_adjustments.copy()
        best_phases = self.phase_adjustments.copy()
        
        for iteration in range(n_iterations):
            # Optimize weights
            for name in self.weight_adjustments:
                for w in weight_values:
                    self.weight_adjustments[name] = w
                    mae = self._compute_mae(all_mechanisms, all_targets)
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = self.weight_adjustments.copy()
                self.weight_adjustments = best_weights.copy()
            
            # Optimize phases
            for name in self.phase_adjustments:
                for p in phase_values:
                    self.phase_adjustments[name] = p
                    mae = self._compute_mae(all_mechanisms, all_targets)
                    if mae < best_mae:
                        best_mae = mae
                        best_phases = self.phase_adjustments.copy()
                self.phase_adjustments = best_phases.copy()
            
            print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
        
        self.weight_adjustments = best_weights
        self.phase_adjustments = best_phases
        
        print("\nLearned parameters:")
        for name in self.mechanism_weights:
            w = self.mechanism_weights[name] * self.weight_adjustments[name]
            p = self.phase_adjustments[name]
            print(f"  {name}: weight={w:.3f}, phase={np.degrees(p):.0f}°")
        
        return best_mae
    
    def _compute_mae(self, all_mechanisms, all_targets):
        """Compute MAE with current parameters."""
        total_mae = 0
        
        for mechs, target in zip(all_mechanisms, all_targets):
            total = None
            target_shape = target.shape
            
            for name, depth_cue in mechs.items():
                w = self.mechanism_weights[name] * self.weight_adjustments[name]
                if w <= 0:
                    continue
                
                phase = self.phase_adjustments[name]
                
                if depth_cue.shape != target_shape:
                    pil = Image.fromarray((depth_cue * 255).astype(np.uint8))
                    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                    depth_cue = np.array(pil).astype(np.float32) / 255.0
                
                complex_cue = depth_cue * np.exp(1j * phase)
                
                if total is None:
                    total = w * complex_cue
                else:
                    total = total + w * complex_cue
            
            pred = _normalize(gaussian_filter(np.abs(total), sigma=2.0))
            total_mae += np.mean(np.abs(pred - target))
        
        return total_mae / len(all_mechanisms)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_physical_model_experiment(n_train: int = 50, n_test: int = 10):
    """
    Compare physical mechanism model to output-space model.
    """
    print("=" * 70)
    print("EXPERIMENT: Physical Mechanisms vs Output Space")
    print("=" * 70)
    print()
    print("Hypothesis: Modeling the GENERATIVE PROCESS (light, optics, geometry)")
    print("should outperform correlating OUTPUT FEATURES with depth.")
    print()
    print("Physical mechanisms:")
    print("  1. Shading (Lambertian: I ∝ cos(θ))")
    print("  2. Shadows (occlusion from light)")
    print("  3. Defocus blur (depth of field)")
    print("  4. Chromatic aberration (R/B focus difference)")
    print("  5. Texture gradient (perspective)")
    print("  6. Vanishing point geometry")
    print("  7. Polarization proxy (specular vs diffuse)")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
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
            train_data.append((rgb, depth))
        else:
            test_data.append((rgb, depth))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Train physical model
    print("\n" + "=" * 60)
    print("Training Physical Mechanism Model")
    print("=" * 60)
    
    model = PhysicalDepthModel()
    train_mae = model.learn_parameters(train_data, n_iterations=3)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    test_errors = []
    for rgb, depth in test_data:
        pred = model.predict(rgb)
        
        if depth.shape != pred.shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        test_errors.append(np.mean(np.abs(pred - depth)))
    
    print(f"\n  Physical Model Test MAE: {np.mean(test_errors):.4f}")
    
    # Compare to previous best
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n  Physical Mechanisms:    {np.mean(test_errors):.4f}")
    print(f"  Output Space (best):    0.199")
    
    diff = np.mean(test_errors) - 0.199
    if diff < 0:
        print(f"\n  ✓ Physical model is better by {-diff:.4f}!")
    else:
        print(f"\n  Physical model is {diff:.4f} behind")
        print("\n  Analysis: Which mechanisms contribute?")
        for name in model.mechanism_weights:
            w = model.mechanism_weights[name] * model.weight_adjustments[name]
            if w > 0.01:
                print(f"    {name}: {w:.3f}")
    
    return model


def estimate_multiple_lights(gray: np.ndarray, n_lights: int = 3) -> List[Tuple[float, float, float]]:
    """
    Estimate multiple light source directions.
    
    Different regions of the image may be lit by different sources.
    Use clustering on gradient directions to find multiple lights.
    """
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Only consider strong gradients
    threshold = np.percentile(magnitude, 80)
    mask = magnitude > threshold
    
    if mask.sum() < 100:
        # Not enough gradients, return single light
        lx, ly = estimate_light_direction(gray)
        return [(lx, ly, 1.0)]
    
    # Get gradient directions at strong gradient points
    gx = grad_x[mask]
    gy = grad_y[mask]
    mags = magnitude[mask]
    
    # Cluster by angle
    angles = np.arctan2(gy, gx)
    
    # Simple k-means-like clustering on angles
    lights = []
    angle_bins = np.linspace(-np.pi, np.pi, n_lights + 1)
    
    for i in range(n_lights):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
        if bin_mask.sum() > 10:
            # Weighted average direction in this bin
            avg_angle = np.average(angles[bin_mask], weights=mags[bin_mask])
            weight = mags[bin_mask].sum() / mags.sum()
            
            # Light direction is opposite to gradient
            lx = -np.cos(avg_angle)
            ly = -np.sin(avg_angle)
            lights.append((lx, ly, weight))
    
    if not lights:
        lx, ly = estimate_light_direction(gray)
        return [(lx, ly, 1.0)]
    
    return lights


def extract_multi_light_shading(gray: np.ndarray) -> np.ndarray:
    """
    Extract shading depth considering multiple light sources.
    
    Each light creates its own shading pattern.
    The combination reveals more about 3D structure.
    """
    lights = estimate_multiple_lights(gray, n_lights=3)
    
    total_shading = np.zeros_like(gray)
    
    for lx, ly, weight in lights:
        shading = extract_shading_depth(gray, (lx, ly))
        total_shading = total_shading + weight * shading
    
    return _normalize(total_shading)


def extract_occlusion_boundaries(gray: np.ndarray) -> np.ndarray:
    """
    Extract depth discontinuities from occlusion boundaries.
    
    Occlusion boundaries are where one surface ends and another begins.
    They have specific edge characteristics:
    - Strong intensity gradient
    - Often accompanied by texture discontinuity
    """
    # Edge detection at multiple scales
    edges_fine = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
    
    gray_smooth = gaussian_filter(gray, sigma=3.0)
    edges_coarse = np.sqrt(sobel(gray_smooth, axis=0)**2 + sobel(gray_smooth, axis=1)**2)
    
    # Occlusion boundaries persist across scales
    # Texture edges disappear at coarse scale
    occlusion = np.minimum(edges_fine, edges_coarse * 2)
    
    return _normalize(occlusion)


def run_multi_light_experiment(n_train: int = 50, n_test: int = 10):
    """
    Test multi-light and occlusion boundary mechanisms.
    """
    print("=" * 70)
    print("EXPERIMENT: Multi-Light Physical Model")
    print("=" * 70)
    print()
    print("Additional mechanisms:")
    print("  - Multiple light source estimation")
    print("  - Occlusion boundary detection")
    print()
    
    # Load data
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    train_data = []
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
            train_data.append((rgb, depth))
        else:
            test_data.append((rgb, depth))
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Extract all mechanisms including new ones
    print("\nExtracting mechanisms...")
    
    all_mechanisms = []
    all_targets = []
    
    for rgb, depth in train_data:
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        
        mechs = {
            'multi_light_shading': extract_multi_light_shading(gray),
            'occlusion': extract_occlusion_boundaries(gray),
            'shadow': extract_shadow_depth(gray),
            'defocus': extract_defocus_depth(gray),
            'vanishing': extract_vanishing_point_depth(gray),
            'vertical': np.tile(np.linspace(1, 0, gray.shape[0]).reshape(-1, 1), (1, gray.shape[1])),
        }
        
        target_shape = gray.shape
        if depth.shape != target_shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        all_mechanisms.append(mechs)
        all_targets.append(depth)
    
    # Compute correlations
    print("\nMechanism correlations with depth:")
    for name in all_mechanisms[0].keys():
        corrs = []
        for mechs, target in zip(all_mechanisms, all_targets):
            corr = np.corrcoef(mechs[name].flatten(), target.flatten())[0, 1]
            if not np.isnan(corr):
                corrs.append(abs(corr))
        print(f"  {name}: |r| = {np.mean(corrs):.3f}")
    
    # Learn optimal combination
    print("\nLearning optimal combination...")
    
    # Initialize weights by correlation
    mechanism_weights = {}
    for name in all_mechanisms[0].keys():
        corrs = []
        for mechs, target in zip(all_mechanisms, all_targets):
            corr = np.corrcoef(mechs[name].flatten(), target.flatten())[0, 1]
            if not np.isnan(corr):
                corrs.append(abs(corr))
        mechanism_weights[name] = np.mean(corrs)
    
    # Normalize
    total = sum(mechanism_weights.values())
    mechanism_weights = {k: v/total for k, v in mechanism_weights.items()}
    
    # Coordinate descent for phases
    phase_adjustments = {k: 0.0 for k in mechanism_weights}
    phase_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    best_mae = float('inf')
    best_phases = phase_adjustments.copy()
    
    for iteration in range(3):
        for name in phase_adjustments:
            for p in phase_values:
                phase_adjustments[name] = p
                
                # Compute MAE
                total_mae = 0
                for mechs, target in zip(all_mechanisms, all_targets):
                    total_complex = None
                    for mech_name, mech_val in mechs.items():
                        w = mechanism_weights[mech_name]
                        phase = phase_adjustments[mech_name]
                        complex_val = mech_val * np.exp(1j * phase)
                        
                        if total_complex is None:
                            total_complex = w * complex_val
                        else:
                            total_complex = total_complex + w * complex_val
                    
                    pred = _normalize(gaussian_filter(np.abs(total_complex), sigma=2.0))
                    total_mae += np.mean(np.abs(pred - target))
                
                mae = total_mae / len(all_mechanisms)
                if mae < best_mae:
                    best_mae = mae
                    best_phases = phase_adjustments.copy()
            
            phase_adjustments = best_phases.copy()
        
        print(f"  Iteration {iteration + 1}: MAE = {best_mae:.4f}")
    
    phase_adjustments = best_phases
    
    print("\nLearned parameters:")
    for name in mechanism_weights:
        print(f"  {name}: weight={mechanism_weights[name]:.3f}, phase={np.degrees(phase_adjustments[name]):.0f}°")
    
    # Test
    print("\nTesting...")
    test_errors = []
    
    for rgb, depth in test_data:
        if rgb.ndim == 3:
            gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        else:
            gray = rgb.copy()
        gray = _normalize(gray)
        
        mechs = {
            'multi_light_shading': extract_multi_light_shading(gray),
            'occlusion': extract_occlusion_boundaries(gray),
            'shadow': extract_shadow_depth(gray),
            'defocus': extract_defocus_depth(gray),
            'vanishing': extract_vanishing_point_depth(gray),
            'vertical': np.tile(np.linspace(1, 0, gray.shape[0]).reshape(-1, 1), (1, gray.shape[1])),
        }
        
        target_shape = gray.shape
        if depth.shape != target_shape:
            pil = Image.fromarray((depth * 255).astype(np.uint8))
            pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            depth = np.array(pil).astype(np.float32) / 255.0
        
        total_complex = None
        for name, mech_val in mechs.items():
            w = mechanism_weights[name]
            phase = phase_adjustments[name]
            complex_val = mech_val * np.exp(1j * phase)
            
            if total_complex is None:
                total_complex = w * complex_val
            else:
                total_complex = total_complex + w * complex_val
        
        pred = _normalize(gaussian_filter(np.abs(total_complex), sigma=2.0))
        test_errors.append(np.mean(np.abs(pred - depth)))
    
    print(f"\n  Multi-Light Model Test MAE: {np.mean(test_errors):.4f}")
    print(f"  Previous best:              0.199")
    
    diff = np.mean(test_errors) - 0.199
    if diff < 0:
        print(f"\n  ✓ NEW BEST! Improvement: {-diff:.4f}")
    else:
        print(f"\n  Still {diff:.4f} behind")
    
    return mechanism_weights, phase_adjustments


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PART 1: Basic Physical Mechanisms")
    print("="*70)
    model = run_physical_model_experiment(n_train=50, n_test=10)
    
    print("\n" + "="*70)
    print("PART 2: Multi-Light Physical Model")
    print("="*70)
    weights, phases = run_multi_light_experiment(n_train=50, n_test=10)
