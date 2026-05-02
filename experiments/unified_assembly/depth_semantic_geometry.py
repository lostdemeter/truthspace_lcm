#!/usr/bin/env python3
"""
DEEP DIVE: Can We Discover Semantic Priors Geometrically?

The Question:
- DA learned "sky is far", "face is close" from millions of labeled images
- Can we discover these same relationships from GEOMETRY alone?

Three Hypotheses to Test:

1. COLOR-DEPTH CORRELATIONS
   - Blue wavelength is shorter → scatters more → "blue = far" (Rayleigh scattering)
   - Warm colors absorb less atmosphere → "warm = close"
   - This is PHYSICS, not learned correlation!

2. OBJECT BOUNDARIES VIA SELF-SIMILARITY
   - Objects are regions of self-similar texture
   - Boundaries are where self-similarity breaks
   - This is GEOMETRIC, not semantic!

3. SEMANTIC CATEGORIES FROM GEOMETRIC SIGNATURES
   - "Sky" has high blue, low texture, at top → geometric signature
   - "Face" has specific color distribution, texture pattern → geometric signature
   - Categories EMERGE from geometric clustering, not labels!

The TruthSpace Hypothesis:
If semantic priors are fundamentally geometric, they should EMERGE from
self-assembly without training data. The structure should encode the
navigation rules.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.spatial.distance import cdist
# KMeans removed - not needed for core experiment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
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


# =============================================================================
# HYPOTHESIS 1: Color-Depth from Physics (Rayleigh Scattering)
# =============================================================================

def rayleigh_depth(rgb: np.ndarray) -> np.ndarray:
    """
    Derive depth from Rayleigh scattering physics.
    
    Rayleigh scattering: intensity ∝ 1/λ⁴
    - Blue (λ≈450nm) scatters most → dominates distant objects
    - Red (λ≈700nm) scatters least → dominates close objects
    
    The ratio R/B encodes atmospheric depth:
    - High R/B → close (less scattering)
    - Low R/B → far (more blue scattering)
    
    This is PHYSICS, not a learned correlation!
    """
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    
    # Avoid division by zero
    B = np.maximum(B, 0.01)
    
    # Rayleigh ratio: R/B indicates atmospheric depth
    # High R/B = close, Low R/B = far
    rayleigh_ratio = R / B
    
    # Normalize and invert (high ratio = close = low depth value)
    depth = 1 - _normalize(rayleigh_ratio)
    
    return depth


def chromatic_aberration_depth(rgb: np.ndarray) -> np.ndarray:
    """
    Derive depth from chromatic aberration physics.
    
    Different wavelengths focus at different distances:
    - Blue focuses closer to lens
    - Red focuses farther from lens
    
    The blur difference between R and B channels encodes depth:
    - Sharp R, blurry B → close
    - Sharp B, blurry R → far
    
    This is OPTICS, not a learned correlation!
    """
    R = rgb[:, :, 0].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    
    # Compute local sharpness (gradient magnitude)
    R_sharp = np.sqrt(sobel(R, axis=0)**2 + sobel(R, axis=1)**2)
    B_sharp = np.sqrt(sobel(B, axis=0)**2 + sobel(B, axis=1)**2)
    
    # Smooth to get local average
    R_sharp = gaussian_filter(R_sharp, sigma=3)
    B_sharp = gaussian_filter(B_sharp, sigma=3)
    
    # Avoid division by zero
    total_sharp = R_sharp + B_sharp + 0.01
    
    # Ratio of sharpness: high B_sharp/R_sharp = far
    depth = B_sharp / total_sharp
    
    return _normalize(depth)


# =============================================================================
# HYPOTHESIS 2: Object Boundaries via Self-Similarity
# =============================================================================

def self_similarity_boundaries(rgb: np.ndarray, patch_size: int = 8) -> np.ndarray:
    """
    Detect object boundaries via self-similarity breakdown.
    
    Within an object, patches are self-similar.
    At boundaries, self-similarity breaks.
    
    This is GEOMETRIC - no semantic labels needed!
    """
    h, w = rgb.shape[:2]
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # Compute local variance (measure of texture)
    local_mean = uniform_filter(gray, size=patch_size)
    local_sq_mean = uniform_filter(gray**2, size=patch_size)
    local_var = local_sq_mean - local_mean**2
    
    # Compute gradient of local variance (boundaries = high gradient)
    var_grad = np.sqrt(sobel(local_var, axis=0)**2 + sobel(local_var, axis=1)**2)
    
    # Boundaries are where variance changes rapidly
    boundaries = _normalize(var_grad)
    
    return boundaries


def texture_coherence_regions(rgb: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Segment image into coherent texture regions.
    
    Objects are regions of coherent texture.
    This emerges from geometry, not semantic labels!
    """
    h, w = rgb.shape[:2]
    
    # Extract texture features at each pixel
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # Local statistics as texture features
    local_mean = uniform_filter(gray, size=8)
    local_var = uniform_filter(gray**2, size=8) - local_mean**2
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Stack features
    features = np.stack([
        local_mean.flatten(),
        np.sqrt(local_var.flatten() + 0.001),
        grad_mag.flatten(),
        rgb[:,:,0].flatten(),  # Color
        rgb[:,:,1].flatten(),
        rgb[:,:,2].flatten(),
    ], axis=1)
    
    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 0.001)
    
    # Cluster into regions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    labels = kmeans.fit_predict(features)
    
    return labels.reshape(h, w)


# =============================================================================
# HYPOTHESIS 3: Semantic Categories from Geometric Signatures
# =============================================================================

def geometric_sky_detector(rgb: np.ndarray) -> np.ndarray:
    """
    Detect "sky" from geometric signature alone.
    
    Sky signature (no semantic label needed):
    - High blue channel
    - Low texture (smooth)
    - At top of image
    - High brightness
    
    This is a GEOMETRIC SIGNATURE, not a learned category!
    """
    h, w = rgb.shape[:2]
    
    # Blue dominance
    blue_dom = rgb[:,:,2] / (rgb.sum(axis=2) + 0.01)
    
    # Low texture (smooth regions)
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    texture = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
    smoothness = 1 - _normalize(texture)
    
    # Vertical position (top = more likely sky)
    y_coords = np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
    
    # Brightness
    brightness = gray
    
    # Combine: sky = blue + smooth + top + bright
    sky_score = (
        0.3 * blue_dom + 
        0.3 * smoothness + 
        0.2 * y_coords + 
        0.2 * brightness
    )
    
    return _normalize(sky_score)


def geometric_ground_detector(rgb: np.ndarray) -> np.ndarray:
    """
    Detect "ground" from geometric signature alone.
    
    Ground signature:
    - At bottom of image
    - Often brown/green (earth tones)
    - Medium texture
    - Horizontal gradients (perspective lines)
    """
    h, w = rgb.shape[:2]
    
    # Vertical position (bottom = more likely ground)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    
    # Earth tones (green/brown)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    earth_tone = (G + 0.5 * R) / (R + G + B + 0.01)
    
    # Horizontal gradient dominance (perspective lines)
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    grad_x = np.abs(sobel(gray, axis=1))
    grad_y = np.abs(sobel(gray, axis=0))
    horiz_dom = grad_x / (grad_x + grad_y + 0.01)
    
    # Combine
    ground_score = (
        0.4 * y_coords + 
        0.3 * earth_tone + 
        0.3 * horiz_dom
    )
    
    return _normalize(ground_score)


def geometric_object_detector(rgb: np.ndarray) -> np.ndarray:
    """
    Detect "objects" (foreground) from geometric signature.
    
    Object signature:
    - High texture (detail)
    - Color contrast with surroundings
    - Often in middle of image
    - Distinct boundaries
    """
    h, w = rgb.shape[:2]
    
    # High texture
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    texture = np.sqrt(sobel(gray, axis=0)**2 + sobel(gray, axis=1)**2)
    texture = gaussian_filter(texture, sigma=2)
    
    # Color saliency (difference from local mean)
    local_color = np.stack([
        uniform_filter(rgb[:,:,c], size=20) for c in range(3)
    ], axis=2)
    color_diff = np.sqrt(((rgb - local_color)**2).sum(axis=2))
    
    # Center bias (objects often in middle)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
    center_dist = np.sqrt((y_coords - 0.5)**2 + (x_coords - 0.5)**2)
    center_bias = 1 - center_dist / center_dist.max()
    
    # Combine
    object_score = (
        0.4 * _normalize(texture) + 
        0.4 * _normalize(color_diff) + 
        0.2 * center_bias
    )
    
    return _normalize(object_score)


# =============================================================================
# COMBINED GEOMETRIC SEMANTIC DEPTH
# =============================================================================

def geometric_semantic_depth(rgb: np.ndarray) -> np.ndarray:
    """
    Combine all geometric semantic detectors into depth estimate.
    
    This uses NO training data - only geometric signatures!
    """
    h, w = rgb.shape[:2]
    
    # Detect semantic regions geometrically
    sky = geometric_sky_detector(rgb)
    ground = geometric_ground_detector(rgb)
    objects = geometric_object_detector(rgb)
    
    # Physics-based depth
    rayleigh = rayleigh_depth(rgb)
    chromatic = chromatic_aberration_depth(rgb)
    
    # Vertical baseline
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    vertical = 0.6 * y_coords + 0.1
    
    # Combine: weight by confidence
    # Sky = far (depth = 1)
    # Ground = varies with y
    # Objects = close (depth = 0)
    
    depth = (
        0.3 * vertical +           # Geometric baseline
        0.2 * sky +                # Sky = far
        0.1 * (1 - objects) +      # Objects = close (invert)
        0.2 * rayleigh +           # Physics: Rayleigh
        0.2 * chromatic            # Physics: chromatic aberration
    )
    
    return _normalize(depth)


# =============================================================================
# EXPERIMENT: Test Each Hypothesis
# =============================================================================

def run_semantic_geometry_experiment(n_images: int = 30):
    """Test if semantic priors can emerge from geometry."""
    
    print("=" * 70)
    print("DEEP DIVE: Can We Discover Semantic Priors Geometrically?")
    print("=" * 70)
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect errors for each approach
    results = {
        'vertical': [],
        'rayleigh': [],
        'chromatic': [],
        'sky_detector': [],
        'ground_detector': [],
        'object_detector': [],
        'combined_geometric': [],
    }
    
    for i, img_id in enumerate(available_ids[:n_images]):
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
        
        # Physics-based
        rayleigh = rayleigh_depth(rgb_small)
        chromatic = chromatic_aberration_depth(rgb_small)
        
        # Geometric semantic detectors
        sky = geometric_sky_detector(rgb_small)
        ground = geometric_ground_detector(rgb_small)
        objects = geometric_object_detector(rgb_small)
        
        # Combined
        combined = geometric_semantic_depth(rgb_small)
        
        # Compute errors vs DA (semantic ground truth)
        results['vertical'].append(np.mean(np.abs(vertical - da_depth_small)))
        results['rayleigh'].append(np.mean(np.abs(rayleigh - da_depth_small)))
        results['chromatic'].append(np.mean(np.abs(chromatic - da_depth_small)))
        results['sky_detector'].append(np.mean(np.abs(sky - da_depth_small)))
        results['ground_detector'].append(np.mean(np.abs(ground - da_depth_small)))
        results['object_detector'].append(np.mean(np.abs((1-objects) - da_depth_small)))
        results['combined_geometric'].append(np.mean(np.abs(combined - da_depth_small)))
    
    # Print results
    print("=" * 60)
    print("RESULTS: MAE vs Depth Anything (Semantic Ground Truth)")
    print("=" * 60)
    print()
    
    for name, errors in sorted(results.items(), key=lambda x: np.mean(x[1])):
        print(f"  {name:<25}: {np.mean(errors):.4f}")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    
    best = min(results.items(), key=lambda x: np.mean(x[1]))
    print(f"Best geometric approach: {best[0]} (MAE = {np.mean(best[1]):.4f})")
    print()
    
    vertical_mae = np.mean(results['vertical'])
    combined_mae = np.mean(results['combined_geometric'])
    
    if combined_mae < vertical_mae:
        improvement = (vertical_mae - combined_mae) / vertical_mae * 100
        print(f"Combined geometric IMPROVES on vertical by {improvement:.1f}%!")
        print("→ Semantic priors CAN emerge from geometry!")
    else:
        print("Combined geometric does NOT improve on vertical.")
        print("→ Semantic priors may require training data.")
    
    return results


def create_semantic_geometry_visualization(n_images: int = 3):
    """Visualize the geometric semantic detectors."""
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    fig = plt.figure(figsize=(24, 6 * n_images))
    fig.suptitle('Can Semantic Priors Emerge from Geometry?\n'
                 'Testing Physics-Based and Geometric Signature Approaches',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_images, 8, figure=fig, hspace=0.3, wspace=0.15)
    
    for row, img_id in enumerate(available_ids[:n_images]):
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
        
        # Compute all detectors
        rayleigh = rayleigh_depth(rgb_small)
        sky = geometric_sky_detector(rgb_small)
        ground = geometric_ground_detector(rgb_small)
        objects = geometric_object_detector(rgb_small)
        combined = geometric_semantic_depth(rgb_small)
        
        # Plot
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(rgb_small)
        ax1.set_title('Original', fontsize=9)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(da_depth_small, cmap='magma')
        ax2.set_title('DA Depth\n(Semantic Truth)', fontsize=9)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(rayleigh, cmap='magma')
        mae = np.mean(np.abs(rayleigh - da_depth_small))
        ax3.set_title(f'Rayleigh Physics\n(MAE: {mae:.3f})', fontsize=9)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.imshow(sky, cmap='magma')
        ax4.set_title('Sky Detector\n(Geometric)', fontsize=9)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[row, 4])
        ax5.imshow(ground, cmap='magma')
        ax5.set_title('Ground Detector\n(Geometric)', fontsize=9)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[row, 5])
        ax6.imshow(objects, cmap='magma')
        ax6.set_title('Object Detector\n(Geometric)', fontsize=9)
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[row, 6])
        ax7.imshow(combined, cmap='magma')
        mae = np.mean(np.abs(combined - da_depth_small))
        ax7.set_title(f'Combined Geometric\n(MAE: {mae:.3f})', fontsize=9)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[row, 7])
        # Show where combined beats vertical
        y_coords = np.tile(np.linspace(0, 1, new_h).reshape(-1, 1), (1, new_w))
        vertical = _normalize(0.6 * y_coords + 0.1)
        combined_err = np.abs(combined - da_depth_small)
        vertical_err = np.abs(vertical - da_depth_small)
        improvement = vertical_err - combined_err
        ax8.imshow(improvement, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
        ax8.set_title('Improvement\n(Green=Geo Better)', fontsize=9)
        ax8.axis('off')
    
    output_file = OUTPUT_PATH / "semantic_geometry_deep_dive.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def analyze_what_emerges(n_images: int = 30):
    """
    Analyze what semantic structure EMERGES from geometry.
    
    Key question: Do geometric signatures correlate with DA's semantic regions?
    """
    print()
    print("=" * 70)
    print("ANALYSIS: What Semantic Structure Emerges from Geometry?")
    print("=" * 70)
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect correlations
    sky_corr = []
    ground_corr = []
    object_corr = []
    rayleigh_corr = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
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
        
        # Compute detectors
        sky = geometric_sky_detector(rgb_small)
        ground = geometric_ground_detector(rgb_small)
        objects = geometric_object_detector(rgb_small)
        rayleigh = rayleigh_depth(rgb_small)
        
        # Correlations with DA depth
        sky_corr.append(np.corrcoef(sky.flatten(), da_depth_small.flatten())[0, 1])
        ground_corr.append(np.corrcoef(ground.flatten(), da_depth_small.flatten())[0, 1])
        object_corr.append(np.corrcoef((1-objects).flatten(), da_depth_small.flatten())[0, 1])
        rayleigh_corr.append(np.corrcoef(rayleigh.flatten(), da_depth_small.flatten())[0, 1])
    
    print("Correlation with DA Depth (Semantic Ground Truth):")
    print("-" * 50)
    print(f"  Sky detector:      {np.mean(sky_corr):.3f}")
    print(f"  Ground detector:   {np.mean(ground_corr):.3f}")
    print(f"  Object detector:   {np.mean(object_corr):.3f}")
    print(f"  Rayleigh physics:  {np.mean(rayleigh_corr):.3f}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 50)
    
    if np.mean(sky_corr) > 0.3:
        print("✓ Sky detector correlates with DA depth!")
        print("  → 'Sky = far' CAN emerge from geometry (blue + smooth + top)")
    else:
        print("✗ Sky detector does NOT correlate well with DA depth")
        print("  → 'Sky = far' may require semantic understanding")
    
    if np.mean(rayleigh_corr) > 0.3:
        print("✓ Rayleigh physics correlates with DA depth!")
        print("  → Color-depth relationship IS physics, not learned!")
    else:
        print("✗ Rayleigh physics does NOT correlate well")
        print("  → Indoor scenes break the atmospheric assumption")
    
    return {
        'sky_corr': np.mean(sky_corr),
        'ground_corr': np.mean(ground_corr),
        'object_corr': np.mean(object_corr),
        'rayleigh_corr': np.mean(rayleigh_corr)
    }


if __name__ == "__main__":
    # Run main experiment
    results = run_semantic_geometry_experiment(n_images=30)
    
    # Analyze what emerges
    correlations = analyze_what_emerges(n_images=30)
    
    # Create visualization
    viz_file = create_semantic_geometry_visualization(n_images=3)
    
    print()
    print("=" * 70)
    print("CONCLUSION: Can Semantic Priors Emerge from Geometry?")
    print("=" * 70)
    print()
    print("The answer depends on WHICH semantic priors:")
    print()
    print("1. COLOR-DEPTH (Rayleigh scattering)")
    print(f"   Correlation: {correlations['rayleigh_corr']:.3f}")
    if correlations['rayleigh_corr'] > 0.3:
        print("   → YES! This is physics, not learned.")
    else:
        print("   → PARTIAL. Works outdoors, fails indoors.")
    print()
    print("2. SKY DETECTION (blue + smooth + top)")
    print(f"   Correlation: {correlations['sky_corr']:.3f}")
    if correlations['sky_corr'] > 0.3:
        print("   → YES! Geometric signature works.")
    else:
        print("   → NO. Requires semantic understanding.")
    print()
    print("3. OBJECT DETECTION (texture + saliency)")
    print(f"   Correlation: {correlations['object_corr']:.3f}")
    if correlations['object_corr'] > 0.3:
        print("   → YES! Objects have geometric signatures.")
    else:
        print("   → NO. Object = close is semantic, not geometric.")
