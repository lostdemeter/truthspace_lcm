#!/usr/bin/env python3
"""
TRUE Ambiguity Test: When Does Self-Assembly Actually Fail?

Previous tests showed 92-96% improvement because objects were spatially separated.
This test introduces TRUE ambiguity:

1. OCCLUSION: Objects overlap in 2D projection
2. TEXTURE: Same texture at different depths
3. RANDOM DEPTH: Color/position don't predict depth

The goal: Find the MINIMAL information needed to resolve ambiguity.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def generate_random_depth_scene(width: int = 128, height: int = 128, seed: int = None):
    """
    Generate a scene where color/position do NOT predict depth.
    
    This is the TRUE test: can self-assembly work when there's
    no geometric relationship between appearance and depth?
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random RGB image
    rgb = np.random.rand(height, width, 3) * 0.5 + 0.25
    
    # Smooth it to look more natural
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        rgb[:,:,c] = gaussian_filter(rgb[:,:,c], sigma=5)
    
    # Random depth (INDEPENDENT of RGB!)
    depth = np.random.rand(height, width)
    depth = gaussian_filter(depth, sigma=10)
    
    return {'rgb': rgb, 'depth': _normalize(depth)}


def generate_texture_at_depth_scene(width: int = 128, height: int = 128, seed: int = None):
    """
    Same texture appears at different depths.
    
    This tests: can self-assembly learn that texture doesn't predict depth?
    """
    if seed is not None:
        np.random.seed(seed)
    
    from scipy.ndimage import gaussian_filter
    
    # Create a repeating texture
    texture = np.zeros((height, width, 3))
    for i in range(0, width, 16):
        for j in range(0, height, 16):
            color = np.random.rand(3) * 0.5 + 0.25
            texture[j:j+16, i:i+16] = color
    
    # Smooth slightly
    for c in range(3):
        texture[:,:,c] = gaussian_filter(texture[:,:,c], sigma=1)
    
    # Depth is based on REGIONS, not texture
    depth = np.zeros((height, width))
    n_regions = np.random.randint(3, 6)
    for _ in range(n_regions):
        cx, cy = np.random.randint(0, width), np.random.randint(0, height)
        r = np.random.randint(20, 50)
        d = np.random.rand()
        
        y, x = np.ogrid[:height, :width]
        mask = (x - cx)**2 + (y - cy)**2 < r**2
        depth[mask] = d
    
    depth = gaussian_filter(depth, sigma=3)
    
    return {'rgb': texture, 'depth': _normalize(depth)}


def generate_occlusion_scene(width: int = 128, height: int = 128, seed: int = None):
    """
    Objects occlude each other - same 2D position, different depths.
    
    This is the hardest case: position doesn't disambiguate.
    """
    if seed is not None:
        np.random.seed(seed)
    
    rgb = np.ones((height, width, 3)) * 0.5  # Gray background
    depth = np.ones((height, width)) * 0.9  # Far background
    
    # Add overlapping circles at different depths
    n_circles = np.random.randint(4, 8)
    circles = []
    
    for _ in range(n_circles):
        cx = np.random.randint(20, width - 20)
        cy = np.random.randint(20, height - 20)
        r = np.random.randint(15, 40)
        d = np.random.rand() * 0.8 + 0.1  # Depth 0.1 to 0.9
        color = np.random.rand(3) * 0.6 + 0.2
        circles.append((cx, cy, r, d, color))
    
    # Sort by depth (far to near) so near objects occlude far ones
    circles.sort(key=lambda c: -c[3])
    
    for cx, cy, r, d, color in circles:
        y, x = np.ogrid[:height, :width]
        mask = (x - cx)**2 + (y - cy)**2 < r**2
        rgb[mask] = color
        depth[mask] = d
    
    return {'rgb': rgb, 'depth': _normalize(depth)}


def test_ambiguity_level(generate_fn, name: str, n_train: int = 30, n_test: int = 10):
    """Test self-assembly on a specific ambiguity type."""
    
    # Generate training data
    train_X = []
    train_y = []
    
    for i in range(n_train):
        data = generate_fn(seed=i)
        h, w = data['depth'].shape
        
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
        
        features = np.stack([
            data['rgb'][:,:,0],
            data['rgb'][:,:,1],
            data['rgb'][:,:,2],
            y_coords,
            x_coords,
        ], axis=2)
        
        train_X.append(features.reshape(-1, 5))
        train_y.append(data['depth'].flatten())
    
    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    
    # Fit linear model
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(np.column_stack([X, np.ones(len(y))]), y, rcond=None)
    
    # Test
    test_errors = []
    vertical_errors = []
    random_errors = []
    
    for i in range(n_test):
        data = generate_fn(seed=1000 + i)
        h, w = data['depth'].shape
        
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
        
        features = np.stack([
            data['rgb'][:,:,0],
            data['rgb'][:,:,1],
            data['rgb'][:,:,2],
            y_coords,
            x_coords,
        ], axis=2).reshape(-1, 5)
        
        pred = features @ coeffs[:-1] + coeffs[-1]
        true_depth = data['depth'].flatten()
        
        test_errors.append(np.mean(np.abs(pred - true_depth)))
        vertical_errors.append(np.mean(np.abs(y_coords.flatten() - true_depth)))
        random_errors.append(np.mean(np.abs(np.random.rand(len(true_depth)) - true_depth)))
    
    return {
        'name': name,
        'mae_vertical': np.mean(vertical_errors),
        'mae_rgb': np.mean(test_errors),
        'mae_random': np.mean(random_errors),
        'improvement': (np.mean(vertical_errors) - np.mean(test_errors)) / np.mean(vertical_errors) * 100
    }


def run_ambiguity_experiment():
    """Test all ambiguity types."""
    print("=" * 70)
    print("TRUE AMBIGUITY TEST: When Does Self-Assembly Actually Fail?")
    print("=" * 70)
    print()
    
    tests = [
        (generate_random_depth_scene, "Random Depth (no correlation)"),
        (generate_texture_at_depth_scene, "Texture at Random Depths"),
        (generate_occlusion_scene, "Overlapping Objects"),
    ]
    
    results = []
    
    for generate_fn, name in tests:
        print(f"Testing: {name}...")
        result = test_ambiguity_level(generate_fn, name)
        results.append(result)
        
        print(f"  Vertical MAE: {result['mae_vertical']:.4f}")
        print(f"  RGB+XY MAE:   {result['mae_rgb']:.4f}")
        print(f"  Random MAE:   {result['mae_random']:.4f}")
        print(f"  Improvement:  {result['improvement']:+.1f}%")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: True Ambiguity Results")
    print("=" * 70)
    print()
    print(f"{'Test':<35} {'Vertical':<10} {'RGB+XY':<10} {'Improve':<10}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['name']:<35} {r['mae_vertical']:.4f}     {r['mae_rgb']:.4f}     {r['improvement']:+.1f}%")
    
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    for r in results:
        if r['improvement'] < 10:
            print(f"FAILURE: {r['name']}")
            print(f"  → Self-assembly CANNOT resolve this ambiguity!")
            print(f"  → RGB+position don't predict depth here.")
            print()
    
    return results


def create_ambiguity_visualization():
    """Visualize each ambiguity type."""
    
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('TRUE Ambiguity: When Self-Assembly Fails\n'
                 'Testing Cases Where Color/Position Don\'t Predict Depth',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.2)
    
    tests = [
        (generate_random_depth_scene, "Random Depth"),
        (generate_texture_at_depth_scene, "Texture at Depths"),
        (generate_occlusion_scene, "Occlusion"),
    ]
    
    for row, (generate_fn, title) in enumerate(tests):
        data = generate_fn(seed=42)
        h, w = data['depth'].shape
        
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(data['rgb'])
        ax1.set_title(title, fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(data['depth'], cmap='magma')
        if row == 0:
            ax2.set_title('True Depth', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(y_coords, cmap='magma')
        if row == 0:
            ax3.set_title('Vertical Baseline', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[row, 3])
        # Correlation between RGB and depth
        rgb_flat = data['rgb'].reshape(-1, 3)
        depth_flat = data['depth'].flatten()
        corr_r = np.corrcoef(rgb_flat[:,0], depth_flat)[0,1]
        corr_g = np.corrcoef(rgb_flat[:,1], depth_flat)[0,1]
        corr_b = np.corrcoef(rgb_flat[:,2], depth_flat)[0,1]
        
        ax4.bar(['R', 'G', 'B'], [corr_r, corr_g, corr_b], color=['red', 'green', 'blue'])
        ax4.set_ylim(-1, 1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if row == 0:
            ax4.set_title('Color-Depth Corr', fontsize=10)
        ax4.set_ylabel('Correlation')
    
    output_file = OUTPUT_PATH / "true_ambiguity_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Run ambiguity experiment
    results = run_ambiguity_experiment()
    
    # Create visualization
    viz_file = create_ambiguity_visualization()
    
    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("When color/position DON'T predict depth, self-assembly FAILS.")
    print()
    print("This is the fundamental limit:")
    print("  - Self-assembly can only discover relationships that EXIST in the data")
    print("  - If depth is independent of appearance, there's nothing to discover")
    print("  - Real photos have PARTIAL correlation (some signal, lots of noise)")
    print()
    print("The question becomes:")
    print("  What ADDITIONAL information would resolve the ambiguity?")
    print("  - Motion parallax (video)")
    print("  - Stereo disparity (two cameras)")
    print("  - Focus/defocus (depth of field)")
    print("  - Semantic understanding (knowing what things ARE)")
