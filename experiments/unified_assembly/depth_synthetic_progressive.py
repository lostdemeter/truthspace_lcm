#!/usr/bin/env python3
"""
Progressive Complexity: Where Does Self-Assembly Break?

We start with a simple synthetic scene where color = depth,
then progressively add complexity to discover:

1. At what point does color-depth mapping become ambiguous?
2. What additional information would resolve the ambiguity?
3. What is the MINIMAL information needed for depth self-assembly?

Complexity Levels:
- Level 0: Each object has unique color (trivial)
- Level 1: Objects can share colors (color ambiguity)
- Level 2: Same object at different depths (depth ambiguity)
- Level 3: Varying lighting (lighting ambiguity)
- Level 4: Textured surfaces (texture ambiguity)
- Level 5: Occlusion (visibility ambiguity)

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray


class ProgressiveScene:
    """Scene with controllable complexity levels."""
    
    def __init__(self, width: int = 128, height: int = 128):
        self.width = width
        self.height = height
        self.objects = []
        self.light_dir = np.array([-1.0, -1.0, 1.0])
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        
    def add_sphere(self, center, radius, color):
        self.objects.append(Sphere(
            center=np.array(center),
            radius=radius,
            color=np.array(color)
        ))
    
    def render(self) -> dict:
        """Render with complete ground truth."""
        rgb = np.zeros((self.height, self.width, 3))
        depth = np.full((self.height, self.width), 100.0)
        
        # Sky background
        for y in range(self.height):
            for x in range(self.width):
                rgb[y, x] = np.array([0.5, 0.7, 1.0])
        
        # Ground plane at y = -1
        for y in range(self.height):
            for x in range(self.width):
                # Ray from camera
                ndc_x = (2 * x / self.width - 1) * 0.5
                ndc_y = (1 - 2 * y / self.height) * 0.5
                ray_dir = np.array([ndc_x, ndc_y, 1.0])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Ground plane intersection
                if ray_dir[1] < -0.001:
                    t = (-1.0 - 0) / ray_dir[1]
                    if t > 0 and t < depth[y, x]:
                        depth[y, x] = t
                        # Checkerboard pattern
                        hit = t * ray_dir
                        checker = (int(hit[0] * 2) + int(hit[2] * 2)) % 2
                        rgb[y, x] = np.array([0.3, 0.25, 0.2]) if checker else np.array([0.5, 0.45, 0.4])
        
        # Spheres
        for obj in self.objects:
            for y in range(self.height):
                for x in range(self.width):
                    ndc_x = (2 * x / self.width - 1) * 0.5
                    ndc_y = (1 - 2 * y / self.height) * 0.5
                    ray_dir = np.array([ndc_x, ndc_y, 1.0])
                    ray_dir = ray_dir / np.linalg.norm(ray_dir)
                    
                    # Ray-sphere intersection
                    oc = -obj.center
                    a = np.dot(ray_dir, ray_dir)
                    b = 2.0 * np.dot(oc, ray_dir)
                    c = np.dot(oc, oc) - obj.radius * obj.radius
                    disc = b * b - 4 * a * c
                    
                    if disc >= 0:
                        t = (-b - np.sqrt(disc)) / (2.0 * a)
                        if t > 0.001 and t < depth[y, x]:
                            depth[y, x] = t
                            hit = t * ray_dir
                            normal = (hit - obj.center) / obj.radius
                            # Diffuse shading
                            ndotl = max(0, np.dot(normal, -self.light_dir))
                            rgb[y, x] = obj.color * (0.2 + 0.8 * ndotl)
        
        return {'rgb': np.clip(rgb, 0, 1), 'depth': depth}


def create_level_0_scene():
    """Level 0: Unique colors (trivial mapping)."""
    scene = ProgressiveScene()
    scene.add_sphere((0, 0, 5), 0.8, (1.0, 0.2, 0.2))  # Red
    scene.add_sphere((1.2, 0.3, 7), 0.6, (0.2, 1.0, 0.2))  # Green
    scene.add_sphere((-1.0, -0.2, 9), 0.7, (0.2, 0.2, 1.0))  # Blue
    return scene


def create_level_1_scene():
    """Level 1: Shared colors (color ambiguity)."""
    scene = ProgressiveScene()
    # Two red spheres at different depths!
    scene.add_sphere((0, 0, 4), 0.6, (1.0, 0.2, 0.2))  # Red, close
    scene.add_sphere((1.0, 0.2, 8), 0.8, (1.0, 0.2, 0.2))  # Red, far
    scene.add_sphere((-0.8, -0.1, 6), 0.5, (0.2, 0.8, 0.2))  # Green
    return scene


def create_level_2_scene():
    """Level 2: Same object type at many depths."""
    scene = ProgressiveScene()
    # Many similar spheres at different depths
    np.random.seed(42)
    for i in range(6):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-0.3, 0.5)
        z = 4 + i * 1.5  # Increasing depth
        r = np.random.uniform(0.3, 0.6)
        # All similar grayish colors
        gray = np.random.uniform(0.4, 0.6)
        scene.add_sphere((x, y, z), r, (gray, gray, gray))
    return scene


def create_level_3_scene():
    """Level 3: Varying lighting direction."""
    scene = ProgressiveScene()
    scene.add_sphere((0, 0, 5), 0.8, (0.8, 0.8, 0.8))
    scene.add_sphere((1.2, 0.3, 7), 0.6, (0.8, 0.8, 0.8))
    # Change light direction
    scene.light_dir = np.array([1.0, -0.5, 0.5])
    scene.light_dir = scene.light_dir / np.linalg.norm(scene.light_dir)
    return scene


def test_self_assembly_at_level(level: int, n_train: int = 20, n_test: int = 10):
    """Test self-assembly at a given complexity level."""
    
    create_funcs = {
        0: create_level_0_scene,
        1: create_level_1_scene,
        2: create_level_2_scene,
        3: create_level_3_scene,
    }
    
    create_fn = create_funcs.get(level, create_level_0_scene)
    
    # Generate training data
    train_X = []
    train_y = []
    
    for _ in range(n_train):
        scene = create_fn()
        # Add some randomness
        for obj in scene.objects:
            obj.center = obj.center + np.random.randn(3) * 0.2
        
        data = scene.render()
        h, w = data['depth'].shape
        
        # Features: RGB + vertical
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        features = np.stack([
            data['rgb'][:,:,0],
            data['rgb'][:,:,1],
            data['rgb'][:,:,2],
            y_coords,
        ], axis=2)
        
        train_X.append(features.reshape(-1, 4))
        train_y.append(_normalize(data['depth']).flatten())
    
    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    
    # Fit linear model
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(np.column_stack([X, np.ones(len(y))]), y, rcond=None)
    
    # Test
    test_errors = []
    vertical_errors = []
    
    for _ in range(n_test):
        scene = create_fn()
        for obj in scene.objects:
            obj.center = obj.center + np.random.randn(3) * 0.2
        
        data = scene.render()
        h, w = data['depth'].shape
        
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        features = np.stack([
            data['rgb'][:,:,0],
            data['rgb'][:,:,1],
            data['rgb'][:,:,2],
            y_coords,
        ], axis=2).reshape(-1, 4)
        
        pred = features @ coeffs[:-1] + coeffs[-1]
        true_depth = _normalize(data['depth']).flatten()
        
        test_errors.append(np.mean(np.abs(pred - true_depth)))
        vertical_errors.append(np.mean(np.abs(y_coords.flatten() - true_depth)))
    
    return {
        'level': level,
        'mae_vertical': np.mean(vertical_errors),
        'mae_rgb': np.mean(test_errors),
        'improvement': (np.mean(vertical_errors) - np.mean(test_errors)) / np.mean(vertical_errors) * 100
    }


def run_progressive_experiment():
    """Test all complexity levels."""
    print("=" * 70)
    print("PROGRESSIVE COMPLEXITY: Where Does Self-Assembly Break?")
    print("=" * 70)
    print()
    
    results = []
    
    for level in range(4):
        level_names = {
            0: "Unique colors (trivial)",
            1: "Shared colors (ambiguity)",
            2: "Many similar objects",
            3: "Varying lighting",
        }
        
        print(f"Testing Level {level}: {level_names[level]}...")
        result = test_self_assembly_at_level(level)
        results.append(result)
        
        print(f"  Vertical MAE: {result['mae_vertical']:.4f}")
        print(f"  RGB+V MAE:    {result['mae_rgb']:.4f}")
        print(f"  Improvement:  {result['improvement']:.1f}%")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY: How Complexity Affects Self-Assembly")
    print("=" * 60)
    print()
    print(f"{'Level':<30} {'Vertical':<12} {'RGB+V':<12} {'Improvement':<12}")
    print("-" * 66)
    
    for r in results:
        level_names = {
            0: "Unique colors",
            1: "Shared colors",
            2: "Many similar objects",
            3: "Varying lighting",
        }
        name = level_names[r['level']]
        print(f"{name:<30} {r['mae_vertical']:.4f}       {r['mae_rgb']:.4f}       {r['improvement']:+.1f}%")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    
    # Find where it breaks
    for i, r in enumerate(results):
        if r['improvement'] < 50:
            print(f"Self-assembly degrades at Level {r['level']}!")
            print(f"  This is where color-depth mapping becomes ambiguous.")
            break
    else:
        print("Self-assembly works at all tested levels!")
    
    return results


def create_progressive_visualization():
    """Visualize each complexity level."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Progressive Complexity: Where Does Self-Assembly Break?\n'
                 'Testing Color-Depth Mapping Ambiguity',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.2)
    
    create_funcs = [
        (create_level_0_scene, "Level 0: Unique Colors"),
        (create_level_1_scene, "Level 1: Shared Colors"),
        (create_level_2_scene, "Level 2: Similar Objects"),
        (create_level_3_scene, "Level 3: Varying Light"),
    ]
    
    for row, (create_fn, title) in enumerate(create_funcs):
        scene = create_fn()
        data = scene.render()
        h, w = data['depth'].shape
        
        # Vertical baseline
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        vertical = _normalize(y_coords)
        
        # RGB
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(data['rgb'])
        ax1.set_title(title if row == 0 else f"Level {row}", fontsize=10)
        ax1.axis('off')
        if row == 0:
            ax1.set_ylabel('RGB', fontsize=10)
        
        # True depth
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(_normalize(data['depth']), cmap='magma')
        if row == 0:
            ax2.set_title('True Depth', fontsize=10)
        ax2.axis('off')
        
        # Vertical baseline
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(vertical, cmap='magma')
        if row == 0:
            ax3.set_title('Vertical Baseline', fontsize=10)
        ax3.axis('off')
        
        # Error
        ax4 = fig.add_subplot(gs[row, 3])
        error = np.abs(_normalize(data['depth']) - vertical)
        ax4.imshow(error, cmap='hot')
        mae = np.mean(error)
        if row == 0:
            ax4.set_title(f'Error (MAE: {mae:.3f})', fontsize=10)
        else:
            ax4.set_title(f'MAE: {mae:.3f}', fontsize=10)
        ax4.axis('off')
    
    output_file = OUTPUT_PATH / "progressive_complexity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Run progressive experiment
    results = run_progressive_experiment()
    
    # Create visualization
    viz_file = create_progressive_visualization()
    
    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("The progression reveals WHERE ambiguity enters:")
    print()
    print("  Level 0: Color uniquely identifies objects → depth is trivial")
    print("  Level 1: Same color, different depths → need POSITION to disambiguate")
    print("  Level 2: Many similar objects → need CONTEXT to disambiguate")
    print("  Level 3: Varying lighting → need LIGHT SOURCE to disambiguate")
    print()
    print("Each level of ambiguity requires additional STRUCTURE to resolve.")
    print("This is what self-assembly must discover from the data.")
