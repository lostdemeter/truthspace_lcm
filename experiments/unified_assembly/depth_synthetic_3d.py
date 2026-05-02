#!/usr/bin/env python3
"""
Synthetic 3D Environment for Depth Self-Assembly

The Problem with Photographs:
- A photo is a 2D projection of 3D reality
- Light interference patterns have been collapsed
- We can't know: light direction, surface normals, material properties
- Too many unknowns to solve geometrically

The Solution: Synthetic 3D with Complete Ground Truth
- We CREATE the 3D scene programmatically
- We KNOW exact depth at every pixel
- We KNOW surface normals, lighting, materials
- We can discover what information self-assembly NEEDS

What We Can Learn:
1. Which geometric features actually predict depth?
2. What information is lost in the 2D projection?
3. What would a self-assembler need to recover depth?
4. Are there geometric invariants that survive projection?

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


@dataclass
class Light:
    """Light source in 3D space."""
    direction: np.ndarray  # Normalized direction vector
    color: np.ndarray      # RGB color
    intensity: float       # Brightness
    
    def __post_init__(self):
        self.direction = self.direction / np.linalg.norm(self.direction)


@dataclass
class Material:
    """Surface material properties."""
    albedo: np.ndarray     # Base color (RGB)
    roughness: float       # 0 = mirror, 1 = diffuse
    metallic: float        # 0 = dielectric, 1 = metal


@dataclass 
class Sphere:
    """Sphere primitive."""
    center: np.ndarray
    radius: float
    material: Material


@dataclass
class Plane:
    """Infinite plane primitive."""
    point: np.ndarray      # Point on plane
    normal: np.ndarray     # Normal vector
    material: Material
    
    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)


class SyntheticScene:
    """
    A synthetic 3D scene with complete ground truth.
    
    We can render:
    - RGB image (what a camera sees)
    - Depth map (exact distance to camera)
    - Normal map (surface orientation)
    - Lighting contribution (how much light hits each point)
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.objects = []
        self.lights = []
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_dir = np.array([0.0, 0.0, 1.0])
        self.fov = 60.0  # degrees
        
    def add_sphere(self, center: Tuple[float, float, float], radius: float,
                   color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                   roughness: float = 0.5):
        material = Material(
            albedo=np.array(color),
            roughness=roughness,
            metallic=0.0
        )
        self.objects.append(Sphere(
            center=np.array(center),
            radius=radius,
            material=material
        ))
    
    def add_plane(self, point: Tuple[float, float, float],
                  normal: Tuple[float, float, float],
                  color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                  roughness: float = 0.8):
        material = Material(
            albedo=np.array(color),
            roughness=roughness,
            metallic=0.0
        )
        self.objects.append(Plane(
            point=np.array(point),
            normal=np.array(normal),
            material=material
        ))
    
    def add_light(self, direction: Tuple[float, float, float],
                  color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  intensity: float = 1.0):
        self.lights.append(Light(
            direction=np.array(direction),
            color=np.array(color),
            intensity=intensity
        ))
    
    def _ray_sphere_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                               sphere: Sphere) -> Tuple[float, np.ndarray]:
        """Ray-sphere intersection. Returns (t, normal) or (inf, None)."""
        oc = ray_origin - sphere.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return np.inf, None
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            return np.inf, None
        
        hit_point = ray_origin + t * ray_dir
        normal = (hit_point - sphere.center) / sphere.radius
        return t, normal
    
    def _ray_plane_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                              plane: Plane) -> Tuple[float, np.ndarray]:
        """Ray-plane intersection. Returns (t, normal) or (inf, None)."""
        denom = np.dot(plane.normal, ray_dir)
        if abs(denom) < 1e-6:
            return np.inf, None
        
        t = np.dot(plane.point - ray_origin, plane.normal) / denom
        if t < 0.001:
            return np.inf, None
        
        return t, plane.normal
    
    def _trace_ray(self, ray_origin: np.ndarray, ray_dir: np.ndarray):
        """Trace a ray and return hit info."""
        closest_t = np.inf
        closest_normal = None
        closest_material = None
        
        for obj in self.objects:
            if isinstance(obj, Sphere):
                t, normal = self._ray_sphere_intersect(ray_origin, ray_dir, obj)
            elif isinstance(obj, Plane):
                t, normal = self._ray_plane_intersect(ray_origin, ray_dir, obj)
            else:
                continue
            
            if t < closest_t:
                closest_t = t
                closest_normal = normal
                closest_material = obj.material
        
        return closest_t, closest_normal, closest_material
    
    def _shade(self, hit_point: np.ndarray, normal: np.ndarray, 
               material: Material) -> np.ndarray:
        """Compute shading at a point."""
        color = np.zeros(3)
        
        # Ambient
        ambient = 0.1
        color += ambient * material.albedo
        
        # Diffuse from each light
        for light in self.lights:
            # Lambert diffuse
            ndotl = max(0, np.dot(normal, -light.direction))
            diffuse = ndotl * light.intensity
            color += diffuse * material.albedo * light.color
        
        return np.clip(color, 0, 1)
    
    def render(self) -> dict:
        """
        Render the scene and return complete ground truth.
        
        Returns dict with:
        - 'rgb': RGB image
        - 'depth': Depth map (distance to camera)
        - 'normals': Surface normal map
        - 'lighting': Lighting contribution map
        """
        rgb = np.zeros((self.height, self.width, 3))
        depth = np.full((self.height, self.width), np.inf)
        normals = np.zeros((self.height, self.width, 3))
        lighting = np.zeros((self.height, self.width))
        
        # Compute ray directions for each pixel
        aspect = self.width / self.height
        fov_rad = np.radians(self.fov)
        
        for y in range(self.height):
            for x in range(self.width):
                # Normalized device coordinates
                ndc_x = (2 * x / self.width - 1) * aspect * np.tan(fov_rad / 2)
                ndc_y = (1 - 2 * y / self.height) * np.tan(fov_rad / 2)
                
                ray_dir = np.array([ndc_x, ndc_y, 1.0])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                t, normal, material = self._trace_ray(self.camera_pos, ray_dir)
                
                if t < np.inf:
                    hit_point = self.camera_pos + t * ray_dir
                    
                    depth[y, x] = t
                    normals[y, x] = normal
                    rgb[y, x] = self._shade(hit_point, normal, material)
                    
                    # Compute total lighting contribution
                    total_light = 0.1  # ambient
                    for light in self.lights:
                        ndotl = max(0, np.dot(normal, -light.direction))
                        total_light += ndotl * light.intensity
                    lighting[y, x] = total_light
                else:
                    # Sky/background
                    rgb[y, x] = np.array([0.5, 0.7, 1.0])  # Sky blue
                    depth[y, x] = 100.0  # Far
                    normals[y, x] = np.array([0, 0, -1])
                    lighting[y, x] = 1.0
        
        return {
            'rgb': rgb,
            'depth': depth,
            'normals': normals,
            'lighting': lighting
        }


def create_test_scene() -> SyntheticScene:
    """Create a simple test scene with known geometry."""
    scene = SyntheticScene(width=256, height=256)
    
    # Ground plane
    scene.add_plane(
        point=(0, -1, 0),
        normal=(0, 1, 0),
        color=(0.4, 0.35, 0.3),  # Brown
        roughness=0.9
    )
    
    # Spheres at different depths
    scene.add_sphere(center=(-1.5, 0, 4), radius=0.8, color=(0.8, 0.2, 0.2))  # Red, close
    scene.add_sphere(center=(0, 0.3, 6), radius=1.0, color=(0.2, 0.8, 0.2))   # Green, mid
    scene.add_sphere(center=(1.5, -0.2, 8), radius=0.6, color=(0.2, 0.2, 0.8)) # Blue, far
    
    # Main light
    scene.add_light(direction=(-1, -1, 1), color=(1.0, 0.95, 0.9), intensity=0.8)
    
    # Fill light
    scene.add_light(direction=(1, -0.5, 0.5), color=(0.6, 0.7, 1.0), intensity=0.3)
    
    return scene


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def analyze_geometric_features(render_data: dict):
    """
    Analyze which geometric features predict depth.
    
    With ground truth, we can measure EXACTLY how well each feature
    correlates with true depth.
    """
    rgb = render_data['rgb']
    depth = render_data['depth']
    normals = render_data['normals']
    lighting = render_data['lighting']
    
    h, w = depth.shape
    
    # Normalize depth for analysis
    depth_norm = _normalize(depth)
    
    # Extract features
    features = {}
    
    # 1. Vertical position (our baseline)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    features['vertical'] = y_coords
    
    # 2. Color channels
    features['red'] = rgb[:, :, 0]
    features['green'] = rgb[:, :, 1]
    features['blue'] = rgb[:, :, 2]
    
    # 3. Brightness
    features['brightness'] = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # 4. Lighting (ground truth - not available in photos!)
    features['lighting'] = lighting
    
    # 5. Normal Y component (up-facing surfaces)
    features['normal_y'] = normals[:, :, 1]
    
    # 6. Normal Z component (camera-facing surfaces)
    features['normal_z'] = normals[:, :, 2]
    
    # Compute correlations with depth
    print("=" * 60)
    print("FEATURE CORRELATIONS WITH TRUE DEPTH")
    print("=" * 60)
    print()
    
    correlations = {}
    for name, feature in features.items():
        # Flatten and compute correlation
        corr = np.corrcoef(feature.flatten(), depth_norm.flatten())[0, 1]
        correlations[name] = corr
        print(f"  {name:<15}: {corr:+.3f}")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Best predictors of depth:")
    for name, corr in sorted_corr[:3]:
        print(f"  {name}: {corr:+.3f}")
    
    print()
    print("Key insight: Which features are AVAILABLE in photos?")
    print("  - vertical: YES (image coordinates)")
    print("  - color/brightness: YES (pixel values)")
    print("  - lighting: NO (collapsed into pixel values)")
    print("  - normals: NO (not directly observable)")
    
    return correlations


def test_self_assembly_with_ground_truth(n_scenes: int = 10):
    """
    Test self-assembly when we have COMPLETE ground truth.
    
    This tells us: if self-assembly had access to all information,
    how well could it work?
    """
    print()
    print("=" * 70)
    print("EXPERIMENT: Self-Assembly with Complete Ground Truth")
    print("=" * 70)
    print()
    
    # Generate multiple scenes
    all_features = []
    all_depths = []
    
    for i in range(n_scenes):
        scene = SyntheticScene(width=128, height=128)
        
        # Ground plane
        scene.add_plane(point=(0, -1, 0), normal=(0, 1, 0), 
                       color=(0.4 + 0.1*np.random.randn(), 0.35, 0.3))
        
        # Random spheres
        n_spheres = np.random.randint(2, 5)
        for _ in range(n_spheres):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-0.5, 1)
            z = np.random.uniform(3, 10)
            r = np.random.uniform(0.3, 1.0)
            color = np.random.uniform(0.2, 0.9, 3)
            scene.add_sphere(center=(x, y, z), radius=r, color=tuple(color))
        
        # Random lighting
        light_dir = np.random.randn(3)
        light_dir[1] = -abs(light_dir[1])  # Light from above
        scene.add_light(direction=tuple(light_dir), intensity=0.8)
        
        # Render
        data = scene.render()
        
        h, w = data['depth'].shape
        
        # Extract features (including ground truth that photos don't have)
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        
        features = np.stack([
            data['rgb'][:,:,0],           # R
            data['rgb'][:,:,1],           # G
            data['rgb'][:,:,2],           # B
            y_coords,                      # Vertical position
            data['lighting'],              # GROUND TRUTH: lighting
            data['normals'][:,:,1],        # GROUND TRUTH: normal Y
            data['normals'][:,:,2],        # GROUND TRUTH: normal Z
        ], axis=2)
        
        all_features.append(features.reshape(-1, 7))
        all_depths.append(_normalize(data['depth']).flatten())
    
    # Stack all data
    X = np.vstack(all_features)
    y = np.concatenate(all_depths)
    
    print(f"Total samples: {len(y)}")
    
    # Simple linear regression to see how well features predict depth
    # With ground truth features
    X_with_gt = X  # All 7 features including normals/lighting
    X_without_gt = X[:, :4]  # Only RGB + vertical (what photos have)
    
    # Fit linear models
    from numpy.linalg import lstsq
    
    # With ground truth
    coeffs_gt, residuals_gt, _, _ = lstsq(
        np.column_stack([X_with_gt, np.ones(len(y))]), y, rcond=None
    )
    pred_gt = X_with_gt @ coeffs_gt[:-1] + coeffs_gt[-1]
    mae_gt = np.mean(np.abs(pred_gt - y))
    
    # Without ground truth
    coeffs_no_gt, residuals_no_gt, _, _ = lstsq(
        np.column_stack([X_without_gt, np.ones(len(y))]), y, rcond=None
    )
    pred_no_gt = X_without_gt @ coeffs_no_gt[:-1] + coeffs_no_gt[-1]
    mae_no_gt = np.mean(np.abs(pred_no_gt - y))
    
    # Vertical only
    vertical = X[:, 3:4]
    coeffs_v, _, _, _ = lstsq(
        np.column_stack([vertical, np.ones(len(y))]), y, rcond=None
    )
    pred_v = vertical @ coeffs_v[:-1] + coeffs_v[-1]
    mae_v = np.mean(np.abs(pred_v - y))
    
    print()
    print("=" * 60)
    print("RESULTS: Linear Prediction of Depth")
    print("=" * 60)
    print()
    print(f"  Vertical only:           MAE = {mae_v:.4f}")
    print(f"  RGB + Vertical:          MAE = {mae_no_gt:.4f}")
    print(f"  RGB + Vertical + GT:     MAE = {mae_gt:.4f}")
    print()
    
    improvement_rgb = (mae_v - mae_no_gt) / mae_v * 100
    improvement_gt = (mae_v - mae_gt) / mae_v * 100
    
    print(f"  RGB adds:                {improvement_rgb:.1f}% improvement")
    print(f"  Ground truth adds:       {improvement_gt:.1f}% improvement")
    print()
    
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    print("The gap between 'RGB + Vertical' and 'RGB + Vertical + GT'")
    print("is the information that is LOST when 3D collapses to 2D.")
    print()
    print("This is what photographs CAN'T tell us:")
    print("  - Surface normals (orientation)")
    print("  - Lighting contribution (how much light hits each point)")
    print()
    print("This is what self-assembly would need to INFER from structure.")
    
    return {
        'mae_vertical': mae_v,
        'mae_rgb': mae_no_gt,
        'mae_gt': mae_gt
    }


def create_visualization():
    """Create visualization of synthetic scene and ground truth."""
    
    scene = create_test_scene()
    data = scene.render()
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Synthetic 3D Scene with Complete Ground Truth\n'
                 'What Information is Lost in 2D Projection?',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Row 1: What we CAN see
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(data['rgb'])
    ax1.set_title('RGB Image\n(What camera sees)', fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(_normalize(data['depth']), cmap='magma')
    ax2.set_title('True Depth\n(Ground truth)', fontsize=10)
    ax2.axis('off')
    
    # Vertical baseline
    h, w = data['depth'].shape
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    vertical = _normalize(0.6 * y_coords + 0.1)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(vertical, cmap='magma')
    ax3.set_title('Vertical Baseline\n(What we estimate)', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    error = np.abs(_normalize(data['depth']) - vertical)
    ax4.imshow(error, cmap='hot')
    ax4.set_title('Error\n(What we miss)', fontsize=10)
    ax4.axis('off')
    
    # Row 2: What we CAN'T see (ground truth only)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(data['lighting'], cmap='gray')
    ax5.set_title('Lighting\n(NOT in photos)', fontsize=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    normal_vis = (data['normals'] + 1) / 2  # Map [-1,1] to [0,1]
    ax6.imshow(normal_vis)
    ax6.set_title('Surface Normals\n(NOT in photos)', fontsize=10)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(data['normals'][:,:,1], cmap='RdBu', vmin=-1, vmax=1)
    ax7.set_title('Normal Y (up/down)\n(NOT in photos)', fontsize=10)
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(data['normals'][:,:,2], cmap='RdBu', vmin=-1, vmax=1)
    ax8.set_title('Normal Z (facing)\n(NOT in photos)', fontsize=10)
    ax8.axis('off')
    
    output_file = OUTPUT_PATH / "synthetic_3d_ground_truth.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Create and visualize a test scene
    print("Creating synthetic 3D scene...")
    scene = create_test_scene()
    data = scene.render()
    
    # Analyze feature correlations
    correlations = analyze_geometric_features(data)
    
    # Test self-assembly with ground truth
    results = test_self_assembly_with_ground_truth(n_scenes=10)
    
    # Create visualization
    viz_file = create_visualization()
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("With synthetic 3D, we have COMPLETE ground truth.")
    print("This reveals what information is LOST in 2D projection:")
    print()
    print("  1. Surface normals - orientation of surfaces")
    print("  2. Lighting contribution - how light interacts with geometry")
    print("  3. Material properties - how surfaces reflect light")
    print()
    print("These are the 'unknowns' that make depth from photos so hard.")
    print("Self-assembly needs to INFER these from structure alone.")
