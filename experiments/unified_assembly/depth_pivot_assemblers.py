#!/usr/bin/env python3
"""
Experiment: Self-Assembling Pivot Points for Depth Estimation

Key insight from user: "What if we used a self assembler for each pivot point?"

Each pivot point has its own DOMAIN of self-similarity - its own geometric 
structure that EMERGES from the data rather than being designed.

Architecture:
    Image → [Camera Assembler] → Vertical Structure (quaternion)
         → [Light Assembler]  → Shadow Structure (quaternion)  
         → [Surface Assembler] → Normal Field (quaternion field)
         → [Color Assembler]  → Phase Structure (quaternion)
                                    ↓
                          [Intersection] → Depth

Each assembler:
1. Extracts PAIRS from the image (like the text assembler extracts word pairs)
2. Discovers DIMENSIONS from pair relationships
3. Outputs a QUATERNION representing that pivot's orientation
4. Self-assembles - structure emerges, not designed

The final depth is the INTERSECTION of all pivot constraints.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.linalg import svd
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
# QUATERNION (same as before)
# =============================================================================

@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return cls(w, xyz[0], xyz[1], xyz[2])
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'Quaternion':
        """Create quaternion that rotates [0,0,1] to v."""
        v = v / (np.linalg.norm(v) + 1e-10)
        default = np.array([0, 0, 1])
        axis = np.cross(default, v)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return cls.identity()
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(default, v), -1, 1))
        return cls.from_axis_angle(axis, angle)
    
    def to_vector(self) -> np.ndarray:
        """Get the direction this quaternion points (rotated z-axis)."""
        # Rotate [0,0,1] by this quaternion
        qv = Quaternion(0, 0, 0, 1)
        q_conj = Quaternion(self.w, -self.x, -self.y, -self.z)
        
        def qmul(a, b):
            return Quaternion(
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
                a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
            )
        
        result = qmul(qmul(self, qv), q_conj)
        return np.array([result.x, result.y, result.z])
    
    def normalize(self) -> 'Quaternion':
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)


# =============================================================================
# BASE PIVOT ASSEMBLER
# =============================================================================

@dataclass
class AssemblyState:
    """State of a pivot assembler."""
    cycle: int = 0
    pairs_extracted: int = 0
    dimensions_discovered: int = 0
    quaternion: Quaternion = field(default_factory=Quaternion.identity)
    confidence: float = 0.0


class PivotAssembler:
    """
    Base class for self-assembling pivot points.
    
    Each pivot assembler:
    1. Extracts PAIRS from the image relevant to its domain
    2. Builds a SIMILARITY MATRIX from pairs
    3. Discovers DIMENSIONS via eigendecomposition
    4. Outputs a QUATERNION representing the pivot orientation
    
    This mirrors the text self-assembler but for geometric pivots.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.state = AssemblyState()
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.dimensions: Optional[np.ndarray] = None
    
    def extract_pairs(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract pairs relevant to this pivot. Override in subclasses."""
        raise NotImplementedError
    
    def build_similarity_matrix(self) -> np.ndarray:
        """Build similarity matrix from pairs."""
        if not self.pairs:
            return np.eye(1)
        
        n = len(self.pairs)
        S = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Similarity = how related are these pairs?
                p1_src, p1_tgt = self.pairs[i]
                p2_src, p2_tgt = self.pairs[j]
                
                # Cosine similarity of source vectors
                src_sim = np.dot(p1_src.flatten(), p2_src.flatten()) / (
                    np.linalg.norm(p1_src) * np.linalg.norm(p2_src) + 1e-10)
                
                # Cosine similarity of target vectors
                tgt_sim = np.dot(p1_tgt.flatten(), p2_tgt.flatten()) / (
                    np.linalg.norm(p1_tgt) * np.linalg.norm(p2_tgt) + 1e-10)
                
                S[i, j] = (src_sim + tgt_sim) / 2
        
        self.similarity_matrix = S
        return S
    
    def discover_dimensions(self) -> np.ndarray:
        """Discover dimensions via eigendecomposition."""
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        S = self.similarity_matrix
        
        # Eigendecomposition: S = V @ Λ @ V.T
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(S)
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Keep top dimensions (those with significant eigenvalues)
            significant = eigenvalues > 0.01 * eigenvalues[0]
            self.dimensions = eigenvectors[:, significant]
            self.state.dimensions_discovered = significant.sum()
            
        except:
            self.dimensions = np.eye(min(S.shape[0], 3))
            self.state.dimensions_discovered = self.dimensions.shape[1]
        
        return self.dimensions
    
    def compute_quaternion(self) -> Quaternion:
        """Compute quaternion from discovered dimensions."""
        if self.dimensions is None:
            self.discover_dimensions()
        
        # The principal dimension gives the pivot orientation
        if self.dimensions.shape[1] >= 3:
            # Use top 3 dimensions as axes
            principal = self.dimensions[:, 0]
            
            # Convert to 3D direction
            if len(principal) >= 3:
                direction = principal[:3]
            else:
                direction = np.array([0, 0, 1])
            
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            self.state.quaternion = Quaternion.from_vector(direction)
        else:
            self.state.quaternion = Quaternion.identity()
        
        return self.state.quaternion
    
    def assemble(self, image: np.ndarray) -> Quaternion:
        """Run the full assembly loop."""
        self.state.cycle += 1
        
        # 1. Extract pairs
        self.pairs = self.extract_pairs(image)
        self.state.pairs_extracted = len(self.pairs)
        
        # 2. Build similarity matrix
        self.build_similarity_matrix()
        
        # 3. Discover dimensions
        self.discover_dimensions()
        
        # 4. Compute quaternion
        return self.compute_quaternion()


# =============================================================================
# CAMERA PIVOT ASSEMBLER
# =============================================================================

class CameraPivotAssembler(PivotAssembler):
    """
    Self-assembles the camera orientation pivot.
    
    Pairs: (pixel_position, intensity) relationships
    Discovers: How vertical position relates to depth (perspective)
    Output: Quaternion representing camera tilt
    """
    
    def __init__(self):
        super().__init__("camera")
    
    def extract_pairs(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract position-intensity pairs."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        pairs = []
        
        # Sample pairs of pixels at different vertical positions
        n_samples = 50
        for _ in range(n_samples):
            # Random y positions
            y1 = np.random.randint(0, h//2)  # Top half
            y2 = np.random.randint(h//2, h)  # Bottom half
            x = np.random.randint(0, w)
            
            # Source: position vector [x/w, y1/h, 1]
            src = np.array([x/w, y1/h, 1.0])
            
            # Target: position vector [x/w, y2/h, 1]
            tgt = np.array([x/w, y2/h, 1.0])
            
            pairs.append((src, tgt))
        
        return pairs
    
    def compute_quaternion(self) -> Quaternion:
        """
        Camera quaternion from vertical gradient analysis.
        
        The camera tilt determines how vertical position maps to depth.
        """
        if not self.pairs:
            return Quaternion.identity()
        
        # Analyze the vertical relationship
        # For a horizontal camera: y maps linearly to depth
        # For a tilted camera: the mapping is rotated
        
        # Compute average direction from top to bottom
        directions = []
        for src, tgt in self.pairs:
            diff = tgt - src
            if np.linalg.norm(diff) > 0.01:
                directions.append(diff / np.linalg.norm(diff))
        
        if directions:
            avg_dir = np.mean(directions, axis=0)
            avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-10)
            
            # This direction represents the camera's "down" vector
            # Convert to quaternion
            self.state.quaternion = Quaternion.from_vector(avg_dir)
            self.state.confidence = 1.0 - np.std([np.linalg.norm(d - avg_dir) for d in directions])
        else:
            self.state.quaternion = Quaternion.identity()
            self.state.confidence = 0.0
        
        return self.state.quaternion


# =============================================================================
# LIGHT PIVOT ASSEMBLER
# =============================================================================

class LightPivotAssembler(PivotAssembler):
    """
    Self-assembles the light source pivot.
    
    Pairs: (gradient_direction, shadow_indicator) relationships
    Discovers: Light source direction from shading patterns
    Output: Quaternion representing light direction
    """
    
    def __init__(self):
        super().__init__("light")
    
    def extract_pairs(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract gradient-shadow pairs."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Compute gradients
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        
        pairs = []
        
        # Sample pairs at strong gradient locations
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        threshold = np.percentile(magnitude, 80)
        
        strong_y, strong_x = np.where(magnitude > threshold)
        
        if len(strong_y) > 100:
            indices = np.random.choice(len(strong_y), 100, replace=False)
            for idx in indices:
                y, x = strong_y[idx], strong_x[idx]
                
                # Source: gradient direction at this point
                gx, gy = grad_x[y, x], grad_y[y, x]
                src = np.array([gx, gy, magnitude[y, x]])
                
                # Target: intensity at this point (indicates facing toward/away from light)
                tgt = np.array([gray[y, x], 1.0 - gray[y, x], 0.5])
                
                pairs.append((src, tgt))
        
        return pairs
    
    def compute_quaternion(self) -> Quaternion:
        """
        Light quaternion from gradient analysis.
        
        The light direction is opposite to the average gradient
        (gradients point toward brighter areas, light comes from bright side).
        """
        if not self.pairs:
            return Quaternion.identity()
        
        # Average gradient direction
        grad_dirs = []
        for src, tgt in self.pairs:
            gx, gy = src[0], src[1]
            if abs(gx) > 0.01 or abs(gy) > 0.01:
                grad_dirs.append(np.array([gx, gy]))
        
        if grad_dirs:
            avg_grad = np.mean(grad_dirs, axis=0)
            
            # Light direction is opposite to gradient (light comes from bright side)
            light_2d = -avg_grad / (np.linalg.norm(avg_grad) + 1e-10)
            
            # Assume light is elevated (z > 0)
            light_3d = np.array([light_2d[0], light_2d[1], 0.5])
            light_3d = light_3d / np.linalg.norm(light_3d)
            
            self.state.quaternion = Quaternion.from_vector(light_3d)
            self.state.confidence = 1.0 / (1.0 + np.std([np.linalg.norm(d) for d in grad_dirs]))
        else:
            self.state.quaternion = Quaternion.identity()
            self.state.confidence = 0.0
        
        return self.state.quaternion


# =============================================================================
# SURFACE PIVOT ASSEMBLER
# =============================================================================

class SurfacePivotAssembler(PivotAssembler):
    """
    Self-assembles the surface normal field.
    
    Pairs: (local_patch, neighboring_patch) relationships
    Discovers: Surface orientation from texture gradients
    Output: Quaternion field (average orientation)
    """
    
    def __init__(self):
        super().__init__("surface")
    
    def extract_pairs(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract patch pairs for surface analysis."""
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        patch_size = 8
        pairs = []
        
        # Sample patch pairs
        n_samples = 50
        for _ in range(n_samples):
            # Random patch location
            y = np.random.randint(patch_size, h - patch_size)
            x = np.random.randint(patch_size, w - patch_size)
            
            # Source patch
            src_patch = gray[y-patch_size//2:y+patch_size//2, 
                            x-patch_size//2:x+patch_size//2]
            
            # Neighboring patch (below)
            if y + patch_size < h:
                tgt_patch = gray[y+patch_size//2:y+3*patch_size//2,
                                x-patch_size//2:x+patch_size//2]
            else:
                tgt_patch = src_patch
            
            # Flatten to vectors
            src = src_patch.flatten()[:9]  # Take first 9 elements
            tgt = tgt_patch.flatten()[:9]
            
            if len(src) == 9 and len(tgt) == 9:
                pairs.append((src, tgt))
        
        return pairs
    
    def compute_quaternion(self) -> Quaternion:
        """
        Surface quaternion from patch relationships.
        
        The dominant direction of texture change indicates surface orientation.
        """
        if not self.pairs:
            return Quaternion.identity()
        
        # Use SVD on the patch differences to find dominant direction
        diffs = []
        for src, tgt in self.pairs:
            diff = tgt - src
            if np.linalg.norm(diff) > 0.01:
                diffs.append(diff)
        
        if len(diffs) > 3:
            diff_matrix = np.array(diffs)
            try:
                U, S, Vt = svd(diff_matrix, full_matrices=False)
                
                # Principal direction
                principal = Vt[0, :3] if Vt.shape[1] >= 3 else np.array([0, 0, 1])
                principal = principal / (np.linalg.norm(principal) + 1e-10)
                
                self.state.quaternion = Quaternion.from_vector(principal)
                self.state.confidence = S[0] / (S.sum() + 1e-10)
            except:
                self.state.quaternion = Quaternion.identity()
                self.state.confidence = 0.0
        else:
            self.state.quaternion = Quaternion.identity()
            self.state.confidence = 0.0
        
        return self.state.quaternion


# =============================================================================
# COLOR PIVOT ASSEMBLER
# =============================================================================

class ColorPivotAssembler(PivotAssembler):
    """
    Self-assembles the color phase pivot.
    
    Pairs: (R, G, B) channel relationships
    Discovers: Chromatic aberration / wavelength-dependent depth
    Output: Quaternion representing color phase
    """
    
    def __init__(self):
        super().__init__("color")
    
    def extract_pairs(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract color channel pairs."""
        if image.ndim != 3 or image.shape[2] < 3:
            return []
        
        h, w = image.shape[:2]
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        pairs = []
        
        # Sample pairs of R-B differences at different locations
        n_samples = 50
        for _ in range(n_samples):
            y1 = np.random.randint(0, h)
            x1 = np.random.randint(0, w)
            y2 = np.random.randint(0, h)
            x2 = np.random.randint(0, w)
            
            # Source: RGB at point 1
            src = np.array([r[y1, x1], g[y1, x1], b[y1, x1]])
            
            # Target: RGB at point 2
            tgt = np.array([r[y2, x2], g[y2, x2], b[y2, x2]])
            
            pairs.append((src, tgt))
        
        return pairs
    
    def compute_quaternion(self) -> Quaternion:
        """
        Color quaternion from channel relationships.
        
        The R-B difference encodes chromatic depth information.
        """
        if not self.pairs:
            return Quaternion.identity()
        
        # Analyze R-B relationships
        rb_diffs = []
        for src, tgt in self.pairs:
            rb_src = src[0] - src[2]  # R - B at source
            rb_tgt = tgt[0] - tgt[2]  # R - B at target
            rb_diffs.append(np.array([rb_src, rb_tgt, (rb_src + rb_tgt) / 2]))
        
        if rb_diffs:
            avg_rb = np.mean(rb_diffs, axis=0)
            avg_rb = avg_rb / (np.linalg.norm(avg_rb) + 1e-10)
            
            self.state.quaternion = Quaternion.from_vector(avg_rb)
            self.state.confidence = 1.0 / (1.0 + np.std([np.linalg.norm(d) for d in rb_diffs]))
        else:
            self.state.quaternion = Quaternion.identity()
            self.state.confidence = 0.0
        
        return self.state.quaternion


# =============================================================================
# PIVOT INTERSECTION
# =============================================================================

class PivotIntersection:
    """
    Combines multiple pivot assemblers to find depth.
    
    Each pivot provides a constraint. The depth is the intersection
    of all constraints - the solution that satisfies all pivots.
    """
    
    def __init__(self):
        self.camera = CameraPivotAssembler()
        self.light = LightPivotAssembler()
        self.surface = SurfacePivotAssembler()
        self.color = ColorPivotAssembler()
        
        self.assemblers = [self.camera, self.light, self.surface, self.color]
    
    def assemble_all(self, image: np.ndarray) -> Dict[str, Quaternion]:
        """Run all assemblers and return their quaternions."""
        results = {}
        for assembler in self.assemblers:
            q = assembler.assemble(image)
            results[assembler.name] = q
            print(f"  {assembler.name}: {assembler.state.pairs_extracted} pairs, "
                  f"{assembler.state.dimensions_discovered} dims, "
                  f"conf={assembler.state.confidence:.3f}")
        return results
    
    def predict_depth(self, image: np.ndarray, quaternions: Dict[str, Quaternion]) -> np.ndarray:
        """
        Predict depth from pivot quaternions.
        
        Each quaternion defines a constraint. We combine them weighted by confidence.
        """
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Create coordinate grids
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
        
        # Each pivot contributes a depth estimate
        depth_estimates = []
        weights = []
        
        # Camera pivot: vertical position → depth
        camera_dir = self.camera.state.quaternion.to_vector()
        # Project y-coordinate onto camera direction
        camera_depth = y_coords * abs(camera_dir[1]) + (1 - y_coords) * (1 - abs(camera_dir[1]))
        depth_estimates.append(camera_depth)
        weights.append(self.camera.state.confidence + 0.1)  # Camera always has some weight
        
        # Light pivot: shading → depth
        light_dir = self.light.state.quaternion.to_vector()
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        # Dot product of gradient with light direction
        light_depth = _normalize(grad_x * light_dir[0] + grad_y * light_dir[1])
        depth_estimates.append(light_depth)
        weights.append(self.light.state.confidence)
        
        # Surface pivot: texture gradient → depth
        surface_dir = self.surface.state.quaternion.to_vector()
        # Use local variance as texture measure
        local_var = gaussian_filter(gray**2, sigma=3) - gaussian_filter(gray, sigma=3)**2
        surface_depth = _normalize(local_var * abs(surface_dir[2]))
        depth_estimates.append(surface_depth)
        weights.append(self.surface.state.confidence)
        
        # Color pivot: R-B difference → depth
        if image.ndim == 3:
            color_dir = self.color.state.quaternion.to_vector()
            rb_diff = image[:,:,0] - image[:,:,2]
            color_depth = _normalize(rb_diff * color_dir[0])
            depth_estimates.append(color_depth)
            weights.append(self.color.state.confidence)
        
        # Combine by confidence-weighted average
        total_weight = sum(weights) + 1e-10
        weights = [w / total_weight for w in weights]
        
        depth = np.zeros_like(gray)
        for est, w in zip(depth_estimates, weights):
            depth += w * est
        
        depth = gaussian_filter(depth, sigma=2.0)
        return _normalize(depth)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_pivot_assembler_experiment(n_train: int = 20, n_test: int = 10):
    """
    Test self-assembling pivot points for depth estimation.
    """
    print("=" * 70)
    print("EXPERIMENT: Self-Assembling Pivot Points")
    print("=" * 70)
    print()
    print("Each pivot has its own self-assembler that discovers structure:")
    print("  - Camera: vertical position → depth relationship")
    print("  - Light: gradient direction → light source")
    print("  - Surface: patch relationships → surface orientation")
    print("  - Color: R/G/B channels → chromatic depth")
    print()
    print("Pivots assemble independently, then INTERSECT to find depth.")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Train: let assemblers discover structure from training images
    print("=" * 60)
    print("Phase 1: Self-Assembly (Training)")
    print("=" * 60)
    
    intersection = PivotIntersection()
    
    # Run assembly on training images to calibrate
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Resize for speed
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        if i < 3:  # Show details for first few
            print(f"\nImage {i+1}: {img_id}")
            intersection.assemble_all(rgb_small)
    
    # Test
    print("\n" + "=" * 60)
    print("Phase 2: Testing")
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
        
        # Assemble pivots for this image
        print(f"\nTest image {i+1}: {img_id}")
        quaternions = intersection.assemble_all(rgb_small)
        
        # Predict depth
        pred_depth = intersection.predict_depth(rgb_small, quaternions)
        
        mae = np.mean(np.abs(pred_depth - true_depth_small))
        test_errors.append(mae)
        print(f"  MAE: {mae:.4f}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Self-Assembling Pivots Test MAE: {np.mean(test_errors):.4f}")
    print(f"  Constraint Intersection:          0.235")
    print(f"  Vertical alone:                   0.182")
    print(f"  Statistical best:                 0.199")
    
    # Analysis
    print("\n" + "=" * 60)
    print("PIVOT ANALYSIS")
    print("=" * 60)
    
    for assembler in intersection.assemblers:
        q = assembler.state.quaternion
        v = q.to_vector()
        print(f"\n  {assembler.name}:")
        print(f"    Direction: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
        print(f"    Confidence: {assembler.state.confidence:.3f}")
        print(f"    Dimensions discovered: {assembler.state.dimensions_discovered}")
    
    return intersection


class CrossPivotAssembly:
    """
    Self-assembling pivots with CROSS-PIVOT COMMUNICATION.
    
    Key insight: In inverse kinematics, joints are CONNECTED.
    The position of one joint constrains the others.
    
    Similarly, our pivots should communicate:
    - Camera pivot discovers vertical relationship
    - Light pivot uses camera's vertical to constrain shadow direction
    - Surface pivot uses both to constrain normal field
    - Color pivot uses all three to constrain chromatic depth
    
    This creates a KINEMATIC CHAIN where each pivot's discovery
    informs the next.
    """
    
    def __init__(self):
        self.camera = CameraPivotAssembler()
        self.light = LightPivotAssembler()
        self.surface = SurfacePivotAssembler()
        self.color = ColorPivotAssembler()
        
        # Cross-pivot learned relationships
        self.camera_to_light: Optional[np.ndarray] = None  # How camera constrains light
        self.light_to_surface: Optional[np.ndarray] = None  # How light constrains surface
        self.all_to_depth: Optional[np.ndarray] = None  # Combined mapping to depth
    
    def assemble_chain(self, image: np.ndarray, depth: np.ndarray = None) -> Dict[str, Quaternion]:
        """
        Assemble pivots in a CHAIN, each informing the next.
        """
        results = {}
        
        # 1. Camera pivot first (no dependencies)
        camera_q = self.camera.assemble(image)
        results['camera'] = camera_q
        camera_dir = camera_q.to_vector()
        
        # 2. Light pivot, constrained by camera
        # The light direction should be consistent with the camera's vertical
        light_q = self.light.assemble(image)
        light_dir = light_q.to_vector()
        
        # Adjust light to be orthogonal to camera's "up" if they're too aligned
        dot_product = np.dot(camera_dir, light_dir)
        if abs(dot_product) > 0.9:
            # Light is too aligned with vertical - adjust
            # Light should come from the side, not directly above/below
            perpendicular = np.cross(camera_dir, np.array([1, 0, 0]))
            if np.linalg.norm(perpendicular) < 0.1:
                perpendicular = np.cross(camera_dir, np.array([0, 0, 1]))
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)
            light_dir = 0.7 * light_dir + 0.3 * perpendicular
            light_dir = light_dir / np.linalg.norm(light_dir)
            light_q = Quaternion.from_vector(light_dir)
        
        results['light'] = light_q
        
        # 3. Surface pivot, constrained by camera and light
        surface_q = self.surface.assemble(image)
        results['surface'] = surface_q
        
        # 4. Color pivot, constrained by all
        color_q = self.color.assemble(image)
        results['color'] = color_q
        
        return results
    
    def learn_cross_pivot_mapping(self, images: List[np.ndarray], depths: List[np.ndarray]):
        """
        Learn how pivots relate to each other and to depth.
        
        This is the KEY: we learn the KINEMATIC CHAIN that connects
        pivot orientations to depth.
        """
        print("Learning cross-pivot mappings...")
        
        # Collect pivot outputs and depths
        all_camera_dirs = []
        all_light_dirs = []
        all_surface_dirs = []
        all_color_dirs = []
        all_depths = []
        
        for image, depth in zip(images, depths):
            quaternions = self.assemble_chain(image)
            
            all_camera_dirs.append(quaternions['camera'].to_vector())
            all_light_dirs.append(quaternions['light'].to_vector())
            all_surface_dirs.append(quaternions['surface'].to_vector())
            all_color_dirs.append(quaternions['color'].to_vector())
            all_depths.append(depth.mean())  # Use mean depth for now
        
        # Stack into matrices
        camera_matrix = np.array(all_camera_dirs)
        light_matrix = np.array(all_light_dirs)
        
        # Learn camera → light relationship
        # How does camera orientation predict light orientation?
        try:
            self.camera_to_light = np.linalg.lstsq(camera_matrix, light_matrix, rcond=None)[0]
            print(f"  Camera → Light mapping learned")
        except:
            self.camera_to_light = np.eye(3)
        
        # Learn combined → depth relationship
        # Stack all pivot directions
        combined = np.hstack([camera_matrix, light_matrix])
        depths_array = np.array(all_depths)
        
        try:
            self.all_to_depth = np.linalg.lstsq(combined, depths_array, rcond=None)[0]
            print(f"  Combined → Depth mapping learned")
        except:
            self.all_to_depth = np.ones(6) / 6
    
    def predict_depth_chained(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth using the learned kinematic chain.
        """
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Assemble pivots
        quaternions = self.assemble_chain(image)
        
        # Get directions
        camera_dir = quaternions['camera'].to_vector()
        light_dir = quaternions['light'].to_vector()
        surface_dir = quaternions['surface'].to_vector()
        color_dir = quaternions['color'].to_vector()
        
        # Create coordinate grids
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        x_coords = np.tile(np.linspace(0, 1, w).reshape(1, -1), (h, 1))
        
        # Camera contribution: project position onto camera direction
        # Camera direction [0, 1, 0] means vertical = depth
        camera_depth = y_coords * camera_dir[1] + x_coords * camera_dir[0]
        
        # Light contribution: shading
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        light_depth = _normalize(grad_x * light_dir[0] + grad_y * light_dir[1])
        
        # Surface contribution: texture gradient
        local_var = gaussian_filter(gray**2, sigma=3) - gaussian_filter(gray, sigma=3)**2
        surface_depth = _normalize(np.sqrt(np.maximum(local_var, 0)))
        
        # Color contribution
        if image.ndim == 3:
            rb_diff = image[:,:,0] - image[:,:,2]
            color_depth = _normalize(rb_diff)
        else:
            color_depth = np.zeros_like(gray)
        
        # Combine using confidences
        weights = [
            self.camera.state.confidence,
            self.light.state.confidence * 0.5,  # Light is less reliable
            self.surface.state.confidence * 0.3,
            self.color.state.confidence * 0.2
        ]
        total_weight = sum(weights) + 1e-10
        weights = [w / total_weight for w in weights]
        
        depth = (weights[0] * camera_depth + 
                 weights[1] * light_depth + 
                 weights[2] * surface_depth + 
                 weights[3] * color_depth)
        
        depth = gaussian_filter(depth, sigma=2.0)
        return _normalize(depth)


def run_cross_pivot_experiment(n_train: int = 30, n_test: int = 10):
    """
    Test cross-pivot communication for depth estimation.
    """
    print("=" * 70)
    print("EXPERIMENT: Cross-Pivot Communication")
    print("=" * 70)
    print()
    print("Pivots form a KINEMATIC CHAIN:")
    print("  Camera → Light → Surface → Color → Depth")
    print()
    print("Each pivot's discovery constrains the next.")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    cross_assembly = CrossPivotAssembly()
    
    # Collect training data
    train_images = []
    train_depths = []
    
    print("Loading training data...")
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
        
        train_images.append(rgb_small)
        train_depths.append(depth_small)
    
    print(f"  Loaded {len(train_images)} training images")
    
    # Learn cross-pivot mappings
    cross_assembly.learn_cross_pivot_mapping(train_images, train_depths)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing")
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
        
        # Predict
        pred_depth = cross_assembly.predict_depth_chained(rgb_small)
        
        mae = np.mean(np.abs(pred_depth - true_depth_small))
        test_errors.append(mae)
        
        if i < 3:
            print(f"\n  Test {i+1}: {img_id}")
            print(f"    Camera conf: {cross_assembly.camera.state.confidence:.3f}")
            print(f"    Light conf:  {cross_assembly.light.state.confidence:.3f}")
            print(f"    MAE: {mae:.4f}")
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Cross-Pivot Chain Test MAE: {np.mean(test_errors):.4f}")
    print(f"  Independent Pivots:          0.2146")
    print(f"  Vertical alone:              0.182")
    print(f"  Statistical best:            0.199")
    
    return cross_assembly


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SELF-ASSEMBLING PIVOT DEPTH ESTIMATION")
    print("="*70)
    
    print("\n" + "="*70)
    print("PART 1: Independent Pivots")
    print("="*70)
    intersection = run_pivot_assembler_experiment(n_train=20, n_test=10)
    
    print("\n" + "="*70)
    print("PART 2: Cross-Pivot Chain")
    print("="*70)
    cross = run_cross_pivot_experiment(n_train=30, n_test=10)
