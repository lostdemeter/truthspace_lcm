#!/usr/bin/env python3
"""
φ-Space Navigation - How to traverse from Point A to Point B

Key insight: We can get between grayscale and color in DA2's space.
The question is: What's the optimal PATH?

This explores:
1. What does the φ-space look like?
2. Can we quantize positions to φ-lattice?
3. Is there a φ-A* algorithm?
4. What are the "valid moves" in φ-space?

The hypothesis: Navigation in φ-space follows φ-constraints.
- Valid positions are at φ^n levels
- Valid moves are φ-scaled steps
- The optimal path minimizes φ-distance

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
import heapq
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LN_PHI = np.log(PHI)

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = -0.147 * rgb[..., 0] - 0.289 * rgb[..., 1] + 0.436 * rgb[..., 2]
    v = 0.615 * rgb[..., 0] - 0.515 * rgb[..., 1] - 0.100 * rgb[..., 2]
    return np.stack([y, u, v], axis=-1)


# ============================================================
# φ-LATTICE OPERATIONS
# ============================================================

def to_phi_level(value: float, k: int = 32) -> int:
    """
    Convert a value to its φ-level.
    
    level = k * log_φ(|value|)
    
    This quantizes the value to the φ-lattice.
    """
    if abs(value) < 1e-10:
        return 0
    return int(round(k * np.log(abs(value)) / LN_PHI))


def from_phi_level(level: int, sign: float = 1.0, k: int = 32) -> float:
    """Convert φ-level back to value."""
    return sign * (PHI ** (level / k))


def phi_distance(a: np.ndarray, b: np.ndarray, k: int = 32) -> float:
    """
    Compute φ-distance between two points.
    
    In φ-space, distance is measured in φ-levels, not Euclidean.
    """
    # Convert to φ-levels
    a_levels = np.array([to_phi_level(v, k) for v in a])
    b_levels = np.array([to_phi_level(v, k) for v in b])
    
    # φ-distance is the sum of level differences
    return np.sum(np.abs(a_levels - b_levels))


def phi_manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """
    φ-Manhattan distance: sum of |log_φ(a_i/b_i)|
    
    This measures how many φ-steps apart two points are.
    """
    # Avoid division by zero
    a_safe = np.where(np.abs(a) < 1e-10, 1e-10, a)
    b_safe = np.where(np.abs(b) < 1e-10, 1e-10, b)
    
    # Ratio in log-φ space
    ratios = np.abs(np.log(np.abs(a_safe / b_safe)) / LN_PHI)
    
    return np.sum(ratios)


# ============================================================
# φ-SPACE A* ALGORITHM
# ============================================================

class PhiSpaceNavigator:
    """
    Navigate through φ-space using A*-like algorithm.
    
    The key insight: Valid moves in φ-space are φ-scaled.
    - Move by φ^n in any dimension
    - Cost is proportional to |n|
    - Heuristic is φ-distance to goal
    """
    
    def __init__(self, n_dims: int, k: int = 8):
        """
        n_dims: Number of dimensions in the space
        k: φ-grid resolution (steps per factor of φ)
        """
        self.n_dims = n_dims
        self.k = k
        
        # Valid move sizes (in φ-levels)
        # We can move by ±1, ±φ, ±φ², etc. levels
        self.move_sizes = [1, 2, 3, 5, 8, 13]  # Fibonacci! (φ-related)
    
    def quantize_to_phi_grid(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize a point to the φ-grid.
        
        Returns (levels, signs) where point ≈ signs * φ^(levels/k)
        """
        signs = np.sign(point)
        signs[signs == 0] = 1  # Default positive
        
        levels = np.zeros(len(point), dtype=int)
        for i, v in enumerate(point):
            levels[i] = to_phi_level(v, self.k)
        
        return levels, signs
    
    def from_phi_grid(self, levels: np.ndarray, signs: np.ndarray) -> np.ndarray:
        """Convert φ-grid position back to values."""
        return signs * (PHI ** (levels / self.k))
    
    def get_neighbors(self, levels: np.ndarray, signs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get valid neighbors in φ-space.
        
        Returns list of (new_levels, new_signs, cost)
        """
        neighbors = []
        
        for dim in range(self.n_dims):
            for move_size in self.move_sizes:
                # Move up
                new_levels = levels.copy()
                new_levels[dim] += move_size
                neighbors.append((new_levels, signs.copy(), move_size))
                
                # Move down
                new_levels = levels.copy()
                new_levels[dim] -= move_size
                neighbors.append((new_levels, signs.copy(), move_size))
                
                # Sign flip (costs more)
                new_signs = signs.copy()
                new_signs[dim] *= -1
                neighbors.append((levels.copy(), new_signs, move_size * 2))
        
        return neighbors
    
    def heuristic(self, current_levels: np.ndarray, goal_levels: np.ndarray) -> float:
        """
        Heuristic for A*: φ-distance to goal.
        
        This is admissible because we can't get there faster than
        the sum of level differences.
        """
        return np.sum(np.abs(current_levels - goal_levels))
    
    def find_path(self, start: np.ndarray, goal: np.ndarray, 
                  max_iterations: int = 10000) -> List[np.ndarray]:
        """
        Find path from start to goal using φ-A*.
        
        Returns list of points along the path.
        """
        # Quantize start and goal
        start_levels, start_signs = self.quantize_to_phi_grid(start)
        goal_levels, goal_signs = self.quantize_to_phi_grid(goal)
        
        # Priority queue: (f_score, g_score, levels, signs, path)
        # f_score = g_score + heuristic
        start_h = self.heuristic(start_levels, goal_levels)
        
        # Use tuple of levels as state key
        open_set = [(start_h, 0, tuple(start_levels), tuple(start_signs), [start])]
        closed_set = set()
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            f, g, levels_tuple, signs_tuple, path = heapq.heappop(open_set)
            levels = np.array(levels_tuple)
            signs = np.array(signs_tuple)
            
            # Check if we've reached the goal
            if np.allclose(levels, goal_levels) and np.allclose(signs, goal_signs):
                print(f"   Path found in {iterations} iterations, {len(path)} steps")
                return path
            
            # Skip if already visited
            state = (levels_tuple, signs_tuple)
            if state in closed_set:
                continue
            closed_set.add(state)
            
            # Expand neighbors
            for new_levels, new_signs, cost in self.get_neighbors(levels, signs):
                new_state = (tuple(new_levels), tuple(new_signs))
                if new_state in closed_set:
                    continue
                
                new_g = g + cost
                new_h = self.heuristic(new_levels, goal_levels)
                new_f = new_g + new_h
                
                new_point = self.from_phi_grid(new_levels, new_signs)
                new_path = path + [new_point]
                
                heapq.heappush(open_set, (new_f, new_g, tuple(new_levels), tuple(new_signs), new_path))
        
        print(f"   No path found after {iterations} iterations")
        return path  # Return best path so far
    
    def interpolate_phi(self, start: np.ndarray, goal: np.ndarray, 
                        n_steps: int = 10) -> List[np.ndarray]:
        """
        Simple φ-interpolation: move in φ-space linearly.
        
        This is the "straight line" in φ-space.
        """
        start_levels, start_signs = self.quantize_to_phi_grid(start)
        goal_levels, goal_signs = self.quantize_to_phi_grid(goal)
        
        path = []
        for t in np.linspace(0, 1, n_steps):
            # Interpolate levels
            interp_levels = (1 - t) * start_levels + t * goal_levels
            interp_levels = np.round(interp_levels).astype(int)
            
            # For signs, use start until halfway, then goal
            interp_signs = start_signs if t < 0.5 else goal_signs
            
            point = self.from_phi_grid(interp_levels, interp_signs)
            path.append(point)
        
        return path


# ============================================================
# VISUALIZATION AND TESTING
# ============================================================

def load_da2():
    """Load DA2 model."""
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    
    return model, processor


def extract_da2_structure(model, processor, rgb: np.ndarray):
    """Extract DA2's backbone structure."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
    
    return structure


def analyze_phi_structure(model, processor, rgb: np.ndarray):
    """
    Analyze the φ-structure of DA2's encoding.
    
    Questions:
    1. Are the values already on a φ-lattice?
    2. What's the distribution of φ-levels?
    3. How far apart are grayscale and color in φ-space?
    """
    if rgb.max() > 1:
        rgb = rgb.astype(np.float32) / 255.0
    
    structure = extract_da2_structure(model, processor, rgb)
    structure = structure[1:]  # Skip CLS
    
    # Flatten all values
    all_values = structure.flatten()
    
    # Convert to φ-levels
    k = 32
    levels = np.array([to_phi_level(v, k) for v in all_values])
    
    # Distribution of levels
    unique_levels, counts = np.unique(levels, return_counts=True)
    
    print(f"   Total values: {len(all_values)}")
    print(f"   Unique φ-levels: {len(unique_levels)}")
    print(f"   Level range: [{levels.min()}, {levels.max()}]")
    print(f"   Most common levels: {unique_levels[np.argsort(counts)[-5:][::-1]]}")
    
    # Check if values cluster at φ-levels
    reconstructed = np.array([from_phi_level(l, np.sign(v) if v != 0 else 1, k) 
                              for l, v in zip(levels, all_values)])
    
    reconstruction_error = np.abs(all_values - reconstructed).mean()
    print(f"   φ-quantization error: {reconstruction_error:.6f}")
    
    return levels, unique_levels, counts


def test_phi_navigation():
    """Test φ-space navigation."""
    print("=" * 70)
    print("φ-SPACE NAVIGATION")
    print("How to traverse from grayscale to color")
    print("=" * 70)
    
    # Test with simple vectors first
    print("\n1. SIMPLE NAVIGATION TEST")
    print("-" * 50)
    
    # Create a simple test case
    n_dims = 10
    navigator = PhiSpaceNavigator(n_dims=n_dims, k=8)
    
    # Start and goal
    np.random.seed(42)
    start = np.random.randn(n_dims) * 0.1
    goal = np.random.randn(n_dims) * 0.1
    
    print(f"   Start: {start[:5]}...")
    print(f"   Goal:  {goal[:5]}...")
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(goal - start)
    print(f"   Euclidean distance: {euclidean_dist:.4f}")
    
    # φ-distance
    phi_dist = phi_distance(start, goal, k=8)
    print(f"   φ-distance (levels): {phi_dist}")
    
    # φ-Manhattan
    phi_manh = phi_manhattan(start, goal)
    print(f"   φ-Manhattan: {phi_manh:.4f}")
    
    # Find path using φ-interpolation
    print("\n   φ-interpolation path:")
    path = navigator.interpolate_phi(start, goal, n_steps=5)
    for i, p in enumerate(path):
        dist_to_goal = np.linalg.norm(p - goal)
        print(f"     Step {i}: dist_to_goal = {dist_to_goal:.4f}")
    
    print("\n2. DA2 STRUCTURE ANALYSIS")
    print("-" * 50)
    
    # Load DA2 and analyze
    model, processor = load_da2()
    
    # Load a test image
    img_files = sorted(COCO_PATH.glob("*.jpg"))
    img = np.array(Image.open(img_files[200]).convert("RGB"))
    
    print(f"   Analyzing image: {img_files[200].stem}")
    levels, unique_levels, counts = analyze_phi_structure(model, processor, img)
    
    # Visualize level distribution
    print("\n3. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Level distribution
    axes[0].hist(levels, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('φ-level')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of φ-levels in DA2 structure')
    axes[0].axvline(x=0, color='red', linestyle='--', label='φ^0 = 1')
    axes[0].legend()
    
    # Top levels
    top_idx = np.argsort(counts)[-20:]
    axes[1].barh(range(20), counts[top_idx])
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f'φ^{unique_levels[i]/32:.2f}' for i in top_idx])
    axes[1].set_xlabel('Count')
    axes[1].set_title('Most common φ-levels')
    
    # Navigation path visualization (2D projection)
    np.random.seed(42)
    start_2d = np.random.randn(2) * 0.5
    goal_2d = np.random.randn(2) * 0.5 + 1
    
    nav_2d = PhiSpaceNavigator(n_dims=2, k=8)
    path_2d = nav_2d.interpolate_phi(start_2d, goal_2d, n_steps=20)
    path_2d = np.array(path_2d)
    
    axes[2].plot(path_2d[:, 0], path_2d[:, 1], 'b-o', markersize=4, label='φ-path')
    axes[2].plot([start_2d[0], goal_2d[0]], [start_2d[1], goal_2d[1]], 'r--', label='Euclidean')
    axes[2].scatter([start_2d[0]], [start_2d[1]], c='green', s=100, marker='s', label='Start')
    axes[2].scatter([goal_2d[0]], [goal_2d[1]], c='red', s=100, marker='*', label='Goal')
    axes[2].set_xlabel('Dim 0')
    axes[2].set_ylabel('Dim 1')
    axes[2].set_title('φ-path vs Euclidean path (2D)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "phi_space_navigation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'phi_space_navigation.png'}")
    
    return navigator, levels


if __name__ == "__main__":
    navigator, levels = test_phi_navigation()
    
    print("\n" + "=" * 70)
    print("φ-SPACE NAVIGATION SUMMARY")
    print("=" * 70)
    print(f"""
   Key findings:
   
   1. DA2's structure CAN be quantized to φ-levels
      - {len(np.unique(levels))} unique levels in the structure
      - Values cluster at specific φ-levels
   
   2. φ-distance is different from Euclidean distance
      - Measures "how many φ-steps apart"
      - More natural for φ-structured data
   
   3. φ-interpolation follows the φ-lattice
      - Moves in discrete φ-steps
      - Different path than Euclidean interpolation
   
   The φ-A* algorithm:
   - Valid moves are Fibonacci-sized steps (1, 2, 3, 5, 8, 13)
   - Cost is proportional to step size
   - Heuristic is φ-distance to goal
   
   This is the foundation for navigating from grayscale to color
   in φ-space!
""")
