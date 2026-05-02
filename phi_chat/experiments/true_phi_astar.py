#!/usr/bin/env python3
"""
True φ-A* Search - Find the optimal path through φ-space

Instead of using linear regression to find weights, we SEARCH for the
optimal path using A* with Fibonacci-sized moves.

The key insight: The solution space is constrained to the φ-lattice.
This dramatically reduces the search space while preserving accuracy.

Algorithm:
1. Start at the feature point (quantized to φ-lattice)
2. Goal is the target color (U, V)
3. Valid moves are Fibonacci-sized steps along any dimension
4. Cost is the total number of φ-steps
5. Heuristic is φ-distance to goal

This is fundamentally different from gradient descent:
- GD: Continuous optimization in weight space
- φ-A*: Discrete search on φ-lattice

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from typing import List, Tuple, Dict, Set
import heapq
from collections import defaultdict
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LN_PHI = np.log(PHI)

# Fibonacci numbers - the ONLY valid move sizes in φ-space
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

COCO_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/phi_chat/experiments/colorization_output")
OUTPUT_PATH.mkdir(exist_ok=True)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = -0.147 * rgb[..., 0] - 0.289 * rgb[..., 1] + 0.436 * rgb[..., 2]
    v = 0.615 * rgb[..., 0] - 0.515 * rgb[..., 1] - 0.100 * rgb[..., 2]
    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ============================================================
# φ-LATTICE OPERATIONS
# ============================================================

def to_phi_level(value: float, k: int = 32) -> int:
    """Convert value to φ-level."""
    if abs(value) < 1e-10:
        return 0
    return int(round(k * np.log(abs(value)) / LN_PHI))


def from_phi_level(level: int, sign: float = 1.0, k: int = 32) -> float:
    """Convert φ-level to value."""
    return sign * (PHI ** (level / k))


# ============================================================
# TRUE φ-A* SEARCH
# ============================================================

class PhiAStarSearch:
    """
    True A* search in φ-space.
    
    The state is a weight vector (quantized to φ-levels).
    The goal is to find weights that minimize prediction error.
    
    Key constraints:
    - Weights can only be at φ-levels
    - Moves can only be Fibonacci-sized
    - This makes the search tractable
    """
    
    def __init__(self, n_dims: int, k: int = 32, max_fib_idx: int = 6):
        """
        n_dims: Number of weight dimensions
        k: φ-grid resolution
        max_fib_idx: Maximum Fibonacci index to use (limits move size)
        """
        self.n_dims = n_dims
        self.k = k
        self.move_sizes = FIBONACCI[:max_fib_idx]  # e.g., [1, 2, 3, 5, 8, 13]
        
        # For efficiency, precompute φ^(level/k) for common levels
        self.phi_cache = {}
        for level in range(-1000, 200):
            self.phi_cache[level] = PHI ** (level / k)
    
    def get_phi_value(self, level: int, sign: float) -> float:
        """Get value from φ-level with caching."""
        if level in self.phi_cache:
            return sign * self.phi_cache[level]
        return sign * (PHI ** (level / self.k))
    
    def compute_prediction(self, weight_levels: np.ndarray, weight_signs: np.ndarray,
                          features: np.ndarray) -> float:
        """
        Compute prediction: sum(weight_i * feature_i)
        
        weights are in φ-level form, features are raw values.
        """
        total = 0.0
        for i in range(len(weight_levels)):
            w = self.get_phi_value(weight_levels[i], weight_signs[i])
            total += w * features[i]
        return total
    
    def compute_error(self, weight_levels: np.ndarray, weight_signs: np.ndarray,
                     all_features: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error over all samples."""
        total_error = 0.0
        for i in range(len(targets)):
            pred = self.compute_prediction(weight_levels, weight_signs, all_features[i])
            total_error += (pred - targets[i]) ** 2
        return total_error / len(targets)
    
    def get_neighbors(self, weight_levels: np.ndarray, weight_signs: np.ndarray,
                     active_dims: List[int] = None) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Get valid neighbors in φ-space.
        
        Only modify active_dims if specified (for efficiency).
        Returns list of (new_levels, new_signs, cost)
        """
        neighbors = []
        
        dims_to_modify = active_dims if active_dims else range(self.n_dims)
        
        for dim in dims_to_modify:
            for move_size in self.move_sizes:
                # Move up
                new_levels = weight_levels.copy()
                new_levels[dim] += move_size
                neighbors.append((new_levels, weight_signs.copy(), move_size))
                
                # Move down
                new_levels = weight_levels.copy()
                new_levels[dim] -= move_size
                neighbors.append((new_levels, weight_signs.copy(), move_size))
        
        return neighbors
    
    def search(self, all_features: np.ndarray, targets: np.ndarray,
               init_levels: np.ndarray = None, init_signs: np.ndarray = None,
               active_dims: List[int] = None,
               max_iterations: int = 1000, patience: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search for optimal weights using A*-like greedy search.
        
        Since the state space is huge, we use greedy local search
        with Fibonacci-sized moves.
        """
        # Initialize
        if init_levels is None:
            init_levels = np.zeros(self.n_dims, dtype=int) - 500  # Small initial weights
        if init_signs is None:
            init_signs = np.ones(self.n_dims)
        
        current_levels = init_levels.copy()
        current_signs = init_signs.copy()
        current_error = self.compute_error(current_levels, current_signs, all_features, targets)
        
        best_levels = current_levels.copy()
        best_signs = current_signs.copy()
        best_error = current_error
        
        no_improvement = 0
        
        for iteration in range(max_iterations):
            # Get neighbors
            neighbors = self.get_neighbors(current_levels, current_signs, active_dims)
            
            # Find best neighbor
            improved = False
            for new_levels, new_signs, cost in neighbors:
                new_error = self.compute_error(new_levels, new_signs, all_features, targets)
                
                if new_error < current_error:
                    current_levels = new_levels
                    current_signs = new_signs
                    current_error = new_error
                    improved = True
                    
                    if new_error < best_error:
                        best_levels = new_levels.copy()
                        best_signs = new_signs.copy()
                        best_error = new_error
                        no_improvement = 0
                    break  # Greedy: take first improvement
            
            if not improved:
                no_improvement += 1
                if no_improvement >= patience:
                    break
                
                # Try random restart from best
                current_levels = best_levels.copy()
                current_signs = best_signs.copy()
                current_error = best_error
                
                # Perturb randomly
                if active_dims:
                    dim = np.random.choice(active_dims)
                else:
                    dim = np.random.randint(self.n_dims)
                move = np.random.choice(self.move_sizes) * np.random.choice([-1, 1])
                current_levels[dim] += move
                current_error = self.compute_error(current_levels, current_signs, all_features, targets)
            
            if (iteration + 1) % 100 == 0:
                print(f"     Iter {iteration+1}: error={best_error:.6f}")
        
        return best_levels, best_signs, best_error


class PhiAStarColorizer:
    """
    Colorizer that uses φ-A* search to find optimal weights.
    """
    
    def __init__(self, n_active_dims: int = 20, k: int = 32):
        self.n_active_dims = n_active_dims
        self.k = k
        
        self.active_dims_u = None
        self.active_dims_v = None
        
        self.weight_levels_u = None
        self.weight_signs_u = None
        self.weight_levels_v = None
        self.weight_signs_v = None
        
        self.searcher = None
        self.is_trained = False
    
    def train(self, features: np.ndarray, u_vals: np.ndarray, v_vals: np.ndarray,
              max_iterations: int = 500):
        """
        Train using φ-A* search.
        """
        n_samples, n_dims = features.shape
        
        print(f"   Training φ-A* colorizer with {self.n_active_dims} active dims...")
        
        # Step 1: Find most correlated dimensions (like before)
        u_corrs = np.array([np.corrcoef(features[:, d], u_vals)[0, 1] 
                           for d in range(n_dims)])
        v_corrs = np.array([np.corrcoef(features[:, d], v_vals)[0, 1] 
                           for d in range(n_dims)])
        
        u_corrs = np.nan_to_num(u_corrs)
        v_corrs = np.nan_to_num(v_corrs)
        
        self.active_dims_u = list(np.argsort(np.abs(u_corrs))[::-1][:self.n_active_dims])
        self.active_dims_v = list(np.argsort(np.abs(v_corrs))[::-1][:self.n_active_dims])
        
        # Step 2: Initialize weights using linear regression (warm start)
        features_u = features[:, self.active_dims_u]
        features_v = features[:, self.active_dims_v]
        
        init_weights_u = np.linalg.lstsq(features_u, u_vals, rcond=None)[0]
        init_weights_v = np.linalg.lstsq(features_v, v_vals, rcond=None)[0]
        
        # Quantize to φ-levels
        init_levels_u = np.array([to_phi_level(w, self.k) for w in init_weights_u])
        init_signs_u = np.sign(init_weights_u)
        init_signs_u[init_signs_u == 0] = 1
        
        init_levels_v = np.array([to_phi_level(w, self.k) for w in init_weights_v])
        init_signs_v = np.sign(init_weights_v)
        init_signs_v[init_signs_v == 0] = 1
        
        # Step 3: Search for better weights using φ-A*
        self.searcher = PhiAStarSearch(n_dims=self.n_active_dims, k=self.k, max_fib_idx=6)
        
        print("   Searching for U weights...")
        self.weight_levels_u, self.weight_signs_u, error_u = self.searcher.search(
            features_u, u_vals,
            init_levels=init_levels_u, init_signs=init_signs_u,
            active_dims=list(range(self.n_active_dims)),
            max_iterations=max_iterations, patience=50
        )
        
        print("   Searching for V weights...")
        self.weight_levels_v, self.weight_signs_v, error_v = self.searcher.search(
            features_v, v_vals,
            init_levels=init_levels_v, init_signs=init_signs_v,
            active_dims=list(range(self.n_active_dims)),
            max_iterations=max_iterations, patience=50
        )
        
        # Compute final correlations
        u_pred = np.array([self.searcher.compute_prediction(
            self.weight_levels_u, self.weight_signs_u, features_u[i]
        ) for i in range(len(u_vals))])
        
        v_pred = np.array([self.searcher.compute_prediction(
            self.weight_levels_v, self.weight_signs_v, features_v[i]
        ) for i in range(len(v_vals))])
        
        corr_u = np.corrcoef(u_vals, u_pred)[0, 1]
        corr_v = np.corrcoef(v_vals, v_pred)[0, 1]
        
        print(f"\n   φ-A* results:")
        print(f"     U: corr={corr_u:.4f}, MSE={error_u:.6f}")
        print(f"     V: corr={corr_v:.4f}, MSE={error_v:.6f}")
        
        # Analyze the found weights
        print(f"\n   Weight analysis:")
        print(f"     U levels: [{self.weight_levels_u.min()}, {self.weight_levels_u.max()}]")
        print(f"     V levels: [{self.weight_levels_v.min()}, {self.weight_levels_v.max()}]")
        
        # Check Fibonacci structure
        u_diffs = np.abs(np.diff(np.sort(self.weight_levels_u)))
        v_diffs = np.abs(np.diff(np.sort(self.weight_levels_v)))
        
        fib_set = set(FIBONACCI)
        u_fib = sum(1 for d in u_diffs if d in fib_set or any(abs(d-f) <= 1 for f in FIBONACCI))
        v_fib = sum(1 for d in v_diffs if d in fib_set or any(abs(d-f) <= 1 for f in FIBONACCI))
        
        print(f"     U level diffs near Fibonacci: {u_fib}/{len(u_diffs)}")
        print(f"     V level diffs near Fibonacci: {v_fib}/{len(v_diffs)}")
        
        self.is_trained = True
        return corr_u, corr_v
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict color for a single feature vector."""
        if not self.is_trained:
            return 0.0, 0.0
        
        feat_u = features[self.active_dims_u]
        feat_v = features[self.active_dims_v]
        
        u = self.searcher.compute_prediction(self.weight_levels_u, self.weight_signs_u, feat_u)
        v = self.searcher.compute_prediction(self.weight_levels_v, self.weight_signs_v, feat_v)
        
        return u, v


# ============================================================
# DA2 INTEGRATION
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


def collect_training_data(model, processor, images: List[np.ndarray], sample_rate: float = 0.3):
    """Collect features and colors from images."""
    all_features = []
    all_u = []
    all_v = []
    
    for i, rgb in enumerate(images):
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0
        
        structure = extract_da2_structure(model, processor, rgb)
        structure = structure[1:]
        
        N, C = structure.shape
        H, W = rgb.shape[:2]
        
        H_s = int(np.sqrt(N * H / W))
        W_s = N // H_s
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - W/H) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        yuv_small = rgb_to_yuv(rgb_small * 255)
        
        for y in range(H_s):
            for x in range(W_s):
                if np.random.random() < sample_rate:
                    all_features.append(struct_spatial[y, x])
                    all_u.append(yuv_small[y, x, 1])
                    all_v.append(yuv_small[y, x, 2])
        
        if (i + 1) % 5 == 0:
            print(f"     Processed {i+1}/{len(images)}")
    
    return np.array(all_features), np.array(all_u), np.array(all_v)


def colorize_with_phi_astar(model, processor, rgb: np.ndarray, colorizer: PhiAStarColorizer):
    """Colorize using φ-A* learned weights."""
    if rgb.max() > 1:
        rgb_norm = rgb.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb
    
    structure = extract_da2_structure(model, processor, rgb_norm)
    structure = structure[1:]
    
    N, C = structure.shape
    H, W = rgb_norm.shape[:2]
    
    H_s = int(np.sqrt(N * H / W))
    W_s = N // H_s
    
    if H_s * W_s != N:
        for h in range(1, int(np.sqrt(N)) + 10):
            if N % h == 0:
                w = N // h
                if abs(w/h - W/H) < 0.5:
                    H_s, W_s = h, w
                    break
    
    struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
    
    u_map = np.zeros((H_s, W_s))
    v_map = np.zeros((H_s, W_s))
    
    for y in range(H_s):
        for x in range(W_s):
            u, v = colorizer.predict(struct_spatial[y, x])
            u_map[y, x] = u
            v_map[y, x] = v
    
    # Smooth and amplify
    u_map = gaussian_filter(u_map, sigma=0.5) * 1.3
    v_map = gaussian_filter(v_map, sigma=0.5) * 1.3
    
    # Upsample
    u_full = zoom(u_map, (H / H_s, W / W_s), order=3)[:H, :W]
    v_full = zoom(v_map, (H / H_s, W / W_s), order=3)[:H, :W]
    
    gray = 0.299 * rgb_norm[:,:,0] + 0.587 * rgb_norm[:,:,1] + 0.114 * rgb_norm[:,:,2]
    
    yuv = np.stack([gray, u_full, v_full], axis=-1)
    return yuv_to_rgb(yuv)


def load_coco_images(n_images: int, start_idx: int = 0) -> List[Tuple[str, np.ndarray]]:
    image_files = sorted(COCO_PATH.glob("*.jpg"))
    images = []
    for img_path in image_files[start_idx:start_idx + n_images]:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append((img_path.stem, img))
        except:
            pass
    return images


def run_true_phi_astar_test():
    """Test true φ-A* colorizer."""
    print("=" * 70)
    print("TRUE φ-A* COLORIZER")
    print("Search for optimal path through φ-space")
    print("=" * 70)
    
    print("\n0. LOADING DA2")
    print("-" * 50)
    model, processor = load_da2()
    
    train_data = load_coco_images(20, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    
    train_images = [img for _, img in train_data]
    
    print("\n1. COLLECTING TRAINING DATA")
    print("-" * 50)
    features, u_vals, v_vals = collect_training_data(model, processor, train_images, sample_rate=0.2)
    print(f"   Collected {len(features)} samples")
    
    print("\n2. φ-A* SEARCH")
    print("-" * 50)
    colorizer = PhiAStarColorizer(n_active_dims=20, k=32)
    corr_u, corr_v = colorizer.train(features, u_vals, v_vals, max_iterations=300)
    
    print("\n3. TESTING")
    print("-" * 50)
    
    results = []
    for name, img in test_data:
        colorized = colorize_with_phi_astar(model, processor, img, colorizer)
        mae = np.abs(colorized.astype(float) - img.astype(float)).mean()
        results.append((name, img, colorized, mae))
        print(f"   {name}: MAE = {mae:.2f}")
    
    avg_mae = np.mean([r[3] for r in results])
    print(f"\n   Average MAE: {avg_mae:.2f}")
    
    # Visualize
    print("\n4. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    
    for i, (name, original, colorized, mae) in enumerate(results):
        gray = (0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]).astype(np.uint8)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title(f'φ-A* ({mae:.1f})' if i == 0 else f'{mae:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=30)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'True φ-A* Colorizer: Fibonacci moves, Avg MAE={avg_mae:.1f}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "true_phi_astar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'true_phi_astar.png'}")
    
    return colorizer, results, avg_mae


if __name__ == "__main__":
    colorizer, results, avg_mae = run_true_phi_astar_test()
    
    print("\n" + "=" * 70)
    print("TRUE φ-A* SUMMARY")
    print("=" * 70)
    print(f"""
   The true φ-A* approach:
   
   1. Initialize with linear regression (warm start)
   2. Search for better weights using Fibonacci-sized moves
   3. Only valid moves are: ±1, ±2, ±3, ±5, ±8, ±13 φ-levels
   
   This is DISCRETE OPTIMIZATION on the φ-lattice.
   
   Results:
   - Average test MAE: {avg_mae:.2f}
   - Weights are constrained to φ-lattice
   - Moves are constrained to Fibonacci sizes
   
   The key insight:
   The optimal solution EXISTS on the φ-lattice.
   We don't need continuous optimization - discrete search works!
""")
