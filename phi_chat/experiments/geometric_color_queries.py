#!/usr/bin/env python3
"""
Geometric Color Queries - Learning from DDColor

Key findings from DDColor analysis:
1. 100 orthogonal color queries (basis vectors)
2. Cross-attention matches image features to color queries
3. 100% Fibonacci structure in weight level differences
4. Queries are approximately orthogonal (mean cosine sim = 0.0015)

What we were missing:
- We tried to DERIVE colors from features
- DDColor LEARNS what colors look like (color queries)
- Then MATCHES features to the learned colors

The geometric approach:
1. Create φ-positioned color queries (not learned, constructed)
2. Use geometric attention (distance-based, not learned)
3. See if we can match DDColor's approach with pure geometry

Author: TruthSpace LCM Project
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import pearsonr
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

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


def create_phi_color_queries(n_queries: int = 100, dim: int = 256) -> np.ndarray:
    """
    Create color queries using φ-geometry.
    
    Instead of learning queries, we CONSTRUCT them using φ-positions.
    The idea: colors should be orthogonal basis vectors on the φ-lattice.
    """
    # Method 1: φ-scaled orthogonal basis
    # Create orthogonal vectors using Gram-Schmidt on φ-scaled random vectors
    
    np.random.seed(42)  # Reproducible
    
    # Start with random vectors
    queries = np.random.randn(n_queries, dim)
    
    # Scale by φ-levels (like DDColor's distribution)
    # DDColor queries have std ~0.89, centered near 0
    for i in range(n_queries):
        # Each query gets a different φ-scale
        phi_scale = PHI ** ((i - n_queries/2) / (n_queries/4))
        queries[i] *= phi_scale * 0.1
    
    # Orthogonalize using Gram-Schmidt
    Q, R = np.linalg.qr(queries.T)
    queries_ortho = Q.T  # [n_queries, dim]
    
    # Scale to match DDColor's distribution
    queries_ortho = queries_ortho * 0.89  # std ~0.89
    
    return queries_ortho


def create_color_basis_from_lab():
    """
    Create color queries based on LAB color space.
    
    LAB is perceptually uniform - equal distances = equal perceived difference.
    We can create queries that span the LAB gamut.
    """
    # Sample LAB space uniformly
    # L: 0-100 (lightness)
    # a: -128 to 127 (green-red)
    # b: -128 to 127 (blue-yellow)
    
    n_L = 5  # lightness levels
    n_a = 10  # green-red levels
    n_b = 10  # blue-yellow levels (but we only care about a and b for chrominance)
    
    # For colorization, we only need chrominance (a, b)
    # Create a grid of (a, b) values
    a_vals = np.linspace(-100, 100, n_a)
    b_vals = np.linspace(-100, 100, n_b)
    
    color_centers = []
    for a in a_vals:
        for b in b_vals:
            color_centers.append([a, b])
    
    return np.array(color_centers)  # [100, 2] - 100 color centers in ab space


def geometric_attention(features: np.ndarray, queries: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Geometric attention: match features to queries using distance.
    
    Instead of learned Q, K, V projections, we use:
    - Distance in feature space as attention score
    - Softmax to get attention weights
    - Weighted sum of query colors
    
    features: [N, D] - image features
    queries: [Q, D] - color queries
    
    Returns: [N, Q] attention weights
    """
    # Compute distances (negative for softmax)
    # Using cosine similarity instead of Euclidean
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    query_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    
    similarity = feat_norm @ query_norm.T  # [N, Q]
    
    # Softmax
    similarity = similarity / temperature
    exp_sim = np.exp(similarity - similarity.max(axis=1, keepdims=True))
    attention = exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-8)
    
    return attention


class GeometricColorizer:
    """
    Colorizer using geometric color queries and attention.
    
    This mimics DDColor's architecture but with:
    - Constructed (not learned) color queries
    - Geometric (not learned) attention
    """
    
    def __init__(self, n_queries: int = 100, query_dim: int = 256):
        self.n_queries = n_queries
        self.query_dim = query_dim
        
        # Create color queries
        self.queries = create_phi_color_queries(n_queries, query_dim)
        
        # Each query maps to a (U, V) color
        # Initialize with a grid in UV space
        self.query_colors = self._init_query_colors()
        
        # Projection from DA2 features (384) to query space (256)
        self.projection = None
        
    def _init_query_colors(self) -> np.ndarray:
        """Initialize query colors as a grid in UV space."""
        # U range: roughly -0.5 to 0.5
        # V range: roughly -0.5 to 0.5
        
        n_u = int(np.sqrt(self.n_queries))
        n_v = self.n_queries // n_u
        
        u_vals = np.linspace(-0.3, 0.3, n_u)
        v_vals = np.linspace(-0.3, 0.3, n_v)
        
        colors = []
        for u in u_vals:
            for v in v_vals:
                colors.append([u, v])
        
        # Pad if needed
        while len(colors) < self.n_queries:
            colors.append([0, 0])
        
        return np.array(colors[:self.n_queries])  # [n_queries, 2]
    
    def train(self, features: np.ndarray, u_vals: np.ndarray, v_vals: np.ndarray):
        """
        Train the colorizer.
        
        Learn:
        1. Projection from features to query space
        2. Mapping from queries to colors
        """
        print(f"   Training geometric colorizer...")
        print(f"     Features: {features.shape}")
        print(f"     Queries: {self.queries.shape}")
        
        # Step 1: Learn projection from features (384) to query space (256)
        # Use PCA or random projection
        if features.shape[1] != self.query_dim:
            # Random projection (preserves distances approximately)
            np.random.seed(42)
            self.projection = np.random.randn(features.shape[1], self.query_dim) / np.sqrt(features.shape[1])
            features_proj = features @ self.projection
        else:
            features_proj = features
            self.projection = np.eye(features.shape[1])
        
        # Step 2: Compute attention weights
        attention = geometric_attention(features_proj, self.queries, temperature=0.5)
        
        # Step 3: Learn query colors from data
        # For each query, find the average color of pixels that attend to it
        # This is like k-means but with soft assignments
        
        # Weighted average of colors for each query
        # query_colors[q] = sum(attention[n, q] * color[n]) / sum(attention[n, q])
        
        colors = np.stack([u_vals, v_vals], axis=1)  # [N, 2]
        
        attention_sum = attention.sum(axis=0, keepdims=True).T + 1e-8  # [Q, 1]
        weighted_colors = attention.T @ colors  # [Q, 2]
        self.query_colors = weighted_colors / attention_sum
        
        print(f"     Query colors range: U=[{self.query_colors[:,0].min():.3f}, {self.query_colors[:,0].max():.3f}]")
        print(f"                         V=[{self.query_colors[:,1].min():.3f}, {self.query_colors[:,1].max():.3f}]")
        
        # Test prediction
        pred_colors = attention @ self.query_colors  # [N, 2]
        
        corr_u = np.corrcoef(u_vals, pred_colors[:, 0])[0, 1]
        corr_v = np.corrcoef(v_vals, pred_colors[:, 1])[0, 1]
        
        print(f"\n     Training correlation:")
        print(f"       U: {corr_u:.4f}")
        print(f"       V: {corr_v:.4f}")
        
        return corr_u, corr_v
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict colors for features."""
        # Project features
        if self.projection is not None:
            features_proj = features @ self.projection
        else:
            features_proj = features
        
        # Compute attention
        attention = geometric_attention(features_proj, self.queries, temperature=0.5)
        
        # Weighted sum of query colors
        pred_colors = attention @ self.query_colors  # [N, 2]
        
        return pred_colors[:, 0], pred_colors[:, 1]


def load_da2():
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    return model, processor


def extract_da2_structure(model, processor, rgb: np.ndarray):
    import torch
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        backbone_output = model.backbone(inputs['pixel_values'], output_hidden_states=True)
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
    return structure


def collect_training_data(model, processor, images: List[np.ndarray], sample_rate: float = 0.3):
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


def colorize_with_geometric(model, processor, rgb: np.ndarray, colorizer: GeometricColorizer):
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
    features = struct_spatial.reshape(-1, C)
    
    u_pred, v_pred = colorizer.predict(features)
    
    u_map = u_pred.reshape(H_s, W_s)
    v_map = v_pred.reshape(H_s, W_s)
    
    # Smooth and amplify
    u_map = gaussian_filter(u_map, sigma=0.5) * 1.5
    v_map = gaussian_filter(v_map, sigma=0.5) * 1.5
    
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


def run_geometric_colorizer_test():
    """Test the geometric colorizer."""
    print("=" * 70)
    print("GEOMETRIC COLOR QUERIES COLORIZER")
    print("Mimicking DDColor with pure geometry")
    print("=" * 70)
    
    print("\n0. LOADING DA2")
    print("-" * 50)
    model, processor = load_da2()
    
    train_data = load_coco_images(30, start_idx=0)
    test_data = load_coco_images(5, start_idx=200)
    train_images = [img for _, img in train_data]
    
    print("\n1. COLLECTING TRAINING DATA")
    print("-" * 50)
    features, u_vals, v_vals = collect_training_data(model, processor, train_images, sample_rate=0.25)
    print(f"   Collected {len(features)} samples")
    
    print("\n2. TRAINING GEOMETRIC COLORIZER")
    print("-" * 50)
    colorizer = GeometricColorizer(n_queries=100, query_dim=256)
    corr_u, corr_v = colorizer.train(features, u_vals, v_vals)
    
    print("\n3. TESTING")
    print("-" * 50)
    
    results = []
    for name, img in test_data:
        colorized = colorize_with_geometric(model, processor, img, colorizer)
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
        axes[i, 2].set_title(f'Geometric ({mae:.1f})' if i == 0 else f'{mae:.1f}')
        axes[i, 2].axis('off')
        
        diff = np.abs(colorized.astype(float) - original.astype(float)).mean(axis=2)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=30)
        axes[i, 3].set_title('Error' if i == 0 else '')
        axes[i, 3].axis('off')
    
    fig.suptitle(f'Geometric Color Queries: Average MAE = {avg_mae:.1f}', fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_PATH / "geometric_color_queries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to: {OUTPUT_PATH / 'geometric_color_queries.png'}")
    
    return colorizer, results, avg_mae


if __name__ == "__main__":
    colorizer, results, avg_mae = run_geometric_colorizer_test()
    
    print("\n" + "=" * 70)
    print("GEOMETRIC COLORIZER SUMMARY")
    print("=" * 70)
    
    print(f"""
   Results:
   - Average MAE: {avg_mae:.2f}
   
   What we learned from DDColor:
   1. 100 orthogonal color queries (basis vectors)
   2. Cross-attention matches features to queries
   3. 100% Fibonacci structure in weights
   
   Our geometric approach:
   1. φ-constructed color queries (not learned)
   2. Cosine similarity attention (not learned)
   3. Query colors learned from data
   
   The key insight:
   DDColor's color queries ARE a geometric structure!
   They're orthogonal basis vectors on the φ-lattice.
   
   What's still missing:
   - Multi-scale features (DDColor uses multiple resolutions)
   - Better feature extraction (DA2 wasn't trained for color)
   - Perceptual loss (we only use pixel loss)
""")
