#!/usr/bin/env python3
"""
φ-Based Reimplementation of DA2

Using our dimension mapping, we build a geometric decoder that:
1. Uses known dimension-to-feature correlations
2. Applies φ-scaled weights
3. Combines dimensions geometrically to reconstruct depth

This is the culmination of our reverse engineering work.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")

# ============================================================================
# DIMENSION MAPPING (from our analysis)
# ============================================================================

# Depth-encoding dimensions with their correlations
DEPTH_DIMS = {
    318: -0.614, 76: -0.466, 271: 0.380, 153: 0.368, 80: -0.338,
    32: 0.335, 234: -0.334, 321: 0.329, 4: -0.324, 278: 0.322,
    168: -0.320, 142: -0.317, 102: -0.316, 375: -0.314, 99: 0.308,
    247: 0.297, 229: 0.296, 62: -0.294, 383: -0.294, 23: 0.659,
}

# Y-position encoding (vertical)
Y_POSITION_DIMS = {
    23: 0.955, 164: -0.447, 142: -0.333, 70: -0.283, 186: 0.280,
}

# X-position encoding (horizontal)
X_POSITION_DIMS = {
    181: -0.793, 379: -0.748, 288: 0.675, 383: -0.613, 28: 0.555,
}

# Center distance (close-up detection)
CENTER_DIST_DIMS = {
    262: -0.780, 311: -0.274, 343: -0.254, 151: 0.248, 266: -0.234,
}

# Luminance encoding
LUMINANCE_DIMS = {
    323: -0.716, 172: -0.297, 89: 0.219, 200: 0.194, 92: -0.183,
}

# Edge encoding
EDGE_DIMS = {
    359: -0.263, 121: 0.190, 349: 0.171, 46: 0.169, 290: 0.138,
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def load_da2():
    """Load DA2 model."""
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    
    return model, processor


def extract_structure(model, processor, rgb: np.ndarray):
    """Extract DA2's backbone structure."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
        
        full_output = model(inputs['pixel_values'])
        da2_depth = full_output.predicted_depth.squeeze().numpy()
    
    return structure, _normalize(da2_depth)


# ============================================================================
# φ-DECODER
# ============================================================================

class PhiDecoder:
    """
    Geometric depth decoder using φ-scaled dimension weights.
    """
    
    def __init__(self):
        # Build weight vector for all 384 dimensions
        self.weights = np.zeros(384)
        
        # Apply φ-scaling based on correlation strength
        for dim, corr in DEPTH_DIMS.items():
            if abs(corr) > 0.3:
                self.weights[dim] = corr / PHI
            else:
                self.weights[dim] = corr / (PHI ** 2)
        
        # Normalize weights
        self.weights = self.weights / np.abs(self.weights).sum()
        
        # Store dimension info for analysis
        self.depth_dims = list(DEPTH_DIMS.keys())
        self.n_active_dims = len(self.depth_dims)
    
    def decode(self, structure: np.ndarray) -> np.ndarray:
        """
        Decode depth from structure using φ-weighted dimensions.
        
        Args:
            structure: [N, 384] or [H, W, 384] tensor
        
        Returns:
            depth: [N] or [H, W] depth values
        """
        if len(structure.shape) == 2:
            # [N, 384] -> [N]
            depth = structure @ self.weights
        else:
            # [H, W, 384] -> [H, W]
            depth = np.tensordot(structure, self.weights, axes=([2], [0]))
        
        return _normalize(depth)


class AdaptivePhiDecoder:
    """
    Adaptive decoder that adjusts based on image characteristics.
    
    Uses center_dist dimensions to detect close-ups and adjust strategy.
    """
    
    def __init__(self):
        self.base_decoder = PhiDecoder()
        
        # Close-up detection weights
        self.closeup_weights = np.zeros(384)
        for dim, corr in CENTER_DIST_DIMS.items():
            self.closeup_weights[dim] = corr
        
        # Luminance adjustment weights
        self.luminance_weights = np.zeros(384)
        for dim, corr in LUMINANCE_DIMS.items():
            self.luminance_weights[dim] = corr / PHI
    
    def detect_closeup(self, structure: np.ndarray) -> float:
        """
        Detect if image is a close-up based on center_dist dimensions.
        
        Returns score 0-1 (higher = more close-up like)
        """
        # Use mean activation of center_dist dimensions
        if len(structure.shape) == 2:
            features = structure.mean(axis=0)
        else:
            features = structure.mean(axis=(0, 1))
        
        # Center distance score (inverted because negative correlation)
        center_score = -np.dot(features, self.closeup_weights)
        
        # Normalize to 0-1
        return np.clip((center_score + 1) / 2, 0, 1)
    
    def decode(self, structure: np.ndarray) -> np.ndarray:
        """
        Adaptive decoding based on image type.
        """
        closeup_score = self.detect_closeup(structure)
        
        # Base depth from φ-decoder
        base_depth = self.base_decoder.decode(structure)
        
        # For close-ups, add luminance contribution
        if len(structure.shape) == 2:
            lum_depth = structure @ self.luminance_weights
        else:
            lum_depth = np.tensordot(structure, self.luminance_weights, axes=([2], [0]))
        
        lum_depth = _normalize(lum_depth)
        
        # Blend based on closeup score
        # Close-ups: more luminance, less base
        # Normal: more base, less luminance
        alpha = closeup_score * 0.3  # Max 30% luminance contribution
        
        combined = (1 - alpha) * base_depth + alpha * lum_depth
        
        return _normalize(combined)


# ============================================================================
# TESTING
# ============================================================================

def test_phi_decoder(model, processor, n_test: int = 12):
    """Test the φ-decoder against DA2."""
    print("\n" + "=" * 70)
    print("TESTING φ-DECODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Include outliers
    test_ids = available_ids[40:40+n_test]
    outlier_ids = ["000000002587", "000000003501"]
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    # Create decoders
    basic_decoder = PhiDecoder()
    adaptive_decoder = AdaptivePhiDecoder()
    
    print(f"\n  Basic φ-Decoder: {basic_decoder.n_active_dims} active dimensions")
    print(f"  Adaptive φ-Decoder: adds luminance adjustment for close-ups")
    
    results = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        structure, da2_depth = extract_structure(model, processor, rgb)
        
        # Skip CLS token
        structure = structure[1:]
        N, C = structure.shape
        
        depth_h, depth_w = da2_depth.shape
        H_s = depth_h // 14
        W_s = depth_w // 14
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        # Decode with both decoders
        basic_depth = basic_decoder.decode(struct_spatial)
        adaptive_depth = adaptive_decoder.decode(struct_spatial)
        closeup_score = adaptive_decoder.detect_closeup(struct_spatial)
        
        # Upscale
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        basic_resized = _normalize(zoom(basic_depth, (zoom_h, zoom_w), order=3))
        adaptive_resized = _normalize(zoom(adaptive_depth, (zoom_h, zoom_w), order=3))
        
        # Metrics
        basic_corr = np.corrcoef(basic_resized.flatten(), da2_depth.flatten())[0, 1]
        adaptive_corr = np.corrcoef(adaptive_resized.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2_depth': da2_depth,
            'basic_depth': basic_resized,
            'adaptive_depth': adaptive_resized,
            'basic_corr': basic_corr,
            'adaptive_corr': adaptive_corr,
            'closeup_score': closeup_score,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        print(f"  {img_id}: basic={basic_corr:.3f}, adaptive={adaptive_corr:.3f}, "
              f"closeup={closeup_score:.2f}{marker}")
    
    return results


def create_visualization(results: list):
    """Visualize φ-decoder results."""
    
    # Sort: outliers first
    results_sorted = sorted(results, key=lambda x: -x['is_outlier'])
    
    n_images = min(len(results_sorted), 10)
    
    fig = plt.figure(figsize=(20, 3.5 * n_images + 1))
    fig.suptitle('φ-Based DA2 Reimplementation\n'
                 'Using dimension mapping to decode depth geometrically',
                 fontsize=14, fontweight='bold', y=0.99)
    
    gs = gridspec.GridSpec(n_images + 1, 5, figure=fig, hspace=0.2, wspace=0.1,
                          height_ratios=[1] * n_images + [0.3])
    
    headers = ['Original', 'DA2 Depth', 'φ-Decoder', 'Adaptive φ', 'Difference']
    
    for row, r in enumerate(results_sorted[:n_images]):
        # Original
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(r['rgb'])
        if row == 0:
            ax.set_title(headers[0], fontsize=10)
        if r['is_outlier']:
            ax.set_ylabel(f"OUTLIER\n{r['img_id'][-4:]}", fontsize=8, color='red')
        ax.axis('off')
        
        # DA2 depth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(r['da2_depth'], cmap='magma')
        if row == 0:
            ax.set_title(headers[1], fontsize=10)
        ax.axis('off')
        
        # Basic φ-decoder
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(r['basic_depth'], cmap='magma')
        title = f'{headers[2]} ({r["basic_corr"]:.3f})' if row == 0 else f'{r["basic_corr"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Adaptive φ-decoder
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(r['adaptive_depth'], cmap='magma')
        title = f'{headers[3]} ({r["adaptive_corr"]:.3f})' if row == 0 else f'{r["adaptive_corr"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Difference
        ax = fig.add_subplot(gs[row, 4])
        diff = r['adaptive_depth'] - r['da2_depth']
        ax.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        if row == 0:
            ax.set_title(headers[4], fontsize=10)
        ax.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[n_images, :])
    ax_summary.axis('off')
    
    avg_basic = np.mean([r['basic_corr'] for r in results])
    avg_adaptive = np.mean([r['adaptive_corr'] for r in results])
    
    outlier_results = [r for r in results if r['is_outlier']]
    if outlier_results:
        outlier_basic = np.mean([r['basic_corr'] for r in outlier_results])
        outlier_adaptive = np.mean([r['adaptive_corr'] for r in outlier_results])
    else:
        outlier_basic = outlier_adaptive = 0
    
    summary = f"""
    φ-DECODER RESULTS
    Basic φ-Decoder: {len(DEPTH_DIMS)} dimensions, φ-scaled weights
    Adaptive φ-Decoder: adds luminance adjustment for close-ups
    
    Overall: Basic={avg_basic:.4f}, Adaptive={avg_adaptive:.4f}
    Outliers: Basic={outlier_basic:.4f}, Adaptive={outlier_adaptive:.4f}
    
    Key: We decode DA2's depth using ONLY the dimension mapping we discovered.
    No training, no learned weights - just geometric interpretation of DA2's structure.
    """
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    output_file = OUTPUT_PATH / "da2_phi_reimplementation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Test φ-decoder
    results = test_phi_decoder(model, processor, n_test=10)
    
    # Visualize
    viz_file = create_visualization(results)
    
    # Summary
    avg_basic = np.mean([r['basic_corr'] for r in results])
    avg_adaptive = np.mean([r['adaptive_corr'] for r in results])
    
    outlier_results = [r for r in results if r['is_outlier']]
    
    print("\n" + "=" * 70)
    print("φ-REIMPLEMENTATION COMPLETE")
    print("=" * 70)
    print(f"\n  Basic φ-Decoder: {avg_basic:.4f} average correlation")
    print(f"  Adaptive φ-Decoder: {avg_adaptive:.4f} average correlation")
    
    if outlier_results:
        print(f"\n  Outlier Performance:")
        for r in outlier_results:
            print(f"    {r['img_id']}: basic={r['basic_corr']:.3f}, adaptive={r['adaptive_corr']:.3f}")
    
    print()
    print("  This decoder uses NO learned weights - only the dimension mapping")
    print("  we discovered through correlation analysis.")
    print()
    print("  The φ-scaling provides a geometric interpretation of DA2's structure.")
