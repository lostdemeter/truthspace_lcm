#!/usr/bin/env python3
"""
Multi-Scale φ-Decoder: Using Multiple Backbone Layers

DA2's neck uses features from multiple backbone layers and fuses them.
We'll extract all backbone hidden states and do φ-weighted fusion.

The hypothesis: each layer captures different scales of information,
and φ-weighting can combine them optimally.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, gaussian_filter
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")
COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


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


def extract_all_layers(model, processor, rgb: np.ndarray):
    """Extract all backbone hidden states."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        
        # Get all hidden states
        hidden_states = []
        for hs in backbone_output.hidden_states:
            hidden_states.append(hs.squeeze().numpy())
        
        # Get final depth
        full_output = model(inputs['pixel_values'])
        da2_depth = full_output.predicted_depth.squeeze().numpy()
    
    return hidden_states, _normalize(da2_depth)


def analyze_layer_structure(hidden_states: list):
    """Analyze the structure of each layer."""
    print("\n" + "=" * 70)
    print("BACKBONE LAYER ANALYSIS")
    print("=" * 70)
    
    print(f"\n  Number of layers: {len(hidden_states)}")
    print("\n  Layer shapes:")
    for i, hs in enumerate(hidden_states):
        print(f"    Layer {i}: {hs.shape}")
    
    return len(hidden_states)


def collect_multilayer_data(model, processor, n_images: int = 25):
    """Collect data from multiple backbone layers."""
    print("\n  Collecting multi-layer data...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # First pass: determine layer structure
    sample_path = COCO_VAL_PATH / f"{available_ids[0]}.jpg"
    sample_rgb = np.array(Image.open(sample_path).convert("RGB")).astype(np.float32) / 255.0
    sample_layers, _ = extract_all_layers(model, processor, sample_rgb)
    
    n_layers = len(sample_layers)
    analyze_layer_structure(sample_layers)
    
    # We'll use layers that have the same spatial structure (skip first few)
    # DINOv2 typically has: [CLS + patches] format for each layer
    usable_layers = []
    for i, layer in enumerate(sample_layers):
        if len(layer.shape) == 2 and layer.shape[0] > 100:  # Has patches
            usable_layers.append(i)
    
    print(f"\n  Usable layers (with patch structure): {usable_layers}")
    
    # Collect data from last 4 layers (most relevant for depth)
    layers_to_use = usable_layers[-4:] if len(usable_layers) >= 4 else usable_layers
    print(f"  Using layers: {layers_to_use}")
    
    all_layer_features = {layer_idx: [] for layer_idx in layers_to_use}
    all_depths = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        hidden_states, da2_depth = extract_all_layers(model, processor, rgb)
        
        depth_h, depth_w = da2_depth.shape
        
        for layer_idx in layers_to_use:
            layer = hidden_states[layer_idx]
            
            # Skip CLS token
            if len(layer.shape) == 2:
                layer = layer[1:]
            
            N, C = layer.shape
            H_s = depth_h // 14
            W_s = depth_w // 14
            
            if H_s * W_s != N:
                for h in range(1, int(np.sqrt(N)) + 10):
                    if N % h == 0:
                        w = N // h
                        if abs(w/h - depth_w/depth_h) < 0.5:
                            H_s, W_s = h, w
                            break
            
            layer_spatial = layer[:H_s*W_s].reshape(H_s, W_s, C)
            
            # Downsample depth to match
            depth_small = np.array(
                Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))
            ) / 255.0
            
            # Collect patches
            for y in range(H_s):
                for x in range(W_s):
                    all_layer_features[layer_idx].append(layer_spatial[y, x])
                    
                    # Only add depth once (from first layer)
                    if layer_idx == layers_to_use[0]:
                        all_depths.append(depth_small[y, x])
        
        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{n_images}")
    
    # Convert to arrays
    for layer_idx in layers_to_use:
        all_layer_features[layer_idx] = np.array(all_layer_features[layer_idx])
    all_depths = np.array(all_depths)
    
    print(f"\n  Collected {len(all_depths)} patches")
    for layer_idx in layers_to_use:
        print(f"    Layer {layer_idx}: {all_layer_features[layer_idx].shape}")
    
    return all_layer_features, all_depths, layers_to_use


def build_multilayer_phi_decoder(layer_features: dict, depths: np.ndarray, 
                                  layers_to_use: list, n_dims_per_layer: int = 30):
    """Build φ-decoder using multiple layers with φ-weighted fusion."""
    print("\n" + "=" * 70)
    print("BUILDING MULTI-LAYER φ-DECODER")
    print("=" * 70)
    
    # For each layer, find top depth-correlated dimensions
    layer_weights = {}
    layer_dims = {}
    layer_corrs = {}
    
    for layer_idx in layers_to_use:
        features = layer_features[layer_idx]
        
        correlations = []
        for dim in range(features.shape[1]):
            corr, _ = pearsonr(features[:, dim], depths)
            correlations.append((dim, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        dims = [c[0] for c in correlations[:n_dims_per_layer]]
        corrs = [c[1] for c in correlations[:n_dims_per_layer]]
        
        layer_dims[layer_idx] = dims
        layer_corrs[layer_idx] = corrs
        
        # Build weights for this layer
        weights = np.zeros(features.shape[1])
        for i, dim in enumerate(dims):
            weights[dim] = corrs[i]  # Start with correlation as weight
        
        layer_weights[layer_idx] = weights
        
        # Report
        max_corr = max(abs(c) for c in corrs)
        print(f"\n  Layer {layer_idx}:")
        print(f"    Top dims: {dims[:5]}")
        print(f"    Max correlation: {max_corr:.4f}")
    
    # Now optimize φ-weights for layer fusion
    print("\n  Optimizing φ-weighted layer fusion...")
    
    def compute_layer_predictions(layer_features, layer_weights):
        """Compute prediction from each layer."""
        preds = {}
        for layer_idx, features in layer_features.items():
            weights = layer_weights[layer_idx]
            pred = features @ weights
            pred = _normalize(pred)
            preds[layer_idx] = pred
        return preds
    
    layer_preds = compute_layer_predictions(layer_features, layer_weights)
    
    # Stack predictions for fusion
    pred_stack = np.column_stack([layer_preds[idx] for idx in layers_to_use])
    
    def objective(fusion_exponents):
        """Optimize fusion weights as φ^exponent."""
        fusion_weights = np.array([PHI ** exp for exp in fusion_exponents])
        fusion_weights = fusion_weights / fusion_weights.sum()
        
        fused = pred_stack @ fusion_weights
        fused = _normalize(fused)
        
        corr = np.corrcoef(fused, depths)[0, 1]
        return -corr
    
    # Initialize: later layers get higher weight (φ-scaled)
    n_layers = len(layers_to_use)
    init_exponents = np.array([i * 0.5 for i in range(n_layers)])
    
    result = minimize(
        objective,
        init_exponents,
        method='L-BFGS-B',
        bounds=[(-2, 3) for _ in range(n_layers)]
    )
    
    optimal_exponents = result.x
    optimal_weights = np.array([PHI ** exp for exp in optimal_exponents])
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    fusion_corr = -result.fun
    
    print(f"\n  Fusion results:")
    print(f"    Optimal exponents: {[f'{e:.2f}' for e in optimal_exponents]}")
    print(f"    Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")
    print(f"    Fused correlation: {fusion_corr:.4f}")
    
    # Compare to single-layer
    for layer_idx in layers_to_use:
        single_corr = np.corrcoef(layer_preds[layer_idx], depths)[0, 1]
        print(f"    Layer {layer_idx} alone: {single_corr:.4f}")
    
    return layer_weights, layer_dims, optimal_exponents, fusion_corr


def test_multilayer_decoder(model, processor, layer_weights: dict, layer_dims: dict,
                            fusion_exponents: np.ndarray, layers_to_use: list, n_test: int = 10):
    """Test multi-layer φ-decoder on images."""
    print("\n" + "=" * 70)
    print("TESTING MULTI-LAYER φ-DECODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[30:30+n_test]
    outlier_ids = ["000000002587", "000000003501"]
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    fusion_weights = np.array([PHI ** exp for exp in fusion_exponents])
    fusion_weights = fusion_weights / fusion_weights.sum()
    
    results = []
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        hidden_states, da2_depth = extract_all_layers(model, processor, rgb)
        
        depth_h, depth_w = da2_depth.shape
        
        # Get predictions from each layer
        layer_preds = []
        
        for layer_idx in layers_to_use:
            layer = hidden_states[layer_idx]
            
            if len(layer.shape) == 2:
                layer = layer[1:]
            
            N, C = layer.shape
            H_s = depth_h // 14
            W_s = depth_w // 14
            
            if H_s * W_s != N:
                for h in range(1, int(np.sqrt(N)) + 10):
                    if N % h == 0:
                        w = N // h
                        if abs(w/h - depth_w/depth_h) < 0.5:
                            H_s, W_s = h, w
                            break
            
            layer_spatial = layer[:H_s*W_s].reshape(H_s, W_s, C)
            
            # Decode this layer
            weights = layer_weights[layer_idx]
            pred = np.tensordot(layer_spatial, weights, axes=([2], [0]))
            pred = _normalize(pred)
            
            # Upscale
            zoom_h = depth_h / H_s
            zoom_w = depth_w / W_s
            pred_upscaled = zoom(pred, (zoom_h, zoom_w), order=3)
            pred_upscaled = _normalize(pred_upscaled)
            
            layer_preds.append(pred_upscaled)
        
        # Fuse layers with φ-weights
        fused = np.zeros_like(layer_preds[0])
        for i, pred in enumerate(layer_preds):
            fused += fusion_weights[i] * pred
        fused = _normalize(fused)
        
        # Also get single-layer (last layer) for comparison
        single_layer = layer_preds[-1]
        
        # Metrics
        corr_single = np.corrcoef(single_layer.flatten(), da2_depth.flatten())[0, 1]
        corr_fused = np.corrcoef(fused.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2': da2_depth,
            'single': single_layer,
            'fused': fused,
            'corr_single': corr_single,
            'corr_fused': corr_fused,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        improvement = corr_fused - corr_single
        print(f"  {img_id}: single={corr_single:.3f}, fused={corr_fused:.3f} ({improvement:+.3f}){marker}")
    
    return results


def create_visualization(results: list, fusion_exponents: np.ndarray, layers_to_use: list):
    """Visualize multi-layer fusion results."""
    
    fig = plt.figure(figsize=(18, 3.5 * len(results) + 1.5))
    fig.suptitle('Multi-Layer φ-Decoder: φ-Weighted Layer Fusion\n'
                 f'Layers {layers_to_use}, exponents: {[f"{e:.2f}" for e in fusion_exponents]}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(len(results) + 1, 5, figure=fig, hspace=0.15, wspace=0.05,
                          height_ratios=[1] * len(results) + [0.3])
    
    for row, r in enumerate(results):
        # Original
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(r['rgb'])
        label = f"{'OUTLIER ' if r['is_outlier'] else ''}{r['img_id'][-4:]}"
        ax.set_ylabel(label, fontsize=8, color='red' if r['is_outlier'] else 'black')
        if row == 0:
            ax.set_title('Original', fontsize=10)
        ax.axis('off')
        
        # DA2
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(r['da2'], cmap='magma')
        if row == 0:
            ax.set_title('DA2', fontsize=10)
        ax.axis('off')
        
        # Single layer
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(r['single'], cmap='magma')
        title = f'Single Layer ({r["corr_single"]:.3f})' if row == 0 else f'{r["corr_single"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Fused
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(r['fused'], cmap='magma')
        title = f'φ-Fused ({r["corr_fused"]:.3f})' if row == 0 else f'{r["corr_fused"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Difference
        ax = fig.add_subplot(gs[row, 4])
        diff = r['fused'] - r['da2']
        ax.imshow(diff, cmap='RdBu', vmin=-0.15, vmax=0.15)
        if row == 0:
            ax.set_title('Difference', fontsize=10)
        ax.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[len(results), :])
    ax_summary.axis('off')
    
    avg_single = np.mean([r['corr_single'] for r in results])
    avg_fused = np.mean([r['corr_fused'] for r in results])
    improvement = avg_fused - avg_single
    
    normal_results = [r for r in results if not r['is_outlier']]
    outlier_results = [r for r in results if r['is_outlier']]
    
    normal_fused = np.mean([r['corr_fused'] for r in normal_results]) if normal_results else 0
    outlier_fused = np.mean([r['corr_fused'] for r in outlier_results]) if outlier_results else 0
    
    summary = f"""
    MULTI-LAYER φ-DECODER RESULTS
    
    Single layer (last):   {avg_single:.4f}
    φ-Fused (4 layers):    {avg_fused:.4f} ({improvement:+.4f})
    
    Normal images:         {normal_fused:.4f}
    Outliers:              {outlier_fused:.4f}
    
    Fusion weights: {[f'φ^{e:.1f}' for e in fusion_exponents]}
    """
    color = 'lightgreen' if avg_fused > 0.93 else 'lightyellow'
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color))
    
    output_file = OUTPUT_PATH / "da2_multiscale_phi.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect multi-layer data
    layer_features, depths, layers_to_use = collect_multilayer_data(
        model, processor, n_images=25
    )
    
    # Build multi-layer φ-decoder
    layer_weights, layer_dims, fusion_exponents, train_corr = build_multilayer_phi_decoder(
        layer_features, depths, layers_to_use, n_dims_per_layer=30
    )
    
    # Test
    results = test_multilayer_decoder(
        model, processor, layer_weights, layer_dims, 
        fusion_exponents, layers_to_use, n_test=8
    )
    
    # Visualize
    viz_file = create_visualization(results, fusion_exponents, layers_to_use)
    
    # Summary
    avg_single = np.mean([r['corr_single'] for r in results])
    avg_fused = np.mean([r['corr_fused'] for r in results])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Single layer: {avg_single:.4f}")
    print(f"  φ-Fused: {avg_fused:.4f}")
    print(f"  Improvement: {avg_fused - avg_single:+.4f}")
