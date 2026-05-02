#!/usr/bin/env python3
"""
DA2 Neck/Head Analysis: What Post-Processing Does DA2 Apply?

The φ-decoder output looks blocky compared to DA2's smooth output.
DA2 has a "neck" and "head" that process the backbone features.

Let's analyze:
1. What does DA2's neck/head architecture look like?
2. Can we extract intermediate outputs to see the refinement stages?
3. Can we replicate this with φ-geometric post-processing?

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, gaussian_filter, sobel, uniform_filter
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


def analyze_da2_architecture(model):
    """Analyze DA2's architecture to understand the neck/head."""
    print("\n" + "=" * 70)
    print("DA2 ARCHITECTURE ANALYSIS")
    print("=" * 70)
    
    print("\n  Model components:")
    for name, module in model.named_children():
        print(f"    {name}: {type(module).__name__}")
        
        # Look deeper into neck and head
        if name in ['neck', 'head']:
            print(f"\n    {name} details:")
            for sub_name, sub_module in module.named_children():
                print(f"      {sub_name}: {type(sub_module).__name__}")
                
                # Even deeper for key components
                if hasattr(sub_module, 'named_children'):
                    for ssn, ssm in list(sub_module.named_children())[:3]:
                        print(f"        {ssn}: {type(ssm).__name__}")
    
    return model


def extract_all_stages(model, processor, rgb: np.ndarray):
    """Extract outputs from all stages of DA2."""
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    stages = {}
    
    with torch.no_grad():
        # Stage 1: Backbone
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        stages['backbone'] = backbone_output.hidden_states[-1].squeeze().numpy()
        
        # Get all hidden states
        for i, hs in enumerate(backbone_output.hidden_states):
            stages[f'backbone_layer_{i}'] = hs.squeeze().numpy()
        
        # Stage 2: Neck (feature reassembly)
        # The neck takes backbone features and processes them
        backbone_features = backbone_output.feature_maps
        
        # Try to get neck output
        try:
            neck_output = model.neck(backbone_features)
            if isinstance(neck_output, (list, tuple)):
                for i, no in enumerate(neck_output):
                    stages[f'neck_{i}'] = no.squeeze().numpy()
            else:
                stages['neck'] = neck_output.squeeze().numpy()
        except Exception as e:
            print(f"    Could not extract neck output: {e}")
        
        # Stage 3: Full output
        full_output = model(inputs['pixel_values'])
        stages['final'] = full_output.predicted_depth.squeeze().numpy()
    
    return stages


def analyze_spatial_refinement(stages: dict):
    """Analyze how spatial resolution changes through stages."""
    print("\n" + "=" * 70)
    print("SPATIAL REFINEMENT ANALYSIS")
    print("=" * 70)
    
    print("\n  Stage shapes:")
    for name, data in stages.items():
        if isinstance(data, np.ndarray):
            print(f"    {name}: {data.shape}")


def test_post_processing_methods(model, processor, rgb: np.ndarray):
    """Test different post-processing methods to match DA2."""
    import torch
    from scipy.optimize import minimize
    
    print("\n" + "=" * 70)
    print("TESTING POST-PROCESSING METHODS")
    print("=" * 70)
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        backbone_output = model.backbone(inputs['pixel_values'], output_hidden_states=True)
        structure = backbone_output.hidden_states[-1].squeeze().numpy()
        
        full_output = model(inputs['pixel_values'])
        da2_depth = _normalize(full_output.predicted_depth.squeeze().numpy())
    
    # Build simple φ-decoder
    structure = structure[1:]  # Skip CLS
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
    
    # Simple weighted sum (use mean as proxy)
    phi_raw = struct_spatial.mean(axis=2)
    phi_raw = _normalize(phi_raw)
    
    # Upscale to DA2 resolution
    zoom_h = depth_h / H_s
    zoom_w = depth_w / W_s
    
    results = {}
    
    # Method 1: Simple bicubic
    phi_bicubic = zoom(phi_raw, (zoom_h, zoom_w), order=3)
    phi_bicubic = _normalize(phi_bicubic)
    results['bicubic'] = phi_bicubic
    
    # Method 2: Bicubic + Gaussian smoothing
    phi_smooth = gaussian_filter(phi_bicubic, sigma=2.0)
    phi_smooth = _normalize(phi_smooth)
    results['bicubic+gaussian'] = phi_smooth
    
    # Method 3: Bilateral-like filter (edge-preserving)
    # Approximate with guided filter approach
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    gray_resized = np.array(Image.fromarray((gray * 255).astype(np.uint8)).resize((depth_w, depth_h))) / 255.0
    
    # Edge-aware smoothing: smooth more where image is smooth
    edges = np.sqrt(sobel(gray_resized, axis=0)**2 + sobel(gray_resized, axis=1)**2)
    edges = _normalize(edges)
    
    # Blend: keep original where edges are strong, smooth where edges are weak
    smooth_weight = 1 - edges
    phi_edge_aware = smooth_weight * phi_smooth + (1 - smooth_weight) * phi_bicubic
    phi_edge_aware = _normalize(phi_edge_aware)
    results['edge_aware'] = phi_edge_aware
    
    # Method 4: Multi-scale refinement (like DA2's neck)
    # Process at multiple scales and combine
    scales = [1.0, 0.5, 0.25]
    multi_scale = np.zeros_like(phi_bicubic)
    
    for scale in scales:
        if scale < 1.0:
            h_s = int(depth_h * scale)
            w_s = int(depth_w * scale)
            downscaled = np.array(Image.fromarray((phi_bicubic * 255).astype(np.uint8)).resize((w_s, h_s))) / 255.0
            upscaled = np.array(Image.fromarray((downscaled * 255).astype(np.uint8)).resize((depth_w, depth_h))) / 255.0
        else:
            upscaled = phi_bicubic
        
        multi_scale += upscaled * (PHI ** (1 - scale))  # φ-weighted
    
    multi_scale = _normalize(multi_scale)
    results['multi_scale'] = multi_scale
    
    # Method 5: Iterative refinement
    phi_iter = phi_bicubic.copy()
    for _ in range(3):
        # Smooth
        smoothed = gaussian_filter(phi_iter, sigma=1.5)
        # Sharpen edges
        laplacian = phi_iter - smoothed
        phi_iter = smoothed + 0.3 * laplacian
        phi_iter = _normalize(phi_iter)
    results['iterative'] = phi_iter
    
    # Method 6: φ-scaled Gaussian pyramid
    pyramid = [phi_bicubic]
    current = phi_bicubic
    for i in range(3):
        sigma = PHI ** i
        smoothed = gaussian_filter(current, sigma=sigma)
        pyramid.append(smoothed)
        current = smoothed
    
    # Reconstruct with φ-weights
    phi_pyramid = np.zeros_like(phi_bicubic)
    for i, level in enumerate(pyramid):
        weight = PHI ** (-i)
        phi_pyramid += weight * level
    phi_pyramid = _normalize(phi_pyramid)
    results['phi_pyramid'] = phi_pyramid
    
    # Compute correlations
    print("\n  Post-processing method comparison:")
    print("-" * 50)
    
    for name, processed in results.items():
        corr = np.corrcoef(processed.flatten(), da2_depth.flatten())[0, 1]
        mae = np.mean(np.abs(processed - da2_depth))
        print(f"    {name:20s}: Corr={corr:.4f}, MAE={mae:.4f}")
    
    return results, da2_depth, rgb


def visualize_post_processing(results: dict, da2_depth: np.ndarray, rgb: np.ndarray):
    """Visualize different post-processing methods."""
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Post-Processing Methods: Matching DA2\'s Smoothness',
                 fontsize=14, fontweight='bold')
    
    n_methods = len(results)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.2, wspace=0.1)
    
    # Row 1: Original, DA2, and first 2 methods
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    ax.set_title('Original', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(da2_depth, cmap='magma')
    ax.set_title('DA2 (target)', fontsize=10)
    ax.axis('off')
    
    method_names = list(results.keys())
    
    for i, name in enumerate(method_names[:2]):
        ax = fig.add_subplot(gs[0, 2 + i])
        corr = np.corrcoef(results[name].flatten(), da2_depth.flatten())[0, 1]
        ax.imshow(results[name], cmap='magma')
        ax.set_title(f'{name}\n({corr:.3f})', fontsize=9)
        ax.axis('off')
    
    # Row 2: Remaining methods
    for i, name in enumerate(method_names[2:6]):
        ax = fig.add_subplot(gs[1, i])
        corr = np.corrcoef(results[name].flatten(), da2_depth.flatten())[0, 1]
        ax.imshow(results[name], cmap='magma')
        ax.set_title(f'{name}\n({corr:.3f})', fontsize=9)
        ax.axis('off')
    
    output_file = OUTPUT_PATH / "da2_post_processing.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def test_full_pipeline_with_best_postprocess(model, processor):
    """Test full φ-decoder + best post-processing on multiple images."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)
    
    import torch
    from scipy.optimize import minimize
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect training data for φ-decoder
    print("\n  Building φ-decoder...")
    all_features = []
    all_depths = []
    
    for img_id in available_ids[:20]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            backbone_output = model.backbone(inputs['pixel_values'], output_hidden_states=True)
            structure = backbone_output.hidden_states[-1].squeeze().numpy()
            full_output = model(inputs['pixel_values'])
            da2_depth = _normalize(full_output.predicted_depth.squeeze().numpy())
        
        structure = structure[1:]
        N, C = structure.shape
        depth_h, depth_w = da2_depth.shape
        H_s, W_s = depth_h // 14, depth_w // 14
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        for y in range(H_s):
            for x in range(W_s):
                all_features.append(struct_spatial[y, x])
                all_depths.append(depth_small[y, x])
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    
    # Build φ-decoder
    correlations = []
    for dim in range(all_features.shape[1]):
        corr, _ = pearsonr(all_features[:, dim], all_depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    dims = [c[0] for c in correlations[:100]]
    corrs = [c[1] for c in correlations[:100]]
    selected = all_features[:, dims]
    
    def objective(exponents):
        weights = np.array([np.sign(corrs[i]) * (PHI ** exponents[i]) for i in range(100)])
        weights = weights / np.abs(weights).sum()
        pred = selected @ weights
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return -np.corrcoef(pred, all_depths)[0, 1]
    
    init_exp = np.array([np.log(abs(c) + 0.1) / np.log(PHI) for c in corrs])
    result = minimize(objective, init_exp, method='L-BFGS-B', bounds=[(-3, 3)] * 100)
    
    weights = np.zeros(384)
    for i, dim in enumerate(dims):
        weights[dim] = np.sign(corrs[i]) * (PHI ** result.x[i])
    weights = weights / np.abs(weights).sum()
    
    print(f"    φ-decoder training correlation: {-result.fun:.4f}")
    
    # Test on images with best post-processing
    test_ids = available_ids[40:46]
    outlier_ids = ["000000002587", "000000003501"]
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    results = []
    
    print("\n  Testing with iterative refinement post-processing:")
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            backbone_output = model.backbone(inputs['pixel_values'], output_hidden_states=True)
            structure = backbone_output.hidden_states[-1].squeeze().numpy()
            full_output = model(inputs['pixel_values'])
            da2_depth = _normalize(full_output.predicted_depth.squeeze().numpy())
        
        structure = structure[1:]
        N, C = structure.shape
        depth_h, depth_w = da2_depth.shape
        H_s, W_s = depth_h // 14, depth_w // 14
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        
        # φ-decode
        phi_raw = np.tensordot(struct_spatial, weights, axes=([2], [0]))
        phi_raw = _normalize(phi_raw)
        
        # Upscale
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        phi_upscaled = zoom(phi_raw, (zoom_h, zoom_w), order=3)
        phi_upscaled = _normalize(phi_upscaled)
        
        # Best post-processing: iterative refinement
        phi_refined = phi_upscaled.copy()
        for _ in range(5):
            smoothed = gaussian_filter(phi_refined, sigma=1.5)
            laplacian = phi_refined - smoothed
            phi_refined = smoothed + 0.2 * laplacian
            phi_refined = _normalize(phi_refined)
        
        # Compute metrics
        corr_raw = np.corrcoef(phi_upscaled.flatten(), da2_depth.flatten())[0, 1]
        corr_refined = np.corrcoef(phi_refined.flatten(), da2_depth.flatten())[0, 1]
        
        is_outlier = img_id in outlier_ids
        marker = " [OUTLIER]" if is_outlier else ""
        print(f"    {img_id}: raw={corr_raw:.3f}, refined={corr_refined:.3f}{marker}")
        
        rgb_display = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))) / 255.0
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2': da2_depth,
            'phi_raw': phi_upscaled,
            'phi_refined': phi_refined,
            'corr_raw': corr_raw,
            'corr_refined': corr_refined,
            'is_outlier': is_outlier
        })
    
    return results


def create_final_visualization(results: list):
    """Create final comparison visualization."""
    
    fig = plt.figure(figsize=(16, 3.5 * len(results) + 1))
    fig.suptitle('φ-Decoder + Iterative Refinement vs DA2',
                 fontsize=14, fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(len(results) + 1, 5, figure=fig, hspace=0.15, wspace=0.05,
                          height_ratios=[1] * len(results) + [0.25])
    
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
        
        # φ Raw
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(r['phi_raw'], cmap='magma')
        title = f'φ Raw ({r["corr_raw"]:.3f})' if row == 0 else f'{r["corr_raw"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # φ Refined
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(r['phi_refined'], cmap='magma')
        title = f'φ Refined ({r["corr_refined"]:.3f})' if row == 0 else f'{r["corr_refined"]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Difference
        ax = fig.add_subplot(gs[row, 4])
        diff = r['phi_refined'] - r['da2']
        ax.imshow(diff, cmap='RdBu', vmin=-0.15, vmax=0.15)
        if row == 0:
            ax.set_title('Difference', fontsize=10)
        ax.axis('off')
    
    # Summary
    ax_summary = fig.add_subplot(gs[len(results), :])
    ax_summary.axis('off')
    
    avg_raw = np.mean([r['corr_raw'] for r in results])
    avg_refined = np.mean([r['corr_refined'] for r in results])
    
    normal_results = [r for r in results if not r['is_outlier']]
    outlier_results = [r for r in results if r['is_outlier']]
    
    normal_avg = np.mean([r['corr_refined'] for r in normal_results]) if normal_results else 0
    outlier_avg = np.mean([r['corr_refined'] for r in outlier_results]) if outlier_results else 0
    
    summary = f"""
    φ-DECODER + ITERATIVE REFINEMENT
    
    Raw φ-decoder:     {avg_raw:.4f}
    + Refinement:      {avg_refined:.4f}
    
    Normal images:     {normal_avg:.4f}
    Outliers:          {outlier_avg:.4f}
    """
    ax_summary.text(0.5, 0.5, summary, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    output_file = OUTPUT_PATH / "da2_phi_refined.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Analyze architecture
    analyze_da2_architecture(model)
    
    # Test on a sample image
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    sample_id = available_ids[42]
    img_path = COCO_VAL_PATH / f"{sample_id}.jpg"
    rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    
    # Test post-processing methods
    results, da2_depth, rgb_sample = test_post_processing_methods(model, processor, rgb)
    
    # Visualize
    visualize_post_processing(results, da2_depth, rgb_sample)
    
    # Full pipeline test
    full_results = test_full_pipeline_with_best_postprocess(model, processor)
    
    # Final visualization
    create_final_visualization(full_results)
    
    # Summary
    avg_refined = np.mean([r['corr_refined'] for r in full_results])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Average correlation with refinement: {avg_refined:.4f}")
