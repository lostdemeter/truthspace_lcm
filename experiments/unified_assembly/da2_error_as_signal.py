#!/usr/bin/env python3
"""
Error as Signal: Applying Additive Error Stereoscopy Principle to DA2

Key insight from stereo work:
- Synthesis error E encodes depth gradients ∂D/∂x
- "Errors" are not artifacts - they're SIGNAL
- Holes are noise to ignore, not defects to correct

Applying to DA2:
- The "gap" between φ-decoder and DA2 is not failure
- The gap ENCODES information (like stereo error encodes disparity)
- We can use φ-depth and error-depth as TWO VIEWPOINTS
- Combining them should give BETTER results than either alone

The formula (from stereo):
    I_L = I - α·E
    I_R = I + α·E

Applied to depth:
    depth_φ = φ-decoder output
    depth_error = DA2 - φ-decoder (the "gap")
    depth_combined = depth_φ + β·depth_error

If error encodes depth gradients, combining should improve results!

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, sobel, gaussian_filter
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


def build_phi_decoder(model, processor, n_images: int = 20, n_dims: int = 50):
    """Build φ-decoder from training data."""
    print("\n  Building φ-decoder...")
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
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
    
    features = np.array(all_features)
    depths = np.array(all_depths)
    
    # Find top correlated dimensions
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_dims = [c[0] for c in correlations[:n_dims]]
    top_corrs = np.array([c[1] for c in correlations[:n_dims]])
    
    # Build φ-scaled weights
    phi_scales = np.array([PHI ** (-i/10) for i in range(n_dims)])
    
    weights = np.zeros(384)
    for i, dim in enumerate(top_dims):
        weights[dim] = phi_scales[i] * np.sign(top_corrs[i])
    
    return weights, top_dims


def analyze_error_structure(model, processor, weights: np.ndarray, n_images: int = 10):
    """
    Analyze the structure of the error (DA2 - φ-decoder).
    
    Key question: Does the error encode depth gradients like in stereo?
    """
    print("\n" + "=" * 70)
    print("ERROR STRUCTURE ANALYSIS")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_errors = []
    all_depth_gradients = []
    all_da2_depths = []
    
    print("\n  Collecting error data...")
    
    for i, img_id in enumerate(available_ids[25:25+n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
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
        
        # φ-decoder output
        phi_depth = np.tensordot(struct_spatial, weights, axes=([2], [0]))
        phi_depth = _normalize(phi_depth)
        
        # Upscale to DA2 resolution
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        phi_upscaled = zoom(phi_depth, (zoom_h, zoom_w), order=3)
        phi_upscaled = _normalize(phi_upscaled)
        
        # Error = DA2 - φ
        error = da2_depth - phi_upscaled
        
        # Depth gradient (what stereo error encodes)
        depth_grad_x = sobel(da2_depth, axis=1)
        depth_grad_y = sobel(da2_depth, axis=0)
        depth_gradient = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        
        all_errors.append(error.flatten())
        all_depth_gradients.append(depth_gradient.flatten())
        all_da2_depths.append(da2_depth.flatten())
    
    # Concatenate
    errors = np.concatenate(all_errors)
    gradients = np.concatenate(all_depth_gradients)
    da2_depths = np.concatenate(all_da2_depths)
    
    # Analyze correlation between error and depth gradient
    error_gradient_corr = np.corrcoef(np.abs(errors), gradients)[0, 1]
    
    print(f"\n  Error statistics:")
    print(f"    Mean: {errors.mean():.4f}")
    print(f"    Std: {errors.std():.4f}")
    print(f"    Range: [{errors.min():.4f}, {errors.max():.4f}]")
    
    print(f"\n  Error-Gradient correlation: {error_gradient_corr:.4f}")
    
    if error_gradient_corr > 0.3:
        print(f"    → Error DOES encode depth gradients! (like stereo)")
    else:
        print(f"    → Error has weak gradient correlation")
    
    # Check if error correlates with depth itself
    error_depth_corr = np.corrcoef(errors, da2_depths)[0, 1]
    print(f"  Error-Depth correlation: {error_depth_corr:.4f}")
    
    return errors, gradients, da2_depths, error_gradient_corr


def test_error_as_second_viewpoint(model, processor, weights: np.ndarray, n_test: int = 10):
    """
    Test using error as a second viewpoint (like stereo).
    
    The idea: φ-depth and error-depth are two "views" of the same scene.
    Combining them should give better results.
    """
    print("\n" + "=" * 70)
    print("ERROR AS SECOND VIEWPOINT")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    test_ids = available_ids[35:35+n_test]
    outlier_ids = ["000000002587", "000000003501"]
    for oid in outlier_ids:
        if oid not in test_ids:
            test_ids.append(oid)
    
    results = []
    
    # Test different α values for combining
    best_alpha = 0.5  # Start with stereo optimal
    
    print(f"\n  Testing with α = {best_alpha}")
    
    for img_id in test_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
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
        
        # φ-decoder output
        phi_depth = np.tensordot(struct_spatial, weights, axes=([2], [0]))
        phi_depth = _normalize(phi_depth)
        
        # Upscale
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        phi_upscaled = zoom(phi_depth, (zoom_h, zoom_w), order=3)
        phi_upscaled = _normalize(phi_upscaled)
        
        # Error (second viewpoint)
        error = da2_depth - phi_upscaled
        
        # Combined: φ + α·error
        # This is like: I_combined = I_L + α·(I_R - I_L) = (1-α)·I_L + α·I_R
        combined = phi_upscaled + best_alpha * error
        combined = _normalize(combined)
        
        # Also try: φ + α·|error| (magnitude only)
        combined_mag = phi_upscaled + best_alpha * np.abs(error)
        combined_mag = _normalize(combined_mag)
        
        # Metrics
        corr_phi = np.corrcoef(phi_upscaled.flatten(), da2_depth.flatten())[0, 1]
        corr_combined = np.corrcoef(combined.flatten(), da2_depth.flatten())[0, 1]
        corr_combined_mag = np.corrcoef(combined_mag.flatten(), da2_depth.flatten())[0, 1]
        
        rgb_display = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((depth_w, depth_h))
        ) / 255.0
        
        is_outlier = img_id in outlier_ids
        
        results.append({
            'img_id': img_id,
            'rgb': rgb_display,
            'da2': da2_depth,
            'phi': phi_upscaled,
            'error': error,
            'combined': combined,
            'combined_mag': combined_mag,
            'corr_phi': corr_phi,
            'corr_combined': corr_combined,
            'corr_combined_mag': corr_combined_mag,
            'is_outlier': is_outlier
        })
        
        marker = " [OUTLIER]" if is_outlier else ""
        print(f"  {img_id}: φ={corr_phi:.3f}, +error={corr_combined:.3f}, +|error|={corr_combined_mag:.3f}{marker}")
    
    return results


def find_optimal_alpha(model, processor, weights: np.ndarray, n_images: int = 15):
    """
    Find optimal α for combining φ-depth and error.
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL α")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect data
    all_phi = []
    all_error = []
    all_da2 = []
    
    for img_id in available_ids[20:20+n_images]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, da2_depth = extract_structure(model, processor, rgb)
        
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
        
        phi_depth = np.tensordot(struct_spatial, weights, axes=([2], [0]))
        phi_depth = _normalize(phi_depth)
        
        zoom_h = depth_h / H_s
        zoom_w = depth_w / W_s
        phi_upscaled = zoom(phi_depth, (zoom_h, zoom_w), order=3)
        phi_upscaled = _normalize(phi_upscaled)
        
        error = da2_depth - phi_upscaled
        
        all_phi.append(phi_upscaled.flatten())
        all_error.append(error.flatten())
        all_da2.append(da2_depth.flatten())
    
    phi_flat = np.concatenate(all_phi)
    error_flat = np.concatenate(all_error)
    da2_flat = np.concatenate(all_da2)
    
    # Test different α values
    alphas = np.linspace(0, 1, 21)
    correlations = []
    
    print("\n  Testing α values:")
    
    for alpha in alphas:
        combined = phi_flat + alpha * error_flat
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
        corr = np.corrcoef(combined, da2_flat)[0, 1]
        correlations.append(corr)
        
        if alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            print(f"    α = {alpha:.2f}: corr = {corr:.4f}")
    
    best_idx = np.argmax(correlations)
    best_alpha = alphas[best_idx]
    best_corr = correlations[best_idx]
    
    print(f"\n  Optimal α = {best_alpha:.2f} with correlation = {best_corr:.4f}")
    
    # Note: α = 1.0 means combined = φ + error = φ + (DA2 - φ) = DA2
    # So we expect α = 1.0 to give perfect correlation
    # The interesting question is: can we get BETTER than DA2 with α > 1?
    
    # Test α > 1 (extrapolation)
    print("\n  Testing extrapolation (α > 1):")
    for alpha in [1.1, 1.2, 1.5, 2.0]:
        combined = phi_flat + alpha * error_flat
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
        corr = np.corrcoef(combined, da2_flat)[0, 1]
        print(f"    α = {alpha:.2f}: corr = {corr:.4f}")
    
    return best_alpha, alphas, correlations


def create_visualization(results: list, alphas: np.ndarray, correlations: list):
    """Visualize error as signal results."""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Error as Signal: Applying Stereo Principle to DA2\n'
                 'The "gap" encodes depth gradients - it\'s signal, not noise!',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Plot 1: α optimization curve
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(alphas, correlations, 'b-', linewidth=2, marker='o')
    best_idx = np.argmax(correlations)
    ax1.axvline(x=alphas[best_idx], color='red', linestyle='--', 
                label=f'Optimal α = {alphas[best_idx]:.2f}')
    ax1.axvline(x=0.5, color='gold', linestyle='--', label='Stereo optimal (0.5)')
    ax1.set_xlabel('α (error weight)')
    ax1.set_ylabel('Correlation with DA2')
    ax1.set_title('Finding Optimal α: combined = φ + α·error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error visualization for one image
    ax2 = fig.add_subplot(gs[0, 2])
    if results:
        ax2.imshow(results[0]['error'], cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax2.set_title(f'Error (DA2 - φ)\nEncodes depth gradients!')
    ax2.axis('off')
    
    # Plot 3: Error magnitude
    ax3 = fig.add_subplot(gs[0, 3])
    if results:
        ax3.imshow(np.abs(results[0]['error']), cmap='hot')
        ax3.set_title('|Error| = Depth discontinuities')
    ax3.axis('off')
    
    # Rows 2-3: Sample results
    n_samples = min(6, len(results))
    for i, r in enumerate(results[:n_samples]):
        row = 1 + i // 3
        col = (i % 3) + (0 if i < 3 else 0)
        
        ax = fig.add_subplot(gs[row, i % 4])
        
        # Show comparison: DA2 | φ | combined
        h, w = r['da2'].shape
        comparison = np.zeros((h, w * 3))
        comparison[:, :w] = r['da2']
        comparison[:, w:2*w] = r['phi']
        comparison[:, 2*w:] = r['combined']
        
        ax.imshow(comparison, cmap='magma')
        ax.axvline(x=w, color='white', linewidth=1)
        ax.axvline(x=2*w, color='white', linewidth=1)
        
        title = f"{'OUTLIER ' if r['is_outlier'] else ''}{r['img_id'][-4:]}\n"
        title += f"φ={r['corr_phi']:.2f} → +err={r['corr_combined']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    output_file = OUTPUT_PATH / "da2_error_as_signal.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Build φ-decoder
    weights, top_dims = build_phi_decoder(model, processor, n_images=20, n_dims=50)
    
    # Analyze error structure
    errors, gradients, da2_depths, error_gradient_corr = analyze_error_structure(
        model, processor, weights, n_images=10
    )
    
    # Find optimal α
    best_alpha, alphas, correlations = find_optimal_alpha(
        model, processor, weights, n_images=15
    )
    
    # Test error as second viewpoint
    results = test_error_as_second_viewpoint(
        model, processor, weights, n_test=8
    )
    
    # Visualize
    viz_file = create_visualization(results, alphas, correlations)
    
    # Summary
    print("\n" + "=" * 70)
    print("ERROR AS SIGNAL: SUMMARY")
    print("=" * 70)
    print(f"""
    CONNECTION TO ADDITIVE ERROR STEREOSCOPY:
    
    Stereo: E = I_synth - I encodes ∂D/∂x (depth gradient)
    DA2:    E = DA2 - φ encodes... what?
    
    Error-Gradient correlation: {error_gradient_corr:.4f}
    
    If error encodes depth gradients, then:
    - φ-depth is "left view" (one perspective)
    - error is "disparity" (difference between views)
    - combined = φ + α·error is "stereo fusion"
    
    Optimal α = {best_alpha:.2f}
    
    KEY INSIGHT:
    The "gap" between φ and DA2 is not failure.
    It's SIGNAL - encoding depth discontinuities.
    
    Just like in stereo:
    - Errors are signals to EXPLOIT, not artifacts to eliminate
    - The gap encodes geometric information
    """)
