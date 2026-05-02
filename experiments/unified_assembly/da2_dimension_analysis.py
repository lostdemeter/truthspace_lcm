#!/usr/bin/env python3
"""
Analyzing What Each Dimension Does in DA2's Structure

We've proven that DA2's backbone output can be decoded with a simple
linear projection (50 PCA dimensions → depth). Now let's understand
what each dimension represents.

Goals:
1. Visualize what each PCA dimension encodes
2. Identify interpretable dimensions (global depth, edges, objects, etc.)
3. See if we can improve specific dimensions geometrically

This is the key to understanding DA2's "knowledge" geometrically.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.ndimage import zoom
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
    
    print("Loading Depth Anything V2...")
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
        structure = backbone_output.hidden_states[-1]
        
        full_output = model(inputs['pixel_values'])
        depth = full_output.predicted_depth.squeeze().numpy()
    
    return structure.squeeze().numpy(), _normalize(depth)


def learn_interpretable_transcoder(model, processor, n_train: int = 20):
    """
    Learn transcoder and analyze what each dimension represents.
    """
    print("\n" + "=" * 70)
    print("LEARNING INTERPRETABLE TRANSCODER")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    all_features = []
    all_depths = []
    all_positions = []  # Track spatial positions
    
    pixels_per_image = 500
    
    print(f"\nCollecting samples from {n_train} images...")
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        structure, depth = extract_structure(model, processor, rgb)
        
        # Skip CLS token
        structure = structure[1:]
        N, C = structure.shape
        
        # Find spatial dimensions
        depth_h, depth_w = depth.shape
        patch_size = 14
        H_s = depth_h // patch_size
        W_s = depth_w // patch_size
        
        if H_s * W_s != N:
            for h in range(1, int(np.sqrt(N)) + 10):
                if N % h == 0:
                    w = N // h
                    if abs(w/h - depth_w/depth_h) < 0.5:
                        H_s, W_s = h, w
                        break
        
        struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
        depth_small = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        # Sample positions
        np.random.seed(i)
        for _ in range(pixels_per_image):
            y = np.random.randint(0, H_s)
            x = np.random.randint(0, W_s)
            
            all_features.append(struct_spatial[y, x])
            all_depths.append(depth_small[y, x])
            all_positions.append((y / H_s, x / W_s))  # Normalized position
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_train}")
    
    all_features = np.array(all_features)
    all_depths = np.array(all_depths)
    all_positions = np.array(all_positions)
    
    print(f"\n  Collected {len(all_features)} samples")
    
    # PCA
    feature_mean = all_features.mean(axis=0)
    features_centered = all_features - feature_mean
    
    U, S, Vt = svd(features_centered, full_matrices=False)
    
    # Keep top components
    n_components = 20  # Fewer for interpretability
    pca_components = Vt[:n_components]
    pca_features = features_centered @ pca_components.T
    
    # Variance explained by each component
    var_explained = (S[:n_components] ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    
    print(f"\n  Variance explained by top {n_components} components:")
    for i in range(min(10, n_components)):
        print(f"    PC{i}: {var_explained[i]*100:.2f}% (cumulative: {cumvar[i]*100:.2f}%)")
    
    # Learn weights for each component
    X = np.column_stack([pca_features, np.ones(len(pca_features))])
    weights, _, _, _ = np.linalg.lstsq(X, all_depths, rcond=None)
    
    # Analyze what each component correlates with
    print("\n  Component analysis:")
    print("-" * 50)
    
    component_info = []
    
    for i in range(n_components):
        pc_values = pca_features[:, i]
        
        # Correlation with depth
        corr_depth = np.corrcoef(pc_values, all_depths)[0, 1]
        
        # Correlation with vertical position
        corr_y = np.corrcoef(pc_values, all_positions[:, 0])[0, 1]
        
        # Correlation with horizontal position
        corr_x = np.corrcoef(pc_values, all_positions[:, 1])[0, 1]
        
        # Weight in final prediction
        weight = weights[i]
        
        # Contribution to depth prediction
        contribution = abs(weight) * np.std(pc_values)
        
        component_info.append({
            'index': i,
            'var_explained': var_explained[i],
            'corr_depth': corr_depth,
            'corr_y': corr_y,
            'corr_x': corr_x,
            'weight': weight,
            'contribution': contribution
        })
        
        # Interpret the component
        interpretation = []
        if abs(corr_y) > 0.3:
            interpretation.append(f"vertical ({'top=high' if corr_y > 0 else 'bottom=high'})")
        if abs(corr_x) > 0.3:
            interpretation.append(f"horizontal ({'right=high' if corr_x > 0 else 'left=high'})")
        if abs(corr_depth) > 0.3:
            interpretation.append(f"depth ({'far=high' if corr_depth > 0 else 'close=high'})")
        
        interp_str = ", ".join(interpretation) if interpretation else "complex pattern"
        
        print(f"    PC{i}: var={var_explained[i]*100:.1f}%, "
              f"depth_corr={corr_depth:.2f}, y_corr={corr_y:.2f}, "
              f"weight={weight:.4f} → {interp_str}")
    
    return {
        'feature_mean': feature_mean,
        'pca_components': pca_components,
        'weights': weights,
        'n_components': n_components,
        'component_info': component_info,
        'var_explained': var_explained
    }


def visualize_dimensions(model, processor, transcoder: dict, img_id: str):
    """
    Visualize what each dimension captures for a specific image.
    """
    print(f"\nVisualizing dimensions for {img_id}...")
    
    img_path = COCO_VAL_PATH / f"{img_id}.jpg"
    rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    
    structure, da2_depth = extract_structure(model, processor, rgb)
    
    # Process structure
    structure = structure[1:]  # Skip CLS
    N, C = structure.shape
    
    depth_h, depth_w = da2_depth.shape
    patch_size = 14
    H_s = depth_h // patch_size
    W_s = depth_w // patch_size
    
    if H_s * W_s != N:
        for h in range(1, int(np.sqrt(N)) + 10):
            if N % h == 0:
                w = N // h
                if abs(w/h - depth_w/depth_h) < 0.5:
                    H_s, W_s = h, w
                    break
    
    struct_spatial = structure[:H_s*W_s].reshape(H_s, W_s, C)
    struct_flat = struct_spatial.reshape(-1, C)
    
    # Project to PCA space
    features_centered = struct_flat - transcoder['feature_mean']
    pca_features = features_centered @ transcoder['pca_components'].T
    
    # Reshape each component to spatial
    n_show = min(8, transcoder['n_components'])
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'What Each Dimension Captures\nImage: {img_id}',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, n_show + 1, figure=fig, hspace=0.3, wspace=0.2)
    
    # Row 1: Original and DA2 depth
    ax_rgb = fig.add_subplot(gs[0, 0])
    rgb_display = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s * 4, H_s * 4))) / 255.0
    ax_rgb.imshow(rgb_display)
    ax_rgb.set_title('Original', fontsize=10)
    ax_rgb.axis('off')
    
    ax_da2 = fig.add_subplot(gs[0, 1])
    da2_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s * 4, H_s * 4))) / 255.0
    ax_da2.imshow(da2_small, cmap='magma')
    ax_da2.set_title('DA2 Depth', fontsize=10)
    ax_da2.axis('off')
    
    # Show individual components
    for i in range(n_show):
        pc_map = pca_features[:, i].reshape(H_s, W_s)
        pc_upscaled = zoom(pc_map, 4, order=1)
        
        info = transcoder['component_info'][i]
        
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(pc_upscaled, cmap='RdBu', vmin=-3, vmax=3)
        ax.set_title(f'PC{i}\nvar={info["var_explained"]*100:.1f}%\nw={info["weight"]:.3f}', fontsize=8)
        ax.axis('off')
    
    # Row 3: Weighted contributions
    for i in range(n_show):
        pc_map = pca_features[:, i].reshape(H_s, W_s)
        weighted = pc_map * transcoder['weights'][i]
        weighted_upscaled = zoom(weighted, 4, order=1)
        
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(weighted_upscaled, cmap='magma')
        ax.set_title(f'PC{i} × weight', fontsize=8)
        ax.axis('off')
    
    # Final reconstruction
    pred_flat = pca_features @ transcoder['weights'][:-1] + transcoder['weights'][-1]
    pred_map = pred_flat.reshape(H_s, W_s)
    pred_upscaled = zoom(_normalize(pred_map), 4, order=3)
    
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_pred.imshow(pred_upscaled, cmap='magma')
    ax_pred.set_title('Reconstructed', fontsize=10)
    ax_pred.axis('off')
    
    output_file = OUTPUT_PATH / f"da2_dimensions_{img_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved to {output_file}")
    return output_file


def identify_improvement_opportunities(transcoder: dict):
    """
    Identify which dimensions could be improved geometrically.
    """
    print("\n" + "=" * 70)
    print("IDENTIFYING IMPROVEMENT OPPORTUNITIES")
    print("=" * 70)
    
    component_info = transcoder['component_info']
    
    # Sort by contribution to final prediction
    sorted_by_contribution = sorted(component_info, key=lambda x: x['contribution'], reverse=True)
    
    print("\n  Components ranked by contribution to depth prediction:")
    print("-" * 60)
    
    for info in sorted_by_contribution[:10]:
        i = info['index']
        
        # Identify what this component does
        if abs(info['corr_y']) > 0.5:
            role = "VERTICAL GRADIENT"
            improvement = "Could be replaced with pure geometric y-coordinate"
        elif abs(info['corr_depth']) > 0.7:
            role = "GLOBAL DEPTH"
            improvement = "Could be enhanced with φ-scaling"
        elif abs(info['corr_x']) > 0.3:
            role = "HORIZONTAL PATTERN"
            improvement = "May encode left-right asymmetry"
        else:
            role = "COMPLEX FEATURE"
            improvement = "Likely encodes object boundaries or texture"
        
        print(f"\n  PC{i}: {role}")
        print(f"    Contribution: {info['contribution']:.4f}")
        print(f"    Correlations: depth={info['corr_depth']:.2f}, y={info['corr_y']:.2f}, x={info['corr_x']:.2f}")
        print(f"    Improvement: {improvement}")
    
    return sorted_by_contribution


def create_summary_visualization(transcoder: dict, sorted_components: list):
    """Create a summary visualization of the dimension analysis."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('DA2 Dimension Analysis: Understanding the Geometric Structure',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Variance explained
    ax1 = fig.add_subplot(gs[0, 0])
    var = transcoder['var_explained']
    ax1.bar(range(len(var)), var * 100, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance per Component')
    
    # 2. Cumulative variance
    ax2 = fig.add_subplot(gs[0, 1])
    cumvar = np.cumsum(var) * 100
    ax2.plot(cumvar, 'g-o', markersize=4)
    ax2.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title('Cumulative Variance')
    ax2.legend()
    
    # 3. Correlation with depth
    ax3 = fig.add_subplot(gs[0, 2])
    corr_depth = [c['corr_depth'] for c in transcoder['component_info']]
    colors = ['green' if c > 0.3 else 'red' if c < -0.3 else 'gray' for c in corr_depth]
    ax3.bar(range(len(corr_depth)), corr_depth, color=colors)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Correlation with Depth')
    ax3.set_title('Depth Correlation per Component')
    
    # 4. Contribution to prediction
    ax4 = fig.add_subplot(gs[1, 0])
    contributions = [c['contribution'] for c in sorted_components[:15]]
    indices = [c['index'] for c in sorted_components[:15]]
    ax4.barh(range(len(contributions)), contributions, color='orange')
    ax4.set_yticks(range(len(contributions)))
    ax4.set_yticklabels([f'PC{i}' for i in indices])
    ax4.set_xlabel('Contribution')
    ax4.set_title('Top Contributors to Depth')
    ax4.invert_yaxis()
    
    # 5. Correlation with position
    ax5 = fig.add_subplot(gs[1, 1])
    corr_y = [c['corr_y'] for c in transcoder['component_info']]
    corr_x = [c['corr_x'] for c in transcoder['component_info']]
    ax5.scatter(corr_x, corr_y, c=corr_depth, cmap='RdYlGn', s=100, alpha=0.7)
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    ax5.axvline(x=0, color='gray', linewidth=0.5)
    ax5.set_xlabel('Correlation with X (horizontal)')
    ax5.set_ylabel('Correlation with Y (vertical)')
    ax5.set_title('Spatial Correlations\n(color = depth correlation)')
    plt.colorbar(ax5.collections[0], ax=ax5, label='Depth corr')
    
    # 6. Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Find key components
    vertical_pcs = [c for c in transcoder['component_info'] if abs(c['corr_y']) > 0.4]
    depth_pcs = [c for c in transcoder['component_info'] if abs(c['corr_depth']) > 0.5]
    
    summary = f"""
    DIMENSION ANALYSIS SUMMARY
    
    Total components: {transcoder['n_components']}
    
    Key findings:
    
    1. VERTICAL GRADIENT components:
       {[c['index'] for c in vertical_pcs]}
       These encode top-bottom depth gradient
       → Could be replaced with pure geometry
    
    2. DEPTH-CORRELATED components:
       {[c['index'] for c in depth_pcs]}
       These directly encode depth information
       → Core of DA2's knowledge
    
    3. Top 3 contributors:
       PC{sorted_components[0]['index']}: {sorted_components[0]['contribution']:.3f}
       PC{sorted_components[1]['index']}: {sorted_components[1]['contribution']:.3f}
       PC{sorted_components[2]['index']}: {sorted_components[2]['contribution']:.3f}
    
    IMPROVEMENT STRATEGY:
    - Replace vertical components with φ-scaled y
    - Keep depth-correlated components
    - Analyze complex components for edges
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    output_file = OUTPUT_PATH / "da2_dimension_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSummary saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    model, processor = load_da2()
    
    # Learn interpretable transcoder
    transcoder = learn_interpretable_transcoder(model, processor, n_train=20)
    
    # Identify improvement opportunities
    sorted_components = identify_improvement_opportunities(transcoder)
    
    # Visualize dimensions for a sample image
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    if depth_files:
        sample_id = depth_files[25].stem.replace("_depth", "")
        visualize_dimensions(model, processor, transcoder, sample_id)
    
    # Create summary
    create_summary_visualization(transcoder, sorted_components)
    
    print("\n" + "=" * 70)
    print("DIMENSION ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("We now understand what each dimension in DA2's structure encodes.")
    print("This opens the door to geometric improvements:")
    print("  1. Replace learned vertical gradients with pure φ-geometry")
    print("  2. Enhance depth-correlated components with φ-scaling")
    print("  3. Analyze complex components for edge detection")
