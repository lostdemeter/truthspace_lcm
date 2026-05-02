#!/usr/bin/env python3
"""
Analyzing DA2's Handling of Outlier Images

We have outliers (banana, food bowl) where our φ-transcoder fails but DA2 succeeds.
This means DA2 has learned something we haven't captured geometrically.

Goals:
1. Compare DA2 activations on outliers vs good images
2. Find which layers/dimensions handle close-up objects
3. Extract the mechanism that DA2 uses for these cases

If we can find WHERE in DA2 this is handled, we can:
- Extract that specific knowledge
- Map it to our geometric framework
- Improve our transcoder for outliers

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
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


def get_all_activations(model, processor, rgb: np.ndarray):
    """
    Get activations from ALL layers of DA2.
    
    Returns a dict with:
    - hidden_states: list of backbone hidden states
    - feature_maps: backbone feature maps
    - neck_features: neck layer outputs
    - head_features: head layer outputs
    - final_depth: predicted depth
    """
    import torch
    
    pil_image = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=pil_image, return_tensors="pt")
    
    activations = {}
    
    with torch.no_grad():
        # Get backbone activations
        backbone_output = model.backbone(
            inputs['pixel_values'],
            output_hidden_states=True
        )
        
        activations['hidden_states'] = [
            hs.squeeze().numpy() for hs in backbone_output.hidden_states
        ]
        
        if backbone_output.feature_maps:
            activations['feature_maps'] = [
                fm.squeeze().numpy() for fm in backbone_output.feature_maps
            ]
        
        # Get full model output
        full_output = model(inputs['pixel_values'])
        activations['final_depth'] = full_output.predicted_depth.squeeze().numpy()
    
    return activations


def analyze_activation_differences(model, processor, good_ids: list, outlier_ids: list):
    """
    Compare activations between good images and outliers.
    
    Look for dimensions that behave differently.
    """
    print("\n" + "=" * 70)
    print("ANALYZING ACTIVATION DIFFERENCES: Good vs Outliers")
    print("=" * 70)
    
    good_activations = []
    outlier_activations = []
    
    print("\nProcessing good images...")
    for img_id in good_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        acts = get_all_activations(model, processor, rgb)
        good_activations.append(acts)
        print(f"  {img_id}: processed")
    
    print("\nProcessing outlier images...")
    for img_id in outlier_ids:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        acts = get_all_activations(model, processor, rgb)
        outlier_activations.append(acts)
        print(f"  {img_id}: processed")
    
    # Compare hidden states at each layer
    print("\n" + "-" * 50)
    print("HIDDEN STATE ANALYSIS")
    print("-" * 50)
    
    n_layers = len(good_activations[0]['hidden_states'])
    
    layer_differences = []
    
    for layer_idx in range(n_layers):
        # Get mean activation per dimension for good vs outlier
        good_means = []
        outlier_means = []
        
        for acts in good_activations:
            hs = acts['hidden_states'][layer_idx]
            if len(hs.shape) == 2:  # [N, C]
                good_means.append(hs.mean(axis=0))  # Mean over positions
            else:
                good_means.append(hs.mean())
        
        for acts in outlier_activations:
            hs = acts['hidden_states'][layer_idx]
            if len(hs.shape) == 2:
                outlier_means.append(hs.mean(axis=0))
            else:
                outlier_means.append(hs.mean())
        
        good_mean = np.mean(good_means, axis=0)
        outlier_mean = np.mean(outlier_means, axis=0)
        
        # Find dimensions with largest difference
        if isinstance(good_mean, np.ndarray):
            diff = np.abs(good_mean - outlier_mean)
            top_diff_dims = np.argsort(diff)[-10:][::-1]
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            layer_differences.append({
                'layer': layer_idx,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'top_dims': top_diff_dims,
                'top_diffs': diff[top_diff_dims]
            })
            
            print(f"\n  Layer {layer_idx}:")
            print(f"    Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
            print(f"    Top differing dims: {top_diff_dims[:5]}")
    
    return good_activations, outlier_activations, layer_differences


def find_discriminative_dimensions(good_activations: list, outlier_activations: list):
    """
    Find dimensions that discriminate between good and outlier images.
    
    These are the dimensions that DA2 uses to handle close-ups differently.
    """
    print("\n" + "=" * 70)
    print("FINDING DISCRIMINATIVE DIMENSIONS")
    print("=" * 70)
    
    # Use the last hidden state (most processed)
    good_features = []
    outlier_features = []
    
    for acts in good_activations:
        hs = acts['hidden_states'][-1]  # Last layer
        if len(hs.shape) == 2:
            # Skip CLS token, get mean over patches
            good_features.append(hs[1:].mean(axis=0))
    
    for acts in outlier_activations:
        hs = acts['hidden_states'][-1]
        if len(hs.shape) == 2:
            outlier_features.append(hs[1:].mean(axis=0))
    
    good_features = np.array(good_features)
    outlier_features = np.array(outlier_features)
    
    print(f"\n  Good features shape: {good_features.shape}")
    print(f"  Outlier features shape: {outlier_features.shape}")
    
    # Compute mean and std for each group
    good_mean = good_features.mean(axis=0)
    good_std = good_features.std(axis=0)
    outlier_mean = outlier_features.mean(axis=0)
    outlier_std = outlier_features.std(axis=0)
    
    # Fisher's discriminant ratio: (μ1 - μ2)² / (σ1² + σ2²)
    fisher_ratio = (good_mean - outlier_mean)**2 / (good_std**2 + outlier_std**2 + 1e-10)
    
    # Top discriminative dimensions
    top_dims = np.argsort(fisher_ratio)[-20:][::-1]
    
    print("\n  Top 20 discriminative dimensions (Fisher ratio):")
    for i, dim in enumerate(top_dims[:10]):
        print(f"    Dim {dim}: ratio={fisher_ratio[dim]:.4f}, "
              f"good_mean={good_mean[dim]:.3f}, outlier_mean={outlier_mean[dim]:.3f}")
    
    return {
        'fisher_ratio': fisher_ratio,
        'top_dims': top_dims,
        'good_mean': good_mean,
        'outlier_mean': outlier_mean,
        'good_std': good_std,
        'outlier_std': outlier_std
    }


def analyze_depth_statistics(good_activations: list, outlier_activations: list):
    """
    Analyze how depth predictions differ between good and outlier images.
    """
    print("\n" + "=" * 70)
    print("DEPTH PREDICTION STATISTICS")
    print("=" * 70)
    
    print("\n  Good images depth stats:")
    for i, acts in enumerate(good_activations):
        depth = acts['final_depth']
        depth_norm = _normalize(depth)
        print(f"    Image {i}: mean={depth_norm.mean():.3f}, std={depth_norm.std():.3f}, "
              f"range=[{depth_norm.min():.3f}, {depth_norm.max():.3f}]")
    
    print("\n  Outlier images depth stats:")
    for i, acts in enumerate(outlier_activations):
        depth = acts['final_depth']
        depth_norm = _normalize(depth)
        print(f"    Image {i}: mean={depth_norm.mean():.3f}, std={depth_norm.std():.3f}, "
              f"range=[{depth_norm.min():.3f}, {depth_norm.max():.3f}]")
    
    # Key insight: outliers might have different depth distributions
    good_stds = [_normalize(a['final_depth']).std() for a in good_activations]
    outlier_stds = [_normalize(a['final_depth']).std() for a in outlier_activations]
    
    print(f"\n  Average depth std - Good: {np.mean(good_stds):.3f}, Outlier: {np.mean(outlier_stds):.3f}")


def create_analysis_visualization(good_activations, outlier_activations, 
                                   discriminative_info, good_ids, outlier_ids):
    """Visualize the analysis."""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('DA2 Outlier Analysis: What Makes Close-ups Different?',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Good images and their depth
    for i, (acts, img_id) in enumerate(zip(good_activations[:2], good_ids[:2])):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if img_path.exists():
            rgb = np.array(Image.open(img_path).convert("RGB"))
            
            ax = fig.add_subplot(gs[0, i*2])
            ax.imshow(rgb)
            ax.set_title(f'Good: {img_id}', fontsize=9)
            ax.axis('off')
            
            ax = fig.add_subplot(gs[0, i*2+1])
            ax.imshow(_normalize(acts['final_depth']), cmap='magma')
            ax.set_title('DA2 Depth', fontsize=9)
            ax.axis('off')
    
    # Row 2: Outlier images and their depth
    for i, (acts, img_id) in enumerate(zip(outlier_activations[:2], outlier_ids[:2])):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        if img_path.exists():
            rgb = np.array(Image.open(img_path).convert("RGB"))
            
            ax = fig.add_subplot(gs[1, i*2])
            ax.imshow(rgb)
            ax.set_title(f'Outlier: {img_id}', fontsize=9)
            ax.axis('off')
            
            ax = fig.add_subplot(gs[1, i*2+1])
            ax.imshow(_normalize(acts['final_depth']), cmap='magma')
            ax.set_title('DA2 Depth', fontsize=9)
            ax.axis('off')
    
    # Row 3: Discriminative analysis
    ax = fig.add_subplot(gs[2, 0:2])
    fisher = discriminative_info['fisher_ratio']
    ax.bar(range(len(fisher)), fisher, color='steelblue', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Fisher Ratio')
    ax.set_title('Discriminative Power per Dimension\n(Higher = More Different for Outliers)')
    
    # Highlight top dimensions
    top_dims = discriminative_info['top_dims'][:5]
    for dim in top_dims:
        ax.axvline(x=dim, color='red', alpha=0.3, linewidth=2)
    
    ax = fig.add_subplot(gs[2, 2:4])
    # Show mean activation difference for top dimensions
    top_20 = discriminative_info['top_dims'][:20]
    good_vals = discriminative_info['good_mean'][top_20]
    outlier_vals = discriminative_info['outlier_mean'][top_20]
    
    x = np.arange(len(top_20))
    width = 0.35
    ax.bar(x - width/2, good_vals, width, label='Good Images', color='green', alpha=0.7)
    ax.bar(x + width/2, outlier_vals, width, label='Outliers', color='red', alpha=0.7)
    ax.set_xlabel('Top Discriminative Dimensions')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Activation Comparison: Good vs Outlier')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in top_20], rotation=45, fontsize=7)
    ax.legend()
    
    output_file = OUTPUT_PATH / "da2_outlier_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def extract_outlier_correction(discriminative_info, n_dims: int = 10):
    """
    Extract the dimensions that could correct outlier predictions.
    
    These are the dimensions where outliers differ most from good images.
    We can use these to detect and correct close-up images.
    """
    print("\n" + "=" * 70)
    print("EXTRACTING OUTLIER CORRECTION MECHANISM")
    print("=" * 70)
    
    top_dims = discriminative_info['top_dims'][:n_dims]
    
    print(f"\n  Top {n_dims} discriminative dimensions: {list(top_dims)}")
    
    # Create a simple detector: if these dimensions have outlier-like values,
    # the image is probably a close-up
    good_mean = discriminative_info['good_mean'][top_dims]
    outlier_mean = discriminative_info['outlier_mean'][top_dims]
    
    # Threshold: midpoint between good and outlier means
    threshold = (good_mean + outlier_mean) / 2
    
    # Direction: which way indicates outlier
    direction = np.sign(outlier_mean - good_mean)
    
    print("\n  Outlier detection rule:")
    print("    For each top dimension, check if activation is on the outlier side")
    print(f"    Dimensions: {list(top_dims)}")
    print(f"    Thresholds: {threshold}")
    print(f"    Outlier direction: {direction}")
    
    return {
        'top_dims': top_dims,
        'threshold': threshold,
        'direction': direction,
        'good_mean': good_mean,
        'outlier_mean': outlier_mean
    }


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Define good and outlier images based on our previous experiments
    # Good: high correlation with our transcoder
    good_ids = [
        "000000002532",  # Skier - 0.961
        "000000002923",  # Landscape - 0.968
        "000000003255",  # Sheep - 0.975
        "000000002592",  # Cup - 0.930
    ]
    
    # Outliers: low correlation with our transcoder
    outlier_ids = [
        "000000002587",  # Banana - 0.125
        "000000003501",  # Food bowl - 0.512
    ]
    
    # Analyze activation differences
    good_acts, outlier_acts, layer_diffs = analyze_activation_differences(
        model, processor, good_ids, outlier_ids
    )
    
    # Find discriminative dimensions
    disc_info = find_discriminative_dimensions(good_acts, outlier_acts)
    
    # Analyze depth statistics
    analyze_depth_statistics(good_acts, outlier_acts)
    
    # Extract correction mechanism
    correction = extract_outlier_correction(disc_info)
    
    # Visualize
    viz_file = create_analysis_visualization(
        good_acts, outlier_acts, disc_info, good_ids, outlier_ids
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Key findings:")
    print(f"  - Top discriminative dimensions: {list(disc_info['top_dims'][:5])}")
    print(f"  - These dimensions have different activation patterns for close-ups")
    print()
    print("Next steps:")
    print("  1. Use these dimensions to detect close-up images")
    print("  2. Apply different geometric strategy for close-ups")
    print("  3. Or: incorporate these specific dimensions into our transcoder")
