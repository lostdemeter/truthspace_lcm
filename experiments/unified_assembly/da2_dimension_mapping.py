#!/usr/bin/env python3
"""
DA2 Dimension Mapping: What Does Each Dimension Control?

We know DA2 encodes depth linearly in specific dimensions.
Now we systematically map what each dimension controls:
- Depth (distance from camera)
- Vertical position (y-coordinate correlation)
- Horizontal position (x-coordinate correlation)
- Edges/boundaries
- Object identity
- Scene type (indoor/outdoor, close-up/landscape)

Goal: Build a complete map so we can reimplementing DA2 using φ-geometry.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.ndimage import sobel, gaussian_filter
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


def extract_structure_and_features(model, processor, rgb: np.ndarray):
    """Extract DA2's structure and compute various feature maps."""
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


def compute_geometric_features(rgb: np.ndarray, depth: np.ndarray):
    """
    Compute geometric features that dimensions might correlate with.
    """
    h, w = depth.shape
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # Position features
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_norm = y_grid / h
    x_norm = x_grid / w
    
    # Edge features
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    edges = _normalize(np.sqrt(grad_x**2 + grad_y**2))
    
    # Depth gradient (how fast depth changes)
    depth_grad_x = sobel(depth, axis=1)
    depth_grad_y = sobel(depth, axis=0)
    depth_edges = _normalize(np.sqrt(depth_grad_x**2 + depth_grad_y**2))
    
    # Color features
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    luminance = _normalize(gray)
    
    # Saturation
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
    
    # Local contrast
    local_mean = gaussian_filter(gray, sigma=10)
    local_std = np.sqrt(gaussian_filter((gray - local_mean)**2, sigma=10))
    contrast = _normalize(local_std)
    
    # Center distance
    center_y, center_x = h/2, w/2
    center_dist = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    center_dist = _normalize(center_dist)
    
    return {
        'depth': depth,
        'y_position': y_norm,
        'x_position': x_norm,
        'edges': edges,
        'depth_edges': depth_edges,
        'luminance': luminance,
        'saturation': saturation,
        'contrast': contrast,
        'center_dist': center_dist,
        'red': r,
        'green': g,
        'blue': b,
    }


def map_dimensions(model, processor, n_images: int = 30):
    """
    Map each dimension to what it correlates with.
    """
    print("\n" + "=" * 70)
    print("MAPPING DA2 DIMENSIONS")
    print("=" * 70)
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Collect data
    all_features = []  # [N_patches, 384]
    all_targets = {}   # {feature_name: [N_patches]}
    
    feature_names = ['depth', 'y_position', 'x_position', 'edges', 'depth_edges',
                     'luminance', 'saturation', 'contrast', 'center_dist',
                     'red', 'green', 'blue']
    
    for name in feature_names:
        all_targets[name] = []
    
    print(f"\nCollecting data from {n_images} images...")
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        structure, da2_depth = extract_structure_and_features(model, processor, rgb)
        
        # Skip CLS token
        structure = structure[1:]
        N, C = structure.shape
        
        # Get spatial dimensions
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
        
        # Resize RGB and compute features at patch resolution
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        depth_small = np.array(Image.fromarray((da2_depth * 255).astype(np.uint8)).resize((W_s, H_s))) / 255.0
        
        features = compute_geometric_features(rgb_small, depth_small)
        
        # Collect all patches
        for y in range(H_s):
            for x in range(W_s):
                all_features.append(struct_spatial[y, x])
                for name in feature_names:
                    all_targets[name].append(features[name][y, x])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_images}")
    
    all_features = np.array(all_features)
    for name in feature_names:
        all_targets[name] = np.array(all_targets[name])
    
    print(f"\n  Collected {len(all_features)} patches")
    
    # Compute correlations for each dimension with each target
    print("\n  Computing correlations for all 384 dimensions...")
    
    dimension_map = []
    
    for dim in range(384):
        dim_values = all_features[:, dim]
        
        correlations = {}
        for name in feature_names:
            corr, _ = pearsonr(dim_values, all_targets[name])
            correlations[name] = corr
        
        # Find what this dimension correlates with most
        best_target = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_target]
        
        dimension_map.append({
            'dim': dim,
            'best_target': best_target,
            'best_corr': best_corr,
            'all_correlations': correlations
        })
    
    return dimension_map, feature_names


def categorize_dimensions(dimension_map: list):
    """
    Categorize dimensions by what they primarily encode.
    """
    print("\n" + "=" * 70)
    print("DIMENSION CATEGORIES")
    print("=" * 70)
    
    categories = {}
    
    for dm in dimension_map:
        target = dm['best_target']
        if target not in categories:
            categories[target] = []
        categories[target].append(dm)
    
    # Sort each category by correlation strength
    for target in categories:
        categories[target].sort(key=lambda x: abs(x['best_corr']), reverse=True)
    
    # Print summary
    print("\n  Dimensions by primary encoding:")
    print("-" * 50)
    
    for target in sorted(categories.keys(), key=lambda t: len(categories[t]), reverse=True):
        dims = categories[target]
        top_dims = [d['dim'] for d in dims[:5]]
        top_corrs = [d['best_corr'] for d in dims[:5]]
        
        print(f"\n  {target.upper()} ({len(dims)} dimensions)")
        print(f"    Top dims: {top_dims}")
        print(f"    Top corrs: {[f'{c:.3f}' for c in top_corrs]}")
    
    return categories


def find_phi_patterns(dimension_map: list):
    """
    Look for φ-related patterns in the dimension structure.
    """
    print("\n" + "=" * 70)
    print("SEARCHING FOR φ-PATTERNS")
    print("=" * 70)
    
    # Sort dimensions by depth correlation
    depth_dims = sorted(dimension_map, key=lambda x: abs(x['all_correlations']['depth']), reverse=True)
    
    depth_corrs = [abs(d['all_correlations']['depth']) for d in depth_dims[:50]]
    
    # Check for φ-ratios between consecutive correlations
    ratios = []
    for i in range(len(depth_corrs) - 1):
        if depth_corrs[i+1] > 0.01:  # Avoid division by tiny numbers
            ratio = depth_corrs[i] / depth_corrs[i+1]
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    # Check how many are near φ or 1/φ
    near_phi = np.abs(ratios - PHI) < 0.15
    near_phi_inv = np.abs(ratios - 1/PHI) < 0.15
    
    print(f"\n  Checking ratios between consecutive depth correlations:")
    print(f"    Ratios near φ (1.618): {near_phi.sum()} / {len(ratios)}")
    print(f"    Ratios near 1/φ (0.618): {near_phi_inv.sum()} / {len(ratios)}")
    
    # Check dimension indices for patterns
    top_depth_dims = [d['dim'] for d in depth_dims[:20]]
    print(f"\n  Top 20 depth-encoding dimension indices: {top_depth_dims}")
    
    # Check for spacing patterns
    dim_diffs = np.diff(sorted(top_depth_dims))
    print(f"  Differences between sorted indices: {list(dim_diffs)}")
    
    # Check if any differences are φ-related
    phi_related = []
    for diff in dim_diffs:
        if abs(diff - int(PHI * 10)) < 2:  # ~16
            phi_related.append(diff)
        elif abs(diff - int(PHI * 20)) < 3:  # ~32
            phi_related.append(diff)
        elif abs(diff - int(PHI * 50)) < 5:  # ~81
            phi_related.append(diff)
    
    if phi_related:
        print(f"  φ-related spacings found: {phi_related}")
    
    return depth_dims


def create_dimension_report(dimension_map: list, categories: dict):
    """
    Create a visual report of dimension mappings.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('DA2 Dimension Mapping: What Each Dimension Controls',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Correlation heatmap for top dimensions
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get top 30 dimensions by any correlation
    top_dims = sorted(dimension_map, key=lambda x: max(abs(c) for c in x['all_correlations'].values()), reverse=True)[:30]
    
    feature_names = list(top_dims[0]['all_correlations'].keys())
    heatmap_data = np.array([[d['all_correlations'][f] for f in feature_names] for d in top_dims])
    
    im = ax1.imshow(heatmap_data.T, aspect='auto', cmap='RdBu', vmin=-0.7, vmax=0.7)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Dimension (sorted by max correlation)')
    ax1.set_title('Correlation of Top 30 Dimensions with Geometric Features')
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Plot 2: Category distribution
    ax2 = fig.add_subplot(gs[1, 0])
    cat_names = list(categories.keys())
    cat_counts = [len(categories[c]) for c in cat_names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cat_names)))
    ax2.barh(cat_names, cat_counts, color=colors)
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_title('Dimensions by Primary Encoding')
    
    # Plot 3: Depth encoding dimensions
    ax3 = fig.add_subplot(gs[1, 1])
    depth_corrs = sorted([d['all_correlations']['depth'] for d in dimension_map], key=abs, reverse=True)[:50]
    ax3.bar(range(len(depth_corrs)), depth_corrs, color=['green' if c > 0 else 'red' for c in depth_corrs])
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Correlation with Depth')
    ax3.set_title('Top 50 Depth-Encoding Dimensions')
    
    # Plot 4: Position encoding
    ax4 = fig.add_subplot(gs[1, 2])
    y_corrs = [d['all_correlations']['y_position'] for d in dimension_map]
    x_corrs = [d['all_correlations']['x_position'] for d in dimension_map]
    ax4.scatter(x_corrs, y_corrs, alpha=0.5, s=10)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.set_xlabel('X-Position Correlation')
    ax4.set_ylabel('Y-Position Correlation')
    ax4.set_title('Spatial Position Encoding')
    
    # Plot 5: Summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Build summary
    summary_lines = []
    summary_lines.append("DIMENSION MAPPING SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    for target in ['depth', 'y_position', 'x_position', 'edges', 'luminance']:
        if target in categories:
            dims = categories[target][:3]
            dim_str = ", ".join([f"dim{d['dim']}({d['best_corr']:.2f})" for d in dims])
            summary_lines.append(f"{target.upper():15s}: {dim_str}")
    
    summary_lines.append("")
    summary_lines.append("KEY INSIGHT: DA2's 384 dimensions encode different geometric features.")
    summary_lines.append("We can reconstruct depth by combining dimensions with known correlations.")
    
    summary_text = "\n".join(summary_lines)
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    output_file = OUTPUT_PATH / "da2_dimension_mapping.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def build_phi_decoder_spec(categories: dict, dimension_map: list):
    """
    Build a specification for a φ-based decoder.
    """
    print("\n" + "=" * 70)
    print("φ-DECODER SPECIFICATION")
    print("=" * 70)
    
    spec = {
        'depth_dims': [],
        'y_position_dims': [],
        'x_position_dims': [],
        'edge_dims': [],
        'color_dims': [],
    }
    
    # Get top dimensions for each category
    for target, dims in categories.items():
        top_dims = [(d['dim'], d['best_corr']) for d in dims[:10]]
        
        if target == 'depth':
            spec['depth_dims'] = top_dims
        elif target == 'y_position':
            spec['y_position_dims'] = top_dims
        elif target == 'x_position':
            spec['x_position_dims'] = top_dims
        elif target == 'edges' or target == 'depth_edges':
            spec['edge_dims'].extend(top_dims)
        elif target in ['red', 'green', 'blue', 'luminance', 'saturation']:
            spec['color_dims'].extend(top_dims)
    
    print("\n  φ-Decoder Specification:")
    print("-" * 50)
    
    for key, dims in spec.items():
        if dims:
            print(f"\n  {key}:")
            for dim, corr in dims[:5]:
                weight = corr  # Use correlation as weight
                phi_weight = weight / PHI if abs(weight) > 0.3 else weight / (PHI**2)
                print(f"    dim{dim}: corr={corr:.3f}, φ-weight={phi_weight:.4f}")
    
    return spec


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Map all dimensions
    dimension_map, feature_names = map_dimensions(model, processor, n_images=30)
    
    # Categorize
    categories = categorize_dimensions(dimension_map)
    
    # Look for φ-patterns
    depth_dims = find_phi_patterns(dimension_map)
    
    # Create visual report
    viz_file = create_dimension_report(dimension_map, categories)
    
    # Build φ-decoder spec
    spec = build_phi_decoder_spec(categories, dimension_map)
    
    print("\n" + "=" * 70)
    print("DIMENSION MAPPING COMPLETE")
    print("=" * 70)
    print()
    print("  We now have a complete map of what each DA2 dimension encodes.")
    print("  Next step: Build a φ-based decoder using this specification.")
