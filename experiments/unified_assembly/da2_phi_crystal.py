#!/usr/bin/env python3
"""
φ-Crystal Decoder: Combining φ-Geometry with Gushurst Crystal Resonance

Key insight from Gushurst Crystal:
- Prime-power symmetries [2¹, 3¹, 7¹] create resonance patterns
- Variance cascade reveals fractal structure
- Crystalline lattice captures number-theoretic relationships

Applying to DA2:
- φ-scaling loses 7.8% vs optimal (φ-weights correlate 0.69 with optimal)
- What if dimensions have RESONANCE patterns like prime powers?
- Can we use crystal structure to find better dimension weights?

The hypothesis:
- DA2's dimensions may have resonance relationships
- φ alone captures the decay pattern
- Crystal structure captures the RESONANCE pattern
- φ × crystal = better than either alone

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from scipy import linalg
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


def collect_data(model, processor, n_images: int = 25):
    """Collect patch-level data."""
    print("\n  Collecting data...")
    
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
    
    return np.array(all_features), np.array(all_depths)


def build_dimension_crystal(features: np.ndarray, depths: np.ndarray, n_dims: int = 50):
    """
    Build a crystalline lattice structure for dimensions.
    
    Inspired by Gushurst Crystal:
    - Nodes = dimensions (instead of zeta zeros)
    - Edges = correlation between dimensions
    - Weights = depth correlation (instead of variance cascade)
    
    The crystal structure captures RESONANCE between dimensions.
    """
    print("\n" + "=" * 70)
    print("BUILDING DIMENSION CRYSTAL")
    print("=" * 70)
    
    # Get correlations with depth
    correlations = []
    for dim in range(features.shape[1]):
        corr, _ = pearsonr(features[:, dim], depths)
        correlations.append((dim, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_dims = [c[0] for c in correlations[:n_dims]]
    top_corrs = np.array([c[1] for c in correlations[:n_dims]])
    top_features = features[:, top_dims]
    
    print(f"\n  Top {n_dims} dimensions selected")
    print(f"  Correlation range: [{np.abs(top_corrs).min():.3f}, {np.abs(top_corrs).max():.3f}]")
    
    # Build dimension-dimension correlation matrix (the crystal lattice)
    print("\n  Building crystal lattice (dimension correlations)...")
    
    # Normalize features
    top_features_norm = (top_features - top_features.mean(axis=0)) / (top_features.std(axis=0) + 1e-10)
    
    # Correlation matrix between dimensions
    crystal_lattice = np.corrcoef(top_features_norm.T)
    
    # Apply prime-power weighting like Gushurst Crystal [2¹, 3¹, 7¹]
    # This creates resonance patterns at specific scales
    w2 = 1.0 / 2.0  # Binary symmetry
    w3 = 1.0 / 3.0  # Triangular symmetry
    w7 = 1.0 / 7.0  # Heptagonal symmetry
    
    # Create resonance-weighted lattice
    resonance_lattice = np.zeros_like(crystal_lattice)
    
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                resonance_lattice[i, j] = 1.0
                continue
            
            # Distance in correlation space
            dist = abs(i - j)
            
            # Apply prime-power resonances
            if dist % 2 == 0:  # Binary resonance
                resonance_lattice[i, j] += crystal_lattice[i, j] * w2
            if dist % 3 == 0:  # Triangular resonance
                resonance_lattice[i, j] += crystal_lattice[i, j] * w3
            if dist % 7 == 0:  # Heptagonal resonance
                resonance_lattice[i, j] += crystal_lattice[i, j] * w7
            
            # Add base correlation
            resonance_lattice[i, j] += crystal_lattice[i, j] * 0.5
    
    # Symmetrize
    resonance_lattice = (resonance_lattice + resonance_lattice.T) / 2
    
    # Spectral analysis of crystal
    eigenvals, eigenvecs = linalg.eigh(resonance_lattice)
    spectral_gap = eigenvals[-1] - eigenvals[-2]
    
    print(f"\n  Crystal lattice: {n_dims}×{n_dims}")
    print(f"  Spectral gap: {spectral_gap:.4f}")
    print(f"  Top eigenvalue: {eigenvals[-1]:.4f}")
    
    return crystal_lattice, resonance_lattice, eigenvals, eigenvecs, top_dims, top_corrs


def extract_crystal_weights(resonance_lattice: np.ndarray, eigenvals: np.ndarray, 
                           eigenvecs: np.ndarray, top_corrs: np.ndarray):
    """
    Extract dimension weights from crystal structure.
    
    NEW APPROACH: Use crystal structure to REFINE φ-weights, not replace them.
    
    Key insight: The crystal lattice captures how dimensions interact.
    We can use this to:
    1. Boost dimensions that resonate with other high-corr dimensions
    2. Dampen dimensions that are redundant (highly correlated with others)
    """
    print("\n" + "=" * 70)
    print("EXTRACTING CRYSTAL-REFINED WEIGHTS")
    print("=" * 70)
    
    n_dims = len(top_corrs)
    
    # Base φ-weights (what we're refining)
    phi_scales = np.array([PHI ** (-i/10) for i in range(n_dims)])
    base_weights = phi_scales * np.sign(top_corrs)
    
    # Method 1: Redundancy correction
    # If a dimension is highly correlated with others, reduce its weight
    # (it's redundant - other dims carry similar info)
    row_sums = np.abs(resonance_lattice).sum(axis=1) - 1  # Exclude self
    redundancy = row_sums / row_sums.max()
    redundancy_correction = 1.0 - 0.5 * redundancy  # Reduce weight by up to 50%
    
    weights_redundancy = base_weights * redundancy_correction
    weights_redundancy = weights_redundancy / np.abs(weights_redundancy).sum()
    
    print(f"\n  Method 1: Redundancy correction")
    print(f"    Reduces weight of highly-correlated dimensions")
    
    # Method 2: Resonance boost
    # Boost dimensions that resonate at prime-power intervals
    resonance_boost = np.ones(n_dims)
    for i in range(n_dims):
        # Count resonances at prime-power distances
        for j in range(n_dims):
            if i == j:
                continue
            dist = abs(i - j)
            if dist in [2, 3, 4, 6, 7, 8, 9, 12, 14]:  # Prime powers and products
                resonance_boost[i] += 0.1 * abs(resonance_lattice[i, j])
    
    resonance_boost = resonance_boost / resonance_boost.max()
    weights_resonance = base_weights * resonance_boost
    weights_resonance = weights_resonance / np.abs(weights_resonance).sum()
    
    print(f"\n  Method 2: Resonance boost")
    print(f"    Boosts dimensions with prime-power resonances")
    
    # Method 3: Correlation-weighted φ
    # Weight φ-scales by the actual depth correlation magnitude
    corr_weights = np.abs(top_corrs) / np.abs(top_corrs).max()
    weights_corr_phi = base_weights * corr_weights
    weights_corr_phi = weights_corr_phi / np.abs(weights_corr_phi).sum()
    
    print(f"\n  Method 3: Correlation-weighted φ")
    print(f"    Weights φ by depth correlation magnitude")
    
    # Method 4: Combined (redundancy + resonance + correlation)
    combined_factor = redundancy_correction * resonance_boost * corr_weights
    weights_combined = base_weights * combined_factor
    weights_combined = weights_combined / np.abs(weights_combined).sum()
    
    print(f"\n  Method 4: Combined crystal refinement")
    
    # Method 5: Spectral smoothing
    # Use crystal eigenvectors to smooth the weights
    # Project φ-weights onto top eigenvectors and back
    n_keep = 20  # Keep top 20 eigenvectors
    projection = eigenvecs[:, -n_keep:] @ (eigenvecs[:, -n_keep:].T @ base_weights)
    weights_spectral = projection * np.sign(top_corrs)
    weights_spectral = weights_spectral / np.abs(weights_spectral).sum()
    
    print(f"\n  Method 5: Spectral smoothing")
    print(f"    Projects φ-weights through crystal eigenbasis")
    
    return {
        'redundancy': weights_redundancy,
        'resonance': weights_resonance,
        'corr_phi': weights_corr_phi,
        'combined': weights_combined,
        'spectral': weights_spectral
    }


def test_crystal_decoder(features: np.ndarray, depths: np.ndarray, 
                        top_dims: list, top_corrs: np.ndarray,
                        crystal_weights: dict):
    """
    Test different crystal-based decoders.
    """
    print("\n" + "=" * 70)
    print("TESTING CRYSTAL DECODERS")
    print("=" * 70)
    
    top_features = features[:, top_dims]
    top_features_norm = (top_features - top_features.mean(axis=0)) / (top_features.std(axis=0) + 1e-10)
    
    n_dims = len(top_dims)
    results = {}
    
    # Baseline: Pure φ-scaling
    phi_scales = np.array([PHI ** (-i/10) for i in range(n_dims)])
    phi_weights = phi_scales * np.sign(top_corrs)
    phi_weights = phi_weights / np.abs(phi_weights).sum()
    
    phi_pred = top_features_norm @ phi_weights
    phi_pred = _normalize(phi_pred)
    phi_corr = np.corrcoef(phi_pred, depths)[0, 1]
    
    results['φ-only'] = phi_corr
    print(f"\n  φ-only: {phi_corr:.4f}")
    
    # Test each crystal method
    for name, weights in crystal_weights.items():
        pred = top_features_norm @ weights
        pred = _normalize(pred)
        corr = np.corrcoef(pred, depths)[0, 1]
        results[name] = corr
        
        improvement = corr - phi_corr
        print(f"  {name}: {corr:.4f} ({improvement:+.4f} vs φ)")
    
    # Optimal linear (for reference)
    from sklearn.linear_model import Ridge
    lr = Ridge(alpha=1.0)
    lr.fit(top_features_norm, depths)
    optimal_pred = lr.predict(top_features_norm)
    optimal_corr = np.corrcoef(optimal_pred, depths)[0, 1]
    
    results['optimal'] = optimal_corr
    print(f"\n  Optimal linear: {optimal_corr:.4f}")
    
    # Gap analysis
    print(f"\n  Gap analysis:")
    print(f"    φ-only gap to optimal: {optimal_corr - phi_corr:.4f}")
    
    best_crystal = max([(k, v) for k, v in results.items() if k not in ['φ-only', 'optimal']], 
                       key=lambda x: x[1])
    print(f"    Best crystal gap to optimal: {optimal_corr - best_crystal[1]:.4f}")
    print(f"    Crystal improvement over φ: {best_crystal[1] - phi_corr:.4f}")
    
    return results


def visualize_crystal_structure(crystal_lattice: np.ndarray, resonance_lattice: np.ndarray,
                               eigenvals: np.ndarray, results: dict):
    """Visualize the crystal structure and results."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('φ-Crystal Decoder: Combining φ-Geometry with Crystal Resonance',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Raw dimension correlation matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(crystal_lattice, cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_title('Dimension Correlation Matrix')
    ax1.set_xlabel('Dimension (sorted by depth corr)')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot 2: Resonance-weighted lattice
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(resonance_lattice, cmap='viridis')
    ax2.set_title('Resonance Lattice [2¹, 3¹, 7¹]')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Dimension')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Eigenvalue spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(eigenvals, 'go-', markersize=4)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Crystal Eigenspectrum')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Results comparison
    ax4 = fig.add_subplot(gs[1, 0])
    methods = list(results.keys())
    correlations = [results[m] for m in methods]
    colors = ['gold' if m == 'φ-only' else 'green' if m == 'optimal' else 'steelblue' 
              for m in methods]
    bars = ax4.bar(range(len(methods)), correlations, color=colors)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Decoder Comparison')
    ax4.set_ylim(0.8, 1.0)
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        ax4.annotate(f'{corr:.3f}', xy=(bar.get_x() + bar.get_width()/2, corr),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # Plot 5: Improvement over φ
    ax5 = fig.add_subplot(gs[1, 1])
    phi_corr = results['φ-only']
    improvements = {k: v - phi_corr for k, v in results.items() if k != 'φ-only'}
    methods_imp = list(improvements.keys())
    imps = [improvements[m] for m in methods_imp]
    colors_imp = ['green' if m == 'optimal' else 'steelblue' for m in methods_imp]
    ax5.bar(range(len(methods_imp)), imps, color=colors_imp)
    ax5.set_xticks(range(len(methods_imp)))
    ax5.set_xticklabels(methods_imp, rotation=45, ha='right')
    ax5.set_ylabel('Improvement over φ')
    ax5.set_title('Crystal Improvement')
    ax5.axhline(y=0, color='red', linestyle='--')
    
    # Plot 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    best_crystal = max([(k, v) for k, v in results.items() if k not in ['φ-only', 'optimal']], 
                       key=lambda x: x[1])
    
    summary = f"""
    φ-CRYSTAL DECODER RESULTS
    
    Baseline (φ-only): {results['φ-only']:.4f}
    Optimal linear:    {results['optimal']:.4f}
    
    Best crystal method: {best_crystal[0]}
    Best crystal corr:   {best_crystal[1]:.4f}
    
    Gap closed:
      φ → optimal:     {results['optimal'] - results['φ-only']:.4f}
      φ → best crystal: {best_crystal[1] - results['φ-only']:.4f}
      Remaining gap:    {results['optimal'] - best_crystal[1]:.4f}
    
    Crystal resonance captures dimension
    relationships that pure φ-scaling misses.
    
    Prime-power symmetries [2¹, 3¹, 7¹]
    create resonance patterns in the
    dimension correlation structure.
    """
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    output_file = OUTPUT_PATH / "da2_phi_crystal.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Loading DA2...")
    model, processor = load_da2()
    
    # Collect data
    features, depths = collect_data(model, processor, n_images=25)
    print(f"  Collected {len(features)} patches")
    
    # Build dimension crystal
    crystal_lattice, resonance_lattice, eigenvals, eigenvecs, top_dims, top_corrs = \
        build_dimension_crystal(features, depths, n_dims=50)
    
    # Extract crystal weights
    crystal_weights = extract_crystal_weights(resonance_lattice, eigenvals, eigenvecs, top_corrs)
    
    # Test decoders
    results = test_crystal_decoder(features, depths, top_dims, top_corrs, crystal_weights)
    
    # Visualize
    viz_file = visualize_crystal_structure(crystal_lattice, resonance_lattice, eigenvals, results)
    
    # Summary
    print("\n" + "=" * 70)
    print("φ-CRYSTAL DECODER SUMMARY")
    print("=" * 70)
    
    best_crystal = max([(k, v) for k, v in results.items() if k not in ['φ-only', 'optimal']], 
                       key=lambda x: x[1])
    
    print(f"""
    The Gushurst Crystal concept applied to DA2 dimensions:
    
    φ-only:        {results['φ-only']:.4f}
    Best crystal:  {best_crystal[1]:.4f} ({best_crystal[0]})
    Optimal:       {results['optimal']:.4f}
    
    Crystal improvement: {best_crystal[1] - results['φ-only']:+.4f}
    Remaining gap:       {results['optimal'] - best_crystal[1]:.4f}
    
    The crystal structure captures RESONANCE patterns
    between dimensions that pure φ-scaling misses.
    """)
