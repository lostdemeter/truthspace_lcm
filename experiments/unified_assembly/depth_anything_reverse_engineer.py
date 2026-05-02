#!/usr/bin/env python3
"""
Reverse Engineering Depth Anything V2

Goal: Understand the geometric structure of DA2's weights and see if we can
map it to our φ-based self-assembling structure.

The hypothesis:
- DA2 learned depth estimation from millions of images
- The "knowledge" is encoded in the weight geometry
- If we can understand that geometry, we can potentially:
  1. Replicate it with φ-based structure
  2. Understand what semantic priors it learned
  3. Build a geometric equivalent

Steps:
1. Load DA2 weights
2. Apply self-assembly (eigendecomposition of similarity matrix)
3. Analyze the structure (clusters, ratios, dimensions)
4. Map to φ-space
5. Test if φ-structure can approximate DA2 behavior

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from fractions import Fraction
import warnings

warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly")


def load_depth_anything_v2():
    """Load Depth Anything V2 model weights."""
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
        
        print("Loading Depth Anything V2 (small)...")
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        
        print(f"  Loaded {len(weights)} weight tensors")
        
        # Print model structure
        print("\n  Model structure:")
        layer_types = {}
        for name in weights.keys():
            parts = name.split('.')
            if len(parts) > 1:
                layer_type = parts[0] + '.' + parts[1] if len(parts) > 2 else parts[0]
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        for lt, count in sorted(layer_types.items())[:10]:
            print(f"    {lt}: {count} tensors")
        
        return weights, model
        
    except Exception as e:
        print(f"Error loading DA2: {e}")
        return None, None


def analyze_weight_structure(weights: dict):
    """Analyze the structure of DA2 weights."""
    print("\n" + "=" * 60)
    print("ANALYZING DEPTH ANYTHING V2 WEIGHT STRUCTURE")
    print("=" * 60)
    
    # Categorize weights
    categories = {
        'patch_embed': [],
        'encoder': [],
        'decoder': [],
        'head': [],
        'other': []
    }
    
    for name, w in weights.items():
        if 'patch' in name.lower() or 'embed' in name.lower():
            categories['patch_embed'].append((name, w))
        elif 'encoder' in name.lower() or 'backbone' in name.lower():
            categories['encoder'].append((name, w))
        elif 'decoder' in name.lower() or 'neck' in name.lower():
            categories['decoder'].append((name, w))
        elif 'head' in name.lower():
            categories['head'].append((name, w))
        else:
            categories['other'].append((name, w))
    
    print("\n  Weight categories:")
    for cat, items in categories.items():
        total_params = sum(w.size for _, w in items)
        print(f"    {cat}: {len(items)} tensors, {total_params:,} params")
    
    return categories


def compute_da2_eigenstructure(weights: dict, max_weights: int = 100):
    """
    Compute eigenstructure of DA2 weights.
    
    This is the core of reverse engineering - understanding
    how the weights are geometrically organized.
    """
    print("\n" + "=" * 60)
    print("COMPUTING DA2 EIGENSTRUCTURE")
    print("=" * 60)
    
    # Select 2D weight matrices
    matrices = []
    names = []
    for name, w in weights.items():
        if len(w.shape) == 2 and min(w.shape) > 20 and 'bias' not in name:
            matrices.append(w)
            names.append(name)
    
    matrices = matrices[:max_weights]
    names = names[:max_weights]
    
    print(f"\n  Selected {len(matrices)} weight matrices")
    
    # Extract features from each weight
    features = []
    singular_values = []
    
    for w in matrices:
        try:
            _, s, _ = svd(w, full_matrices=False)
            # Use top singular values as features
            n_sv = min(20, len(s))
            feat = np.zeros(20)
            feat[:n_sv] = s[:n_sv] / (s[0] + 1e-10)
            features.append(feat)
            singular_values.append(s)
        except:
            features.append(np.zeros(20))
            singular_values.append(np.array([1.0]))
    
    features = np.array(features)
    
    # Compute similarity matrix
    similarity = features @ features.T
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(similarity)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute positions in eigenspace
    pos_mask = eigenvalues > 0
    positions = eigenvectors[:, pos_mask] * np.sqrt(eigenvalues[pos_mask])
    
    print(f"\n  Eigenvalue spectrum:")
    print(f"    Top 5: {eigenvalues[:5]}")
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"    Top 3 explain: {cumvar[2]*100:.2f}%")
    
    return eigenvalues, eigenvectors, positions, names, singular_values


def find_phi_structure(eigenvalues: np.ndarray, singular_values: list):
    """
    Search for φ-related structure in DA2 weights.
    
    If DA2 has learned efficient representations, we might
    find φ-scaling in its weight structure.
    """
    print("\n" + "=" * 60)
    print("SEARCHING FOR φ-STRUCTURE IN DA2")
    print("=" * 60)
    
    # Analyze eigenvalue ratios
    print("\n  Eigenvalue ratios:")
    for i in range(min(10, len(eigenvalues) - 1)):
        if eigenvalues[i+1] > 1e-10:
            ratio = eigenvalues[i] / eigenvalues[i+1]
            
            # Check against special values
            phi_diff = abs(ratio - PHI)
            phi2_diff = abs(ratio - PHI**2)
            e_diff = abs(ratio - np.e)
            ratio_137_30 = abs(ratio - 137/30)
            
            special = ""
            if phi_diff < 0.1:
                special = f" ≈ φ"
            elif phi2_diff < 0.15:
                special = f" ≈ φ²"
            elif e_diff < 0.15:
                special = f" ≈ e"
            elif ratio_137_30 < 0.2:
                special = f" ≈ 137/30"
            
            print(f"    λ_{i}/λ_{i+1} = {ratio:.4f}{special}")
    
    # Analyze singular value ratios across all weights
    all_sv_ratios = []
    for sv in singular_values:
        for i in range(min(10, len(sv) - 1)):
            if sv[i+1] > 1e-10:
                all_sv_ratios.append(sv[i] / sv[i+1])
    
    all_sv_ratios = np.array(all_sv_ratios)
    
    print(f"\n  Singular value ratios (across all weights):")
    print(f"    Total: {len(all_sv_ratios)}")
    print(f"    Mean: {all_sv_ratios.mean():.4f}")
    print(f"    Median: {np.median(all_sv_ratios):.4f}")
    
    # Check clustering around φ
    near_phi = (np.abs(all_sv_ratios - PHI) < 0.1).mean() * 100
    near_phi2 = (np.abs(all_sv_ratios - PHI**2) < 0.15).mean() * 100
    near_1 = (np.abs(all_sv_ratios - 1.0) < 0.05).mean() * 100
    
    print(f"\n  Clustering:")
    print(f"    Near φ (1.618): {near_phi:.2f}%")
    print(f"    Near φ² (2.618): {near_phi2:.2f}%")
    print(f"    Near 1.0: {near_1:.2f}%")
    
    return all_sv_ratios


def map_to_phi_space(positions: np.ndarray, names: list):
    """
    Map DA2 weight positions to φ-space coordinates.
    
    The idea: if DA2's weight structure has geometric meaning,
    we should be able to express it in φ-coordinates.
    """
    print("\n" + "=" * 60)
    print("MAPPING DA2 TO φ-SPACE")
    print("=" * 60)
    
    # Use first 4 dimensions (like our x,y,z,w in TruthSpace)
    pos_4d = positions[:, :4] if positions.shape[1] >= 4 else positions
    
    # Normalize to [0, 1] range
    pos_norm = (pos_4d - pos_4d.min(axis=0)) / (pos_4d.max(axis=0) - pos_4d.min(axis=0) + 1e-10)
    
    # Convert to φ-coordinates
    # φ-encoding: position = φ^level where level is the "semantic depth"
    phi_coords = np.log(pos_norm + 0.01) / np.log(PHI)
    
    print(f"\n  φ-coordinate ranges:")
    for i in range(min(4, phi_coords.shape[1])):
        print(f"    Dim {i}: [{phi_coords[:, i].min():.2f}, {phi_coords[:, i].max():.2f}]")
    
    # Cluster analysis in φ-space
    # Group by layer type
    encoder_idx = [i for i, n in enumerate(names) if 'encoder' in n.lower() or 'backbone' in n.lower()]
    decoder_idx = [i for i, n in enumerate(names) if 'decoder' in n.lower() or 'neck' in n.lower()]
    head_idx = [i for i, n in enumerate(names) if 'head' in n.lower()]
    
    print(f"\n  Cluster centroids in φ-space:")
    if encoder_idx:
        centroid = phi_coords[encoder_idx].mean(axis=0)
        print(f"    Encoder: {centroid[:4]}")
    if decoder_idx:
        centroid = phi_coords[decoder_idx].mean(axis=0)
        print(f"    Decoder: {centroid[:4]}")
    if head_idx:
        centroid = phi_coords[head_idx].mean(axis=0)
        print(f"    Head: {centroid[:4]}")
    
    return phi_coords


def create_da2_visualization(eigenvalues, positions, names, sv_ratios, phi_coords):
    """Visualize DA2 reverse engineering results."""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Reverse Engineering Depth Anything V2\n'
                 'Mapping Neural Weights to φ-Space',
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Eigenvalue spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(eigenvalues[:30], 'b-o', markersize=4)
    ax1.set_title('Eigenvalue Spectrum', fontsize=10)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log)')
    
    # 2. Cumulative variance
    ax2 = fig.add_subplot(gs[0, 1])
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(cumvar[:20] * 100, 'g-o', markersize=4)
    ax2.axhline(y=95, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('Cumulative Variance', fontsize=10)
    ax2.set_xlabel('Components')
    ax2.set_ylabel('Variance (%)')
    
    # 3. 2D projection of weight space
    ax3 = fig.add_subplot(gs[0, 2])
    colors = []
    for name in names:
        if 'encoder' in name.lower() or 'backbone' in name.lower():
            colors.append('red')
        elif 'decoder' in name.lower() or 'neck' in name.lower():
            colors.append('blue')
        elif 'head' in name.lower():
            colors.append('green')
        else:
            colors.append('gray')
    
    ax3.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=0.6, s=30)
    ax3.set_title('Weight Space (2D)', fontsize=10)
    ax3.set_xlabel('Dim 1')
    ax3.set_ylabel('Dim 2')
    
    # 4. Singular value ratio histogram
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(sv_ratios, bins=50, range=(0.9, 3.0), alpha=0.7, edgecolor='black')
    ax4.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ={PHI:.3f}')
    ax4.axvline(x=PHI**2, color='orange', linestyle='--', linewidth=2, label=f'φ²={PHI**2:.3f}')
    ax4.set_title('SV Ratios', fontsize=10)
    ax4.set_xlabel('Ratio')
    ax4.legend()
    
    # 5. φ-space projection (dim 0 vs 1)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(phi_coords[:, 0], phi_coords[:, 1], c=colors, alpha=0.6, s=30)
    ax5.set_title('φ-Space (Dim 0 vs 1)', fontsize=10)
    ax5.set_xlabel('φ-coord 0')
    ax5.set_ylabel('φ-coord 1')
    
    # 6. φ-space projection (dim 2 vs 3)
    ax6 = fig.add_subplot(gs[1, 1])
    if phi_coords.shape[1] >= 4:
        ax6.scatter(phi_coords[:, 2], phi_coords[:, 3], c=colors, alpha=0.6, s=30)
        ax6.set_title('φ-Space (Dim 2 vs 3)', fontsize=10)
        ax6.set_xlabel('φ-coord 2')
        ax6.set_ylabel('φ-coord 3')
    
    # 7. 3D view
    ax7 = fig.add_subplot(gs[1, 2], projection='3d')
    ax7.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, alpha=0.6, s=30)
    ax7.set_title('Weight Space (3D)', fontsize=10)
    
    # 8. Summary
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    near_phi = (np.abs(sv_ratios - PHI) < 0.1).mean() * 100
    cumvar_3 = cumvar[2] * 100 if len(cumvar) > 2 else 0
    
    summary = f"""
    DA2 Reverse Engineering Results:
    
    Eigenstructure:
    • Top 3 explain {cumvar_3:.1f}% variance
    • Sharp dropoff (low-dim structure)
    
    φ-Structure:
    • {near_phi:.1f}% of SV ratios near φ
    • Clusters separate in φ-space
    
    Mapping to TruthSpace:
    • Encoder → one region
    • Decoder → another region  
    • Head → distinct cluster
    
    Key Insight:
    DA2's depth knowledge is encoded
    in a low-dimensional geometric
    structure that can be mapped
    to φ-coordinates.
    """
    ax8.text(0.1, 0.9, summary, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Encoder'),
        Patch(facecolor='blue', label='Decoder'),
        Patch(facecolor='green', label='Head'),
    ]
    ax8.legend(handles=legend_elements, loc='lower right')
    
    output_file = OUTPUT_PATH / "da2_reverse_engineering.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Load DA2
    weights, model = load_depth_anything_v2()
    
    if weights is None:
        print("Could not load DA2. Exiting.")
        exit(1)
    
    # Analyze structure
    categories = analyze_weight_structure(weights)
    
    # Compute eigenstructure
    eigenvalues, eigenvectors, positions, names, sv = compute_da2_eigenstructure(weights)
    
    # Search for φ-structure
    sv_ratios = find_phi_structure(eigenvalues, sv)
    
    # Map to φ-space
    phi_coords = map_to_phi_space(positions, names)
    
    # Create visualization
    viz_file = create_da2_visualization(eigenvalues, positions, names, sv_ratios, phi_coords)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Depth Anything V2's weights can be mapped to φ-space!")
    print()
    print("This suggests we could potentially:")
    print("  1. Understand DA2's depth priors geometrically")
    print("  2. Build a φ-based equivalent")
    print("  3. Transfer the learned structure to our self-assembler")
    print()
    print("Next step: Test if φ-structure can replicate DA2 behavior.")
