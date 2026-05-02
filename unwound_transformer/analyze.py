#!/usr/bin/env python3
"""
Geometric Analysis of Transformer Computation
==============================================

Examines the full computation chain to discover:
1. Hidden state trajectory patterns
2. Attention geometry
3. φ-level alignment
4. Cross-layer invariants
"""

import numpy as np
from model import UnwoundQwen2
from geometry import (
    analyze_trace, 
    print_analysis_summary,
    compute_trajectory_curvature,
    find_attention_anchors,
    compute_layer_similarity_matrix,
    analyze_weight_spectrum,
    find_geometric_invariants
)

PHI = (1 + np.sqrt(5)) / 2


def main():
    print("=" * 70)
    print("GEOMETRIC ANALYSIS OF TRANSFORMER COMPUTATION")
    print("=" * 70)
    
    model = UnwoundQwen2()
    
    # Analyze multiple traces to find invariants
    print("\n--- Collecting traces for analysis ---")
    traces = []
    np.random.seed(123)
    
    for i in range(10):
        A = np.random.randint(100, 5000)
        B = np.random.randint(100, 5000)
        trace = model.forward_with_trace(A, B)
        traces.append(trace)
        print(f"  Trace {i+1}: ({A}, {B}) -> {trace.predicted_token} ('{model.decode_token(trace.predicted_token)}')")
    
    # Analyze first trace in detail
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF FIRST TRACE")
    print("=" * 70)
    
    analysis = analyze_trace(traces[0])
    print_analysis_summary(analysis)
    
    # Trajectory curvature
    print("\n--- Trajectory Curvature ---")
    curvatures = compute_trajectory_curvature(analysis.hidden_directions)
    high_curvature = [(i+1, c) for i, c in enumerate(curvatures) if c > 0.5]
    print(f"  High curvature layers (>0.5): {high_curvature}")
    
    # Attention anchors
    print("\n--- Attention Anchors ---")
    anchors = find_attention_anchors(analysis, threshold=0.8)
    for layer, heads in sorted(anchors.items()):
        print(f"  Layer {layer}: heads {heads}")
    
    # Find invariants across traces
    print("\n" + "=" * 70)
    print("CROSS-TRACE INVARIANTS")
    print("=" * 70)
    
    invariants = find_geometric_invariants(traces)
    
    print("\n--- Consistent Direction Layers ---")
    print(f"  Layers with consistent direction change: {invariants['consistent_direction_layers']}")
    
    print("\n--- Norm Growth Ratios ---")
    ratios = invariants['mean_norm_ratios']
    print(f"  Mean ratios per layer: {[f'{r:.3f}' for r in ratios[:10]]}...")
    print(f"  φ-aligned layers: {invariants['phi_aligned_layers']}")
    print(f"  1/φ-aligned layers: {invariants['inv_phi_aligned_layers']}")
    
    # Analyze weight structure
    print("\n" + "=" * 70)
    print("WEIGHT MATRIX ANALYSIS")
    print("=" * 70)
    
    print("\n--- Layer Similarity Matrix ---")
    sim_matrix = compute_layer_similarity_matrix(model)
    
    # Find most similar layer pairs
    similar_pairs = []
    for i in range(28):
        for j in range(i+1, 28):
            if sim_matrix[i, j] > 0.5:
                similar_pairs.append((i, j, sim_matrix[i, j]))
    
    similar_pairs.sort(key=lambda x: -x[2])
    print(f"  Most similar layer pairs (W_q cosine > 0.5):")
    for i, j, sim in similar_pairs[:5]:
        print(f"    Layers {i}-{j}: {sim:.4f}")
    
    # Analyze W_q spectrum for a few layers
    print("\n--- Weight Spectrum Analysis ---")
    for layer_idx in [0, 13, 27]:
        spectrum = analyze_weight_spectrum(model.layers[layer_idx]['W_q'])
        print(f"  Layer {layer_idx} W_q:")
        print(f"    Effective rank: {spectrum['effective_rank']}")
        print(f"    Condition number: {spectrum['condition_number']:.2f}")
        print(f"    φ-aligned singular values (top 20): {spectrum['phi_aligned_count']}/20")
    
    # Key geometric findings
    print("\n" + "=" * 70)
    print("KEY GEOMETRIC FINDINGS")
    print("=" * 70)
    
    # 1. Norm growth pattern
    all_norms = np.array([analyze_trace(t).hidden_norms for t in traces])
    mean_norms = np.mean(all_norms, axis=0)
    
    print("\n1. HIDDEN STATE NORM GROWTH")
    print(f"   Embedding -> Layer 0: {mean_norms[1]/mean_norms[0]:.2f}x (large jump)")
    print(f"   Layer 0 -> Layer 27: {mean_norms[-1]/mean_norms[1]:.2f}x (gradual growth)")
    
    # 2. Direction stability
    all_changes = np.array([analyze_trace(t).direction_changes for t in traces])
    mean_changes = np.mean(all_changes, axis=0)
    
    stable_layers = np.where(mean_changes > 0.95)[0]
    print(f"\n2. DIRECTION STABILITY")
    print(f"   Stable layers (cos > 0.95): {stable_layers.tolist()}")
    print(f"   Mean direction change: {np.mean(mean_changes):.4f}")
    
    # 3. φ-alignment
    all_phi = np.array([analyze_trace(t).phi_residuals for t in traces])
    mean_phi = np.mean(all_phi, axis=0)
    
    phi_aligned = np.where(mean_phi < 0.1)[0]
    print(f"\n3. φ-ALIGNMENT")
    print(f"   φ-aligned layers (residual < 0.1): {phi_aligned.tolist()}")
    print(f"   Mean φ-residual: {np.mean(mean_phi):.4f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
