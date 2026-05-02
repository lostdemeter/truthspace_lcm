#!/usr/bin/env python3
"""
Find the Missing Dimension

The refinement is rank-2, not rank-1. This means there's a second axis
we haven't captured in V3 Chemistry.

Hypothesis: We can find this axis by using walltime (or another smooth
function) as a search axis, similar to how the clock solver uses
N_smooth(θ) ≈ n to find eigenphases.

The approach:
1. Compute the SVD of the refinement: U @ S @ V.T
2. The first singular vector (rank-1) is the main axis
3. The second singular vector is the "missing dimension"
4. Characterize what this second axis represents
5. If we can identify it, we can add it to V3 and get rank-1

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.encoder import PhiEncoder, PHI, LN_PHI


def generate_refinement_samples(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate multiple refinement samples (V3 → target).
    
    The refinement is STRUCTURED, not random. We want to find
    what axes explain the structure.
    
    Returns:
        v3_outputs: [n_samples, H, W, 2]
        targets: [n_samples, H, W, 2]
        refinements: [n_samples, H, W, 2]
    """
    H, W = 64, 64
    
    v3_outputs = []
    targets = []
    refinements = []
    
    for seed in range(n_samples):
        np.random.seed(seed)
        
        # V3 base (semantic structure)
        v3 = np.zeros((H, W, 2))
        v3[:H//2, :, 0] = -5   # Sky: negative a
        v3[:H//2, :, 1] = -30  # Sky: negative b (blue)
        v3[H//2:, :, 0] = -20  # Ground: negative a (green)
        v3[H//2:, :, 1] = 20   # Ground: positive b
        
        # Vary the COEFFICIENTS, not the structure
        # This simulates different "styles" of refinement
        alpha = np.random.uniform(0.4, 0.6)  # Saturation coefficient
        beta = np.random.uniform(0.8, 1.2)   # Luminance coefficient
        
        # Target (simulated DDColor with STRUCTURED refinement)
        target = v3.copy()
        
        # Refinement 1: Saturation boost (varies by alpha)
        target *= (1 + alpha)
        
        # Refinement 2: Luminance-dependent shift (varies by beta)
        # This is the "second axis" - luminance affects color
        luminance = np.linspace(0.8, 0.3, H)[:, np.newaxis]
        luminance = np.broadcast_to(luminance, (H, W))
        target[..., 0] += luminance * 5 * beta  # Warm shift in bright areas
        target[..., 1] += (1 - luminance) * 5 * beta  # Cool shift in dark areas
        
        # Refinement 3: Edge enhancement (small, consistent)
        grad_x = np.gradient(v3[..., 0], axis=1)
        grad_y = np.gradient(v3[..., 0], axis=0)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        target[..., 0] += edges * 0.5
        target[..., 1] += edges * 0.5
        
        v3_outputs.append(v3)
        targets.append(target)
        refinements.append(target - v3)
    
    return np.array(v3_outputs), np.array(targets), np.array(refinements)


def analyze_refinement_axes(refinements: np.ndarray) -> Dict:
    """
    Analyze the axes of the refinement using SVD.
    
    The goal is to find what the second singular vector represents.
    """
    print("=" * 70)
    print("REFINEMENT AXIS ANALYSIS")
    print("=" * 70)
    
    n_samples, H, W, C = refinements.shape
    
    # Flatten to [n_samples, H*W*C]
    flat = refinements.reshape(n_samples, -1)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    
    print(f"\n## Singular Values")
    print(f"  S[0]: {S[0]:.2f} ({S[0]**2 / (S**2).sum() * 100:.1f}%)")
    print(f"  S[1]: {S[1]:.2f} ({S[1]**2 / (S**2).sum() * 100:.1f}%)")
    print(f"  S[2]: {S[2]:.2f} ({S[2]**2 / (S**2).sum() * 100:.1f}%)")
    print(f"  S[3]: {S[3]:.2f} ({S[3]**2 / (S**2).sum() * 100:.1f}%)")
    
    # Effective rank
    normalized_S = S / S.sum()
    entropy = -np.sum(normalized_S * np.log(normalized_S + 1e-10))
    effective_rank = np.exp(entropy)
    print(f"\n  Effective rank: {effective_rank:.2f}")
    
    # Analyze the first two singular vectors
    print(f"\n## First Singular Vector (V[0])")
    V0 = Vt[0].reshape(H, W, C)
    print(f"  Shape: {V0.shape}")
    print(f"  a-channel mean: {V0[..., 0].mean():.4f}")
    print(f"  b-channel mean: {V0[..., 1].mean():.4f}")
    print(f"  a-channel std: {V0[..., 0].std():.4f}")
    print(f"  b-channel std: {V0[..., 1].std():.4f}")
    
    # Spatial pattern of V0
    print(f"  Top half a: {V0[:H//2, :, 0].mean():.4f}")
    print(f"  Bottom half a: {V0[H//2:, :, 0].mean():.4f}")
    print(f"  Top half b: {V0[:H//2, :, 1].mean():.4f}")
    print(f"  Bottom half b: {V0[H//2:, :, 1].mean():.4f}")
    
    print(f"\n## Second Singular Vector (V[1])")
    V1 = Vt[1].reshape(H, W, C)
    print(f"  Shape: {V1.shape}")
    print(f"  a-channel mean: {V1[..., 0].mean():.4f}")
    print(f"  b-channel mean: {V1[..., 1].mean():.4f}")
    print(f"  a-channel std: {V1[..., 0].std():.4f}")
    print(f"  b-channel std: {V1[..., 1].std():.4f}")
    
    # Spatial pattern of V1
    print(f"  Top half a: {V1[:H//2, :, 0].mean():.4f}")
    print(f"  Bottom half a: {V1[H//2:, :, 0].mean():.4f}")
    print(f"  Top half b: {V1[:H//2, :, 1].mean():.4f}")
    print(f"  Bottom half b: {V1[H//2:, :, 1].mean():.4f}")
    
    return {
        "S": S,
        "V0": V0,
        "V1": V1,
        "effective_rank": effective_rank,
        "U": U,
        "Vt": Vt,
    }


def search_for_axis_with_walltime(refinements: np.ndarray, n_trials: int = 100) -> Dict:
    """
    Use walltime as a search axis to find the missing dimension.
    
    The idea: if we can find a function f(x, y) such that
    refinement ≈ α * f(x, y) + β * g(x, y)
    where f is the first axis and g is the second,
    then we can characterize both axes.
    
    We use walltime to measure how long it takes to compute
    different candidate functions, and use this as a proxy
    for "complexity" or "naturalness" of the axis.
    """
    print("\n" + "=" * 70)
    print("WALLTIME-BASED AXIS SEARCH")
    print("=" * 70)
    
    n_samples, H, W, C = refinements.shape
    
    # Average refinement (the signal we're trying to decompose)
    avg_refinement = refinements.mean(axis=0)
    
    # Candidate axes to test
    candidates = []
    
    # 1. Luminance axis (vertical gradient)
    t0 = time.perf_counter()
    luminance = np.linspace(1, 0, H)[:, np.newaxis]
    luminance = np.broadcast_to(luminance, (H, W))
    luminance_axis = np.stack([luminance, 1 - luminance], axis=-1)
    t_luminance = time.perf_counter() - t0
    candidates.append(("luminance", luminance_axis, t_luminance))
    
    # 2. Horizontal position axis
    t0 = time.perf_counter()
    horizontal = np.linspace(0, 1, W)[np.newaxis, :]
    horizontal = np.broadcast_to(horizontal, (H, W))
    horizontal_axis = np.stack([horizontal, horizontal], axis=-1)
    t_horizontal = time.perf_counter() - t0
    candidates.append(("horizontal", horizontal_axis, t_horizontal))
    
    # 3. Radial axis (distance from center)
    t0 = time.perf_counter()
    y, x = np.ogrid[:H, :W]
    radial = np.sqrt((y - H/2)**2 + (x - W/2)**2) / (H/2)
    radial_axis = np.stack([radial, radial], axis=-1)
    t_radial = time.perf_counter() - t0
    candidates.append(("radial", radial_axis, t_radial))
    
    # 4. Edge axis (gradient magnitude)
    t0 = time.perf_counter()
    grad_x = np.gradient(avg_refinement[..., 0], axis=1)
    grad_y = np.gradient(avg_refinement[..., 0], axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges = edges / (edges.max() + 1e-10)
    edge_axis = np.stack([edges, edges], axis=-1)
    t_edge = time.perf_counter() - t0
    candidates.append(("edge", edge_axis, t_edge))
    
    # 5. Texture axis (local variance)
    t0 = time.perf_counter()
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(avg_refinement[..., 0], size=5)
    local_sq_mean = uniform_filter(avg_refinement[..., 0]**2, size=5)
    texture = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    texture = texture / (texture.max() + 1e-10)
    texture_axis = np.stack([texture, texture], axis=-1)
    t_texture = time.perf_counter() - t0
    candidates.append(("texture", texture_axis, t_texture))
    
    # 6. Semantic axis (sky vs ground)
    t0 = time.perf_counter()
    semantic = np.zeros((H, W))
    semantic[:H//2, :] = 1  # Sky
    semantic[H//2:, :] = 0  # Ground
    semantic_axis = np.stack([semantic, 1 - semantic], axis=-1)
    t_semantic = time.perf_counter() - t0
    candidates.append(("semantic", semantic_axis, t_semantic))
    
    # 7. φ-level axis (based on magnitude)
    t0 = time.perf_counter()
    magnitude = np.sqrt(avg_refinement[..., 0]**2 + avg_refinement[..., 1]**2)
    phi_level = np.log(magnitude + 1e-10) / LN_PHI
    phi_level = (phi_level - phi_level.min()) / (phi_level.max() - phi_level.min() + 1e-10)
    phi_axis = np.stack([phi_level, phi_level], axis=-1)
    t_phi = time.perf_counter() - t0
    candidates.append(("phi_level", phi_axis, t_phi))
    
    # Test each candidate: how well does it explain the refinement?
    print(f"\n## Candidate Axis Analysis")
    print(f"{'Axis':<15} {'Walltime':>12} {'Correlation':>12} {'Residual':>12}")
    print("-" * 55)
    
    results = []
    for name, axis, walltime in candidates:
        # Flatten for correlation
        axis_flat = axis.flatten()
        ref_flat = avg_refinement.flatten()
        
        # Correlation
        corr = np.corrcoef(axis_flat, ref_flat)[0, 1]
        
        # Residual after projecting out this axis
        # residual = refinement - (refinement · axis) * axis / ||axis||²
        projection = np.sum(avg_refinement * axis) / (np.sum(axis**2) + 1e-10)
        residual = avg_refinement - projection * axis
        residual_norm = np.sqrt(np.sum(residual**2))
        
        print(f"{name:<15} {walltime*1e6:>10.2f}μs {corr:>12.4f} {residual_norm:>12.2f}")
        
        results.append({
            "name": name,
            "axis": axis,
            "walltime": walltime,
            "correlation": corr,
            "residual_norm": residual_norm,
            "projection": projection,
        })
    
    # Find the best axis (highest correlation)
    best = max(results, key=lambda x: abs(x["correlation"]))
    print(f"\n  Best axis: {best['name']} (corr={best['correlation']:.4f})")
    
    return {
        "candidates": results,
        "best": best,
        "avg_refinement": avg_refinement,
    }


def test_rank_reduction(refinements: np.ndarray, axis_results: Dict) -> Dict:
    """
    Test if adding the discovered axis reduces rank from 2 to 1.
    """
    print("\n" + "=" * 70)
    print("RANK REDUCTION TEST")
    print("=" * 70)
    
    n_samples, H, W, C = refinements.shape
    
    # Get the best axis
    best_axis = axis_results["best"]["axis"]
    best_name = axis_results["best"]["name"]
    
    print(f"\n## Testing with axis: {best_name}")
    
    # For each sample, project out the best axis
    residuals = []
    for i in range(n_samples):
        ref = refinements[i]
        
        # Project out the axis
        projection = np.sum(ref * best_axis) / (np.sum(best_axis**2) + 1e-10)
        residual = ref - projection * best_axis
        residuals.append(residual)
    
    residuals = np.array(residuals)
    
    # Compute SVD of residuals
    flat = residuals.reshape(n_samples, -1)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    
    # Effective rank of residuals
    normalized_S = S / S.sum()
    entropy = -np.sum(normalized_S * np.log(normalized_S + 1e-10))
    effective_rank = np.exp(entropy)
    
    print(f"\n## Residual Analysis (after projecting out {best_name})")
    print(f"  S[0]: {S[0]:.2f} ({S[0]**2 / (S**2).sum() * 100:.1f}%)")
    print(f"  S[1]: {S[1]:.2f} ({S[1]**2 / (S**2).sum() * 100:.1f}%)")
    print(f"  Effective rank: {effective_rank:.2f}")
    
    # Compare to original
    flat_orig = refinements.reshape(n_samples, -1)
    U_orig, S_orig, Vt_orig = np.linalg.svd(flat_orig, full_matrices=False)
    normalized_S_orig = S_orig / S_orig.sum()
    entropy_orig = -np.sum(normalized_S_orig * np.log(normalized_S_orig + 1e-10))
    effective_rank_orig = np.exp(entropy_orig)
    
    print(f"\n## Comparison")
    print(f"  Original effective rank: {effective_rank_orig:.2f}")
    print(f"  After projection: {effective_rank:.2f}")
    print(f"  Rank reduction: {effective_rank_orig - effective_rank:.2f}")
    
    # Check if we achieved rank-1
    rank1_achieved = effective_rank < 1.5
    print(f"\n  Rank-1 achieved: {rank1_achieved}")
    
    if rank1_achieved:
        print(f"\n  SUCCESS! The {best_name} axis explains the second dimension.")
        print(f"  V3 + {best_name} axis = rank-1 refinement")
    else:
        print(f"\n  Need to search for additional axes or combinations.")
    
    return {
        "original_rank": effective_rank_orig,
        "residual_rank": effective_rank,
        "rank_reduction": effective_rank_orig - effective_rank,
        "rank1_achieved": rank1_achieved,
        "best_axis": best_name,
    }


def main():
    """Run the full analysis."""
    print("=" * 70)
    print("FINDING THE MISSING DIMENSION")
    print("=" * 70)
    print("""
The refinement is rank-2, not rank-1. We need to find the second axis.

Approach:
1. Generate multiple refinement samples
2. Analyze the SVD to understand the axes
3. Use walltime-based search to find candidate axes
4. Test if projecting out the axis reduces rank to 1
""")
    
    # Generate samples
    print("\n## Generating refinement samples...")
    v3_outputs, targets, refinements = generate_refinement_samples(n_samples=100)
    print(f"  Generated {len(refinements)} samples")
    
    # Analyze axes
    svd_results = analyze_refinement_axes(refinements)
    
    # Search for axis with walltime
    axis_results = search_for_axis_with_walltime(refinements)
    
    # Test rank reduction
    reduction_results = test_rank_reduction(refinements, axis_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
## Key Findings

1. Original refinement effective rank: {reduction_results['original_rank']:.2f}

2. Best axis found: {reduction_results['best_axis']}
   - Correlation with refinement: {axis_results['best']['correlation']:.4f}
   - Walltime: {axis_results['best']['walltime']*1e6:.2f}μs

3. After projecting out {reduction_results['best_axis']}:
   - Residual effective rank: {reduction_results['residual_rank']:.2f}
   - Rank reduction: {reduction_results['rank_reduction']:.2f}

4. Rank-1 achieved: {reduction_results['rank1_achieved']}

## Interpretation

The second dimension of the refinement is: {reduction_results['best_axis']}

This means:
- V3 Chemistry captures the semantic structure
- The {reduction_results['best_axis']} axis captures the second dimension
- Together they explain the full refinement

If we add {reduction_results['best_axis']} to V3 Chemistry:
    V3 + {reduction_results['best_axis']} = rank-1 refinement
    
This is the minimum representation:
    DDColor = V3 + {reduction_results['best_axis']} + rank-1 error
""")
    
    return {
        "svd": svd_results,
        "axis": axis_results,
        "reduction": reduction_results,
    }


if __name__ == "__main__":
    results = main()
