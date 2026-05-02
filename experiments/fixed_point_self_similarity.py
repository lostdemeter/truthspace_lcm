#!/usr/bin/env python3
"""
Fixed Point Self-Similarity Analysis
======================================

Applying the Multifold Gushurst Optimization Protocol (MGOP) and
Equation Discovery Protocol (EDP) to analyze token fixed points.

Key Questions:
1. Do fixed points exhibit self-similarity (fractal structure)?
2. Are there φ-patterns in the fixed point coordinates?
3. What is the geometry of the vocabulary space?

Protocols Applied:
- MGOP Phase 1: Fractal Peel (spatial projection)
- MGOP Phase 3: Fractal Depth Probe (multi-scale)
- MGOP Phase 4: Zeta Resonance Test (number-theoretic)
- EDP Phase 4: Error Analysis (φ-patterns)

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import signal
from scipy.fft import fft, fftfreq

PHI = 1.6180339887498949
PHI_INV = 1 / PHI  # 0.618...


def collect_fixed_points():
    """
    Collect token fixed points (mean h_after for each token).
    """
    print("\n" + "=" * 70)
    print("Collecting Token Fixed Points")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Diverse prompts to collect fixed points
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The elephant is a large animal that",
        "The lion is a large animal that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "In the beginning there was",
        "Once upon a time there was a",
        "The meaning of life is",
        "Mathematics is the language of",
    ]
    
    n_tokens = 15
    
    token_h_afters = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_prev = outputs.hidden_states[-1][0, -1, :].clone()
        
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                h_curr = outputs.hidden_states[-1][0, -1, :].clone()
                token = outputs.logits[0, -1, :].argmax().item()
            
            token_h_afters[token].append(h_curr)
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # Compute fixed points (mean h_after)
    fixed_points = {}
    for token, h_afters in token_h_afters.items():
        if len(h_afters) >= 2:  # Need multiple samples
            fixed_points[token] = {
                "target": torch.stack(h_afters).mean(dim=0),
                "std": torch.stack(h_afters).std(dim=0),
                "n": len(h_afters),
                "text": tokenizer.decode([token]),
            }
    
    print(f"Collected {len(fixed_points)} fixed points")
    
    del model
    
    return fixed_points, tokenizer


def mgop_phase1_fractal_peel(fixed_points):
    """
    MGOP Phase 1: Fractal Peel (Spatial Projection)
    
    Analyze the spatial structure of fixed points.
    """
    print("\n" + "=" * 70)
    print("MGOP Phase 1: Fractal Peel")
    print("=" * 70)
    
    # Stack all fixed points into a matrix
    tokens = list(fixed_points.keys())
    targets = torch.stack([fixed_points[t]["target"] for t in tokens])
    
    print(f"Fixed point matrix shape: {targets.shape}")
    
    # 1. Compute pairwise similarities
    targets_norm = F.normalize(targets, dim=1)
    similarity = targets_norm @ targets_norm.T
    
    # Remove diagonal
    mask = ~torch.eye(len(tokens), dtype=torch.bool)
    off_diag = similarity[mask]
    
    print(f"\n--- Pairwise Similarity Statistics ---")
    print(f"  Mean: {off_diag.mean().item():.4f}")
    print(f"  Std:  {off_diag.std().item():.4f}")
    print(f"  Min:  {off_diag.min().item():.4f}")
    print(f"  Max:  {off_diag.max().item():.4f}")
    
    # 2. Compute autocorrelation of similarity matrix
    sim_np = similarity.numpy()
    autocorr = signal.correlate2d(sim_np, sim_np, mode='same')
    autocorr_norm = autocorr / autocorr.max()
    
    # Find peak (excluding center)
    center = np.array(autocorr_norm.shape) // 2
    autocorr_norm[center[0], center[1]] = 0
    peak_idx = np.unravel_index(np.argmax(autocorr_norm), autocorr_norm.shape)
    peak_value = autocorr_norm[peak_idx]
    peak_offset = (peak_idx[0] - center[0], peak_idx[1] - center[1])
    
    print(f"\n--- Autocorrelation Analysis ---")
    print(f"  Peak offset: {peak_offset}")
    print(f"  Peak value (resfrac): {peak_value:.4f}")
    
    # 3. Compute SVD to find principal structure
    U, S, Vh = torch.linalg.svd(targets)
    
    # Variance explained
    total_var = (S ** 2).sum()
    var_explained = (S ** 2).cumsum(0) / total_var
    
    print(f"\n--- SVD Analysis ---")
    print(f"  Top 10 singular values:")
    for i in range(min(10, len(S))):
        print(f"    S[{i}] = {S[i].item():.2f} ({var_explained[i].item()*100:.1f}% cumulative)")
    
    # How many components for 90% variance?
    n_90 = (var_explained < 0.9).sum().item() + 1
    print(f"  Components for 90% variance: {n_90}")
    
    # 4. Resfrac score (predictability)
    # If peak_value > 0.5, structure is ergodic (random)
    # If peak_value < 0.5, structure is exploitable
    
    is_ergodic = peak_value > 0.5
    
    print(f"\n--- Fractal Peel Decision ---")
    print(f"  Resfrac (ρ): {peak_value:.4f}")
    if is_ergodic:
        print(f"  Status: ERGODIC (ρ > 0.5) - structure may be random")
    else:
        print(f"  Status: STRUCTURED (ρ < 0.5) - exploitable patterns exist")
    
    return {
        "similarity": similarity,
        "autocorr": autocorr_norm,
        "peak_offset": peak_offset,
        "resfrac": peak_value,
        "svd": (U, S, Vh),
        "var_explained": var_explained,
        "is_ergodic": is_ergodic,
    }


def mgop_phase3_fractal_depth(fixed_points):
    """
    MGOP Phase 3: Fractal Depth Probe (Multi-Scale Projection)
    
    Measure fractal structure at multiple scales.
    """
    print("\n" + "=" * 70)
    print("MGOP Phase 3: Fractal Depth Probe")
    print("=" * 70)
    
    tokens = list(fixed_points.keys())
    targets = torch.stack([fixed_points[t]["target"] for t in tokens])
    
    # 1. Multi-scale analysis via progressive SVD
    scales = [1, 2, 4, 8, 16]
    
    print("\n--- Multi-Scale SVD Analysis ---")
    
    scale_results = []
    for scale in scales:
        # Downsample by averaging groups of dimensions
        n_dims = targets.shape[1]
        new_dims = n_dims // scale
        
        if new_dims < 10:
            break
        
        # Reshape and average
        targets_scaled = targets[:, :new_dims * scale].reshape(len(tokens), new_dims, scale).mean(dim=2)
        
        # SVD
        U, S, Vh = torch.linalg.svd(targets_scaled)
        
        # Effective rank (number of significant singular values)
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()
        
        # Zipf exponent (fit S[i] ~ 1/i^α)
        log_i = np.log(np.arange(1, len(S) + 1))
        log_S = np.log(S.numpy() + 1e-10)
        
        # Linear regression
        slope, intercept = np.polyfit(log_i[:min(50, len(S))], log_S[:min(50, len(S))], 1)
        zipf_exponent = -slope
        
        scale_results.append({
            "scale": scale,
            "dims": new_dims,
            "effective_rank": effective_rank,
            "zipf_exponent": zipf_exponent,
            "top_S": S[:5].tolist(),
        })
        
        print(f"\n  Scale {scale} ({new_dims} dims):")
        print(f"    Effective rank: {effective_rank:.2f}")
        print(f"    Zipf exponent: {zipf_exponent:.4f}")
        print(f"    Target (1/φ): {PHI_INV:.4f}")
        print(f"    Deviation: {abs(zipf_exponent - PHI_INV):.4f}")
    
    # 2. Self-similarity test
    # Check if Zipf exponent is consistent across scales
    exponents = [r["zipf_exponent"] for r in scale_results]
    exponent_var = np.var(exponents)
    
    print(f"\n--- Self-Similarity Test ---")
    print(f"  Zipf exponents: {[f'{e:.4f}' for e in exponents]}")
    print(f"  Variance: {exponent_var:.6f}")
    
    is_self_similar = exponent_var < 0.01
    print(f"  Self-similar: {'YES' if is_self_similar else 'NO'}")
    
    # 3. φ-Zipf test
    # Is the exponent close to 1/φ?
    mean_exponent = np.mean(exponents)
    phi_deviation = abs(mean_exponent - PHI_INV)
    
    print(f"\n--- φ-Zipf Test ---")
    print(f"  Mean exponent: {mean_exponent:.4f}")
    print(f"  Target (1/φ): {PHI_INV:.4f}")
    print(f"  Deviation: {phi_deviation:.4f}")
    
    is_phi_zipf = phi_deviation < 0.1
    print(f"  φ-Zipf: {'YES' if is_phi_zipf else 'NO'}")
    
    return {
        "scale_results": scale_results,
        "exponents": exponents,
        "exponent_var": exponent_var,
        "is_self_similar": is_self_similar,
        "mean_exponent": mean_exponent,
        "phi_deviation": phi_deviation,
        "is_phi_zipf": is_phi_zipf,
    }


def mgop_phase4_zeta_resonance(fixed_points):
    """
    MGOP Phase 4: Zeta Resonance Test (Number-Theoretic Projection)
    
    Check if fixed point structure resonates with fundamental constants.
    """
    print("\n" + "=" * 70)
    print("MGOP Phase 4: Zeta Resonance Test")
    print("=" * 70)
    
    tokens = list(fixed_points.keys())
    targets = torch.stack([fixed_points[t]["target"] for t in tokens])
    
    # 1. Extract characteristic frequencies from fixed points
    # Use FFT on the principal components
    U, S, Vh = torch.linalg.svd(targets)
    
    # Top principal component
    pc1 = Vh[0].numpy()
    
    # FFT
    fft_result = np.abs(fft(pc1))
    freqs = fftfreq(len(pc1))
    
    # Find peaks
    peak_indices = signal.find_peaks(fft_result[:len(fft_result)//2], height=np.max(fft_result)*0.1)[0]
    peak_freqs = freqs[peak_indices]
    peak_values = fft_result[peak_indices]
    
    print(f"\n--- FFT Peak Analysis ---")
    print(f"  Number of peaks: {len(peak_indices)}")
    if len(peak_indices) > 0:
        print(f"  Top 5 peak frequencies:")
        sorted_idx = np.argsort(peak_values)[::-1][:5]
        for i in sorted_idx:
            print(f"    f = {peak_freqs[i]:.6f}, amplitude = {peak_values[i]:.2f}")
    
    # 2. Compute spacings between fixed points
    # Use pairwise distances
    targets_norm = F.normalize(targets, dim=1)
    distances = 1 - (targets_norm @ targets_norm.T)
    
    # Get off-diagonal distances
    mask = ~torch.eye(len(tokens), dtype=torch.bool)
    off_diag_dist = distances[mask].numpy()
    
    # Histogram of distances
    hist, bin_edges = np.histogram(off_diag_dist, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find characteristic spacing (mode)
    mode_idx = np.argmax(hist)
    characteristic_spacing = bin_centers[mode_idx]
    
    print(f"\n--- Distance Distribution ---")
    print(f"  Mean distance: {off_diag_dist.mean():.4f}")
    print(f"  Std distance: {off_diag_dist.std():.4f}")
    print(f"  Characteristic spacing (mode): {characteristic_spacing:.4f}")
    
    # 3. Test resonance with φ
    # Check if characteristic spacing is related to φ
    phi_ratios = [
        ("φ^0", 1.0),
        ("φ^-1", PHI_INV),
        ("φ^-2", PHI_INV**2),
        ("φ^1", PHI),
        ("1/2", 0.5),
        ("1/3", 1/3),
        ("2/3", 2/3),
    ]
    
    print(f"\n--- φ-Resonance Test ---")
    best_match = None
    best_error = float('inf')
    
    for name, ratio in phi_ratios:
        error = abs(characteristic_spacing - ratio)
        print(f"  {name} ({ratio:.4f}): error = {error:.4f}")
        if error < best_error:
            best_error = error
            best_match = name
    
    print(f"\n  Best match: {best_match} (error = {best_error:.4f})")
    
    # 4. Test for 137/30 ratio (fine structure)
    # Look for this ratio in the singular value distribution
    S_ratios = []
    for i in range(len(S) - 1):
        if S[i+1] > 1e-6:
            S_ratios.append((S[i] / S[i+1]).item())
    
    fine_structure = 137 / 30  # ≈ 4.567
    
    print(f"\n--- Fine Structure (137/30) Test ---")
    print(f"  Target ratio: {fine_structure:.4f}")
    
    # Find closest ratio
    if S_ratios:
        closest_idx = np.argmin([abs(r - fine_structure) for r in S_ratios])
        closest_ratio = S_ratios[closest_idx]
        print(f"  Closest S[i]/S[i+1] ratio: {closest_ratio:.4f} at i={closest_idx}")
        print(f"  Deviation: {abs(closest_ratio - fine_structure):.4f}")
    
    return {
        "peak_freqs": peak_freqs.tolist() if len(peak_freqs) > 0 else [],
        "characteristic_spacing": characteristic_spacing,
        "best_phi_match": best_match,
        "best_phi_error": best_error,
        "S_ratios": S_ratios[:10] if S_ratios else [],
    }


def edp_phase4_phi_patterns(fixed_points):
    """
    EDP Phase 4: Error Analysis (φ-Patterns)
    
    Search for φ-patterns in fixed point coordinates.
    """
    print("\n" + "=" * 70)
    print("EDP Phase 4: φ-Pattern Search")
    print("=" * 70)
    
    tokens = list(fixed_points.keys())
    targets = torch.stack([fixed_points[t]["target"] for t in tokens])
    
    # 1. Analyze coordinate distribution
    all_coords = targets.flatten().numpy()
    
    print(f"\n--- Coordinate Distribution ---")
    print(f"  Mean: {all_coords.mean():.6f}")
    print(f"  Std: {all_coords.std():.6f}")
    print(f"  Min: {all_coords.min():.6f}")
    print(f"  Max: {all_coords.max():.6f}")
    
    # 2. Search for φ^k patterns in coordinate magnitudes
    coord_mags = np.abs(all_coords)
    
    # Bin by φ^k
    phi_bins = {}
    for k in range(-15, 15):
        phi_k = PHI ** k
        # Count coordinates within 10% of φ^k
        count = np.sum((coord_mags > phi_k * 0.9) & (coord_mags < phi_k * 1.1))
        if count > 0:
            phi_bins[k] = count
    
    print(f"\n--- φ^k Binning ---")
    for k, count in sorted(phi_bins.items(), key=lambda x: -x[1])[:10]:
        phi_k = PHI ** k
        print(f"  φ^{k:+3d} = {phi_k:.6f}: {count} coordinates ({count/len(all_coords)*100:.2f}%)")
    
    # 3. Search for (n/d) × φ^k patterns in top singular values
    U, S, Vh = torch.linalg.svd(targets)
    
    print(f"\n--- Singular Value φ-Patterns ---")
    
    def find_phi_pattern(value, max_n=50, max_d=50, max_k=15):
        """Find (n/d) × φ^k approximation."""
        best = None
        best_error = float('inf')
        
        for k in range(-max_k, max_k):
            phi_k = PHI ** k
            for d in range(1, max_d + 1):
                for n in range(-max_n, max_n + 1):
                    if n == 0:
                        continue
                    approx = (n / d) * phi_k
                    error = abs(value - approx)
                    if error < best_error:
                        best_error = error
                        best = (n, d, k, approx, error)
        
        return best
    
    for i in range(min(10, len(S))):
        s_val = S[i].item()
        pattern = find_phi_pattern(s_val)
        if pattern:
            n, d, k, approx, error = pattern
            rel_error = error / abs(s_val) if s_val != 0 else float('inf')
            clean = abs(n) <= 20 and d <= 20 and rel_error < 0.01
            marker = "← CLEAN!" if clean else ""
            print(f"  S[{i}] = {s_val:.4f} ≈ ({n}/{d}) × φ^{k} = {approx:.4f} (err={rel_error:.4f}) {marker}")
    
    # 4. Analyze ratios between consecutive singular values
    print(f"\n--- Singular Value Ratios ---")
    
    for i in range(min(10, len(S) - 1)):
        if S[i+1] > 1e-6:
            ratio = (S[i] / S[i+1]).item()
            
            # Check if ratio is close to φ or powers of φ
            phi_matches = []
            for k in range(-3, 4):
                phi_k = PHI ** k
                if abs(ratio - phi_k) < 0.1:
                    phi_matches.append(f"φ^{k}")
            
            match_str = f" ≈ {', '.join(phi_matches)}" if phi_matches else ""
            print(f"  S[{i}]/S[{i+1}] = {ratio:.4f}{match_str}")
    
    return {
        "phi_bins": phi_bins,
        "top_S": S[:10].tolist(),
    }


def analyze_fixed_point_geometry(fixed_points, tokenizer):
    """
    Analyze the geometry of fixed points in vocabulary space.
    """
    print("\n" + "=" * 70)
    print("Fixed Point Geometry Analysis")
    print("=" * 70)
    
    tokens = list(fixed_points.keys())
    targets = torch.stack([fixed_points[t]["target"] for t in tokens])
    
    # 1. Cluster analysis
    # Group tokens by similarity
    targets_norm = F.normalize(targets, dim=1)
    similarity = targets_norm @ targets_norm.T
    
    # Find highly similar pairs
    print(f"\n--- Highly Similar Fixed Point Pairs ---")
    
    pairs = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            sim = similarity[i, j].item()
            if sim > 0.9:
                pairs.append((tokens[i], tokens[j], sim))
    
    pairs.sort(key=lambda x: -x[2])
    
    for t1, t2, sim in pairs[:10]:
        text1 = fixed_points[t1]["text"]
        text2 = fixed_points[t2]["text"]
        print(f"  {text1!r} ↔ {text2!r}: sim = {sim:.4f}")
    
    # 2. Find clusters using simple thresholding
    print(f"\n--- Fixed Point Clusters (sim > 0.8) ---")
    
    visited = set()
    clusters = []
    
    for i in range(len(tokens)):
        if i in visited:
            continue
        
        cluster = [i]
        visited.add(i)
        
        for j in range(len(tokens)):
            if j not in visited and similarity[i, j] > 0.8:
                cluster.append(j)
                visited.add(j)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    for cluster in clusters[:5]:
        texts = [fixed_points[tokens[i]]["text"] for i in cluster]
        print(f"  Cluster: {texts}")
    
    # 3. Analyze fixed point norms
    norms = torch.norm(targets, dim=1)
    
    print(f"\n--- Fixed Point Norms ---")
    print(f"  Mean: {norms.mean().item():.2f}")
    print(f"  Std: {norms.std().item():.2f}")
    print(f"  Min: {norms.min().item():.2f} ({fixed_points[tokens[norms.argmin()]]['text']!r})")
    print(f"  Max: {norms.max().item():.2f} ({fixed_points[tokens[norms.argmax()]]['text']!r})")
    
    # Check if norms follow φ-pattern
    norm_ratio = norms.max() / norms.min()
    print(f"  Max/Min ratio: {norm_ratio.item():.4f}")
    print(f"  φ^2: {PHI**2:.4f}")
    print(f"  φ^3: {PHI**3:.4f}")
    
    return {
        "clusters": clusters,
        "similar_pairs": pairs[:10],
        "norms": norms.tolist(),
    }


if __name__ == "__main__":
    # 1. Collect fixed points
    fixed_points, tokenizer = collect_fixed_points()
    
    # 2. MGOP Phase 1: Fractal Peel
    phase1_results = mgop_phase1_fractal_peel(fixed_points)
    
    # 3. MGOP Phase 3: Fractal Depth Probe
    phase3_results = mgop_phase3_fractal_depth(fixed_points)
    
    # 4. MGOP Phase 4: Zeta Resonance Test
    phase4_results = mgop_phase4_zeta_resonance(fixed_points)
    
    # 5. EDP Phase 4: φ-Pattern Search
    edp_results = edp_phase4_phi_patterns(fixed_points)
    
    # 6. Geometry Analysis
    geometry_results = analyze_fixed_point_geometry(fixed_points, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Fixed Point Self-Similarity Analysis")
    print("=" * 70)
    
    print(f"\n1. Fractal Peel (MGOP Phase 1):")
    print(f"   Resfrac: {phase1_results['resfrac']:.4f}")
    print(f"   Ergodic: {phase1_results['is_ergodic']}")
    
    print(f"\n2. Fractal Depth (MGOP Phase 3):")
    print(f"   Self-similar: {phase3_results['is_self_similar']}")
    print(f"   φ-Zipf: {phase3_results['is_phi_zipf']}")
    print(f"   Mean Zipf exponent: {phase3_results['mean_exponent']:.4f} (target 1/φ = {PHI_INV:.4f})")
    
    print(f"\n3. Zeta Resonance (MGOP Phase 4):")
    print(f"   Characteristic spacing: {phase4_results['characteristic_spacing']:.4f}")
    print(f"   Best φ match: {phase4_results['best_phi_match']} (error = {phase4_results['best_phi_error']:.4f})")
    
    print(f"\n4. φ-Patterns (EDP Phase 4):")
    print(f"   Top φ^k bins: {list(edp_results['phi_bins'].keys())[:5]}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if phase3_results['is_phi_zipf']:
        print("\n✓ Fixed points exhibit φ-Zipf distribution!")
        print("  The singular values follow S[i] ~ 1/i^(1/φ)")
        print("  This is the self-similar balance point.")
    
    if not phase1_results['is_ergodic']:
        print("\n✓ Fixed points have exploitable structure!")
        print("  Resfrac < 0.5 indicates non-random patterns.")
    
    if phase3_results['is_self_similar']:
        print("\n✓ Fixed points are self-similar across scales!")
        print("  The same structure appears at different resolutions.")
