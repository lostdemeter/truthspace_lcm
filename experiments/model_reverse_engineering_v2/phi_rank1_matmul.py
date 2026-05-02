#!/usr/bin/env python3
"""
Rank-1 Level Matmul — The Hidden Dimension Exploited
======================================================

DISCOVERY: The level (exponent) matrix is RANK 1.
  level[j,i] ≈ σ₁ × u₁[j] × v₁[i]

This means:
  W[j,i] = sign[j,i] × φ^(u[j] × v[i])
          = sign[j,i] × magnitude_field[j,i]

where magnitude_field is a RANK-1 outer product of two vectors.

The weight matrix separates into:
  - SIGN: binary (±1), full-rank → carries the INFORMATION
  - MAGNITUDE: rank-1, two vectors → carries the SCALE

Computation model:
  1. Precompute v (column scales) and u (row scales) — O(N) each
  2. Discretize u into K buckets. For each bucket k:
     a. Scale input: x_k[i] = φ^(u_k × v[i]) × x[i]  — O(N)
     b. Binary matmul: y_rows = sign_rows @ x_k          — add/sub only!
  3. Total: O(K × N) scaling + O(N × out_f) binary ops

The binary matmul is add/subtract, not multiply.
On XOR hardware: trivially parallel.
"""

import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')


def levels(W):
    return W.exponents.astype(np.int32) // PHI_GRID


# ============================================================================
# Extract the rank-1 structure
# ============================================================================

def extract_rank1(W):
    """
    SVD of the level matrix → rank-1 decomposition.
    Returns u (row scales), v (col scales), sigma (overall scale).
    """
    lvl = levels(W).astype(np.float32)
    U, sigma, Vt = np.linalg.svd(lvl, full_matrices=False)
    
    # Rank-1: level ≈ sigma[0] * U[:,0] * Vt[0,:]
    u = U[:, 0] * sigma[0]  # row scales (out_f,)
    v = Vt[0, :]            # col scales (in_f,)
    
    # Verify
    lvl_rank1 = np.outer(u, v)
    residual = np.linalg.norm(lvl - lvl_rank1) / np.linalg.norm(lvl)
    
    return u, v, sigma, residual


# ============================================================================
# Discretize row scales
# ============================================================================

def discretize_u(u, K):
    """
    Quantize row scales into K buckets.
    Returns bucket indices and representative values.
    """
    # Use quantile-based binning for even distribution
    percentiles = np.linspace(0, 100, K + 1)
    edges = np.percentile(u, percentiles)
    
    bucket_idx = np.digitize(u, edges[1:-1])  # 0 to K-1
    
    # Representative value per bucket (mean of members)
    bucket_vals = np.zeros(K)
    bucket_counts = np.zeros(K, dtype=int)
    for k in range(K):
        mask = bucket_idx == k
        if np.any(mask):
            bucket_vals[k] = np.mean(u[mask])
            bucket_counts[k] = np.sum(mask)
    
    return bucket_idx, bucket_vals, bucket_counts


# ============================================================================
# Rank-1 Discretized Matmul
# ============================================================================

def rank1_matmul(W, x_float, K_values=None, name=""):
    """
    The full pipeline:
    1. Extract rank-1 level: u (rows), v (cols)
    2. Discretize u into K buckets
    3. For each bucket: scale input, binary matmul
    """
    if K_values is None:
        K_values = [1, 2, 4, 8, 16, 32, 64]
    
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    full = x_float @ W_dec.T
    
    u, v, sigma, residual = extract_rank1(W)
    signs = W.signs.astype(np.float32)  # (out_f, in_f)
    
    print(f"\n  Rank-1 level structure ({name}):")
    print(f"    Level residual (rank-1): {residual:.4f}")
    print(f"    σ₁/σ₂ ratio: {sigma[0]/sigma[1]:.1f}")
    print(f"    u range: [{u.min():.2f}, {u.max():.2f}]")
    print(f"    v range: [{v.min():.4f}, {v.max():.4f}]")
    
    # Exact rank-1 matmul (continuous u, no discretization)
    # W_r1[j,i] = sign[j,i] × φ^(u[j] × v[i])
    mag_r1 = PHI ** np.outer(u, v).astype(np.float64)
    W_r1 = signs.astype(np.float64) * mag_r1
    result_r1 = (x_float.astype(np.float64) @ W_r1.T).astype(np.float32)
    corr_r1 = np.corrcoef(full.flatten(), result_r1.flatten())[0, 1]
    print(f"    Rank-1 level (exact u): corr={corr_r1:.6f}")
    
    # Discretized versions
    print(f"\n    Discretized u (K buckets):")
    
    for K in K_values:
        if K > out_f:
            break
        
        bucket_idx, bucket_vals, bucket_counts = discretize_u(u, K)
        
        # Precompute K scaled inputs
        t0 = time.perf_counter()
        scaled_inputs = np.zeros((K, in_f), dtype=np.float64)
        for k in range(K):
            scaled_inputs[k] = PHI ** (bucket_vals[k] * v).astype(np.float64) * x_float[0]
        t_scale = time.perf_counter() - t0
        
        # Binary matmul per bucket
        t0 = time.perf_counter()
        result = np.zeros((1, out_f), dtype=np.float32)
        for k in range(K):
            rows = np.where(bucket_idx == k)[0]
            if len(rows) == 0:
                continue
            # sign[rows, :] @ scaled_inputs[k, :]
            # This is binary (sign) × float, = add/subtract
            result[0, rows] = signs[rows] @ scaled_inputs[k].astype(np.float32)
        t_binary = time.perf_counter() - t0
        
        corr = np.corrcoef(full.flatten(), result.flatten())[0, 1]
        rel_err = np.linalg.norm(full - result) / np.linalg.norm(full)
        
        # Top-k
        tk_full = set(np.argsort(np.abs(full[0]))[-100:])
        tk_res = set(np.argsort(np.abs(result[0]))[-100:])
        topk = len(tk_full & tk_res) / 100
        
        print(f"      K={K:>3d}: corr={corr:.6f}, rel_err={rel_err:.4f}, "
              f"top100={topk:.0%}, scale={t_scale*1000:.2f}ms, "
              f"binary={t_binary*1000:.2f}ms")
    
    # BLAS baseline
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = x_float @ W_dec.T
        times.append(time.perf_counter() - t0)
    t_blas = np.median(times) * 1000
    print(f"\n    BLAS baseline: {t_blas:.3f}ms")
    
    return corr_r1


# ============================================================================
# How many distinct u values really exist?
# ============================================================================

def analyze_u_distribution(W, name=""):
    """
    The row scale u[j] — how many distinct values?
    If u is itself structured (e.g., repeats per head), K is small.
    """
    u, v, sigma, _ = extract_rank1(W)
    out_f = len(u)
    
    print(f"\n  Row scale u distribution ({name}):")
    print(f"    {out_f} values, range [{u.min():.3f}, {u.max():.3f}]")
    print(f"    Mean={np.mean(u):.3f}, Std={np.std(u):.3f}")
    
    # Histogram
    hist, edges = np.histogram(u, bins=20)
    print(f"    Distribution (20 bins):")
    max_count = max(hist)
    for i, count in enumerate(hist):
        bar = "█" * int(count / max_count * 40)
        print(f"      [{edges[i]:>6.2f}, {edges[i+1]:>6.2f}): {count:>5d} {bar}")
    
    # Is u structured by heads? (28 heads × 128 head_dim)
    if out_f == 3584:  # q_proj
        u_heads = u.reshape(28, 128)
        head_means = np.mean(u_heads, axis=1)
        head_stds = np.std(u_heads, axis=1)
        
        print(f"\n    Per-head u structure (28 heads × 128):")
        print(f"      Cross-head std of means: {np.std(head_means):.4f}")
        print(f"      Mean within-head std: {np.mean(head_stds):.4f}")
        print(f"      Ratio: {np.std(head_means)/np.mean(head_stds):.3f}")
        
        for h in [0, 7, 14, 21, 27]:
            print(f"      Head {h:2d}: mean={head_means[h]:.3f}, std={head_stds[h]:.3f}")
    
    # Column scale v
    print(f"\n  Column scale v distribution ({name}):")
    print(f"    {len(v)} values, range [{v.min():.4f}, {v.max():.4f}]")
    print(f"    Mean={np.mean(v):.4f}, Std={np.std(v):.4f}")
    
    return u, v


# ============================================================================
# Sign Matrix: The Real Information
# ============================================================================

def analyze_sign_information(W, name=""):
    """
    The sign matrix is where the information lives.
    How compressible is it? 
    """
    sgn = W.signs.astype(np.float32)
    out_f, in_f = sgn.shape
    
    # Total bits: out_f × in_f (binary)
    total_bits = out_f * in_f
    
    print(f"\n  Sign matrix information ({name}):")
    print(f"    Shape: {out_f}×{in_f} = {total_bits:,} bits raw")
    
    # SVD of sign matrix
    U, sigma, Vt = np.linalg.svd(sgn, full_matrices=False)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    
    r50 = np.searchsorted(energy, 0.50) + 1
    r90 = np.searchsorted(energy, 0.90) + 1
    r95 = np.searchsorted(energy, 0.95) + 1
    r99 = np.searchsorted(energy, 0.99) + 1
    
    print(f"    Sign SVD: rank50={r50}, rank90={r90}, "
          f"rank95={r95}, rank99={r99}/{min(out_f,in_f)}")
    
    # Entropy of sign matrix
    # Signs are balanced (~50% each), so per-element entropy ≈ 1 bit
    pos_frac = np.mean(sgn > 0)
    per_elem_entropy = -(pos_frac * np.log2(pos_frac + 1e-10) + 
                         (1-pos_frac) * np.log2(1-pos_frac + 1e-10))
    
    print(f"    Per-element entropy: {per_elem_entropy:.3f} bits (max=1.0)")
    print(f"    Total naive entropy: {total_bits * per_elem_entropy:,.0f} bits "
          f"= {total_bits * per_elem_entropy / 8 / 1024:.1f} KB")
    
    # But correlation between columns?
    # If sign columns are correlated, effective entropy is less
    sample_cols = np.random.choice(in_f, min(100, in_f), replace=False)
    col_corrs = []
    for i in range(len(sample_cols)):
        for j in range(i+1, min(len(sample_cols), i+10)):
            c = np.corrcoef(sgn[:, sample_cols[i]], sgn[:, sample_cols[j]])[0, 1]
            col_corrs.append(abs(c))
    
    print(f"    Mean |column correlation|: {np.mean(col_corrs):.4f}")
    print(f"    Max |column correlation|: {np.max(col_corrs):.4f}")
    
    return r90


# ============================================================================
# The Full Picture: Rank-1 Magnitude + Binary Sign = Complete Factoring
# ============================================================================

def full_picture(W, x_float, name=""):
    """
    Final synthesis: how accurate is rank-1 level + exact sign?
    And what would the hardware look like?
    """
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    full = x_float @ W_dec.T
    
    u, v, sigma, _ = extract_rank1(W)
    sgn = W.signs.astype(np.float32)
    
    # Reconstruction: sign × φ^(u⊗v)
    mag = PHI ** np.outer(u, v).astype(np.float64)
    W_recon = sgn.astype(np.float64) * mag
    result = (x_float.astype(np.float64) @ W_recon.T).astype(np.float32)
    
    corr = np.corrcoef(full.flatten(), result.flatten())[0, 1]
    
    # Also: rank-2 level
    lvl = levels(W).astype(np.float32)
    U_full, sig, Vt_full = np.linalg.svd(lvl, full_matrices=False)
    lvl_r2 = (U_full[:, :2] * sig[:2]) @ Vt_full[:2, :]
    mag_r2 = PHI ** lvl_r2.astype(np.float64)
    W_r2 = sgn.astype(np.float64) * mag_r2
    result_r2 = (x_float.astype(np.float64) @ W_r2.T).astype(np.float32)
    corr_r2 = np.corrcoef(full.flatten(), result_r2.flatten())[0, 1]
    
    print(f"\n  {'='*60}")
    print(f"  COMPLETE FACTORING ({name})")
    print(f"  {'='*60}")
    print(f"    W[j,i] = sign[j,i] × φ^(u[j]×v[i])")
    print(f"")
    print(f"    Stored: sign matrix ({out_f}×{in_f} bits)")
    print(f"           + u vector ({out_f} floats)")  
    print(f"           + v vector ({in_f} floats)")
    print(f"    Storage: {out_f*in_f/8/1024:.0f} KB (signs) + "
          f"{(out_f+in_f)*4/1024:.0f} KB (u,v)")
    print(f"    vs original: {out_f*in_f*2/1024:.0f} KB (int16 exponents + int8 signs)")
    print(f"")
    print(f"    Rank-1 level accuracy: corr={corr:.6f}")
    print(f"    Rank-2 level accuracy: corr={corr_r2:.6f}")
    print(f"")
    print(f"    Computation: for each of K row groups,")
    print(f"      x_scaled = φ^(u_k × v) ⊙ x   — O(N) per group")
    print(f"      y_rows = sign_rows @ x_scaled  — BINARY: add/sub only")
    print(f"    Total: O(K×N + N×out_f) binary ops")
    

def run():
    print("=" * 70)
    print("  RANK-1 LEVEL MATMUL")
    print("  The hidden dimension: level[j,i] = u[j] × v[i]")
    print("  Like π in the Gaussian — the structure was always there")
    print("=" * 70)

    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    for wname in ['q_proj', 'gate_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        print(f"\n{'='*70}\n  {wname} ({W.shape[0]}×{W.shape[1]})\n{'='*70}")
        
        analyze_u_distribution(W, wname)
        analyze_sign_information(W, wname)
        
        x = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        rank1_matmul(W, x, [1, 2, 4, 8, 16, 32], wname)
        full_picture(W, x, wname)
        
        W.clear_cache()


if __name__ == '__main__':
    run()
