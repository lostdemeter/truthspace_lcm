#!/usr/bin/env python3
"""
Hidden Dimension v2 — Finding the axis of symmetry.

The Gaussian has π because e^(-x²) has RADIAL symmetry in 2D.
The weight matrix has a hidden dimension — not heads (tested, failed).

Two hypotheses:
  A) The φ-level IS the hidden dimension. At each level, the binary
     mask M_level = (level_map == L) might have low-rank structure.
     If so: W = Σ_L v_L × M_L where each M_L is low-rank.
     
  B) The layer IS the hidden dimension. Across layers 0-27, the
     tetromino assignment at position (j,i) might be consistent.
     "The structure doesn't change shape."

  C) Level × Sign: maybe the structure is in how SIGN varies at
     fixed level. Within each level, sign could be low-rank.
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
# A) Per-Level Mask Rank
# ============================================================================

def test_level_mask_rank(W, name=""):
    """
    For each φ-level L, the binary mask M_L = (level_map == L) shows
    which positions are at that magnitude. Is M_L low-rank?
    
    If M_L has rank r << min(out, in), then:
      M_L @ x can be computed in O(r × (out + in)) instead of O(out × in)
      W = Σ_L v_L × M_L gives total O(n_levels × r × (out + in))
    """
    lvl = levels(W)
    out_f, in_f = lvl.shape
    signs = W.signs
    
    unique_levels = np.unique(lvl)
    level_counts = {int(l): np.sum(lvl == l) for l in unique_levels}
    total = out_f * in_f
    
    print(f"\n  Per-level binary mask rank ({name}, {out_f}×{in_f}):")
    print(f"    Unique levels: {len(unique_levels)}")
    
    # Sort by frequency
    sorted_levels = sorted(level_counts.items(), key=lambda x: -x[1])
    
    for L, count in sorted_levels[:10]:
        frac = count / total
        mask = (lvl == L).astype(np.float32)  # (out_f, in_f)
        
        # SVD — only need top singular values
        # For large matrices, use randomized SVD
        k = min(64, min(out_f, in_f))
        U, sigma, Vt = np.linalg.svd(mask, full_matrices=False)
        sigma = sigma[:k]
        
        energy = np.cumsum(sigma**2) / np.sum(sigma**2 + 1e-10)
        r50 = np.searchsorted(energy, 0.50) + 1
        r90 = np.searchsorted(energy, 0.90) + 1
        r95 = np.searchsorted(energy, 0.95) + 1
        
        # Effective rank
        eff_rank = int(np.sum(sigma > sigma[0] * 0.01))
        
        print(f"    Level {L:>3d} ({frac:>5.1%}): rank50={r50}, rank90={r90}, "
              f"rank95={r95}, eff_rank={eff_rank}")
    
    # Now: sign pattern WITHIN each level
    print(f"\n  Sign pattern rank at fixed level ({name}):")
    for L, count in sorted_levels[:6]:
        mask = (lvl == L)
        # Sign at this level: +1 or -1 where mask is True, 0 elsewhere
        sign_at_level = np.where(mask, signs, 0).astype(np.float32)
        
        _, sigma, _ = np.linalg.svd(sign_at_level, full_matrices=False)
        sigma = sigma[:min(64, len(sigma))]
        energy = np.cumsum(sigma**2) / np.sum(sigma**2 + 1e-10)
        r50 = np.searchsorted(energy, 0.50) + 1
        r90 = np.searchsorted(energy, 0.90) + 1
        
        print(f"    Level {L:>3d}: sign_rank50={r50}, sign_rank90={r90}")
    
    return sorted_levels


# ============================================================================
# B) Cross-Layer Consistency
# ============================================================================

def test_cross_layer_consistency(wname):
    """
    "The structure doesn't change shape."
    Load the same weight type from multiple layers and compare.
    """
    print(f"\n  Cross-layer consistency ({wname}):")
    
    layers_to_load = [0, 1, 7, 14, 27]
    layer_data = {}
    
    for L_idx in layers_to_load:
        layer_dir = os.path.join(MODEL_DIR, f'layer_{L_idx:02d}')
        fpath = os.path.join(layer_dir, f'{wname}.npz')
        if not os.path.exists(fpath):
            print(f"    Layer {L_idx}: not found, skipping")
            continue
        W = PhiEncoded.load(fpath)
        layer_data[L_idx] = {
            'levels': levels(W),
            'signs': W.signs.copy(),
            'shape': W.shape
        }
        W.clear_cache()
    
    if len(layer_data) < 2:
        print(f"    Not enough layers found.")
        return
    
    available = sorted(layer_data.keys())
    print(f"    Loaded layers: {available}")
    
    # Level agreement between layers
    print(f"\n    Level agreement (fraction of positions with same φ-level):")
    for i, L1 in enumerate(available):
        for L2 in available[i+1:]:
            agree = np.mean(layer_data[L1]['levels'] == layer_data[L2]['levels'])
            print(f"      Layer {L1} vs {L2}: {agree:.1%}")
    
    # Sign agreement
    print(f"\n    Sign agreement:")
    for i, L1 in enumerate(available):
        for L2 in available[i+1:]:
            agree = np.mean(layer_data[L1]['signs'] == layer_data[L2]['signs'])
            print(f"      Layer {L1} vs {L2}: {agree:.1%}")
    
    # Tetromino ID agreement (level × sign)
    print(f"\n    Full tetromino agreement (level + sign):")
    for i, L1 in enumerate(available):
        for L2 in available[i+1:]:
            lvl_agree = layer_data[L1]['levels'] == layer_data[L2]['levels']
            sgn_agree = layer_data[L1]['signs'] == layer_data[L2]['signs']
            full_agree = lvl_agree & sgn_agree
            print(f"      Layer {L1} vs {L2}: {np.mean(full_agree):.1%}")
    
    # Level distribution shift
    print(f"\n    Level distribution per layer:")
    for L_idx in available:
        lvl = layer_data[L_idx]['levels']
        print(f"      Layer {L_idx:2d}: mean={np.mean(lvl):.2f}, "
              f"std={np.std(lvl):.2f}, "
              f"mode={np.bincount(lvl.flatten() - lvl.min()).argmax() + lvl.min()}")
    
    # Correlation of the ACTUAL decoded weight matrices
    print(f"\n    Weight correlation (decoded float):")
    decoded = {}
    for L_idx in available:
        layer_dir = os.path.join(MODEL_DIR, f'layer_{L_idx:02d}')
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        decoded[L_idx] = W.decode_cached().flatten()
        W.clear_cache()
    
    for i, L1 in enumerate(available):
        for L2 in available[i+1:]:
            corr = np.corrcoef(decoded[L1], decoded[L2])[0, 1]
            print(f"      Layer {L1} vs {L2}: corr={corr:.6f}")


# ============================================================================
# C) The φ-Level as Continuous Dimension
# ============================================================================

def test_level_as_dimension(W, name=""):
    """
    Instead of binary masks per level, treat the level as a CONTINUOUS
    dimension. The weight matrix becomes:
    
    W[j, i] = sign[j,i] × φ^(level[j,i])
    
    We know sign and level are both (out, in) matrices.
    The level matrix: what's ITS structure?
    """
    lvl = levels(W).astype(np.float32)
    sgn = W.signs.astype(np.float32)
    out_f, in_f = lvl.shape
    
    print(f"\n  Level matrix structure ({name}):")
    
    # SVD of level matrix
    U, sigma, Vt = np.linalg.svd(lvl, full_matrices=False)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    r50 = np.searchsorted(energy, 0.50) + 1
    r90 = np.searchsorted(energy, 0.90) + 1
    r95 = np.searchsorted(energy, 0.95) + 1
    
    print(f"    Level matrix rank50={r50}, rank90={r90}, rank95={r95}/{min(out_f,in_f)}")
    print(f"    Top-5 singular values: {sigma[:5]}")
    
    # SVD of sign matrix
    U, sigma_s, Vt = np.linalg.svd(sgn, full_matrices=False)
    energy_s = np.cumsum(sigma_s**2) / np.sum(sigma_s**2)
    r50s = np.searchsorted(energy_s, 0.50) + 1
    r90s = np.searchsorted(energy_s, 0.90) + 1
    
    print(f"    Sign matrix rank50={r50s}, rank90={r90s}/{min(out_f,in_f)}")
    
    # Key question: is level low-rank even though the full weight isn't?
    # If level = U_L @ S_L @ V_L^T (low rank), then:
    # W[j,i] = sign[j,i] × φ^(Σ_r u_r[j] × s_r × v_r[i])
    # This isn't linearly decomposable, but the EXPONENT is.
    
    # What if we factor: W = sign ⊙ exp(level × ln(φ))
    # log(|W|) = level × ln(φ)
    # If level is low-rank, then log(|W|) is low-rank!
    
    if r90 < min(out_f, in_f) * 0.5:
        print(f"    → Level matrix IS low-rank! The exponent structure has hidden symmetry")
        print(f"      log(|W|) = (low-rank matrix) × ln(φ)")
    else:
        print(f"    → Level matrix is high-rank ({r90}/{min(out_f,in_f)})")
    
    # Decomposition test: rank-K level approximation
    for K in [1, 5, 10, 50, 100]:
        if K > min(out_f, in_f):
            break
        # Reconstruct level with rank K
        U_k = U[:, :K]  # actually need to recompute
        # Recompute for sign matrix
        pass
    
    return r90


# ============================================================================
# D) Combined: level_matrix as the hidden structure
# ============================================================================

def test_low_rank_level_matmul(W, x_float, name=""):
    """
    If the level matrix is low-rank, we can approximate:
    
    level ≈ U_K @ diag(s_K) @ V_K^T   (rank K)
    W[j,i] ≈ sign[j,i] × φ^(level_approx[j,i])
    
    This preserves the SIGN exactly and approximates the MAGNITUDE
    via a low-rank exponent field.
    """
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    full = x_float @ W_dec.T
    
    lvl = levels(W).astype(np.float32)
    sgn = W.signs.astype(np.float32)
    
    # Full SVD of level matrix
    U, sigma, Vt = np.linalg.svd(lvl, full_matrices=False)
    
    print(f"\n  Low-rank level matmul ({name}):")
    
    for K in [1, 2, 5, 10, 25, 50, 100]:
        if K > min(out_f, in_f):
            break
        
        # Rank-K level approximation
        lvl_approx = (U[:, :K] * sigma[:K]) @ Vt[:K, :]
        mag_approx = PHI ** lvl_approx.astype(np.float64)
        W_approx = sgn.astype(np.float64) * mag_approx
        
        result = (x_float.astype(np.float64) @ W_approx.T).astype(np.float32)
        corr = np.corrcoef(full.flatten(), result.flatten())[0, 1]
        
        print(f"    Rank-{K:>3d} level: corr={corr:.6f}")
    
    # Compare: rank-K SVD of the WEIGHT matrix itself
    print(f"  vs rank-K SVD of full weight matrix:")
    U_w, sigma_w, Vt_w = np.linalg.svd(W_dec, full_matrices=False)
    for K in [1, 2, 5, 10, 25, 50, 100]:
        if K > min(out_f, in_f):
            break
        W_k = (U_w[:, :K] * sigma_w[:K]) @ Vt_w[:K, :]
        result_k = x_float @ W_k.T
        corr_k = np.corrcoef(full.flatten(), result_k.flatten())[0, 1]
        print(f"    Rank-{K:>3d} weight: corr={corr_k:.6f}")


def run():
    print("=" * 70)
    print("  HIDDEN DIMENSION v2 — Finding the axis of symmetry")
    print("  Heads didn't work. What dimension reveals the structure?")
    print("=" * 70)

    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    # q_proj
    W = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    print(f"\n{'='*70}\n  q_proj ({W.shape[0]}×{W.shape[1]})\n{'='*70}")

    test_level_mask_rank(W, "q_proj")
    test_level_as_dimension(W, "q_proj")
    x = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
    test_low_rank_level_matmul(W, x, "q_proj")
    W.clear_cache()

    # Cross-layer
    print(f"\n{'='*70}\n  Cross-Layer Analysis\n{'='*70}")
    test_cross_layer_consistency('q_proj')

    # gate_proj
    W = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    print(f"\n{'='*70}\n  gate_proj ({W.shape[0]}×{W.shape[1]})\n{'='*70}")
    test_level_as_dimension(W, "gate_proj")
    x_g = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
    test_low_rank_level_matmul(W, x_g, "gate_proj")
    W.clear_cache()

    print(f"\n{'='*70}\n  SYNTHESIS\n{'='*70}")


if __name__ == '__main__':
    run()
