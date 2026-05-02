#!/usr/bin/env python3
"""
Level-Grouped φ-Matmul: Reorder computation by φ-level.

The key insight: meaning is relational, shapes are arbitrary. We choose
the coordinate system that makes computation sequential.

Instead of:
    y[j] = Σ_i  sign_w[j,i] * LUT[exp_w[j,i] + exp_x[i]]   (random access)

We regroup by weight level:
    y[j] = Σ_level  φ^(level/K) * Σ_{i∈group(level)}  sign_w[j,i] * x_decoded[i]

The inner sum is a sign-weighted gather of sequential input values.
The outer sum has ~30-50 terms (one per distinct level).
No LUT random access. Just sequential reads + XOR + accumulate.

This IS DC 138's φ-Level Matmul, but now we understand why:
we're choosing the ordering that makes our binary operations sequential.
"""

import os
import sys
import time
import numpy as np
import numba as nb
from numba import njit, prange, typed, types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID, LOG_PHI
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid, get_lut


# ============================================================================
# LEVEL-GROUPED WEIGHT STRUCTURE
# ============================================================================

def quantize_levels(exponents, bin_width=1):
    """
    Quantize exponents to discrete levels.
    
    bin_width=1: every distinct exponent is its own level (finest grain)
    bin_width=16: group into bins of 16 (coarser, fewer levels)
    """
    return (exponents // bin_width) * bin_width


def build_level_groups(W: PhiEncoded, bin_width=16):
    """
    Pre-process a weight matrix into level-grouped structure.
    
    For each row j:
      - Group columns by quantized weight exponent level
      - Store: level -> (column_indices, signs_at_those_columns)
      - Pre-compute: φ^(level/128) for each level
    
    Returns:
        groups: list of row_groups, where each row_group is a list of
                (level_magnitude, col_indices, col_signs)
        n_levels_per_row: array of how many levels each row uses
    """
    out_f, in_f = W.shape
    w_exps = W.exponents.astype(np.int32)
    w_signs = W.signs
    
    all_row_groups = []
    n_levels = np.zeros(out_f, dtype=np.int32)
    
    for j in range(out_f):
        row_exps = w_exps[j]
        row_signs = w_signs[j]
        
        # Quantize to levels
        levels = quantize_levels(row_exps, bin_width)
        unique_levels = np.unique(levels)
        
        row_group = []
        for lev in unique_levels:
            mask = levels == lev
            indices = np.where(mask)[0].astype(np.int32)
            signs = row_signs[mask]
            
            # Pre-compute the magnitude for this level
            # Use the center of the bin as representative
            magnitude = float(PHI ** ((lev + bin_width / 2) / PHI_GRID))
            
            row_group.append((magnitude, indices, signs))
        
        # Sort by magnitude descending (process largest contributions first)
        row_group.sort(key=lambda x: -x[0])
        all_row_groups.append(row_group)
        n_levels[j] = len(unique_levels)
    
    return all_row_groups, n_levels


def build_level_arrays(W: PhiEncoded, bin_width=16):
    """
    Build flattened arrays suitable for Numba from level-grouped structure.
    
    Returns:
        row_starts: int32 (out_f+1,) — CSR-style row offsets into levels
        level_mags: float32 (total_levels,) — magnitude per level
        level_starts: int32 (total_levels+1,) — offsets into indices/signs
        col_indices: int32 (total_elements,) — column indices
        col_signs: int8 (total_elements,) — signs at those columns
    """
    groups, n_levels = build_level_groups(W, bin_width)
    out_f = len(groups)
    
    # Count totals
    total_levels = sum(len(g) for g in groups)
    total_elements = sum(sum(len(indices) for _, indices, _ in g) for g in groups)
    
    row_starts = np.zeros(out_f + 1, dtype=np.int32)
    level_mags = np.zeros(total_levels, dtype=np.float32)
    level_starts = np.zeros(total_levels + 1, dtype=np.int32)
    col_indices = np.zeros(total_elements, dtype=np.int32)
    col_signs = np.zeros(total_elements, dtype=np.int8)
    
    lev_idx = 0
    elem_idx = 0
    
    for j, row_group in enumerate(groups):
        row_starts[j] = lev_idx
        
        for mag, indices, signs in row_group:
            level_mags[lev_idx] = mag
            level_starts[lev_idx] = elem_idx
            
            n = len(indices)
            col_indices[elem_idx:elem_idx + n] = indices
            col_signs[elem_idx:elem_idx + n] = signs
            
            elem_idx += n
            lev_idx += 1
    
    row_starts[out_f] = lev_idx
    level_starts[total_levels] = elem_idx
    
    return row_starts, level_mags, level_starts, col_indices, col_signs


# ============================================================================
# NUMBA LEVEL-GROUPED MATMUL
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def level_matmul_numba(row_starts, level_mags, level_starts,
                       col_indices, col_signs, x_float):
    """
    Level-grouped φ-matmul via Numba.
    
    For each output row j:
        y[j] = Σ_level  magnitude[level] × Σ_{i∈level}  sign[i] × x[i]
    
    The inner sum is sequential gather + sign multiply + accumulate.
    The outer sum has ~30-50 terms (one per level).
    No random LUT access. All sequential.
    
    Args:
        row_starts: CSR row offsets into levels
        level_mags: magnitude per level group
        level_starts: CSR offsets into col_indices/col_signs
        col_indices: column indices per level
        col_signs: signs per column
        x_float: float32 input (batch, in_f) — already decoded
    
    Returns:
        result: float32 (batch, out_f)
    """
    batch = x_float.shape[0]
    out_f = len(row_starts) - 1
    
    result = np.zeros((batch, out_f), dtype=np.float32)
    
    for b in prange(batch):
        for j in range(out_f):
            acc = np.float32(0.0)
            
            # Iterate over levels for this row
            for lev in range(row_starts[j], row_starts[j + 1]):
                mag = level_mags[lev]
                
                # Sign-weighted sum of inputs at this level
                level_acc = np.float32(0.0)
                for k in range(level_starts[lev], level_starts[lev + 1]):
                    ci = col_indices[k]
                    s = col_signs[k]
                    level_acc += np.float32(s) * x_float[b, ci]
                
                acc += mag * level_acc
            
            result[b, j] = acc
    
    return result


@njit(parallel=True, cache=True, fastmath=True)
def level_matmul_numba_phi(row_starts, level_mags, level_starts,
                           col_indices, col_signs,
                           x_signs, x_exps, lut, lut_min):
    """
    Level-grouped φ-matmul operating on φ-encoded input.
    
    Same structure but input is (signs, exponents) not float.
    Uses LUT for input decoding but the access pattern is now
    determined by the LEVEL GROUPS (sorted), not random.
    
    For each level group, the input indices are clustered,
    so LUT accesses are more cache-friendly.
    """
    batch = x_signs.shape[0]
    out_f = len(row_starts) - 1
    lut_max = len(lut) - 1
    
    result = np.zeros((batch, out_f), dtype=np.float32)
    
    for b in prange(batch):
        for j in range(out_f):
            acc = np.float32(0.0)
            
            for lev in range(row_starts[j], row_starts[j + 1]):
                mag = level_mags[lev]
                level_acc = np.float32(0.0)
                
                for k in range(level_starts[lev], level_starts[lev + 1]):
                    ci = col_indices[k]
                    s = col_signs[k] * x_signs[b, ci]
                    
                    # Decode input magnitude via LUT
                    e = np.int32(x_exps[b, ci])
                    idx = e - lut_min
                    if idx < 0:
                        idx = 0
                    elif idx > lut_max:
                        idx = lut_max
                    
                    level_acc += np.float32(s) * lut[idx]
                
                acc += mag * level_acc
            
            result[b, j] = acc
    
    return result


# ============================================================================
# DIMENSION-REORDERED MATMUL
# ============================================================================

def build_dimension_permutation(W: PhiEncoded):
    """
    Find a permutation of input dimensions that makes exponent access
    more sequential across all rows.
    
    Strategy: sort columns by their median exponent. This way, adjacent
    columns tend to have similar exponents, making the gather pattern
    within each level group more sequential.
    """
    median_exps = np.median(W.exponents.astype(np.float64), axis=0)
    perm = np.argsort(median_exps).astype(np.int32)
    return perm


def apply_permutation(W: PhiEncoded, perm):
    """Permute the columns (input dimensions) of a weight matrix."""
    new_signs = W.signs[:, perm].copy()
    new_exps = W.exponents[:, perm].copy()
    return PhiEncoded(signs=new_signs, exponents=new_exps)


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark():
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
    
    print("=" * 70)
    print("  Level-Grouped φ-Matmul Benchmark")
    print("  Reorder by shape: sequential access instead of random LUT")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    
    # Test on q_proj (3584×3584) and gate_proj (18944×3584)
    for name, shape_label in [('q_proj', '3584×3584'), ('gate_proj', '18944×3584')]:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        
        print(f"\n{'─'*70}")
        print(f"  {name} ({shape_label})")
        print(f"{'─'*70}")
        
        np.random.seed(42)
        x_float = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        x_enc = PhiEncoded.encode(x_float)
        
        # --- BLAS baseline ---
        _ = W.decode_cached()  # warm cache
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result_blas = phi_matmul_hybrid(W, x_float)
            times.append(time.perf_counter() - t0)
        t_blas = np.median(times)
        print(f"\n  BLAS (cached decode):     {t_blas*1000:8.2f}ms")
        
        # --- Level analysis ---
        for bin_width in [1, 8, 16, 32, 64]:
            groups, n_levels = build_level_groups(W, bin_width=bin_width)
            avg_levels = np.mean(n_levels)
            min_levels = np.min(n_levels)
            max_levels = np.max(n_levels)
            print(f"  bin_width={bin_width:3d}: "
                  f"avg {avg_levels:.0f} levels/row "
                  f"(range {min_levels}-{max_levels})")
        
        # --- Build level arrays for best bin_width ---
        for bw in [16, 32, 64]:
            print(f"\n  --- Level-grouped (bin_width={bw}) ---")
            t0 = time.perf_counter()
            rs, lm, ls, ci, cs = build_level_arrays(W, bin_width=bw)
            t_build = time.perf_counter() - t0
            print(f"  Build time: {t_build*1000:.0f}ms")
            
            n_levels_total = len(lm)
            n_elements = len(ci)
            print(f"  Total levels: {n_levels_total:,}")
            print(f"  Total elements: {n_elements:,} (of {W.shape[0]*W.shape[1]:,})")
            
            # Warm JIT
            _ = level_matmul_numba(rs, lm, ls, ci, cs, x_float)
            
            # Benchmark with float input
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                result_level = level_matmul_numba(rs, lm, ls, ci, cs, x_float)
                times.append(time.perf_counter() - t0)
            t_level = np.median(times)
            
            corr = np.corrcoef(result_blas.flatten(), result_level.flatten())[0, 1]
            
            # Top-k agreement
            top_blas = set(np.argsort(np.abs(result_blas[0]))[-100:])
            top_level = set(np.argsort(np.abs(result_level[0]))[-100:])
            topk = len(top_blas & top_level) / 100
            
            print(f"  Float input:  {t_level*1000:8.2f}ms "
                  f"({t_level/t_blas:6.1f}× vs BLAS)  "
                  f"corr={corr:.6f}  top100={topk:.0%}")
            
            # Benchmark with φ-encoded input
            lut = (PHI ** (np.arange(-25000, 5001, dtype=np.int32) / PHI_GRID)).astype(np.float32)
            
            _ = level_matmul_numba_phi(rs, lm, ls, ci, cs,
                                        np.ascontiguousarray(x_enc.signs),
                                        np.ascontiguousarray(x_enc.exponents),
                                        lut, np.int32(-25000))
            
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                result_phi = level_matmul_numba_phi(
                    rs, lm, ls, ci, cs,
                    np.ascontiguousarray(x_enc.signs),
                    np.ascontiguousarray(x_enc.exponents),
                    lut, np.int32(-25000))
                times.append(time.perf_counter() - t0)
            t_phi = np.median(times)
            
            corr_phi = np.corrcoef(result_blas.flatten(), result_phi.flatten())[0, 1]
            print(f"  φ-int input:  {t_phi*1000:8.2f}ms "
                  f"({t_phi/t_blas:6.1f}× vs BLAS)  "
                  f"corr={corr_phi:.6f}")
        
        # --- Dimension-reordered variant ---
        print(f"\n  --- Dimension-reordered + Level-grouped (bin=32) ---")
        perm = build_dimension_permutation(W)
        W_reord = apply_permutation(W, perm)
        x_reord = x_float[:, perm]
        
        rs2, lm2, ls2, ci2, cs2 = build_level_arrays(W_reord, bin_width=32)
        _ = level_matmul_numba(rs2, lm2, ls2, ci2, cs2, x_reord)
        
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result_reord = level_matmul_numba(rs2, lm2, ls2, ci2, cs2, x_reord)
            times.append(time.perf_counter() - t0)
        t_reord = np.median(times)
        
        corr_reord = np.corrcoef(result_blas.flatten(), result_reord.flatten())[0, 1]
        print(f"  Reordered:    {t_reord*1000:8.2f}ms "
              f"({t_reord/t_blas:6.1f}× vs BLAS)  "
              f"corr={corr_reord:.6f}")
        
        W.clear_cache()
    
    # --- Final summary ---
    print(f"\n{'='*70}")
    print(f"  KEY INSIGHT")
    print(f"{'='*70}")
    print(f"  Since meaning is relational and shapes are arbitrary,")
    print(f"  we choose the coordinate system that makes computation")
    print(f"  sequential. Level-grouping replaces random LUT access")
    print(f"  with ~30-50 sequential gather-multiply-sum operations.")
    print(f"  The binary structure (signs) is preserved exactly.")
    print(f"  The magnitude (levels) becomes the outer loop.")


if __name__ == '__main__':
    run_benchmark()
