#!/usr/bin/env python3
"""
Simultaneous Mode-Sum Matmul
==============================

"Everything happens at the same time." — No clock, no traversal.
The shape IS the computation, read all at once.

Key hypothesis: each input column has a CONSISTENT tetromino identity
across output rows. If so:

  y[j] = Σ_i W[j,i] × x[i]
       = Σ_i tet_value[tet_id[j,i]] × x[i]

If tet_id[j,i] ≈ col_mode[i] (same for all j), then:
       ≈ Σ_t tet_value[t] × row_count[j,t] × (Σ_{i: col_mode=t} x[i]) / col_count[t]

Step 1: Compute 67 mode sums: S_t = Σ x[i] where col_mode[i] = t  → O(N)
Step 2: Each output: y[j] = Σ_t w_eff[j,t] × S_t                  → O(67)
Total: O(N + 67 × out_f) instead of O(N × out_f)

But this ONLY works if columns have consistent tetromino identity.
This script tests that hypothesis first, then builds the matmul.
"""

import os
import sys
import time
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')


def phi_to_tet_ids(W: PhiEncoded):
    """Map φ-encoded weights to tetromino IDs (sign × φ-level)."""
    exps = W.exponents.astype(np.int32)
    levels = exps // PHI_GRID
    sign_bit = (W.signs > 0).astype(np.int16)
    return levels.astype(np.int16) * 2 + sign_bit


# ============================================================================
# TEST 1: Column Consistency
# ============================================================================

def test_column_consistency(W: PhiEncoded, name=""):
    """
    Test: does each input column have a consistent tetromino across rows?
    
    If column i has the SAME tet_id in every row, then we can factor
    the matmul into mode sums. If not, the factoring is approximate.
    
    Measures:
    - Mode fraction: what % of rows agree with the column's mode?
    - Entropy: how spread are the tet_ids within each column?
    """
    tet_ids = phi_to_tet_ids(W)
    out_f, in_f = tet_ids.shape
    
    print(f"\n  Column consistency ({name}, {out_f}×{in_f}):")
    
    # For each column, find the mode (most common tet_id) and its fraction
    mode_fracs = np.zeros(in_f)
    col_modes = np.zeros(in_f, dtype=np.int16)
    col_entropies = np.zeros(in_f)
    
    for i in range(in_f):
        col = tet_ids[:, i]
        vals, counts = np.unique(col, return_counts=True)
        max_idx = np.argmax(counts)
        col_modes[i] = vals[max_idx]
        mode_fracs[i] = counts[max_idx] / out_f
        
        # Entropy
        probs = counts / out_f
        col_entropies[i] = -np.sum(probs * np.log2(probs + 1e-10))
    
    print(f"    Mean column mode fraction: {np.mean(mode_fracs):.1%}")
    print(f"    Median column mode fraction: {np.median(mode_fracs):.1%}")
    print(f"    Min column mode fraction: {np.min(mode_fracs):.1%}")
    print(f"    Max column mode fraction: {np.max(mode_fracs):.1%}")
    print(f"    Columns with >50% agreement: {np.mean(mode_fracs > 0.5):.1%}")
    print(f"    Columns with >90% agreement: {np.mean(mode_fracs > 0.9):.1%}")
    print(f"    Mean column entropy: {np.mean(col_entropies):.2f} bits")
    print(f"    (Perfect consistency = 0 bits, uniform over 67 = {np.log2(67):.1f} bits)")
    
    return col_modes, mode_fracs, col_entropies


# ============================================================================
# TEST 2: Row Consistency (transpose view)
# ============================================================================

def test_row_consistency(W: PhiEncoded, name=""):
    """
    Same test but for rows: does each output row have a dominant mode?
    """
    tet_ids = phi_to_tet_ids(W)
    out_f, in_f = tet_ids.shape
    
    print(f"\n  Row consistency ({name}):")
    
    row_mode_fracs = np.zeros(out_f)
    for j in range(out_f):
        row = tet_ids[j]
        vals, counts = np.unique(row, return_counts=True)
        row_mode_fracs[j] = np.max(counts) / in_f
    
    print(f"    Mean row mode fraction: {np.mean(row_mode_fracs):.1%}")
    print(f"    Rows with >50% agreement: {np.mean(row_mode_fracs > 0.5):.1%}")
    
    return row_mode_fracs


# ============================================================================
# TEST 3: Sign-Only Consistency
# ============================================================================

def test_sign_consistency(W: PhiEncoded, name=""):
    """
    Test sign consistency per column.
    
    Even if the LEVEL varies per row, the SIGN might be consistent.
    If sign is consistent per column, we can at least factor the
    sign structure (the AIG binary core) out of the matmul.
    """
    signs = W.signs  # (out_f, in_f), int8: -1 or +1
    out_f, in_f = signs.shape
    
    # For each column, what fraction of rows have the same sign?
    pos_frac = np.mean(signs > 0, axis=0)  # fraction positive per column
    sign_consistency = np.maximum(pos_frac, 1 - pos_frac)  # max(pos, neg)
    
    print(f"\n  Sign consistency ({name}):")
    print(f"    Mean sign consistency: {np.mean(sign_consistency):.1%}")
    print(f"    Columns with >60% same sign: {np.mean(sign_consistency > 0.6):.1%}")
    print(f"    Columns with >80% same sign: {np.mean(sign_consistency > 0.8):.1%}")
    print(f"    Columns with >95% same sign: {np.mean(sign_consistency > 0.95):.1%}")
    
    return sign_consistency


# ============================================================================
# TEST 4: Level-Only Consistency (ignoring sign)
# ============================================================================

def test_level_consistency(W: PhiEncoded, name=""):
    """
    Test level (magnitude) consistency per column.
    
    Even if sign varies, the LEVEL might be consistent.
    If level is consistent per column, each column has a fixed scale,
    and only the sign pattern matters per row.
    """
    exps = W.exponents.astype(np.int32)
    levels = exps // PHI_GRID  # integer φ-level
    out_f, in_f = levels.shape
    
    level_mode_fracs = np.zeros(in_f)
    col_level_modes = np.zeros(in_f, dtype=np.int16)
    
    for i in range(in_f):
        col = levels[:, i]
        vals, counts = np.unique(col, return_counts=True)
        max_idx = np.argmax(counts)
        col_level_modes[i] = vals[max_idx]
        level_mode_fracs[i] = counts[max_idx] / out_f
    
    print(f"\n  Level consistency ({name}):")
    print(f"    Mean level mode fraction: {np.mean(level_mode_fracs):.1%}")
    print(f"    Columns with >50% same level: {np.mean(level_mode_fracs > 0.5):.1%}")
    print(f"    Columns with >90% same level: {np.mean(level_mode_fracs > 0.9):.1%}")
    
    # If level is consistent but sign isn't, we can write:
    # W[j,i] ≈ sign[j,i] × φ^(col_level[i])
    # y[j] = Σ_i sign[j,i] × φ^(col_level[i]) × x[i]
    #       = Σ_i sign[j,i] × (φ^(col_level[i]) × x[i])
    # Let x_scaled[i] = φ^(col_level[i]) × x[i]  (O(N), done once)
    # y[j] = Σ_i sign[j,i] × x_scaled[i]  (binary inner product!)
    
    print(f"    If consistent: matmul reduces to BINARY inner products")
    print(f"    (sign[j,:] · x_scaled = XOR-accumulate)")
    
    return col_level_modes, level_mode_fracs


# ============================================================================
# TEST 5: Factored Matmul (level-consistent approximation)
# ============================================================================

def factored_matmul(W: PhiEncoded, x_float, name=""):
    """
    If levels are column-consistent, factor the matmul:
    
    W[j,i] ≈ sign[j,i] × φ^(col_level[i])
    
    Step 1: x_scaled[i] = φ^(col_level[i]) × x[i]        — O(N)
    Step 2: y[j] = sign_row[j] · x_scaled = Σ sign[j,i] × x_scaled[i] — O(N) per row
    
    Total: O(N × out_f) but with BINARY operations (sign multiply = XOR).
    The multiply step becomes a sign-weighted sum.
    
    On hardware: this is N×out_f XOR-accumulates, no float multiplies.
    """
    W_decoded = W.decode_cached()
    out_f, in_f = W.shape
    
    # Full result (ground truth)
    full_result = x_float @ W_decoded.T
    
    # Column level modes
    exps = W.exponents.astype(np.int32)
    levels = exps // PHI_GRID
    
    col_level_modes = np.zeros(in_f, dtype=np.int16)
    for i in range(in_f):
        vals, counts = np.unique(levels[:, i], return_counts=True)
        col_level_modes[i] = vals[np.argmax(counts)]
    
    # Step 1: Scale input by column level
    col_magnitudes = PHI ** col_level_modes.astype(np.float64)
    x_scaled = x_float[0] * col_magnitudes  # (in_f,)
    
    # Step 2: Binary inner product: y[j] = sign[j,:] · x_scaled
    signs = W.signs.astype(np.float32)  # (out_f, in_f)
    result_factored = signs @ x_scaled.astype(np.float32)  # (out_f,)
    result_factored = result_factored.reshape(1, -1)
    
    corr_factored = np.corrcoef(full_result.flatten(), result_factored.flatten())[0, 1]
    rel_err = np.linalg.norm(full_result - result_factored) / np.linalg.norm(full_result)
    
    # Top-k agreement
    top_full = set(np.argsort(np.abs(full_result[0]))[-100:])
    top_fact = set(np.argsort(np.abs(result_factored[0]))[-100:])
    topk = len(top_full & top_fact) / 100
    
    print(f"\n  Factored matmul ({name}):")
    print(f"    Correlation: {corr_factored:.6f}")
    print(f"    Relative error: {rel_err:.4f}")
    print(f"    Top-100 overlap: {topk:.0%}")
    
    # Step 2b: Use ACTUAL per-element levels (not mode)
    # W[j,i] = sign[j,i] × φ^(level[j,i])
    # We can separate: W = sign ⊙ magnitude
    # where magnitude[j,i] = φ^(level[j,i])
    #
    # This is exact for the quantized weights.
    # The matmul: y = (sign ⊙ magnitude) @ x
    # If we precompute magnitude @ x... no, that gives a scalar per row.
    # 
    # But: y[j] = Σ_i sign[j,i] × magnitude[j,i] × x[i]
    # If magnitude[j,i] = col_mag[i] (column consistent), then:
    # y[j] = Σ_i sign[j,i] × (col_mag[i] × x[i])
    # = sign_row[j] · x_scaled
    
    # Let's also test the exact quantized version
    tet_magnitudes = PHI ** levels.astype(np.float64)  # (out_f, in_f)
    W_quantized = signs * tet_magnitudes.astype(np.float32)
    result_quantized = x_float @ W_quantized.T
    
    corr_quant = np.corrcoef(full_result.flatten(), result_quantized.flatten())[0, 1]
    
    print(f"    Exact quantized matmul corr: {corr_quant:.6f}")
    print(f"    (This is the tetromino representation accuracy)")
    
    # Speed comparison
    # Full BLAS
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = x_float @ W_decoded.T
        times.append(time.perf_counter() - t0)
    t_full = np.median(times)
    
    # Factored: sign @ x_scaled
    x_s = x_scaled.astype(np.float32).copy()
    s = signs.copy()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = s @ x_s
        times.append(time.perf_counter() - t0)
    t_factored = np.median(times)
    
    print(f"\n    Speed:")
    print(f"      Full BLAS: {t_full*1000:.3f}ms")
    print(f"      Factored (sign @ x_scaled): {t_factored*1000:.3f}ms")
    print(f"      Ratio: {t_factored/t_full:.2f}×")
    print(f"      (Both are BLAS — sign matrix is still float32)")
    print(f"      On XOR hardware: sign @ x = binary op, ~100× faster")
    
    return corr_factored, corr_quant


# ============================================================================
# TEST 6: Mode-Sum Matmul (the simultaneous version)
# ============================================================================

def mode_sum_matmul(W: PhiEncoded, x_float, name=""):
    """
    The fully simultaneous mode-sum approach.
    
    Everything happens at once:
    1. Assign each column to its mode tetromino → O(precomputed)
    2. Compute 67 mode sums: S_t = Σ x[i] where col_mode[i] = t → O(N)
    3. For each output row j: y[j] = Σ_t w_eff[j,t] × S_t → O(67)
    
    w_eff[j,t] is precomputed: sum of actual tet_values in row j 
    for all positions with col_mode t.
    
    Total: O(N + 67 × out_f) — one pass over input, one over output.
    """
    W_decoded = W.decode_cached()
    out_f, in_f = W.shape
    
    full_result = x_float @ W_decoded.T
    
    # Tetromino IDs
    tet_ids = phi_to_tet_ids(W)
    unique_tets = np.unique(tet_ids)
    n_tets = len(unique_tets)
    tet_to_idx = {int(t): i for i, t in enumerate(unique_tets)}
    
    # Tet values
    tet_values = np.zeros(n_tets, dtype=np.float64)
    for t in unique_tets:
        level = int(t) // 2
        sign = (int(t) % 2) * 2 - 1
        tet_values[tet_to_idx[int(t)]] = sign * (PHI ** level)
    
    # Column modes
    col_modes = np.zeros(in_f, dtype=np.int16)
    for i in range(in_f):
        vals, counts = np.unique(tet_ids[:, i], return_counts=True)
        col_modes[i] = vals[np.argmax(counts)]
    
    col_mode_idx = np.array([tet_to_idx[int(m)] for m in col_modes])
    
    # Step 1: Compute mode sums — O(N)
    t0 = time.perf_counter()
    S = np.zeros(n_tets, dtype=np.float64)
    for t_idx in range(n_tets):
        mask = col_mode_idx == t_idx
        S[t_idx] = np.sum(x_float[0, mask])
    t_sums = time.perf_counter() - t0
    
    # Step 2: Precompute w_eff[j, t] = sum of tet_values for positions
    # in row j that have col_mode t
    # This is a one-time precomputation
    t0 = time.perf_counter()
    w_eff = np.zeros((out_f, n_tets), dtype=np.float64)
    for j in range(out_f):
        for t_idx in range(n_tets):
            uid = int(unique_tets[t_idx])
            # Positions in row j where col_mode matches this tet
            col_mask = col_mode_idx == t_idx
            # Actual tet values at those positions in this row
            row_tets = tet_ids[j, col_mask]
            # Sum actual values (not all same as col_mode!)
            for rt in np.unique(row_tets):
                rt_count = np.sum(row_tets == rt)
                rt_level = int(rt) // 2
                rt_sign = (int(rt) % 2) * 2 - 1
                w_eff[j, t_idx] += rt_sign * (PHI ** rt_level) * rt_count
    t_precompute = time.perf_counter() - t0
    
    # Step 3: y[j] = w_eff[j,:] @ S — O(67 × out_f)
    t0 = time.perf_counter()
    result_mode = (w_eff @ S).reshape(1, -1).astype(np.float32)
    t_matmul = time.perf_counter() - t0
    
    corr_mode = np.corrcoef(full_result.flatten(), result_mode.flatten())[0, 1]
    rel_err = np.linalg.norm(full_result - result_mode) / np.linalg.norm(full_result)
    
    top_full = set(np.argsort(np.abs(full_result[0]))[-100:])
    top_mode = set(np.argsort(np.abs(result_mode[0]))[-100:])
    topk = len(top_full & top_mode) / 100
    
    print(f"\n  Mode-sum matmul ({name}):")
    print(f"    Mode sums (S_t): {t_sums*1000:.3f}ms")
    print(f"    Precompute w_eff: {t_precompute:.1f}s (one-time)")
    print(f"    Final matmul (w_eff @ S): {t_matmul*1000:.4f}ms")
    print(f"    Correlation: {corr_mode:.6f}")
    print(f"    Relative error: {rel_err:.4f}")
    print(f"    Top-100 overlap: {topk:.0%}")
    
    # Now: what if w_eff is precomputed?
    # Inference = mode_sums + w_eff @ S
    # = O(N) + O(67 × out_f)
    # For q_proj: O(3584) + O(67 × 3584) = O(240k) vs O(12.8M) = 53×
    
    # But WAIT — w_eff includes the actual per-position tet values,
    # not just the mode assumption. So this should be EXACT for the
    # tetromino-quantized weights, not approximate.
    
    # Let's verify: w_eff @ S should give EXACT tetromino matmul if
    # S captured the correct input sums per mode group.
    # The approximation is: we GROUP inputs by col_mode, so inputs
    # with different col_modes but same position get different S values.
    # But each position has exactly one col_mode, so S_t sums exactly
    # the inputs assigned to mode t. And w_eff[j,t] sums the actual
    # (not mode) tet_values for those positions in row j.
    # So: w_eff[j,:] @ S = Σ_t Σ_{i:col_mode=t} actual_tet_value[j,i] × x[i]
    #                     = Σ_i actual_tet_value[j,i] × x[i]
    #                     = exact tetromino matmul!
    
    # The correlation should match the tetromino quantization accuracy.
    # Let's verify:
    tet_magnitudes = PHI ** (W.exponents.astype(np.int32) // PHI_GRID).astype(np.float64)
    W_tet = W.signs.astype(np.float64) * tet_magnitudes
    result_exact_tet = (x_float.astype(np.float64) @ W_tet.T).astype(np.float32)
    corr_exact_tet = np.corrcoef(full_result.flatten(), result_exact_tet.flatten())[0, 1]
    
    # And check if mode-sum matches exact-tet
    corr_mode_vs_tet = np.corrcoef(result_exact_tet.flatten(), result_mode.flatten())[0, 1]
    
    print(f"\n    Exact tetromino matmul corr vs float: {corr_exact_tet:.6f}")
    print(f"    Mode-sum corr vs exact-tet: {corr_mode_vs_tet:.6f}")
    print(f"    (Should be 1.0 if mode grouping is lossless)")
    
    return corr_mode, w_eff, S


# ============================================================================
# MAIN
# ============================================================================

def run_analysis():
    print("=" * 70)
    print("  Simultaneous Mode-Sum Matmul")
    print("  'Everything happens at the same time'")
    print("  No clock. No traversal. The shape IS the computation.")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj', 'gate_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({W.shape[0]}×{W.shape[1]})")
        print(f"{'='*70}")
        
        # Test consistencies
        col_modes, mode_fracs, entropies = test_column_consistency(W, wname)
        test_row_consistency(W, wname)
        test_sign_consistency(W, wname)
        col_levels, level_fracs = test_level_consistency(W, wname)
        
        x_float = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        
        # Factored matmul (level-consistent approximation)
        factored_matmul(W, x_float, wname)
        
        # Mode-sum matmul (simultaneous)
        mode_sum_matmul(W, x_float, wname)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  CONCLUSIONS")
    print(f"{'='*70}")
    print(f"""
  The shape doesn't change. The question is:
  how much of the shape can we read simultaneously?
  
  If columns have consistent tetrominoes:
    → 67 mode sums, O(N + 67×out_f), ~53× fewer ops
    
  If columns have consistent LEVELS (but varying signs):
    → Pre-scale input by level, then BINARY matmul
    → Same ops but all multiplies become sign flips
    → On XOR hardware: massive speedup
    
  If neither is consistent:
    → The shape genuinely varies per (row, col)
    → Need full matmul, but structure still constrains hardware design
""")


if __name__ == '__main__':
    run_analysis()
