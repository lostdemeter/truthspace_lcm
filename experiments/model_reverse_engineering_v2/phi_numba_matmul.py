#!/usr/bin/env python3
"""
Numba-accelerated φ-integer matmul.

The pure φ-integer matmul is faster per-element than float decode + numpy
(46.5ms vs 78ms for q_proj), but Python loop overhead makes the full
layer slower. Numba JIT compilation eliminates this overhead.

Core operation per output element:
    y[j] = Σ_i  sign_w[j,i] * sign_x[i] * LUT[exp_w[j,i] + exp_x[i]]
         = Σ_i  (XOR) * (table_read[ADD])

This is 1 XOR + 1 ADD + 1 table lookup + 1 multiply + 1 accumulate
per element. At C speed with SIMD, this should be very fast.

Benchmark against:
  - NumPy hybrid (decode + BLAS matmul)
  - NumPy pure (Python-loop φ-integer)
  - Numba φ-integer (JIT-compiled)
  - Numba sparse φ-integer (with adaptive LCD mask)
"""

import os
import sys
import time
import numpy as np
import numba as nb
from numba import njit, prange

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID, LOG_PHI
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid, phi_matmul_pure, get_lut

# ============================================================================
# PRE-COMPUTE LUT FOR NUMBA
# ============================================================================

EXP_MIN = -25000
EXP_MAX = 5000
LUT_SIZE = EXP_MAX - EXP_MIN + 1

def build_numba_lut():
    """Build the φ-LUT as a plain numpy array for Numba access."""
    exponents = np.arange(EXP_MIN, EXP_MAX + 1, dtype=np.int32)
    return (PHI ** (exponents / PHI_GRID)).astype(np.float32)

# Global LUT — built once
PHI_LUT = build_numba_lut()


# ============================================================================
# NUMBA JIT φ-MATMUL (DENSE)
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def phi_matmul_numba(w_signs, w_exps, x_signs, x_exps, lut, lut_min):
    """
    φ-integer matmul via Numba JIT.
    
    y[b, j] = Σ_i sign_w[j,i] * sign_x[b,i] * lut[exp_w[j,i] + exp_x[b,i]]
    
    All inner operations are integer (XOR=multiply, ADD) + table lookup.
    Numba compiles this to native code with SIMD vectorization.
    
    Args:
        w_signs: int8 (out_f, in_f)
        w_exps: int16 (out_f, in_f)
        x_signs: int8 (batch, in_f)
        x_exps: int16 (batch, in_f)
        lut: float32 (lut_size,) — pre-computed φ^(e/128)
        lut_min: int32 — minimum exponent in LUT
    
    Returns:
        result: float32 (batch, out_f)
    """
    batch = x_signs.shape[0]
    out_f = w_signs.shape[0]
    in_f = w_signs.shape[1]
    lut_max_idx = len(lut) - 1
    
    result = np.zeros((batch, out_f), dtype=np.float32)
    
    for b in prange(batch):
        for j in range(out_f):
            acc = np.float32(0.0)
            for i in range(in_f):
                # Sign XOR (multiply of -1/+1 = XOR equivalent)
                s = w_signs[j, i] * x_signs[b, i]
                
                # Exponent ADD
                e = np.int32(w_exps[j, i]) + np.int32(x_exps[b, i])
                
                # LUT lookup (clamp to range)
                idx = e - lut_min
                if idx < 0:
                    idx = 0
                elif idx > lut_max_idx:
                    idx = lut_max_idx
                
                # Accumulate: sign * magnitude
                acc += np.float32(s) * lut[idx]
            
            result[b, j] = acc
    
    return result


@njit(parallel=True, cache=True, fastmath=True)
def phi_matmul_numba_sparse(w_signs, w_exps, x_signs, x_exps, 
                             mask, lut, lut_min):
    """
    Sparse φ-integer matmul: only compute where mask[j,i] is True.
    
    Same as dense but skips masked-out terms.
    """
    batch = x_signs.shape[0]
    out_f = w_signs.shape[0]
    in_f = w_signs.shape[1]
    lut_max_idx = len(lut) - 1
    
    result = np.zeros((batch, out_f), dtype=np.float32)
    
    for b in prange(batch):
        for j in range(out_f):
            acc = np.float32(0.0)
            for i in range(in_f):
                if not mask[j, i]:
                    continue
                
                s = w_signs[j, i] * x_signs[b, i]
                e = np.int32(w_exps[j, i]) + np.int32(x_exps[b, i])
                
                idx = e - lut_min
                if idx < 0:
                    idx = 0
                elif idx > lut_max_idx:
                    idx = lut_max_idx
                
                acc += np.float32(s) * lut[idx]
            
            result[b, j] = acc
    
    return result


# ============================================================================
# ADAPTIVE MASK BUILDER (same as in phi_integer_inference.py)
# ============================================================================

def build_adaptive_mask(W: PhiEncoded, coverage=0.999):
    """Keep minimum terms per row to cover `coverage` of total magnitude."""
    exps = W.exponents.astype(np.float64)
    out_f, in_f = exps.shape
    magnitudes = PHI ** (exps / PHI_GRID)
    
    mask = np.zeros((out_f, in_f), dtype=np.bool_)
    terms_kept = np.zeros(out_f, dtype=np.int32)
    
    for j in range(out_f):
        row_mag = magnitudes[j]
        total_mag = np.sum(row_mag)
        if total_mag < 1e-20:
            continue
        sorted_idx = np.argsort(-row_mag)
        cumsum = np.cumsum(row_mag[sorted_idx])
        n_keep = np.searchsorted(cumsum, coverage * total_mag) + 1
        n_keep = min(n_keep, in_f)
        mask[j, sorted_idx[:n_keep]] = True
        terms_kept[j] = n_keep
    
    density = np.sum(mask) / mask.size
    return mask, density, np.mean(terms_kept)


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark():
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
    
    print("=" * 70)
    print("  Numba-Accelerated φ-Integer Matmul Benchmark")
    print("=" * 70)
    
    # Load q_proj weights
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    W_q = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    print(f"\n  Weight: q_proj {W_q.shape}")
    print(f"  Storage: {W_q.storage_bytes() / 1e6:.1f} MB")
    
    # Create input
    np.random.seed(42)
    x_float = np.random.randn(1, W_q.shape[1]).astype(np.float32) * 0.02
    x_enc = PhiEncoded.encode(x_float)
    
    # Ensure contiguous arrays for Numba
    w_signs = np.ascontiguousarray(W_q.signs)
    w_exps = np.ascontiguousarray(W_q.exponents)
    x_signs = np.ascontiguousarray(x_enc.signs)
    x_exps = np.ascontiguousarray(x_enc.exponents)
    
    # ---- 1. Hybrid baseline ----
    print(f"\n  --- Hybrid (decode → BLAS) ---")
    # Warm the cache
    _ = phi_matmul_hybrid(W_q, x_float)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result_hybrid = phi_matmul_hybrid(W_q, x_float)
        times.append(time.perf_counter() - t0)
    t_hybrid = np.median(times)
    print(f"  Time: {t_hybrid*1000:.1f}ms (median of 5)")
    
    # ---- 2. NumPy pure baseline ----
    print(f"\n  --- NumPy Pure φ-integer ---")
    t0 = time.perf_counter()
    result_numpy = phi_matmul_pure(W_q, x_enc.signs, x_enc.exponents)
    t_numpy = time.perf_counter() - t0
    corr_np = np.corrcoef(result_hybrid.flatten(), result_numpy.flatten())[0, 1]
    print(f"  Time: {t_numpy*1000:.1f}ms")
    print(f"  Correlation: {corr_np:.6f}")
    
    # ---- 3. Numba dense φ-integer ----
    print(f"\n  --- Numba Dense φ-integer (JIT) ---")
    # Warm up JIT compilation
    print(f"  Compiling (first call)...")
    t0 = time.perf_counter()
    result_numba = phi_matmul_numba(w_signs, w_exps, x_signs, x_exps,
                                     PHI_LUT, np.int32(EXP_MIN))
    t_compile = time.perf_counter() - t0
    print(f"  First call (includes compilation): {t_compile*1000:.1f}ms")
    
    # Benchmark after compilation
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result_numba = phi_matmul_numba(w_signs, w_exps, x_signs, x_exps,
                                         PHI_LUT, np.int32(EXP_MIN))
        times.append(time.perf_counter() - t0)
    t_numba = np.median(times)
    
    corr_nb = np.corrcoef(result_hybrid.flatten(), result_numba.flatten())[0, 1]
    max_diff = np.max(np.abs(result_hybrid - result_numba))
    print(f"  Time: {t_numba*1000:.1f}ms (median of 10)")
    print(f"  Correlation: {corr_nb:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    
    # ---- 4. Numba sparse with adaptive mask ----
    print(f"\n  --- Numba Sparse φ-integer (99% coverage) ---")
    mask_99, density_99, avg_kept = build_adaptive_mask(W_q, coverage=0.99)
    mask_99_c = np.ascontiguousarray(mask_99)
    
    # Warm up
    _ = phi_matmul_numba_sparse(w_signs, w_exps, x_signs, x_exps,
                                 mask_99_c, PHI_LUT, np.int32(EXP_MIN))
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result_sparse = phi_matmul_numba_sparse(w_signs, w_exps, x_signs, x_exps,
                                                 mask_99_c, PHI_LUT, np.int32(EXP_MIN))
        times.append(time.perf_counter() - t0)
    t_sparse = np.median(times)
    
    corr_sp = np.corrcoef(result_hybrid.flatten(), result_sparse.flatten())[0, 1]
    print(f"  Mask density: {density_99:.1%} ({avg_kept:.0f}/{W_q.shape[1]} avg/row)")
    print(f"  Time: {t_sparse*1000:.1f}ms (median of 10)")
    print(f"  Correlation: {corr_sp:.6f}")
    
    # ---- 5. Numba sparse with 95% coverage ----
    print(f"\n  --- Numba Sparse φ-integer (95% coverage) ---")
    mask_95, density_95, avg_kept_95 = build_adaptive_mask(W_q, coverage=0.95)
    mask_95_c = np.ascontiguousarray(mask_95)
    
    _ = phi_matmul_numba_sparse(w_signs, w_exps, x_signs, x_exps,
                                 mask_95_c, PHI_LUT, np.int32(EXP_MIN))
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result_95 = phi_matmul_numba_sparse(w_signs, w_exps, x_signs, x_exps,
                                             mask_95_c, PHI_LUT, np.int32(EXP_MIN))
        times.append(time.perf_counter() - t0)
    t_sparse95 = np.median(times)
    
    corr_95 = np.corrcoef(result_hybrid.flatten(), result_95.flatten())[0, 1]
    print(f"  Mask density: {density_95:.1%} ({avg_kept_95:.0f}/{W_q.shape[1]} avg/row)")
    print(f"  Time: {t_sparse95*1000:.1f}ms (median of 10)")
    print(f"  Correlation: {corr_95:.6f}")
    
    # ---- Now test on larger matrices (gate_proj: 18944×3584) ----
    print(f"\n{'─'*70}")
    print(f"  Large matrix: gate_proj (18944×3584)")
    print(f"{'─'*70}")
    
    W_gate = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    wg_signs = np.ascontiguousarray(W_gate.signs)
    wg_exps = np.ascontiguousarray(W_gate.exponents)
    
    # Hybrid
    _ = phi_matmul_hybrid(W_gate, x_float)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result_gate_hybrid = phi_matmul_hybrid(W_gate, x_float)
        times.append(time.perf_counter() - t0)
    t_gate_hybrid = np.median(times)
    print(f"  Hybrid:         {t_gate_hybrid*1000:.1f}ms")
    
    # Numba dense
    _ = phi_matmul_numba(wg_signs, wg_exps, x_signs, x_exps,
                          PHI_LUT, np.int32(EXP_MIN))
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result_gate_numba = phi_matmul_numba(wg_signs, wg_exps, x_signs, x_exps,
                                              PHI_LUT, np.int32(EXP_MIN))
        times.append(time.perf_counter() - t0)
    t_gate_numba = np.median(times)
    corr_gate = np.corrcoef(result_gate_hybrid.flatten(), result_gate_numba.flatten())[0, 1]
    print(f"  Numba dense:    {t_gate_numba*1000:.1f}ms (corr={corr_gate:.6f})")
    
    # Numba sparse 99%
    mask_gate, density_gate, avg_gate = build_adaptive_mask(W_gate, coverage=0.99)
    mask_gate_c = np.ascontiguousarray(mask_gate)
    
    _ = phi_matmul_numba_sparse(wg_signs, wg_exps, x_signs, x_exps,
                                 mask_gate_c, PHI_LUT, np.int32(EXP_MIN))
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result_gate_sp = phi_matmul_numba_sparse(wg_signs, wg_exps, x_signs, x_exps,
                                                  mask_gate_c, PHI_LUT, np.int32(EXP_MIN))
        times.append(time.perf_counter() - t0)
    t_gate_sparse = np.median(times)
    corr_gate_sp = np.corrcoef(result_gate_hybrid.flatten(), result_gate_sp.flatten())[0, 1]
    print(f"  Numba sparse99: {t_gate_sparse*1000:.1f}ms "
          f"({density_gate:.1%} density, corr={corr_gate_sp:.6f})")
    
    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY: q_proj (3584×3584)")
    print(f"{'='*70}")
    print(f"  {'Method':<30s} {'Time':>8s}  {'vs Hybrid':>10s}  {'Corr':>10s}")
    print(f"  {'─'*62}")
    print(f"  {'Hybrid (decode+BLAS)':<30s} {t_hybrid*1000:>7.1f}ms  {'1.00×':>10s}  {'baseline':>10s}")
    print(f"  {'NumPy pure φ-int':<30s} {t_numpy*1000:>7.1f}ms  {t_numpy/t_hybrid:>9.2f}×  {corr_np:>10.6f}")
    print(f"  {'Numba dense φ-int':<30s} {t_numba*1000:>7.1f}ms  {t_numba/t_hybrid:>9.2f}×  {corr_nb:>10.6f}")
    print(f"  {'Numba sparse 99%':<30s} {t_sparse*1000:>7.1f}ms  {t_sparse/t_hybrid:>9.2f}×  {corr_sp:>10.6f}")
    print(f"  {'Numba sparse 95%':<30s} {t_sparse95*1000:>7.1f}ms  {t_sparse95/t_hybrid:>9.2f}×  {corr_95:>10.6f}")
    
    print(f"\n  gate_proj (18944×3584):")
    print(f"  {'Hybrid':<30s} {t_gate_hybrid*1000:>7.1f}ms")
    print(f"  {'Numba dense':<30s} {t_gate_numba*1000:>7.1f}ms  {t_gate_numba/t_gate_hybrid:>9.2f}×")
    print(f"  {'Numba sparse 99%':<30s} {t_gate_sparse*1000:>7.1f}ms  {t_gate_sparse/t_gate_hybrid:>9.2f}×")
    
    # Estimate full-layer and full-model
    # Per layer: 4 attention matmuls (3584² each) + 3 MLP matmuls (18944×3584 or 3584×18944)
    # Total: 4 × 3584² + 2 × 18944×3584 + 1 × 3584×18944 = ~220M params
    attn_time = 4 * t_numba  # 4 attention projections
    mlp_time = 3 * t_gate_numba  # 3 MLP projections (approx)
    layer_time = attn_time + mlp_time
    model_time = 28 * layer_time
    
    print(f"\n  Estimated full model (28 layers, Numba dense):")
    print(f"    Per layer: {layer_time*1000:.0f}ms")
    print(f"    Full model: {model_time:.1f}s")
    print(f"    Tokens/sec: {1/model_time:.2f} (single-token decode)")


if __name__ == '__main__':
    run_benchmark()
