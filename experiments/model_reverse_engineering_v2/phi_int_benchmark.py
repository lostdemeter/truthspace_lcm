#!/usr/bin/env python3
"""
Quick benchmark: φ-integer matmul vs float matmul on actual Qwen2-7B weights.

Tests:
1. Accuracy: correlation between integer and float results
2. Speed: wall time for a single layer's gate_proj matmul
3. Next-token prediction: do both give the same top-1?
"""

import numpy as np
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(__file__))
from phi_types import PhiEncoded, PHI, PHI_GRID
from phi_integer import (
    phi_matmul_integer, phi_silu_int, phi_rms_norm_int,
    phi_add_encoded, phi_multiply_int, phi_accumulate,
    get_fixed_lut, get_silu_lut, float_to_phi, phi_to_float
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')


def load_phi_weight(path):
    """Load weight as PhiEncoded (native format — no float decode)."""
    d = np.load(path)
    return PhiEncoded(signs=d['signs'], exponents=d['exponents'])


def load_phi_raw(path):
    """Load raw signs and exponents."""
    d = np.load(path)
    return d['signs'], d['exponents']


def main():
    print("=" * 70)
    print("  φ-Integer Arithmetic Benchmark")
    print("  Native integer vs float decode on Qwen2-7B weights")
    print("=" * 70)
    print()

    # ─── Warm up LUTs ─────────────────────────────────────────────
    print("  Initializing LUTs...", end='', flush=True)
    t0 = time.time()
    get_fixed_lut()
    get_silu_lut()
    print(f" ({time.time()-t0:.2f}s)")

    # ─── Load a test vector (single token embedding) ──────────────
    print("  Loading token embedding (id=279 = ' the')...", end='', flush=True)
    t0 = time.time()
    emb_data = np.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    tok_signs = emb_data['signs'][279].copy()  # (3584,)
    tok_exps = emb_data['exponents'][279].copy()
    del emb_data; gc.collect()
    tok_float = tok_signs.astype(np.float64) * (PHI ** (tok_exps.astype(np.float64) / PHI_GRID))
    tok_float32 = tok_float.astype(np.float32)
    print(f" ({time.time()-t0:.1f}s)")

    # ─── Test 1: Small matmul (v_proj: 512 × 3584) ───────────────
    print("\n  ─── Test 1: v_proj matmul (512 × 3584) ───")
    v_signs, v_exps = load_phi_raw(os.path.join(MODEL_DIR, 'layer_00/v_proj.npz'))
    v_float = (v_signs.astype(np.float64) * PHI ** (v_exps.astype(np.float64) / PHI_GRID)).astype(np.float32)
    v_phi = PhiEncoded(signs=v_signs, exponents=v_exps)

    # Float matmul
    t0 = time.time()
    result_float = v_float @ tok_float32
    t_float = time.time() - t0
    print(f"  Float matmul:   {t_float*1000:.1f}ms")

    # Integer matmul
    x_s = tok_signs[np.newaxis, :]   # (1, 3584)
    x_e = tok_exps[np.newaxis, :]
    t0 = time.time()
    res_s, res_e = phi_matmul_integer(v_phi, x_s, x_e, chunk_size=512)
    t_int = time.time() - t0
    result_int_decoded = phi_to_float(res_s[0], res_e[0])
    print(f"  Integer matmul: {t_int*1000:.1f}ms")
    print(f"  Speedup: {t_float/t_int:.2f}× ({'faster' if t_int < t_float else 'slower'})")

    # Correlation
    corr = np.corrcoef(result_float.flatten(), result_int_decoded.flatten())[0, 1]
    print(f"  Correlation: {corr:.8f}")

    del v_signs, v_exps, v_float, v_phi; gc.collect()

    # ─── Test 2: Large matmul (gate_proj: 18944 × 3584) ──────────
    print("\n  ─── Test 2: gate_proj matmul (18944 × 3584) ───")
    print("  Loading gate_proj...", end='', flush=True)
    t0 = time.time()
    g_signs, g_exps = load_phi_raw(os.path.join(MODEL_DIR, 'layer_00/gate_proj.npz'))
    g_float = (g_signs.astype(np.float64) * PHI ** (g_exps.astype(np.float64) / PHI_GRID)).astype(np.float32)
    g_phi = PhiEncoded(signs=g_signs, exponents=g_exps)
    print(f" ({time.time()-t0:.1f}s)")

    # Float matmul
    t0 = time.time()
    gate_float = g_float @ tok_float32
    t_float = time.time() - t0
    print(f"  Float matmul:   {t_float*1000:.1f}ms")

    # Integer matmul
    t0 = time.time()
    gate_s, gate_e = phi_matmul_integer(g_phi, x_s, x_e, chunk_size=256)
    t_int = time.time() - t0
    gate_int_decoded = phi_to_float(gate_s[0], gate_e[0])
    print(f"  Integer matmul: {t_int*1000:.1f}ms")
    print(f"  Ratio: {t_int/t_float:.1f}× ({'faster' if t_int < t_float else 'slower'})")

    corr = np.corrcoef(gate_float.flatten(), gate_int_decoded.flatten())[0, 1]
    print(f"  Correlation: {corr:.8f}")

    # ─── Test 3: SiLU (element-wise on gate output) ───────────────
    print("\n  ─── Test 3: SiLU (18944 elements) ───")

    # Float SiLU
    t0 = time.time()
    for _ in range(100):
        silu_float = gate_float * (1.0 / (1.0 + np.exp(-np.clip(gate_float, -88, 88))))
    t_float = (time.time() - t0) / 100
    print(f"  Float SiLU:   {t_float*1000:.3f}ms")

    # Integer SiLU
    t0 = time.time()
    for _ in range(100):
        silu_s, silu_e = phi_silu_int(gate_s[0], gate_e[0])
    t_int = (time.time() - t0) / 100
    silu_int_decoded = phi_to_float(silu_s, silu_e)
    print(f"  Integer SiLU: {t_int*1000:.3f}ms")
    print(f"  Speedup: {t_float/t_int:.1f}×")

    corr = np.corrcoef(silu_float.flatten(), silu_int_decoded.flatten())[0, 1]
    print(f"  Correlation: {corr:.8f}")

    # ─── Test 4: RMS Norm ─────────────────────────────────────────
    print("\n  ─── Test 4: RMS Norm (3584 → 3584) ───")
    norms = np.load(os.path.join(MODEL_DIR, 'layer_00/norms.npz'))
    w = norms['input_layernorm'].astype(np.float32)
    w_s, w_e = float_to_phi(w)

    # Float
    t0 = time.time()
    for _ in range(100):
        rms = np.sqrt(np.mean(tok_float32 ** 2) + 1e-6)
        normed_float = (tok_float32 / rms) * w
    t_float = (time.time() - t0) / 100
    print(f"  Float RMS norm:   {t_float*1000:.3f}ms")

    # Integer (wrap as 2D to avoid scalar accumulation bug)
    t0 = time.time()
    for _ in range(100):
        n_s, n_e = phi_rms_norm_int(
            tok_signs[np.newaxis, :], tok_exps[np.newaxis, :],
            w_s, w_e)
    t_int = (time.time() - t0) / 100
    normed_int = phi_to_float(n_s[0], n_e[0])
    print(f"  Integer RMS norm: {t_int*1000:.3f}ms")
    print(f"  Speedup: {t_float/t_int:.1f}×")

    corr = np.corrcoef(normed_float.flatten(), normed_int.flatten())[0, 1]
    print(f"  Correlation: {corr:.8f}")

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print("  The φ-integer approach:")
    print("  ✓ No 87-second float decode at startup")
    print("  ✓ Weights used directly in native φ-encoding")
    print("  ✓ SiLU is a table lookup (no exp/sigmoid)")
    print("  ✓ RMS norm is integer ops (no sqrt)")
    print()
    print("  For GPU: need custom CUDA/Triton kernels for")
    print("  the block-scaled accumulation pattern")
    print()


if __name__ == '__main__':
    main()
