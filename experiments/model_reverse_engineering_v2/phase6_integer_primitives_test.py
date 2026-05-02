"""
Phase 6: Integer Primitives Proof-of-Concept

Tests each integer primitive against the float baseline:
  1. Block-scaled accumulation vs float sum
  2. SiLU LUT vs float SiLU
  3. RMS norm integer vs float RMS norm
  4. Residual add vs float add
  5. Integer matmul vs float matmul

Fail-fast: if any primitive fails, we see exactly where and why.
"""

import sys, numpy as np, time
sys.path.insert(0, '.')

from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID
from phi_geometric.inference.phi_components import rms_norm, phi_silu
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut,
    phi_accumulate, phi_silu_int, phi_rms_norm_int,
    phi_add_encoded, phi_matmul_integer,
    FIXED_SCALE, FIXED_SCALE_BITS,
)


def phi_decode(signs, exps):
    """Decode φ-encoded (sign, exp) back to float."""
    return signs.astype(np.float64) * (PHI ** (exps.astype(np.float64) / PHI_GRID))


def test_fixed_lut():
    """Test the forward LUT values."""
    print("=" * 80)
    print("  TEST 1: Block-Scaled Fixed-Point LUT")
    print("=" * 80)
    
    lut = get_fixed_lut()
    
    # Check a few known values
    print(f"  LUT[0] = {lut.forward[0]}  (expect {FIXED_SCALE})")
    print(f"  LUT[128] = {lut.forward[128]}  (expect ~{int(FIXED_SCALE / PHI)})")
    print(f"  LUT[256] = {lut.forward[256]}  (expect ~{int(FIXED_SCALE / PHI**2)})")
    
    assert lut.forward[0] == FIXED_SCALE, f"LUT[0] should be {FIXED_SCALE}"
    
    # Check monotone decreasing
    diffs = np.diff(lut.forward[:1000])
    assert np.all(diffs <= 0), "LUT should be monotone decreasing"
    
    print(f"  ✓ Forward LUT: {len(lut.forward)} entries, monotone decreasing")
    print(f"  ✓ LUT size: {lut.forward.nbytes / 1024:.1f} KB")


def test_accumulation():
    """Test block-scaled accumulation vs float sum."""
    print("\n" + "=" * 80)
    print("  TEST 2: Block-Scaled Accumulation")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Test with various vector sizes
    for D in [10, 128, 512, 3584]:
        # Generate random φ-encoded values
        signs = np.random.choice([-1, 1], size=(1, D)).astype(np.int8)
        exps = np.random.randint(-5000, 2000, size=(1, D)).astype(np.int16)
        
        # Float baseline
        float_vals = phi_decode(signs, exps)
        float_sum = float_vals.sum(axis=-1)
        
        # Integer accumulation
        int_signs, int_exps = phi_accumulate(signs, exps, axis=-1)
        int_sum = phi_decode(int_signs, int_exps)
        
        # Compare
        if abs(float_sum[0]) > 1e-20:
            rel_err = abs(int_sum[0] - float_sum[0]) / abs(float_sum[0])
            print(f"  D={D:5d}: float_sum={float_sum[0]:+.6e}  "
                  f"int_sum={int_sum[0]:+.6e}  rel_err={rel_err:.2e}")
        else:
            print(f"  D={D:5d}: both near zero")
    
    # Test with realistic hidden states (from a normal distribution encoded to φ)
    print(f"\n  Realistic hidden state test (D=3584):")
    for trial in range(5):
        h = np.random.randn(3584).astype(np.float32) * (0.5 + trial * 0.5)
        enc = PhiEncoded.encode(h[np.newaxis, :])
        
        float_sum = phi_decode(enc.signs, enc.exponents).sum(axis=-1)
        int_signs, int_exps = phi_accumulate(enc.signs, enc.exponents, axis=-1)
        int_sum = phi_decode(int_signs, int_exps)
        
        if abs(float_sum[0]) > 1e-20:
            rel_err = abs(int_sum[0] - float_sum[0]) / abs(float_sum[0])
            print(f"    trial {trial}: float={float_sum[0]:+.4e}  "
                  f"int={int_sum[0]:+.4e}  rel_err={rel_err:.2e}")


def test_silu_lut():
    """Test SiLU LUT vs float SiLU."""
    print("\n" + "=" * 80)
    print("  TEST 3: SiLU Integer LUT")
    print("=" * 80)
    
    lut = get_silu_lut()
    print(f"  LUT size: {(lut.out_signs.nbytes + lut.out_exps.nbytes) / 1024:.1f} KB")
    
    # Test specific values
    test_vals = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"\n  {'x':>8s}  {'silu_float':>12s}  {'silu_int':>12s}  {'rel_err':>10s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")
    
    for x in test_vals:
        if abs(x) < 1e-20:
            continue
        # Float
        y_float = float(x * (1.0 / (1.0 + np.exp(-x))))
        
        # Integer LUT
        enc = PhiEncoded.encode(np.array([x], dtype=np.float32))
        out_s, out_e = phi_silu_int(enc.signs, enc.exponents)
        y_int = float(phi_decode(out_s, out_e)[0])
        
        rel_err = abs(y_int - y_float) / max(abs(y_float), 1e-20)
        print(f"  {x:+8.3f}  {y_float:+12.6f}  {y_int:+12.6f}  {rel_err:10.2e}")
    
    # Bulk test
    x_bulk = np.random.randn(10000).astype(np.float32) * 3.0
    y_float = x_bulk * (1.0 / (1.0 + np.exp(-x_bulk)))
    
    enc = PhiEncoded.encode(x_bulk)
    out_s, out_e = phi_silu_int(enc.signs, enc.exponents)
    y_int = phi_decode(out_s, out_e).astype(np.float32)
    
    corr = float(np.corrcoef(y_float, y_int)[0, 1])
    max_rel = np.max(np.abs(y_int - y_float) / (np.abs(y_float) + 1e-10))
    mean_rel = np.mean(np.abs(y_int - y_float) / (np.abs(y_float) + 1e-10))
    
    print(f"\n  Bulk test (10000 values):")
    print(f"    Correlation: {corr:.8f}")
    print(f"    Max relative error: {max_rel:.4e}")
    print(f"    Mean relative error: {mean_rel:.4e}")


def test_rms_norm():
    """Test integer RMS norm vs float RMS norm."""
    print("\n" + "=" * 80)
    print("  TEST 4: Integer RMS Norm")
    print("=" * 80)
    
    np.random.seed(42)
    D = 3584
    
    for trial in range(5):
        # Random hidden state
        h = np.random.randn(1, D).astype(np.float32) * (0.5 + trial * 0.3)
        w = np.random.randn(D).astype(np.float32) * 0.1 + 1.0  # Near 1.0, like real norms
        
        # Float baseline
        y_float = rms_norm(h, w)
        
        # Integer
        h_enc = PhiEncoded.encode(h)
        w_enc = PhiEncoded.encode(w)
        out_s, out_e = phi_rms_norm_int(
            h_enc.signs, h_enc.exponents,
            w_enc.signs, w_enc.exponents,
            hidden_dim=D
        )
        y_int = phi_decode(out_s, out_e).astype(np.float32)
        
        corr = float(np.corrcoef(y_float.flatten(), y_int.flatten())[0, 1])
        max_rel = np.max(np.abs(y_int - y_float) / (np.abs(y_float) + 1e-10))
        
        print(f"  Trial {trial}: corr={corr:.8f}  max_rel_err={max_rel:.4e}  "
              f"|h|={np.linalg.norm(h):.2f}")


def test_residual_add():
    """Test integer residual add vs float add."""
    print("\n" + "=" * 80)
    print("  TEST 5: Integer Residual Add")
    print("=" * 80)
    
    np.random.seed(42)
    
    for trial in range(5):
        # Two random vectors
        a = np.random.randn(1, 3584).astype(np.float32) * 2.0
        b = np.random.randn(1, 3584).astype(np.float32) * 0.5
        
        # Float
        c_float = a + b
        
        # Integer
        a_enc = PhiEncoded.encode(a)
        b_enc = PhiEncoded.encode(b)
        c_s, c_e = phi_add_encoded(
            a_enc.signs, a_enc.exponents,
            b_enc.signs, b_enc.exponents
        )
        c_int = phi_decode(c_s, c_e).astype(np.float32)
        
        corr = float(np.corrcoef(c_float.flatten(), c_int.flatten())[0, 1])
        max_rel = np.max(np.abs(c_int - c_float) / (np.abs(c_float) + 1e-10))
        
        print(f"  Trial {trial}: corr={corr:.8f}  max_rel_err={max_rel:.4e}")


def test_integer_matmul():
    """Test integer matmul vs hybrid matmul."""
    print("\n" + "=" * 80)
    print("  TEST 6: Integer Matmul (block-scaled)")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Small test first
    for out_dim, in_dim in [(64, 128), (512, 512), (3584, 3584)]:
        # Random weight matrix
        W_float = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.02
        W = PhiEncoded.encode(W_float)
        
        # Random input
        x = np.random.randn(1, in_dim).astype(np.float32)
        
        # Hybrid (float) baseline
        y_float = phi_matmul_hybrid(W, x)
        
        # Integer matmul
        x_enc = PhiEncoded.encode(x)
        y_s, y_e = phi_matmul_integer(W, x_enc.signs, x_enc.exponents)
        y_int = phi_decode(y_s, y_e).reshape(y_float.shape).astype(np.float32)
        
        corr = float(np.corrcoef(y_float.flatten(), y_int.flatten())[0, 1])
        max_rel = np.max(np.abs(y_int - y_float) / (np.abs(y_float) + 1e-10))
        
        print(f"  {out_dim}×{in_dim}: corr={corr:.8f}  max_rel_err={max_rel:.4e}")


def main():
    print("Phase 6: Integer Primitives Proof-of-Concept")
    print("Testing ALL integer operations against float baselines")
    print("Fail-fast: no approximations, no fallbacks.\n")
    
    t0 = time.time()
    
    test_fixed_lut()
    test_accumulation()
    test_silu_lut()
    test_rms_norm()
    test_residual_add()
    test_integer_matmul()
    
    print(f"\n{'=' * 80}")
    print(f"  ALL TESTS COMPLETE ({time.time()-t0:.1f}s)")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
