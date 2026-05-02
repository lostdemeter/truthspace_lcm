"""
Verify the 4-state SiLU LUT (Doc 254).

Tests:
1. Gate code boundaries are correct at ±log(φ)
2. Near-zero sign preservation (the key fix)
3. Backward compatibility: phi_silu_int unchanged
4. phi_silu_4state returns correct gate codes
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from phi_geometric.inference.phi_integer import (
    PhiSiLULUT, phi_silu_int, phi_silu_4state,
    GATE_CONTRACT, GATE_PRESERVE_N, GATE_PRESERVE_P, GATE_EXPAND,
    LOG_PHI_BOUNDARY, PHI_GRID, EXP_MIN,
)
from phi_geometric.inference.phi_types import PHI, LOG_PHI

print("=" * 60)
print("4-STATE SiLU LUT VERIFICATION")
print("=" * 60)

lut = PhiSiLULUT()

# Test 1: Gate code boundary verification
print("\n--- Test 1: Gate code boundaries at ±log(φ) ---")
test_values = [
    (-5.0,   GATE_CONTRACT,   "deep negative"),
    (-1.0,   GATE_CONTRACT,   "moderate negative"),
    (-0.5,   GATE_CONTRACT,   "just past -log(φ)"),
    (-0.48,  GATE_PRESERVE_N, "just inside PRESERVE-"),
    (-0.1,   GATE_PRESERVE_N, "negative near zero"),
    (-0.001, GATE_PRESERVE_N, "tiny negative"),
    (0.0,    GATE_PRESERVE_P, "exact zero"),
    (0.001,  GATE_PRESERVE_P, "tiny positive"),
    (0.1,    GATE_PRESERVE_P, "positive near zero"),
    (0.48,   GATE_PRESERVE_P, "just inside PRESERVE+"),
    (0.5,    GATE_EXPAND,     "just past +log(φ)"),
    (1.0,    GATE_EXPAND,     "moderate positive"),
    (5.0,    GATE_EXPAND,     "deep positive"),
]

GATE_NAMES = {0: "CONTRACT", 1: "PRESERVE-", 2: "PRESERVE+", 3: "EXPAND"}
all_pass = True

for x_val, expected_code, desc in test_values:
    # Encode x_val to (sign, exp)
    sign = np.int8(1 if x_val >= 0 else -1)
    if abs(x_val) < 1e-45:
        exp_val = np.int16(EXP_MIN)
    else:
        exp_val = np.int16(round(PHI_GRID * np.log(abs(x_val)) / LOG_PHI))
    
    signs = np.array([sign], dtype=np.int8)
    exps = np.array([exp_val], dtype=np.int16)
    
    _, _, gate_codes = phi_silu_4state(signs, exps)
    actual_code = gate_codes[0]
    
    ok = actual_code == expected_code
    if not ok:
        all_pass = False
    
    status = "✓" if ok else "✗"
    print(f"  {status} x={x_val:+7.3f} ({desc:25s}) → {GATE_NAMES[actual_code]:10s} "
          f"(expected {GATE_NAMES[expected_code]})")

print(f"\n  Boundary test: {'PASS' if all_pass else 'FAIL'}")

# Test 2: Near-zero sign preservation
print("\n--- Test 2: Near-zero sign preservation ---")
# For very negative x, SiLU output is tiny but NEGATIVE
# The old LUT would force sign=+1, the new one preserves sign=-1
test_neg = [(-10.0, -1), (-20.0, -1), (-30.0, -1)]
sign_pass = True

for x_val, expected_sign in test_neg:
    sign = np.int8(-1)
    exp_val = np.int16(round(PHI_GRID * np.log(abs(x_val)) / LOG_PHI))
    signs = np.array([sign], dtype=np.int8)
    exps = np.array([exp_val], dtype=np.int16)
    
    out_s, out_e = phi_silu_int(signs, exps)
    actual_sign = out_s[0]
    
    ok = actual_sign == expected_sign
    if not ok:
        sign_pass = False
    
    # Also compute actual SiLU for reference
    y = x_val * (1.0 / (1.0 + np.exp(-x_val)))
    
    status = "✓" if ok else "✗"
    print(f"  {status} x={x_val:+6.1f} → sign={actual_sign:+d} (expected {expected_sign:+d}), "
          f"actual SiLU={y:.2e}")

print(f"\n  Sign preservation test: {'PASS' if sign_pass else 'FAIL'}")

# Test 3: Backward compatibility
print("\n--- Test 3: Backward compatibility ---")
np.random.seed(42)
N = 10000
rand_signs = np.random.choice([-1, 1], size=N).astype(np.int8)
rand_exps = np.random.randint(-5000, 3000, size=N).astype(np.int16)

out_s_2, out_e_2 = phi_silu_int(rand_signs, rand_exps)
out_s_4, out_e_4, gate_c = phi_silu_4state(rand_signs, rand_exps)

# Signs and exps should match between 2-state and 4-state calls
signs_match = np.array_equal(out_s_2, out_s_4)
exps_match = np.array_equal(out_e_2, out_e_4)

print(f"  Signs match (2-state vs 4-state): {signs_match}")
print(f"  Exps match (2-state vs 4-state): {exps_match}")
print(f"\n  Backward compatibility: {'PASS' if signs_match and exps_match else 'FAIL'}")

# Test 4: Gate code distribution on random data
print("\n--- Test 4: Gate code distribution ---")
counts = np.bincount(gate_c.astype(np.int32), minlength=4)
for code, name in GATE_NAMES.items():
    pct = counts[code] / N * 100
    print(f"  {name:10s}: {counts[code]:5d} ({pct:5.1f}%)")

# Test 5: Verify SiLU accuracy on PRESERVE region
print("\n--- Test 5: SiLU accuracy in PRESERVE region ---")
preserve_mask = (gate_c == GATE_PRESERVE_N) | (gate_c == GATE_PRESERVE_P)
n_preserve = preserve_mask.sum()
print(f"  PRESERVE channels: {n_preserve}/{N} ({n_preserve/N*100:.1f}%)")

if n_preserve > 0:
    # Reconstruct float values
    x_float = rand_signs[preserve_mask].astype(np.float64) * PHI ** (rand_exps[preserve_mask].astype(np.float64) / PHI_GRID)
    y_actual = x_float * (1.0 / (1.0 + np.exp(-np.clip(x_float, -500, 500))))
    y_encoded = out_s_4[preserve_mask].astype(np.float64) * PHI ** (out_e_4[preserve_mask].astype(np.float64) / PHI_GRID)
    
    # Relative error
    rel_err = np.abs(y_encoded - y_actual) / (np.abs(y_actual) + 1e-30)
    print(f"  Relative error (median): {np.median(rel_err):.4f}")
    print(f"  Relative error (95th):   {np.percentile(rel_err, 95):.4f}")
    print(f"  Relative error (max):    {np.max(rel_err):.4f}")

print("\n" + "=" * 60)
overall = all_pass and sign_pass and signs_match and exps_match
print(f"OVERALL: {'ALL TESTS PASS' if overall else 'SOME TESTS FAILED'}")
print("=" * 60)
