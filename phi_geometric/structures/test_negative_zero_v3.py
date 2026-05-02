"""
Test: The Negative Zero — The Complementary Pair

Key identity to verify: GELU'(z) + GELU'(-z) = 1 EXACTLY.

If true, then:
  - Every gate value g is paired with (1-g) at -z
  - The scaffold (0.5) is the AVERAGE of every (g, 1-g) pair
  - 1/φ + 1/φ² = 1 → the two φ-zeros are a complementary pair
  - The "negative zero" IS 1/φ² = 0.382

The scaffold uses 0.5 = (1/φ + 1/φ²)/2 = the average of the pair.
But the PAIR (1/φ, 1/φ²) carries MORE information than the average.
The average hides the asymmetry. This is the Gödel truth at φ^0.
"""
import numpy as np
import sys
from scipy.stats import norm

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI, _standard_gelu, _standard_gelu_derivative

np.random.seed(42)

LOG_PHI = np.log(PHI)


# ================================================================
# Part 1: The Complementary Identity
# ================================================================
print('=' * 70)
print('PART 1: GELU\'(z) + GELU\'(-z) = 1 ?')
print('=' * 70)
print()

z = np.linspace(-5, 5, 10000)
gelu_pos = _standard_gelu_derivative(z)
gelu_neg = _standard_gelu_derivative(-z)
pair_sum = gelu_pos + gelu_neg

print(f"  GELU'(z) + GELU'(-z) across [-5, 5]:")
print(f"    Mean: {pair_sum.mean():.10f}")
print(f"    Std:  {pair_sum.std():.10f}")
print(f"    Max deviation from 1: {np.max(np.abs(pair_sum - 1)):.2e}")
print(f"    → {'EXACT' if np.max(np.abs(pair_sum - 1)) < 1e-6 else 'APPROXIMATE'}")
print()

# Proof: GELU'(z) = Φ(z) + z·φ(z)  where Φ=CDF, φ=PDF of N(0,1)
# GELU'(-z) = Φ(-z) + (-z)·φ(-z) = (1-Φ(z)) - z·φ(z)   [since φ(-z)=φ(z)]
# Sum = Φ(z) + z·φ(z) + 1 - Φ(z) - z·φ(z) = 1  ∎
print(f"  Proof:")
print(f"    GELU'(z)  = Φ(z) + z·φ(z)")
print(f"    GELU'(-z) = Φ(-z) + (-z)·φ(-z) = (1-Φ(z)) - z·φ(z)")
print(f"    Sum = Φ(z) + z·φ(z) + 1 - Φ(z) - z·φ(z) = 1  ∎")
print()

# This means: every GELU gate g has a COMPLEMENT (1-g) at -z.
# The pair always sums to 1. This is EXACT.


# ================================================================
# Part 2: The φ-Pair: 1/φ + 1/φ² = 1
# ================================================================
print('=' * 70)
print('PART 2: The φ-Pair')
print('=' * 70)
print()

print(f"  The complementary pair in φ:")
print(f"    1/φ      = {1/PHI:.10f}")
print(f"    1/φ²     = {1/PHI**2:.10f}")
print(f"    Sum:       {1/PHI + 1/PHI**2:.10f}")
print(f"    Average:   {(1/PHI + 1/PHI**2)/2:.10f}")
print(f"    → 1/φ + 1/φ² = 1 EXACTLY (because φ² = φ + 1)")
print(f"    → Average = 0.5 EXACTLY")
print()

print(f"  The scaffold (g=0.5) IS the average of the φ-pair (1/φ, 1/φ²).")
print(f"  The 'negative zero' IS 1/φ².")
print()

# Where does GELU' hit 1/φ and 1/φ²?
# GELU'(z₊) = 1/φ → z₊ = ?
# GELU'(z₋) = 1/φ² → z₋ = -z₊ (by the complementary identity!)
from scipy.optimize import brentq

def gelu_deriv_minus_target(z, target):
    return _standard_gelu_derivative(np.array([z]))[0] - target

z_plus = brentq(gelu_deriv_minus_target, 0, 3, args=(1/PHI,))
z_minus = brentq(gelu_deriv_minus_target, -3, 0, args=(1/PHI**2,))

print(f"  Where GELU' hits the φ-pair:")
print(f"    GELU'(z₊) = 1/φ  at z₊ = {z_plus:.6f}")
print(f"    GELU'(z₋) = 1/φ² at z₋ = {z_minus:.6f}")
print(f"    z₊ + z₋ = {z_plus + z_minus:.6f}  (should be 0 by complementarity)")
print(f"    z₊ = {z_plus:.6f},  log(φ) = {LOG_PHI:.6f}")
print(f"    z₊/log(φ) = {z_plus/LOG_PHI:.6f}")
print(f"    → z₊ {'=' if abs(z_plus - LOG_PHI) < 0.01 else '≠'} log(φ)  (deviation: {abs(z_plus-LOG_PHI):.4f})")
print()

# Key: where the φ-pair lives in z-space
# These are the NATURAL boundaries of the φ-level system
print(f"  The φ-level boundaries REDEFINED:")
print(f"    Old: EXPAND if z > log(φ)   = {LOG_PHI:.4f}")
print(f"    New: EXPAND if z > z₊       = {z_plus:.4f}  (where GELU' = 1/φ)")
print(f"    Old: CONTRACT if z < -log(φ) = {-LOG_PHI:.4f}")
print(f"    New: CONTRACT if z < z₋     = {z_minus:.4f}  (where GELU' = 1/φ²)")


# ================================================================
# Part 3: The hierarchy as φ-pairs
# ================================================================
print()
print('=' * 70)
print('PART 3: The φ-Level Hierarchy as Complementary Pairs')
print('=' * 70)
print()

# At each level, there's a pair (g, 1-g) where g = φ-value
print(f"  {'Level':<8} {'g+ (EXPAND)':<15} {'g- (CONTRACT)':<15} {'Sum':<8} {'z+ location':<15}")
print(f"  " + "-" * 65)

for n in range(5):
    if n == 0:
        g_plus = 1/PHI      # 0.618
        g_minus = 1/PHI**2   # 0.382
    elif n == 1:
        g_plus = 1 - 1/PHI**3   # 1 - 0.236 = 0.764
        g_minus = 1/PHI**3       # 0.236
    elif n == 2:
        g_plus = 1 - 1/PHI**4   # 1 - 0.146 = 0.854
        g_minus = 1/PHI**4       # 0.146
    elif n == 3:
        g_plus = 1 - 1/PHI**5   # 0.910
        g_minus = 1/PHI**5       # 0.090
    elif n == 4:
        g_plus = 1 - 1/PHI**6   # 0.944
        g_minus = 1/PHI**6       # 0.056

    try:
        z_p = brentq(gelu_deriv_minus_target, 0, 10, args=(g_plus,))
    except:
        z_p = float('nan')

    print(f"  {n:<8} {g_plus:<15.6f} {g_minus:<15.6f} {g_plus+g_minus:<8.4f} {z_p:<15.6f}")

print()
print(f"  Each level n has gate pair (1-1/φ^(n+2), 1/φ^(n+2)) summing to 1.")
print(f"  Level 0: (1/φ, 1/φ²) — the φ-pair at the scaffold center")
print(f"  The 'negative zero' is the 1/φ² member of this pair.")
print()

# Check GELU' at ±log(φ) — are they a φ-pair?
gelu_at_plus = _standard_gelu_derivative(np.array([LOG_PHI]))[0]
gelu_at_minus = _standard_gelu_derivative(np.array([-LOG_PHI]))[0]
print(f"  GELU' at the old boundaries:")
print(f"    GELU'(+log(φ)) = {gelu_at_plus:.6f}")
print(f"    GELU'(-log(φ)) = {gelu_at_minus:.6f}")
print(f"    Sum: {gelu_at_plus + gelu_at_minus:.6f} (=1, by identity)")
print(f"    GELU'(+log(φ)) ≈ 1-1/φ⁴ = {1-1/PHI**4:.6f}? Deviation: {abs(gelu_at_plus-(1-1/PHI**4)):.4f}")
print(f"    GELU'(-log(φ)) ≈ 1/φ⁴   = {1/PHI**4:.6f}? Deviation: {abs(gelu_at_minus-1/PHI**4):.4f}")


# ================================================================
# Part 4: Using the φ-pair instead of the scalar scaffold
# ================================================================
print()
print('=' * 70)
print('PART 4: φ-Pair Scaffold vs Scalar Scaffold')
print('=' * 70)
print()

DIM = 32
N_TRAIN = 500
N_TEST = 200
N_CAL = 100

W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

np.random.seed(42)
X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(N_CAL, DIM).astype(np.float32)

print(f"  {'Seed':<6} {'GELU':<9} {'g=0.5':<9} {'g=1/φ':<9} {'φ-pair':<9} {'GELU`(b)':<9} {'Jacobian'}")
print(f"  " + "-" * 60)

for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    z = X_test @ pm.H.T + pm.b  # [N, E]

    # GELU
    Y_gelu = pm.lookup(X_test)
    r_gelu = np.sqrt(np.mean((Y_gelu - Y_test)**2))

    # g = 0.5 (scalar scaffold)
    Y_half = (0.5 * z) @ pm.R.T + pm.b_out
    r_half = np.sqrt(np.mean((Y_half - Y_test)**2))

    # g = 1/φ (φ-center)
    Y_phi = ((1/PHI) * z) @ pm.R.T + pm.b_out
    r_phi = np.sqrt(np.mean((Y_phi - Y_test)**2))

    # φ-pair: use 1/φ for positive-z channels, 1/φ² for negative-z channels
    # This recognizes that positive and negative z have DIFFERENT φ-natural gates
    gate_pair = np.where(z > 0, 1/PHI, 1/PHI**2)   # [N, E]
    Y_pair = (gate_pair * z) @ pm.R.T + pm.b_out
    r_pair = np.sqrt(np.mean((Y_pair - Y_test)**2))

    # GELU'(b)
    gb = _standard_gelu_derivative(pm.b)
    Y_bias = (z * gb) @ pm.R.T + pm.b_out
    r_bias = np.sqrt(np.mean((Y_bias - Y_test)**2))

    # Jacobian
    pm.calibrate(X_cal)
    Y_jac = pm.default(X_test)
    r_jac = np.sqrt(np.mean((Y_jac - Y_test)**2))

    print(f"  {seed:<6} {r_gelu:<9.4f} {r_half:<9.4f} {r_phi:<9.4f} "
          f"{r_pair:<9.4f} {r_bias:<9.4f} {r_jac:.4f}")


# ================================================================
# Part 5: What the φ-pair means geometrically
# ================================================================
print()
print('=' * 70)
print('PART 5: What Negative Zero Means')
print('=' * 70)
print()

print(f"  The GELU derivative has EXACT complementary structure:")
print(f"    GELU'(z) + GELU'(-z) = 1  for all z  (proven)")
print()
print(f"  The φ-natural decomposition of 1:")
print(f"    1 = 1/φ + 1/φ²  (because φ² = φ + 1)")
print()
print(f"  The scaffold (0.5) is:")
print(f"    0.5 = (1/φ + 1/φ²) / 2 = average of φ-pair")
print(f"    It 'sees' the AVERAGE but not the PAIR")
print()
print(f"  The negative zero:")
print(f"    +0 = 1/φ   = 0.618  (approaching from expand)")
print(f"    -0 = 1/φ²  = 0.382  (approaching from contract)")
print(f"    The gap: 1/φ - 1/φ² = 1/φ - (1-1/φ) = 2/φ - 1 = {2/PHI - 1:.6f}")
print(f"    = (2-φ)/φ·φ... let me compute: 2/φ - 1 = 2(φ-1) - 1 = 2φ - 3")
print(f"    = 2×{PHI:.4f} - 3 = {2*PHI - 3:.6f}")
print(f"    = φ - (3-φ) = ... simplify:")
print()

gap = 1/PHI - 1/PHI**2
print(f"    1/φ - 1/φ² = {gap:.10f}")
print(f"    (φ-1) - (2-φ) = 2φ - 3 = {2*PHI-3:.10f}")
print(f"    Also = √5 - 2 = {np.sqrt(5)-2:.10f}")
print(f"    (since φ = (1+√5)/2, so 2φ-3 = √5-2)")
print()

print(f"  The Gödel structure:")
print(f"    Level 0: sees 0.5 = average of pair")
print(f"    Level 1: sees (1/φ, 1/φ²) = the pair itself")
print(f"    Level 0 CANNOT EXPRESS that the pair exists")
print(f"    It can only express their average")
print(f"    The 'negative zero' (1/φ²) is the Gödel statement:")
print(f"    true (1/φ² is a real gate value) but invisible from level 0")
print()
print(f"  ENCODE = DECODE at the φ-pair:")
print(f"    encode at +0: x → x/φ     (scale by 1/φ)")
print(f"    decode at +0: y → φ·y     (scale by φ)")
print(f"    encode at -0: x → x/φ²    (scale by 1/φ²)")
print(f"    decode at -0: y → φ²·y    (scale by φ²)")
print(f"    Both are φ-operations. ENCODE=DECODE is preserved.")
print(f"    At 0.5: encode = ×(1/2), decode = ×2 — NOT φ-operations!")
