"""
Test: Negative Zero v2 — The φ-Center Hypothesis

Key observations from v1:
  1. step1/step2 = 2.61 ≈ φ² (exactly) — missing level between them
  2. Adding MORE nonlinearity (±0 split, exact gate) makes things WORSE
  3. The scaffold works because it's CONSTANT, not because it's accurate

New hypothesis: "Negative zero" = the gap between 0.5 and 1/φ.
  - 0.5 is the GAUSSIAN center (Φ(0), sigmoid(0))
  - 1/φ = 0.618 is the φ-NATURAL center
  - 1/φ - 1/2 = 0.118 ≈ 12% — the missing gap!

The bias pushes gate from 0.5 → 0.599, which is TOWARD 1/φ.

Test: sweep gate values to find the optimal CONSTANT gate.
If optimal ≈ 1/φ, the φ-center hypothesis is confirmed.
"""
import numpy as np
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI, _standard_gelu, _standard_gelu_derivative

np.random.seed(42)

LOG_PHI = np.log(PHI)

DIM = 32
N_TRAIN = 500
N_TEST = 200

W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(100, DIM).astype(np.float32)


# ================================================================
# Part 1: Sweep constant gate values
# ================================================================
print('=' * 70)
print('PART 1: Optimal Constant Gate Value')
print('=' * 70)
print()

gate_values = np.linspace(0.1, 0.95, 200)
rmses_by_gate = {}

# Average over seeds
for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    z_test = X_test @ pm.H.T + pm.b

    for g in gate_values:
        # Linear approximation with constant gate g: y ≈ R @ (g * z) + b_out
        # But we also need to account for the gate applied to the bias term
        Y_approx = (g * z_test) @ pm.R.T + pm.b_out
        rmse = np.sqrt(np.mean((Y_approx - Y_test)**2))
        if g not in rmses_by_gate:
            rmses_by_gate[g] = []
        rmses_by_gate[g].append(rmse)

# Find optimal
avg_rmses = {g: np.mean(rs) for g, rs in rmses_by_gate.items()}
optimal_gate = min(avg_rmses, key=avg_rmses.get)
optimal_rmse = avg_rmses[optimal_gate]

# Also get specific values
rmse_at_half = avg_rmses[min(gate_values, key=lambda g: abs(g - 0.5))]
rmse_at_phi = avg_rmses[min(gate_values, key=lambda g: abs(g - 1/PHI))]
rmse_at_phi2 = avg_rmses[min(gate_values, key=lambda g: abs(g - 1/PHI**2))]

print(f"  Optimal constant gate value: {optimal_gate:.4f}")
print(f"  Optimal RMSE:                {optimal_rmse:.4f}")
print()
print(f"  Comparison with φ-values:")
print(f"    g = 0.5     (scaffold):  RMSE = {rmse_at_half:.4f}")
print(f"    g = 1/φ²    (= 0.382):   RMSE = {rmse_at_phi2:.4f}")
print(f"    g = 1/φ     (= 0.618):   RMSE = {rmse_at_phi:.4f}")
print(f"    g = optimal (= {optimal_gate:.3f}):  RMSE = {optimal_rmse:.4f}")
print()

# How close is optimal to known constants?
print(f"  Distance from known constants:")
print(f"    |optimal - 0.5|   = {abs(optimal_gate - 0.5):.4f}")
print(f"    |optimal - 1/φ|   = {abs(optimal_gate - 1/PHI):.4f}")
print(f"    |optimal - 1/φ²|  = {abs(optimal_gate - 1/PHI**2):.4f}")
print(f"    |optimal - √(2/π)/2| = {abs(optimal_gate - np.sqrt(2/np.pi)/2):.4f}")
print(f"    |optimal - 1/2+1/(2φ²)| = {abs(optimal_gate - (0.5 + 0.5/PHI**2)):.4f}")
print()

# The landscape around the optimum
print(f"  Gate value landscape (φ markers):")
for g_name, g_val in [
    ("1/φ²", 1/PHI**2),
    ("0.5 (GELU'(0))", 0.5),
    ("OPTIMAL", optimal_gate),
    ("1/φ (= φ-1)", 1/PHI),
    ("√(2/π) (GELU''(0))", np.sqrt(2/np.pi)),
    ("1/√φ", 1/np.sqrt(PHI)),
]:
    rmse_here = avg_rmses[min(gate_values, key=lambda g: abs(g - g_val))]
    marker = " ◄◄◄" if abs(g_val - optimal_gate) < 0.01 else ""
    print(f"    g = {g_val:.4f} ({g_name:>20s}):  RMSE = {rmse_here:.4f}{marker}")


# ================================================================
# Part 2: Per-channel optimal gate
# ================================================================
print()
print('=' * 70)
print('PART 2: Per-Channel Optimal Gate')
print('=' * 70)
print()

# For one seed, find the optimal gate PER CHANNEL
pm = PhiMap(DIM, expansion=4, gate='gelu')
pm.init_random(seed=42)
pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

z_test_42 = X_test @ pm.H.T + pm.b   # [N, E]

# Brute force: for each channel, sweep gate value
channel_optimal_gates = []
for ch in range(pm.E):
    best_g = 0.5
    best_rmse = float('inf')
    for g in np.linspace(0.1, 0.95, 100):
        z_modified = _standard_gelu(z_test_42).copy()
        z_modified[:, ch] = g * z_test_42[:, ch]
        Y_approx = z_modified @ pm.R.T + pm.b_out
        rmse = np.sqrt(np.mean((Y_approx - Y_test)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_g = g
    channel_optimal_gates.append(best_g)

channel_optimal_gates = np.array(channel_optimal_gates)
gate_at_bias = _standard_gelu_derivative(pm.b)

print(f"  Per-channel optimal gate statistics:")
print(f"    Mean:   {channel_optimal_gates.mean():.4f}")
print(f"    Std:    {channel_optimal_gates.std():.4f}")
print(f"    Min:    {channel_optimal_gates.min():.4f}")
print(f"    Max:    {channel_optimal_gates.max():.4f}")
print()
print(f"  Comparison:")
print(f"    Mean optimal gate:  {channel_optimal_gates.mean():.4f}")
print(f"    0.5 (scaffold):    0.5000")
print(f"    1/φ (φ-center):    {1/PHI:.4f}")
print(f"    Mean GELU'(b):      {gate_at_bias.mean():.4f}")
print()

# Correlation between optimal gate and GELU'(b)
corr = np.corrcoef(channel_optimal_gates, gate_at_bias)[0, 1]
print(f"  Correlation(optimal_gate, GELU'(b)): {corr:.4f}")

# Distribution: how many channels prefer > 1/φ, between 0.5 and 1/φ, < 0.5?
above_phi = (channel_optimal_gates > 1/PHI).sum()
between = ((channel_optimal_gates >= 0.5) & (channel_optimal_gates <= 1/PHI)).sum()
below_half = (channel_optimal_gates < 0.5).sum()
print(f"\n  Distribution of optimal gates:")
print(f"    > 1/φ (0.618):   {above_phi}/{pm.E} ({above_phi/pm.E*100:.1f}%)")
print(f"    0.5 to 1/φ:      {between}/{pm.E} ({between/pm.E*100:.1f}%)")
print(f"    < 0.5:           {below_half}/{pm.E} ({below_half/pm.E*100:.1f}%)")


# ================================================================
# Part 3: The 0.5 vs 1/φ experiment with Jacobian comparison
# ================================================================
print()
print('=' * 70)
print('PART 3: φ-Scaffold vs Half-Scaffold vs Jacobian')
print('=' * 70)
print()

print(f"  {'Seed':<6} {'GELU':<9} {'g=0.5':<9} {'g=1/φ':<9} {'GELU`(b)':<9} {'Jacobian':<9} {'g=opt'}")
print(f"  " + "-" * 60)

for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    z = X_test @ pm.H.T + pm.b

    # GELU
    Y_gelu = pm.lookup(X_test)
    r_gelu = np.sqrt(np.mean((Y_gelu - Y_test)**2))

    # g = 0.5
    Y_half = (0.5 * z) @ pm.R.T + pm.b_out
    r_half = np.sqrt(np.mean((Y_half - Y_test)**2))

    # g = 1/φ
    Y_phi = ((1/PHI) * z) @ pm.R.T + pm.b_out
    r_phi = np.sqrt(np.mean((Y_phi - Y_test)**2))

    # GELU'(b) per channel
    gb = _standard_gelu_derivative(pm.b)
    Y_bias = (z * gb) @ pm.R.T + pm.b_out
    r_bias = np.sqrt(np.mean((Y_bias - Y_test)**2))

    # Jacobian
    pm.calibrate(X_cal)
    Y_jac = pm.default(X_test)
    r_jac = np.sqrt(np.mean((Y_jac - Y_test)**2))

    # Optimal constant (use the optimal_gate from sweep)
    Y_opt = (optimal_gate * z) @ pm.R.T + pm.b_out
    r_opt = np.sqrt(np.mean((Y_opt - Y_test)**2))

    print(f"  {seed:<6} {r_gelu:<9.4f} {r_half:<9.4f} {r_phi:<9.4f} "
          f"{r_bias:<9.4f} {r_jac:<9.4f} {r_opt:.4f}")


# ================================================================
# Part 4: The Negative Zero Identity
# ================================================================
print()
print('=' * 70)
print('PART 4: The Negative Zero')
print('=' * 70)
print()

print(f"  Key values:")
print(f"    GELU'(0)   = 0.5000  (the Gaussian zero)")
print(f"    1/φ        = {1/PHI:.4f}  (the φ-natural zero)")
print(f"    1/φ - 1/2  = {1/PHI - 0.5:.4f}  ('negative zero' = the gap)")
print()
print(f"    Optimal constant gate = {optimal_gate:.4f}")
print(f"    Gap from 0.5:     {optimal_gate - 0.5:.4f}")
print(f"    Gap from 1/φ:     {optimal_gate - 1/PHI:.4f}")
print()

# The encode = decode test at different centers
print(f"  ENCODE = DECODE test:")
print(f"    At g = 0.5:  encode(x) = x/2,   decode(y) = 2y      → encode·decode scale = 1.0")
print(f"    At g = 1/φ:  encode(x) = x/φ,   decode(y) = φy      → encode·decode scale = 1.0")
print(f"    At g = 0.5:  the 'decode amplification' = 2.000")
print(f"    At g = 1/φ:  the 'decode amplification' = φ = {PHI:.4f}")
print()
print(f"    Difference: the φ-center encodes/decodes with φ (self-similar)")
print(f"    The half-center encodes/decodes with 2 (not φ, breaks self-similarity)")
print()

# Check: is the optimal gate between 0.5 and 1/φ, and what fraction of the way?
if 0.5 <= optimal_gate <= 1/PHI:
    fraction = (optimal_gate - 0.5) / (1/PHI - 0.5)
    print(f"    Optimal is {fraction*100:.1f}% of the way from 0.5 to 1/φ")
elif optimal_gate > 1/PHI:
    print(f"    Optimal EXCEEDS 1/φ by {optimal_gate - 1/PHI:.4f}")
elif optimal_gate < 0.5:
    print(f"    Optimal is BELOW 0.5 by {0.5 - optimal_gate:.4f}")

# Does 1/φ - 0.5 relate to any φ-constant?
neg_zero = 1/PHI - 0.5
print(f"\n  'Negative zero' = 1/φ - 1/2 = {neg_zero:.6f}")
print(f"    = 1/(2φ²) ?  → {1/(2*PHI**2):.6f}  (deviation: {abs(neg_zero - 1/(2*PHI**2)):.6f})")
print(f"    = (φ-1)/2 ?  → {(PHI-1)/2:.6f}  (≡ 1/(2φ), same as 1/φ - 1/2)")
print(f"    = 1/2 · 1/φ  → {0.5/PHI:.6f}")
print(f"    YES: 1/φ - 1/2 = 1/(2φ) EXACTLY")
print(f"    Because 1/φ = (φ-1)/φ·(1/1) = 1 - 1/φ... no")
print(f"    Actually: 1/φ - 1/2 = (2-φ)/(2φ) = (2-1.618)/(2×1.618) = 0.382/3.236")
print(f"    = 1/φ² / (2φ) ... hmm")
print()
print(f"  Direct identity: 1/φ - 1/2 = 1/(2φ)")
print(f"    Proof: 1/φ = (φ-1) = 0.618...")
print(f"    1/φ - 1/2 = φ - 1 - 1/2 = φ - 3/2")
print(f"    Hmm no. 1/φ = φ - 1 = 0.618.  1/φ - 1/2 = 0.118")
print(f"    1/(2φ) = 1/(2×1.618) = 0.309.  Not the same.")
print()
print(f"    Exact: 1/φ - 1/2 = (2 - φ)/(2φ)")
print(f"    Since φ² = φ + 1, we have 2 - φ = 2 - φ")
print(f"    = (2-φ)/(2φ) = {(2-PHI)/(2*PHI):.6f}")
print(f"    Verified: {neg_zero:.6f}")
