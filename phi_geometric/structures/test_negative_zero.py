"""
Test: The Negative Zero Hypothesis

At φ^0 (the PRESERVE region), the scaffold treats everything as x/2.
But GELU is NOT symmetric within this region:

  GELU(+ε) ≈ ε/2 + ε²·√(2/π)/4  (pushed MORE positive)
  GELU(-ε) ≈ -ε/2 + ε²·√(2/π)/4  (pushed LESS negative)

The curvature = GELU''(0) = √(2/π) ≈ φ/2 (within 1.38%).

This means φ^0 has TWO halves:
  φ^(+0): z ∈ [0, log(φ)]     — positive preserve, GELU amplifies
  φ^(-0): z ∈ [-log(φ), 0]    — negative preserve, GELU softens

The ternary system can't express this from within (Gödel).
Splitting level 0 should capture the missing 12%.

Key predictions:
  1. GELU derivative is very different at +0 vs -0
  2. Splitting PRESERVE into ±0 halves captures additional structure
  3. The convergence follows φ-structure (each level captures ~1/φ of remainder)
  4. The curvature φ/2 at the splitting point is not coincidence
"""
import numpy as np
import sys
from scipy.special import erf
from scipy.stats import norm

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI, _standard_gelu, _standard_gelu_derivative

np.random.seed(42)

LOG_PHI = np.log(PHI)  # ≈ 0.481


# ================================================================
# Part 1: The asymmetry within PRESERVE
# ================================================================
print('=' * 70)
print('PART 1: The Asymmetry at φ^0')
print('=' * 70)
print()

z_range = np.linspace(-LOG_PHI, LOG_PHI, 1000)
gelu_deriv = _standard_gelu_derivative(z_range)

# Split at 0
pos_mask = z_range >= 0
neg_mask = z_range < 0

mean_deriv_pos = gelu_deriv[pos_mask].mean()
mean_deriv_neg = gelu_deriv[neg_mask].mean()
mean_deriv_all = gelu_deriv.mean()

print(f"  GELU'(z) within PRESERVE region [-log(φ), +log(φ)]:")
print(f"    Overall mean:     {mean_deriv_all:.4f}  (scaffold assumes 0.5)")
print(f"    φ^(+0) mean:      {mean_deriv_pos:.4f}  (z ∈ [0, log(φ)])")
print(f"    φ^(-0) mean:      {mean_deriv_neg:.4f}  (z ∈ [-log(φ), 0])")
print(f"    Ratio +0/-0:      {mean_deriv_pos/mean_deriv_neg:.4f}")
print(f"    Asymmetry:        {(mean_deriv_pos - mean_deriv_neg):.4f}")
print()

# The curvature
gelu_second = np.sqrt(2 / np.pi)
print(f"  GELU''(0) = √(2/π) = {gelu_second:.6f}")
print(f"  φ/2                = {PHI/2:.6f}")
print(f"  Match:               {abs(gelu_second - PHI/2) / gelu_second * 100:.2f}% deviation")
print()

# What does GELU' look like at specific points?
print(f"  GELU'(z) at key points within PRESERVE:")
for z_val in [-LOG_PHI, -LOG_PHI/2, -LOG_PHI/PHI, 0, LOG_PHI/PHI, LOG_PHI/2, LOG_PHI]:
    d = _standard_gelu_derivative(np.array([z_val]))[0]
    region = "φ^(-0)" if z_val < 0 else ("φ^(+0)" if z_val > 0 else "CENTER")
    label = ""
    if abs(z_val) < 1e-10:
        label = " = 0"
    elif abs(z_val - LOG_PHI) < 1e-10:
        label = " = log(φ)"
    elif abs(z_val + LOG_PHI) < 1e-10:
        label = " = -log(φ)"
    elif abs(z_val - LOG_PHI/2) < 1e-6:
        label = " = log(φ)/2"
    elif abs(z_val + LOG_PHI/2) < 1e-6:
        label = " = -log(φ)/2"
    elif abs(z_val - LOG_PHI/PHI) < 1e-6:
        label = " = log(φ)/φ"
    elif abs(z_val + LOG_PHI/PHI) < 1e-6:
        label = " = -log(φ)/φ"
    print(f"    z = {z_val:+.4f}{label:>14s}  →  GELU'(z) = {d:.4f}  [{region}]")


# ================================================================
# Part 2: Does splitting φ^0 close the 12% gap?
# ================================================================
print()
print('=' * 70)
print('PART 2: Splitting φ^0 Into ±0')
print('=' * 70)
print()

DIM = 32
N_TRAIN = 500
N_TEST = 200
N_CAL = 100

W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(N_CAL, DIM).astype(np.float32)

results_all = []

for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    z_test = X_test @ pm.H.T + pm.b   # [N, E]
    z_cal = X_cal @ pm.H.T + pm.b

    # 1. Full GELU (nonlinear)
    Y_nl = pm.lookup(X_test)
    rmse_nl = np.sqrt(np.mean((Y_nl - Y_test)**2))

    # 2. Mean Jacobian (statistical, 100 calibration)
    pm.calibrate(X_cal)
    Y_jac = pm.default(X_test)
    rmse_jac = np.sqrt(np.mean((Y_jac - Y_test)**2))

    # 3. Scaffold: ½R@H (geometric, 0 data)
    gate_half = np.full(pm.E, 0.5)
    S3 = (pm.R * gate_half) @ pm.H
    b3 = pm.R @ (pm.b * 0.5) + pm.b_out
    Y_s3 = X_test @ S3.T + b3
    rmse_s3 = np.sqrt(np.mean((Y_s3 - Y_test)**2))

    # 4. Bias-corrected: GELU'(b)R@H (geometric, 0 data)
    gate_bias = _standard_gelu_derivative(pm.b)
    S4 = (pm.R * gate_bias) @ pm.H
    b4 = pm.R @ _standard_gelu(pm.b) + pm.b_out
    Y_s4 = X_test @ S4.T + b4
    rmse_s4 = np.sqrt(np.mean((Y_s4 - Y_test)**2))

    # 5. ±0 split: use GELU'(b + σ) for +0 region, GELU'(b - σ) for -0 region
    # The key insight: for each channel, z = H@x + b.
    # The spread of z around b depends on the input distribution.
    # But we can estimate this GEOMETRICALLY from the hyperplane norm.
    # ||H_i|| gives the sensitivity of channel i to input perturbations.
    # For unit Gaussian input, std(z_i) ≈ ||H_i||.
    h_norms = np.sqrt(np.sum(pm.H**2, axis=1))  # [E]

    # For each channel, the effective gate at +0 and -0:
    # When z > b (positive perturbation): effective gate ≈ GELU'(b + h_norm*δ)
    # When z < b (negative perturbation): effective gate ≈ GELU'(b - h_norm*δ)
    # We use the EXPECTED gate derivative for each half of the distribution.
    # For a Gaussian with mean b and std h_norm:
    #   E[GELU'(z) | z > b] = average of GELU' over [b, b + 2*h_norm] approximately
    #   E[GELU'(z) | z < b] = average of GELU' over [b - 2*h_norm, b] approximately

    # But we want to do this WITHOUT calibration data.
    # Use the geometric fact: GELU'(b + δ) ≈ GELU'(b) + GELU''(b)·δ
    # The second derivative captures the curvature.
    # For the positive half: E[δ | δ > 0] ≈ h_norm * √(2/π) (half-normal mean)
    # For the negative half: E[δ | δ < 0] ≈ -h_norm * √(2/π)

    half_normal_mean = h_norms * np.sqrt(2 / np.pi)  # [E]

    # GELU second derivative at bias point
    # GELU'(x) = Φ(x) + x·φ(x) where Φ=CDF, φ=PDF of standard normal
    # GELU''(x) = 2·φ(x) + x·φ'(x) = 2·φ(x) - x²·φ(x) = φ(x)(2 - x²)
    gelu_second_at_b = norm.pdf(pm.b) * (2 - pm.b**2)  # [E]

    gate_pos = gate_bias + gelu_second_at_b * half_normal_mean   # GELU' for +0
    gate_neg = gate_bias - gelu_second_at_b * half_normal_mean   # GELU' for -0

    # For each test input, each channel: use gate_pos if z > b, gate_neg if z < b
    pos_channels = z_test > pm.b[np.newaxis, :]   # [N, E]
    gate_split = np.where(pos_channels, gate_pos[np.newaxis, :], gate_neg[np.newaxis, :])

    # Reconstruction with ±0 split
    gelu_approx = z_test * gate_split   # [N, E]
    Y_s5 = gelu_approx @ pm.R.T + pm.b_out
    rmse_s5 = np.sqrt(np.mean((Y_s5 - Y_test)**2))

    # 6. For comparison: PERFECT per-channel gate (what if we knew the exact per-input GELU'?)
    gate_exact = _standard_gelu_derivative(z_test)  # [N, E]
    gelu_linear_exact = z_test * gate_exact
    Y_s6 = gelu_linear_exact @ pm.R.T + pm.b_out
    rmse_s6 = np.sqrt(np.mean((Y_s6 - Y_test)**2))

    results_all.append({
        'seed': seed,
        'rmse_nl': rmse_nl,
        'rmse_jac': rmse_jac,
        'rmse_scaffold': rmse_s3,
        'rmse_bias': rmse_s4,
        'rmse_split': rmse_s5,
        'rmse_exact_gate': rmse_s6,
    })

print(f"  {'Seed':<6} {'GELU':<9} {'Jacobian':<9} {'½R@H':<9} {'GELU`(b)':<9} {'±0 split':<9} {'exact g`'}")
print(f"  " + "-" * 60)
for r in results_all:
    print(f"  {r['seed']:<6} {r['rmse_nl']:<9.4f} {r['rmse_jac']:<9.4f} "
          f"{r['rmse_scaffold']:<9.4f} {r['rmse_bias']:<9.4f} "
          f"{r['rmse_split']:<9.4f} {r['rmse_exact_gate']:.4f}")

# Averages
avg = {k: np.mean([r[k] for r in results_all]) for k in results_all[0] if k != 'seed'}
print(f"\n  {'AVG':<6} {avg['rmse_nl']:<9.4f} {avg['rmse_jac']:<9.4f} "
      f"{avg['rmse_scaffold']:<9.4f} {avg['rmse_bias']:<9.4f} "
      f"{avg['rmse_split']:<9.4f} {avg['rmse_exact_gate']:.4f}")

# Gap analysis
improvement_jac = (avg['rmse_nl'] - avg['rmse_jac']) / avg['rmse_nl'] * 100
improvement_scaffold = (avg['rmse_nl'] - avg['rmse_scaffold']) / avg['rmse_nl'] * 100
improvement_bias = (avg['rmse_nl'] - avg['rmse_bias']) / avg['rmse_nl'] * 100
improvement_split = (avg['rmse_nl'] - avg['rmse_split']) / avg['rmse_nl'] * 100
improvement_exact = (avg['rmse_nl'] - avg['rmse_exact_gate']) / avg['rmse_nl'] * 100

print(f"\n  Improvement over full GELU:")
print(f"    Jacobian (100 cal):     {improvement_jac:+.2f}%  (TARGET)")
print(f"    Scaffold ½R@H (0 cal):  {improvement_scaffold:+.2f}%")
print(f"    Bias GELU'(b) (0 cal):  {improvement_bias:+.2f}%")
print(f"    ±0 split (0 cal):       {improvement_split:+.2f}%")
print(f"    Exact gate' (oracle):   {improvement_exact:+.2f}%")

print(f"\n  Fraction of Jacobian advantage captured (geometrically):")
if improvement_jac != 0:
    print(f"    Scaffold:   {improvement_scaffold/improvement_jac*100:.1f}%")
    print(f"    + Bias:     {improvement_bias/improvement_jac*100:.1f}%")
    print(f"    + ±0 split: {improvement_split/improvement_jac*100:.1f}%")
    print(f"    Exact gate: {improvement_exact/improvement_jac*100:.1f}%  (theoretical max)")


# ================================================================
# Part 3: The φ-structure of convergence
# ================================================================
print()
print('=' * 70)
print('PART 3: Does the Convergence Follow φ?')
print('=' * 70)
print()

# The incremental improvements
total_improvement = improvement_jac
step1 = improvement_scaffold                        # scaffold
step2 = improvement_bias - improvement_scaffold     # bias correction
step3 = improvement_split - improvement_bias        # ±0 split
step4 = improvement_jac - improvement_split         # remaining (statistical)

print(f"  Incremental improvements:")
print(f"    Step 1 (scaffold):      {step1:+.4f}%")
print(f"    Step 2 (bias):          {step2:+.4f}%")
print(f"    Step 3 (±0 split):      {step3:+.4f}%")
print(f"    Step 4 (remaining):     {step4:+.4f}%  (this is the only statistical part)")
print()

# Check for φ-ratios between steps
if abs(step2) > 1e-6:
    ratio_1_2 = step1 / step2
    print(f"  Ratios between steps:")
    print(f"    step1/step2 = {ratio_1_2:.4f}  (φ = {PHI:.4f}, φ² = {PHI**2:.4f})")
if abs(step3) > 1e-6:
    ratio_2_3 = step2 / step3
    print(f"    step2/step3 = {ratio_2_3:.4f}")
if abs(step4) > 1e-6 and abs(step3) > 1e-6:
    ratio_3_4 = step3 / step4
    print(f"    step3/step4 = {ratio_3_4:.4f}")
print()

# Fibonacci-like: does each step ≈ sum of next two?
print(f"  Fibonacci property (each step ≈ sum of next two):")
print(f"    step1 = {step1:.4f},  step2 + step3 = {step2+step3:.4f}  "
      f"(ratio: {step1/(step2+step3) if abs(step2+step3)>1e-6 else float('inf'):.4f})")
print(f"    step2 = {step2:.4f},  step3 + step4 = {step3+step4:.4f}  "
      f"(ratio: {step2/(step3+step4) if abs(step3+step4)>1e-6 else float('inf'):.4f})")


# ================================================================
# Part 4: The Gödel connection — self-referential incompleteness
# ================================================================
print()
print('=' * 70)
print('PART 4: The Gödel Structure')
print('=' * 70)
print()

print(f"  The hierarchy of φ-levels, each invisible from the one above:")
print()
print(f"  Level 0: Scaffold (½R@H)")
print(f"    Assumes GELU' = 0.5 everywhere")
print(f"    Cannot see: per-channel bias shift")
print(f"    Captures: {improvement_scaffold/improvement_jac*100:.1f}% of Jacobian")
print()
print(f"  Level 1: Bias correction (GELU'(b))")
print(f"    Reads each channel's resting φ-level from bias")
print(f"    Cannot see: ±0 asymmetry within PRESERVE")
print(f"    Captures: {improvement_bias/improvement_jac*100:.1f}% of Jacobian")
print()
print(f"  Level 2: ±0 split")
print(f"    Splits PRESERVE into positive and negative halves")
print(f"    Cannot see: per-input exact position within each half")
print(f"    Captures: {improvement_split/improvement_jac*100:.1f}% of Jacobian")
print()
print(f"  Level ∞: Exact GELU' (oracle)")
print(f"    Knows exact gate derivative at every point")
print(f"    Captures: {improvement_exact/improvement_jac*100:.1f}% of Jacobian")
print()

print(f"  The Gödel-like insight:")
print(f"    Each level contains a truth it cannot express:")
print(f"    - Level 0 can't express: 'not all channels have GELU' = 0.5'")
print(f"    - Level 1 can't express: '+0 and -0 are different'")
print(f"    - Level 2 can't express: 'each input's exact position matters'")
print(f"    - Level N can't express: the structure at level N+1")
print()
print(f"    The system is always incomplete — but the residual at each level")
print(f"    is self-similar (same φ-structure at smaller scale).")
print(f"    This IS the fractal: φ all the way down.")
print()

# The key: does the curvature = φ/2 show up in the ±0 correction?
print(f"  Curvature check:")
print(f"    GELU''(0) = √(2/π) = {np.sqrt(2/np.pi):.6f}")
print(f"    φ/2                = {PHI/2:.6f}")
print(f"    Deviation:           {abs(np.sqrt(2/np.pi) - PHI/2)/np.sqrt(2/np.pi)*100:.2f}%")
print()

# Average over seeds: what fraction of the ±0 correction comes from the curvature term?
pm_42 = PhiMap(DIM, expansion=4, gate='gelu')
pm_42.init_random(seed=42)
pm_42.fit(X_train, Y_train, n_iter=2000, lr=0.005)
h_norms_42 = np.sqrt(np.sum(pm_42.H**2, axis=1))
half_normal_42 = h_norms_42 * np.sqrt(2/np.pi)
gelu_second_42 = norm.pdf(pm_42.b) * (2 - pm_42.b**2)
correction_magnitude = np.abs(gelu_second_42 * half_normal_42)

print(f"  ±0 correction magnitude per channel:")
print(f"    Mean:   {correction_magnitude.mean():.4f}")
print(f"    Std:    {correction_magnitude.std():.4f}")
print(f"    This is the CURVATURE × SPREAD at each channel")
print(f"    = GELU''(b) × ||H_i|| × √(2/π)")
print(f"    ≈ (φ/2) × ||H_i|| × √(2/π)  at the scaffold center")
