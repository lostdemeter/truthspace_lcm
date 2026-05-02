"""
Test: Close the scaffold-Jacobian gap geometrically.

The scaffold (½R@H) assumes GELU'(z) = 0.5 everywhere.
The Jacobian uses E[GELU'(z)] from calibration data.
The gap is 2.82%.

Hypothesis: The bias b shifts each channel's resting point away from
the φ^0 center. GELU'(b) gives the channel's intrinsic gate derivative
at zero input. Using R @ diag(GELU'(b)) @ H should close the gap
WITHOUT calibration data — because b is part of the structure, not data.

This would prove: the Jacobian's advantage was always geometric
(the bias defines each channel's φ-level), not statistical (averaging
over inputs "denoises").
"""
import numpy as np
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI, _standard_gelu_derivative, _standard_gelu

np.random.seed(42)

LOG_PHI = np.log(PHI)

DIM = 32
N_TRAIN = 500
N_TEST = 200
N_CAL = 100

# Target function
W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(N_CAL, DIM).astype(np.float32)

print('=' * 70)
print('Can We Close the Gap Geometrically?')
print('=' * 70)
print()

results = []

for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    # 1. Nonlinear (full GELU)
    Y_nl = pm.lookup(X_test)
    rmse_nl = np.sqrt(np.mean((Y_nl - Y_test)**2))

    # 2. Mean Jacobian (statistical, needs calibration)
    pm.calibrate(X_cal)
    Y_jac = pm.default(X_test)
    rmse_jac = np.sqrt(np.mean((Y_jac - Y_test)**2))

    # 3. Pure scaffold: (1/2) R @ H  (geometric, no data)
    S_half = 0.5 * pm.R @ pm.H
    b_half = pm.R @ (pm.b * 0.5) + pm.b_out
    Y_half = X_test @ S_half.T + b_half
    rmse_half = np.sqrt(np.mean((Y_half - Y_test)**2))

    # 4. Bias-corrected scaffold: R @ diag(GELU'(b)) @ H  (geometric, no data)
    # The bias b is the pre-GELU value at zero input.
    # GELU'(b) is the gate derivative at the channel's resting point.
    gate_at_bias = _standard_gelu_derivative(pm.b)  # [E]
    S_bias = (pm.R * gate_at_bias[np.newaxis, :]) @ pm.H   # [D, D]
    b_bias = pm.R @ _standard_gelu(pm.b) + pm.b_out        # [D]
    Y_bias = X_test @ S_bias.T + b_bias
    rmse_bias = np.sqrt(np.mean((Y_bias - Y_test)**2))

    # 5. How close is GELU'(b) to E[GELU'(z)]?
    z_cal = X_cal @ pm.H.T + pm.b
    mean_deriv = _standard_gelu_derivative(z_cal).mean(axis=0)
    bias_deriv = _standard_gelu_derivative(pm.b)
    deriv_corr = np.corrcoef(mean_deriv, bias_deriv)[0, 1]
    deriv_diff = np.mean(np.abs(mean_deriv - bias_deriv))

    results.append({
        'seed': seed,
        'rmse_nl': rmse_nl,
        'rmse_jac': rmse_jac,
        'rmse_half': rmse_half,
        'rmse_bias': rmse_bias,
        'deriv_corr': deriv_corr,
        'deriv_diff': deriv_diff,
    })

# Print results
print(f"  {'Seed':<6} {'Nonlinear':<11} {'Jacobian':<11} {'½R@H':<11} {'GELU`(b)R@H':<13} {'GELU`(b)↔E[GELU`]'}")
print(f"  " + "-" * 75)
for r in results:
    print(f"  {r['seed']:<6} {r['rmse_nl']:<11.4f} {r['rmse_jac']:<11.4f} "
          f"{r['rmse_half']:<11.4f} {r['rmse_bias']:<13.4f} "
          f"corr={r['deriv_corr']:.3f} Δ={r['deriv_diff']:.4f}")

print()
print("  Legend:")
print("    Nonlinear:    Full GELU (baseline)")
print("    Jacobian:     R @ diag(E[GELU'(z)]) @ H (statistical, 100 calibration samples)")
print("    ½R@H:         Pure scaffold (geometric, 0 samples)")
print("    GELU'(b)R@H:  Bias-corrected scaffold (geometric, 0 samples)")

# Summary
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()

avg_nl = np.mean([r['rmse_nl'] for r in results])
avg_jac = np.mean([r['rmse_jac'] for r in results])
avg_half = np.mean([r['rmse_half'] for r in results])
avg_bias = np.mean([r['rmse_bias'] for r in results])

print(f"  Average RMSE across 5 seeds:")
print(f"    Nonlinear (full GELU):         {avg_nl:.4f}")
print(f"    Mean Jacobian (100 cal):       {avg_jac:.4f}  ({(avg_jac-avg_nl)/avg_nl*100:+.2f}% vs GELU)")
print(f"    Scaffold ½R@H (0 cal):         {avg_half:.4f}  ({(avg_half-avg_nl)/avg_nl*100:+.2f}% vs GELU)")
print(f"    Bias-corrected GELU'(b) (0 cal): {avg_bias:.4f}  ({(avg_bias-avg_nl)/avg_nl*100:+.2f}% vs GELU)")
print()

gap_half_to_jac = (avg_half - avg_jac) / avg_jac * 100
gap_bias_to_jac = (avg_bias - avg_jac) / avg_jac * 100
print(f"  Gap between geometric and statistical:")
print(f"    ½R@H → Jacobian:         {gap_half_to_jac:+.2f}%")
print(f"    GELU'(b)R@H → Jacobian:  {gap_bias_to_jac:+.2f}%")
print(f"    Gap closed:              {(1 - abs(gap_bias_to_jac)/abs(gap_half_to_jac))*100:.1f}%")
print()

if avg_bias < avg_half:
    if avg_bias <= avg_jac * 1.01:
        print(f"  → GELU'(b) CLOSES THE GAP — pure geometry matches statistics")
        print(f"    The Jacobian's advantage was always geometric: the bias defines")
        print(f"    each channel's φ-level, and GELU'(b) reads that level directly.")
    else:
        print(f"  → GELU'(b) PARTIALLY CLOSES THE GAP")
        print(f"    The bias captures the per-channel φ-level shift,")
        print(f"    but input statistics contribute a residual correction.")
else:
    print(f"  → GELU'(b) does NOT help — the correction is data-dependent")


# Additional: what if bias IS the mean input?
# Check if the bias is doing what the mean Jacobian does
print()
print('=' * 70)
print('DETAIL: What Does the Bias Encode?')
print('=' * 70)
print()

for seed in [42]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    b = pm.b
    print(f"  Bias statistics:")
    print(f"    Mean:  {b.mean():.4f}")
    print(f"    Std:   {b.std():.4f}")
    print(f"    Min:   {b.min():.4f}")
    print(f"    Max:   {b.max():.4f}")
    print(f"    % negative: {(b < 0).mean()*100:.1f}%")
    print()

    # What φ-region does each bias put its channel in?
    expand = (b > LOG_PHI).sum()
    preserve = ((b >= -LOG_PHI) & (b <= LOG_PHI)).sum()
    contract = (b < -LOG_PHI).sum()
    print(f"  Bias φ-region classification:")
    print(f"    EXPAND   (b > log(φ)):   {expand}/{len(b)} ({expand/len(b)*100:.1f}%)")
    print(f"    PRESERVE (|b| ≤ log(φ)): {preserve}/{len(b)} ({preserve/len(b)*100:.1f}%)")
    print(f"    CONTRACT (b < -log(φ)):  {contract}/{len(b)} ({contract/len(b)*100:.1f}%)")
    print()

    # GELU'(b) for each region
    gelu_deriv_b = _standard_gelu_derivative(b)
    print(f"  GELU'(b) by region:")
    if expand > 0:
        print(f"    EXPAND:   mean={gelu_deriv_b[b > LOG_PHI].mean():.4f}")
    print(f"    PRESERVE: mean={gelu_deriv_b[(b >= -LOG_PHI) & (b <= LOG_PHI)].mean():.4f}")
    if contract > 0:
        print(f"    CONTRACT: mean={gelu_deriv_b[b < -LOG_PHI].mean():.4f}")
    print()
    print(f"  The bias defines each channel's RESTING φ-level:")
    print(f"  - Channels with b ≈ 0: resting at φ^0 (GELU' ≈ 0.5)")
    print(f"  - Channels with b >> 0: resting at EXPAND (GELU' → 1)")
    print(f"  - Channels with b << 0: resting at CONTRACT (GELU' → 0)")
    print(f"  - The bias IS the channel's default position in φ-space")
