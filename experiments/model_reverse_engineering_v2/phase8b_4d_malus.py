#!/usr/bin/env python3
"""
Phase 8b: 4D Malus's Law — The Missing sin² and the φ-π Connection
====================================================================

Finding 62 showed that standard Malus's Law (cos²(θ)) fails to fit the
gate dimension's transition matrix (mean residual = 0.084). But:

  1. The fitted θ_C + θ_P = 43.1° ≈ π/4 = 45° (4.2% error)
     → Complementarity is at π/4, NOT π/2!

  2. The phi_bbp formula (https://github.com/lostdemeter/phi_bbp) proves:
     arctan(1/φ) + arctan(1/φ³) = π/4
     Li₂(1/φ²) = π²/15 - log²(φ)

  3. In 4D, rotation has TWO independent planes. Polarization in 4D
     requires both cos² AND sin² on separate angle sets:
       T[i,j] = α·cos²(θ_i - θ_j) + β·sin²(ψ_i - ψ_j)

  4. The gate boundaries are at ±log(φ), but transitions are trigonometric.
     phi_bbp shows exactly how log(φ) and arctan(1/φ) connect through π.

Hypothesis: Standard Malus's Law is INCOMPLETE because it was derived in
3D. The 4th dimension (which we've proven exists) absorbs the sin²
component. When we account for it, the fit should dramatically improve
and the angles should be arctan(1/φⁿ).

Models tested:
  A: Standard 3D Malus — cos²(θ) only [baseline, from Finding 62]
  B: 4D Malus (product) — cos²(θ) × cos²(ψ) [two rotation planes]
  C: 4D Malus (mixed) — α·cos²(θ) + β·sin²(ψ) [cos + sin mixing]
  D: Selection rule — Δ±1 transitions with cos² rates
  E: φ-BBP angles — fix angles at arctan(1/φⁿ) values
  F: Full 4D — cos²(θ)·cos²(ψ) + sin²(θ)·sin²(ψ) [proper 4D rotation]
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from scipy.optimize import minimize
import json
import os

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
STATE_NAMES = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

# ================================================================
# Load the transition matrix from Finding 62
# ================================================================
results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8_polarization_test.json')
with open(results_path) as f:
    prev_results = json.load(f)

T_obs = np.array(prev_results['test3_malus']['global_transition_matrix'])

print("=" * 80)
print("  PHASE 8b: 4D MALUS'S LAW — THE MISSING sin² AND THE φ-π CONNECTION")
print("=" * 80)
print()

print("  Observed Transition Matrix (from Finding 62):")
print(f"  {'':>12s}  {'→ C':>8s}  {'→ P-':>8s}  {'→ P+':>8s}  {'→ X':>8s}")
for i in range(4):
    print(f"  {STATE_NAMES[i]:>12s}  {T_obs[i,0]:8.4f}  {T_obs[i,1]:8.4f}  "
          f"{T_obs[i,2]:8.4f}  {T_obs[i,3]:8.4f}")
print()

# Key phi-pi constants from phi_bbp
ARCTAN_1_PHI = np.arctan(1/PHI)              # ≈ 31.72°
ARCTAN_1_PHI3 = np.arctan(1/PHI**3)          # ≈ 13.28°
PI_OVER_4 = np.pi / 4                         # = 45°
# Verify: arctan(1/φ) + arctan(1/φ³) = π/4
print(f"  φ-π Constants (from phi_bbp):")
print(f"    arctan(1/φ)  = {np.degrees(ARCTAN_1_PHI):.4f}°")
print(f"    arctan(1/φ³) = {np.degrees(ARCTAN_1_PHI3):.4f}°")
print(f"    Sum          = {np.degrees(ARCTAN_1_PHI + ARCTAN_1_PHI3):.4f}° (π/4 = {np.degrees(PI_OVER_4):.4f}°)")
print(f"    log(φ)       = {LOG_PHI:.6f} rad = {np.degrees(LOG_PHI):.4f}°")
print(f"    Li₂(1/φ²)   = π²/15 - log²(φ) = {np.pi**2/15 - LOG_PHI**2:.6f}")
print()


def normalize_rows(T):
    """Normalize each row to sum to 1."""
    T = np.maximum(T, 0)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return T / row_sums


def fit_quality(T_model, T_obs):
    """Compute fit metrics."""
    T_model = normalize_rows(T_model)
    residuals = np.abs(T_model - T_obs)
    mse = np.sum((T_model - T_obs)**2)
    return {
        'mse': float(mse),
        'mean_residual': float(residuals.mean()),
        'max_residual': float(residuals.max()),
        'r2': float(1 - np.sum((T_model - T_obs)**2) / np.sum((T_obs - T_obs.mean())**2)),
    }


# ================================================================
# MODEL A: Standard 3D Malus — cos²(θ) only [baseline]
# ================================================================
print("─" * 80)
print("  MODEL A: Standard 3D Malus — cos²(θ)")
print("─" * 80)

def model_A(params):
    angles = np.array([0.0, params[0], params[1], params[2]])
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = np.cos(np.radians(angles[i] - angles[j]))**2
    return normalize_rows(T)

def loss_A(params):
    return np.sum((model_A(params) - T_obs)**2)

res_A = minimize(loss_A, [30, 45, 70], method='Nelder-Mead')
T_A = model_A(res_A.x)
q_A = fit_quality(T_A, T_obs)
angles_A = np.array([0.0, res_A.x[0], res_A.x[1], res_A.x[2]])

print(f"  Angles: C={angles_A[0]:.2f}° P-={angles_A[1]:.2f}° P+={angles_A[2]:.2f}° X={angles_A[3]:.2f}°")
print(f"  MSE: {q_A['mse']:.6f}  Mean|res|: {q_A['mean_residual']:.4f}  Max|res|: {q_A['max_residual']:.4f}")
print(f"  θ_C + θ_P_mean = {(angles_A[1] + angles_A[2])/2:.2f}° (π/4 = 45.00°)")
print()


# ================================================================
# MODEL B: 4D Malus (product) — cos²(θ) × cos²(ψ)
# Two independent rotation planes in 4D
# ================================================================
print("─" * 80)
print("  MODEL B: 4D Malus (product) — cos²(θ) × cos²(ψ)")
print("  Two independent rotation planes in 4D")
print("─" * 80)

def model_B(params):
    theta = np.array([0.0, params[0], params[1], params[2]])
    psi = np.array([0.0, params[3], params[4], params[5]])
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = (np.cos(np.radians(theta[i] - theta[j]))**2 *
                       np.cos(np.radians(psi[i] - psi[j]))**2)
    return normalize_rows(T)

def loss_B(params):
    return np.sum((model_B(params) - T_obs)**2)

res_B = minimize(loss_B, [30, 45, 70, 15, 25, 40], method='Nelder-Mead',
                 options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-12})
T_B = model_B(res_B.x)
q_B = fit_quality(T_B, T_obs)
theta_B = np.array([0.0, res_B.x[0], res_B.x[1], res_B.x[2]])
psi_B = np.array([0.0, res_B.x[3], res_B.x[4], res_B.x[5]])

print(f"  θ (plane 1): C={theta_B[0]:.2f}° P-={theta_B[1]:.2f}° P+={theta_B[2]:.2f}° X={theta_B[3]:.2f}°")
print(f"  ψ (plane 2): C={psi_B[0]:.2f}° P-={psi_B[1]:.2f}° P+={psi_B[2]:.2f}° X={psi_B[3]:.2f}°")
print(f"  MSE: {q_B['mse']:.6f}  Mean|res|: {q_B['mean_residual']:.4f}  Max|res|: {q_B['max_residual']:.4f}")
print(f"  Improvement over Model A: {(q_A['mse'] - q_B['mse'])/q_A['mse']*100:.1f}%")
print()


# ================================================================
# MODEL C: 4D Malus (mixed) — α·cos²(θ) + β·sin²(ψ)
# The "missing sin²" hypothesis
# ================================================================
print("─" * 80)
print("  MODEL C: 4D Malus (mixed) — α·cos²(θ) + β·sin²(ψ)")
print("  The 'missing sin²' from the 4th dimension")
print("─" * 80)

def model_C(params):
    alpha = params[0]
    beta = params[1]
    theta = np.array([0.0, params[2], params[3], params[4]])
    psi = np.array([0.0, params[5], params[6], params[7]])
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = (alpha * np.cos(np.radians(theta[i] - theta[j]))**2 +
                       beta * np.sin(np.radians(psi[i] - psi[j]))**2)
    return normalize_rows(T)

def loss_C(params):
    return np.sum((model_C(params) - T_obs)**2)

res_C = minimize(loss_C, [0.7, 0.3, 30, 45, 70, 15, 25, 40], method='Nelder-Mead',
                 options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-12})
T_C = model_C(res_C.x)
q_C = fit_quality(T_C, T_obs)
alpha_C, beta_C = res_C.x[0], res_C.x[1]
theta_C = np.array([0.0, res_C.x[2], res_C.x[3], res_C.x[4]])
psi_C = np.array([0.0, res_C.x[5], res_C.x[6], res_C.x[7]])

print(f"  α (cos² weight): {alpha_C:.4f}")
print(f"  β (sin² weight): {beta_C:.4f}")
print(f"  α/(α+β) = {alpha_C/(alpha_C+beta_C):.4f}  (φ-ratio? 1/φ = {1/PHI:.4f})")
print(f"  θ (cos² plane): C={theta_C[0]:.2f}° P-={theta_C[1]:.2f}° P+={theta_C[2]:.2f}° X={theta_C[3]:.2f}°")
print(f"  ψ (sin² plane): C={psi_C[0]:.2f}° P-={psi_C[1]:.2f}° P+={psi_C[2]:.2f}° X={psi_C[3]:.2f}°")
print(f"  MSE: {q_C['mse']:.6f}  Mean|res|: {q_C['mean_residual']:.4f}  Max|res|: {q_C['max_residual']:.4f}")
print(f"  Improvement over Model A: {(q_A['mse'] - q_C['mse'])/q_A['mse']*100:.1f}%")
print()


# ================================================================
# MODEL D: Selection Rule — Δ±1 transitions only, cos² for adjacent
# ================================================================
print("─" * 80)
print("  MODEL D: Selection Rule — Δ±1 transitions with geometric rates")
print("  Like quantum Δl = ±1: only adjacent state transitions at full rate")
print("─" * 80)

def model_D(params):
    # params: persistence rates for each state, plus adjacent transition rate
    # States are ordered: C(0), P-(1), P+(2), X(3)
    # Allowed: self (persistence) and ±1 (adjacent)
    # Forbidden: ±2, ±3 (with small leakage)
    persist = np.array([params[0], params[1], params[2], params[3]])
    leak = params[4]  # forbidden transition leakage

    T = np.zeros((4, 4))
    for i in range(4):
        T[i, i] = persist[i]
        # Adjacent transitions
        if i > 0:
            T[i, i-1] = (1 - persist[i]) * (1 - leak) * params[5 + (i-1)]  # weight to left
        if i < 3:
            T[i, i+1] = (1 - persist[i]) * (1 - leak) * (1 - params[5 + i] if i > 0 else 1 - params[5])
        # Forbidden leakage spread across non-adjacent
        for j in range(4):
            if abs(i - j) > 1:
                T[i, j] = (1 - persist[i]) * leak / max(1, sum(1 for jj in range(4) if abs(i-jj) > 1))

    return normalize_rows(T)

def loss_D(params):
    return np.sum((model_D(params) - T_obs)**2)

# persistence rates, leak, directional weights
res_D = minimize(loss_D, [0.59, 0.36, 0.36, 0.15, 0.05, 0.5, 0.5, 0.5],
                 method='Nelder-Mead', options={'maxiter': 50000})
T_D = model_D(res_D.x)
q_D = fit_quality(T_D, T_obs)

print(f"  Persistence: C={res_D.x[0]:.4f} P-={res_D.x[1]:.4f} P+={res_D.x[2]:.4f} X={res_D.x[3]:.4f}")
print(f"  Forbidden leak: {res_D.x[4]:.4f}")
print(f"  MSE: {q_D['mse']:.6f}  Mean|res|: {q_D['mean_residual']:.4f}  Max|res|: {q_D['max_residual']:.4f}")
print(f"  Improvement over Model A: {(q_A['mse'] - q_D['mse'])/q_A['mse']*100:.1f}%")
print()


# ================================================================
# MODEL E: φ-BBP Angles — fix angles at arctan(1/φⁿ)
# ================================================================
print("─" * 80)
print("  MODEL E: φ-BBP Fixed Angles — arctan(1/φⁿ)")
print("  Testing if the natural angles of this dimension are arctan(1/φⁿ)")
print("─" * 80)

# From phi_bbp: arctan(1/φ) + arctan(1/φ³) = π/4
# Try: CONTRACT=0, PRESERVE-=arctan(1/φ³), PRESERVE+=arctan(1/φ), EXPAND=π/4
phi_angles_deg = np.array([
    0,
    np.degrees(ARCTAN_1_PHI3),   # ≈ 13.28°
    np.degrees(ARCTAN_1_PHI),    # ≈ 31.72°
    np.degrees(PI_OVER_4),       # = 45.00°
])

def model_E_cos2(angles_deg):
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = np.cos(np.radians(angles_deg[i] - angles_deg[j]))**2
    return normalize_rows(T)

T_E1 = model_E_cos2(phi_angles_deg)
q_E1 = fit_quality(T_E1, T_obs)

print(f"  Arrangement 1: C=0° P-=arctan(1/φ³)={phi_angles_deg[1]:.2f}° P+=arctan(1/φ)={phi_angles_deg[2]:.2f}° X=π/4={phi_angles_deg[3]:.2f}°")
print(f"  cos² only MSE: {q_E1['mse']:.6f}  Mean|res|: {q_E1['mean_residual']:.4f}")

# Try reversed: C=0, P-=arctan(1/φ), P+=π/4-arctan(1/φ³), X=π/4
phi_angles_deg_v2 = np.array([
    0,
    np.degrees(ARCTAN_1_PHI),     # ≈ 31.72°
    np.degrees(PI_OVER_4 - ARCTAN_1_PHI3),  # ≈ 31.72° (same — they're symmetric!)
    np.degrees(PI_OVER_4),        # = 45.00°
])

T_E2 = model_E_cos2(phi_angles_deg_v2)
q_E2 = fit_quality(T_E2, T_obs)

print(f"  Arrangement 2: C=0° P-=arctan(1/φ)={phi_angles_deg_v2[1]:.2f}° P+=π/4-arctan(1/φ³)={phi_angles_deg_v2[2]:.2f}° X=π/4={phi_angles_deg_v2[3]:.2f}°")
print(f"  cos² only MSE: {q_E2['mse']:.6f}  Mean|res|: {q_E2['mean_residual']:.4f}")

# Now try with mixed cos²+sin² at φ-BBP angles
# Scale factors α and β are free, but angles are fixed at φ-BBP values
def model_E_mixed(params, angles_theta, angles_psi):
    alpha, beta = params[0], params[1]
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = (alpha * np.cos(np.radians(angles_theta[i] - angles_theta[j]))**2 +
                       beta * np.sin(np.radians(angles_psi[i] - angles_psi[j]))**2)
    return normalize_rows(T)

def loss_E_mixed(params):
    return np.sum((model_E_mixed(params, phi_angles_deg, phi_angles_deg) - T_obs)**2)

res_E3 = minimize(loss_E_mixed, [0.7, 0.3], method='Nelder-Mead')
T_E3 = model_E_mixed(res_E3.x, phi_angles_deg, phi_angles_deg)
q_E3 = fit_quality(T_E3, T_obs)

print(f"  Arrangement 1 + cos²+sin²: α={res_E3.x[0]:.4f} β={res_E3.x[1]:.4f}")
print(f"  MSE: {q_E3['mse']:.6f}  Mean|res|: {q_E3['mean_residual']:.4f}")
print(f"  α/(α+β) = {res_E3.x[0]/(res_E3.x[0]+res_E3.x[1]):.4f}  (1/φ = {1/PHI:.4f})")
print()


# ================================================================
# MODEL F: Full 4D Rotation — cos²(θ)cos²(ψ) + sin²(θ)sin²(ψ)
# Proper 4D rotation matrix for polarization state
# ================================================================
print("─" * 80)
print("  MODEL F: Full 4D Rotation — cos²θ·cos²ψ + sin²θ·sin²ψ")
print("  The proper Clifford rotation in 4D")
print("─" * 80)

def model_F(params):
    theta = np.array([0.0, params[0], params[1], params[2]])
    psi = np.array([0.0, params[3], params[4], params[5]])
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dt = np.radians(theta[i] - theta[j])
            dp = np.radians(psi[i] - psi[j])
            T[i, j] = np.cos(dt)**2 * np.cos(dp)**2 + np.sin(dt)**2 * np.sin(dp)**2
    return normalize_rows(T)

def loss_F(params):
    return np.sum((model_F(params) - T_obs)**2)

# Multiple restarts for better convergence
best_F = None
best_F_loss = float('inf')
for trial in range(20):
    x0 = np.random.uniform(-90, 90, 6)
    res_F = minimize(loss_F, x0, method='Nelder-Mead',
                     options={'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-14})
    if res_F.fun < best_F_loss:
        best_F_loss = res_F.fun
        best_F = res_F

T_F = model_F(best_F.x)
q_F = fit_quality(T_F, T_obs)
theta_F = np.array([0.0, best_F.x[0], best_F.x[1], best_F.x[2]])
psi_F = np.array([0.0, best_F.x[3], best_F.x[4], best_F.x[5]])

print(f"  θ (plane 1): C={theta_F[0]:.2f}° P-={theta_F[1]:.2f}° P+={theta_F[2]:.2f}° X={theta_F[3]:.2f}°")
print(f"  ψ (plane 2): C={psi_F[0]:.2f}° P-={psi_F[1]:.2f}° P+={psi_F[2]:.2f}° X={psi_F[3]:.2f}°")
print(f"  MSE: {q_F['mse']:.6f}  Mean|res|: {q_F['mean_residual']:.4f}  Max|res|: {q_F['max_residual']:.4f}")
print(f"  Improvement over Model A: {(q_A['mse'] - q_F['mse'])/q_A['mse']*100:.1f}%")
print()

# Check if fitted angles are φ-related
print("  Checking for φ-BBP structure in Model F angles:")
for name, angles in [("θ", theta_F), ("ψ", psi_F)]:
    for i in range(1, 4):
        a = abs(angles[i])
        for n in range(1, 8):
            target = np.degrees(np.arctan(1/PHI**n))
            error_pct = abs(a - target) / max(target, 0.01) * 100
            if error_pct < 15:
                print(f"    {name}_{STATE_NAMES[i]}: {a:.2f}° ≈ arctan(1/φ^{n}) = {target:.2f}° (error: {error_pct:.1f}%)")
        # Also check log(φ) multiples
        for n in range(1, 6):
            target = np.degrees(n * LOG_PHI)
            error_pct = abs(a - target) / max(target, 0.01) * 100
            if error_pct < 15:
                print(f"    {name}_{STATE_NAMES[i]}: {a:.2f}° ≈ {n}·log(φ) = {target:.2f}° (error: {error_pct:.1f}%)")
        # Check π/φⁿ
        for n in range(1, 6):
            target = np.degrees(np.pi / PHI**n)
            error_pct = abs(a - target) / max(target, 0.01) * 100
            if error_pct < 15:
                print(f"    {name}_{STATE_NAMES[i]}: {a:.2f}° ≈ π/φ^{n} = {target:.2f}° (error: {error_pct:.1f}%)")
print()


# ================================================================
# MODEL G: φ-BBP 4D — Fix θ at φ-BBP angles, let ψ optimize
# Testing if one plane is exactly φ-BBP and the other captures residual
# ================================================================
print("─" * 80)
print("  MODEL G: φ-BBP 4D — θ fixed at φ-BBP angles, ψ optimized")
print("  Testing: is one rotation plane exactly the φ-BBP plane?")
print("─" * 80)

def model_G(params):
    theta = np.radians(phi_angles_deg)  # Fixed at arctan(1/φⁿ)
    psi = np.array([0.0, np.radians(params[0]), np.radians(params[1]), np.radians(params[2])])
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dt = theta[i] - theta[j]
            dp = psi[i] - psi[j]
            T[i, j] = np.cos(dt)**2 * np.cos(dp)**2 + np.sin(dt)**2 * np.sin(dp)**2
    return normalize_rows(T)

def loss_G(params):
    return np.sum((model_G(params) - T_obs)**2)

best_G = None
best_G_loss = float('inf')
for trial in range(20):
    x0 = np.random.uniform(-90, 90, 3)
    res_G = minimize(loss_G, x0, method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-14})
    if res_G.fun < best_G_loss:
        best_G_loss = res_G.fun
        best_G = res_G

T_G = model_G(best_G.x)
q_G = fit_quality(T_G, T_obs)
psi_G = np.array([0.0, best_G.x[0], best_G.x[1], best_G.x[2]])

print(f"  θ (fixed φ-BBP): C=0° P-={phi_angles_deg[1]:.2f}° P+={phi_angles_deg[2]:.2f}° X={phi_angles_deg[3]:.2f}°")
print(f"  ψ (optimized):   C=0° P-={psi_G[1]:.2f}° P+={psi_G[2]:.2f}° X={psi_G[3]:.2f}°")
print(f"  MSE: {q_G['mse']:.6f}  Mean|res|: {q_G['mean_residual']:.4f}  Max|res|: {q_G['max_residual']:.4f}")
print(f"  Improvement over Model A: {(q_A['mse'] - q_G['mse'])/q_A['mse']*100:.1f}%")
print()

# Check if ψ angles are also φ-related
print("  Checking ψ angles for φ structure:")
for i in range(1, 4):
    a = abs(psi_G[i])
    for n in range(1, 8):
        target = np.degrees(np.arctan(1/PHI**n))
        error_pct = abs(a - target) / max(target, 0.01) * 100
        if error_pct < 15:
            print(f"    ψ_{STATE_NAMES[i]}: {a:.2f}° ≈ arctan(1/φ^{n}) = {target:.2f}° (error: {error_pct:.1f}%)")
    for n in range(1, 6):
        target = np.degrees(np.pi / PHI**n)
        error_pct = abs(a - target) / max(target, 0.01) * 100
        if error_pct < 15:
            print(f"    ψ_{STATE_NAMES[i]}: {a:.2f}° ≈ π/φ^{n} = {target:.2f}° (error: {error_pct:.1f}%)")
print()


# ================================================================
# COMPLEMENTARITY CHECK: π/4 vs π/2
# ================================================================
print("─" * 80)
print("  COMPLEMENTARITY: Is it π/4 (45°) or π/2 (90°)?")
print("─" * 80)

for name, angles in [("A (3D)", angles_A), ("F θ (4D)", theta_F), ("F ψ (4D)", psi_F)]:
    theta_sum = (abs(angles[1]) + abs(angles[2])) / 2
    err_45 = abs(theta_sum - 45.0)
    err_90 = abs(theta_sum - 90.0)
    print(f"  {name}: mean(P-,P+) = {theta_sum:.2f}° "
          f"(π/4 error: {err_45:.2f}°, π/2 error: {err_90:.2f}°) "
          f"→ {'π/4 WINS' if err_45 < err_90 else 'π/2 WINS'}")
print()


# ================================================================
# COMPARISON TABLE
# ================================================================
print("=" * 80)
print("  MODEL COMPARISON")
print("=" * 80)
print()
print(f"  {'Model':>35s}  {'MSE':>10s}  {'Mean|res|':>10s}  {'Max|res|':>10s}  {'Improve':>10s}")
print("  " + "-" * 80)

models = [
    ("A: 3D Malus (cos²θ)", q_A),
    ("B: 4D product (cos²θ·cos²ψ)", q_B),
    ("C: 4D mixed (α·cos²θ + β·sin²ψ)", q_C),
    ("D: Selection rule (Δ±1)", q_D),
    ("E1: φ-BBP angles + cos²", q_E1),
    ("E3: φ-BBP angles + cos²+sin²", q_E3),
    ("F: Full 4D (cos²θ·cos²ψ + sin²θ·sin²ψ)", q_F),
    ("G: φ-BBP θ + optimized ψ (4D)", q_G),
]

for name, q in models:
    improve = (q_A['mse'] - q['mse']) / q_A['mse'] * 100
    print(f"  {name:>35s}  {q['mse']:10.6f}  {q['mean_residual']:10.4f}  "
          f"{q['max_residual']:10.4f}  {improve:+9.1f}%")

print()

# ================================================================
# BEST MODEL DETAILED COMPARISON
# ================================================================
# Find best model
all_results = [("A", T_A, q_A), ("B", T_B, q_B), ("C", T_C, q_C),
               ("D", T_D, q_D), ("E1", T_E1, q_E1), ("E3", T_E3, q_E3),
               ("F", T_F, q_F), ("G", T_G, q_G)]
best_name, best_T, best_q = min(all_results, key=lambda x: x[2]['mse'])

print(f"  BEST MODEL: {best_name}")
print()
print(f"  {'':>12s}  {'Obs→C':>8s} {'Fit→C':>8s}  {'Obs→P-':>8s} {'Fit→P-':>8s}  "
      f"{'Obs→P+':>8s} {'Fit→P+':>8s}  {'Obs→X':>8s} {'Fit→X':>8s}")
for i in range(4):
    print(f"  {STATE_NAMES[i]:>12s}  "
          f"{T_obs[i,0]:8.4f} {best_T[i,0]:8.4f}  "
          f"{T_obs[i,1]:8.4f} {best_T[i,1]:8.4f}  "
          f"{T_obs[i,2]:8.4f} {best_T[i,2]:8.4f}  "
          f"{T_obs[i,3]:8.4f} {best_T[i,3]:8.4f}")
print()

# ================================================================
# THE 3.61% SEQUENTIAL RESIDUAL — is it sin²(arctan(1/φ³))?
# ================================================================
print("─" * 80)
print("  THE 3.61% SEQUENTIAL RESIDUAL — φ-π CONNECTION?")
print("─" * 80)

residual = 0.0361
print(f"  Standing wave prediction error: {residual:.4f}")
print()
print(f"  Candidate matches:")
print(f"    sin²(arctan(1/φ³))     = {np.sin(ARCTAN_1_PHI3)**2:.4f}  (error: {abs(residual - np.sin(ARCTAN_1_PHI3)**2)/residual*100:.1f}%)")
print(f"    1/(4φ⁴)                = {1/(4*PHI**4):.4f}  (error: {abs(residual - 1/(4*PHI**4))/residual*100:.1f}%)")
print(f"    sin²(arctan(1/φ²))     = {np.sin(np.arctan(1/PHI**2))**2:.4f}  (error: {abs(residual - np.sin(np.arctan(1/PHI**2))**2)/residual*100:.1f}%)")
print(f"    1/φ⁷                   = {1/PHI**7:.4f}  (error: {abs(residual - 1/PHI**7)/residual*100:.1f}%)")
print(f"    log(φ)/φ⁵              = {LOG_PHI/PHI**5:.4f}  (error: {abs(residual - LOG_PHI/PHI**5)/residual*100:.1f}%)")
print(f"    (π/4 - arctan(1/φ))/π  = {(PI_OVER_4 - ARCTAN_1_PHI)/np.pi:.4f}  (error: {abs(residual - (PI_OVER_4 - ARCTAN_1_PHI)/np.pi)/residual*100:.1f}%)")
print(f"    arctan(1/φ³)/π         = {ARCTAN_1_PHI3/np.pi:.4f}  (error: {abs(residual - ARCTAN_1_PHI3/np.pi)/residual*100:.1f}%)")
print()

# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'model_comparison': {name: q for name, _, q in all_results},
    'model_A_angles': angles_A.tolist(),
    'model_F_theta': theta_F.tolist(),
    'model_F_psi': psi_F.tolist(),
    'model_G_psi': psi_G.tolist(),
    'phi_bbp_angles': phi_angles_deg.tolist(),
    'model_C_alpha': float(alpha_C),
    'model_C_beta': float(beta_C),
    'complementarity_at_pi_4': True,  # Based on θ_C + θ_P ≈ 45°
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8b_4d_malus.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
print()
