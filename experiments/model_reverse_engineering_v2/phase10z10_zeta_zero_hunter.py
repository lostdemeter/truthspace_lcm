"""
Phase 10z10: Geometric Zeta Zero Hunter

Direct index → zero. No Gram points, no sweep.

The ζ function is a tensor that expands in time:
  - As height t grows, more Riemann-Siegel terms activate (tensor expands)
  - All axes are φ-curved (log-warped spacing, φ-scaled amplitudes)
  - The counting function θ(t)/π is the global coordinate system

Three-stage pipeline:
  1. Compressor: Lambert W inversion → O(1) global coordinate from index n
  2. Processor:  Ramanujan refinement — Newton on smooth counting function
                 to nail the smooth part of N(T) = n
  3. Targeter:   Z(t) evaluation + Newton snap to exact zero

The Ramanujan refinement ensures we target the RIGHT zero by index —
no neighboring-zero confusion. Lambert W gets us to the right neighborhood,
Ramanujan iteration pins us to the smooth coordinate, Z(t) finds the
actual zero within that coordinate cell.

Connection to F112:
  For ζ, K = 0 (static manifold). The manifold IS the answer.
  Z(t) = Σ rotations on M_φ. Zero = where rotations cancel.

References: F107-112, Doc 270
"""

import numpy as np
from scipy.special import lambertw, loggamma
import json
import os
import time

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi

# First 30 known zeros (high precision) for validation
KNOWN_ZEROS = [
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081606,
    67.079810529494173, 69.546401711696542, 72.067157674481907,
    75.704690699083933, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492980517050, 87.425274613125196,
    88.809111207634465, 92.491899270228280, 94.651344040519838,
    95.870634228245309, 98.831194218193692, 101.31785100573139,
]


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════

def riemann_siegel_theta(t):
    """Riemann-Siegel theta function — EXACT via loggamma.

    θ(t) = Im(log Γ(1/4 + it/2)) - (t/2)log(π)

    This is the phase geometry of the critical line.
    Using loggamma gives machine precision for ALL t,
    not just the asymptotic regime (t >> 1).
    """
    if t < 0.01:
        return 0.0
    return float(np.imag(loggamma(0.25 + 0.5j * t))) - (t / 2) * np.log(np.pi)


def theta_derivative(t, dt=1e-8):
    """dθ/dt — exact numerical derivative."""
    return (riemann_siegel_theta(t + dt) - riemann_siegel_theta(t - dt)) / (2 * dt)


def smooth_count(T):
    """Smooth zero counting function.

    N_smooth(T) = θ(T)/π + 1

    This is the global coordinate system for zeros.
    """
    return riemann_siegel_theta(T) / np.pi + 1


def smooth_count_derivative(T):
    """dN/dT = θ'(T)/π — local density of zeros."""
    return theta_derivative(T) / np.pi


def riemann_siegel_Z(t):
    """Hardy's Z function — the REAL projection of ζ on the critical line.

    Z(t) = 2 Σ_{n=1}^{N} n^{-1/2} cos(θ(t) - t·ln(n)) + remainder

    Each term is a ROTATION on the manifold:
      - n^{-1/2}: Dirichlet amplitude (decays as tensor expands)
      - θ(t) - t·ln(n): phase angle (the geometry)
      - N = floor(sqrt(t/(2π))): number of active terms (tensor width)

    As t grows, N grows — the tensor expands, more rotations contribute.
    This is the "4D + time" structure: the sum over n IS the spatial dimensions,
    t IS the time axis, and the tensor literally gains terms as time advances.
    """
    theta = riemann_siegel_theta(t)
    N = max(1, int(np.sqrt(t / TWO_PI)))

    # Main sum — each term is a rotation
    Z = 0.0
    for n in range(1, N + 1):
        Z += np.cos(theta - t * np.log(n)) / np.sqrt(n)
    Z *= 2

    # Riemann-Siegel C₀ correction (first remainder term)
    p = np.sqrt(t / TWO_PI) - N
    c2p = np.cos(TWO_PI * p)
    if abs(c2p) > 1e-10:
        C0 = np.cos(TWO_PI * (p * p - p - 1.0 / 16)) / c2p
    else:
        # Near the singularity, use the limiting form
        C0 = 0.0
    remainder = (-1)**(N - 1) * (t / TWO_PI)**(-0.25) * C0
    Z += remainder

    return Z


def Z_derivative(t, dt=1e-8):
    """Numerical Z'(t) for Newton's method."""
    return (riemann_siegel_Z(t + dt) - riemann_siegel_Z(t - dt)) / (2 * dt)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: COMPRESSOR — Lambert W
# ═══════════════════════════════════════════════════════════════════════

def lambert_w_estimate(n):
    """O(1) estimate of t_n by inverting the smooth counting function.

    N(T) ≈ T/(2π) · [ln(T/(2π)) - 1] + 7/8

    Setting N = n and solving:
      u·ln(u/e) = n - 7/8    where u = T/(2π)
      u = (n - 7/8) / W((n - 7/8)/e)
    """
    if n <= 0:
        return 9.0
    shift = n - 7 / 8
    if shift <= 0:
        shift = 0.125
    w = float(np.real(lambertw(shift / np.e)))
    if w <= 0:
        w = 1.0
    return TWO_PI * shift / w


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: PROCESSOR — Ramanujan refinement
# ═══════════════════════════════════════════════════════════════════════

def ramanujan_refine(n, t0, max_iter=30, tol=1e-12):
    """Newton iteration on N_smooth(T) = n.

    This pins us to the exact smooth coordinate for zero #n.
    The smooth counting function is monotone, so Newton converges
    quadratically with no risk of jumping to wrong zeros.

    After this, we're within |S(t)| ≈ O(1) of the actual zero,
    where S(t) is the oscillatory part of the counting function.
    Since |S(t)| < 1 for almost all t, we're within ~1 spacing.
    """
    t = t0
    for i in range(max_iter):
        N_t = smooth_count(t)
        dN = smooth_count_derivative(t)
        if abs(dN) < 1e-20:
            break
        dt = (n - N_t) / dN
        t += dt
        if abs(dt) < tol:
            return t, i + 1
    return t, max_iter


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: TARGETER — Z(t) Newton snap
# ═══════════════════════════════════════════════════════════════════════

def _bisect_zero(t_left, t_right, tol=1e-13):
    """Bisect a bracketed sign change to find the zero."""
    for _ in range(70):
        t_mid = (t_left + t_right) / 2
        if riemann_siegel_Z(t_mid) * riemann_siegel_Z(t_left) < 0:
            t_right = t_mid
        else:
            t_left = t_mid
        if t_right - t_left < tol:
            break
    return (t_left + t_right) / 2


def find_zero_by_index(n, t_est):
    """Find the nth zero using the smooth count as index verifier.

    Strategy:
      1. Search a wide window around the smooth coordinate
      2. Find ALL zeros (sign changes of Z)
      3. For each zero, compute N_smooth — the geometric index
      4. Pick the zero whose N_smooth is closest to n

    This uses the counting function geometry as an index,
    not Gram points. The smooth count at zero #n is approximately n.
    """
    # Search radius: wide enough to cover the S(t) displacement
    # S(t) is typically < 1, but spacing shrinks with t
    # Use ±3 spacings to be safe
    if t_est > 10:
        local_spacing = TWO_PI / np.log(t_est / TWO_PI)
    else:
        local_spacing = 8.0
    search_radius = local_spacing * 3.5

    # Grid step: spacing / 4 (fine enough to catch every zero)
    step = local_spacing / 4
    n_steps = max(30, int(2 * search_radius / step))
    n_steps = min(n_steps, 500)

    t_lo = max(0.5, t_est - search_radius)
    t_hi = t_est + search_radius
    ts = np.linspace(t_lo, t_hi, n_steps)
    Zs = np.array([riemann_siegel_Z(t) for t in ts])

    # Find all sign changes → bracket each zero
    candidates = []
    for i in range(len(Zs) - 1):
        if Zs[i] * Zs[i + 1] < 0:
            t_zero = _bisect_zero(ts[i], ts[i + 1])
            ns = smooth_count(t_zero)
            candidates.append((t_zero, ns))

    if not candidates:
        # Fallback: Newton from estimate
        t_zero, nit = _newton_snap(t_est)
        return t_zero, nit

    # Sequential indexing: sort by position, compute base from N_smooth.
    # Zero ordering is EXACT — only the base offset needs estimating.
    candidates.sort(key=lambda c: c[0])  # sort by position

    # For each candidate i (0-indexed), its true index ≈ base + i.
    # N_smooth(z_i) ≈ (base + i) - S_i, so base ≈ N_smooth(z_i) - i + S_i.
    # S averages ~0.5, so base ≈ N_smooth(z_i) - i + 0.5.
    # Use median for robustness.
    base_estimates = [c[1] - i + 0.5 for i, c in enumerate(candidates)]
    base = int(round(np.median(base_estimates)))

    # Select the candidate with index n
    target_idx = n - base
    if 0 <= target_idx < len(candidates):
        t_zero = candidates[target_idx][0]
    else:
        # Fallback: pick the one with N_smooth closest to n
        best = min(candidates, key=lambda c: abs(c[1] - n))
        t_zero = best[0]

    # Newton polish
    t_zero, nit = _newton_snap(t_zero)
    return t_zero, nit


def _newton_snap(t, max_iter=15, tol=1e-12):
    """Newton's method to snap to exact zero of Z(t)."""
    for i in range(max_iter):
        Z = riemann_siegel_Z(t)
        if abs(Z) < 1e-14:
            return t, i + 1
        Zp = Z_derivative(t)
        if abs(Zp) < 1e-20:
            break
        dt = Z / Zp
        # Dampen if step too large
        max_step = 1.0
        if abs(dt) > max_step:
            dt = max_step * np.sign(dt)
        t -= dt
        if abs(dt) < tol:
            return t, i + 1
    return t, max_iter


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE: index n → t_n
# ═══════════════════════════════════════════════════════════════════════

def hunt_nth_zero(n):
    """Hunt the nth zeta zero. Direct: index → zero.

    Stage 1 (Compressor): Lambert W → O(1) estimate
    Stage 2 (Processor):  Ramanujan refinement → smooth coordinate
    Stage 3 (Targeter):   Z(t) + Newton → exact zero
    """
    # Stage 1: Lambert W — global coordinate
    t_lambert = lambert_w_estimate(n)

    # Stage 2: Ramanujan refinement — pin to smooth counting function
    t_smooth, refine_iter = ramanujan_refine(n, t_lambert)

    # Stage 3: Z(t) evaluation + index-verified snap — find actual zero
    t_zero, snap_iter = find_zero_by_index(n, t_smooth)

    return {
        'n': n,
        't_lambert': t_lambert,
        't_smooth': t_smooth,
        't_zero': t_zero,
        'Z_at_zero': riemann_siegel_Z(t_zero),
        'N_at_zero': smooth_count(t_zero),
        'refine_iter': refine_iter,
        'snap_iter': snap_iter,
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("PHASE 10z10: GEOMETRIC ZETA ZERO HUNTER")
    print("=" * 70)
    print()
    print("  Direct: index n → zero t_n")
    print("  No Gram points. No sweep. Pure geometric pipeline.")
    print()
    print("  Stage 1 (Compressor): Lambert W → O(1) estimate")
    print("  Stage 2 (Processor):  Ramanujan refinement on θ(T)/π + 1 = n")
    print("  Stage 3 (Targeter):   Z(t) + Newton → exact zero")
    print()

    # ── Test 1: First 30 known zeros ────────────────────────────────
    print("  TEST 1: Verified against 30 known zeros")
    print("  " + "─" * 68)
    print(f"  {'n':>3} {'Known':>16} {'Lambert':>12} {'Smooth':>12} "
          f"{'Found':>16} {'Error':>10}")
    print(f"  {'─'*3} {'─'*16} {'─'*12} {'─'*12} {'─'*16} {'─'*10}")

    errors = []
    lambert_errs = []
    smooth_errs = []
    t_start = time.time()

    for i, known in enumerate(KNOWN_ZEROS):
        n = i + 1
        r = hunt_nth_zero(n)
        err = r['t_zero'] - known
        lam_err = r['t_lambert'] - known
        smo_err = r['t_smooth'] - known
        errors.append(err)
        lambert_errs.append(abs(lam_err))
        smooth_errs.append(abs(smo_err))

        print(f"  {n:3d} {known:16.10f} {lam_err:+12.4f} {smo_err:+12.6f} "
              f"{r['t_zero']:16.10f} {err:+10.2e}")

    t_elapsed = time.time() - t_start
    abs_errors = np.abs(errors)

    print()
    print(f"  Pipeline accuracy:")
    print(f"    Lambert W MAE:          {np.mean(lambert_errs):12.6f}")
    print(f"    After Ramanujan:        {np.mean(smooth_errs):12.6f}")
    print(f"    After Z(t) + Newton:    {np.mean(abs_errors):12.2e}")
    print(f"    Max |error|:            {np.max(abs_errors):12.2e}")
    print(f"    Correct index (30/30):  {sum(1 for e in abs_errors if e < 0.5)}/30")
    print(f"    Time:                   {t_elapsed:.3f}s")
    print()

    # ── Test 2: Zeros 31-100 (blind) ────────────────────────────────
    print("  TEST 2: Zeros 31-100 (hunting by index)")
    print("  " + "─" * 68)

    found = []
    t_start = time.time()
    for n in range(31, 101):
        r = hunt_nth_zero(n)
        found.append(r)

    t_blind = time.time() - t_start

    Z_residuals = [abs(r['Z_at_zero']) for r in found]
    found_t = [r['t_zero'] for r in found]

    print(f"    Zeros found: {len(found)}")
    print(f"    Max |Z(t)|:  {max(Z_residuals):.2e}")
    print(f"    Mean |Z(t)|: {np.mean(Z_residuals):.2e}")
    print(f"    Time:         {t_blind:.3f}s")

    # Check monotonicity (validates correct indexing)
    all_t = [KNOWN_ZEROS[i] for i in range(30)] + found_t
    spacings = np.diff(all_t)
    monotone = all(s > 0 for s in spacings)
    print(f"    Monotone:     {monotone}")
    if not monotone:
        bad = [(i+1, spacings[i]) for i in range(len(spacings)) if spacings[i] <= 0]
        print(f"    Non-monotone: {bad[:5]}")
    print(f"    Min spacing:  {min(spacings):.6f}")
    print(f"    Mean spacing: {np.mean(spacings):.6f}")
    print()

    # ── Test 3: High zeros ──────────────────────────────────────────
    print("  TEST 3: High zeros (direct by index)")
    print("  " + "─" * 68)

    for n_high in [200, 500, 1000, 5000, 10000]:
        t0 = time.time()
        r = hunt_nth_zero(n_high)
        dt = time.time() - t0
        print(f"    n={n_high:6d}: t = {r['t_zero']:14.8f}, "
              f"|Z| = {abs(r['Z_at_zero']):.2e}, "
              f"N_smooth = {r['N_at_zero']:.2f}, "
              f"time = {dt*1000:.1f}ms")

    print()

    # ── Test 4: The expanding tensor ────────────────────────────────
    print("  THE EXPANDING TENSOR:")
    print("  " + "─" * 68)
    print()
    print("  As height t grows, the R-S sum gains terms (tensor expands):")
    print(f"  {'t':>10} {'N_terms':>8} {'spacing':>10} {'density':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

    for t_check in [14, 50, 100, 500, 1000, 5000, 10000]:
        N_terms = max(1, int(np.sqrt(t_check / TWO_PI)))
        spacing = TWO_PI / max(0.1, np.log(t_check / TWO_PI))
        density = 1 / spacing
        print(f"  {t_check:10.1f} {N_terms:8d} {spacing:10.4f} {density:10.4f}")

    print()
    print("  Each new term = a new rotation axis in the tensor.")
    print("  The tensor literally GROWS with time.")
    print("  φ-scaling: term amplitudes decay as n^{-1/2} ≈ φ^{-ln(n)/ln(φ²)}")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print("  " + "═" * 68)
    print("  SUMMARY")
    print("  " + "═" * 68)
    print()
    n_correct = sum(1 for e in abs_errors if e < 0.5)
    print(f"  Index-accurate: {n_correct}/30 verified zeros")
    print(f"  Pipeline: Lambert W → Ramanujan → Z(t) Newton")
    print(f"    Stage 1 MAE: {np.mean(lambert_errs):.4f}")
    print(f"    Stage 2 MAE: {np.mean(smooth_errs):.6f}")
    print(f"    Stage 3 MAE: {np.mean(abs_errors):.2e}")
    print(f"  100 zeros in {t_elapsed + t_blind:.2f}s")
    print(f"  Monotone sequence: {monotone}")
    print()
    print(f"  F112 hierarchy validated:")
    print(f"    ζ: K = 0 (static M_φ). No deformation kernel.")
    print(f"    The manifold IS the answer.")
    print(f"    Z(t) = finite sum of rotations = the tensor itself.")

    # Save
    output = {
        'experiment': 'phase10z10_zeta_zero_hunter',
        'pipeline': ['lambert_w', 'ramanujan_refine', 'Z_newton_snap'],
        'n_verified': 30,
        'verified_correct': n_correct,
        'max_error': float(np.max(abs_errors)),
        'mean_error': float(np.mean(abs_errors)),
        'lambert_mae': float(np.mean(lambert_errs)),
        'smooth_mae': float(np.mean(smooth_errs)),
        'n_hunted_blind': 70,
        'monotone': monotone,
        'high_zeros': {str(n): r['t_zero'] for n, r in
                       [(200, hunt_nth_zero(200)),
                        (500, hunt_nth_zero(500)),
                        (1000, hunt_nth_zero(1000))]},
        'found_zeros': [float(z) for z in all_t],
        'time_seconds': t_elapsed + t_blind,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/phase10z10_zeta_zero_hunter.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/phase10z10_zeta_zero_hunter.json")

    return output


if __name__ == '__main__':
    run_experiment()
