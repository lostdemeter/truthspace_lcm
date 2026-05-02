#!/usr/bin/env python3
"""
Pinned Threshold Learning v3: Precision Breakpoint Detection

v2 finding: Pure hinge decomposition (96.2%) beats ridge regression (90.1%).
Ridge regression HURTS because it fits smooth gate approximations instead
of the true piecewise-linear structure. The analytical formula is better.

The remaining 3.8% error comes from ONE source: imprecise breakpoint locations.
When training points don't tightly bracket a transition, the detected position
is off, which corrupts the amplitude estimate.

v3 attacks breakpoint precision from three angles:

1. POSITION REFINEMENT: After hinge detection, search nearby positions for
   each breakpoint to minimize training reconstruction error.

2. RE-ESTIMATION: At refined positions, re-compute amplitudes from the data
   using the correct analytical formula.

3. GREEDY SEARCH: Skip hinge detection entirely — enumerate candidate
   breakpoint positions and greedily add the one that explains the most
   training residual. O(n_bp * n_candidates * n_train).

4. HIGH-S STEPS: Use s=20+ for step transitions (sharp boundaries) while
   keeping s=φ² for ramps (smooth transitions).
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PHI = (1 + np.sqrt(5)) / 2
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)
C_GEOMETRIC = (4 - np.pi) / (6 * np.pi)


def ideal_gate(x):
    x = np.asarray(x, dtype=np.float64)
    f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))


def gate_step(x, threshold, s):
    """Smooth step: ~1 for x > threshold, ~0 for x < threshold."""
    return (ideal_gate(s * (x - (threshold - 0.5))) -
            ideal_gate(s * (x - (threshold + 0.5)))) / s


def gate_ramp(x, threshold, s):
    """Smooth ramp: ~(x-threshold) for x > threshold, ~0 for x < threshold."""
    return ideal_gate(s * (x - threshold)) / s


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def fn_tolower(x):
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    result[(x >= 65) & (x <= 90)] += 32
    return result

def fn_secret(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < 50, 2*x, 100 - x)

def fn_rot13(x):
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    result[(x >= 65) & (x <= 77)] += 13
    result[(x >= 78) & (x <= 90)] -= 13
    result[(x >= 97) & (x <= 109)] += 13
    result[(x >= 110) & (x <= 122)] -= 13
    return result

def fn_abs_centered(x):
    return np.abs(np.asarray(x, dtype=np.float64) - 64)

def fn_sawtooth(x):
    return np.asarray(x, dtype=np.float64) % 32

def fn_clamp(x):
    return np.clip(np.asarray(x, dtype=np.float64), 30, 100)

def fn_relu_shifted(x):
    return np.maximum(0, np.asarray(x, dtype=np.float64) - 40)

def fn_staircase(x):
    return np.floor(np.asarray(x, dtype=np.float64) / 16) * 16


# ============================================================================
# CORE: Segment Detection (from v2)
# ============================================================================

def detect_segments(xs, rs, slope_tol=0.15):
    """Detect contiguous segments of approximately constant slope."""
    if len(xs) < 2:
        return [(0, len(xs)-1, 0, rs[0] if len(rs) > 0 else 0)]

    dx = np.diff(xs)
    dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)

    segments = []
    seg_start = 0
    seg_slopes = [slopes[0]]

    for i in range(1, len(slopes)):
        current_slope = np.median(seg_slopes)
        tol = max(slope_tol, abs(current_slope) * 0.15)

        if abs(slopes[i] - current_slope) > tol:
            seg_xs = xs[seg_start:i+1]
            seg_rs = rs[seg_start:i+1]
            if len(seg_xs) >= 2:
                slope = np.polyfit(seg_xs, seg_rs, 1)[0]
            else:
                slope = current_slope
            intercept = np.median(seg_rs - slope * seg_xs)
            segments.append((seg_start, i, slope, intercept))
            seg_start = i
            seg_slopes = [slopes[i]]
        else:
            seg_slopes.append(slopes[i])

    seg_xs = xs[seg_start:]
    seg_rs = rs[seg_start:]
    if len(seg_xs) >= 2:
        slope = np.polyfit(seg_xs, seg_rs, 1)[0]
    else:
        slope = np.median(seg_slopes)
    intercept = np.median(seg_rs - slope * seg_xs)
    segments.append((seg_start, len(xs)-1, slope, intercept))

    return segments


def classify_transition(prev_slope, prev_intercept, curr_slope, curr_intercept, bp):
    """Classify a transition. Returns list of ('step'|'ramp', bp, value)."""
    primitives = []
    delta_slope = curr_slope - prev_slope
    r_prev = prev_slope * bp + prev_intercept
    r_curr = curr_slope * bp + curr_intercept
    jump = r_curr - r_prev

    if abs(jump) > 0.3:
        primitives.append(('step', bp, jump))
    if abs(delta_slope) > 0.03:
        primitives.append(('ramp', bp, delta_slope))

    return primitives


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

def evaluate_primitives(x_vals, base_slope, base_intercept, primitives, s_step, s_ramp):
    """Evaluate a set of primitives at given x values.
    
    primitives: list of ('step'|'ramp', position, amplitude)
    """
    r = base_intercept + base_slope * x_vals
    for ptype, bp, amp in primitives:
        if ptype == 'step':
            r = r + amp * gate_step(x_vals, bp, s_step)
        elif ptype == 'ramp':
            r = r + amp * gate_ramp(x_vals, bp, s_ramp)
    return x_vals + r


def training_error(train_x, train_y, base_slope, base_intercept, primitives,
                    s_step, s_ramp):
    """Compute sum of squared errors on training data."""
    pred = evaluate_primitives(train_x, base_slope, base_intercept, primitives,
                                s_step, s_ramp)
    return np.sum((pred - train_y) ** 2)


# ============================================================================
# APPROACH 1: Pure Hinge (v2 baseline — 96.2%)
# ============================================================================

def approach_hinge(train_x, train_y, test_x, s=None, slope_tol=0.15):
    """Direct hinge decomposition. The v2 winner."""
    s = s or PHI ** 2

    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    segments = detect_segments(xs, rs, slope_tol)
    if len(segments) == 0:
        return test_x.copy()

    _, _, base_slope, base_intercept = segments[0]
    primitives = []

    for i in range(1, len(segments)):
        ps, pe, pslope, pintercept = segments[i-1]
        cs, ce, cslope, cintercept = segments[i]
        bp = (xs[pe] + xs[cs]) / 2
        primitives.extend(classify_transition(pslope, pintercept, cslope, cintercept, bp))

    return evaluate_primitives(test_x, base_slope, base_intercept, primitives, s, s)


# ============================================================================
# APPROACH 2: Hinge + Position Refinement
# ============================================================================

def approach_hinge_refined(train_x, train_y, test_x, s=None, slope_tol=0.15,
                            search_radius=4.0, search_step=0.5):
    """Hinge decomposition + breakpoint position search.
    
    After initial detection, search nearby positions for each breakpoint
    to minimize training reconstruction error. This fixes the localization
    problem without changing the structure.
    """
    s = s or PHI ** 2

    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    segments = detect_segments(xs, rs, slope_tol)
    if len(segments) == 0:
        return test_x.copy()

    _, _, base_slope, base_intercept = segments[0]
    primitives = []

    for i in range(1, len(segments)):
        ps, pe, pslope, pintercept = segments[i-1]
        cs, ce, cslope, cintercept = segments[i]
        bp = (xs[pe] + xs[cs]) / 2
        primitives.extend(classify_transition(pslope, pintercept, cslope, cintercept, bp))

    if len(primitives) == 0:
        return evaluate_primitives(test_x, base_slope, base_intercept, [], s, s)

    # Position refinement: search nearby positions for each breakpoint
    best_primitives = list(primitives)
    for i in range(len(best_primitives)):
        ptype, bp, amp = best_primitives[i]
        best_err = training_error(train_x, train_y, base_slope, base_intercept,
                                   best_primitives, s, s)
        best_bp = bp
        best_amp = amp

        for offset in np.arange(-search_radius, search_radius + search_step/2, search_step):
            candidate_bp = bp + offset
            # Try with original amplitude
            test_prims = list(best_primitives)
            test_prims[i] = (ptype, candidate_bp, amp)
            err = training_error(train_x, train_y, base_slope, base_intercept,
                                  test_prims, s, s)
            if err < best_err:
                best_err = err
                best_bp = candidate_bp
                best_amp = amp

        best_primitives[i] = (ptype, best_bp, best_amp)

    # Re-estimate amplitudes at refined positions
    # For each primitive, compute the optimal amplitude given the others
    for iteration in range(2):  # 2 passes of amplitude refinement
        for i in range(len(best_primitives)):
            ptype, bp, amp = best_primitives[i]

            # Compute contribution of all OTHER primitives
            other_prims = best_primitives[:i] + best_primitives[i+1:]
            pred_others = evaluate_primitives(train_x, base_slope, base_intercept,
                                               other_prims, s, s)
            target_residual = train_y - pred_others  # what this primitive needs to explain

            # Compute basis function for this primitive
            if ptype == 'step':
                basis = gate_step(train_x, bp, s)
            else:
                basis = gate_ramp(train_x, bp, s)

            # Optimal amplitude: minimize ||basis * amp - target||²
            # amp = (basis · target) / (basis · basis)
            basis_sq = np.sum(basis ** 2)
            if basis_sq > 1e-10:
                optimal_amp = np.sum(basis * target_residual) / basis_sq
                best_primitives[i] = (ptype, bp, optimal_amp)

    return evaluate_primitives(test_x, base_slope, base_intercept,
                                best_primitives, s, s)


# ============================================================================
# APPROACH 3: Hinge Refined + High-s Steps
# ============================================================================

def approach_hinge_refined_highs(train_x, train_y, test_x, base_s=None,
                                   step_s=20.0, slope_tol=0.15,
                                   search_radius=4.0, search_step=0.5):
    """Hinge + position refinement + high sharpness for steps."""
    base_s = base_s or PHI ** 2

    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    segments = detect_segments(xs, rs, slope_tol)
    if len(segments) == 0:
        return test_x.copy()

    _, _, base_slope, base_intercept = segments[0]
    primitives = []

    for i in range(1, len(segments)):
        ps, pe, pslope, pintercept = segments[i-1]
        cs, ce, cslope, cintercept = segments[i]
        bp = (xs[pe] + xs[cs]) / 2
        primitives.extend(classify_transition(pslope, pintercept, cslope, cintercept, bp))

    if len(primitives) == 0:
        return evaluate_primitives(test_x, base_slope, base_intercept, [], base_s, base_s)

    # Position refinement with per-type sharpness
    best_primitives = list(primitives)
    for i in range(len(best_primitives)):
        ptype, bp, amp = best_primitives[i]
        best_err = training_error(train_x, train_y, base_slope, base_intercept,
                                   best_primitives, step_s, base_s)
        best_bp = bp

        for offset in np.arange(-search_radius, search_radius + search_step/2, search_step):
            candidate_bp = bp + offset
            test_prims = list(best_primitives)
            test_prims[i] = (ptype, candidate_bp, amp)
            err = training_error(train_x, train_y, base_slope, base_intercept,
                                  test_prims, step_s, base_s)
            if err < best_err:
                best_err = err
                best_bp = candidate_bp

        best_primitives[i] = (ptype, best_bp, amp)

    # Amplitude refinement
    for iteration in range(2):
        for i in range(len(best_primitives)):
            ptype, bp, amp = best_primitives[i]
            other_prims = best_primitives[:i] + best_primitives[i+1:]
            pred_others = evaluate_primitives(train_x, base_slope, base_intercept,
                                               other_prims, step_s, base_s)
            target_residual = train_y - pred_others

            s_i = step_s if ptype == 'step' else base_s
            if ptype == 'step':
                basis = gate_step(train_x, bp, s_i)
            else:
                basis = gate_ramp(train_x, bp, s_i)

            basis_sq = np.sum(basis ** 2)
            if basis_sq > 1e-10:
                optimal_amp = np.sum(basis * target_residual) / basis_sq
                best_primitives[i] = (ptype, bp, optimal_amp)

    return evaluate_primitives(test_x, base_slope, base_intercept,
                                best_primitives, step_s, base_s)


# ============================================================================
# APPROACH 4: Greedy Breakpoint Search
# ============================================================================

def approach_greedy(train_x, train_y, test_x, s=None, max_primitives=20,
                     candidate_step=0.5):
    """Greedy breakpoint search: enumerate positions, add best one at a time.
    
    Skip hinge detection entirely. Instead:
    1. Start with linear fit (base_slope, base_intercept)
    2. Try adding a STEP or RAMP at every candidate position
    3. Add the one that reduces training error most
    4. Repeat until error is small or max primitives reached
    """
    s = s or PHI ** 2

    # Initial linear fit on residual
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    if len(xs) >= 2:
        coeffs = np.polyfit(xs, rs, 1)
        base_slope = coeffs[0]
        base_intercept = coeffs[1]
    else:
        base_slope = 0
        base_intercept = rs[0] if len(rs) > 0 else 0

    primitives = []
    x_min, x_max = xs[0], xs[-1]

    # Candidate positions: every 0.5 from min to max
    candidates = np.arange(x_min, x_max + candidate_step/2, candidate_step)

    for prim_iter in range(max_primitives):
        current_err = training_error(train_x, train_y, base_slope, base_intercept,
                                      primitives, s, s)

        if current_err < 0.5 * len(train_x):  # average error < 0.5 per point
            break

        best_improvement = 0
        best_new_prim = None

        # Precompute what existing primitives predict
        pred_current = evaluate_primitives(train_x, base_slope, base_intercept,
                                            primitives, s, s)
        remaining = train_y - pred_current

        for cand in candidates:
            # Try STEP at this position
            step_basis = gate_step(train_x, cand, s)
            step_sq = np.sum(step_basis ** 2)
            if step_sq > 1e-10:
                step_amp = np.sum(step_basis * remaining) / step_sq
                step_improvement = np.sum(remaining ** 2) - np.sum((remaining - step_amp * step_basis) ** 2)

                if step_improvement > best_improvement:
                    best_improvement = step_improvement
                    best_new_prim = ('step', cand, step_amp)

            # Try RAMP at this position
            ramp_basis = gate_ramp(train_x, cand, s)
            ramp_sq = np.sum(ramp_basis ** 2)
            if ramp_sq > 1e-10:
                ramp_amp = np.sum(ramp_basis * remaining) / ramp_sq
                ramp_improvement = np.sum(remaining ** 2) - np.sum((remaining - ramp_amp * ramp_basis) ** 2)

                if ramp_improvement > best_improvement:
                    best_improvement = ramp_improvement
                    best_new_prim = ('ramp', cand, ramp_amp)

        if best_new_prim is None or best_improvement < 1.0:
            break

        primitives.append(best_new_prim)

        # After adding, re-estimate ALL amplitudes jointly
        # This prevents greedy coupling errors
        n_prims = len(primitives)
        H = np.zeros((len(train_x), n_prims))
        for j, (ptype, bp, _) in enumerate(primitives):
            if ptype == 'step':
                H[:, j] = gate_step(train_x, bp, s)
            else:
                H[:, j] = gate_ramp(train_x, bp, s)

        target = train_y - (train_x + base_intercept + base_slope * train_x)
        if n_prims > 0:
            # Least squares (unregularized — we know the structure)
            amps, _, _, _ = np.linalg.lstsq(H, target, rcond=None)
            primitives = [(ptype, bp, float(amps[j])) for j, (ptype, bp, _) in enumerate(primitives)]

    return evaluate_primitives(test_x, base_slope, base_intercept, primitives, s, s)


# ============================================================================
# APPROACH 5: Greedy + High-s + Refinement (the full v3 pipeline)
# ============================================================================

def approach_v3_full(train_x, train_y, test_x, base_s=None, step_s=20.0,
                      max_primitives=20, candidate_step=1.0,
                      refine_radius=2.0, refine_step=0.25):
    """Full v3 pipeline: greedy search + position refinement + high-s steps.
    
    1. Greedy search to find candidate primitives
    2. Position refinement for each one
    3. Amplitude re-estimation
    4. High sharpness for steps
    """
    base_s = base_s or PHI ** 2

    # Initial linear fit on residual
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    if len(xs) >= 2:
        coeffs = np.polyfit(xs, rs, 1)
        base_slope = coeffs[0]
        base_intercept = coeffs[1]
    else:
        base_slope = 0
        base_intercept = rs[0] if len(rs) > 0 else 0

    primitives = []
    x_min, x_max = xs[0], xs[-1]
    candidates = np.arange(x_min, x_max + candidate_step/2, candidate_step)

    # Phase 1: Greedy search (using base_s for detection)
    for prim_iter in range(max_primitives):
        pred_current = evaluate_primitives(train_x, base_slope, base_intercept,
                                            primitives, base_s, base_s)
        remaining = train_y - pred_current
        current_err = np.sum(remaining ** 2)

        if current_err < 0.25 * len(train_x):
            break

        best_improvement = 0
        best_new_prim = None

        for cand in candidates:
            for ptype in ['step', 'ramp']:
                if ptype == 'step':
                    basis = gate_step(train_x, cand, base_s)
                else:
                    basis = gate_ramp(train_x, cand, base_s)

                basis_sq = np.sum(basis ** 2)
                if basis_sq > 1e-10:
                    amp = np.sum(basis * remaining) / basis_sq
                    improvement = np.sum(remaining ** 2) - np.sum((remaining - amp * basis) ** 2)

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_new_prim = (ptype, cand, amp)

        if best_new_prim is None or best_improvement < 0.5:
            break

        primitives.append(best_new_prim)

        # Re-estimate amplitudes jointly
        n_prims = len(primitives)
        H = np.zeros((len(train_x), n_prims))
        for j, (ptype, bp, _) in enumerate(primitives):
            if ptype == 'step':
                H[:, j] = gate_step(train_x, bp, base_s)
            else:
                H[:, j] = gate_ramp(train_x, bp, base_s)

        target = train_y - (train_x + base_intercept + base_slope * train_x)
        if n_prims > 0:
            amps, _, _, _ = np.linalg.lstsq(H, target, rcond=None)
            primitives = [(ptype, bp, float(amps[j])) for j, (ptype, bp, _) in enumerate(primitives)]

    if len(primitives) == 0:
        return evaluate_primitives(test_x, base_slope, base_intercept, [], base_s, base_s)

    # Phase 2: Position refinement with high-s for steps
    for i in range(len(primitives)):
        ptype, bp, amp = primitives[i]
        best_err = training_error(train_x, train_y, base_slope, base_intercept,
                                   primitives, step_s, base_s)
        best_bp = bp

        for offset in np.arange(-refine_radius, refine_radius + refine_step/2, refine_step):
            candidate_bp = bp + offset
            test_prims = list(primitives)
            test_prims[i] = (ptype, candidate_bp, amp)
            err = training_error(train_x, train_y, base_slope, base_intercept,
                                  test_prims, step_s, base_s)
            if err < best_err:
                best_err = err
                best_bp = candidate_bp

        primitives[i] = (ptype, best_bp, amp)

    # Phase 3: Final amplitude re-estimation with high-s
    for iteration in range(3):
        for i in range(len(primitives)):
            ptype, bp, amp = primitives[i]
            other_prims = primitives[:i] + primitives[i+1:]
            pred_others = evaluate_primitives(train_x, base_slope, base_intercept,
                                               other_prims, step_s, base_s)
            target_residual = train_y - pred_others

            s_i = step_s if ptype == 'step' else base_s
            if ptype == 'step':
                basis = gate_step(train_x, bp, s_i)
            else:
                basis = gate_ramp(train_x, bp, s_i)

            basis_sq = np.sum(basis ** 2)
            if basis_sq > 1e-10:
                optimal_amp = np.sum(basis * target_residual) / basis_sq
                primitives[i] = (ptype, bp, optimal_amp)

    return evaluate_primitives(test_x, base_slope, base_intercept,
                                primitives, step_s, base_s)


# ============================================================================
# APPROACH 6: Best-of-all ensemble
# ============================================================================

def approach_ensemble(train_x, train_y, test_x, base_s=None, step_s=20.0):
    """Run all approaches, pick best on training data."""
    base_s = base_s or PHI ** 2
    
    candidates = [
        ('hinge', lambda: approach_hinge(train_x, train_y, test_x, base_s)),
        ('hinge_refined', lambda: approach_hinge_refined(train_x, train_y, test_x, base_s)),
        ('hinge_highs', lambda: approach_hinge_refined_highs(train_x, train_y, test_x, base_s, step_s)),
        ('greedy', lambda: approach_greedy(train_x, train_y, test_x, base_s)),
        ('v3_full', lambda: approach_v3_full(train_x, train_y, test_x, base_s, step_s)),
    ]
    
    # Also try different slope tolerances
    for tol in [0.08, 0.12, 0.20]:
        candidates.append(
            (f'hinge_tol{tol}', lambda t=tol: approach_hinge(train_x, train_y, test_x, base_s, t))
        )
        candidates.append(
            (f'refined_tol{tol}', lambda t=tol: approach_hinge_refined(train_x, train_y, test_x, base_s, t))
        )
    
    best_pred = None
    best_err = np.inf
    best_name = "none"
    
    for name, fn in candidates:
        try:
            pred = fn()
            # Evaluate on training data
            train_pred_fn = None
            if 'hinge' in name and 'refined' not in name and 'highs' not in name:
                tol = 0.15
                if 'tol' in name:
                    tol = float(name.split('tol')[1])
                train_pred_fn = approach_hinge(train_x, train_y, train_x, base_s, tol)
            elif 'refined' in name and 'highs' not in name:
                tol = 0.15
                if 'tol' in name:
                    tol = float(name.split('tol')[1])
                train_pred_fn = approach_hinge_refined(train_x, train_y, train_x, base_s, tol)
            elif name == 'hinge_highs':
                train_pred_fn = approach_hinge_refined_highs(train_x, train_y, train_x, base_s, step_s)
            elif name == 'greedy':
                train_pred_fn = approach_greedy(train_x, train_y, train_x, base_s)
            elif name == 'v3_full':
                train_pred_fn = approach_v3_full(train_x, train_y, train_x, base_s, step_s)
            
            if train_pred_fn is not None:
                err = np.sum((train_pred_fn - train_y) ** 2)
            else:
                err = np.sum((pred[:len(train_y)] - train_y) ** 2) if len(pred) >= len(train_y) else np.inf
            
            if err < best_err:
                best_err = err
                best_pred = pred
                best_name = name
        except Exception:
            continue
    
    return best_pred if best_pred is not None else test_x.copy(), best_name


# ============================================================================
# BENCHMARK
# ============================================================================

print("=" * 70)
print("PINNED LEARNING v3: Precision Breakpoint Detection")
print("=" * 70)

test_functions = [
    ("tolower",        fn_tolower,       128, 50),
    ("secret_fn",      fn_secret,        100, 40),
    ("ROT13",          fn_rot13,         128, 60),
    ("abs_centered",   fn_abs_centered,  128, 40),
    ("sawtooth_32",    fn_sawtooth,      128, 50),
    ("clamp_30_100",   fn_clamp,         128, 40),
    ("relu_shifted",   fn_relu_shifted,  128, 30),
    ("staircase_16",   fn_staircase,     128, 50),
]

approaches_list = [
    ("hinge",           lambda tx, ty, sx: approach_hinge(tx, ty, sx)),
    ("hinge_refined",   lambda tx, ty, sx: approach_hinge_refined(tx, ty, sx)),
    ("hinge_highs",     lambda tx, ty, sx: approach_hinge_refined_highs(tx, ty, sx)),
    ("greedy",          lambda tx, ty, sx: approach_greedy(tx, ty, sx)),
    ("v3_full",         lambda tx, ty, sx: approach_v3_full(tx, ty, sx)),
]

all_results = {}

for fn_name, fn, input_range, n_train in test_functions:
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)

    np.random.seed(42)
    train_idx = np.sort(np.random.choice(input_range, n_train, replace=False))
    train_x = train_idx.astype(np.float64)
    train_y = fn(train_x)

    print(f"\n{'='*60}")
    print(f"  {fn_name} ({n_train} train / {input_range} test)")
    print(f"{'='*60}")

    results = {}

    for aname, afn in approaches_list:
        t0 = time.perf_counter()
        try:
            pred = afn(train_x, train_y, test_x)
        except Exception as e:
            print(f"  {aname:<17s}: ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0

        exact = int(np.sum(np.round(pred) == test_y))
        max_err = float(np.abs(pred - test_y).max())
        results[aname] = (pred, exact, max_err, elapsed)

        print(f"  {aname:<17s}: {exact:>4d}/{input_range} exact, "
              f"max_err={max_err:>8.4f}, time={elapsed:.4f}s")

    # Ensemble
    t0 = time.perf_counter()
    ens_pred, ens_name = approach_ensemble(train_x, train_y, test_x)
    elapsed = time.perf_counter() - t0
    ens_exact = int(np.sum(np.round(ens_pred) == test_y))
    ens_maxe = float(np.abs(ens_pred - test_y).max())
    results['ensemble'] = (ens_pred, ens_exact, ens_maxe, elapsed)
    print(f"  {'ensemble':<17s}: {ens_exact:>4d}/{input_range} exact, "
          f"max_err={ens_maxe:>8.4f}, time={elapsed:.4f}s  (via {ens_name})")

    all_results[fn_name] = results


# ============================================================================
# SAMPLE EFFICIENCY
# ============================================================================

print()
print("=" * 70)
print("SAMPLE EFFICIENCY")
print("=" * 70)

sample_counts = [3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 128]
sample_results = {}

for fn_name, fn, input_range, _ in test_functions[:5]:
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)

    print(f"\n  {fn_name}:")
    fn_samples = {}

    for n_s in sample_counts:
        if n_s > input_range:
            continue
        np.random.seed(42)
        if n_s >= input_range:
            tx_s = np.arange(input_range, dtype=np.float64)
        else:
            tx_s = np.sort(np.random.choice(input_range, n_s, replace=False)).astype(np.float64)
        ty_s = fn(tx_s)

        h_pred = approach_hinge(tx_s, ty_s, test_x)
        h_exact = int(np.sum(np.round(h_pred) == test_y))

        hr_pred = approach_hinge_refined(tx_s, ty_s, test_x)
        hr_exact = int(np.sum(np.round(hr_pred) == test_y))

        g_pred = approach_greedy(tx_s, ty_s, test_x)
        g_exact = int(np.sum(np.round(g_pred) == test_y))

        fn_samples[n_s] = (h_exact, hr_exact, g_exact)
        print(f"    {n_s:>3d} ex: hinge={h_exact:>3d}  refined={hr_exact:>3d}  "
              f"greedy={g_exact:>3d}  /{input_range}")

    sample_results[fn_name] = fn_samples


# ============================================================================
# STEP SHARPNESS SWEEP
# ============================================================================

print()
print("=" * 70)
print("STEP SHARPNESS SWEEP (hinge_refined_highs)")
print("=" * 70)

step_s_values = [PHI**2, 5.0, 10.0, 20.0, 50.0, 100.0]
sharpness_results = {}

for fn_name, fn, input_range, n_train in test_functions[:3]:
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)
    np.random.seed(42)
    train_x = np.sort(np.random.choice(input_range, n_train, replace=False)).astype(np.float64)
    train_y = fn(train_x)

    print(f"\n  {fn_name}:")
    fn_sharp = {}
    for ss in step_s_values:
        pred = approach_hinge_refined_highs(train_x, train_y, test_x, step_s=ss)
        exact = int(np.sum(np.round(pred) == test_y))
        max_err = float(np.abs(pred - test_y).max())
        fn_sharp[ss] = (exact, max_err)
        print(f"    step_s={ss:>6.2f}: {exact:>4d}/{input_range} exact, max_err={max_err:.4f}")

    sharpness_results[fn_name] = fn_sharp


# ============================================================================
# ERROR ANALYSIS: Where do the remaining errors occur?
# ============================================================================

print()
print("=" * 70)
print("ERROR ANALYSIS: Where are the remaining errors?")
print("=" * 70)

for fn_name, fn, input_range, n_train in test_functions[:5]:
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)
    np.random.seed(42)
    train_x = np.sort(np.random.choice(input_range, n_train, replace=False)).astype(np.float64)
    train_y = fn(train_x)

    pred = approach_hinge_refined_highs(train_x, train_y, test_x)
    errors = np.where(np.round(pred) != test_y)[0]

    print(f"\n  {fn_name}: {len(errors)} errors at x = {errors.tolist()[:20]}")
    if len(errors) > 0:
        for e in errors[:5]:
            print(f"    x={e}: true={test_y[e]:.0f}, pred={pred[e]:.2f}, "
                  f"rounded={np.round(pred[e]):.0f}, err={abs(pred[e]-test_y[e]):.4f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)
fig.suptitle('Pinned Learning v3: Precision Breakpoint Detection',
             fontsize=15, fontweight='bold', y=1.01)

colors = {'hinge': '#F44336', 'hinge_refined': '#2196F3',
          'hinge_highs': '#4CAF50', 'greedy': '#FF9800',
          'v3_full': '#9C27B0', 'ensemble': '#00BCD4'}

# Row 1-2: Function plots
for idx, (fn_name, fn, input_range, n_train) in enumerate(test_functions):
    row = idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])

    tx = np.arange(input_range, dtype=np.float64)
    ty = fn(tx)
    ax.plot(tx, ty, 'k-', linewidth=2, alpha=0.3, label='True')

    if fn_name in all_results:
        sorted_r = sorted(all_results[fn_name].items(), key=lambda x: (-x[1][1], x[1][2]))
        for rank, (aname, (pred, exact, maxe, _)) in enumerate(sorted_r[:3]):
            ax.plot(tx, pred, color=colors.get(aname, 'gray'), linewidth=1.2,
                    alpha=0.8, label=f'{aname}: {exact}/{input_range}')

    ax.set_title(fn_name, fontweight='bold', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

# Row 3: Accuracy + Sample efficiency
ax_acc = fig.add_subplot(gs[2, 0:2])
approach_names = [n for n, _ in approaches_list] + ['ensemble']
bar_labels = ['Hinge\n(v2)', 'Hinge\nRefined', 'Hinge\nHigh-s', 'Greedy\nSearch',
              'v3 Full\nPipeline', 'Ensemble']
bar_colors_list = [colors.get(n, 'gray') for n in approach_names]

avg_accs = []
for aname in approach_names:
    accs = []
    for fn_name, _, input_range, _ in test_functions:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            accs.append(exact / input_range * 100)
    avg_accs.append(np.mean(accs) if accs else 0)

bars = ax_acc.bar(bar_labels, avg_accs, color=bar_colors_list, alpha=0.85)
for bar, acc in zip(bars, avg_accs):
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax_acc.set_ylabel('Average accuracy (%)')
ax_acc.set_title('Average Accuracy Across 8 Functions', fontweight='bold')
ax_acc.set_ylim(0, 110)
ax_acc.grid(True, alpha=0.3, axis='y')

# Sample efficiency
ax_se = fig.add_subplot(gs[2, 2:4])
se_colors = {'tolower': 'blue', 'secret_fn': 'red', 'ROT13': 'green',
             'abs_centered': 'orange', 'sawtooth_32': 'purple'}
for fn_name in list(sample_results.keys())[:3]:
    results = sample_results[fn_name]
    ns = sorted(results.keys())
    h_accs = [results[n][0] for n in ns]
    hr_accs = [results[n][1] for n in ns]
    g_accs = [results[n][2] for n in ns]

    ax_se.plot(ns, hr_accs, '-o', color=se_colors[fn_name], linewidth=2.5,
               markersize=6, label=f'refined {fn_name}')
    ax_se.plot(ns, h_accs, '--s', color=se_colors[fn_name], linewidth=1,
               markersize=4, alpha=0.4, label=f'hinge {fn_name}')
    ax_se.plot(ns, g_accs, ':^', color=se_colors[fn_name], linewidth=1,
               markersize=4, alpha=0.4, label=f'greedy {fn_name}')

ax_se.axhline(128, color='gray', linestyle=':', alpha=0.5)
ax_se.set_xlabel('Training examples')
ax_se.set_ylabel('Exact matches')
ax_se.set_title('Sample Efficiency Comparison', fontweight='bold')
ax_se.legend(fontsize=6, ncol=3)
ax_se.grid(True, alpha=0.3)

# Row 4: Heatmap + Sharpness sweep + Insight
ax_hm = fig.add_subplot(gs[3, 0:2])
fn_names_list = [fn_name for fn_name, _, _, _ in test_functions]
heatmap_data = []
for fn_name in fn_names_list:
    row = []
    for aname in approach_names:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            inp = 100 if fn_name == 'secret_fn' else 128
            row.append(exact / inp * 100)
        else:
            row.append(0)
    heatmap_data.append(row)

hm = ax_hm.imshow(heatmap_data, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')
ax_hm.set_xticks(range(len(approach_names)))
ax_hm.set_xticklabels(bar_labels, fontsize=8)
ax_hm.set_yticks(range(len(fn_names_list)))
ax_hm.set_yticklabels(fn_names_list, fontsize=9)
for i in range(len(fn_names_list)):
    for j in range(len(approach_names)):
        ax_hm.text(j, i, f'{heatmap_data[i][j]:.0f}', ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if heatmap_data[i][j] < 60 else 'black')
plt.colorbar(hm, ax=ax_hm, shrink=0.8, label='Accuracy %')
ax_hm.set_title('Per-Function Accuracy Heatmap', fontweight='bold')

# Sharpness sweep
ax_sh = fig.add_subplot(gs[3, 2])
for fn_name in sharpness_results:
    ss_vals = sorted(sharpness_results[fn_name].keys())
    accs = [sharpness_results[fn_name][sv][0] for sv in ss_vals]
    inp = 100 if fn_name == 'secret_fn' else 128
    ax_sh.semilogx(ss_vals, accs, '-o', label=fn_name, linewidth=2, markersize=5)
ax_sh.axvline(PHI**2, color='gray', linestyle='--', alpha=0.5, label=f'φ²')
ax_sh.set_xlabel('Step sharpness (s)')
ax_sh.set_ylabel('Exact matches')
ax_sh.set_title('Step Sharpness Effect', fontweight='bold')
ax_sh.legend(fontsize=8)
ax_sh.grid(True, alpha=0.3)

# Insight panel
ax_ins = fig.add_subplot(gs[3, 3])
ax_ins.axis('off')
insight = (
    "v3 KEY FINDINGS\n"
    "═══════════════════════\n\n"
    "1. POSITION REFINEMENT\n"
    "   Search ±4 around each\n"
    "   detected breakpoint.\n"
    "   Fixes localization error\n"
    "   from sparse data.\n\n"
    "2. AMPLITUDE RE-ESTIMATION\n"
    "   After refining positions,\n"
    "   re-compute optimal amp\n"
    "   for each primitive.\n\n"
    "3. GREEDY SEARCH\n"
    "   Enumerate all candidate\n"
    "   positions. Add the one\n"
    "   that explains the most.\n\n"
    "4. HIGH-s STEPS\n"
    "   s=20 for steps makes\n"
    "   transitions near-perfect\n"
    "   at integer boundaries."
)
ax_ins.text(0.05, 0.95, insight, transform=ax_ins.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/tmp/geometric_pinned_v3.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print("=" * 70)
print("FINAL SUMMARY: v3")
print("=" * 70)

print(f"\n  {'Function':<15s}", end="")
for label in bar_labels:
    print(f"  {label.replace(chr(10),' '):>12s}", end="")
print()
print("  " + "-" * 87)

for fn_name, _, input_range, _ in test_functions:
    print(f"  {fn_name:<15s}", end="")
    for aname in approach_names:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            print(f"  {exact:>4d}/{input_range:<3d}    ", end="")
        else:
            print(f"  {'N/A':>12s}", end="")
    print()

print(f"\n  {'AVERAGE':<15s}", end="")
for aname in approach_names:
    accs = []
    for fn_name, _, input_range, _ in test_functions:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            accs.append(exact / input_range * 100)
    avg = np.mean(accs) if accs else 0
    print(f"  {avg:>7.1f}%    ", end="")
print()

# Perfect scores
print(f"\n  PERFECT SCORES (100%):")
for fn_name, _, input_range, _ in test_functions:
    if fn_name in all_results:
        for aname, (_, exact, _, _) in all_results[fn_name].items():
            if exact == input_range:
                print(f"    {fn_name:<15s}: {aname}")

print(f"\nSaved: /tmp/geometric_pinned_v3.png")
