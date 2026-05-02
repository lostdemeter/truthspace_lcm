#!/usr/bin/env python3
"""
Pinned Threshold Learning v2.1: Refined Geometric Program Synthesis

Key insight from v2: Hinge decomposition (96.2%) dominates all other approaches.
But remaining errors come from:
  1. Imprecise breakpoint locations (sparse data doesn't bracket boundaries tightly)
  2. Noisy amplitude estimates from finite differences
  3. Missing breakpoints when training points are far from transitions

v2.1 attacks all three:
  A. DETECT: Hinge decomposition finds breakpoint STRUCTURE (types and locations)
  B. REFINE: Ridge regression fits OPTIMAL weights to detected basis functions
  C. ITERATE: Residual analysis catches MISSED breakpoints
  D. ADAPT: Per-primitive sharpness tuning maximizes transition accuracy

The combined approach: detect → refine → iterate → adapt
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
    """Ideal Gate in numpy (float64)."""
    x = np.asarray(x, dtype=np.float64)
    f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))


# ============================================================================
# PRIMITIVES: The three atoms of geometric computing
# ============================================================================

def gate_step(x, threshold, s):
    """Step: ≈1 for x > threshold, ≈0 for x < threshold.
    Implemented as narrow RECT (pair of gate neurons)."""
    g_lo = ideal_gate(s * (x - (threshold - 0.5))) / s
    g_hi = ideal_gate(s * (x - (threshold + 0.5))) / s
    return g_lo - g_hi


def gate_ramp(x, threshold, s):
    """Ramp: ≈(x-threshold) for x > threshold, ≈0 for x < threshold."""
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
# CORE: Segment Detection
# ============================================================================

def detect_segments(xs, rs, slope_tol=0.15):
    """Detect contiguous segments of approximately constant slope.
    
    Returns list of (start_idx, end_idx, slope, intercept).
    Uses median-based robust slope estimation.
    """
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
            # Close current segment
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
    
    # Final segment
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
    """Classify a transition between two segments.
    
    Returns list of primitives: ('step', bp, height) and/or ('ramp', bp, delta_slope).
    """
    primitives = []
    
    delta_slope = curr_slope - prev_slope
    
    # Value continuity check
    r_prev = prev_slope * bp + prev_intercept
    r_curr = curr_slope * bp + curr_intercept
    jump = r_curr - r_prev
    
    if abs(jump) > 0.3:
        primitives.append(('step', bp, jump))
    if abs(delta_slope) > 0.03:
        primitives.append(('ramp', bp, delta_slope))
    
    return primitives


# ============================================================================
# APPROACH 1: Pure Hinge (from v2 — baseline)
# ============================================================================

def approach_hinge(train_x, train_y, test_x, s=None, slope_tol=0.15):
    """Direct hinge decomposition: detect segments, classify transitions, construct."""
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
    
    # Evaluate
    r_test = base_intercept + base_slope * test_x
    for ptype, bp, value in primitives:
        if ptype == 'step':
            r_test += value * gate_step(test_x, bp, s)
        elif ptype == 'ramp':
            r_test += value * gate_ramp(test_x, bp, s)
    
    return test_x + r_test


# ============================================================================
# APPROACH 2: Hinge + Ridge Refinement
# ============================================================================

def approach_hinge_ridge(train_x, train_y, test_x, s=None, slope_tol=0.15,
                          alpha=0.001):
    """Detect breakpoints via hinge, then fit optimal weights via ridge regression.
    
    The key insight: hinge finds WHERE breakpoints are (the hard part).
    Ridge regression finds the OPTIMAL amplitudes (the easy part).
    This separates detection from fitting.
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs
    
    segments = detect_segments(xs, rs, slope_tol)
    if len(segments) == 0:
        return test_x.copy()
    
    # Collect all detected breakpoints and their types
    breakpoint_info = []
    for i in range(1, len(segments)):
        ps, pe, pslope, pintercept = segments[i-1]
        cs, ce, cslope, cintercept = segments[i]
        bp = (xs[pe] + xs[cs]) / 2
        prims = classify_transition(pslope, pintercept, cslope, cintercept, bp)
        breakpoint_info.extend(prims)
    
    if len(breakpoint_info) == 0:
        # Pure linear
        coeffs = np.polyfit(train_x, train_y, 1)
        return coeffs[0] * test_x + coeffs[1]
    
    # Build basis matrix: one column per primitive, plus linear and constant terms
    n_prims = len(breakpoint_info)
    n_basis = n_prims + 2  # primitives + slope + intercept
    
    H_train = np.zeros((len(train_x), n_basis))
    H_train[:, 0] = 1.0  # constant
    H_train[:, 1] = train_x  # linear term
    
    for j, (ptype, bp, _) in enumerate(breakpoint_info):
        if ptype == 'step':
            H_train[:, j+2] = gate_step(train_x, bp, s)
        elif ptype == 'ramp':
            H_train[:, j+2] = gate_ramp(train_x, bp, s)
    
    # Ridge regression: minimize ||H @ w - y||² + α||w||²
    r_train = train_y - train_x  # fit the residual (skip connection handles identity)
    # But we included linear term in H, so fit y directly and subtract x
    # Actually, let's fit r = y - x with basis including constant, linear, and primitives
    reg = alpha * len(train_x) * np.eye(n_basis)
    reg[0, 0] = 0  # don't regularize the constant
    reg[1, 1] = 0  # don't regularize the linear term
    
    w = np.linalg.solve(H_train.T @ H_train + reg, H_train.T @ r_train)
    
    # Predict
    H_test = np.zeros((len(test_x), n_basis))
    H_test[:, 0] = 1.0
    H_test[:, 1] = test_x
    for j, (ptype, bp, _) in enumerate(breakpoint_info):
        if ptype == 'step':
            H_test[:, j+2] = gate_step(test_x, bp, s)
        elif ptype == 'ramp':
            H_test[:, j+2] = gate_ramp(test_x, bp, s)
    
    return test_x + H_test @ w


# ============================================================================
# APPROACH 3: Iterative Hinge + Ridge (detect → refine → iterate)
# ============================================================================

def approach_iterative(train_x, train_y, test_x, s=None, max_iters=3,
                        slope_tol=0.15, alpha=0.001):
    """Iterative refinement: detect breakpoints, fit, analyze residual, repeat.
    
    Round 1: Hinge decomposition on original data
    Round 2+: Hinge decomposition on RESIDUAL from previous round
    Final: Ridge regression on ALL detected basis functions
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs_sorted = train_x[order]
    ys_sorted = train_y[order]
    
    all_primitives = []
    current_residual = ys_sorted - xs_sorted
    
    for iteration in range(max_iters):
        # Detect segments in current residual
        segments = detect_segments(xs_sorted, current_residual, slope_tol)
        
        if len(segments) <= 1:
            # No more structure to find
            break
        
        # Classify transitions
        new_prims = []
        for i in range(1, len(segments)):
            ps, pe, pslope, pintercept = segments[i-1]
            cs, ce, cslope, cintercept = segments[i]
            bp = (xs_sorted[pe] + xs_sorted[cs]) / 2
            prims = classify_transition(pslope, pintercept, cslope, cintercept, bp)
            new_prims.extend(prims)
        
        if len(new_prims) == 0:
            break
        
        # Deduplicate: don't add primitives too close to existing ones
        for ptype, bp, val in new_prims:
            is_dup = False
            for etype, ebp, _ in all_primitives:
                if etype == ptype and abs(ebp - bp) < 1.0:
                    is_dup = True
                    break
            if not is_dup:
                all_primitives.append((ptype, bp, val))
        
        # Build basis and fit with ridge to get current residual
        if len(all_primitives) > 0:
            n_basis = len(all_primitives) + 2
            H = np.zeros((len(xs_sorted), n_basis))
            H[:, 0] = 1.0
            H[:, 1] = xs_sorted
            
            for j, (ptype, bp, _) in enumerate(all_primitives):
                if ptype == 'step':
                    H[:, j+2] = gate_step(xs_sorted, bp, s)
                elif ptype == 'ramp':
                    H[:, j+2] = gate_ramp(xs_sorted, bp, s)
            
            r_target = ys_sorted - xs_sorted
            reg = alpha * len(xs_sorted) * np.eye(n_basis)
            reg[0, 0] = 0
            reg[1, 1] = 0
            
            w = np.linalg.solve(H.T @ H + reg, H.T @ r_target)
            
            # Update residual
            current_residual = r_target - H @ w
            
            if np.abs(current_residual).max() < 0.5:
                break
    
    if len(all_primitives) == 0:
        # Fall back to linear
        coeffs = np.polyfit(train_x, train_y, 1)
        return coeffs[0] * test_x + coeffs[1]
    
    # Final prediction
    n_basis = len(all_primitives) + 2
    H_test = np.zeros((len(test_x), n_basis))
    H_test[:, 0] = 1.0
    H_test[:, 1] = test_x
    for j, (ptype, bp, _) in enumerate(all_primitives):
        if ptype == 'step':
            H_test[:, j+2] = gate_step(test_x, bp, s)
        elif ptype == 'ramp':
            H_test[:, j+2] = gate_ramp(test_x, bp, s)
    
    return test_x + H_test @ w


# ============================================================================
# APPROACH 4: Adaptive Sharpness
# ============================================================================

def approach_adaptive_sharpness(train_x, train_y, test_x, base_s=None,
                                 slope_tol=0.15, alpha=0.001):
    """Like hinge+ridge, but with per-primitive adaptive sharpness.
    
    Steps get HIGH sharpness (sharp transitions).
    Ramps get MODERATE sharpness (smooth transitions).
    Sharpness scales with 1/gap_to_nearest_breakpoint.
    """
    base_s = base_s or PHI ** 2
    
    order = np.argsort(train_x)
    xs = train_x[order]
    ys = train_y[order]
    rs = ys - xs
    
    segments = detect_segments(xs, rs, slope_tol)
    if len(segments) <= 1:
        coeffs = np.polyfit(train_x, train_y, 1)
        return coeffs[0] * test_x + coeffs[1]
    
    primitives = []
    for i in range(1, len(segments)):
        ps, pe, pslope, pintercept = segments[i-1]
        cs, ce, cslope, cintercept = segments[i]
        bp = (xs[pe] + xs[cs]) / 2
        prims = classify_transition(pslope, pintercept, cslope, cintercept, bp)
        primitives.extend(prims)
    
    if len(primitives) == 0:
        coeffs = np.polyfit(train_x, train_y, 1)
        return coeffs[0] * test_x + coeffs[1]
    
    # Compute minimum distance between breakpoints
    bps = np.array([bp for _, bp, _ in primitives])
    bps_unique = np.unique(bps)
    if len(bps_unique) > 1:
        min_gap = np.min(np.diff(bps_unique))
    else:
        min_gap = 10.0
    
    # Adaptive sharpness: steps get higher s, ramps get base s
    # Also scale s based on gap: tighter breakpoints need sharper gates
    primitive_sharpness = []
    for ptype, bp, val in primitives:
        if ptype == 'step':
            # Steps need sharp transitions — use higher s
            # Scale with 1/gap to avoid crosstalk with nearby breakpoints
            s_prim = max(base_s, min(50.0, 2.0 / max(min_gap, 0.1)))
        else:
            # Ramps are naturally smooth — base s is fine
            s_prim = base_s
        primitive_sharpness.append(s_prim)
    
    # Build basis with per-primitive sharpness
    n_basis = len(primitives) + 2
    H_train = np.zeros((len(train_x), n_basis))
    H_train[:, 0] = 1.0
    H_train[:, 1] = train_x
    
    for j, ((ptype, bp, _), s_j) in enumerate(zip(primitives, primitive_sharpness)):
        if ptype == 'step':
            H_train[:, j+2] = gate_step(train_x, bp, s_j)
        elif ptype == 'ramp':
            H_train[:, j+2] = gate_ramp(train_x, bp, s_j)
    
    # Ridge regression
    r_train = train_y - train_x
    reg = alpha * len(train_x) * np.eye(n_basis)
    reg[0, 0] = 0
    reg[1, 1] = 0
    w = np.linalg.solve(H_train.T @ H_train + reg, H_train.T @ r_train)
    
    # Predict
    H_test = np.zeros((len(test_x), n_basis))
    H_test[:, 0] = 1.0
    H_test[:, 1] = test_x
    for j, ((ptype, bp, _), s_j) in enumerate(zip(primitives, primitive_sharpness)):
        if ptype == 'step':
            H_test[:, j+2] = gate_step(test_x, bp, s_j)
        elif ptype == 'ramp':
            H_test[:, j+2] = gate_ramp(test_x, bp, s_j)
    
    return test_x + H_test @ w


# ============================================================================
# APPROACH 5: Full Pipeline (detect → classify → ridge → iterate → adapt)
# ============================================================================

def approach_full_pipeline(train_x, train_y, test_x, base_s=None,
                            slope_tol=0.15, alpha=0.001, max_iters=3):
    """The complete pipeline combining all improvements.
    
    1. Detect segments in residual
    2. Classify transitions (STEP/RAMP)
    3. Iteratively find more breakpoints in ridge-regression residual
    4. Apply adaptive per-primitive sharpness
    5. Final ridge regression with all basis functions
    """
    base_s = base_s or PHI ** 2
    
    order = np.argsort(train_x)
    xs_sorted = train_x[order]
    ys_sorted = train_y[order]
    
    all_primitives = []
    current_residual = ys_sorted - xs_sorted
    
    # Phase 1: Iterative detection
    for iteration in range(max_iters):
        segments = detect_segments(xs_sorted, current_residual, slope_tol)
        
        if len(segments) <= 1:
            break
        
        new_prims = []
        for i in range(1, len(segments)):
            ps, pe, pslope, pintercept = segments[i-1]
            cs, ce, cslope, cintercept = segments[i]
            bp = (xs_sorted[pe] + xs_sorted[cs]) / 2
            prims = classify_transition(pslope, pintercept, cslope, cintercept, bp)
            new_prims.extend(prims)
        
        if len(new_prims) == 0:
            break
        
        # Deduplicate
        for ptype, bp, val in new_prims:
            is_dup = False
            for etype, ebp, _ in all_primitives:
                if etype == ptype and abs(ebp - bp) < 1.0:
                    is_dup = True
                    break
            if not is_dup:
                all_primitives.append((ptype, bp, val))
        
        # Fit current set and compute residual
        if len(all_primitives) > 0:
            n_basis = len(all_primitives) + 2
            H = np.zeros((len(xs_sorted), n_basis))
            H[:, 0] = 1.0
            H[:, 1] = xs_sorted
            for j, (ptype, bp, _) in enumerate(all_primitives):
                if ptype == 'step':
                    H[:, j+2] = gate_step(xs_sorted, bp, base_s)
                elif ptype == 'ramp':
                    H[:, j+2] = gate_ramp(xs_sorted, bp, base_s)
            
            r_target = ys_sorted - xs_sorted
            reg = alpha * len(xs_sorted) * np.eye(n_basis)
            reg[0, 0] = 0
            reg[1, 1] = 0
            w_iter = np.linalg.solve(H.T @ H + reg, H.T @ r_target)
            current_residual = r_target - H @ w_iter
            
            if np.abs(current_residual).max() < 0.3:
                break
    
    if len(all_primitives) == 0:
        coeffs = np.polyfit(train_x, train_y, 1)
        return coeffs[0] * test_x + coeffs[1]
    
    # Phase 2: Adaptive sharpness
    bps = np.array([bp for _, bp, _ in all_primitives])
    bps_unique = np.unique(bps)
    if len(bps_unique) > 1:
        min_gap = np.min(np.diff(bps_unique))
    else:
        min_gap = 10.0
    
    primitive_sharpness = []
    for ptype, bp, val in all_primitives:
        if ptype == 'step':
            s_prim = max(base_s, min(50.0, 2.0 / max(min_gap, 0.1)))
        else:
            s_prim = base_s
        primitive_sharpness.append(s_prim)
    
    # Phase 3: Final ridge regression with adaptive sharpness
    n_basis = len(all_primitives) + 2
    H_train = np.zeros((len(train_x), n_basis))
    H_train[:, 0] = 1.0
    H_train[:, 1] = train_x
    for j, ((ptype, bp, _), s_j) in enumerate(zip(all_primitives, primitive_sharpness)):
        if ptype == 'step':
            H_train[:, j+2] = gate_step(train_x, bp, s_j)
        elif ptype == 'ramp':
            H_train[:, j+2] = gate_ramp(train_x, bp, s_j)
    
    r_train = train_y - train_x
    reg = alpha * len(train_x) * np.eye(n_basis)
    reg[0, 0] = 0
    reg[1, 1] = 0
    w = np.linalg.solve(H_train.T @ H_train + reg, H_train.T @ r_train)
    
    # Predict
    H_test = np.zeros((len(test_x), n_basis))
    H_test[:, 0] = 1.0
    H_test[:, 1] = test_x
    for j, ((ptype, bp, _), s_j) in enumerate(zip(all_primitives, primitive_sharpness)):
        if ptype == 'step':
            H_test[:, j+2] = gate_step(test_x, bp, s_j)
        elif ptype == 'ramp':
            H_test[:, j+2] = gate_ramp(test_x, bp, s_j)
    
    return test_x + H_test @ w


# ============================================================================
# APPROACH 6: Slope-tol sweep (find best slope_tol per function)
# ============================================================================

def approach_autotune(train_x, train_y, test_x, s=None, alpha=0.001):
    """Try multiple slope tolerances and pick best on training data.
    
    The slope_tol parameter controls segment detection sensitivity.
    Too low → over-segments (noise). Too high → under-segments (misses).
    Sweep and pick the one that minimizes training error.
    """
    s = s or PHI ** 2
    
    best_pred = None
    best_train_err = np.inf
    best_tol = None
    
    for slope_tol in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
        try:
            pred = approach_full_pipeline(train_x, train_y, test_x, s,
                                           slope_tol=slope_tol, alpha=alpha)
            # Evaluate on training data
            train_pred = approach_full_pipeline(train_x, train_y, train_x, s,
                                                slope_tol=slope_tol, alpha=alpha)
            train_err = np.sum((train_pred - train_y) ** 2)
            
            if train_err < best_train_err:
                best_train_err = train_err
                best_pred = pred
                best_tol = slope_tol
        except Exception:
            continue
    
    return best_pred if best_pred is not None else test_x.copy(), best_tol


# ============================================================================
# BENCHMARK
# ============================================================================

print("=" * 70)
print("PINNED LEARNING v2.1: Refined Geometric Program Synthesis")
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

approaches = {
    'hinge':        lambda tx, ty, sx: approach_hinge(tx, ty, sx),
    'hinge+ridge':  lambda tx, ty, sx: approach_hinge_ridge(tx, ty, sx),
    'iterative':    lambda tx, ty, sx: approach_iterative(tx, ty, sx),
    'adaptive_s':   lambda tx, ty, sx: approach_adaptive_sharpness(tx, ty, sx),
    'full_pipe':    lambda tx, ty, sx: approach_full_pipeline(tx, ty, sx),
}

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
    
    for aname, afn in approaches.items():
        t0 = time.perf_counter()
        try:
            pred = afn(train_x, train_y, test_x)
        except Exception as e:
            print(f"  {aname:<15s}: ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        
        exact = int(np.sum(np.round(pred) == test_y))
        max_err = float(np.abs(pred - test_y).max())
        results[aname] = (pred, exact, max_err, elapsed)
        
        print(f"  {aname:<15s}: {exact:>4d}/{input_range} exact, "
              f"max_err={max_err:>8.4f}, time={elapsed:.4f}s")
    
    # Autotune
    t0 = time.perf_counter()
    autotune_pred, best_tol = approach_autotune(train_x, train_y, test_x)
    elapsed = time.perf_counter() - t0
    exact_at = int(np.sum(np.round(autotune_pred) == test_y))
    max_err_at = float(np.abs(autotune_pred - test_y).max())
    results['autotune'] = (autotune_pred, exact_at, max_err_at, elapsed)
    print(f"  {'autotune':<15s}: {exact_at:>4d}/{input_range} exact, "
          f"max_err={max_err_at:>8.4f}, time={elapsed:.4f}s  (tol={best_tol})")
    
    all_results[fn_name] = results


# ============================================================================
# SAMPLE EFFICIENCY
# ============================================================================

print()
print("=" * 70)
print("SAMPLE EFFICIENCY: full_pipeline vs hinge")
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
        
        hinge_pred = approach_hinge(tx_s, ty_s, test_x)
        hinge_exact = int(np.sum(np.round(hinge_pred) == test_y))
        
        pipe_pred = approach_full_pipeline(tx_s, ty_s, test_x)
        pipe_exact = int(np.sum(np.round(pipe_pred) == test_y))
        
        fn_samples[n_s] = (hinge_exact, pipe_exact)
        print(f"    {n_s:>3d} examples: hinge={hinge_exact:>3d}/{input_range}  "
              f"full_pipe={pipe_exact:>3d}/{input_range}")
    
    sample_results[fn_name] = fn_samples


# ============================================================================
# ALPHA SWEEP
# ============================================================================

print()
print("=" * 70)
print("REGULARIZATION SWEEP: What alpha works best?")
print("=" * 70)

alpha_values = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
alpha_results = {}

for fn_name, fn, input_range, n_train in test_functions[:3]:
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)
    np.random.seed(42)
    train_x = np.sort(np.random.choice(input_range, n_train, replace=False)).astype(np.float64)
    train_y = fn(train_x)
    
    print(f"\n  {fn_name}:")
    fn_alpha = {}
    for a_val in alpha_values:
        pred = approach_hinge_ridge(train_x, train_y, test_x, alpha=a_val)
        exact = int(np.sum(np.round(pred) == test_y))
        max_err = float(np.abs(pred - test_y).max())
        fn_alpha[a_val] = (exact, max_err)
        print(f"    α={a_val:<8.4f}: {exact:>4d}/{input_range} exact, max_err={max_err:.4f}")
    alpha_results[fn_name] = fn_alpha


# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle('Pinned Learning v2.1: Refined Geometric Program Synthesis',
             fontsize=15, fontweight='bold', y=1.01)

colors = {'hinge': '#F44336', 'hinge+ridge': '#2196F3', 'iterative': '#4CAF50',
          'adaptive_s': '#FF9800', 'full_pipe': '#9C27B0', 'autotune': '#00BCD4'}

# Row 1-2: Function plots (show top 3 approaches per function)
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

# Row 3: Accuracy bar chart + Sample efficiency + Alpha sweep
ax_acc = fig.add_subplot(gs[2, 0:2])
approach_names = list(approaches.keys()) + ['autotune']
approach_labels = ['Hinge', 'Hinge\n+Ridge', 'Iterative', 'Adaptive\nSharpness',
                   'Full\nPipeline', 'Auto-\ntuned']
bar_colors = [colors.get(n, 'gray') for n in approach_names]

avg_accs = []
for aname in approach_names:
    accs = []
    for fn_name, _, input_range, _ in test_functions:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            accs.append(exact / input_range * 100)
    avg_accs.append(np.mean(accs) if accs else 0)

bars = ax_acc.bar(approach_labels, avg_accs, color=bar_colors, alpha=0.85)
for bar, acc in zip(bars, avg_accs):
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax_acc.set_ylabel('Average accuracy (%)')
ax_acc.set_title('Accuracy: Average across 8 functions', fontweight='bold')
ax_acc.set_ylim(0, 110)
ax_acc.grid(True, alpha=0.3, axis='y')

# Sample efficiency
ax_se = fig.add_subplot(gs[2, 2:4])
se_colors = {'tolower': 'blue', 'secret_fn': 'red', 'ROT13': 'green',
             'abs_centered': 'orange', 'sawtooth_32': 'purple'}
for fn_name in list(sample_results.keys()):
    results = sample_results[fn_name]
    ns = sorted(results.keys())
    hinge_accs = [results[n][0] for n in ns]
    pipe_accs = [results[n][1] for n in ns]
    inp = 100 if fn_name == 'secret_fn' else 128
    
    ax_se.plot(ns, pipe_accs, '-o', color=se_colors[fn_name], linewidth=2,
               markersize=5, label=f'pipe {fn_name}')
    ax_se.plot(ns, hinge_accs, '--s', color=se_colors[fn_name], linewidth=1,
               markersize=4, alpha=0.4, label=f'hinge {fn_name}')

ax_se.axhline(128, color='gray', linestyle=':', alpha=0.5)
ax_se.set_xlabel('Training examples')
ax_se.set_ylabel('Exact matches')
ax_se.set_title('Sample Efficiency: full_pipeline vs hinge', fontweight='bold')
ax_se.legend(fontsize=7, ncol=2)
ax_se.grid(True, alpha=0.3)

# Row 4: Heatmap + Alpha sweep + Insight
ax_hm = fig.add_subplot(gs[3, 0:2])
heatmap_data = []
fn_names_list = [fn_name for fn_name, _, _, _ in test_functions]
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

hm = ax_hm.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax_hm.set_xticks(range(len(approach_names)))
ax_hm.set_xticklabels(approach_labels, fontsize=8)
ax_hm.set_yticks(range(len(fn_names_list)))
ax_hm.set_yticklabels(fn_names_list, fontsize=9)
for i in range(len(fn_names_list)):
    for j in range(len(approach_names)):
        ax_hm.text(j, i, f'{heatmap_data[i][j]:.0f}', ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if heatmap_data[i][j] < 50 else 'black')
plt.colorbar(hm, ax=ax_hm, shrink=0.8, label='Accuracy %')
ax_hm.set_title('Per-Function Accuracy Heatmap', fontweight='bold')

# Alpha sweep plot
ax_alpha = fig.add_subplot(gs[3, 2])
for fn_name in alpha_results:
    alphas = sorted(alpha_results[fn_name].keys())
    accs = [alpha_results[fn_name][a][0] for a in alphas]
    inp = 100 if fn_name == 'secret_fn' else 128
    ax_alpha.semilogx([max(a, 1e-5) for a in alphas], accs,
                      '-o', label=fn_name, linewidth=2, markersize=5)
ax_alpha.set_xlabel('Regularization α')
ax_alpha.set_ylabel('Exact matches')
ax_alpha.set_title('Ridge α Sweep', fontweight='bold')
ax_alpha.legend(fontsize=8)
ax_alpha.grid(True, alpha=0.3)

# Insight panel
ax_ins = fig.add_subplot(gs[3, 3])
ax_ins.axis('off')
insight = (
    "v2.1 IMPROVEMENTS\n"
    "═══════════════════════\n\n"
    "KEY INSIGHT:\n"
    "  Separate DETECTION from\n"
    "  FITTING. Hinge decomp\n"
    "  finds WHERE breakpoints\n"
    "  are. Ridge regression\n"
    "  finds OPTIMAL weights.\n\n"
    "ITERATIVE REFINEMENT:\n"
    "  After initial fit, analyze\n"
    "  residual for missed breaks.\n"
    "  Catches subtle transitions.\n\n"
    "ADAPTIVE SHARPNESS:\n"
    "  Steps → high s (sharp)\n"
    "  Ramps → base s (smooth)\n"
    "  Prevents boundary errors.\n\n"
    "AUTOTUNING:\n"
    "  Sweep slope_tol, pick best\n"
    "  on training data."
)
ax_ins.text(0.05, 0.95, insight, transform=ax_ins.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/tmp/geometric_pinned_v2_refined.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print("=" * 70)
print("FINAL SUMMARY: v2.1 Refined")
print("=" * 70)

print(f"\n  {'Function':<15s}", end="")
for label in approach_labels:
    print(f"  {label.replace(chr(10),' '):>10s}", end="")
print()
print(f"  {'-'*15}", end="")
for _ in approach_names:
    print(f"  {'-'*10}", end="")
print()

for fn_name, _, input_range, _ in test_functions:
    print(f"  {fn_name:<15s}", end="")
    for aname in approach_names:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            print(f"  {exact:>4d}/{input_range:<3d} ", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    print()

print(f"\n  {'AVERAGE':<15s}", end="")
for aname in approach_names:
    accs = []
    for fn_name, _, input_range, _ in test_functions:
        if fn_name in all_results and aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            accs.append(exact / input_range * 100)
    avg = np.mean(accs) if accs else 0
    print(f"  {avg:>7.1f}%  ", end="")
print()

# Best per function
print(f"\n  BEST approach per function:")
for fn_name, _, input_range, _ in test_functions:
    if fn_name in all_results:
        best = max(all_results[fn_name].items(), key=lambda x: (x[1][1], -x[1][2]))
        print(f"    {fn_name:<15s}: {best[0]:<15s} {best[1][1]:>4d}/{input_range}")

# Perfect scores
print(f"\n  PERFECT SCORES (100%):")
for fn_name, _, input_range, _ in test_functions:
    if fn_name in all_results:
        for aname, (_, exact, _, _) in all_results[fn_name].items():
            if exact == input_range:
                print(f"    {fn_name:<15s}: {aname}")

print(f"\nSaved: /tmp/geometric_pinned_v2_refined.png")
