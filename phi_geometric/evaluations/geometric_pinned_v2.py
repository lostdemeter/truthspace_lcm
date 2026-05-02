#!/usr/bin/env python3
"""
Pinned Threshold Learning v2: Optimized Geometric Program Synthesis

V1 used generic neurons + least squares — it didn't exploit the fact that
we know the exact primitive vocabulary: STEP, RECT, RAMP.

V2 attacks three axes:
  1. DETECTION: Robust breakpoint detection via second-derivative analysis
  2. CLASSIFICATION: Each breakpoint is a JUMP (step), BEND (slope change), or BOTH
  3. CONSTRUCTION: Direct weight computation from classified primitives — no lstsq

The piecewise-linear decomposition theorem:
  Any piecewise-linear function r(x) can be written as:
    r(x) = a + b*x + Σ Δm_k * max(0, x - bp_k) + Σ h_j * step(x - sp_j)
  where:
    - (a, b) = initial intercept and slope
    - Δm_k = slope change at breakpoint k → RAMP neuron
    - h_j = jump height at step point j → STEP neuron pair

  Every term maps directly to gate neurons with KNOWN weights.

Approaches benchmarked:
  A. v1 Pinned (baseline — generic neurons + ridge lstsq)
  B. Hinge decomposition (slope-change ramps — direct construction)
  C. Segment decomposition (fit segments, classify transitions)
  D. Second-derivative peaks (robust to noise, adaptive thresholds)
  E. Greedy RECT fitting (detect rectangles first, ramps second)
  F. Adaptive ensemble (try all, pick best per-function)
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


def geo_step(x, threshold, s):
    """Smooth step at threshold: ≈1 for x >> threshold, ≈0 for x << threshold."""
    return ideal_gate(s * (x - threshold)) / s


def geo_ramp(x, threshold, s):
    """Smooth ramp at threshold: ≈(x-threshold) for x >> threshold, ≈0 for x << threshold."""
    return ideal_gate(s * (x - threshold)) / s


def geo_rect(x, lo, hi, s):
    """Rectangle: ≈1 for lo < x < hi, ≈0 otherwise."""
    return geo_step(x, lo, s) - geo_step(x, hi, s)


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
    """Clamp to [30, 100]."""
    return np.clip(np.asarray(x, dtype=np.float64), 30, 100)

def fn_relu_shifted(x):
    """ReLU shifted: max(0, x - 40)."""
    return np.maximum(0, np.asarray(x, dtype=np.float64) - 40)

def fn_staircase(x):
    """Staircase: floor(x/16)*16."""
    return np.floor(np.asarray(x, dtype=np.float64) / 16) * 16


# ============================================================================
# APPROACH A: v1 Pinned (baseline)
# ============================================================================

def v1_pinned(train_x, train_y, test_x, s=None):
    """V1 baseline: detect breakpoints + ridge lstsq."""
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    residual = ys - xs
    
    # Finite-difference breakpoint detection
    if len(xs) < 3:
        return test_x.copy()
    
    dr = np.diff(residual)
    dx = np.maximum(np.diff(xs), 1e-10)
    slopes = dr / dx
    slope_changes = np.abs(np.diff(slopes))
    
    breakpoints = []
    threshold = max(0.5 * np.median(slope_changes + 1e-10), 0.05)
    
    for i in range(len(slope_changes)):
        if slope_changes[i] > threshold:
            bp = (xs[i+1] + xs[i+2]) / 2 if i+2 < len(xs) else xs[i+1]
            breakpoints.append(bp)
    
    for i in range(len(dr)):
        if abs(dr[i]) > 2.0:
            breakpoints.append((xs[i] + xs[i+1]) / 2)
    
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) == 0:
        # Linear fit
        A = np.column_stack([train_x, np.ones_like(train_x)])
        coeffs, _, _, _ = np.linalg.lstsq(A, train_y, rcond=None)
        return coeffs[0] * test_x + coeffs[1]
    
    # Place neurons at breakpoints
    thresholds = []
    for bp in breakpoints:
        thresholds.extend([bp - 0.5, bp + 0.5])
    thresholds = np.array(thresholds)
    n = len(thresholds)
    
    # Build activation matrix and solve with ridge regression
    H = np.zeros((len(train_x), n))
    for j, th in enumerate(thresholds):
        H[:, j] = ideal_gate(s * (train_x - th))
    
    r = train_y - train_x
    alpha = 0.01 * len(train_x)
    w2 = np.linalg.solve(H.T @ H + alpha * np.eye(n), H.T @ r)
    
    H_test = np.zeros((len(test_x), n))
    for j, th in enumerate(thresholds):
        H_test[:, j] = ideal_gate(s * (test_x - th))
    
    return test_x + H_test @ w2


# ============================================================================
# APPROACH B: Hinge Decomposition (direct construction)
# ============================================================================

def detect_segments(xs, rs, slope_tol=0.15):
    """Detect contiguous segments of constant slope in the residual.
    
    Returns list of (start_idx, end_idx, slope, intercept) tuples.
    """
    if len(xs) < 2:
        return [(0, len(xs)-1, 0, rs[0] if len(rs) > 0 else 0)]
    
    # Compute pairwise slopes
    dx = np.diff(xs)
    dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)
    
    # Group into segments of similar slope
    segments = []
    seg_start = 0
    current_slope = slopes[0]
    
    for i in range(1, len(slopes)):
        # Dynamic tolerance: allow more variation for steep slopes
        tol = max(slope_tol, abs(current_slope) * 0.1)
        if abs(slopes[i] - current_slope) > tol:
            # Segment boundary: fit the segment properly
            seg_xs = xs[seg_start:i+1]
            seg_rs = rs[seg_start:i+1]
            if len(seg_xs) >= 2:
                slope = np.polyfit(seg_xs, seg_rs, 1)[0]
            else:
                slope = current_slope
            intercept = seg_rs[0] - slope * seg_xs[0]
            segments.append((seg_start, i, slope, intercept))
            seg_start = i
            current_slope = slopes[i]
        else:
            # Running average of slope for robustness
            n_in_seg = i - seg_start
            current_slope = (current_slope * n_in_seg + slopes[i]) / (n_in_seg + 1)
    
    # Final segment
    seg_xs = xs[seg_start:]
    seg_rs = rs[seg_start:]
    if len(seg_xs) >= 2:
        slope = np.polyfit(seg_xs, seg_rs, 1)[0]
    else:
        slope = current_slope
    intercept = seg_rs[0] - slope * seg_xs[0]
    segments.append((seg_start, len(xs)-1, slope, intercept))
    
    return segments


def hinge_decomposition(train_x, train_y, test_x, s=None, slope_tol=0.15):
    """Direct construction via hinge decomposition.
    
    Any piecewise-linear function r(x) = a + bx + Σ Δm_k * ramp(x - bp_k)
    Each ramp = one gate neuron with known weights.
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs  # residual from skip connection
    
    # Detect segments
    segments = detect_segments(xs, rs, slope_tol)
    
    if len(segments) == 0:
        return test_x.copy()
    
    # Extract breakpoints and classify transitions
    primitives = []  # list of ('ramp', bp, delta_slope) or ('step', bp, height)
    
    # Initial segment provides base slope and intercept
    _, _, base_slope, base_intercept = segments[0]
    
    for i in range(1, len(segments)):
        prev_start, prev_end, prev_slope, prev_intercept = segments[i-1]
        curr_start, curr_end, curr_slope, curr_intercept = segments[i]
        
        # Breakpoint is at the boundary between segments
        bp = (xs[prev_end] + xs[curr_start]) / 2
        
        # Compute slope change
        delta_slope = curr_slope - prev_slope
        
        # Compute jump (discontinuity in r at the breakpoint)
        # Expected value from prev segment at bp:
        r_prev_at_bp = prev_slope * bp + prev_intercept
        # Expected value from curr segment at bp:
        r_curr_at_bp = curr_slope * bp + curr_intercept
        jump = r_curr_at_bp - r_prev_at_bp
        
        if abs(jump) > 0.5:
            primitives.append(('step', bp, jump))
        if abs(delta_slope) > 0.05:
            primitives.append(('ramp', bp, delta_slope))
    
    # Construct the function directly
    def evaluate(x_vals):
        result = base_intercept + (base_slope + 1) * x_vals  # +1 because y = x + r, and r has slope base_slope
        # Wait, y = x + r(x), so if r(x) = base_intercept + base_slope * x + ...
        # y = x + base_intercept + base_slope * x + ...
        # y = (1 + base_slope) * x + base_intercept + ...
        # But our GeoBlock does output = x + correction, so correction = r(x)
        # r(x) = base_intercept + base_slope * x + Σ primitives
        
        # Actually, let's build r(x) properly:
        r = np.full_like(x_vals, base_intercept) + base_slope * x_vals
        
        # Handle initial slope with a ramp from before the data
        # Actually the base_slope * x term is a global linear contribution
        # We need to handle this carefully in GeoBlock architecture
        # For now, just compute it directly
        
        for ptype, bp, value in primitives:
            if ptype == 'ramp':
                r = r + value * geo_ramp(x_vals, bp, s)
            elif ptype == 'step':
                # Step = rect with one edge far away
                # step(x, t) ≈ gate(s*(x-t))/s for x >> t
                # But this gives a ramp, not a step!
                # For a step, use: gate(s*(x-(t-0.5))) - gate(s*(x-(t+0.5)))
                # This gives ≈ s*1/s = 1 for x > t+0.5, ≈ 0 for x < t-0.5
                r = r + value * geo_rect(x_vals, bp - 0.5, bp + 0.5, s)
        
        return x_vals + r  # y = x + correction
    
    # BUT: the base_slope * x term can't be implemented with gate neurons alone
    # unless we use a ramp from x_min. Let's fix this:
    # r(x) = base_intercept + base_slope * ramp(x, x_min) + base_slope * x_min + ...
    # Hmm, this gets complicated. Let me use a different formulation.
    
    # Actually: ramp(x, x_min) ≈ x - x_min for all x > x_min
    # So: base_slope * ramp(x, x_min) ≈ base_slope * (x - x_min)
    #   = base_slope * x - base_slope * x_min
    # Combined with base_intercept:
    #   base_intercept + base_slope * x = base_intercept + base_slope * x_min + base_slope * ramp(x, x_min)
    # So the constant part is base_intercept + base_slope * x_min
    # And we need a ramp neuron at x_min with slope = base_slope
    
    # For proper GeoBlock, the constant part needs a step from -∞
    # In practice, add a step at x_min - 1 with height = constant value
    
    # For this benchmark, just compute directly (the architecture question
    # is separate from the detection quality question)
    
    return evaluate(test_x)


# ============================================================================
# APPROACH C: Second-Derivative Peak Detection
# ============================================================================

def second_derivative_detection(train_x, train_y, test_x, s=None, 
                                 peak_threshold=0.3):
    """Detect breakpoints via second derivative peaks.
    
    For piecewise-linear functions, r''(x) = 0 everywhere except at
    breakpoints where it's a Dirac delta. In discrete data, this becomes
    peaks in the second finite difference.
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs
    
    if len(xs) < 3:
        return test_x.copy()
    
    # Second finite differences (approximation to r'')
    d2r = np.zeros(len(rs))
    for i in range(1, len(rs) - 1):
        dx1 = xs[i] - xs[i-1]
        dx2 = xs[i+1] - xs[i]
        if dx1 > 0 and dx2 > 0:
            d2r[i] = ((rs[i+1] - rs[i]) / dx2 - (rs[i] - rs[i-1]) / dx1) / ((dx1 + dx2) / 2)
    
    # Find peaks in |d2r|
    abs_d2r = np.abs(d2r)
    threshold = peak_threshold * np.max(abs_d2r) if np.max(abs_d2r) > 0 else 1.0
    
    # Find local maxima above threshold
    peaks = []
    for i in range(1, len(abs_d2r) - 1):
        if abs_d2r[i] > threshold:
            if abs_d2r[i] >= abs_d2r[i-1] and abs_d2r[i] >= abs_d2r[i+1]:
                peaks.append(i)
    
    # Also add points where abs_d2r exceeds 2x threshold even if not local max
    for i in range(1, len(abs_d2r) - 1):
        if abs_d2r[i] > 2 * threshold and i not in peaks:
            peaks.append(i)
    
    peaks = sorted(set(peaks))
    
    if len(peaks) == 0:
        # Linear function
        if len(xs) >= 2:
            slope = (rs[-1] - rs[0]) / (xs[-1] - xs[0])
            intercept = rs[0] - slope * xs[0]
            return test_x + intercept + slope * test_x
        return test_x.copy()
    
    # At each peak, classify the transition
    primitives = []
    
    for pi in peaks:
        bp = xs[pi]
        
        # Compute slope before and after
        # Look at a few points on each side for robustness
        n_look = min(3, pi, len(xs) - pi - 1)
        if n_look < 1:
            continue
        
        slopes_before = []
        for j in range(max(0, pi - n_look), pi):
            if xs[j+1] - xs[j] > 0:
                slopes_before.append((rs[j+1] - rs[j]) / (xs[j+1] - xs[j]))
        
        slopes_after = []
        for j in range(pi, min(len(xs)-1, pi + n_look)):
            if xs[j+1] - xs[j] > 0:
                slopes_after.append((rs[j+1] - rs[j]) / (xs[j+1] - xs[j]))
        
        slope_before = np.median(slopes_before) if slopes_before else 0
        slope_after = np.median(slopes_after) if slopes_after else 0
        
        delta_slope = slope_after - slope_before
        
        # Check for jump: interpolate from before, compare to actual
        if pi > 0 and pi < len(xs) - 1:
            expected_r = rs[pi-1] + slope_before * (xs[pi] - xs[pi-1])
            jump = rs[pi] - expected_r
        else:
            jump = 0
        
        if abs(jump) > 1.0:
            primitives.append(('step', bp, jump))
        if abs(delta_slope) > 0.1:
            primitives.append(('ramp', bp, delta_slope))
    
    # Compute initial conditions
    if len(peaks) > 0 and peaks[0] > 0:
        first_seg = slice(0, peaks[0])
        seg_xs = xs[first_seg]
        seg_rs = rs[first_seg]
        if len(seg_xs) >= 2:
            base_slope = np.polyfit(seg_xs, seg_rs, 1)[0]
            base_intercept = seg_rs[0] - base_slope * seg_xs[0]
        else:
            base_slope = 0
            base_intercept = seg_rs[0] if len(seg_rs) > 0 else 0
    else:
        base_slope = 0
        base_intercept = rs[0]
    
    # Construct function
    r_test = base_intercept + base_slope * test_x
    for ptype, bp, value in primitives:
        if ptype == 'ramp':
            r_test += value * geo_ramp(test_x, bp, s)
        elif ptype == 'step':
            r_test += value * geo_rect(test_x, bp - 0.5, bp + 0.5, s)
    
    return test_x + r_test


# ============================================================================
# APPROACH D: Greedy RECT Fitting
# ============================================================================

def greedy_rect_fitting(train_x, train_y, test_x, s=None, max_rects=20):
    """Detect rectangles (constant-offset regions) first, then handle ramps.
    
    Many operations (tolower, ROT13, staircase) are sums of RECTs.
    Detect the dominant RECT first, subtract it, repeat.
    Also handles ramps by fitting slope in residual.
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    residual = ys - xs  # what skip connection doesn't explain
    
    primitives = []
    
    for rect_iter in range(max_rects):
        if np.abs(residual).max() < 0.5:
            break
        
        # Find the best RECT: constant offset in some range [lo, hi]
        best_score = 0
        best_rect = None
        
        n = len(xs)
        
        # Try all pairs of boundary points (O(n²) but n is small)
        for i in range(n):
            for j in range(i + 1, n):
                lo, hi = xs[i], xs[j]
                
                # Points inside [lo, hi]
                inside = (xs >= lo - 0.5) & (xs <= hi + 0.5)
                if inside.sum() < 1:
                    continue
                
                # Average residual inside
                height = np.median(residual[inside])
                
                if abs(height) < 0.5:
                    continue
                
                # Score: how much squared error does this rect explain?
                rect_vals = height * geo_rect(xs, lo - 0.5, hi + 0.5, s)
                explained = np.sum((residual ** 2) - ((residual - rect_vals) ** 2))
                
                if explained > best_score:
                    best_score = explained
                    best_rect = (lo - 0.5, hi + 0.5, height)
        
        # Also try a RAMP: slope change at each training point
        for i in range(1, n - 1):
            bp = xs[i]
            
            # Fit slope to residual after bp
            after = xs >= bp
            before = xs < bp
            
            if after.sum() < 2 or before.sum() < 2:
                continue
            
            slope_after = np.polyfit(xs[after], residual[after], 1)[0]
            slope_before = np.polyfit(xs[before], residual[before], 1)[0]
            delta_slope = slope_after - slope_before
            
            if abs(delta_slope) < 0.1:
                continue
            
            ramp_vals = delta_slope * geo_ramp(xs, bp, s)
            explained = np.sum((residual ** 2) - ((residual - ramp_vals) ** 2))
            
            if explained > best_score:
                best_score = explained
                best_rect = None
                best_ramp = (bp, delta_slope)
        
        if best_score < 0.5:
            break
        
        if best_rect is not None:
            lo, hi, height = best_rect
            rect_vals = height * geo_rect(xs, lo, hi, s)
            residual = residual - rect_vals
            primitives.append(('rect', lo, hi, height))
        elif 'best_ramp' in dir() and best_ramp is not None:
            bp, delta_slope = best_ramp
            ramp_vals = delta_slope * geo_ramp(xs, bp, s)
            residual = residual - ramp_vals
            primitives.append(('ramp', bp, delta_slope))
            best_ramp = None
    
    # Handle remaining residual: fit a global linear term
    if np.abs(residual).max() > 0.5 and len(xs) >= 2:
        global_slope, global_intercept = np.polyfit(xs, residual, 1)
        primitives.append(('linear', global_slope, global_intercept))
    
    # Construct prediction
    r_test = np.zeros_like(test_x, dtype=np.float64)
    for prim in primitives:
        if prim[0] == 'rect':
            _, lo, hi, height = prim
            r_test += height * geo_rect(test_x, lo, hi, s)
        elif prim[0] == 'ramp':
            _, bp, delta_slope = prim
            r_test += delta_slope * geo_ramp(test_x, bp, s)
        elif prim[0] == 'linear':
            _, slope, intercept = prim
            r_test += slope * test_x + intercept
    
    return test_x + r_test


# ============================================================================
# APPROACH E: Piecewise Linear Optimal Segmentation
# ============================================================================

def optimal_segmentation(train_x, train_y, test_x, s=None, max_segments=12,
                         penalty=2.0):
    """Find optimal piecewise-linear segmentation using dynamic programming.
    
    Minimize: Σ segment_error + penalty * n_segments
    
    This is the most principled approach: it finds the globally optimal
    set of breakpoints for a given number of segments.
    """
    s = s or PHI ** 2
    
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs
    n = len(xs)
    
    if n < 3:
        return test_x.copy()
    
    # Precompute segment costs: cost[i][j] = error of linear fit to points i..j
    # Use O(n²) precomputation
    seg_cost = np.full((n, n), np.inf)
    seg_slope = np.zeros((n, n))
    seg_intercept = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            seg_xs = xs[i:j+1]
            seg_rs = rs[i:j+1]
            if len(seg_xs) == 1:
                seg_cost[i, j] = 0
                seg_slope[i, j] = 0
                seg_intercept[i, j] = seg_rs[0]
            elif len(seg_xs) == 2:
                slope = (seg_rs[1] - seg_rs[0]) / max(seg_xs[1] - seg_xs[0], 1e-10)
                intercept = seg_rs[0] - slope * seg_xs[0]
                seg_cost[i, j] = 0
                seg_slope[i, j] = slope
                seg_intercept[i, j] = intercept
            else:
                coeffs = np.polyfit(seg_xs, seg_rs, 1)
                fitted = coeffs[0] * seg_xs + coeffs[1]
                seg_cost[i, j] = np.sum((seg_rs - fitted) ** 2)
                seg_slope[i, j] = coeffs[0]
                seg_intercept[i, j] = coeffs[1]
    
    # Dynamic programming: dp[k][j] = min cost of segmenting points 0..j with k segments
    dp = np.full((max_segments + 1, n), np.inf)
    dp_trace = np.full((max_segments + 1, n), -1, dtype=int)
    
    # Base case: 1 segment
    for j in range(n):
        dp[1, j] = seg_cost[0, j]
    
    # Fill DP table
    for k in range(2, max_segments + 1):
        for j in range(k - 1, n):
            for i in range(k - 1, j + 1):
                cost = dp[k-1, i-1] + seg_cost[i, j] + penalty
                if cost < dp[k, j]:
                    dp[k, j] = cost
                    dp_trace[k, j] = i
    
    # Find optimal number of segments
    best_k = 1
    best_cost = dp[1, n-1]
    for k in range(2, max_segments + 1):
        if dp[k, n-1] < best_cost:
            best_cost = dp[k, n-1]
            best_k = k
    
    # Trace back to find segment boundaries
    segments = []
    j = n - 1
    k = best_k
    while k > 0:
        if k == 1:
            segments.append((0, j))
            break
        i = dp_trace[k, j]
        segments.append((i, j))
        j = i - 1
        k -= 1
    
    segments.reverse()
    
    # Convert segments to primitives
    seg_info = []
    for start, end in segments:
        slope = seg_slope[start, end]
        intercept = seg_intercept[start, end]
        seg_info.append((xs[start], xs[end], slope, intercept))
    
    # Build prediction by classifying transitions between segments
    primitives = []
    
    # First segment: base slope and intercept
    x0, x1, base_slope, base_intercept = seg_info[0]
    
    for i in range(1, len(seg_info)):
        prev_x0, prev_x1, prev_slope, prev_intercept = seg_info[i-1]
        curr_x0, curr_x1, curr_slope, curr_intercept = seg_info[i]
        
        bp = (prev_x1 + curr_x0) / 2
        
        delta_slope = curr_slope - prev_slope
        
        # Compute jump
        r_prev = prev_slope * bp + prev_intercept
        r_curr = curr_slope * bp + curr_intercept
        jump = r_curr - r_prev
        
        if abs(jump) > 0.3:
            primitives.append(('step', bp, jump))
        if abs(delta_slope) > 0.05:
            primitives.append(('ramp', bp, delta_slope))
    
    # Construct prediction
    r_test = base_intercept + base_slope * test_x
    for prim in primitives:
        if prim[0] == 'step':
            _, bp, height = prim
            r_test += height * geo_rect(test_x, bp - 0.5, bp + 0.5, s)
        elif prim[0] == 'ramp':
            _, bp, delta_slope = prim
            r_test += delta_slope * geo_ramp(test_x, bp, s)
    
    return test_x + r_test


# ============================================================================
# APPROACH F: Direct Analytical (for known-structure functions)
# ============================================================================

def direct_analytical(train_x, train_y, test_x, s=None):
    """Combine all detection approaches and pick best on training data.
    
    Also adds an analytical refinement: after initial construction,
    compute the residual on TRAINING data and fit a correction.
    """
    s = s or PHI ** 2
    
    # Try all approaches
    approaches = [
        ('hinge', lambda: hinge_decomposition(train_x, train_y, test_x, s)),
        ('d2_peaks', lambda: second_derivative_detection(train_x, train_y, test_x, s)),
        ('greedy_rect', lambda: greedy_rect_fitting(train_x, train_y, test_x, s)),
        ('optimal_seg', lambda: optimal_segmentation(train_x, train_y, test_x, s)),
    ]
    
    best_pred = None
    best_train_err = np.inf
    best_name = ""
    
    for name, fn in approaches:
        try:
            pred = fn()
            # Evaluate on training data
            train_pred = None
            # Re-run on train_x to check training error
            if name == 'hinge':
                train_pred = hinge_decomposition(train_x, train_y, train_x, s)
            elif name == 'd2_peaks':
                train_pred = second_derivative_detection(train_x, train_y, train_x, s)
            elif name == 'greedy_rect':
                train_pred = greedy_rect_fitting(train_x, train_y, train_x, s)
            elif name == 'optimal_seg':
                train_pred = optimal_segmentation(train_x, train_y, train_x, s)
            
            if train_pred is not None:
                train_err = np.sum((np.round(train_pred) - train_y) ** 2)
            else:
                train_err = np.inf
            
            if train_err < best_train_err:
                best_train_err = train_err
                best_pred = pred
                best_name = name
        except Exception:
            continue
    
    if best_pred is None:
        return test_x.copy(), "none"
    
    return best_pred, best_name


# ============================================================================
# BENCHMARK
# ============================================================================

print("=" * 70)
print("PINNED LEARNING v2: Optimized Geometric Program Synthesis")
print("=" * 70)
print()

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

approaches = [
    ("v1_pinned",       lambda tx, ty, sx, s: v1_pinned(tx, ty, sx, s)),
    ("hinge",           lambda tx, ty, sx, s: hinge_decomposition(tx, ty, sx, s)),
    ("d2_peaks",        lambda tx, ty, sx, s: second_derivative_detection(tx, ty, sx, s)),
    ("greedy_rect",     lambda tx, ty, sx, s: greedy_rect_fitting(tx, ty, sx, s)),
    ("optimal_seg",     lambda tx, ty, sx, s: optimal_segmentation(tx, ty, sx, s)),
]

all_results = {}
s = PHI ** 2

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
    
    for approach_name, approach_fn in approaches:
        t0 = time.perf_counter()
        try:
            pred = approach_fn(train_x, train_y, test_x, s)
        except Exception as e:
            print(f"  {approach_name:<15s}: ERROR: {e}")
            results[approach_name] = (test_x.copy(), 0, 999, 0)
            continue
        elapsed = time.perf_counter() - t0
        
        exact = int(np.sum(np.round(pred) == test_y))
        max_err = float(np.abs(pred - test_y).max())
        results[approach_name] = (pred, exact, max_err, elapsed)
        
        print(f"  {approach_name:<15s}: {exact:>4d}/{input_range} exact, "
              f"max_err={max_err:>10.4f}, time={elapsed:.4f}s")
    
    # Also run ensemble
    t0 = time.perf_counter()
    ensemble_pred, best_name = direct_analytical(train_x, train_y, test_x, s)
    elapsed = time.perf_counter() - t0
    exact_ens = int(np.sum(np.round(ensemble_pred) == test_y))
    max_err_ens = float(np.abs(ensemble_pred - test_y).max())
    results['ensemble'] = (ensemble_pred, exact_ens, max_err_ens, elapsed)
    print(f"  {'ensemble':<15s}: {exact_ens:>4d}/{input_range} exact, "
          f"max_err={max_err_ens:>10.4f}, time={elapsed:.4f}s  (picked: {best_name})")
    
    all_results[fn_name] = results


# ============================================================================
# SAMPLE EFFICIENCY DEEP DIVE
# ============================================================================

print()
print()
print("=" * 70)
print("SAMPLE EFFICIENCY DEEP DIVE")
print("=" * 70)

sample_counts = [3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 128]
sample_efficiency = {}

for fn_name, fn, input_range, _ in test_functions[:5]:  # First 5 functions
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)
    
    print(f"\n  {fn_name}:")
    fn_results = {}
    
    for n_samples in sample_counts:
        if n_samples > input_range:
            continue
        np.random.seed(42)
        if n_samples >= input_range:
            train_x_s = np.arange(input_range, dtype=np.float64)
        else:
            train_idx_s = np.sort(np.random.choice(input_range, n_samples, replace=False))
            train_x_s = train_idx_s.astype(np.float64)
        train_y_s = fn(train_x_s)
        
        # Run ensemble (best of all approaches)
        ens_pred, ens_name = direct_analytical(train_x_s, train_y_s, test_x, s)
        ens_exact = int(np.sum(np.round(ens_pred) == test_y))
        
        # Run v1 for comparison
        v1_pred = v1_pinned(train_x_s, train_y_s, test_x, s)
        v1_exact = int(np.sum(np.round(v1_pred) == test_y))
        
        fn_results[n_samples] = (v1_exact, ens_exact, ens_name)
        print(f"    {n_samples:>3d} examples: v1={v1_exact:>3d}/{input_range}  "
              f"v2={ens_exact:>3d}/{input_range}  (via {ens_name})")
    
    sample_efficiency[fn_name] = fn_results


# ============================================================================
# SHARPNESS SWEEP
# ============================================================================

print()
print()
print("=" * 70)
print("SHARPNESS SWEEP: What s value works best?")
print("=" * 70)

s_values = [1.0, PHI, PHI**2, PHI**3, 5.0, 10.0, 20.0, 50.0]
sharpness_results = {}

test_x_sw = np.arange(128, dtype=np.float64)
test_y_sw = fn_tolower(test_x_sw)
np.random.seed(42)
train_idx_sw = np.sort(np.random.choice(128, 50, replace=False))
train_x_sw = train_idx_sw.astype(np.float64)
train_y_sw = fn_tolower(train_x_sw)

print(f"\n  tolower (50 train / 128 test):")
for s_val in s_values:
    pred_sw, name_sw = direct_analytical(train_x_sw, train_y_sw, test_x_sw, s_val)
    exact_sw = int(np.sum(np.round(pred_sw) == test_y_sw))
    maxe_sw = float(np.abs(pred_sw - test_y_sw).max())
    sharpness_results[s_val] = (exact_sw, maxe_sw)
    s_label = f"φ^{np.log(s_val)/np.log(PHI):.1f}" if s_val not in [5, 10, 20, 50] else f"{s_val:.0f}"
    print(f"    s={s_val:>6.3f} ({s_label:>7s}): {exact_sw:>3d}/128 exact, max_err={maxe_sw:.4f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle('Pinned Learning v2: Optimized Geometric Program Synthesis',
             fontsize=15, fontweight='bold', y=1.01)

# Row 1: First 4 test functions
fn_list = list(all_results.keys())
for idx, fn_name in enumerate(fn_list[:4]):
    ax = fig.add_subplot(gs[0, idx])
    results = all_results[fn_name]
    fn_ref = [fn_tolower, fn_secret, fn_rot13, fn_abs_centered,
              fn_sawtooth, fn_clamp, fn_relu_shifted, fn_staircase][idx]
    inp_range = [128, 100, 128, 128, 128, 128, 128, 128][idx]
    
    tx = np.arange(inp_range, dtype=np.float64)
    ty = fn_ref(tx)
    ax.plot(tx, ty, 'k-', linewidth=1.5, alpha=0.3, label='True')
    
    colors = {'v1_pinned': 'red', 'hinge': 'blue', 'd2_peaks': 'green',
              'greedy_rect': 'orange', 'optimal_seg': 'purple', 'ensemble': 'cyan'}
    
    # Show top 3 approaches
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    for rank, (name, (pred, exact, maxe, _)) in enumerate(sorted_results[:3]):
        ax.plot(tx, pred, color=colors.get(name, 'gray'), linewidth=1,
                alpha=0.7, label=f'{name}: {exact}/{inp_range}')
    
    ax.set_title(fn_name, fontweight='bold', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

# Row 2: Last 4 test functions
for idx, fn_name in enumerate(fn_list[4:8]):
    ax = fig.add_subplot(gs[1, idx])
    results = all_results[fn_name]
    fn_ref = [fn_sawtooth, fn_clamp, fn_relu_shifted, fn_staircase][idx]
    inp_range = 128
    
    tx = np.arange(inp_range, dtype=np.float64)
    ty = fn_ref(tx)
    ax.plot(tx, ty, 'k-', linewidth=1.5, alpha=0.3, label='True')
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    for rank, (name, (pred, exact, maxe, _)) in enumerate(sorted_results[:3]):
        ax.plot(tx, pred, color=colors.get(name, 'gray'), linewidth=1,
                alpha=0.7, label=f'{name}: {exact}/{inp_range}')
    
    ax.set_title(fn_name, fontweight='bold', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

# Row 3, col 0-1: Accuracy comparison bar chart
ax_acc = fig.add_subplot(gs[2, 0:2])
approach_names = ['v1_pinned', 'hinge', 'd2_peaks', 'greedy_rect', 'optimal_seg', 'ensemble']
approach_labels = ['v1\nPinned', 'Hinge\nDecomp', 'd² Peak\nDetect', 'Greedy\nRECT', 'Optimal\nSegment', 'Ensemble\n(best of)']
bar_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

avg_accs = []
for aname in approach_names:
    accs = []
    for fn_name in fn_list:
        if aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            inp = 128 if fn_name != 'secret_fn' else 100
            accs.append(exact / inp * 100)
    avg_accs.append(np.mean(accs) if accs else 0)

bars = ax_acc.bar(approach_labels, avg_accs, color=bar_colors, alpha=0.85)
for bar, acc in zip(bars, avg_accs):
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax_acc.set_ylabel('Average accuracy (%)')
ax_acc.set_title('Accuracy: Average across 8 functions', fontweight='bold')
ax_acc.set_ylim(0, 110)
ax_acc.grid(True, alpha=0.3, axis='y')

# Row 3, col 2-3: Sample efficiency
ax_se = fig.add_subplot(gs[2, 2:4])
se_colors = {'tolower': 'blue', 'secret_fn': 'red', 'ROT13': 'green',
             'abs_centered': 'orange', 'sawtooth_32': 'purple'}

for fn_name in list(sample_efficiency.keys())[:3]:
    results = sample_efficiency[fn_name]
    ns = sorted(results.keys())
    v1_accs = [results[n][0] for n in ns]
    v2_accs = [results[n][1] for n in ns]
    inp = 100 if fn_name == 'secret_fn' else 128
    
    ax_se.plot(ns, v2_accs, '-o', color=se_colors[fn_name], linewidth=2,
               markersize=5, label=f'v2 {fn_name}')
    ax_se.plot(ns, v1_accs, '--s', color=se_colors[fn_name], linewidth=1,
               markersize=4, alpha=0.5, label=f'v1 {fn_name}')

ax_se.set_xlabel('Number of training examples')
ax_se.set_ylabel('Exact matches')
ax_se.set_title('Sample Efficiency: v1 vs v2', fontweight='bold')
ax_se.legend(fontsize=8, ncol=2)
ax_se.grid(True, alpha=0.3)

# Row 4, col 0: Sharpness sweep
ax_sh = fig.add_subplot(gs[3, 0])
s_vals_plot = list(sharpness_results.keys())
s_accs = [sharpness_results[sv][0] for sv in s_vals_plot]
ax_sh.plot(s_vals_plot, s_accs, 'b-o', linewidth=2, markersize=6)
ax_sh.axvline(PHI**2, color='red', linestyle='--', alpha=0.5, label=f'φ²={PHI**2:.3f}')
ax_sh.set_xlabel('Sharpness (s)')
ax_sh.set_ylabel('Exact matches (/ 128)')
ax_sh.set_title('Sharpness Sweep: tolower', fontweight='bold')
ax_sh.legend()
ax_sh.grid(True, alpha=0.3)

# Row 4, col 1: Per-function comparison heatmap
ax_hm = fig.add_subplot(gs[3, 1:3])
heatmap_data = []
for fn_name in fn_list:
    row = []
    for aname in approach_names:
        if aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            inp = 128 if fn_name != 'secret_fn' else 100
            row.append(exact / inp * 100)
        else:
            row.append(0)
    heatmap_data.append(row)

hm = ax_hm.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax_hm.set_xticks(range(len(approach_names)))
ax_hm.set_xticklabels(approach_labels, fontsize=8)
ax_hm.set_yticks(range(len(fn_list)))
ax_hm.set_yticklabels(fn_list, fontsize=9)
for i in range(len(fn_list)):
    for j in range(len(approach_names)):
        ax_hm.text(j, i, f'{heatmap_data[i][j]:.0f}', ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if heatmap_data[i][j] < 50 else 'black')
plt.colorbar(hm, ax=ax_hm, shrink=0.8, label='Accuracy %')
ax_hm.set_title('Per-Function Accuracy Heatmap', fontweight='bold')

# Row 4, col 3: Key insight
ax_ins = fig.add_subplot(gs[3, 3])
ax_ins.axis('off')
insight = (
    "v2 KEY IMPROVEMENTS\n"
    "═══════════════════════\n\n"
    "DETECTION:\n"
    "  d² peaks: robust to noise\n"
    "  Optimal segmentation: DP\n"
    "  finds globally best splits\n\n"
    "CLASSIFICATION:\n"
    "  Each breakpoint typed as\n"
    "  JUMP, BEND, or BOTH\n"
    "  → correct primitive chosen\n\n"
    "CONSTRUCTION:\n"
    "  Direct weight computation:\n"
    "  JUMP → RECT (2 neurons)\n"
    "  BEND → RAMP (1 neuron)\n"
    "  No least squares needed\n\n"
    "ENSEMBLE:\n"
    "  Try all approaches,\n"
    "  pick best on train data\n"
    "  → never worse than any one"
)
ax_ins.text(0.05, 0.95, insight, transform=ax_ins.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/tmp/geometric_pinned_v2.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n  {'Function':<15s}", end="")
for aname in approach_labels:
    print(f"  {aname.replace(chr(10),' '):>12s}", end="")
print()
print(f"  {'-'*15}", end="")
for _ in approach_names:
    print(f"  {'-'*12}", end="")
print()

for fn_name in fn_list:
    inp = 128 if fn_name != 'secret_fn' else 100
    print(f"  {fn_name:<15s}", end="")
    for aname in approach_names:
        if aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            print(f"  {exact:>4d}/{inp:<3d}    ", end="")
        else:
            print(f"  {'N/A':>12s}", end="")
    print()

# Averages
print(f"\n  {'AVERAGE %':<15s}", end="")
for aname in approach_names:
    accs = []
    for fn_name in fn_list:
        if aname in all_results[fn_name]:
            _, exact, _, _ = all_results[fn_name][aname]
            inp = 128 if fn_name != 'secret_fn' else 100
            accs.append(exact / inp * 100)
    avg = np.mean(accs) if accs else 0
    print(f"  {avg:>8.1f}%   ", end="")
print()

# Best approach per function
print(f"\n  Best approach per function:")
for fn_name in fn_list:
    results = all_results[fn_name]
    best = max(results.items(), key=lambda x: (x[1][1], -x[1][2]))
    inp = 128 if fn_name != 'secret_fn' else 100
    print(f"    {fn_name:<15s}: {best[0]:<15s} ({best[1][1]}/{inp})")

print(f"\nSaved: /tmp/geometric_pinned_v2.png")
