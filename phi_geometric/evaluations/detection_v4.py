#!/usr/bin/env python3
"""
Detection v4: The Path to 100%

Key findings from the deep dive:
  1. The gate IS capable of 100% at s=φ² with correct breakpoints
  2. Ramps need INTEGER breakpoints, steps need HALF-INTEGER breakpoints
  3. Adjacent same-sign transitions should be MERGED (amplitude-weighted centroid)
  4. Every detection error is Δ=1.0 — all errors are localization, not precision

This version implements:
  A. STEP-FIRST detection (from v3)
  B. TRANSITION MERGING for ramps (amplitude-weighted centroid reconstruction)
  C. CORRECT breakpoint alignment (integers for ramps, half-integers for steps)
  D. STRUCTURAL constraints (consistent amplitudes, RECT pairing)
"""

import math
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2
S8P = np.sqrt(8.0 / np.pi)
CGE = (4 - np.pi) / (6 * np.pi)

def ideal_gate(x):
    x = np.asarray(x, dtype=np.float64)
    f = S8P * x * (1.0 + CGE * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))

def gate_step(x, t, s):
    return (ideal_gate(s * (x - (t - 0.5))) - ideal_gate(s * (x - (t + 0.5)))) / s

def gate_ramp(x, t, s):
    return ideal_gate(s * (x - t)) / s

# Test functions
def fn_tolower(x):
    x = np.asarray(x, dtype=np.float64); r = x.copy()
    r[(x >= 65) & (x <= 90)] += 32; return r
def fn_secret(x):
    return np.where(np.asarray(x, dtype=np.float64) < 50, 2*x, 100 - np.asarray(x, dtype=np.float64))
def fn_rot13(x):
    x = np.asarray(x, dtype=np.float64); r = x.copy()
    r[(x >= 65) & (x <= 77)] += 13; r[(x >= 78) & (x <= 90)] -= 13
    r[(x >= 97) & (x <= 109)] += 13; r[(x >= 110) & (x <= 122)] -= 13; return r
def fn_abs(x): return np.abs(np.asarray(x, dtype=np.float64) - 64)
def fn_saw(x): return np.asarray(x, dtype=np.float64) % 32
def fn_clamp(x): return np.clip(np.asarray(x, dtype=np.float64), 30, 100)
def fn_relu(x): return np.maximum(0, np.asarray(x, dtype=np.float64) - 40)
def fn_stair(x): return np.floor(np.asarray(x, dtype=np.float64) / 16) * 16


# ============================================================================
# CORE DETECTION ENGINE v4
# ============================================================================

def detect_segments(xs, rs, slope_tol=0.15):
    """Detect contiguous segments of approximately constant slope."""
    if len(xs) < 2:
        return [(0, len(xs)-1, 0, rs[0] if len(rs) > 0 else 0)]
    dx = np.diff(xs); dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)
    segments = []; seg_start = 0; seg_slopes = [slopes[0]]
    for i in range(1, len(slopes)):
        cur = np.median(seg_slopes)
        tol = max(slope_tol, abs(cur) * 0.15)
        if abs(slopes[i] - cur) > tol:
            sxs, srs = xs[seg_start:i+1], rs[seg_start:i+1]
            sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else cur
            ic = np.median(srs - sl * sxs)
            segments.append((seg_start, i, sl, ic))
            seg_start = i; seg_slopes = [slopes[i]]
        else:
            seg_slopes.append(slopes[i])
    sxs, srs = xs[seg_start:], rs[seg_start:]
    sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else np.median(seg_slopes)
    ic = np.median(srs - sl * sxs)
    segments.append((seg_start, len(xs)-1, sl, ic))
    return segments


def detect_steps_v4(xs, rs, min_step=3.0):
    """Detect steps: jumps that NEITHER the before NOR after slope can explain.
    
    Key insight: a ramp produces a jump consistent with the slope on one side.
    A step produces a jump that's inconsistent with BOTH sides.
    Use min(|dr - sl_before*dx|, |dr - sl_after*dx|) as the corrected jump.
    """
    if len(xs) < 3: return []
    dx = np.diff(xs); dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)
    steps = []
    for i in range(len(slopes)):
        before_slopes = slopes[max(0, i-3):i]
        after_slopes = slopes[i+1:min(len(slopes), i+4)]
        sl_before = np.median(before_slopes) if len(before_slopes) > 0 else 0
        sl_after = np.median(after_slopes) if len(after_slopes) > 0 else 0
        
        # What jump would each slope predict?
        corrected_before = abs(dr[i] - sl_before * dx[i])
        corrected_after = abs(dr[i] - sl_after * dx[i])
        
        # A step is a jump that NEITHER side can explain
        min_corrected = min(corrected_before, corrected_after)
        
        if min_corrected < min_step:
            continue
        
        # Also verify this slope is anomalous vs context
        abs_slope_here = abs(slopes[i])
        abs_slope_context = max(abs(sl_before), abs(sl_after), 0.01)
        
        if abs_slope_here > 2 * abs_slope_context or min_corrected > min_step * 2:
            # Slope-corrected step height: use the average context slope
            context_slope = (sl_before + sl_after) / 2
            corrected_height = dr[i] - context_slope * dx[i]
            # Half-integer snap for steps
            midpoint = (xs[i] + xs[i+1]) / 2
            bp = round(midpoint - 0.5) + 0.5
            bp = max(xs[i] + 0.1, min(xs[i+1] - 0.1, bp))
            steps.append((i, bp, corrected_height))
    return steps


def merge_transitions(transitions, merge_gap=10.0):
    """Merge adjacent same-sign transitions using amplitude-weighted centroid.
    
    When a single true breakpoint falls between training points that are far
    apart, the hinge decomposition creates TWO transitions (one on each side
    of the gap) with the same sign of slope change. Merging them recovers
    the true position via amplitude-weighted centroid.
    """
    if len(transitions) < 2:
        return transitions
    
    merged = []
    i = 0
    while i < len(transitions):
        bp_i, dm_i = transitions[i]
        
        # Look ahead for same-sign transitions to merge
        j = i + 1
        merge_group = [(bp_i, dm_i)]
        
        while j < len(transitions):
            bp_j, dm_j = transitions[j]
            
            # Same sign and close enough?
            if (dm_i * dm_j > 0) and (bp_j - bp_i < merge_gap):
                merge_group.append((bp_j, dm_j))
                j += 1
            else:
                break
        
        if len(merge_group) == 1:
            merged.append(merge_group[0])
        else:
            # Amplitude-weighted centroid
            total_dm = sum(abs(dm) for _, dm in merge_group)
            if total_dm > 0:
                merged_bp = sum(bp * abs(dm) for bp, dm in merge_group) / total_dm
                merged_dm = sum(dm for _, dm in merge_group)
                merged.append((merged_bp, merged_dm))
        
        i = j
    
    return merged


def structural_inference(step_prims, train_x, train_y, base_slope, 
                         base_intercept, ramp_prims, s=None):
    """Exploit structural patterns to improve step localization.
    
    1. PERIOD REGULARIZATION: If steps have approximately equal spacing
       and amplitude, snap to a regular grid.
    2. RECT PAIRING: If steps come in +h/-h pairs (or +h/-2h/+h triplets),
       enforce consistent widths.
    """
    s = s or PHI ** 2
    if len(step_prims) < 2:
        return step_prims
    
    positions = np.array([bp for _, bp, _ in step_prims])
    amplitudes = np.array([h for _, _, h in step_prims])
    
    # --- PERIOD REGULARIZATION ---
    # Check if all amplitudes are approximately equal (same sign and magnitude)
    abs_amps = np.abs(amplitudes)
    if len(step_prims) >= 3 and np.std(abs_amps) / (np.mean(abs_amps) + 1e-10) < 0.15:
        # Nearly uniform amplitudes → likely periodic
        # Try candidate periods and find best grid fit directly
        spacings = np.diff(positions)
        median_sp = np.median(spacings)
        
        # Candidate periods: median ± 2, plus any common spacing
        candidates = set()
        for sp in spacings:
            for delta in range(-2, 3):
                p = round(sp) + delta
                if p >= 2:
                    candidates.add(p)
                # Also try sp/n for small n (catch sub-harmonics)
                for div in [2, 3]:
                    p2 = round(sp / div)
                    for d2 in range(-1, 2):
                        if p2 + d2 >= 2:
                            candidates.add(p2 + d2)
        
        best_grid_overall = None
        best_score = (-1, 0, float('inf'))
        best_period_overall = None
        
        # Filter out sub-harmonics: period must be at least half the median spacing
        min_period = max(2, np.median(spacings) * 0.5)
        
        for period in candidates:
            if period < min_period:
                continue
            for offset_candidate in np.arange(0.5, period, 1.0):
                grid_positions = []
                for bp in positions:
                    n = round((bp - offset_candidate) / period)
                    grid_positions.append(offset_candidate + n * period)
                deviations = [abs(gp - dp) for gp, dp in zip(grid_positions, positions)]
                max_dev = max(deviations)
                total_dev = sum(deviations)
                n_close = sum(1 for d in deviations if d <= 1.0)
                # Allow max deviation up to period/4 (generous for outliers)
                if max_dev <= min(period / 4, 4):
                    # Primary: maximize positions with Δ≤1
                    # Secondary: prefer LARGEST period (avoid sub-harmonics)
                    # Tertiary: minimize total deviation
                    score = (n_close, period, -total_dev)
                    if score > best_score:
                        best_score = score
                        best_grid_overall = grid_positions
                        best_period_overall = period
        
        if best_grid_overall is not None and best_score[0] >= 0:
            # Verify: grid must not reduce exact match count on training data
            candidate_steps = [('step', gp, h) for gp, (_, _, h) in 
                               zip(best_grid_overall, step_prims)]
            
            def eval_exact(steps):
                r = base_intercept + base_slope * train_x
                for _, bp, val in steps:
                    r += val * gate_step(train_x, bp, s)
                for _, bp, val in ramp_prims:
                    r += val * gate_ramp(train_x, bp, s)
                return int(np.sum(np.round(train_x + r) == np.round(train_y)))
            
            exact_orig = eval_exact(step_prims)
            exact_grid = eval_exact(candidate_steps)
            
            if exact_grid >= exact_orig:  # grid is at least as good
                step_prims = candidate_steps
                return step_prims
    
    # --- NEIGHBOR CONSENSUS ---
    # For each step, check what its left and right neighbors suggest.
    # Only correct if BOTH neighbors agree on the same correction.
    if len(step_prims) >= 3:
        spacings = np.diff(positions)
        # Find modal spacing (most common, rounded)
        from collections import Counter
        rounded_spacings = [round(sp) for sp in spacings]
        spacing_counts = Counter(rounded_spacings)
        # Also count multiples as evidence for the factor
        factor_votes = {}
        for sp_r, cnt in spacing_counts.items():
            if sp_r >= 2:
                factor_votes[sp_r] = factor_votes.get(sp_r, 0) + cnt
                # If a spacing is 2x some value, vote for that value too
                if sp_r % 2 == 0:
                    half = sp_r // 2
                    if half >= 2:
                        factor_votes[half] = factor_votes.get(half, 0) + cnt * 0.3
        
        if factor_votes:
            target = max(factor_votes, key=factor_votes.get)
            if target >= 2 and factor_votes[target] >= 2:
                regularized = list(step_prims)
                for i in range(len(step_prims)):
                    suggestions = []
                    # Left neighbor suggests
                    if i > 0:
                        n_left = round((positions[i] - positions[i-1]) / target)
                        if n_left >= 1:
                            suggestions.append(positions[i-1] + n_left * target)
                    # Right neighbor suggests  
                    if i < len(step_prims) - 1:
                        n_right = round((positions[i+1] - positions[i]) / target)
                        if n_right >= 1:
                            suggestions.append(positions[i+1] - n_right * target)
                    
                    if len(suggestions) == 2:
                        # Both neighbors must agree (within 0.5)
                        if abs(suggestions[0] - suggestions[1]) < 0.5:
                            new_bp = (suggestions[0] + suggestions[1]) / 2
                            # Only apply small corrections
                            if abs(new_bp - positions[i]) <= 1.5 and abs(new_bp - positions[i]) > 0.1:
                                regularized[i] = (step_prims[i][0], new_bp, step_prims[i][2])
                
                # Validate with exact match count
                def eval_exact(steps):
                    r = base_intercept + base_slope * train_x
                    for _, bp, val in steps:
                        r += val * gate_step(train_x, bp, s)
                    for _, bp, val in ramp_prims:
                        r += val * gate_ramp(train_x, bp, s)
                    return int(np.sum(np.round(train_x + r) == np.round(train_y)))
                
                exact_orig = eval_exact(step_prims)
                exact_reg = eval_exact(regularized)
                
                if exact_reg >= exact_orig:
                    step_prims = regularized
    
    # --- RECT PAIR WIDTH CORRECTION ---
    # When exactly 2 steps form a RECT pair (h ≈ -h), the width of the
    # rectangle can be estimated from the fraction of training points
    # that fall inside it. If the detected width differs significantly
    # from the estimate, correct both positions symmetrically.
    if len(step_prims) == 2:
        _, bp1, h1 = step_prims[0]
        _, bp2, h2 = step_prims[1]
        if bp1 > bp2:
            bp1, bp2, h1, h2 = bp2, bp1, h2, h1
        
        # Check for RECT pair: opposite amplitudes
        if abs(h1 + h2) < 0.5 * max(abs(h1), abs(h2)):
            rs = train_y - train_x
            # Count training points with residual matching the "inside" value
            inside_r = h1 if h1 > 0 else h2  # positive step opens the rect
            n_inside = int(np.sum(np.abs(rs - inside_r) < abs(inside_r) * 0.1 + 1))
            R = int(max(train_x)) + 1  # domain [0, max]
            n_total = len(train_x)
            
            # Width estimate from count (need ≥3 inside points for reliability)
            if n_inside >= 3:
                est_width = round(n_inside * R / n_total)
            else:
                est_width = round(bp2 - bp1)  # keep detected width
            detected_width = round(bp2 - bp1)
            
            if n_inside >= 3 and abs(est_width - detected_width) >= 3:
                center = (bp1 + bp2) / 2
                half_w = est_width / 2
                new_bp1 = round(center - half_w - 0.5) + 0.5
                new_bp2 = round(center + half_w - 0.5) + 0.5
                
                # Validate: training accuracy must not decrease
                def eval_exact_rect(steps):
                    r = base_intercept + base_slope * train_x
                    for _, bp, val in steps:
                        r += val * gate_step(train_x, bp, s)
                    for _, bp, val in ramp_prims:
                        r += val * gate_ramp(train_x, bp, s)
                    return int(np.sum(np.round(train_x + r) == np.round(train_y)))
                
                rect_candidate = [('step', new_bp1, h1 if bp1 == step_prims[0][1] else h2),
                                  ('step', new_bp2, h2 if bp2 == step_prims[1][1] else h1)]
                # Preserve original order
                if step_prims[0][1] > step_prims[1][1]:
                    rect_candidate = [rect_candidate[1], rect_candidate[0]]
                
                exact_orig = eval_exact_rect(step_prims)
                exact_rect = eval_exact_rect(rect_candidate)
                
                if exact_rect >= exact_orig:
                    step_prims = rect_candidate
    
    return step_prims


def refine_step_positions(train_x, train_y, base_slope, base_intercept, 
                          step_prims, ramp_prims, s=None):
    """Coordinate descent: try shifting each step by ±1, keep if exact matches improve.
    
    Uses exact match count instead of SSE, because SSE has a systematic bias:
    gate tail leakage pushes breakpoints AWAY from nearby training points,
    even when the closer position is correct.
    """
    s = s or PHI ** 2
    
    def exact_matches(steps):
        r = base_intercept + base_slope * train_x
        for _, bp, val in steps:
            r += val * gate_step(train_x, bp, s)
        for _, bp, val in ramp_prims:
            r += val * gate_ramp(train_x, bp, s)
        return int(np.sum(np.round(train_x + r) == np.round(train_y)))
    
    improved = True
    while improved:
        improved = False
        for si in range(len(step_prims)):
            current_bp = step_prims[si][1]
            current_exact = exact_matches(step_prims)
            
            candidate_lo = list(step_prims)
            candidate_lo[si] = (step_prims[si][0], current_bp - 1.0, step_prims[si][2])
            exact_lo = exact_matches(candidate_lo)
            
            candidate_hi = list(step_prims)
            candidate_hi[si] = (step_prims[si][0], current_bp + 1.0, step_prims[si][2])
            exact_hi = exact_matches(candidate_hi)
            
            best_exact = max(current_exact, exact_lo, exact_hi)
            if best_exact > current_exact:
                improved = True
                if exact_lo > exact_hi:
                    step_prims[si] = candidate_lo[si]
                else:
                    step_prims[si] = candidate_hi[si]
    
    return step_prims


def gap_refine_steps(train_x, train_y, base_slope, base_intercept,
                     step_prims, ramp_prims, s=None):
    """Phase 9: Refine step positions using the known gate curve shape.
    
    For steps in gaps (no training data nearby), use two strategies:
    1. VIRTUAL EVALUATION: evaluate at every integer in the gap using the gate
       model. Check each candidate bp against extrapolated slopes from training
       data on either side. Eliminate candidates that are inconsistent.
    2. CO-LOCATED RAMP: when a step is within 2 units of a ramp with the same
       sign of change, the step should precede the ramp (the discontinuity
       happens before the slope change).
    """
    s = s or PHI ** 2
    sorted_x = np.sort(train_x)
    
    for si in range(len(step_prims)):
        _, bp, h = step_prims[si]
        
        # Find gap boundaries
        before = sorted_x[sorted_x < bp]
        after = sorted_x[sorted_x > bp]
        if len(before) == 0 or len(after) == 0:
            continue
        x_before = before[-1]
        x_after = after[0]
        gap = x_after - x_before
        
        if gap <= 2:  # Step well-localized, no refinement needed
            continue
        
        # Get slopes on each side by fitting nearby training points
        bm = (sorted_x >= x_before - 10) & (sorted_x <= x_before)
        am = (sorted_x >= x_after) & (sorted_x <= x_after + 10)
        xs_b = train_x[np.isin(train_x, sorted_x[bm])]
        ys_b = train_y[np.isin(train_x, sorted_x[bm])]
        xs_a = train_x[np.isin(train_x, sorted_x[am])]
        ys_a = train_y[np.isin(train_x, sorted_x[am])]
        
        if len(xs_b) >= 2:
            slope_b = round(np.polyfit(xs_b, ys_b, 1)[0])
            inter_b = round(np.median(ys_b - slope_b * xs_b))
        else:
            slope_b = round(base_slope + 1)
            inter_b = round(ys_b[0] - slope_b * xs_b[0]) if len(ys_b) > 0 else 0
        
        if len(xs_a) >= 2:
            slope_a = round(np.polyfit(xs_a, ys_a, 1)[0])
            inter_a = round(np.median(ys_a - slope_a * xs_a))
        else:
            slope_a = round(base_slope + 1)
            inter_a = round(ys_a[0] - slope_a * xs_a[0]) if len(ys_a) > 0 else 0
        
        # Only apply virtual evaluation when slopes differ (otherwise all
        # candidates are equally self-consistent)
        slopes_differ = abs(slope_b - slope_a) > 0.5
        
        gap_integers = np.arange(int(x_before), int(x_after) + 1)
        best_score = -1
        best_candidates = []
        
        for candidate in np.arange(x_before + 0.5, x_after, 1.0):
            if slopes_differ:
                # Virtual evaluation: check each gap integer
                test_steps = list(step_prims)
                test_steps[si] = ('step', candidate, h)
                
                correct = 0
                for gx in gap_integers:
                    gxarr = np.array([float(gx)])
                    pred = float(gx) + base_intercept + base_slope * float(gx)
                    for _, sbp, sh in test_steps:
                        pred += sh * gate_step(gxarr, sbp, s)[0]
                    for _, rbp, rv in ramp_prims:
                        pred += rv * gate_ramp(gxarr, rbp, s)[0]
                    rounded = round(pred)
                    
                    # Expected from extrapolation
                    if gx < candidate:
                        expected = round(slope_b * gx + inter_b)
                    else:
                        expected = round(slope_a * gx + inter_a)
                    
                    if rounded == expected:
                        correct += 1
                
                if correct > best_score:
                    best_score = correct
                    best_candidates = [candidate]
                elif correct == best_score:
                    best_candidates.append(candidate)
            else:
                best_candidates.append(candidate)
        
        # Only apply refinement when we have a co-located ramp constraint
        # (provides independent structural signal beyond the circular
        # self-consistency of virtual evaluation alone)
        new_bp = bp  # default: keep current
        
        for _, rbp, rdm in ramp_prims:
            if abs(bp - rbp) < 3.0:
                # Step and ramp co-located — step should precede the ramp
                valid = [c for c in best_candidates if c <= rbp]
                if valid:
                    new_bp = max(valid)
                    break
        
        # Always validate: only apply if training accuracy doesn't decrease
        if new_bp != bp:
            def eval_exact(steps):
                r = base_intercept + base_slope * train_x
                for _, bp_v, val in steps:
                    r += val * gate_step(train_x, bp_v, s)
                for _, bp_v, val in ramp_prims:
                    r += val * gate_ramp(train_x, bp_v, s)
                return int(np.sum(np.round(train_x + r) == np.round(train_y)))
            
            exact_before = eval_exact(step_prims)
            candidate_prims = list(step_prims)
            candidate_prims[si] = ('step', new_bp, h)
            exact_after = eval_exact(candidate_prims)
            
            if exact_after >= exact_before:
                step_prims[si] = ('step', new_bp, h)
    
    return step_prims


def detect_all_v4(train_x, train_y, s=None, slope_tol=0.15):
    """Full v4 detection pipeline: steps first, then ramps with merging.
    
    Returns: (base_slope, base_intercept, step_primitives, ramp_primitives)
    """
    s = s or PHI ** 2
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs
    
    # Phase 1: Detect steps
    raw_steps = detect_steps_v4(xs, rs)
    step_prims = [('step', bp, h) for _, bp, h in raw_steps]
    
    # Phase 2: Remove steps from residual
    rs_corr = rs.copy()
    for idx, bp, height in raw_steps:
        rs_corr[xs > bp] -= height
    
    # Phase 3: Detect segments in step-corrected residual
    segs = detect_segments(xs, rs_corr, slope_tol)
    
    if len(segs) == 0:
        return 0, 0, step_prims, []
    
    base_slope = segs[0][2]
    base_intercept = segs[0][3]
    
    # Phase 4: Classify transitions and collect for merging
    raw_transitions = []
    for i in range(1, len(segs)):
        pe = segs[i-1][1]; cs = segs[i][0]
        bp = (xs[pe] + xs[cs]) / 2
        ds = segs[i][2] - segs[i-1][2]
        
        # Also check for jumps between segments (residual steps missed in phase 1)
        rp = segs[i-1][2] * bp + segs[i-1][3]
        rc = segs[i][2] * bp + segs[i][3]
        jump = rc - rp
        
        if abs(jump) > 2.0:
            step_bp = round(bp - 0.5) + 0.5
            step_bp = max(xs[pe] + 0.1, min(xs[cs] - 0.1, step_bp))
            step_prims.append(('step', step_bp, jump))
        
        if abs(ds) > 0.03:
            raw_transitions.append((bp, ds))
    
    # Phase 5: Merge adjacent same-sign transitions
    merged = merge_transitions(raw_transitions)
    
    # Phase 6: Snap ramp breakpoints to integers
    ramp_prims = []
    for bp, dm in merged:
        bp_int = round(bp)
        ramp_prims.append(('ramp', float(bp_int), dm))
    
    # Phase 7: Structural inference — exploit higher-level patterns
    if len(step_prims) >= 2:
        step_prims = structural_inference(step_prims, train_x, train_y,
                                          base_slope, base_intercept,
                                          ramp_prims, s)
    
    # Phase 8: Refine step positions via coordinate descent on training error
    if len(step_prims) > 0:
        step_prims = refine_step_positions(
            train_x, train_y, base_slope, base_intercept, 
            step_prims, ramp_prims, s)
    
    # Phase 9: Gap refinement — use known gate curve to refine steps in gaps
    if len(step_prims) > 0:
        step_prims = gap_refine_steps(
            train_x, train_y, base_slope, base_intercept,
            step_prims, ramp_prims, s)
    
    return base_slope, base_intercept, step_prims, ramp_prims


def evaluate_v4(test_x, base_slope, base_intercept, step_prims, ramp_prims, s=None):
    """Evaluate the detected decomposition."""
    s = s or PHI ** 2
    r = base_intercept + base_slope * test_x
    for ptype, bp, val in step_prims:
        r += val * gate_step(test_x, bp, s)
    for ptype, bp, val in ramp_prims:
        r += val * gate_ramp(test_x, bp, s)
    return test_x + r


# ============================================================================
# ORACLE with correct breakpoints
# ============================================================================

CORRECT_ORACLE = {
    'tolower':      (0, 0, [('step', 64.5, 32), ('step', 90.5, -32)], []),
    'secret_fn':    (1, 0, [('step', 49.5, -50.0)], [('ramp', 50, -3.0)]),
    'ROT13':        (0, 0, [('step', 64.5, 13), ('step', 77.5, -26),
                             ('step', 90.5, 13), ('step', 96.5, 13),
                             ('step', 109.5, -26), ('step', 122.5, 13)], []),
    'abs_centered': (-2, 64, [], [('ramp', 64, 2.0)]),
    'sawtooth_32':  (0, 0, [('step', 31.5, -32), ('step', 63.5, -32),
                             ('step', 95.5, -32)], []),
    'clamp':        (-1, 30, [], [('ramp', 30, 1.0), ('ramp', 100, -1.0)]),
    'relu_shifted': (-1, 0, [], [('ramp', 40, 1.0)]),
    'staircase':    (-1, 0, [('step', 15.5, 16), ('step', 31.5, 16),
                              ('step', 47.5, 16), ('step', 63.5, 16),
                              ('step', 79.5, 16), ('step', 95.5, 16),
                              ('step', 111.5, 16)], []),
}


# ============================================================================
# BENCHMARK
# ============================================================================

S = PHI ** 2

fns = [
    ("tolower",      fn_tolower,  128, 50),
    ("secret_fn",    fn_secret,   100, 40),
    ("ROT13",        fn_rot13,    128, 60),
    ("abs_centered", fn_abs,      128, 40),
    ("sawtooth_32",  fn_saw,      128, 50),
    ("clamp",        fn_clamp,    128, 40),
    ("relu_shifted", fn_relu,     128, 30),
    ("staircase",    fn_stair,    128, 50),
]

print("=" * 70)
print("DETECTION v4: The Path to 100%")
print("=" * 70)

# First: verify corrected oracle gives 100%
print("\n--- CORRECTED ORACLE VERIFICATION ---")
for name, fn, R, _ in fns:
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    bs, bi, steps, ramps = CORRECT_ORACLE[name]
    pred = evaluate_v4(tx, bs, bi, steps, ramps, S)
    exact = int(np.sum(np.round(pred) == ty))
    print(f"  {name:<15s}: {exact:>4d}/{R} {'✓ PERFECT' if exact == R else '✗ FAIL'}")
    if exact != R:
        errors = np.where(np.round(pred) != ty)[0]
        for e in errors[:5]:
            print(f"    x={e}: true={ty[e]:.0f}, pred={pred[e]:.4f}")

# Main benchmark
print("\n--- v4 DETECTION BENCHMARK ---")
all_results = {}

for name, fn, R, ntr in fns:
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, ntr, replace=False)).astype(np.float64)
    try_y = fn(trx)
    
    print(f"\n{'='*55}")
    print(f"  {name} ({ntr} train / {R} test)")
    print(f"{'='*55}")
    
    # Oracle
    bs_o, bi_o, steps_o, ramps_o = CORRECT_ORACLE[name]
    pred_o = evaluate_v4(tx, bs_o, bi_o, steps_o, ramps_o, S)
    exact_o = int(np.sum(np.round(pred_o) == ty))
    
    # v4 detection
    t0 = time.perf_counter()
    bs_d, bi_d, steps_d, ramps_d = detect_all_v4(trx, try_y, S)
    pred_d = evaluate_v4(tx, bs_d, bi_d, steps_d, ramps_d, S)
    t_d = time.perf_counter() - t0
    exact_d = int(np.sum(np.round(pred_d) == ty))
    
    gap = exact_o - exact_d
    
    print(f"  oracle    : {exact_o:>4d}/{R}")
    print(f"  v4_detect : {exact_d:>4d}/{R}  ({t_d:.4f}s)  gap={gap}")
    
    # Show detected decomposition
    print(f"  detected: base_slope={bs_d:.3f}, base_intercept={bi_d:.2f}")
    for pt, bp, v in steps_d:
        print(f"    step at {bp:>7.1f}, h={v:>7.2f}")
    for pt, bp, v in ramps_d:
        print(f"    ramp at {bp:>7.1f}, Δm={v:>7.3f}")
    
    # Error analysis
    errors = np.where(np.round(pred_d) != ty)[0]
    if len(errors) > 0 and len(errors) <= 15:
        print(f"  errors ({len(errors)}):")
        for e in errors[:8]:
            print(f"    x={e}: true={ty[e]:.0f}, pred={pred_d[e]:.2f}")
    
    all_results[name] = {'oracle': exact_o, 'v4': exact_d, 'gap': gap,
                          'time': t_d, 'R': R}


# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\n  {'Function':<15s}  {'Oracle':>7s}  {'v4':>7s}  {'Gap':>5s}")
print(f"  {'-'*40}")

avgs = {'oracle': [], 'v4': []}
for name, _, R, _ in fns:
    r = all_results[name]
    g = r['gap']
    sign = '+' if g > 0 else ''
    print(f"  {name:<15s}  {r['oracle']:>4d}/{R}  {r['v4']:>4d}/{R}  {sign}{g}")
    avgs['oracle'].append(r['oracle'] / R * 100)
    avgs['v4'].append(r['v4'] / R * 100)

print(f"\n  {'AVERAGE':<15s}  {np.mean(avgs['oracle']):>6.1f}%  {np.mean(avgs['v4']):>6.1f}%")


# Sample efficiency
print(f"\n{'='*70}")
print("SAMPLE EFFICIENCY")
print(f"{'='*70}")
for name, fn, R, _ in [fns[0], fns[2], fns[5], fns[7]]:  # tolower, ROT13, clamp, staircase
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    print(f"\n  {name}:")
    for ns in [10, 20, 30, 50, 80, 100, 128]:
        if ns > R: continue
        np.random.seed(42)
        trx = np.sort(np.random.choice(R, min(ns, R), replace=False)).astype(np.float64)
        try_y = fn(trx)
        bs, bi, steps, ramps = detect_all_v4(trx, try_y, S)
        pred = evaluate_v4(tx, bs, bi, steps, ramps, S)
        exact = int(np.sum(np.round(pred) == ty))
        print(f"    {ns:>3d} ex: {exact:>3d}/{R}")


# Visualization
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Detection v4: Oracle vs v4 vs True', fontsize=14, fontweight='bold')

for idx, (name, fn, R, ntr) in enumerate(fns):
    ax = axes[idx // 4][idx % 4]
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, ntr, replace=False)).astype(np.float64)
    try_y = fn(trx)
    
    # Oracle
    bs_o, bi_o, steps_o, ramps_o = CORRECT_ORACLE[name]
    pred_o = evaluate_v4(tx, bs_o, bi_o, steps_o, ramps_o, S)
    
    # v4
    bs_d, bi_d, steps_d, ramps_d = detect_all_v4(trx, try_y, S)
    pred_d = evaluate_v4(tx, bs_d, bi_d, steps_d, ramps_d, S)
    
    ax.plot(tx, ty, 'k-', lw=2, alpha=0.3, label='True')
    eo = int(np.sum(np.round(pred_o) == ty))
    ed = int(np.sum(np.round(pred_d) == ty))
    ax.plot(tx, pred_o, 'g-', lw=1, alpha=0.6, label=f'oracle: {eo}/{R}')
    ax.plot(tx, pred_d, 'r-', lw=1.2, alpha=0.7, label=f'v4: {ed}/{R}')
    
    # Mark errors
    errs = np.where(np.round(pred_d) != ty)[0]
    if len(errs) > 0 and len(errs) <= 20:
        ax.scatter(errs, ty[errs], c='red', s=30, zorder=5, marker='x')
    
    ax.scatter(trx, try_y, c='blue', s=5, alpha=0.3, zorder=4)
    ax.set_title(f'{name} (v4={ed}/{R})', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/tmp/detection_v4.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: /tmp/detection_v4.png")
