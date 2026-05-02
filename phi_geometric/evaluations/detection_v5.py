#!/usr/bin/env python3
"""
Detection v5: Simplified 3-Phase Pipeline

Inspired by the 12D clock decomposition: structure-first, not correction-first.
The clock system works because it knows the structural parameter (ratio) from 
the start and uses it at every level. Our v4 pipeline detected locally and 
corrected globally across 10 phases. v5 inverts this: classify globally first,
then place breakpoints using structural constraints.

Three phases:
  1. CLASSIFY — residual analysis, transition detection, structure classification
  2. PLACE   — structural placement of breakpoints using global constraints
  3. REFINE  — coordinate descent with exact-match validation

Target: match or beat v4's 99.9% with half the code complexity.
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
# PHASE 1: CLASSIFY — Analyze residual, detect all transitions, classify structure
# ============================================================================

def phase1_classify(xs, ys, slope_tol=0.15):
    """Analyze training data and classify the function's structure.
    
    Returns a structured description:
      - base_slope, base_intercept (the dominant linear trend)
      - raw_steps: [(bp, height), ...] — detected discontinuities
      - raw_ramps: [(bp, delta_slope), ...] — detected slope changes
      - structure: one of 'periodic', 'rect', 'single_step', 'ramp_only', 'mixed'
      - struct_params: dict of structural parameters (period, width, etc.)
    """
    rs = ys - xs
    dx = np.diff(xs)
    dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)
    
    # --- Detect steps (jumps neither side can explain) ---
    raw_steps = []
    for i in range(len(slopes)):
        before = slopes[max(0, i-3):i]
        after = slopes[i+1:min(len(slopes), i+4)]
        sl_b = np.median(before) if len(before) > 0 else 0
        sl_a = np.median(after) if len(after) > 0 else 0
        
        corr_b = abs(dr[i] - sl_b * dx[i])
        corr_a = abs(dr[i] - sl_a * dx[i])
        min_corr = min(corr_b, corr_a)
        
        if min_corr < 3.0:
            continue
        
        abs_here = abs(slopes[i])
        abs_ctx = max(abs(sl_b), abs(sl_a), 0.01)
        
        if abs_here > 2 * abs_ctx or min_corr > 6.0:
            ctx_slope = (sl_b + sl_a) / 2
            height = dr[i] - ctx_slope * dx[i]
            midpoint = (xs[i] + xs[i+1]) / 2
            bp = round(midpoint - 0.5) + 0.5
            bp = max(xs[i] + 0.1, min(xs[i+1] - 0.1, bp))
            raw_steps.append((bp, height))
    
    # --- Remove steps from residual, then detect segments ---
    rs_corr = rs.copy()
    for bp, h in raw_steps:
        rs_corr[xs > bp] -= h
    
    # Segment detection
    segments = []
    seg_start = 0
    seg_slopes = [slopes[0]] if len(slopes) > 0 else [0]
    
    for i in range(1, len(slopes)):
        cur = np.median(seg_slopes)
        tol = max(slope_tol, abs(cur) * 0.15)
        if abs(slopes[i] - cur) > tol:
            sxs, srs = xs[seg_start:i+1], rs_corr[seg_start:i+1]
            sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else cur
            ic = np.median(srs - sl * sxs)
            segments.append((seg_start, i, sl, ic))
            seg_start = i
            seg_slopes = [slopes[i]]
        else:
            seg_slopes.append(slopes[i])
    
    sxs, srs = xs[seg_start:], rs_corr[seg_start:]
    sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else np.median(seg_slopes)
    ic = np.median(srs - sl * sxs)
    segments.append((seg_start, len(xs)-1, sl, ic))
    
    # Base slope/intercept from first segment
    base_slope = segments[0][2] if segments else 0
    base_intercept = segments[0][3] if segments else 0
    
    # --- Detect ramps and residual steps from segment transitions ---
    raw_ramps_pre_merge = []
    for i in range(1, len(segments)):
        pe = segments[i-1][1]
        cs = segments[i][0]
        bp = (xs[pe] + xs[cs]) / 2
        ds = segments[i][2] - segments[i-1][2]
        
        # Check for residual jumps between segments
        rp = segments[i-1][2] * bp + segments[i-1][3]
        rc = segments[i][2] * bp + segments[i][3]
        jump = rc - rp
        if abs(jump) > 2.0:
            step_bp = round(bp - 0.5) + 0.5
            step_bp = max(xs[pe] + 0.1, min(xs[cs] - 0.1, step_bp))
            raw_steps.append((step_bp, jump))
        
        if abs(ds) > 0.03:
            raw_ramps_pre_merge.append((bp, ds))
    
    # Merge adjacent same-sign ramp transitions (amplitude-weighted centroid)
    raw_ramps = []
    i = 0
    while i < len(raw_ramps_pre_merge):
        bp_i, dm_i = raw_ramps_pre_merge[i]
        group = [(bp_i, dm_i)]
        j = i + 1
        while j < len(raw_ramps_pre_merge):
            bp_j, dm_j = raw_ramps_pre_merge[j]
            if dm_i * dm_j > 0 and bp_j - bp_i < 10.0:
                group.append((bp_j, dm_j))
                j += 1
            else:
                break
        if len(group) == 1:
            raw_ramps.append(group[0])
        else:
            total = sum(abs(d) for _, d in group)
            if total > 0:
                merged_bp = sum(b * abs(d) for b, d in group) / total
                merged_dm = sum(d for _, d in group)
                raw_ramps.append((merged_bp, merged_dm))
        i = j
    
    # Snap ramp breakpoints to integers
    raw_ramps = [(round(bp), dm) for bp, dm in raw_ramps]
    
    # --- Classify structure ---
    step_heights = np.array([h for _, h in raw_steps]) if raw_steps else np.array([])
    abs_heights = np.abs(step_heights)
    
    n_steps = len(raw_steps)
    n_ramps = len(raw_ramps)
    
    structure = 'mixed'
    struct_params = {}
    
    if n_steps == 0:
        structure = 'ramp_only'
    elif n_steps == 1:
        structure = 'single_step'
    elif n_steps == 2:
        h1, h2 = step_heights[0], step_heights[1]
        if abs(h1 + h2) < 0.5 * max(abs(h1), abs(h2)):
            structure = 'rect'
            struct_params['h_open'] = h1 if h1 > 0 else h2
            struct_params['h_close'] = h2 if h1 > 0 else h1
    elif n_steps >= 3:
        if len(abs_heights) >= 3 and np.std(abs_heights) / (np.mean(abs_heights) + 1e-10) < 0.15:
            structure = 'periodic'
            positions = np.array([bp for bp, _ in raw_steps])
            spacings = np.diff(positions)
            struct_params['est_period'] = np.median(spacings)
        else:
            # Check for repeated RECT patterns (like ROT13: +h, -2h, +h)
            structure = 'multi_step'
    
    return {
        'base_slope': base_slope,
        'base_intercept': base_intercept,
        'raw_steps': raw_steps,
        'raw_ramps': raw_ramps,
        'structure': structure,
        'struct_params': struct_params,
        'train_x': xs,
        'train_y': ys,
    }


# ============================================================================
# PHASE 2: PLACE — Use structural constraints to place breakpoints optimally
# ============================================================================

def phase2_place(classification, s=None):
    """Place breakpoints using the classified structure.
    
    Instead of placing at gap midpoints and correcting later, use global
    structural knowledge to place breakpoints optimally from the start.
    """
    s = s or PHI ** 2
    structure = classification['structure']
    base_slope = classification['base_slope']
    base_intercept = classification['base_intercept']
    raw_steps = classification['raw_steps']
    raw_ramps = classification['raw_ramps']
    train_x = classification['train_x']
    train_y = classification['train_y']
    struct_params = classification['struct_params']
    
    step_prims = [('step', bp, h) for bp, h in raw_steps]
    ramp_prims = [('ramp', float(bp), dm) for bp, dm in raw_ramps]
    
    # --- Structure-specific placement ---
    
    if structure == 'periodic':
        step_prims = _place_periodic(step_prims, train_x, train_y,
                                      base_slope, base_intercept, ramp_prims,
                                      struct_params, s)
    
    elif structure == 'rect':
        step_prims = _place_rect(step_prims, train_x, train_y,
                                  base_slope, base_intercept, ramp_prims,
                                  struct_params, s)
    
    elif structure == 'single_step':
        step_prims = _place_single_step(step_prims, ramp_prims, train_x, train_y,
                                         base_slope, base_intercept, s)
    
    # For multi_step (≥3 non-uniform steps), try neighbor consensus
    if structure in ('multi_step', 'mixed') and len(step_prims) >= 3:
        step_prims = _neighbor_consensus(step_prims, train_x, train_y,
                                          base_slope, base_intercept, ramp_prims, s)
    
    return base_slope, base_intercept, step_prims, ramp_prims


def _place_periodic(step_prims, train_x, train_y, base_slope, base_intercept,
                    ramp_prims, struct_params, s):
    """Place steps on a regular grid for periodic functions."""
    positions = np.array([bp for _, bp, _ in step_prims])
    spacings = np.diff(positions)
    
    # Find best period: try candidates around estimated period
    est_period = struct_params.get('est_period', np.median(spacings))
    candidates = set()
    for sp in spacings:
        for delta in range(-2, 3):
            p = round(sp) + delta
            if p >= 2:
                candidates.add(p)
            for div in [2, 3]:
                p2 = round(sp / div)
                for d2 in range(-1, 2):
                    if p2 + d2 >= 2:
                        candidates.add(p2 + d2)
    
    # Sub-harmonic filter
    min_period = max(2, np.median(spacings) * 0.5)
    
    best_grid = None
    best_score = (-1, 0, float('inf'))
    
    for period in candidates:
        if period < min_period:
            continue
        for offset in np.arange(0.5, period, 1.0):
            grid = []
            for bp in positions:
                n = round((bp - offset) / period)
                grid.append(offset + n * period)
            devs = [abs(g - p) for g, p in zip(grid, positions)]
            max_dev = max(devs)
            total_dev = sum(devs)
            n_close = sum(1 for d in devs if d <= 1.0)
            
            if max_dev <= min(period / 4, 4):
                score = (n_close, period, -total_dev)
                if score > best_score:
                    best_score = score
                    best_grid = grid
    
    if best_grid is not None:
        candidate = [('step', gp, h) for gp, (_, _, h) in zip(best_grid, step_prims)]
        if _exact_count(candidate, ramp_prims, train_x, train_y,
                        base_slope, base_intercept, s) >= \
           _exact_count(step_prims, ramp_prims, train_x, train_y,
                        base_slope, base_intercept, s):
            return candidate
    
    # Fallback: neighbor consensus for ≥3 steps
    if len(step_prims) >= 3:
        step_prims = _neighbor_consensus(step_prims, train_x, train_y,
                                          base_slope, base_intercept, ramp_prims, s)
    
    return step_prims


def _neighbor_consensus(step_prims, train_x, train_y, base_slope, base_intercept,
                        ramp_prims, s):
    """Correct step positions using left/right neighbor agreement."""
    from collections import Counter
    positions = np.array([bp for _, bp, _ in step_prims])
    spacings = np.diff(positions)
    rounded_sp = [round(sp) for sp in spacings]
    sp_counts = Counter(rounded_sp)
    
    factor_votes = {}
    for sp_r, cnt in sp_counts.items():
        if sp_r >= 2:
            factor_votes[sp_r] = factor_votes.get(sp_r, 0) + cnt
            if sp_r % 2 == 0:
                half = sp_r // 2
                if half >= 2:
                    factor_votes[half] = factor_votes.get(half, 0) + cnt * 0.3
    
    if not factor_votes:
        return step_prims
    
    target = max(factor_votes, key=factor_votes.get)
    if target < 2 or factor_votes[target] < 2:
        return step_prims
    
    regularized = list(step_prims)
    for i in range(len(step_prims)):
        suggestions = []
        if i > 0:
            n_left = round((positions[i] - positions[i-1]) / target)
            if n_left >= 1:
                suggestions.append(positions[i-1] + n_left * target)
        if i < len(step_prims) - 1:
            n_right = round((positions[i+1] - positions[i]) / target)
            if n_right >= 1:
                suggestions.append(positions[i+1] - n_right * target)
        
        if len(suggestions) == 2 and abs(suggestions[0] - suggestions[1]) < 0.5:
            new_bp = (suggestions[0] + suggestions[1]) / 2
            if 0.1 < abs(new_bp - positions[i]) <= 1.5:
                regularized[i] = (step_prims[i][0], new_bp, step_prims[i][2])
    
    if _exact_count(regularized, ramp_prims, train_x, train_y,
                    base_slope, base_intercept, s) >= \
       _exact_count(step_prims, ramp_prims, train_x, train_y,
                    base_slope, base_intercept, s):
        return regularized
    return step_prims


def _place_rect(step_prims, train_x, train_y, base_slope, base_intercept,
                ramp_prims, struct_params, s):
    """Place RECT pair using width estimated from training data fraction."""
    if len(step_prims) != 2:
        return step_prims
    
    _, bp1, h1 = step_prims[0]
    _, bp2, h2 = step_prims[1]
    if bp1 > bp2:
        bp1, bp2, h1, h2 = bp2, bp1, h2, h1
    
    # Estimate RECT width from fraction of "inside" training points
    rs = train_y - train_x
    inside_r = h1 if h1 > 0 else h2
    n_inside = int(np.sum(np.abs(rs - inside_r) < abs(inside_r) * 0.1 + 1))
    R = int(max(train_x)) + 1
    n_total = len(train_x)
    
    # Need ≥3 inside points for reliable estimate
    if n_inside >= 3:
        est_width = round(n_inside * R / n_total)
        detected_width = round(bp2 - bp1)
        
        if abs(est_width - detected_width) >= 3:
            center = (bp1 + bp2) / 2
            half_w = est_width / 2
            new_bp1 = round(center - half_w - 0.5) + 0.5
            new_bp2 = round(center + half_w - 0.5) + 0.5
            
            # Preserve amplitude assignment
            candidate = [('step', new_bp1, h1 if step_prims[0][1] <= step_prims[1][1] else h2),
                         ('step', new_bp2, h2 if step_prims[0][1] <= step_prims[1][1] else h1)]
            if step_prims[0][1] > step_prims[1][1]:
                candidate = [candidate[1], candidate[0]]
            
            if _exact_count(candidate, ramp_prims, train_x, train_y,
                            base_slope, base_intercept, s) >= \
               _exact_count(step_prims, ramp_prims, train_x, train_y,
                            base_slope, base_intercept, s):
                return candidate
    
    return step_prims


def _place_single_step(step_prims, ramp_prims, train_x, train_y,
                       base_slope, base_intercept, s):
    """For a single step co-located with a ramp, step precedes ramp."""
    if len(step_prims) != 1 or len(ramp_prims) == 0:
        return step_prims
    
    _, bp, h = step_prims[0]
    for _, rbp, _ in ramp_prims:
        if abs(bp - rbp) < 3.0:
            # Step should precede the ramp
            candidate_bp = rbp - 0.5
            candidate = [('step', candidate_bp, h)]
            if _exact_count(candidate, ramp_prims, train_x, train_y,
                            base_slope, base_intercept, s) >= \
               _exact_count(step_prims, ramp_prims, train_x, train_y,
                            base_slope, base_intercept, s):
                return candidate
            break
    
    return step_prims


def _exact_count(step_prims, ramp_prims, train_x, train_y,
                 base_slope, base_intercept, s):
    """Count exact matches on training data."""
    r = base_intercept + base_slope * train_x
    for _, bp, val in step_prims:
        r += val * gate_step(train_x, bp, s)
    for _, bp, val in ramp_prims:
        r += val * gate_ramp(train_x, bp, s)
    return int(np.sum(np.round(train_x + r) == np.round(train_y)))


# ============================================================================
# PHASE 3: REFINE — Coordinate descent with exact-match validation
# ============================================================================

def phase3_refine(train_x, train_y, base_slope, base_intercept,
                  step_prims, ramp_prims, s=None, max_iters=3):
    """Coordinate descent: try shifting each step by ±1, keep improvements.
    
    Also handles gap refinement for co-located step+ramp pairs.
    """
    s = s or PHI ** 2
    
    def exact_matches(steps):
        r = base_intercept + base_slope * train_x
        for _, bp, val in steps:
            r += val * gate_step(train_x, bp, s)
        for _, bp, val in ramp_prims:
            r += val * gate_ramp(train_x, bp, s)
        return int(np.sum(np.round(train_x + r) == np.round(train_y)))
    
    # Coordinate descent: ±1 shifts
    for iteration in range(max_iters):
        improved = False
        current_exact = exact_matches(step_prims)
        
        for i in range(len(step_prims)):
            ptype, bp, val = step_prims[i]
            for shift in [-1.0, 1.0]:
                candidate = list(step_prims)
                candidate[i] = (ptype, bp + shift, val)
                new_exact = exact_matches(candidate)
                if new_exact > current_exact:
                    step_prims = candidate
                    current_exact = new_exact
                    improved = True
        
        if not improved:
            break
    
    # Gap refinement: for steps near ramps, step should precede ramp
    sorted_x = np.sort(train_x)
    for si in range(len(step_prims)):
        _, bp, h = step_prims[si]
        
        before = sorted_x[sorted_x < bp]
        after = sorted_x[sorted_x > bp]
        if len(before) == 0 or len(after) == 0:
            continue
        gap = after[0] - before[-1]
        if gap <= 2:
            continue
        
        for _, rbp, _ in ramp_prims:
            if abs(bp - rbp) < 3.0:
                # Try candidates where step precedes ramp
                best_bp = bp
                best_exact = exact_matches(step_prims)
                for candidate_bp in np.arange(before[-1] + 0.5, after[0], 1.0):
                    if candidate_bp <= rbp:
                        test = list(step_prims)
                        test[si] = ('step', candidate_bp, h)
                        ex = exact_matches(test)
                        if ex > best_exact:
                            best_exact = ex
                            best_bp = candidate_bp
                
                if best_bp != bp:
                    test = list(step_prims)
                    test[si] = ('step', best_bp, h)
                    if exact_matches(test) >= exact_matches(step_prims):
                        step_prims = test
                break
    
    return step_prims


# ============================================================================
# MAIN DETECTION: 3-phase pipeline
# ============================================================================

def detect_all_v5(train_x, train_y, s=None, slope_tol=0.15):
    """v5 detection: Classify → Place → Refine.
    
    Returns: (base_slope, base_intercept, step_primitives, ramp_primitives)
    """
    s = s or PHI ** 2
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    
    # Phase 1: Classify
    classification = phase1_classify(xs, ys, slope_tol)
    
    # Phase 2: Place
    base_slope, base_intercept, step_prims, ramp_prims = phase2_place(classification, s)
    
    # Phase 3: Refine
    if len(step_prims) > 0:
        step_prims = phase3_refine(train_x, train_y, base_slope, base_intercept,
                                    step_prims, ramp_prims, s)
    
    return base_slope, base_intercept, step_prims, ramp_prims


def evaluate_v5(test_x, base_slope, base_intercept, step_prims, ramp_prims, s=None):
    """Evaluate the detected decomposition."""
    s = s or PHI ** 2
    r = base_intercept + base_slope * test_x
    for ptype, bp, val in step_prims:
        r += val * gate_step(test_x, bp, s)
    for ptype, bp, val in ramp_prims:
        r += val * gate_ramp(test_x, bp, s)
    return test_x + r


# ============================================================================
# ORACLE
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
print("DETECTION v5: Simplified 3-Phase Pipeline")
print("=" * 70)

# Oracle verification
print("\n--- CORRECTED ORACLE VERIFICATION ---")
for name, fn, R, _ in fns:
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    bs, bi, steps, ramps = CORRECT_ORACLE[name]
    pred = evaluate_v5(tx, bs, bi, steps, ramps, S)
    exact = int(np.sum(np.round(pred) == ty))
    print(f"  {name:<15s}: {exact:>4d}/{R} {'✓ PERFECT' if exact == R else '✗ FAIL'}")

# Main benchmark
print("\n--- v5 DETECTION BENCHMARK ---")
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
    pred_o = evaluate_v5(tx, bs_o, bi_o, steps_o, ramps_o, S)
    exact_o = int(np.sum(np.round(pred_o) == ty))
    
    # v5 detection
    t0 = time.perf_counter()
    bs_d, bi_d, steps_d, ramps_d = detect_all_v5(trx, try_y, S)
    pred_d = evaluate_v5(tx, bs_d, bi_d, steps_d, ramps_d, S)
    t_d = time.perf_counter() - t0
    exact_d = int(np.sum(np.round(pred_d) == ty))
    
    gap = exact_o - exact_d
    
    print(f"  oracle    : {exact_o:>4d}/{R}")
    print(f"  v5_detect : {exact_d:>4d}/{R}  ({t_d:.4f}s)  gap={gap}")
    
    print(f"  detected: base_slope={bs_d:.3f}, base_intercept={bi_d:.2f}")
    for pt, bp, v in steps_d:
        print(f"    step at {bp:>7.1f}, h={v:>7.2f}")
    for pt, bp, v in ramps_d:
        print(f"    ramp at {bp:>7.1f}, Δm={v:>7.3f}")
    
    errors = np.where(np.round(pred_d) != ty)[0]
    if 0 < len(errors) <= 15:
        print(f"  errors ({len(errors)}):")
        for e in errors[:8]:
            print(f"    x={e}: true={ty[e]:.0f}, pred={pred_d[e]:.2f}")
    
    all_results[name] = {'oracle': exact_o, 'v5': exact_d, 'gap': gap,
                          'time': t_d, 'R': R}


# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\n  {'Function':<15s}  {'Oracle':>7s}  {'v5':>7s}  {'Gap':>5s}")
print(f"  {'-'*40}")

avgs = {'oracle': [], 'v5': []}
for name, _, R, _ in fns:
    r = all_results[name]
    g = r['gap']
    sign = '+' if g > 0 else ''
    print(f"  {name:<15s}  {r['oracle']:>4d}/{R}  {r['v5']:>4d}/{R}  {sign}{g}")
    avgs['oracle'].append(r['oracle'] / R * 100)
    avgs['v5'].append(r['v5'] / R * 100)

print(f"\n  {'AVERAGE':<15s}  {np.mean(avgs['oracle']):>6.1f}%  {np.mean(avgs['v5']):>6.1f}%")


# Sample efficiency
print(f"\n{'='*70}")
print("SAMPLE EFFICIENCY")
print(f"{'='*70}")
for name, fn, R, _ in [fns[0], fns[2], fns[5], fns[7]]:
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    print(f"\n  {name}:")
    for ns in [10, 20, 30, 50, 80, 100, 128]:
        if ns > R: continue
        np.random.seed(42)
        trx = np.sort(np.random.choice(R, min(ns, R), replace=False)).astype(np.float64)
        try_y = fn(trx)
        bs, bi, steps, ramps = detect_all_v5(trx, try_y, S)
        pred = evaluate_v5(tx, bs, bi, steps, ramps, S)
        exact = int(np.sum(np.round(pred) == ty))
        print(f"    {ns:>3d} ex: {exact:>3d}/{R}")


# Visualization
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Detection v5: 3-Phase Pipeline (Classify → Place → Refine)', fontsize=14, fontweight='bold')

for idx, (name, fn, R, ntr) in enumerate(fns):
    ax = axes[idx // 4][idx % 4]
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, ntr, replace=False)).astype(np.float64)
    try_y = fn(trx)
    
    bs_o, bi_o, steps_o, ramps_o = CORRECT_ORACLE[name]
    pred_o = evaluate_v5(tx, bs_o, bi_o, steps_o, ramps_o, S)
    
    bs_d, bi_d, steps_d, ramps_d = detect_all_v5(trx, try_y, S)
    pred_d = evaluate_v5(tx, bs_d, bi_d, steps_d, ramps_d, S)
    
    ax.plot(tx, ty, 'k-', lw=2, alpha=0.3, label='True')
    eo = int(np.sum(np.round(pred_o) == ty))
    ed = int(np.sum(np.round(pred_d) == ty))
    ax.plot(tx, pred_o, 'g-', lw=1, alpha=0.6, label=f'oracle: {eo}/{R}')
    ax.plot(tx, pred_d, 'r-', lw=1.2, alpha=0.7, label=f'v5: {ed}/{R}')
    
    errs = np.where(np.round(pred_d) != ty)[0]
    if 0 < len(errs) <= 20:
        ax.scatter(errs, ty[errs], c='red', s=30, zorder=5, marker='x')
    
    ax.scatter(trx, try_y, c='blue', s=5, alpha=0.3, zorder=4)
    ax.set_title(f'{name} (v5={ed}/{R})', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/tmp/detection_v5.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: /tmp/detection_v5.png")
