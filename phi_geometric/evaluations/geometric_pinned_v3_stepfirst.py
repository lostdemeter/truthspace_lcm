#!/usr/bin/env python3
"""
Pinned v3: Step-First Detection

Root cause from v3 error analysis: steps misclassified as ramps when
training data is sparse near boundaries. Fix: detect level shifts FIRST,
remove them, THEN detect slope changes (ramps).
"""

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
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < 50, 2*x, 100 - x)

def fn_rot13(x):
    x = np.asarray(x, dtype=np.float64); r = x.copy()
    r[(x >= 65) & (x <= 77)] += 13; r[(x >= 78) & (x <= 90)] -= 13
    r[(x >= 97) & (x <= 109)] += 13; r[(x >= 110) & (x <= 122)] -= 13
    return r

def fn_abs(x): return np.abs(np.asarray(x, dtype=np.float64) - 64)
def fn_saw(x): return np.asarray(x, dtype=np.float64) % 32
def fn_clamp(x): return np.clip(np.asarray(x, dtype=np.float64), 30, 100)
def fn_relu(x): return np.maximum(0, np.asarray(x, dtype=np.float64) - 40)
def fn_stair(x): return np.floor(np.asarray(x, dtype=np.float64) / 16) * 16


# ============================================================================
# STEP-FIRST DETECTION
# ============================================================================

def detect_steps_contextual(xs, rs, min_step=3.0):
    """Detect steps by checking: large jump + flat context on both sides.
    
    A STEP produces a large isolated jump: the slope at the gap is an outlier
    compared to the slopes before and after. A RAMP produces consistent slopes.
    """
    if len(xs) < 3:
        return []
    dx = np.diff(xs)
    dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)

    steps = []
    for i in range(len(slopes)):
        actual_jump = abs(dr[i])
        if actual_jump < min_step:
            continue

        # Context slopes: what slope do neighbors have?
        before_slopes = slopes[max(0, i-3):i]
        after_slopes = slopes[i+1:min(len(slopes), i+4)]
        
        sl_before = np.median(before_slopes) if len(before_slopes) > 0 else 0
        sl_after = np.median(after_slopes) if len(after_slopes) > 0 else 0
        
        # What jump would we EXPECT from a ramp continuation?
        expected_from_before = abs(sl_before) * dx[i]
        expected_from_after = abs(sl_after) * dx[i]
        expected = max(expected_from_before, expected_from_after)
        
        # Step criterion: actual jump much larger than expected from context
        if actual_jump > max(min_step, expected * 2.5):
            # Verify: this slope is an outlier compared to neighbors
            abs_slope_here = abs(slopes[i])
            abs_slope_context = max(abs(sl_before), abs(sl_after), 0.01)
            
            if abs_slope_here > 3 * abs_slope_context:
                # Slope-corrected step height: subtract ramp contribution
                context_slope = (sl_before + sl_after) / 2
                corrected_height = dr[i] - context_slope * dx[i]
                
                # Snap breakpoint to nearest half-integer
                midpoint = (xs[i] + xs[i+1]) / 2
                bp = round(midpoint - 0.5) + 0.5
                # Keep within the gap
                bp = max(xs[i] + 0.1, min(xs[i+1] - 0.1, bp))
                
                steps.append((i, bp, corrected_height))

    return steps


def remove_steps_from_residual(xs, rs, steps):
    """Subtract detected steps from residual to get step-corrected version."""
    rs_corr = rs.copy()
    for idx, bp, height in steps:
        mask = xs > bp
        rs_corr[mask] -= height
    return rs_corr


def detect_ramps(xs, rs_corr, slope_tol=0.15):
    """Detect slope changes in step-corrected residual."""
    if len(xs) < 3:
        return [], 0, 0
    dx = np.diff(xs)
    dr = np.diff(rs_corr)
    slopes = np.where(dx > 1e-10, dr / dx, 0)

    # Find segments of constant slope
    segments = []
    seg_start = 0
    seg_slopes = [slopes[0]]

    for i in range(1, len(slopes)):
        cur = np.median(seg_slopes)
        tol = max(slope_tol, abs(cur) * 0.15)
        if abs(slopes[i] - cur) > tol:
            sxs = xs[seg_start:i+1]
            srs = rs_corr[seg_start:i+1]
            sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else cur
            ic = np.median(srs - sl * sxs)
            segments.append((seg_start, i, sl, ic))
            seg_start = i
            seg_slopes = [slopes[i]]
        else:
            seg_slopes.append(slopes[i])

    sxs = xs[seg_start:]
    srs = rs_corr[seg_start:]
    sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else np.median(seg_slopes)
    ic = np.median(srs - sl * sxs)
    segments.append((seg_start, len(xs)-1, sl, ic))

    base_slope = segments[0][2] if segments else 0
    base_intercept = segments[0][3] if segments else 0

    ramps = []
    for i in range(1, len(segments)):
        pe = segments[i-1][1]
        cs = segments[i][0]
        bp = (xs[pe] + xs[cs]) / 2
        delta = segments[i][2] - segments[i-1][2]
        if abs(delta) > 0.03:
            ramps.append(('ramp', bp, delta))

    return ramps, base_slope, base_intercept


# ============================================================================
# APPROACH: Step-First Hinge
# ============================================================================

def approach_stepfirst(train_x, train_y, test_x, s=None, slope_tol=0.15):
    """Detect steps first, remove them, then detect ramps."""
    s = s or PHI ** 2
    order = np.argsort(train_x)
    xs, ys = train_x[order], train_y[order]
    rs = ys - xs

    # Phase 1: Detect steps
    raw_steps = detect_steps_contextual(xs, rs)
    step_prims = [('step', bp, h) for _, bp, h in raw_steps]

    # Phase 2: Remove steps, detect ramps
    rs_corr = remove_steps_from_residual(xs, rs, raw_steps)
    ramp_prims, base_slope, base_intercept = detect_ramps(xs, rs_corr, slope_tol)

    # Evaluate
    r_test = base_intercept + base_slope * test_x
    for ptype, bp, val in step_prims:
        r_test += val * gate_step(test_x, bp, s)
    for ptype, bp, val in ramp_prims:
        r_test += val * gate_ramp(test_x, bp, s)

    return test_x + r_test, step_prims, ramp_prims


def approach_best_of(train_x, train_y, test_x, s=None):
    """Run both hinge and step-first, pick better on training exact matches.
    
    SSE is the wrong metric: step-first concentrates error at one boundary point
    (large SSE) but gets more test points right. Use exact-match count instead.
    """
    s = s or PHI ** 2
    
    pred_h = approach_hinge_orig(train_x, train_y, test_x, s)
    pred_sf, steps, ramps = approach_stepfirst(train_x, train_y, test_x, s)
    
    # Evaluate on training data using exact match count
    train_h = approach_hinge_orig(train_x, train_y, train_x, s)
    train_sf, _, _ = approach_stepfirst(train_x, train_y, train_x, s)
    
    exact_h = int(np.sum(np.round(train_h) == np.round(train_y)))
    exact_sf = int(np.sum(np.round(train_sf) == np.round(train_y)))
    
    # Prefer step-first when it detects steps and matches at least as well
    if len(steps) > 0 and exact_sf >= exact_h:
        # Tiebreaker: if equal exact matches, check max training error
        # A bad step classification produces large max error at boundary
        if exact_sf == exact_h:
            maxe_h = float(np.abs(train_h - train_y).max())
            maxe_sf = float(np.abs(train_sf - train_y).max())
            if maxe_sf > maxe_h * 1.5 + 1.0:
                return pred_h, 'hinge', [], []
        return pred_sf, 'step_first', steps, ramps
    elif exact_sf > exact_h:
        return pred_sf, 'step_first', steps, ramps
    else:
        return pred_h, 'hinge', [], []


# Original hinge for comparison
def detect_segments_orig(xs, rs, slope_tol=0.15):
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
            segments.append((seg_start, i, sl, ic)); seg_start = i; seg_slopes = [slopes[i]]
        else:
            seg_slopes.append(slopes[i])
    sxs, srs = xs[seg_start:], rs[seg_start:]
    sl = np.polyfit(sxs, srs, 1)[0] if len(sxs) >= 2 else np.median(seg_slopes)
    ic = np.median(srs - sl * sxs)
    segments.append((seg_start, len(xs)-1, sl, ic))
    return segments

def approach_hinge_orig(train_x, train_y, test_x, s=None, slope_tol=0.15):
    s = s or PHI ** 2
    order = np.argsort(train_x); xs, ys = train_x[order], train_y[order]; rs = ys - xs
    segs = detect_segments_orig(xs, rs, slope_tol)
    if not segs: return test_x.copy()
    _, _, bs, bi = segs[0]; prims = []
    for i in range(1, len(segs)):
        pe = segs[i-1][1]; cs = segs[i][0]; bp = (xs[pe] + xs[cs]) / 2
        ps, pi = segs[i-1][2], segs[i-1][3]; csl, ci = segs[i][2], segs[i][3]
        ds = csl - ps; rp = ps * bp + pi; rc = csl * bp + ci; j = rc - rp
        if abs(j) > 0.3: prims.append(('step', bp, j))
        if abs(ds) > 0.03: prims.append(('ramp', bp, ds))
    r_t = bi + bs * test_x
    for pt, bp, v in prims:
        if pt == 'step': r_t += v * gate_step(test_x, bp, s)
        else: r_t += v * gate_ramp(test_x, bp, s)
    return test_x + r_t


# ============================================================================
# BENCHMARK
# ============================================================================

print("=" * 70)
print("PINNED v3: Step-First Detection")
print("=" * 70)

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

all_res = {}

for fn_name, fn, R, ntr in fns:
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, ntr, replace=False)).astype(np.float64)
    try_y = fn(trx)

    print(f"\n{'='*55}")
    print(f"  {fn_name} ({ntr} train / {R} test)")
    print(f"{'='*55}")

    # Original hinge
    t0 = time.perf_counter()
    pred_h = approach_hinge_orig(trx, try_y, tx)
    t_h = time.perf_counter() - t0
    ex_h = int(np.sum(np.round(pred_h) == ty))
    me_h = float(np.abs(pred_h - ty).max())

    # Step-first
    t0 = time.perf_counter()
    pred_sf, steps, ramps = approach_stepfirst(trx, try_y, tx)
    t_sf = time.perf_counter() - t0
    ex_sf = int(np.sum(np.round(pred_sf) == ty))
    me_sf = float(np.abs(pred_sf - ty).max())

    # Best-of ensemble
    t0 = time.perf_counter()
    pred_bo, which, bo_steps, bo_ramps = approach_best_of(trx, try_y, tx)
    t_bo = time.perf_counter() - t0
    ex_bo = int(np.sum(np.round(pred_bo) == ty))
    me_bo = float(np.abs(pred_bo - ty).max())

    print(f"  hinge_orig  : {ex_h:>4d}/{R} exact, max_err={me_h:>8.4f}, t={t_h:.4f}s")
    print(f"  step_first  : {ex_sf:>4d}/{R} exact, max_err={me_sf:>8.4f}, t={t_sf:.4f}s")
    print(f"    detected: {len(steps)} steps, {len(ramps)} ramps")
    print(f"  best_of     : {ex_bo:>4d}/{R} exact, max_err={me_bo:>8.4f}, t={t_bo:.4f}s  (via {which})")

    # Error locations for best_of
    errs = np.where(np.round(pred_bo) != ty)[0]
    if len(errs) > 0 and len(errs) <= 15:
        print(f"    errors at: {errs.tolist()}")
        for e in errs[:5]:
            print(f"      x={e}: true={ty[e]:.0f}, pred={pred_bo[e]:.2f}")

    all_res[fn_name] = {
        'hinge': (ex_h, me_h, t_h),
        'stepfirst': (ex_sf, me_sf, t_sf),
        'best_of': (ex_bo, me_bo, t_bo),
        'steps': bo_steps, 'ramps': bo_ramps, 'which': which
    }

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\n  {'Function':<15s}  {'Hinge':>8s}  {'StepFirst':>10s}  {'BestOf':>10s}  {'Delta':>8s}")
print(f"  {'-'*65}")

h_avg, sf_avg, bo_avg = [], [], []
for fn_name, _, R, _ in fns:
    if fn_name in all_res:
        ex_h = all_res[fn_name]['hinge'][0]
        ex_sf = all_res[fn_name]['stepfirst'][0]
        ex_bo = all_res[fn_name]['best_of'][0]
        delta = ex_bo - ex_h
        sign = '+' if delta > 0 else ''
        w = all_res[fn_name].get('which', '?')
        print(f"  {fn_name:<15s}  {ex_h:>4d}/{R}  {ex_sf:>6d}/{R}  {ex_bo:>6d}/{R}  {sign}{delta:>5d}  ({w})")
        h_avg.append(ex_h / R * 100)
        sf_avg.append(ex_sf / R * 100)
        bo_avg.append(ex_bo / R * 100)

print(f"\n  {'AVERAGE':<15s}  {np.mean(h_avg):>7.1f}%  {np.mean(sf_avg):>9.1f}%  {np.mean(bo_avg):>9.1f}%  "
      f"{'+' if np.mean(bo_avg) > np.mean(h_avg) else ''}"
      f"{np.mean(bo_avg) - np.mean(h_avg):>6.1f}%")

# Sample efficiency for step-heavy functions
print(f"\n{'='*70}")
print("SAMPLE EFFICIENCY (step-heavy functions)")
print(f"{'='*70}")
for fn_name, fn, R, _ in [fns[0], fns[2], fns[7]]:  # tolower, ROT13, staircase
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    print(f"\n  {fn_name}:")
    for ns in [10, 20, 30, 50, 80, 100, 128]:
        if ns > R: continue
        np.random.seed(42)
        trx = np.sort(np.random.choice(R, min(ns, R), replace=False)).astype(np.float64)
        try_y = fn(trx)
        ph = approach_hinge_orig(trx, try_y, tx)
        pbo, w, _, _ = approach_best_of(trx, try_y, tx)
        eh = int(np.sum(np.round(ph) == ty))
        ebo = int(np.sum(np.round(pbo) == ty))
        print(f"    {ns:>3d} ex: hinge={eh:>3d}  best_of={ebo:>3d}  /{R}  ({w})")

print(f"\nSaved: /tmp/geometric_pinned_v3_stepfirst.png")

# Quick visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Step-First vs Original Hinge', fontsize=14, fontweight='bold')
for idx, (fn_name, fn, R, ntr) in enumerate(fns):
    ax = axes[idx // 4][idx % 4]
    tx = np.arange(R, dtype=np.float64); ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, ntr, replace=False)).astype(np.float64)
    try_y = fn(trx)
    ph = approach_hinge_orig(trx, try_y, tx)
    pbo, w, _, _ = approach_best_of(trx, try_y, tx)
    ax.plot(tx, ty, 'k-', lw=2, alpha=0.3, label='True')
    eh = int(np.sum(np.round(ph) == ty))
    ebo = int(np.sum(np.round(pbo) == ty))
    ax.plot(tx, ph, 'r-', lw=1, alpha=0.7, label=f'hinge: {eh}/{R}')
    ax.plot(tx, pbo, 'b-', lw=1, alpha=0.7, label=f'best_of: {ebo}/{R}')
    ax.set_title(fn_name, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/geometric_pinned_v3_stepfirst.png', dpi=150, bbox_inches='tight')
plt.close()
