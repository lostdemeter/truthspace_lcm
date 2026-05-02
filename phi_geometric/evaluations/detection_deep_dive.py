#!/usr/bin/env python3
"""
Detection Deep Dive: The Path to 100%

Three questions:
  Q1: ORACLE — With perfect breakpoints, does the gate give 100%?
      If no → gate precision is the bottleneck, not detection.
      If yes → detection IS the only bottleneck, and 100% is achievable.

  Q2: ANATOMY — For each error, exactly what went wrong?
      - Mislocalization: breakpoint at wrong position
      - Misclassification: step detected as ramp (or vice versa)
      - Missed: breakpoint not detected at all
      - Amplitude error: breakpoint found but wrong height/slope

  Q3: STRUCTURE — What structural constraints could we exploit?
      - Paired steps (RECTs have matching up/down)
      - Consistent amplitudes (all steps same height)
      - Periodicity (sawtooth, staircase)
      - Boundary alignment (breakpoints at half-integers)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# Test functions with their TRUE decomposition
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
# ORACLE: Perfect breakpoints → gate evaluation
# ============================================================================

# TRUE decompositions: (base_slope, base_intercept, [(type, position, amplitude)])
# r(x) = y - x = base_intercept + base_slope * x + Σ primitives

TRUE_DECOMPOSITIONS = {
    'tolower': {
        'fn': fn_tolower, 'range': 128,
        'base_slope': 0.0, 'base_intercept': 0.0,
        # r(x) = 0 for x<65, 32 for 65≤x≤90, 0 for x>90
        # = 32 * [step(x,64.5) - step(x,90.5)]
        'primitives': [('step', 64.5, 32.0), ('step', 90.5, -32.0)],
        'description': 'RECT: +32 for x in [65,90]'
    },
    'secret_fn': {
        'fn': fn_secret, 'range': 100,
        'base_slope': 0.0, 'base_intercept': 0.0,
        # y = 2x for x<50, 100-x for x≥50
        # r = y - x = x for x<50, 100-2x for x≥50
        # r = 0 + 1*ramp(x,0) + (-3)*ramp(x,50) + ... wait
        # Actually: r(x) = x for x<50, r(x) = 100-2x for x≥50
        # Base: r starts at 0, slope = 1
        # At x=50: slope changes from +1 to -2 (delta = -3)
        # Also: r(49.99) ≈ 50, r(50) = 0 → jump of -50
        # r = 0 + 1*x + (-3)*ramp(x,49.5) + (-50)*step(x,49.5)
        # Wait: r(0)=0, r(49)=49. slope = 1. base_slope=1, base_intercept=0.
        # At x=50: r=100-2*50=0. r(49)=49, r(50)=0. Jump = -49. Slope change = -3.
        # Hmm, let me compute more carefully.
        # y(49) = 2*49 = 98, r(49) = 98-49 = 49
        # y(50) = 100-50 = 50, r(50) = 50-50 = 0
        # Jump in r at 49.5: 0 - 49 = -49... but r at bp=49.5 from left: 49.5, from right: 100-2*49.5-49.5=1
        # Actually r(x) from right = 100-2x for x≥50, so r(49.5)_right = 100-99 = 1
        # r(49.5)_left = 49.5
        # Jump = 1 - 49.5 = -48.5
        # Slope: left = +1, right = -2. delta_slope = -3.
        # So: base_slope=1, base_intercept=0
        # At 49.5: step of -48.5, ramp of -3
        'base_slope': 1.0, 'base_intercept': 0.0,
        'primitives': [('step', 49.5, -48.5), ('ramp', 49.5, -3.0)],
        'description': 'Slope +1 → slope -2 with jump of -48.5 at x=49.5'
    },
    'ROT13': {
        'fn': fn_rot13, 'range': 128,
        'base_slope': 0.0, 'base_intercept': 0.0,
        # r(x): +13 for x in [65,77], -13 for x in [78,90],
        #        +13 for x in [97,109], -13 for x in [110,122]
        # Each is a RECT
        'primitives': [
            ('step', 64.5, 13.0), ('step', 77.5, -13.0),
            ('step', 77.5, -13.0), ('step', 90.5, 13.0),
            ('step', 96.5, 13.0), ('step', 109.5, -13.0),
            ('step', 109.5, -13.0), ('step', 122.5, 13.0),
        ],
        'description': '4 RECTs: ±13 in specific ranges'
    },
    'abs_centered': {
        'fn': fn_abs, 'range': 128,
        # y = |x-64|, r = |x-64| - x = -64 for x<64, x-128 for x≥64
        # Wait: r(0)=64-0=64, r(63)=1-63=-62... no.
        # y(0)=|0-64|=64, r(0)=64-0=64
        # y(63)=|63-64|=1, r(63)=1-63=-62
        # y(64)=0, r(64)=0-64=-64
        # y(65)=1, r(65)=1-65=-64
        # y(127)=63, r(127)=63-127=-64
        # So r(x) = 64-2x for x<64, -64 for x≥64
        # base: r(0)=64, slope=-2 for x<64
        # at x=63.5: r changes from 64-2*63.5=-63 to -64. Hmm not quite.
        # r(x) = |x-64| - x = (64-x) - x = 64-2x for x≤64
        # r(x) = (x-64) - x = -64 for x>64
        # slope left = -2, slope right = 0. delta = +2
        # r at 63.5 from left: 64-127=-63. from right: -64. jump = -1
        'base_slope': -2.0, 'base_intercept': 64.0,
        'primitives': [('ramp', 63.5, 2.0), ('step', 63.5, -1.0)],
        'description': 'V-shape: slope -2 then 0, with small jump'
    },
    'sawtooth_32': {
        'fn': fn_saw, 'range': 128,
        # y = x%32, r = x%32 - x = -floor(x/32)*32
        # r(0..31)=0, r(32..63)=-32, r(64..95)=-64, r(96..127)=-96
        # Pure staircase in residual
        'base_slope': 0.0, 'base_intercept': 0.0,
        'primitives': [
            ('step', 31.5, -32.0), ('step', 63.5, -32.0),
            ('step', 95.5, -32.0),
        ],
        'description': 'Staircase: steps of -32 at period boundaries'
    },
    'clamp': {
        'fn': fn_clamp, 'range': 128,
        # y = clip(x,30,100), r = clip(x,30,100) - x
        # r(x<30) = 30-x, r(30≤x≤100) = 0, r(x>100) = 100-x
        # Left: slope=-1 offset=30. At x=29.5: ramp with delta_slope=+1
        # Right: at x=100.5: ramp with delta_slope=-1
        'base_slope': -1.0, 'base_intercept': 30.0,
        'primitives': [('ramp', 29.5, 1.0), ('ramp', 100.5, -1.0)],
        'description': 'Ramp up to 30, flat, ramp down from 100'
    },
    'relu_shifted': {
        'fn': fn_relu, 'range': 128,
        # y = max(0,x-40), r = max(0,x-40) - x
        # r(x<40) = -x, r(x≥40) = x-40-x = -40
        # base: slope=-1, intercept=0. At x=39.5: ramp delta=+1
        'base_slope': -1.0, 'base_intercept': 0.0,
        'primitives': [('ramp', 39.5, 1.0)],
        'description': 'Single ramp at x=39.5'
    },
    'staircase': {
        'fn': fn_stair, 'range': 128,
        # y = floor(x/16)*16, r = floor(x/16)*16 - x = -(x%16)
        # r(0)=0, r(15)=-15, r(16)=0, r(31)=-15, ...
        # Within each period: slope=-1. At each boundary: jump of +15
        # Wait: r(15)=-15, r(16)=0. Jump = +15. But slope within segment is -1.
        # So: base_slope=-1, base_intercept=0
        # At each boundary x=15.5, 31.5, ..., 111.5: step of +15 and ramp of +1
        # Wait, the slope is -1 everywhere (base), and at each boundary the residual
        # resets. Actually: base is 0 slope 0.
        # r(x) = -(x%16). This is a sawtooth with period 16 and amp 15.
        # In terms of primitives: step of +16 at each boundary (15 from reset + 1 from slope)
        # Hmm, let me think about this more carefully.
        # r(0)=0, r(1)=-1, ..., r(15)=-15, r(16)=0, r(17)=-1, ...
        # The residual decreases by 1 per step within each period.
        # At x=15.5: r goes from -15 to 0 (jump of +15)
        # The base slope would need to be -1 to account for the decrease.
        # But then: r_base = -x. At x=15, r_base=-15, r_actual=-15. OK.
        # At x=16, r_base=-16, r_actual=0. Need +16 from steps.
        # So at x=15.5: step of +16.
        # At x=31.5: r_base=-31.5, accumulated steps so far: +16.
        # r_pred = -31.5 + 16 = -15.5. r_actual(31) = -15. Close but off.
        # Hmm. Actually r(31) = -(31%16) = -15. r_base(31) = -31. Steps so far: +16.
        # -31 + 16 = -15. Yes!
        # r(32) = 0. r_base(32) = -32. Steps: +16. -32+16=-16. Need another +16.
        # So step at 31.5 of +16. Total steps: +32. r(32) = -32+32 = 0. ✓
        'base_slope': -1.0, 'base_intercept': 0.0,
        'primitives': [
            ('step', 15.5, 16.0), ('step', 31.5, 16.0),
            ('step', 47.5, 16.0), ('step', 63.5, 16.0),
            ('step', 79.5, 16.0), ('step', 95.5, 16.0),
            ('step', 111.5, 16.0),
        ],
        'description': 'Steps of +16 every 16 units, base slope -1'
    },
}

# Fix ROT13 decomposition (was doubled)
TRUE_DECOMPOSITIONS['ROT13']['primitives'] = [
    ('step', 64.5, 13.0), ('step', 77.5, -26.0),
    ('step', 90.5, 13.0),
    ('step', 96.5, 13.0), ('step', 109.5, -26.0),
    ('step', 122.5, 13.0),
]

S = PHI ** 2

# ============================================================================
# Q1: ORACLE TEST — Perfect breakpoints → 100%?
# ============================================================================

print("=" * 70)
print("Q1: ORACLE TEST — Do perfect breakpoints give 100%?")
print("=" * 70)

oracle_results = {}

for name, info in TRUE_DECOMPOSITIONS.items():
    fn = info['fn']
    R = info['range']
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    
    bs = info['base_slope']
    bi = info['base_intercept']
    prims = info['primitives']
    
    # Test at multiple sharpness values
    for s_val in [PHI**2, 5.0, 10.0, 20.0, 50.0]:
        r_pred = bi + bs * tx
        for ptype, bp, amp in prims:
            if ptype == 'step':
                r_pred += amp * gate_step(tx, bp, s_val)
            elif ptype == 'ramp':
                r_pred += amp * gate_ramp(tx, bp, s_val)
        
        pred = tx + r_pred
        exact = int(np.sum(np.round(pred) == ty))
        max_err = float(np.abs(pred - ty).max())
        
        if s_val == PHI**2:
            oracle_results[name] = {'exact': exact, 'max_err': max_err, 'R': R}
        
        if s_val == PHI**2 or exact == R:
            pct = exact / R * 100
            marker = " ✓ PERFECT" if exact == R else ""
            print(f"  {name:<15s} s={s_val:>5.2f}: {exact:>4d}/{R} ({pct:5.1f}%) "
                  f"max_err={max_err:.6f}{marker}")
            if exact == R:
                break  # Found minimal s for 100%

# Show where oracle fails at s=φ²
print(f"\nOracle errors at s=φ²:")
for name, info in TRUE_DECOMPOSITIONS.items():
    fn = info['fn']
    R = info['range']
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    
    r_pred = info['base_intercept'] + info['base_slope'] * tx
    for ptype, bp, amp in info['primitives']:
        if ptype == 'step':
            r_pred += amp * gate_step(tx, bp, S)
        elif ptype == 'ramp':
            r_pred += amp * gate_ramp(tx, bp, S)
    pred = tx + r_pred
    
    errors = np.where(np.round(pred) != ty)[0]
    if len(errors) > 0:
        print(f"\n  {name}: {len(errors)} errors")
        for e in errors[:10]:
            print(f"    x={e}: true={ty[e]:.0f}, pred={pred[e]:.4f}, "
                  f"rounded={np.round(pred[e]):.0f}, err={abs(pred[e]-ty[e]):.6f}")

# ============================================================================
# Q2: DETECTION ANATOMY — What does the detector see vs truth?
# ============================================================================

print()
print("=" * 70)
print("Q2: DETECTION ANATOMY — What does the detector see?")
print("=" * 70)

# Import detection functions from v3
def detect_segments(xs, rs, slope_tol=0.15):
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


def detect_steps_contextual(xs, rs, min_step=3.0):
    if len(xs) < 3: return []
    dx = np.diff(xs); dr = np.diff(rs)
    slopes = np.where(dx > 1e-10, dr / dx, 0)
    steps = []
    for i in range(len(slopes)):
        actual_jump = abs(dr[i])
        if actual_jump < min_step: continue
        before_slopes = slopes[max(0, i-3):i]
        after_slopes = slopes[i+1:min(len(slopes), i+4)]
        sl_before = np.median(before_slopes) if len(before_slopes) > 0 else 0
        sl_after = np.median(after_slopes) if len(after_slopes) > 0 else 0
        expected = max(abs(sl_before) * dx[i], abs(sl_after) * dx[i])
        if actual_jump > max(min_step, expected * 2.5):
            abs_slope_here = abs(slopes[i])
            abs_slope_context = max(abs(sl_before), abs(sl_after), 0.01)
            if abs_slope_here > 3 * abs_slope_context:
                context_slope = (sl_before + sl_after) / 2
                corrected_height = dr[i] - context_slope * dx[i]
                midpoint = (xs[i] + xs[i+1]) / 2
                bp = round(midpoint - 0.5) + 0.5
                bp = max(xs[i] + 0.1, min(xs[i+1] - 0.1, bp))
                steps.append((i, bp, corrected_height))
    return steps


for name, info in TRUE_DECOMPOSITIONS.items():
    fn = info['fn']
    R = info['range']
    n_train = {
        'tolower': 50, 'secret_fn': 40, 'ROT13': 60,
        'abs_centered': 40, 'sawtooth_32': 50, 'clamp': 40,
        'relu_shifted': 30, 'staircase': 50
    }[name]
    
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, n_train, replace=False)).astype(np.float64)
    try_y = fn(trx)
    rs = try_y - trx
    
    print(f"\n{'='*55}")
    print(f"  {name}: TRUE = {info['description']}")
    print(f"{'='*55}")
    print(f"  TRUE: base_slope={info['base_slope']}, base_intercept={info['base_intercept']}")
    for ptype, bp, amp in info['primitives']:
        print(f"    {ptype:>4s} at {bp:>6.1f}, amplitude={amp:>7.1f}")
    
    # What step-first detects
    raw_steps = detect_steps_contextual(trx, rs)
    print(f"\n  STEP-FIRST detected {len(raw_steps)} steps:")
    for idx, bp, h in raw_steps:
        print(f"    step at {bp:>6.1f}, height={h:>7.2f} "
              f"(between x={trx[idx]:.0f} and x={trx[idx+1]:.0f}, gap={trx[idx+1]-trx[idx]:.0f})")
    
    # What hinge detects
    segments = detect_segments(trx, rs)
    print(f"\n  HINGE detected {len(segments)} segments:")
    for si, (ss, se, sl, ic) in enumerate(segments):
        print(f"    seg[{si}]: x=[{trx[ss]:.0f},{trx[se]:.0f}], "
              f"slope={sl:>7.3f}, intercept={ic:>7.2f}")
    
    # Hinge transitions
    print(f"  HINGE transitions:")
    for i in range(1, len(segments)):
        pe = segments[i-1][1]; cs = segments[i][0]
        bp = (trx[pe] + trx[cs]) / 2
        ps, pi = segments[i-1][2], segments[i-1][3]
        csl, ci = segments[i][2], segments[i][3]
        ds = csl - ps
        rp = ps * bp + pi; rc = csl * bp + ci; j = rc - rp
        classification = []
        if abs(j) > 0.3: classification.append(f"step(h={j:.2f})")
        if abs(ds) > 0.03: classification.append(f"ramp(Δm={ds:.3f})")
        print(f"    at {bp:>7.1f}: {', '.join(classification) if classification else 'none'}")
    
    # Nearest training points to each true breakpoint
    print(f"\n  TRAINING DATA near true breakpoints:")
    for ptype, bp, amp in info['primitives']:
        dists = np.abs(trx - bp)
        nearest_idx = np.argmin(dists)
        nearest_left = trx[trx < bp]
        nearest_right = trx[trx > bp]
        gap_left = bp - nearest_left[-1] if len(nearest_left) > 0 else float('inf')
        gap_right = nearest_right[0] - bp if len(nearest_right) > 0 else float('inf')
        total_gap = gap_left + gap_right
        print(f"    {ptype:>4s} at {bp:>6.1f}: nearest_left={bp-gap_left:.0f} ({gap_left:.1f} away), "
              f"nearest_right={bp+gap_right:.0f} ({gap_right:.1f} away), total_gap={total_gap:.1f}")


# ============================================================================
# Q3: CAN WE CLOSE THE GAP? Specific analysis per function
# ============================================================================

print()
print("=" * 70)
print("Q3: PER-FUNCTION GAP ANALYSIS")
print("=" * 70)

for name, info in TRUE_DECOMPOSITIONS.items():
    fn = info['fn']
    R = info['range']
    n_train = {
        'tolower': 50, 'secret_fn': 40, 'ROT13': 60,
        'abs_centered': 40, 'sawtooth_32': 50, 'clamp': 40,
        'relu_shifted': 30, 'staircase': 50
    }[name]
    
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, n_train, replace=False)).astype(np.float64)
    try_y = fn(trx)
    
    # Oracle prediction
    r_oracle = info['base_intercept'] + info['base_slope'] * tx
    for ptype, bp, amp in info['primitives']:
        if ptype == 'step':
            r_oracle += amp * gate_step(tx, bp, S)
        elif ptype == 'ramp':
            r_oracle += amp * gate_ramp(tx, bp, S)
    pred_oracle = tx + r_oracle
    oracle_exact = int(np.sum(np.round(pred_oracle) == ty))
    
    # Current best (step-first or hinge)
    # Run step-first manually
    rs = try_y - trx
    raw_steps = detect_steps_contextual(trx, rs)
    step_prims = [('step', bp, h) for _, bp, h in raw_steps]
    
    rs_corr = rs.copy()
    for idx, bp, height in raw_steps:
        rs_corr[trx > bp] -= height
    
    segs = detect_segments(trx, rs_corr)
    bs_det, bi_det = segs[0][2], segs[0][3]
    ramp_prims = []
    for i in range(1, len(segs)):
        pe, cs = segs[i-1][1], segs[i][0]
        bp = (trx[pe] + trx[cs]) / 2
        ds = segs[i][2] - segs[i-1][2]
        if abs(ds) > 0.03:
            ramp_prims.append(('ramp', bp, ds))
    
    r_detected = bi_det + bs_det * tx
    for ptype, bp, val in step_prims:
        r_detected += val * gate_step(tx, bp, S)
    for ptype, bp, val in ramp_prims:
        r_detected += val * gate_ramp(tx, bp, S)
    pred_detected = tx + r_detected
    det_exact = int(np.sum(np.round(pred_detected) == ty))
    
    gap = oracle_exact - det_exact
    
    print(f"\n  {name}: oracle={oracle_exact}/{R}, detected={det_exact}/{R}, gap={gap}")
    
    if gap > 0:
        # Which errors are from detection vs gate?
        oracle_errors = set(np.where(np.round(pred_oracle) != ty)[0])
        detect_errors = set(np.where(np.round(pred_detected) != ty)[0])
        
        gate_errors = oracle_errors  # errors even with perfect breakpoints
        detection_only = detect_errors - oracle_errors  # errors purely from detection
        
        print(f"    Gate errors (irreducible at s=φ²): {len(gate_errors)} at {sorted(gate_errors)[:10]}")
        print(f"    Detection errors (fixable): {len(detection_only)} at {sorted(detection_only)[:15]}")
        
        # For each detection error, identify the cause
        for e in sorted(detection_only)[:10]:
            # Which true breakpoint is this nearest to?
            nearest_bp = None
            min_dist = float('inf')
            for ptype, bp, amp in info['primitives']:
                if abs(e - bp) < min_dist:
                    min_dist = abs(e - bp)
                    nearest_bp = (ptype, bp, amp)
            
            # Was this breakpoint detected?
            detected_near = None
            for ptype, bp, amp in step_prims + ramp_prims:
                if abs(bp - nearest_bp[1]) < 10:
                    detected_near = (ptype, bp, amp)
                    break
            
            if detected_near:
                cause = f"mislocalized: true={nearest_bp[1]:.1f}, detected={detected_near[1]:.1f}, "
                cause += f"Δ={abs(detected_near[1]-nearest_bp[1]):.1f}"
                if detected_near[0] != nearest_bp[0]:
                    cause += f" MISCLASSIFIED: true={nearest_bp[0]}, detected={detected_near[0]}"
                if abs(detected_near[2] - nearest_bp[2]) > 0.5:
                    cause += f" BAD_AMP: true={nearest_bp[2]:.1f}, detected={detected_near[2]:.1f}"
            else:
                cause = f"MISSED breakpoint at {nearest_bp[1]:.1f}"
            
            print(f"      x={e}: true_y={ty[e]:.0f} pred={pred_detected[e]:.2f} → {cause}")


# ============================================================================
# Q4: WHAT SHARPNESS GIVES ORACLE 100%?
# ============================================================================

print()
print("=" * 70)
print("Q4: MINIMUM SHARPNESS FOR ORACLE 100%")
print("=" * 70)

for name, info in TRUE_DECOMPOSITIONS.items():
    fn = info['fn']
    R = info['range']
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    
    found_100 = False
    for s_val in [PHI**2, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]:
        r_pred = info['base_intercept'] + info['base_slope'] * tx
        for ptype, bp, amp in info['primitives']:
            if ptype == 'step':
                r_pred += amp * gate_step(tx, bp, s_val)
            elif ptype == 'ramp':
                r_pred += amp * gate_ramp(tx, bp, s_val)
        pred = tx + r_pred
        exact = int(np.sum(np.round(pred) == ty))
        
        if exact == R:
            print(f"  {name:<15s}: 100% at s≥{s_val:.2f}")
            found_100 = True
            break
    
    if not found_100:
        # Show best
        print(f"  {name:<15s}: CANNOT reach 100% even at s=100 (best={exact}/{R})")
        # Show the errors at s=100
        r_pred = info['base_intercept'] + info['base_slope'] * tx
        for ptype, bp, amp in info['primitives']:
            if ptype == 'step':
                r_pred += amp * gate_step(tx, bp, 100.0)
            elif ptype == 'ramp':
                r_pred += amp * gate_ramp(tx, bp, 100.0)
        pred = tx + r_pred
        errors = np.where(np.round(pred) != ty)[0]
        for e in errors[:5]:
            print(f"    x={e}: true={ty[e]:.0f}, pred={pred[e]:.4f}, err={abs(pred[e]-ty[e]):.6f}")


# ============================================================================
# Q5: GATE PRECISION AT HALF-INTEGER BREAKPOINTS
# ============================================================================

print()
print("=" * 70)
print("Q5: GATE PRECISION — Step/Ramp values at integer offsets")
print("=" * 70)

print("\n  gate_step(x, 0.5, s) at integer x (s=φ²):")
for x in range(-3, 5):
    val = float(gate_step(np.array([x], dtype=np.float64), 0.5, S))
    print(f"    x={x:>3d}: step={val:>12.8f}  (target={'1.0' if x >= 1 else '0.0'})")

print(f"\n  gate_ramp(x, 0.5, s) at integer x (s=φ²):")
for x in range(-3, 5):
    val = float(gate_ramp(np.array([x], dtype=np.float64), 0.5, S))
    expected = max(0, x - 0.5)
    print(f"    x={x:>3d}: ramp={val:>12.8f}  (target={expected:.1f})")

# What's the maximum step_height where gate_step still rounds correctly?
print(f"\n  Maximum step height for correct rounding at s=φ²:")
print(f"    gate_step(0, 0.5, φ²) = {float(gate_step(np.array([0.0]), 0.5, S)):.8f}")
print(f"    For step of height h, error at x=0 = h × {float(gate_step(np.array([0.0]), 0.5, S)):.8f}")
print(f"    Rounds correctly when h × {float(gate_step(np.array([0.0]), 0.5, S)):.8f} < 0.5")
max_h = 0.5 / abs(float(gate_step(np.array([0.0]), 0.5, S)))
print(f"    → Max h = {max_h:.1f}")

print(f"\n    gate_step(1, 0.5, φ²) = {float(gate_step(np.array([1.0]), 0.5, S)):.8f}")
print(f"    For step of height h, error at x=1 = h × (1 - {float(gate_step(np.array([1.0]), 0.5, S)):.8f})")
err_at_1 = 1.0 - float(gate_step(np.array([1.0]), 0.5, S))
print(f"    = h × {err_at_1:.8f}")
max_h_1 = 0.5 / err_at_1
print(f"    → Max h = {max_h_1:.1f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 18))
gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)
fig.suptitle('Detection Deep Dive: Oracle vs Detected vs True',
             fontsize=15, fontweight='bold', y=1.01)

fn_list = list(TRUE_DECOMPOSITIONS.keys())

for idx, name in enumerate(fn_list):
    info = TRUE_DECOMPOSITIONS[name]
    fn = info['fn']
    R = info['range']
    n_train = {
        'tolower': 50, 'secret_fn': 40, 'ROT13': 60,
        'abs_centered': 40, 'sawtooth_32': 50, 'clamp': 40,
        'relu_shifted': 30, 'staircase': 50
    }[name]
    
    row = idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])
    
    tx = np.arange(R, dtype=np.float64)
    ty = fn(tx)
    
    # True function
    ax.plot(tx, ty, 'k-', linewidth=2, alpha=0.3, label='True')
    
    # Oracle (perfect breakpoints)
    r_oracle = info['base_intercept'] + info['base_slope'] * tx
    for ptype, bp, amp in info['primitives']:
        if ptype == 'step':
            r_oracle += amp * gate_step(tx, bp, S)
        elif ptype == 'ramp':
            r_oracle += amp * gate_ramp(tx, bp, S)
    pred_oracle = tx + r_oracle
    oracle_exact = int(np.sum(np.round(pred_oracle) == ty))
    ax.plot(tx, pred_oracle, 'g-', linewidth=1.2, alpha=0.7,
            label=f'oracle: {oracle_exact}/{R}')
    
    # Detected (step-first)
    np.random.seed(42)
    trx = np.sort(np.random.choice(R, n_train, replace=False)).astype(np.float64)
    try_y = fn(trx)
    rs = try_y - trx
    
    raw_steps = detect_steps_contextual(trx, rs)
    step_prims = [('step', bp, h) for _, bp, h in raw_steps]
    rs_corr = rs.copy()
    for idx2, bp, height in raw_steps:
        rs_corr[trx > bp] -= height
    segs = detect_segments(trx, rs_corr)
    bs_det, bi_det = segs[0][2], segs[0][3]
    ramp_prims = []
    for i in range(1, len(segs)):
        pe, cs = segs[i-1][1], segs[i][0]
        bp = (trx[pe] + trx[cs]) / 2
        ds = segs[i][2] - segs[i-1][2]
        if abs(ds) > 0.03:
            ramp_prims.append(('ramp', bp, ds))
    
    r_detected = bi_det + bs_det * tx
    for ptype, bp, val in step_prims:
        r_detected += val * gate_step(tx, bp, S)
    for ptype, bp, val in ramp_prims:
        r_detected += val * gate_ramp(tx, bp, S)
    pred_detected = tx + r_detected
    det_exact = int(np.sum(np.round(pred_detected) == ty))
    ax.plot(tx, pred_detected, 'r-', linewidth=1, alpha=0.7,
            label=f'detected: {det_exact}/{R}')
    
    # Mark errors
    detect_errors = np.where(np.round(pred_detected) != ty)[0]
    if len(detect_errors) > 0 and len(detect_errors) <= 20:
        ax.scatter(detect_errors, ty[detect_errors], c='red', s=30, zorder=5, marker='x')
    
    # Mark training points
    ax.scatter(trx, try_y, c='blue', s=8, alpha=0.3, zorder=4)
    
    # Mark true breakpoints
    for ptype, bp, amp in info['primitives']:
        ax.axvline(bp, color='orange', linestyle=':', alpha=0.4, linewidth=0.8)
    
    ax.set_title(f'{name} (oracle={oracle_exact} det={det_exact})', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)

# Summary panels in row 3-4
ax_summary = fig.add_subplot(gs[2, :2])
ax_summary.axis('off')

summary_text = "DETECTION GAP ANALYSIS\n" + "=" * 40 + "\n\n"
total_oracle = 0
total_detected = 0
total_R = 0
for name, info in TRUE_DECOMPOSITIONS.items():
    R = info['range']
    o = oracle_results.get(name, {}).get('exact', 0)
    # Recompute detected for summary
    total_oracle += o
    total_R += R

summary_text += f"{'Function':<14s} {'Oracle':>7s} {'Detect':>7s} {'Gap':>5s} {'Fixable':>8s}\n"
summary_text += "-" * 45 + "\n"

# Need to recompute...
for name in fn_list:
    R = TRUE_DECOMPOSITIONS[name]['range']
    oracle_ex = oracle_results.get(name, {}).get('exact', 0)
    summary_text += f"{name:<14s} {oracle_ex:>4d}/{R:<3d}\n"

ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Gate precision panel
ax_gate = fig.add_subplot(gs[2, 2:])
xs_fine = np.linspace(-3, 3, 1000)
for s_val, color, label in [(PHI**2, 'blue', f's=φ²'),
                              (5.0, 'green', 's=5'),
                              (10.0, 'orange', 's=10'),
                              (20.0, 'red', 's=20')]:
    step_vals = np.array([float(gate_step(np.array([x]), 0.5, s_val)) for x in xs_fine])
    ax_gate.plot(xs_fine, step_vals, color=color, linewidth=1.5, label=label)
ax_gate.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax_gate.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
for x_int in range(-2, 4):
    ax_gate.axvline(x_int, color='lightgray', linestyle='-', alpha=0.3)
ax_gate.set_xlabel('x')
ax_gate.set_ylabel('gate_step(x, 0.5, s)')
ax_gate.set_title('Gate Step Precision at Half-Integer Breakpoint', fontweight='bold')
ax_gate.legend(fontsize=9)
ax_gate.grid(True, alpha=0.3)

# Ramp precision
ax_ramp = fig.add_subplot(gs[3, :2])
for s_val, color, label in [(PHI**2, 'blue', f's=φ²'),
                              (5.0, 'green', 's=5'),
                              (10.0, 'orange', 's=10'),
                              (20.0, 'red', 's=20')]:
    ramp_vals = np.array([float(gate_ramp(np.array([x]), 0.5, s_val)) for x in xs_fine])
    ideal_ramp = np.maximum(0, xs_fine - 0.5)
    ax_ramp.plot(xs_fine, ramp_vals, color=color, linewidth=1.5, label=label)
ax_ramp.plot(xs_fine, np.maximum(0, xs_fine - 0.5), 'k--', linewidth=1, alpha=0.5, label='ideal')
ax_ramp.set_xlabel('x')
ax_ramp.set_ylabel('gate_ramp(x, 0.5, s)')
ax_ramp.set_title('Gate Ramp Precision at Half-Integer Breakpoint', fontweight='bold')
ax_ramp.legend(fontsize=9)
ax_ramp.grid(True, alpha=0.3)

# Insight
ax_ins = fig.add_subplot(gs[3, 2:])
ax_ins.axis('off')
insight = (
    "PATH TO 100%\n"
    "═══════════════════════════\n\n"
    "Q1: Can the gate do 100%?\n"
    "    → Test with oracle breakpoints\n\n"
    "Q2: Where does detection fail?\n"
    "    → Mislocalization vs\n"
    "      misclassification vs\n"
    "      missed breakpoints\n\n"
    "Q3: What structural constraints\n"
    "    can we exploit?\n"
    "    → Paired steps (RECTs)\n"
    "    → Consistent amplitudes\n"
    "    → Half-integer alignment\n\n"
    "Q4: What sharpness is needed?\n"
    "    → Min s for 100% per function"
)
ax_ins.text(0.05, 0.95, insight, transform=ax_ins.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/tmp/detection_deep_dive.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: /tmp/detection_deep_dive.png")
