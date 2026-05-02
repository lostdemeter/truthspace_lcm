#!/usr/bin/env python3
"""
Pinned Threshold Learning: A Better Way to Learn Geometric Programs

The insight: gradient descent searches blindly for WHERE to put gate
transitions. But we know the structure — every operation is built from
STEP, RECT, and RAMP primitives. The transitions happen at specific
thresholds. If we can DETECT those thresholds from data, we can PIN
them and solve for output weights analytically.

Three learning strategies compared:

1. GRADIENT DESCENT (baseline)
   - Random init → optimize all params with Adam
   - What we did in Part 29: slow, 74/128 best

2. PINNED THRESHOLDS
   - Detect breakpoints in input→output mapping
   - Pin gate transitions at detected breakpoints
   - Solve for output weights via least squares
   - No gradient descent at all

3. RESIDUAL PEELING
   - Start with identity (skip connection)
   - Compute residual: r = y - x
   - Find dominant pattern (step/rect/ramp)
   - Pin a gate, subtract explained part
   - Repeat until residual is small
   - Like boosting, but geometric

4. HYBRID: Pin + Polish
   - Pin thresholds from data
   - Solve W2 analytically
   - Fine-tune with a few gradient steps

Compare: accuracy, speed, and number of training examples needed.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PHI = (1 + np.sqrt(5)) / 2
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)
C_GEOMETRIC = (4 - np.pi) / (6 * np.pi)


def ideal_gate_np(x):
    """Ideal Gate in numpy."""
    x = np.asarray(x, dtype=np.float64)
    f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
    f = np.clip(f, -500, 500)
    return x * (1.0 / (1.0 + np.exp(-f)))


def ideal_gate_torch(x):
    """Ideal Gate in PyTorch."""
    f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
    return x * torch.sigmoid(f)


# ============================================================================
# TEST FUNCTIONS — operations to learn from examples
# ============================================================================

def fn_tolower(x):
    """Uppercase to lowercase: +32 in range [65, 90]."""
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    mask = (x >= 65) & (x <= 90)
    result[mask] += 32
    return result

def fn_secret(x):
    """Piecewise linear: f(x) = 2x if x<50, else 100-x."""
    x = np.asarray(x, dtype=np.float64)
    result = np.where(x < 50, 2*x, 100 - x)
    return result

def fn_rot13(x):
    """ROT13 cipher."""
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    mask_am = (x >= 65) & (x <= 77)
    mask_nz = (x >= 78) & (x <= 90)
    mask_am2 = (x >= 97) & (x <= 109)
    mask_nz2 = (x >= 110) & (x <= 122)
    result[mask_am] += 13
    result[mask_nz] -= 13
    result[mask_am2] += 13
    result[mask_nz2] -= 13
    return result

def fn_abs_centered(x):
    """Absolute value centered at 64: |x - 64|."""
    return np.abs(np.asarray(x, dtype=np.float64) - 64)

def fn_sawtooth(x):
    """Sawtooth: x mod 32."""
    return np.asarray(x, dtype=np.float64) % 32


# ============================================================================
# STRATEGY 1: Gradient Descent (baseline from Part 29)
# ============================================================================

class LearnableGeoBlock(nn.Module):
    """Learnable geometric block with structured initialization."""

    def __init__(self, hidden_dim, input_max=128.0):
        super().__init__()
        s_init = PHI ** 2
        self.W1 = nn.Parameter(torch.full((hidden_dim, 1), s_init))
        thresholds = torch.linspace(0, input_max, hidden_dim)
        self.b1 = nn.Parameter(-s_init * thresholds)
        self.W2 = nn.Parameter(torch.randn(1, hidden_dim) * 2.0)

    def forward(self, x):
        h = ideal_gate_torch(x @ self.W1.T + self.b1)
        return x + h @ self.W2.T


def learn_gradient_descent(train_x, train_y, test_x, test_y,
                           hidden_dim=64, epochs=3000, lr=0.01):
    """Baseline: learn via gradient descent."""
    t0 = time.perf_counter()

    model = LearnableGeoBlock(hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(train_x.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(epochs):
        pred = model(x_t)
        loss = ((pred - y_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        x_test = torch.tensor(test_x.reshape(-1, 1), dtype=torch.float32)
        pred_test = model(x_test).numpy().flatten()

    exact = np.sum(np.round(pred_test) == test_y)
    max_err = np.abs(pred_test - test_y).max()
    return pred_test, exact, max_err, elapsed


# ============================================================================
# STRATEGY 2: Pinned Thresholds
# ============================================================================

def detect_breakpoints(x, y, min_slope_change=0.5):
    """Detect breakpoints in the input-output mapping.

    A breakpoint is where the slope changes significantly.
    Sort by x, compute finite differences, find jumps.
    """
    # Sort by x
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    # Compute residual from identity (skip connection)
    residual = ys - xs

    # Compute slope changes
    if len(xs) < 3:
        return np.array([])

    # Finite differences of residual
    dr = np.diff(residual)
    dx = np.diff(xs)
    # Avoid division by zero
    dx = np.maximum(dx, 1e-10)
    slopes = dr / dx

    # Find where slope changes significantly
    slope_changes = np.abs(np.diff(slopes))

    # Breakpoints are at positions where slope changes most
    breakpoints = []
    threshold = min_slope_change * np.median(slope_changes + 1e-10)
    threshold = max(threshold, min_slope_change * 0.1)

    for i in range(len(slope_changes)):
        if slope_changes[i] > threshold:
            # Breakpoint is between xs[i+1] and xs[i+2]
            bp = (xs[i+1] + xs[i+2]) / 2 if i+2 < len(xs) else xs[i+1]
            breakpoints.append(bp)

    # Also detect jumps in the residual itself
    for i in range(len(dr)):
        if abs(dr[i]) > 2.0:  # significant jump
            bp = (xs[i] + xs[i+1]) / 2
            if bp not in breakpoints:
                breakpoints.append(bp)

    return np.unique(np.array(breakpoints))


def learn_pinned_thresholds(train_x, train_y, test_x, test_y,
                            sharpness=None, n_extra=0):
    """Learn by detecting breakpoints and solving analytically.

    Step 1: Detect breakpoints in training data
    Step 2: Place gate thresholds at breakpoints (±0.5 for each step)
    Step 3: Build gate activation matrix for training data
    Step 4: Solve for output weights via least squares
    """
    t0 = time.perf_counter()

    s = sharpness or PHI ** 2

    # Step 1: Detect breakpoints
    breakpoints = detect_breakpoints(train_x, train_y)

    # Add extra uniformly-spaced thresholds for coverage
    if n_extra > 0:
        extra = np.linspace(train_x.min(), train_x.max(), n_extra + 2)[1:-1]
        breakpoints = np.unique(np.concatenate([breakpoints, extra]))

    if len(breakpoints) == 0:
        # No breakpoints found — pure linear relationship
        # Solve y = a*x + b
        A = np.column_stack([train_x, np.ones_like(train_x)])
        coeffs, _, _, _ = np.linalg.lstsq(A, train_y, rcond=None)
        pred_test = coeffs[0] * test_x + coeffs[1]
        elapsed = time.perf_counter() - t0
        exact = np.sum(np.round(pred_test) == test_y)
        max_err = np.abs(pred_test - test_y).max()
        return pred_test, exact, max_err, elapsed, breakpoints

    # Step 2: Place gate neurons at each breakpoint
    # For each breakpoint bp, create a step pair at bp-0.5 and bp+0.5
    thresholds = []
    for bp in breakpoints:
        thresholds.extend([bp - 0.5, bp + 0.5])
    thresholds = np.array(thresholds)

    n_neurons = len(thresholds)

    # Step 3: Build gate activation matrix
    # Each neuron: gate(s * (x - threshold))
    # Shape: (n_train, n_neurons)
    H_train = np.zeros((len(train_x), n_neurons))
    for j, th in enumerate(thresholds):
        H_train[:, j] = ideal_gate_np(s * (train_x - th))

    # Step 4: Solve for output weights via ridge regression
    # We want: residual = y - x = H @ w2
    # Ridge: minimize ||H @ w2 - residual||² + α||w2||²
    # Solution: w2 = (H^T H + αI)^{-1} H^T residual
    residual = train_y - train_x
    alpha = 0.01 * len(train_x)  # regularization proportional to data size
    HtH = H_train.T @ H_train + alpha * np.eye(n_neurons)
    Htr = H_train.T @ residual
    w2 = np.linalg.solve(HtH, Htr)

    # Predict on test set
    H_test = np.zeros((len(test_x), n_neurons))
    for j, th in enumerate(thresholds):
        H_test[:, j] = ideal_gate_np(s * (test_x - th))

    pred_test = test_x + H_test @ w2

    elapsed = time.perf_counter() - t0
    exact = np.sum(np.round(pred_test) == test_y)
    max_err = np.abs(pred_test - test_y).max()
    return pred_test, exact, max_err, elapsed, breakpoints


# ============================================================================
# STRATEGY 3: Residual Peeling
# ============================================================================

def learn_residual_peeling(train_x, train_y, test_x, test_y,
                           max_layers=8, sharpness=None):
    """Learn by peeling off one geometric primitive at a time.

    Like boosting: fit the largest pattern first, subtract it,
    repeat on the residual.

    Each layer: find the best single step/rect/ramp to explain
    the current residual, pin it, subtract.
    """
    t0 = time.perf_counter()

    s = sharpness or PHI ** 2
    residual_train = train_y - train_x  # what the skip connection doesn't explain

    # Collect all layers for prediction
    layers = []  # list of (type, params)

    for layer_idx in range(max_layers):
        if np.abs(residual_train).max() < 0.5:
            break  # residual is small enough

        # Try to fit a STEP at every possible threshold
        best_score = 0
        best_params = None

        # Candidate thresholds: midpoints between sorted training x values
        sorted_x = np.sort(np.unique(train_x))
        candidates = (sorted_x[:-1] + sorted_x[1:]) / 2

        for th in candidates:
            # Step at threshold th: step(x; th) ≈ 1 when x > th
            step_vals = np.zeros(len(train_x))
            for i, xv in enumerate(train_x):
                lo = ideal_gate_np(s * (xv - (th - 0.5)))
                hi = ideal_gate_np(s * (xv - (th + 0.5)))
                step_vals[i] = (lo - hi) / s

            # Optimal amplitude for this step
            if np.dot(step_vals, step_vals) > 1e-10:
                amplitude = np.dot(step_vals, residual_train) / np.dot(step_vals, step_vals)
            else:
                continue

            # Score: how much residual does this explain?
            explained = amplitude * step_vals
            remaining = residual_train - explained
            score = np.sum(residual_train**2) - np.sum(remaining**2)

            if score > best_score:
                best_score = score
                best_params = ('step', th, amplitude)

        if best_params is None:
            break

        # Apply the best primitive
        typ, th, amp = best_params
        step_train = np.zeros(len(train_x))
        for i, xv in enumerate(train_x):
            lo = ideal_gate_np(s * (xv - (th - 0.5)))
            hi = ideal_gate_np(s * (xv - (th + 0.5)))
            step_train[i] = (lo - hi) / s

        residual_train -= amp * step_train
        layers.append(best_params)

    # Predict on test set
    pred_test = test_x.copy().astype(np.float64)
    for typ, th, amp in layers:
        step_test = np.zeros(len(test_x))
        for i, xv in enumerate(test_x):
            lo = ideal_gate_np(s * (xv - (th - 0.5)))
            hi = ideal_gate_np(s * (xv - (th + 0.5)))
            step_test[i] = (lo - hi) / s
        pred_test += amp * step_test

    elapsed = time.perf_counter() - t0
    exact = np.sum(np.round(pred_test) == test_y)
    max_err = np.abs(pred_test - test_y).max()
    return pred_test, exact, max_err, elapsed, layers


# ============================================================================
# STRATEGY 4: Hybrid — Pin + Polish
# ============================================================================

def learn_hybrid(train_x, train_y, test_x, test_y,
                 sharpness=None, polish_epochs=200, lr=0.01):
    """Pin thresholds from data, then fine-tune with gradient descent.

    Step 1: Detect breakpoints and pin thresholds
    Step 2: Solve W2 analytically (least squares)
    Step 3: Initialize PyTorch model with pinned values
    Step 4: Fine-tune all parameters with a few gradient steps
    """
    t0 = time.perf_counter()

    s = sharpness or PHI ** 2

    # Step 1-2: Use pinned threshold detection
    breakpoints = detect_breakpoints(train_x, train_y)

    # Build threshold list
    thresholds = []
    for bp in breakpoints:
        thresholds.extend([bp - 0.5, bp + 0.5])

    # Add some extra neurons for fine-tuning flexibility
    n_extra = max(8, len(thresholds))
    extra_th = np.linspace(train_x.min(), train_x.max(), n_extra + 2)[1:-1]
    all_thresholds = np.unique(np.concatenate([thresholds, extra_th]))
    n_neurons = len(all_thresholds)

    # Solve for W2 analytically
    H_train = np.zeros((len(train_x), n_neurons))
    for j, th in enumerate(all_thresholds):
        H_train[:, j] = ideal_gate_np(s * (train_x - th))

    residual = train_y - train_x
    w2, _, _, _ = np.linalg.lstsq(H_train, residual, rcond=None)

    # Step 3: Initialize PyTorch model with solved values
    model = LearnableGeoBlock(n_neurons)
    with torch.no_grad():
        model.W1.data = torch.full((n_neurons, 1), s, dtype=torch.float32)
        model.b1.data = torch.tensor(-s * all_thresholds, dtype=torch.float32)
        model.W2.data = torch.tensor(w2.reshape(1, -1), dtype=torch.float32)

    # Step 4: Fine-tune
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(train_x.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(polish_epochs):
        pred = model(x_t)
        loss = ((pred - y_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        x_test = torch.tensor(test_x.reshape(-1, 1), dtype=torch.float32)
        pred_test = model(x_test).numpy().flatten()

    exact = np.sum(np.round(pred_test) == test_y)
    max_err = np.abs(pred_test - test_y).max()
    return pred_test, exact, max_err, elapsed


# ============================================================================
# RUN ALL COMPARISONS
# ============================================================================

print("=" * 70)
print("PINNED THRESHOLD LEARNING: Structure-Aware Optimization")
print("=" * 70)
print()
print("The insight: gradient descent searches blindly for WHERE to put")
print("gate transitions. We know the structure. We can DETECT thresholds")
print("from data and solve for output weights analytically.")
print()

test_functions = [
    ("tolower", fn_tolower, 128, 50),
    ("secret_fn", fn_secret, 100, 40),
    ("ROT13", fn_rot13, 128, 60),
    ("abs_centered", fn_abs_centered, 128, 40),
    ("sawtooth_32", fn_sawtooth, 128, 50),
]

all_results = {}

for fn_name, fn, input_range, n_train in test_functions:
    print(f"\n{'='*60}")
    print(f"  LEARNING: {fn_name}")
    print(f"  Training on {n_train} examples, testing on all {input_range}")
    print(f"{'='*60}")

    # Generate test data (all integers in range)
    test_x = np.arange(input_range, dtype=np.float64)
    test_y = fn(test_x)

    # Generate training data (random subset)
    np.random.seed(42)
    train_idx = np.sort(np.random.choice(input_range, n_train, replace=False))
    train_x = train_idx.astype(np.float64)
    train_y = fn(train_x)

    results = {}

    # Strategy 1: Gradient Descent (best of 3 seeds)
    print(f"\n  Strategy 1: Gradient Descent (3000 epochs, best of 3 seeds)")
    best_gd = None
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        pred, exact, maxe, elapsed = learn_gradient_descent(
            train_x, train_y, test_x, test_y, hidden_dim=64, epochs=3000)
        print(f"    Seed {seed}: {exact}/{input_range} exact, max_err={maxe:.2f}, time={elapsed:.3f}s")
        if best_gd is None or exact > best_gd[1]:
            best_gd = (pred, exact, maxe, elapsed)
    results['grad_descent'] = best_gd
    print(f"    BEST: {best_gd[1]}/{input_range} exact, max_err={best_gd[2]:.2f}, time={best_gd[3]:.3f}s")

    # Strategy 2: Pinned Thresholds
    print(f"\n  Strategy 2: Pinned Thresholds (detect breakpoints, solve analytically)")
    pred, exact, maxe, elapsed, bps = learn_pinned_thresholds(
        train_x, train_y, test_x, test_y)
    results['pinned'] = (pred, exact, maxe, elapsed)
    print(f"    Detected {len(bps)} breakpoints: {bps[:10].round(1).tolist()}"
          f"{'...' if len(bps) > 10 else ''}")
    print(f"    Result: {exact}/{input_range} exact, max_err={maxe:.4f}, time={elapsed:.4f}s")

    # Strategy 2b: Pinned + extra neurons
    pred2, exact2, maxe2, elapsed2, bps2 = learn_pinned_thresholds(
        train_x, train_y, test_x, test_y, n_extra=16)
    results['pinned_extra'] = (pred2, exact2, maxe2, elapsed2)
    print(f"    With 16 extra neurons: {exact2}/{input_range} exact, max_err={maxe2:.4f}")

    # Strategy 3: Residual Peeling
    print(f"\n  Strategy 3: Residual Peeling (greedy primitive fitting)")
    pred3, exact3, maxe3, elapsed3, layers3 = learn_residual_peeling(
        train_x, train_y, test_x, test_y, max_layers=16)
    results['peeling'] = (pred3, exact3, maxe3, elapsed3)
    print(f"    Used {len(layers3)} layers")
    for typ, th, amp in layers3[:5]:
        print(f"      {typ} at {th:.1f}, amplitude={amp:.2f}")
    if len(layers3) > 5:
        print(f"      ... and {len(layers3)-5} more")
    print(f"    Result: {exact3}/{input_range} exact, max_err={maxe3:.4f}, time={elapsed3:.3f}s")

    # Strategy 4: Hybrid (Pin + Polish)
    print(f"\n  Strategy 4: Hybrid (pin thresholds + 200 gradient steps)")
    pred4, exact4, maxe4, elapsed4 = learn_hybrid(
        train_x, train_y, test_x, test_y, polish_epochs=200)
    results['hybrid'] = (pred4, exact4, maxe4, elapsed4)
    print(f"    Result: {exact4}/{input_range} exact, max_err={maxe4:.4f}, time={elapsed4:.3f}s")

    all_results[fn_name] = results

    # Summary for this function
    print(f"\n  SUMMARY for {fn_name}:")
    print(f"  {'Strategy':<25s} {'Exact':>8s} {'Max Err':>10s} {'Time':>10s} {'Speedup':>10s}")
    print(f"  {'-'*65}")

    gd_time = best_gd[3]
    for name, key in [("Gradient Descent", 'grad_descent'),
                      ("Pinned Thresholds", 'pinned'),
                      ("Pinned + Extra", 'pinned_extra'),
                      ("Residual Peeling", 'peeling'),
                      ("Hybrid (Pin+Polish)", 'hybrid')]:
        r = results[key]
        speedup = gd_time / max(r[3], 1e-6)
        print(f"  {name:<25s} {r[1]:>4d}/{input_range:<3d} {r[2]:>10.4f} {r[3]:>9.3f}s {speedup:>9.0f}x")


# ============================================================================
# Sample efficiency: how many examples do we need?
# ============================================================================

print()
print()
print("=" * 70)
print("SAMPLE EFFICIENCY: How many examples does each strategy need?")
print("=" * 70)

sample_counts = [5, 10, 20, 40, 80, 128]
sample_results = {n: {} for n in sample_counts}

test_x_se = np.arange(128, dtype=np.float64)
test_y_se = fn_tolower(test_x_se)

for n_samples in sample_counts:
    np.random.seed(42)
    if n_samples >= 128:
        train_idx_se = np.arange(128)
    else:
        train_idx_se = np.sort(np.random.choice(128, n_samples, replace=False))
    train_x_se = train_idx_se.astype(np.float64)
    train_y_se = fn_tolower(train_x_se)

    # Gradient descent (single seed for speed)
    torch.manual_seed(0)
    _, exact_gd, _, elapsed_gd = learn_gradient_descent(
        train_x_se, train_y_se, test_x_se, test_y_se,
        hidden_dim=64, epochs=3000)

    # Pinned
    _, exact_pin, _, elapsed_pin, _ = learn_pinned_thresholds(
        train_x_se, train_y_se, test_x_se, test_y_se)

    # Hybrid
    _, exact_hyb, _, elapsed_hyb = learn_hybrid(
        train_x_se, train_y_se, test_x_se, test_y_se,
        polish_epochs=200)

    sample_results[n_samples] = {
        'gd': (exact_gd, elapsed_gd),
        'pinned': (exact_pin, elapsed_pin),
        'hybrid': (exact_hyb, elapsed_hyb),
    }

    print(f"  {n_samples:3d} examples: "
          f"GD={exact_gd:3d}/128 ({elapsed_gd:.2f}s)  "
          f"Pinned={exact_pin:3d}/128 ({elapsed_pin:.4f}s)  "
          f"Hybrid={exact_hyb:3d}/128 ({elapsed_hyb:.2f}s)")


# ============================================================================
# Visualization
# ============================================================================

fig = plt.figure(figsize=(22, 18))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1-5: Each test function comparison ---
panel_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
fn_names = ["tolower", "secret_fn", "ROT13", "abs_centered", "sawtooth_32"]
fn_refs = [fn_tolower, fn_secret, fn_rot13, fn_abs_centered, fn_sawtooth]
input_ranges = [128, 100, 128, 128, 128]

for idx, (fn_name, fn_ref, inp_range) in enumerate(zip(fn_names, fn_refs, input_ranges)):
    row, col = panel_positions[idx]
    ax = fig.add_subplot(gs[row, col])

    test_x_p = np.arange(inp_range, dtype=np.float64)
    test_y_p = fn_ref(test_x_p)
    results = all_results[fn_name]

    ax.plot(test_x_p, test_y_p, 'k-', linewidth=1.5, alpha=0.3, label='True')
    ax.plot(test_x_p, results['grad_descent'][0], 'r-', linewidth=1, alpha=0.7,
            label=f"GD: {results['grad_descent'][1]}/{inp_range}")
    ax.plot(test_x_p, results['pinned'][0], 'b-', linewidth=1.5, alpha=0.8,
            label=f"Pinned: {results['pinned'][1]}/{inp_range}")
    ax.plot(test_x_p, results['hybrid'][0], 'g--', linewidth=1, alpha=0.7,
            label=f"Hybrid: {results['hybrid'][1]}/{inp_range}")

    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(f'{fn_name}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- Panel 6: Sample efficiency ---
ax6 = fig.add_subplot(gs[1, 2])
ns = list(sample_results.keys())
gd_acc = [sample_results[n]['gd'][0] for n in ns]
pin_acc = [sample_results[n]['pinned'][0] for n in ns]
hyb_acc = [sample_results[n]['hybrid'][0] for n in ns]

ax6.plot(ns, gd_acc, 'r-o', linewidth=2, markersize=6, label='Gradient Descent')
ax6.plot(ns, pin_acc, 'b-s', linewidth=2, markersize=6, label='Pinned Thresholds')
ax6.plot(ns, hyb_acc, 'g-^', linewidth=2, markersize=6, label='Hybrid')
ax6.axhline(128, color='gray', linestyle=':', alpha=0.5, label='Perfect (128)')
ax6.set_xlabel('Number of training examples')
ax6.set_ylabel('Exact matches (out of 128)')
ax6.set_title('Sample Efficiency: tolower\n(accuracy vs training set size)', fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# --- Panel 7: Speed comparison ---
ax7 = fig.add_subplot(gs[2, 0])

# Aggregate timing across all functions
strategy_names = ['Gradient\nDescent', 'Pinned\nThresholds', 'Pinned\n+ Extra',
                  'Residual\nPeeling', 'Hybrid\n(Pin+Polish)']
strategy_keys = ['grad_descent', 'pinned', 'pinned_extra', 'peeling', 'hybrid']
avg_times = []
for key in strategy_keys:
    times = [all_results[fn][key][3] for fn in fn_names]
    avg_times.append(np.mean(times))

colors = ['#F44336', '#2196F3', '#03A9F4', '#FF9800', '#4CAF50']
bars = ax7.bar(strategy_names, avg_times, color=colors, alpha=0.8)
for bar, t in zip(bars, avg_times):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{t:.3f}s', ha='center', va='bottom', fontsize=9)
ax7.set_ylabel('Average time (seconds)')
ax7.set_title('Speed: Average across 5 functions', fontweight='bold')
ax7.set_yscale('log')
ax7.grid(True, alpha=0.3, axis='y')

# --- Panel 8: Accuracy comparison ---
ax8 = fig.add_subplot(gs[2, 1])

# Aggregate accuracy
avg_acc = {}
for key in strategy_keys:
    accs = []
    for fn_name_inner, fn_ref_inner, inp_range_inner in zip(fn_names, fn_refs, input_ranges):
        accs.append(all_results[fn_name_inner][key][1] / inp_range_inner * 100)
    avg_acc[key] = np.mean(accs)

bars2 = ax8.bar(strategy_names, [avg_acc[k] for k in strategy_keys],
                color=colors, alpha=0.8)
for bar, acc in zip(bars2, [avg_acc[k] for k in strategy_keys]):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{acc:.0f}%', ha='center', va='bottom', fontsize=10)
ax8.set_ylabel('Average accuracy (%)')
ax8.set_title('Accuracy: Average across 5 functions', fontweight='bold')
ax8.set_ylim(0, 110)
ax8.grid(True, alpha=0.3, axis='y')

# --- Panel 9: The insight ---
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
insight_text = (
    "WHY PINNED LEARNING WORKS\n"
    "══════════════════════════════\n\n"
    "Gradient descent searches for:\n"
    "  1. WHERE to put thresholds\n"
    "  2. HOW MUCH to correct\n"
    "  3. HOW SHARP transitions are\n\n"
    "Pinned learning SOLVES:\n"
    "  1. Detect breakpoints from data\n"
    "     (finite differences of residual)\n"
    "  2. Least squares for weights\n"
    "     (one matrix solve, no iteration)\n"
    "  3. Use φ² sharpness (known good)\n\n"
    "Result:\n"
    "  • Orders of magnitude faster\n"
    "  • Often more accurate\n"
    "  • Fewer training examples needed\n"
    "  • No hyperparameter tuning\n\n"
    "The key: STRUCTURE IS INFORMATION.\n"
    "Knowing the gate structure lets\n"
    "us skip the search entirely."
)
ax9.text(0.05, 0.95, insight_text, transform=ax9.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Pinned Threshold Learning: Structure-Aware Geometric Optimization',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig('/tmp/geometric_pinned_learning.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print()
print("Saved: /tmp/geometric_pinned_learning.png")
