"""
Test: Prove the φ-Holographic Map works as a data structure.

Specifically demonstrate the UNIQUE properties:
  1. The mean Jacobian (default) is BETTER than nonlinear lookup
  2. Compression IMPROVES quality (denoising)
  3. Gate codes are locality-preserving
  4. φ-structured initialization outperforms random

This validates the generalization from DDColor's gate field
to a standalone CS data structure.
"""
import numpy as np
import sys
import time

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PhiMapStack, PHI

np.random.seed(42)


# ================================================================
# Test 1: Basic functionality — learn a nonlinear mapping
# ================================================================
print('=' * 70)
print('TEST 1: Learn a Nonlinear Mapping')
print('=' * 70)
print()

DIM = 32
N_TRAIN = 500
N_TEST = 200
N_CAL = 100

# Create a nonlinear target function
W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    """Nonlinear target: tanh of linear transform + quadratic term."""
    return np.tanh(x @ W_true.T) + 0.1 * x**2

# Generate data
X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(N_CAL, DIM).astype(np.float32)

# Train φ-Map
phi_map = PhiMap(DIM, expansion=4, gate='gelu')
phi_map.init_random(seed=42)

print("Training φ-Map...")
t0 = time.time()
final_loss = phi_map.fit(X_train, Y_train, n_iter=2000, lr=0.005, verbose=True)
t1 = time.time()
print(f"  Training time: {t1-t0:.2f}s")
print(f"  Final train loss: {final_loss:.6f}")

# Test nonlinear lookup
Y_pred_nonlinear = phi_map.lookup(X_test)
rmse_nonlinear = np.sqrt(np.mean((Y_pred_nonlinear - Y_test)**2))
print(f"\n  Nonlinear lookup RMSE: {rmse_nonlinear:.4f}")

# Calibrate and test default (mean Jacobian)
phi_map.calibrate(X_cal)
Y_pred_default = phi_map.default(X_test)
rmse_default = np.sqrt(np.mean((Y_pred_default - Y_test)**2))
print(f"  Default (mean Jac) RMSE: {rmse_default:.4f}")
improvement = (rmse_nonlinear - rmse_default) / rmse_nonlinear * 100
print(f"  Improvement: {improvement:+.2f}%")
print(f"  → {'DEFAULT IS BETTER (denoising!)' if rmse_default < rmse_nonlinear else 'Nonlinear is better'}")


# ================================================================
# Test 2: Compression curve — sweet spot exists
# ================================================================
print()
print('=' * 70)
print('TEST 2: Compression Curve — Does a Sweet Spot Exist?')
print('=' * 70)
print()

print(f"  {'Rank %':<10} {'Rank':<8} {'RMSE':<12} {'vs Full Jac':<12} {'vs Nonlinear':<12}")
print(f"  " + "-" * 54)

# Re-calibrate fresh each time
results = []
for rank_pct in [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]:
    phi_map.calibrate(X_cal)  # Fresh calibration
    if rank_pct < 1.0:
        phi_map.compress(rank_pct)
    Y_pred = phi_map.default(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test)**2))
    vs_full = (rmse - rmse_default) / rmse_default * 100
    vs_nonlin = (rmse - rmse_nonlinear) / rmse_nonlinear * 100
    results.append((rank_pct, rmse))
    rank_k = max(1, int(DIM * rank_pct))
    print(f"  {rank_pct*100:>5.0f}%     {rank_k:<8} {rmse:<12.4f} {vs_full:+11.2f}% {vs_nonlin:+11.2f}%")

# Find best
best_pct, best_rmse = min(results, key=lambda x: x[1])
print(f"\n  Best compression: rank {best_pct*100:.0f}% with RMSE {best_rmse:.4f}")
sweet_spot = best_rmse < rmse_default
print(f"  → {'SWEET SPOT EXISTS (compression denoises!)' if sweet_spot else 'No sweet spot'}")


# ================================================================
# Test 3: Locality preservation — similar inputs → similar codes
# ================================================================
print()
print('=' * 70)
print('TEST 3: Locality Preservation')
print('=' * 70)
print()

# Generate pairs at varying distances
n_pairs = 200
base_points = np.random.randn(n_pairs, DIM).astype(np.float32)

distances_input = []
distances_gate = []

for noise_scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    noise = np.random.randn(n_pairs, DIM).astype(np.float32) * noise_scale
    perturbed = base_points + noise

    for i in range(n_pairs):
        d_input = np.linalg.norm(base_points[i] - perturbed[i])
        d_gate = phi_map.similarity(base_points[i:i+1], perturbed[i:i+1])
        distances_input.append(d_input)
        distances_gate.append(d_gate)

# Compute correlation between input distance and gate distance
corr = np.corrcoef(distances_input, distances_gate)[0, 1]
print(f"  Correlation(input_distance, gate_distance): {corr:.4f}")
print(f"  → {'LOCALITY PRESERVED' if corr > 0.5 else 'Weak locality'}")

# Show by noise scale
print(f"\n  {'Input dist':<12} {'Gate dist':<12}")
print(f"  " + "-" * 24)
for noise_scale in [0.01, 0.1, 1.0, 5.0]:
    noise = np.random.randn(50, DIM).astype(np.float32) * noise_scale
    perturbed = base_points[:50] + noise
    gate_dists = []
    input_dists = []
    for i in range(50):
        input_dists.append(np.linalg.norm(base_points[i] - perturbed[i]))
        gate_dists.append(phi_map.similarity(base_points[i:i+1], perturbed[i:i+1]))
    print(f"  {np.mean(input_dists):<12.3f} {np.mean(gate_dists):<12.4f}")


# ================================================================
# Test 4: Gate field statistics
# ================================================================
print()
print('=' * 70)
print('TEST 4: Gate Field Statistics')
print('=' * 70)
print()

stats = phi_map.gate_statistics(X_test)
print(f"  Alive rate (mean):    {stats['alive_rate_mean']:.3f}")
print(f"  Alive rate (std):     {stats['alive_rate_std']:.3f}")
print(f"  Dead channels (<5%):  {stats['dead_channels']}")
print(f"  Code uniqueness:      {stats['code_uniqueness']:.3f}")
print(f"  Effective bits:       {stats['effective_bits']:.1f} / {phi_map.E}")

# Jacobian spectrum
spectrum = phi_map.jacobian_spectrum()
if spectrum is not None:
    cumvar = np.cumsum(spectrum**2) / np.sum(spectrum**2)
    rank50 = np.searchsorted(cumvar, 0.50) + 1
    rank90 = np.searchsorted(cumvar, 0.90) + 1
    print(f"\n  Jacobian spectrum:")
    print(f"    50% variance in {rank50}/{DIM} dims")
    print(f"    90% variance in {rank90}/{DIM} dims")
    print(f"    Condition number: {spectrum[0]/spectrum[-1]:.1f}")

# Parameter counts
params = phi_map.param_count
print(f"\n  Parameters:")
print(f"    Full φ-Map:     {params['full']:,}")
print(f"    Mean Jacobian:  {params['jacobian']:,}")
if params['compression']:
    print(f"    Compression:    {params['compression']*100:.1f}%")


# ================================================================
# Test 5: φ-structured vs random initialization
# ================================================================
print()
print('=' * 70)
print('TEST 5: φ-Structured vs Random Initialization')
print('=' * 70)
print()

# Random init
phi_map_random = PhiMap(DIM, expansion=4, gate='gelu')
phi_map_random.init_random(seed=123)
loss_random = phi_map_random.fit(X_train, Y_train, n_iter=2000, lr=0.005)

# φ-structured init
phi_map_phi = PhiMap(DIM, expansion=4, gate='gelu')
phi_map_phi.init_phi_structured(seed=123)
loss_phi = phi_map_phi.fit(X_train, Y_train, n_iter=2000, lr=0.005)

# Test both
Y_random = phi_map_random.lookup(X_test)
Y_phi = phi_map_phi.lookup(X_test)
rmse_random = np.sqrt(np.mean((Y_random - Y_test)**2))
rmse_phi = np.sqrt(np.mean((Y_phi - Y_test)**2))

print(f"  Random init:       train loss = {loss_random:.6f}, test RMSE = {rmse_random:.4f}")
print(f"  φ-structured init: train loss = {loss_phi:.6f}, test RMSE = {rmse_phi:.4f}")
print(f"  Improvement: {(rmse_random - rmse_phi)/rmse_random*100:+.2f}%")
print(f"  → {'φ-INIT IS BETTER' if rmse_phi < rmse_random else 'Random is better'}")

# Test default (mean Jacobian) for both
phi_map_random.calibrate(X_cal)
phi_map_phi.calibrate(X_cal)

Y_rand_def = phi_map_random.default(X_test)
Y_phi_def = phi_map_phi.default(X_test)
rmse_rand_def = np.sqrt(np.mean((Y_rand_def - Y_test)**2))
rmse_phi_def = np.sqrt(np.mean((Y_phi_def - Y_test)**2))

print(f"\n  Random default RMSE:  {rmse_rand_def:.4f} (vs nonlinear: "
      f"{(rmse_rand_def-rmse_random)/rmse_random*100:+.2f}%)")
print(f"  φ-struct default RMSE: {rmse_phi_def:.4f} (vs nonlinear: "
      f"{(rmse_phi_def-rmse_phi)/rmse_phi*100:+.2f}%)")


# ================================================================
# Test 6: Stacked φ-Map
# ================================================================
print()
print('=' * 70)
print('TEST 6: Stacked φ-Map (Multi-Resolution)')
print('=' * 70)
print()

# Create a deeper target with hierarchical structure
DIM_STACK = 16
N_STACK = 300

X_stack = np.random.randn(N_STACK, DIM_STACK).astype(np.float32)

# Target: composition of nonlinear transforms (what ConvNeXt does)
Y_stack = X_stack.copy()
for _ in range(3):
    W = np.random.randn(DIM_STACK, DIM_STACK).astype(np.float32) * 0.3
    Y_stack = np.tanh(Y_stack @ W.T) + Y_stack  # Residual!

# Single φ-Map
single = PhiMap(DIM_STACK, expansion=4, gate='gelu')
single.init_random(seed=42)
single.fit(X_stack[:200], Y_stack[:200], n_iter=2000, lr=0.005)
single.calibrate(X_stack[:100])

Y_single = single.lookup(X_stack[200:])
Y_single_def = single.default(X_stack[200:])
rmse_single = np.sqrt(np.mean((Y_single - Y_stack[200:])**2))
rmse_single_def = np.sqrt(np.mean((Y_single_def - Y_stack[200:])**2))

# 3-deep stack of φ-Maps (same total computation)
stack_maps = []
X_residual = X_stack[:200].copy()
Y_target = Y_stack[:200].copy()
for i in range(3):
    pm = PhiMap(DIM_STACK, expansion=4, gate='gelu')
    pm.init_random(seed=42 + i)
    # Each map learns the residual
    pm.fit(X_residual, Y_target - X_residual, n_iter=2000, lr=0.005)
    pm.calibrate(X_stack[:100])
    # Update residual
    X_residual = X_residual + pm.lookup(X_residual)
    stack_maps.append(pm)

# Test stacked
X_test_stack = X_stack[200:].copy()
for pm in stack_maps:
    X_test_stack = X_test_stack + pm.lookup(X_test_stack)
rmse_stack = np.sqrt(np.mean((X_test_stack - Y_stack[200:])**2))

# Default (Jacobian) stacked
X_test_def = X_stack[200:].copy()
for pm in stack_maps:
    X_test_def = X_test_def + pm.default(X_test_def)
rmse_stack_def = np.sqrt(np.mean((X_test_def - Y_stack[200:])**2))

print(f"  Single φ-Map nonlinear: {rmse_single:.4f}")
print(f"  Single φ-Map default:   {rmse_single_def:.4f}")
print(f"  Stacked (3 deep) nonlinear: {rmse_stack:.4f}")
print(f"  Stacked (3 deep) default:   {rmse_stack_def:.4f}")
print(f"\n  Stack vs single: {(rmse_single-rmse_stack)/rmse_single*100:+.1f}% (nonlinear)")
print(f"  Default vs nonlinear (stack): {(rmse_stack-rmse_stack_def)/rmse_stack*100:+.1f}%")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY: φ-Holographic Map as Data Structure')
print('=' * 70)
print()
print("Property 1 — Denoising Mean:")
print(f"  Default RMSE: {rmse_default:.4f} vs Nonlinear: {rmse_nonlinear:.4f}")
print(f"  → {'CONFIRMED' if rmse_default <= rmse_nonlinear else 'NOT confirmed'}")
print()
print("Property 2 — Compression Sweet Spot:")
print(f"  Best at rank {best_pct*100:.0f}%: RMSE {best_rmse:.4f}")
print(f"  → {'CONFIRMED' if sweet_spot else 'NOT confirmed'}")
print()
print(f"Property 3 — Locality Preservation:")
print(f"  Distance correlation: {corr:.4f}")
print(f"  → {'CONFIRMED' if corr > 0.5 else 'NOT confirmed'}")
print()
print(f"Property 4 — φ-Structured Init:")
print(f"  φ RMSE: {rmse_phi:.4f} vs Random: {rmse_random:.4f}")
print(f"  → {'CONFIRMED' if rmse_phi < rmse_random else 'NOT confirmed (marginal)'}")
print()
print(f"Property 5 — Stacking Helps:")
print(f"  Stack: {rmse_stack:.4f} vs Single: {rmse_single:.4f}")
print(f"  → {'CONFIRMED' if rmse_stack < rmse_single else 'NOT confirmed'}")
