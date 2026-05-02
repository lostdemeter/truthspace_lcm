"""
Test: The Geometric φ-Map (Doc 247)

Prove that the scaffold (1/2 R @ H) matches the mean Jacobian
WITHOUT calibration data. If so, the "denoising" explanation was
always a shadow of the geometric explanation.

Key predictions:
  1. scaffold(x) = (1/2) R@H@x ≈ mean_jacobian(x)  (no calibration)
  2. The gate field is TERNARY (expand/preserve/contract), not binary
  3. Information at each φ-level is separable
  4. Scaffold captures most of the transform (Doc 132: 99.99% linear)
  5. Content = deviation from scaffold = lives at higher/lower φ-levels
"""
import numpy as np
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI

np.random.seed(42)

LOG_PHI = np.log(PHI)  # ≈ 0.481

DIM = 32
N_TRAIN = 500
N_TEST = 200
N_CAL = 100

# Create nonlinear target
W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)
X_cal = np.random.randn(N_CAL, DIM).astype(np.float32)


# ================================================================
# Test 1: scaffold(x) = (1/2) R@H@x vs mean Jacobian
# ================================================================
print('=' * 70)
print('TEST 1: Scaffold vs Mean Jacobian — Does Geometry Match Statistics?')
print('=' * 70)
print()

# Train a φ-Map
phi_map = PhiMap(DIM, expansion=4, gate='gelu')
phi_map.init_random(seed=42)
phi_map.fit(X_train, Y_train, n_iter=2000, lr=0.005)

# Statistical: mean Jacobian (requires calibration data)
phi_map.calibrate(X_cal)
Y_stat = phi_map.default(X_test)
rmse_stat = np.sqrt(np.mean((Y_stat - Y_test)**2))

# Geometric: scaffold = (1/2) R @ H (NO calibration data)
scaffold_matrix = 0.5 * phi_map.R @ phi_map.H   # [D, D] — intrinsic, no data
scaffold_bias = phi_map.R @ (phi_map.b * 0.5) + phi_map.b_out  # bias contribution
Y_geom = X_test @ scaffold_matrix.T + scaffold_bias
rmse_geom = np.sqrt(np.mean((Y_geom - Y_test)**2))

# Nonlinear (full GELU)
Y_nonlin = phi_map.lookup(X_test)
rmse_nonlin = np.sqrt(np.mean((Y_nonlin - Y_test)**2))

# How close are scaffold and Jacobian?
Y_diff = Y_stat - Y_geom
max_diff = np.max(np.abs(Y_diff))
mean_diff = np.mean(np.abs(Y_diff))

print(f"  Nonlinear (full GELU):     RMSE = {rmse_nonlin:.4f}")
print(f"  Statistical (mean Jac):    RMSE = {rmse_stat:.4f}  (needs {N_CAL} calibration samples)")
print(f"  Geometric (scaffold):      RMSE = {rmse_geom:.4f}  (needs 0 calibration samples)")
print(f"")
print(f"  Scaffold vs Jacobian diff: max = {max_diff:.6f}, mean = {mean_diff:.6f}")
print(f"  → {'MATCH' if mean_diff < 0.01 else 'DIFFER'}")
print()

# WHY they might differ: the mean Jacobian uses E[GELU'(z)], not exactly 0.5
# Let's check what E[GELU'(z)] actually is
from phi_holographic_map import _standard_gelu_derivative
z_cal = X_cal @ phi_map.H.T + phi_map.b
mean_gate_deriv = _standard_gelu_derivative(z_cal).mean(axis=0)
print(f"  E[GELU'(z)] statistics:")
print(f"    Mean:   {mean_gate_deriv.mean():.4f}  (scaffold predicts 0.5)")
print(f"    Std:    {mean_gate_deriv.std():.4f}")
print(f"    Min:    {mean_gate_deriv.min():.4f}")
print(f"    Max:    {mean_gate_deriv.max():.4f}")
print(f"    % within 10% of 0.5: {np.mean(np.abs(mean_gate_deriv - 0.5) < 0.05) * 100:.1f}%")


# ================================================================
# Test 2: Ternary φ-region classification
# ================================================================
print()
print('=' * 70)
print('TEST 2: Ternary φ-Region Classification')
print('=' * 70)
print()

z_test = X_test @ phi_map.H.T + phi_map.b  # [N, E]

# Classify into three regions
log_phi = np.log(PHI)
# For GELU ≈ x·σ(φx), the effective boundary shifts
# But let's check both the theoretical and empirical boundaries

for boundary_name, boundary in [("log(φ)", log_phi), ("log(φ)/φ", log_phi/PHI), ("0.5", 0.5)]:
    expand = (z_test > boundary).mean() * 100
    preserve = ((z_test >= -boundary) & (z_test <= boundary)).mean() * 100
    contract = (z_test < -boundary).mean() * 100

    print(f"  Boundary = ±{boundary_name} (±{boundary:.3f}):")
    print(f"    EXPAND:   {expand:5.1f}%  (φ^+n, amplified)")
    print(f"    PRESERVE: {preserve:5.1f}%  (φ^0, linear)")
    print(f"    CONTRACT: {contract:5.1f}%  (φ^-n, suppressed)")
    print()


# ================================================================
# Test 3: φ-level separation — information at each level
# ================================================================
print()
print('=' * 70)
print('TEST 3: Information at Each φ-Level')
print('=' * 70)
print()

from phi_holographic_map import _standard_gelu

# Full GELU output
gelu_z = _standard_gelu(z_test)

# Scaffold contribution (linear part)
scaffold_z = z_test * 0.5

# Content (deviation from scaffold)
content_z = gelu_z - scaffold_z

# Measure energy at each component
energy_scaffold = np.mean(scaffold_z ** 2)
energy_content = np.mean(content_z ** 2)
energy_total = np.mean(gelu_z ** 2)

print(f"  Energy decomposition:")
print(f"    Total GELU:     {energy_total:.4f}")
print(f"    Scaffold (x/2): {energy_scaffold:.4f}  ({energy_scaffold/energy_total*100:.1f}%)")
print(f"    Content (Δ):    {energy_content:.4f}  ({energy_content/energy_total*100:.1f}%)")
print()

# Now decompose by φ-level
print(f"  Per-φ-level energy breakdown:")
for level_name, mask_fn in [
    ("EXPAND (z > log(φ))", lambda z: z > log_phi),
    ("PRESERVE (|z| ≤ log(φ))", lambda z: np.abs(z) <= log_phi),
    ("CONTRACT (z < -log(φ))", lambda z: z < -log_phi),
]:
    mask = mask_fn(z_test)
    if mask.sum() == 0:
        print(f"    {level_name}: no values")
        continue

    gelu_masked = np.where(mask, gelu_z, 0)
    scaffold_masked = np.where(mask, scaffold_z, 0)
    content_masked = np.where(mask, content_z, 0)

    e_total = np.sum(gelu_masked ** 2) / mask.sum()
    e_scaffold = np.sum(scaffold_masked ** 2) / mask.sum()
    e_content = np.sum(content_masked ** 2) / mask.sum()

    print(f"    {level_name}:")
    print(f"      scaffold={e_scaffold:.4f} ({e_scaffold/(e_total+1e-10)*100:.0f}%), "
          f"content={e_content:.4f} ({e_content/(e_total+1e-10)*100:.0f}%)")


# ================================================================
# Test 4: Reconstruct from scaffold + specific φ-levels
# ================================================================
print()
print('=' * 70)
print('TEST 4: Reconstruction by φ-Level')
print('=' * 70)
print()

# Scaffold only
Y_scaffold = (scaffold_z @ phi_map.R.T) + phi_map.b_out
rmse_scaffold = np.sqrt(np.mean((Y_scaffold - Y_test)**2))

# Scaffold + EXPAND
expand_mask = z_test > log_phi
gelu_expand = np.where(expand_mask, gelu_z, scaffold_z)
Y_expand = (gelu_expand @ phi_map.R.T) + phi_map.b_out
rmse_expand = np.sqrt(np.mean((Y_expand - Y_test)**2))

# Scaffold + CONTRACT
contract_mask = z_test < -log_phi
gelu_contract = np.where(contract_mask, gelu_z, scaffold_z)
Y_contract = (gelu_contract @ phi_map.R.T) + phi_map.b_out
rmse_contract = np.sqrt(np.mean((Y_contract - Y_test)**2))

# Scaffold + EXPAND + CONTRACT (= full GELU)
rmse_full = rmse_nonlin  # Already computed

# What about content ONLY (no scaffold)?
Y_content_only = (content_z @ phi_map.R.T) + phi_map.b_out
rmse_content = np.sqrt(np.mean((Y_content_only - Y_test)**2))

print(f"  {'Reconstruction':<35} {'RMSE':<10} {'vs Full':<12}")
print(f"  " + "-" * 57)
print(f"  {'Full GELU (all levels)':<35} {rmse_full:<10.4f} {'baseline':<12}")
print(f"  {'Scaffold only (φ^0)':<35} {rmse_scaffold:<10.4f} {(rmse_scaffold-rmse_full)/rmse_full*100:+.2f}%")
print(f"  {'Scaffold + EXPAND':<35} {rmse_expand:<10.4f} {(rmse_expand-rmse_full)/rmse_full*100:+.2f}%")
print(f"  {'Scaffold + CONTRACT':<35} {rmse_contract:<10.4f} {(rmse_contract-rmse_full)/rmse_full*100:+.2f}%")
print(f"  {'Content only (no scaffold)':<35} {rmse_content:<10.4f} {(rmse_content-rmse_full)/rmse_full*100:+.2f}%")
print(f"  {'Mean Jacobian (statistical)':<35} {rmse_stat:<10.4f} {(rmse_stat-rmse_full)/rmse_full*100:+.2f}%")


# ================================================================
# Test 5: Does scaffold match Jacobian across multiple trainings?
# ================================================================
print()
print('=' * 70)
print('TEST 5: Scaffold ≈ Jacobian Across Multiple Seeds')
print('=' * 70)
print()

print(f"  {'Seed':<6} {'Nonlinear':<12} {'Jacobian':<12} {'Scaffold':<12} {'|Jac-Scaf|':<12}")
print(f"  " + "-" * 54)

for seed in [42, 123, 456, 789, 1024]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)
    pm.fit(X_train, Y_train, n_iter=2000, lr=0.005)

    # Nonlinear
    Y_nl = pm.lookup(X_test)
    rmse_nl = np.sqrt(np.mean((Y_nl - Y_test)**2))

    # Jacobian (statistical)
    pm.calibrate(X_cal)
    Y_jac = pm.default(X_test)
    rmse_jac = np.sqrt(np.mean((Y_jac - Y_test)**2))

    # Scaffold (geometric)
    S = 0.5 * pm.R @ pm.H
    b_s = pm.R @ (pm.b * 0.5) + pm.b_out
    Y_scaf = X_test @ S.T + b_s
    rmse_scaf = np.sqrt(np.mean((Y_scaf - Y_test)**2))

    diff = np.mean(np.abs(Y_jac - Y_scaf))

    print(f"  {seed:<6} {rmse_nl:<12.4f} {rmse_jac:<12.4f} {rmse_scaf:<12.4f} {diff:<12.6f}")


# ================================================================
# Test 6: The φ-level hierarchy as navigation
# ================================================================
print()
print('=' * 70)
print('TEST 6: φ-Level Navigation')
print('=' * 70)
print()

# Pick a single test point and show its φ-level decomposition
x = X_test[0:1]
y_true = Y_test[0:1]

z = x @ phi_map.H.T + phi_map.b
gelu_out = _standard_gelu(z)

# Count channels at each φ-level
levels = np.floor(np.abs(z[0]) / log_phi).astype(int)
levels[z[0] > 0] = levels[z[0] > 0]     # positive levels
levels[z[0] < 0] = -levels[z[0] < 0]    # negative levels
levels[(np.abs(z[0]) <= log_phi)] = 0     # preserve region

unique_levels, counts = np.unique(levels, return_counts=True)
print(f"  φ-level distribution for one input point:")
print(f"  {'Level':<8} {'Count':<8} {'% of E':<10} {'Region':<12}")
print(f"  " + "-" * 38)
for lev, cnt in sorted(zip(unique_levels, counts), key=lambda x: x[0]):
    region = "PRESERVE" if lev == 0 else ("EXPAND" if lev > 0 else "CONTRACT")
    print(f"  {lev:<8} {cnt:<8} {cnt/phi_map.E*100:<10.1f} {region:<12}")

# Navigate: interpolate between two points through φ-levels
x1 = X_test[0:1]
x2 = X_test[1:2]
z1 = x1 @ phi_map.H.T + phi_map.b
z2 = x2 @ phi_map.H.T + phi_map.b

# Hamming distance of ternary codes
code1 = np.sign(z1[0]) * np.floor(np.abs(z1[0]) / log_phi).astype(int)
code2 = np.sign(z2[0]) * np.floor(np.abs(z2[0]) / log_phi).astype(int)
code1[np.abs(z1[0]) <= log_phi] = 0
code2[np.abs(z2[0]) <= log_phi] = 0

ternary_dist = np.mean(code1 != code2)
binary_dist = phi_map.similarity(x1, x2)

print(f"\n  Navigation between two points:")
print(f"    Binary gate distance:   {binary_dist:.4f}")
print(f"    Ternary φ-level distance: {ternary_dist:.4f}")
print(f"    Ratio: {ternary_dist/max(binary_dist, 1e-10):.2f}")
print(f"    → Ternary encodes {'MORE' if ternary_dist > binary_dist else 'LESS'} distinction")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY: Geometric vs Statistical φ-Map')
print('=' * 70)
print()
print(f"  1. Scaffold ≈ Mean Jacobian?")
print(f"     Mean absolute difference: {mean_diff:.6f}")
print(f"     → {'YES — geometry captures what statistics approximates' if mean_diff < 0.1 else 'PARTIAL — E[GELU`] ≠ 0.5 everywhere'}")
print()
print(f"  2. Scaffold dominates transform?")
print(f"     Scaffold energy: {energy_scaffold/energy_total*100:.1f}% of total")
print(f"     Content energy:  {energy_content/energy_total*100:.1f}% of total")
print(f"     → {'YES' if energy_scaffold > energy_content else 'NO'}")
print()
print(f"  3. Scaffold RMSE vs Jacobian RMSE?")
print(f"     Scaffold: {rmse_geom:.4f} (0 calibration samples)")
print(f"     Jacobian: {rmse_stat:.4f} ({N_CAL} calibration samples)")
print(f"     → {'SCAFFOLD MATCHES' if abs(rmse_geom - rmse_stat) / rmse_stat < 0.05 else 'DIFFER by ' + f'{abs(rmse_geom-rmse_stat)/rmse_stat*100:.1f}%'}")
print()
print(f"  4. Gate field is ternary?")
expand_pct = (z_test > log_phi).mean() * 100
preserve_pct = ((z_test >= -log_phi) & (z_test <= log_phi)).mean() * 100
contract_pct = (z_test < -log_phi).mean() * 100
print(f"     EXPAND: {expand_pct:.1f}%, PRESERVE: {preserve_pct:.1f}%, CONTRACT: {contract_pct:.1f}%")
print(f"     → {'TERNARY' if preserve_pct < 90 and expand_pct > 5 and contract_pct > 5 else 'EFFECTIVELY ' + ('LINEAR' if preserve_pct > 90 else 'BINARY')}")
print()
print(f"  5. No calibration needed?")
print(f"     Scaffold needs: 0 samples")
print(f"     Jacobian needs: {N_CAL} samples")
print(f"     RMSE difference: {abs(rmse_geom-rmse_stat)/rmse_stat*100:.2f}%")
print(f"     → The geometric scaffold captures the statistical Jacobian {'exactly' if mean_diff < 0.01 else 'approximately'}")
