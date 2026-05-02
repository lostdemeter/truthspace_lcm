"""
Convergence Dynamics: Is the system converging TO 1/φ?

Questions:
  1. Training trajectory: How does mean GELU'(b) evolve during training?
  2. Push/pull balance: What fraction of channels push (g>0.5) vs pull (g<0.5)?
  3. Does the optimal constant gate converge to 1/φ with more training?
  4. Does longer training push the system closer to 1/φ?
  5. What's the "true" destination — 1/φ, or something else?
  6. If we constrain the gate to 1/φ, is that strictly better?
  7. Per-channel: are channels converging individually or only on average?
"""
import numpy as np
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PhiMap, PHI, _standard_gelu, _standard_gelu_derivative

LOG_PHI = np.log(PHI)

DIM = 32
N_TRAIN = 500
N_TEST = 200

np.random.seed(42)
W_true = np.random.randn(DIM, DIM).astype(np.float32) * 0.5
def target_fn(x):
    return np.tanh(x @ W_true.T) + 0.1 * x**2

X_train = np.random.randn(N_TRAIN, DIM).astype(np.float32)
Y_train = target_fn(X_train)
X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
Y_test = target_fn(X_test)


# ================================================================
# Part 1: Training Trajectory — tracking gate center over time
# ================================================================
print('=' * 70)
print('PART 1: Training Trajectory of Gate Center')
print('=' * 70)
print()

checkpoints = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
trajectory_data = []

for seed in [42, 123, 456]:
    pm = PhiMap(DIM, expansion=4, gate='gelu')
    pm.init_random(seed=seed)

    # Manual training loop with checkpoints
    E = pm.E
    H = pm.H.copy()
    b = pm.b.copy()
    R = pm.R.copy()
    b_out = pm.b_out.copy()

    lr = 0.005
    prev_iter = 0

    for cp in checkpoints:
        n_iter = cp - prev_iter
        if n_iter <= 0:
            continue

        for _ in range(n_iter):
            z = X_train @ H.T + b
            h = _standard_gelu(z)
            Y_pred = h @ R.T + b_out
            err = Y_pred - Y_train

            dR = err.T @ h / N_TRAIN
            db_out = err.mean(axis=0)
            dh = err @ R
            gz = _standard_gelu_derivative(z)
            dz = dh * gz
            dH = dz.T @ X_train / N_TRAIN
            db = dz.mean(axis=0)

            H -= lr * dH
            b -= lr * db
            R -= lr * dR
            b_out -= lr * db_out

        prev_iter = cp

        # Measure gate statistics at this checkpoint
        gate_at_bias = _standard_gelu_derivative(b)
        mean_gate = gate_at_bias.mean()
        push_frac = (gate_at_bias > 0.5).sum() / E
        pull_frac = (gate_at_bias < 0.5).sum() / E

        # Test RMSE with different gates
        z_test = X_test @ H.T + b
        Y_gelu = _standard_gelu(z_test) @ R.T + b_out
        r_gelu = np.sqrt(np.mean((Y_gelu - Y_test)**2))

        Y_half = (0.5 * z_test) @ R.T + b_out
        r_half = np.sqrt(np.mean((Y_half - Y_test)**2))

        Y_phi = ((1/PHI) * z_test) @ R.T + b_out
        r_phi = np.sqrt(np.mean((Y_phi - Y_test)**2))

        # Find optimal constant gate at this checkpoint
        best_g = 0.5
        best_r = r_half
        for g in np.linspace(0.3, 0.8, 200):
            Y_g = (g * z_test) @ R.T + b_out
            r_g = np.sqrt(np.mean((Y_g - Y_test)**2))
            if r_g < best_r:
                best_r = r_g
                best_g = g

        trajectory_data.append({
            'seed': seed, 'iter': cp,
            'mean_gate': mean_gate, 'push_frac': push_frac,
            'rmse_gelu': r_gelu, 'rmse_half': r_half,
            'rmse_phi': r_phi, 'optimal_g': best_g,
            'gate_std': gate_at_bias.std(),
            'bias_mean': b.mean(), 'bias_std': b.std(),
        })

# Print trajectory
print(f"  {'Iter':<7} {'Mean g(b)':<11} {'Push%':<7} {'g_opt':<7} "
      f"{'RMSE_gelu':<11} {'RMSE_0.5':<11} {'RMSE_1/φ':<11} {'|g(b)-1/φ|'}")
print(f"  " + "-" * 85)

for seed in [42, 123, 456]:
    for d in trajectory_data:
        if d['seed'] != seed:
            continue
        gap = abs(d['mean_gate'] - 1/PHI)
        print(f"  {d['iter']:<7} {d['mean_gate']:<11.5f} {d['push_frac']*100:<7.1f} "
              f"{d['optimal_g']:<7.4f} {d['rmse_gelu']:<11.4f} {d['rmse_half']:<11.4f} "
              f"{d['rmse_phi']:<11.4f} {gap:.5f}")
    print()


# ================================================================
# Part 2: Push/Pull Asymmetry Analysis
# ================================================================
print()
print('=' * 70)
print('PART 2: Push/Pull Asymmetry')
print('=' * 70)
print()

# Use final trained model (seed=42, 20000 iter)
pm = PhiMap(DIM, expansion=4, gate='gelu')
pm.init_random(seed=42)
pm.fit(X_train, Y_train, n_iter=20000, lr=0.005)

gate_at_bias = _standard_gelu_derivative(pm.b)
z_test = X_test @ pm.H.T + pm.b

# For each channel, classify as push or pull
push_channels = gate_at_bias > 0.5
pull_channels = gate_at_bias <= 0.5

print(f"  Channel classification (at 20k iterations):")
print(f"    Push (gate > 0.5):  {push_channels.sum()}/{pm.E} ({push_channels.sum()/pm.E*100:.1f}%)")
print(f"    Pull (gate ≤ 0.5):  {pull_channels.sum()}/{pm.E} ({pull_channels.sum()/pm.E*100:.1f}%)")
print()

# Mean gate for push vs pull channels
if push_channels.sum() > 0:
    push_mean = gate_at_bias[push_channels].mean()
    push_distance_from_phi = abs(push_mean - 1/PHI)
    print(f"  Push channels: mean gate = {push_mean:.5f} (distance from 1/φ: {push_distance_from_phi:.5f})")
if pull_channels.sum() > 0:
    pull_mean = gate_at_bias[pull_channels].mean()
    pull_distance_from_phi2 = abs(pull_mean - 1/PHI**2)
    print(f"  Pull channels: mean gate = {pull_mean:.5f} (distance from 1/φ²: {pull_distance_from_phi2:.5f})")

print()
# Distribution of gates relative to φ-pair
in_phi_plus = ((gate_at_bias > 0.5) & (gate_at_bias <= 1/PHI)).sum()
above_phi = (gate_at_bias > 1/PHI).sum()
in_phi_minus = ((gate_at_bias <= 0.5) & (gate_at_bias >= 1/PHI**2)).sum()
below_phi2 = (gate_at_bias < 1/PHI**2).sum()

print(f"  Distribution relative to φ-pair (1/φ²=0.382, 1/φ=0.618):")
print(f"    Above 1/φ:     {above_phi}/{pm.E} ({above_phi/pm.E*100:.1f}%)")
print(f"    [0.5, 1/φ]:    {in_phi_plus}/{pm.E} ({in_phi_plus/pm.E*100:.1f}%)")
print(f"    [1/φ², 0.5]:   {in_phi_minus}/{pm.E} ({in_phi_minus/pm.E*100:.1f}%)")
print(f"    Below 1/φ²:    {below_phi2}/{pm.E} ({below_phi2/pm.E*100:.1f}%)")

# The push/pull RATIO
push_energy = gate_at_bias[push_channels].sum() if push_channels.sum() > 0 else 0
pull_energy = (1 - gate_at_bias[pull_channels]).sum() if pull_channels.sum() > 0 else 0
print(f"\n  Push/pull energy ratio: {push_energy:.2f} / {pull_energy:.2f} = {push_energy/(pull_energy+1e-10):.4f}")
print(f"  φ ratio:  {PHI:.4f}")
print(f"  φ² ratio: {PHI**2:.4f}")


# ================================================================
# Part 3: Convergence of Optimal Gate
# ================================================================
print()
print('=' * 70)
print('PART 3: Where Is the System Converging?')
print('=' * 70)
print()

# Extract optimal gate trajectory from part 1
for seed in [42]:
    seed_data = [d for d in trajectory_data if d['seed'] == seed]
    print(f"  Seed {seed} — Optimal gate trajectory:")
    print(f"    {'Iter':<7} {'g_opt':<8} {'|g_opt-0.5|':<12} {'|g_opt-1/φ|':<12} {'Moving toward'}")
    print(f"    " + "-" * 55)
    for d in seed_data:
        d05 = abs(d['optimal_g'] - 0.5)
        dphi = abs(d['optimal_g'] - 1/PHI)
        toward = "1/φ" if dphi < d05 else "0.5" if d05 < dphi else "EQUAL"
        print(f"    {d['iter']:<7} {d['optimal_g']:<8.4f} {d05:<12.5f} {dphi:<12.5f} {toward}")

# Also track the mean gate from bias
print()
print(f"  Seed 42 — Mean GELU'(b) trajectory:")
print(f"    {'Iter':<7} {'Mean g(b)':<11} {'|g(b)-0.5|':<12} {'|g(b)-1/φ|':<12} {'% toward 1/φ'}")
print(f"    " + "-" * 55)
for d in [dd for dd in trajectory_data if dd['seed'] == 42]:
    d05 = abs(d['mean_gate'] - 0.5)
    dphi = abs(d['mean_gate'] - 1/PHI)
    # fraction of the 0.5→1/φ gap covered
    frac = (d['mean_gate'] - 0.5) / (1/PHI - 0.5) * 100 if d['mean_gate'] > 0.5 else 0
    print(f"    {d['iter']:<7} {d['mean_gate']:<11.5f} {d05:<12.5f} {dphi:<12.5f} {frac:.1f}%")


# ================================================================
# Part 4: The Theoretically True System
# ================================================================
print()
print('=' * 70)
print('PART 4: The Theoretically True System')
print('=' * 70)
print()

# What if we force the bias so that GELU'(b) = 1/φ for all channels?
# Find the z where GELU'(z) = 1/φ
from scipy.optimize import brentq

def gelu_d_minus_target(z, target):
    return _standard_gelu_derivative(np.array([z]))[0] - target

z_phi = brentq(gelu_d_minus_target, -1, 3, args=(1/PHI,))
print(f"  GELU'(z) = 1/φ at z = {z_phi:.6f}")
print()

# Train but clamp bias to z_phi after each step
pm_constrained = PhiMap(DIM, expansion=4, gate='gelu')
pm_constrained.init_random(seed=42)

# Manual training with constrained bias
H_c = pm_constrained.H.copy()
b_c = np.full(pm_constrained.E, z_phi, dtype=np.float32)  # force all biases to z_phi
R_c = pm_constrained.R.copy()
b_out_c = pm_constrained.b_out.copy()

lr = 0.005
for it in range(20000):
    z = X_train @ H_c.T + b_c
    h = _standard_gelu(z)
    Y_pred = h @ R_c.T + b_out_c
    err = Y_pred - Y_train

    dR = err.T @ h / N_TRAIN
    db_out = err.mean(axis=0)
    dh = err @ R_c
    gz = _standard_gelu_derivative(z)
    dz = dh * gz
    dH = dz.T @ X_train / N_TRAIN

    H_c -= lr * dH
    # b_c stays at z_phi — no update
    R_c -= lr * dR
    b_out_c -= lr * db_out

z_c = X_test @ H_c.T + b_c
Y_constrained_gelu = _standard_gelu(z_c) @ R_c.T + b_out_c
Y_constrained_phi = ((1/PHI) * z_c) @ R_c.T + b_out_c
r_c_gelu = np.sqrt(np.mean((Y_constrained_gelu - Y_test)**2))
r_c_phi = np.sqrt(np.mean((Y_constrained_phi - Y_test)**2))

# Compare with unconstrained
pm_free = PhiMap(DIM, expansion=4, gate='gelu')
pm_free.init_random(seed=42)
pm_free.fit(X_train, Y_train, n_iter=20000, lr=0.005)

z_f = X_test @ pm_free.H.T + pm_free.b
Y_free_gelu = pm_free.lookup(X_test)
Y_free_phi = ((1/PHI) * z_f) @ pm_free.R.T + pm_free.b_out
r_f_gelu = np.sqrt(np.mean((Y_free_gelu - Y_test)**2))
r_f_phi = np.sqrt(np.mean((Y_free_phi - Y_test)**2))

print(f"  Comparison at 20k iterations:")
print(f"    {'Model':<30} {'GELU RMSE':<12} {'g=1/φ RMSE'}")
print(f"    " + "-" * 55)
print(f"    {'Free bias (learned)':<30} {r_f_gelu:<12.4f} {r_f_phi:.4f}")
print(f"    {'Constrained bias (all=z_φ)':<30} {r_c_gelu:<12.4f} {r_c_phi:.4f}")
print(f"    {'Difference':<30} {r_c_gelu-r_f_gelu:<12.4f} {r_c_phi-r_f_phi:.4f}")
print()

# What about: train with NO bias at all (pure scaffold)
H_nb = pm_constrained.H.copy()  # fresh copy
pm_nb = PhiMap(DIM, expansion=4, gate='gelu')
pm_nb.init_random(seed=42)
H_nb = pm_nb.H.copy()
b_nb = np.zeros(pm_nb.E, dtype=np.float32)
R_nb = pm_nb.R.copy()
b_out_nb = pm_nb.b_out.copy()

for it in range(20000):
    z = X_train @ H_nb.T  # no bias at all
    h = _standard_gelu(z)
    Y_pred = h @ R_nb.T + b_out_nb
    err = Y_pred - Y_train

    dR = err.T @ h / N_TRAIN
    db_out = err.mean(axis=0)
    dh = err @ R_nb
    gz = _standard_gelu_derivative(z)
    dz = dh * gz
    dH = dz.T @ X_train / N_TRAIN

    H_nb -= lr * dH
    R_nb -= lr * dR
    b_out_nb -= lr * db_out

z_nb = X_test @ H_nb.T
Y_nb_gelu = _standard_gelu(z_nb) @ R_nb.T + b_out_nb
Y_nb_phi = ((1/PHI) * z_nb) @ R_nb.T + b_out_nb
Y_nb_half = (0.5 * z_nb) @ R_nb.T + b_out_nb
r_nb_gelu = np.sqrt(np.mean((Y_nb_gelu - Y_test)**2))
r_nb_phi = np.sqrt(np.mean((Y_nb_phi - Y_test)**2))
r_nb_half = np.sqrt(np.mean((Y_nb_half - Y_test)**2))

print(f"    {'No bias (b=0, GELU center=0.5)':<30} {r_nb_gelu:<12.4f} {r_nb_phi:.4f}")
print(f"    {'No bias, g=0.5':<30} {r_nb_half:.4f}")
print()

# The critical comparison: with no bias, the GELU center IS 0.5.
# Does g=1/φ still help, or does it need the bias shift?
print(f"  No-bias system: does g=1/φ still help?")
print(f"    GELU:    {r_nb_gelu:.4f}")
print(f"    g=0.5:   {r_nb_half:.4f}")
print(f"    g=1/φ:   {r_nb_phi:.4f}")
if r_nb_phi < r_nb_half:
    print(f"    → YES, g=1/φ is {(1-r_nb_phi/r_nb_half)*100:.2f}% better than g=0.5 even without bias")
else:
    print(f"    → NO, g=0.5 is better without bias (by {(1-r_nb_half/r_nb_phi)*100:.2f}%)")


# ================================================================
# Part 5: The Destination — Asymptotic Analysis
# ================================================================
print()
print('=' * 70)
print('PART 5: Asymptotic Convergence')
print('=' * 70)
print()

# Track how the gap between mean gate and 1/φ changes
print(f"  How fast is the gate approaching 1/φ?")
print(f"  (Using seed=42 trajectory)")
print()

seed42_data = [d for d in trajectory_data if d['seed'] == 42]
print(f"    {'Iter':<7} {'Mean g(b)':<11} {'Gap to 1/φ':<12} {'Gap ratio':<12} {'φ-scaled'}")
print(f"    " + "-" * 60)

prev_gap = None
for d in seed42_data:
    gap = abs(d['mean_gate'] - 1/PHI)
    if prev_gap is not None and gap > 0:
        ratio = prev_gap / gap
        phi_scaled = ratio / PHI
        print(f"    {d['iter']:<7} {d['mean_gate']:<11.5f} {gap:<12.5f} {ratio:<12.4f} {phi_scaled:.4f}")
    else:
        print(f"    {d['iter']:<7} {d['mean_gate']:<11.5f} {gap:<12.5f} {'—':<12} —")
    prev_gap = gap

# Final assessment
final_d = seed42_data[-1]
final_gap = abs(final_d['mean_gate'] - 1/PHI)
initial_gap = abs(seed42_data[0]['mean_gate'] - 1/PHI)
print()
print(f"  Total convergence: gap from {initial_gap:.5f} → {final_gap:.5f}")
print(f"  Reduction: {(1 - final_gap/initial_gap)*100:.1f}%")
print(f"  Final mean gate: {final_d['mean_gate']:.5f}")
print(f"  1/φ = {1/PHI:.5f}")


# ================================================================
# Part 6: Per-Channel Convergence
# ================================================================
print()
print('=' * 70)
print('PART 6: Per-Channel Convergence — Individual Destinations')
print('=' * 70)
print()

# Are individual channels converging to φ-values, or is only the mean?
gate_final = _standard_gelu_derivative(pm_free.b)

# Check each channel's gate against φ-powers
phi_targets = {
    '1/φ⁴': 1/PHI**4,  # 0.146
    '1/φ³': 1/PHI**3,  # 0.236
    '1/φ²': 1/PHI**2,  # 0.382
    '0.5':  0.500,
    '1/φ':  1/PHI,      # 0.618
    '1-1/φ³': 1-1/PHI**3,  # 0.764
    '1-1/φ⁴': 1-1/PHI**4,  # 0.854
}

print(f"  Per-channel gates nearest φ-value:")
channel_nearest = []
for ch in range(pm_free.E):
    g = gate_final[ch]
    nearest = min(phi_targets.items(), key=lambda kv: abs(kv[1] - g))
    channel_nearest.append(nearest[0])

from collections import Counter
counts = Counter(channel_nearest)
print(f"    {'φ-value':<12} {'Count':<8} {'Fraction':<10} {'Expected gate'}")
print(f"    " + "-" * 45)
for name, val in sorted(phi_targets.items(), key=lambda x: x[1]):
    c = counts.get(name, 0)
    print(f"    {name:<12} {c:<8} {c/pm_free.E*100:<10.1f}% {val:.4f}")

# How close are channels to their nearest φ-value?
distances = []
for ch in range(pm_free.E):
    g = gate_final[ch]
    nearest_val = min(phi_targets.values(), key=lambda v: abs(v - g))
    distances.append(abs(g - nearest_val))
distances = np.array(distances)
print(f"\n  Distance to nearest φ-value:")
print(f"    Mean: {distances.mean():.5f}")
print(f"    Max:  {distances.max():.5f}")
print(f"    % within 0.01: {(distances < 0.01).sum()/pm_free.E*100:.1f}%")
print(f"    % within 0.05: {(distances < 0.05).sum()/pm_free.E*100:.1f}%")


# ================================================================
# Part 7: Is 1/φ an Attractor or a Waypoint?
# ================================================================
print()
print('=' * 70)
print('PART 7: Attractor Analysis — 1/φ vs Other Fixed Points')
print('=' * 70)
print()

# The key question: if we start with biases at different positions,
# does training push them toward 1/φ?
start_positions = [
    ('At 0 (GELU center)', 0.0),
    ('At z_φ (GELU\'=1/φ)', z_phi),
    ('At -z_φ (GELU\'=1/φ²)', -z_phi),
    ('At log(φ) (EXPAND boundary)', LOG_PHI),
    ('At -log(φ) (CONTRACT boundary)', -LOG_PHI),
]

print(f"  Starting bias at different positions, tracking where gate converges:")
print(f"    {'Start':<30} {'Init gate':<11} {'Final gate':<11} {'Converged to':<15} {'|final-1/φ|'}")
print(f"    " + "-" * 80)

for name, start_z in start_positions:
    pm_s = PhiMap(DIM, expansion=4, gate='gelu')
    pm_s.init_random(seed=42)

    H_s = pm_s.H.copy()
    b_s = np.full(pm_s.E, start_z, dtype=np.float32)
    R_s = pm_s.R.copy()
    b_out_s = pm_s.b_out.copy()

    for it in range(20000):
        z = X_train @ H_s.T + b_s
        h = _standard_gelu(z)
        Y_pred = h @ R_s.T + b_out_s
        err = Y_pred - Y_train

        dR = err.T @ h / N_TRAIN
        db_out = err.mean(axis=0)
        dh = err @ R_s
        gz = _standard_gelu_derivative(z)
        dz = dh * gz
        dH = dz.T @ X_train / N_TRAIN
        db = dz.mean(axis=0)

        H_s -= lr * dH
        b_s -= lr * db
        R_s -= lr * dR
        b_out_s -= lr * db_out

    init_gate = _standard_gelu_derivative(np.array([start_z]))[0]
    final_gate = _standard_gelu_derivative(b_s).mean()
    gap = abs(final_gate - 1/PHI)

    # What's it closest to?
    closest = min(phi_targets.items(), key=lambda kv: abs(kv[1] - final_gate))

    print(f"    {name:<30} {init_gate:<11.4f} {final_gate:<11.5f} {closest[0]:<15} {gap:.5f}")
