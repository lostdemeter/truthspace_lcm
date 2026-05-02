"""
Test: Do DDColor PW1 biases sit on φ-level attractors?

DDColor biases are deeply negative (mean -0.65 to -2.48).
Our toy model biases converge to φ-levels.
Does the real model show the same structure?

Also: compare the warm/cool vocabulary ratio to the push/pull ratio.
- DDColor vocabulary: 86% warm, 14% cool
- Toy model gates: 89% push, 11% pull
- DDColor Phase 17: input flips 13-21% of channels
"""
import numpy as np
import torch
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/structures')
from phi_holographic_map import PHI, _standard_gelu_derivative

LOG_PHI = np.log(PHI)

# Load DDColor weights
weights_path = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_weights_static.npz'
try:
    weights = np.load(weights_path)
except:
    print("DDColor weights not available, exiting")
    sys.exit(1)

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

# ================================================================
# Part 1: PW1 Bias Distribution → GELU' → φ-levels
# ================================================================
print('=' * 70)
print('PART 1: DDColor PW1 Biases → Gate Values → φ-Levels')
print('=' * 70)
print()

phi_targets = {
    '0 (dead)': 0.0,
    '1/φ⁶': 1/PHI**6,   # 0.056
    '1/φ⁵': 1/PHI**5,   # 0.090
    '1/φ⁴': 1/PHI**4,   # 0.146
    '1/φ³': 1/PHI**3,   # 0.236
    '1/φ²': 1/PHI**2,   # 0.382
    '0.5':  0.500,
    '1/φ':  1/PHI,       # 0.618
    '1-1/φ³': 1-1/PHI**3,  # 0.764
    '1-1/φ⁴': 1-1/PHI**4,  # 0.854
    '1-1/φ⁵': 1-1/PHI**5,  # 0.910
    '1 (full)': 1.0,
}

all_gates = []
all_biases = []

for stage_idx in range(4):
    dim = dims[stage_idx]
    stage_gates = []

    print(f"  Stage {stage_idx} (dim={dim}, expansion=4, channels={4*dim}):")
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        bias_key = f'{prefix}.pwconv1.bias'

        if bias_key in weights:
            bias = weights[bias_key]
            gate = _standard_gelu_derivative(bias)
            stage_gates.extend(gate.tolist())
            all_gates.extend(gate.tolist())
            all_biases.extend(bias.tolist())

    if not stage_gates:
        print(f"    No PW1 biases found")
        continue

    stage_gates = np.array(stage_gates)

    # Gate statistics
    push = (stage_gates > 0.5).sum()
    pull = (stage_gates <= 0.5).sum()
    total = len(stage_gates)

    print(f"    Total channels: {total}")
    print(f"    Push (gate > 0.5): {push} ({push/total*100:.1f}%)")
    print(f"    Pull (gate ≤ 0.5): {pull} ({pull/total*100:.1f}%)")
    print(f"    Mean gate:  {stage_gates.mean():.4f}")
    print(f"    Mean bias:  {np.mean([b for b in all_biases[-total:]]):.4f}")

    # Nearest φ-level for each channel
    from collections import Counter
    nearest = []
    distances = []
    for g in stage_gates:
        best_name = min(phi_targets.items(), key=lambda kv: abs(kv[1] - g))
        nearest.append(best_name[0])
        distances.append(abs(g - best_name[1]))

    counts = Counter(nearest)
    print(f"    φ-level distribution:")
    for name, val in sorted(phi_targets.items(), key=lambda x: x[1]):
        c = counts.get(name, 0)
        if c > 0:
            print(f"      {name:<10}: {c:>5} ({c/total*100:>5.1f}%)")
    print(f"    Mean distance to nearest φ-level: {np.mean(distances):.4f}")
    print(f"    % within 0.05 of φ-level: {(np.array(distances) < 0.05).sum()/total*100:.1f}%")
    print()


# ================================================================
# Part 2: Overall Distribution — The Warm/Cool Connection
# ================================================================
print()
print('=' * 70)
print('PART 2: The Warm/Cool ↔ Push/Pull Connection')
print('=' * 70)
print()

all_gates = np.array(all_gates)
all_biases = np.array(all_biases)

push_total = (all_gates > 0.5).sum()
pull_total = (all_gates <= 0.5).sum()
total = len(all_gates)

print(f"  Overall DDColor PW1 gate distribution (all {total} channels):")
print(f"    Push (gate > 0.5): {push_total} ({push_total/total*100:.1f}%)")
print(f"    Pull (gate ≤ 0.5): {pull_total} ({pull_total/total*100:.1f}%)")
print(f"    Mean gate: {all_gates.mean():.4f}")
print()

# Compare with warm/cool vocabulary
print(f"  The parallels:")
print(f"    DDColor vocabulary: 86% warm / 14% cool")
print(f"    Toy model gates:   89% push / 11% pull")
print(f"    DDColor PW1 gates: {push_total/total*100:.1f}% push / {pull_total/total*100:.1f}% pull")
print()

# Per-stage push percentages
print(f"  Per-stage push percentage:")
offset = 0
for stage_idx in range(4):
    dim = dims[stage_idx]
    n_ch = 4 * dim * depths[stage_idx]
    if offset + n_ch > len(all_gates):
        n_ch = len(all_gates) - offset
    stage_g = all_gates[offset:offset+n_ch]
    push_pct = (stage_g > 0.5).sum() / len(stage_g) * 100 if len(stage_g) > 0 else 0
    pull_pct = (stage_g <= 0.5).sum() / len(stage_g) * 100 if len(stage_g) > 0 else 0
    print(f"    S{stage_idx}: {push_pct:.1f}% push / {pull_pct:.1f}% pull  (mean gate: {stage_g.mean():.4f})")
    offset += n_ch

# The big comparison with Doc 243 pre-GELU % negative
print()
print(f"  Doc 243 pre-GELU '% negative' = pull channels at that spatial position")
print(f"  Our gate-level analysis = pull channels across all positions")
print(f"  These are different views of the same asymmetry")


# ================================================================
# Part 3: Gate Histogram — Do Channels Quantize to φ-Levels?
# ================================================================
print()
print('=' * 70)
print('PART 3: Gate Histogram — φ-Level Quantization')
print('=' * 70)
print()

# Bin the gates and look for peaks at φ-levels
bins = np.linspace(0, 1, 100)
hist, edges = np.histogram(all_gates, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2

# Find peaks
from scipy.signal import find_peaks
peaks, props = find_peaks(hist, height=max(hist)*0.05, distance=3)

print(f"  Histogram peaks (bins with concentrations):")
for p in peaks:
    gate_val = centers[p]
    count = hist[p]
    # Find nearest φ-level
    nearest_phi = min(phi_targets.items(), key=lambda kv: abs(kv[1] - gate_val))
    dist = abs(gate_val - nearest_phi[1])
    print(f"    Peak at gate={gate_val:.3f}, count={count}, "
          f"nearest φ={nearest_phi[0]} ({nearest_phi[1]:.3f}), dist={dist:.4f}")

# Also report the φ-level targets and how many channels are within 0.02 of each
print()
print(f"  Channels within ±0.02 of each φ-level:")
for name, val in sorted(phi_targets.items(), key=lambda x: x[1]):
    if val == 0.0 or val == 1.0:
        continue
    close = ((all_gates > val - 0.02) & (all_gates < val + 0.02)).sum()
    if close > 0:
        print(f"    {name:<10} ({val:.4f}): {close:>5} channels ({close/total*100:.2f}%)")


# ================================================================
# Part 4: The Deep Structure — GELU Regions in DDColor
# ================================================================
print()
print('=' * 70)
print('PART 4: GELU Ternary Regions in DDColor (Using φ-Pair Levels)')
print('=' * 70)
print()

# Use the corrected φ-pair hierarchy from our discovery
# Level 0 boundaries: gate at (1/φ, 1/φ²) → z at ±0.149
# Level 2 boundaries: gate at (1-1/φ⁴, 1/φ⁴) → z at ±0.479 ≈ ±log(φ)

for level, (g_expand, g_contract) in enumerate([
    (1/PHI, 1/PHI**2),           # Level 0
    (1-1/PHI**3, 1/PHI**3),      # Level 1
    (1-1/PHI**4, 1/PHI**4),      # Level 2 ≈ ±log(φ)
    (1-1/PHI**5, 1/PHI**5),      # Level 3
]):
    expand = (all_gates > g_expand).sum()
    preserve = ((all_gates >= g_contract) & (all_gates <= g_expand)).sum()
    contract = (all_gates < g_contract).sum()

    print(f"  Level {level} boundaries: [{g_contract:.4f}, {g_expand:.4f}]")
    print(f"    EXPAND  (gate > {g_expand:.3f}): {expand:>5} ({expand/total*100:.1f}%)")
    print(f"    PRESERVE:                     {preserve:>5} ({preserve/total*100:.1f}%)")
    print(f"    CONTRACT(gate < {g_contract:.3f}): {contract:>5} ({contract/total*100:.1f}%)")
    print()
