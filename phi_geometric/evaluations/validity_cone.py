#!/usr/bin/env python3
"""
The Validity Cone: Scanning for Ground Truth

The flashlight analogy: shine a light into a dimension, everything it
touches is a valid state. The error growth ±1σ IS the cone opening.
Ground truth (GELU) is one point inside this cone.

Experiments:
  1. Cone geometry: opening angle, subspace dimensionality per block
  2. Per-block gate switching: use GELU at block k, Ideal elsewhere
     → which blocks matter most? (critical blocks)
  3. Continuous λ interpolation: gate_λ = (1-λ)·Ideal + λ·GELU per block
     → can we navigate to ground truth with 18 λ parameters?
  4. Cone interior sampling: random λ vectors → all produce valid states?
  5. Greedy scan: optimize λ per block to minimize distance to ground truth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import copy
import math
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

PHI = (1 + np.sqrt(5)) / 2
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)
C_GEOMETRIC = (4 - np.pi) / (6 * np.pi)


class InterpolatedGate(nn.Module):
    """gate_λ(x) = (1-λ)·IdealGate(x) + λ·GELU(x)
    At λ=0: Ideal Gate. At λ=1: GELU (ground truth)."""
    def __init__(self, lam=0.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        # Ideal gate
        f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
        ideal = x * torch.sigmoid(f)
        if self.lam == 0.0:
            return ideal
        elif self.lam == 1.0:
            return F.gelu(x)
        else:
            gelu = F.gelu(x)
            return (1 - self.lam) * ideal + self.lam * gelu


def build_model_with_lambdas(lambdas):
    """Build ConvNeXt with per-block interpolated gates."""
    model = convnext_tiny(weights=weights_enum)
    model.eval()
    gate_idx = 0
    gates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            lam = lambdas[gate_idx] if gate_idx < len(lambdas) else 0.0
            gate = InterpolatedGate(lam)
            setattr(parent, parts[-1], gate)
            gates.append(gate)
            gate_idx += 1
    return model, gates


# ============================================================================
# Setup
# ============================================================================

weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
preprocess = weights_enum.transforms()
image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))

# Reference: full GELU model
model_gelu = convnext_tiny(weights=weights_enum)
model_gelu.eval()

N_IMAGES = 30
N_BLOCKS = 18

# Precompute reference logits
ref_logits = []
ref_preds = []
tensors = []
for idx in range(N_IMAGES):
    img = cv2.imread(images[idx])
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0)
    tensors.append(tensor)
    with torch.no_grad():
        logits = model_gelu(tensor)
    ref_logits.append(logits)
    ref_preds.append(logits.argmax(1).item())

print(f"Prepared {N_IMAGES} images")


# ============================================================================
# 1. CONE GEOMETRY: Opening angle per block
# ============================================================================

print()
print("=" * 80)
print("1. CONE GEOMETRY: Opening angle per block")
print("=" * 80)
print()

# For each block depth, measure the cone radius (RMS) and opening angle
# Cone radius = RMS error. Opening angle = atan(radius / depth)
# Also measure the subspace dimensionality of the error at each depth

# Use full Ideal model (λ=0 everywhere)
lambdas_zero = [0.0] * N_BLOCKS
model_ideal, _ = build_model_with_lambdas(lambdas_zero)

# Hook all stages
stage_feats_gelu = {}
stage_feats_ideal = {}

stage_ids = list(range(8))
hooks_g = []
hooks_i = []
for s in stage_ids:
    hooks_g.append(model_gelu.features[s].register_forward_hook(
        lambda m, inp, out, stage=s: stage_feats_gelu.update({stage: out.detach()})))
    hooks_i.append(model_ideal.features[s].register_forward_hook(
        lambda m, inp, out, stage=s: stage_feats_ideal.update({stage: out.detach()})))

cone_radii = {s: [] for s in stage_ids}
cone_angles = {s: [] for s in stage_ids}
feat_magnitudes = {s: [] for s in stage_ids}

for idx in range(N_IMAGES):
    with torch.no_grad():
        _ = model_gelu(tensors[idx])
        _ = model_ideal(tensors[idx])
    for s in stage_ids:
        diff = stage_feats_gelu[s] - stage_feats_ideal[s]
        radius = diff.pow(2).mean().sqrt().item()
        feat_mag = stage_feats_gelu[s].pow(2).mean().sqrt().item()
        cone_radii[s].append(radius)
        feat_magnitudes[s].append(feat_mag)
        # Opening half-angle: atan(radius / feat_mag)
        if feat_mag > 0:
            cone_angles[s].append(np.arctan(radius / feat_mag) * 180 / np.pi)
        else:
            cone_angles[s].append(0)

for h in hooks_g + hooks_i:
    h.remove()

block_counts = [0, 3, 0, 3, 0, 9, 0, 3]
cum_blocks = np.cumsum(block_counts)

print(f"  {'Stage':<8} {'Blocks':<8} {'Cone radius':<14} {'|Feature|':<14} "
      f"{'Angle (°)':<12} {'Angle/block'}")
print(f"  {'-'*65}")
for s in stage_ids:
    r = np.mean(cone_radii[s])
    f_mag = np.mean(feat_magnitudes[s])
    angle = np.mean(cone_angles[s])
    per_block = angle / max(cum_blocks[s], 1)
    print(f"  s{s:<7} {cum_blocks[s]:<8} {r:<14.6f} {f_mag:<14.4f} "
          f"{angle:<12.4f} {per_block:.4f}")


# ============================================================================
# 2. PER-BLOCK GATE SWITCHING: Which blocks matter most?
# ============================================================================

print()
print("=" * 80)
print("2. CRITICAL BLOCKS: Use GELU at block k, Ideal elsewhere")
print("=" * 80)
print()

# For each block k: set λ_k=1, all others λ=0
# Measure: how much closer to ground truth does this get?

block_importance = []
for k in range(N_BLOCKS):
    lambdas = [0.0] * N_BLOCKS
    lambdas[k] = 1.0  # GELU at this block only
    model_k, _ = build_model_with_lambdas(lambdas)

    l2_list = []
    agree = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            logits_k = model_k(tensors[idx])
        l2_list.append((logits_k - ref_logits[idx]).norm().item())
        if logits_k.argmax(1).item() == ref_preds[idx]:
            agree += 1

    mean_l2 = np.mean(l2_list)
    block_importance.append(mean_l2)
    print(f"  Block {k:2d}: L2 to GELU = {mean_l2:.4f}  agree={agree}/{N_IMAGES}")

# Also measure baseline (all Ideal)
lambdas_zero = [0.0] * N_BLOCKS
model_baseline, _ = build_model_with_lambdas(lambdas_zero)
l2_baseline = []
for idx in range(N_IMAGES):
    with torch.no_grad():
        logits_b = model_baseline(tensors[idx])
    l2_baseline.append((logits_b - ref_logits[idx]).norm().item())
baseline_l2 = np.mean(l2_baseline)

print(f"\n  Baseline (all Ideal): L2 = {baseline_l2:.4f}")
print(f"  Most critical block: {np.argmin(block_importance)} "
      f"(L2 = {np.min(block_importance):.4f})")

# Sort by importance
sorted_blocks = np.argsort(block_importance)
print(f"\n  Block ranking (most helpful → least):")
for rank, k in enumerate(sorted_blocks[:6]):
    improvement = (1 - block_importance[k] / baseline_l2) * 100
    print(f"    #{rank+1}: Block {k} → L2={block_importance[k]:.4f} ({improvement:+.1f}%)")


# ============================================================================
# 3. CUMULATIVE SWITCHING: Add GELU blocks one at a time (best first)
# ============================================================================

print()
print("=" * 80)
print("3. CUMULATIVE SCAN: Add GELU blocks greedily")
print("=" * 80)
print()

# Start with all-Ideal. Greedily switch the most helpful block to GELU.
current_lambdas = [0.0] * N_BLOCKS
remaining = set(range(N_BLOCKS))
cumulative_l2 = [baseline_l2]
cumulative_agree = []
switch_order = []

# Measure baseline agreement
agree_base = 0
for idx in range(N_IMAGES):
    with torch.no_grad():
        lb = model_baseline(tensors[idx])
    if lb.argmax(1).item() == ref_preds[idx]:
        agree_base += 1
cumulative_agree.append(agree_base)

for step in range(N_BLOCKS):
    best_k = None
    best_l2 = float('inf')

    for k in remaining:
        trial = current_lambdas.copy()
        trial[k] = 1.0
        model_trial, _ = build_model_with_lambdas(trial)
        l2_list = []
        for idx in range(min(10, N_IMAGES)):  # quick eval
            with torch.no_grad():
                lt = model_trial(tensors[idx])
            l2_list.append((lt - ref_logits[idx]).norm().item())
        if np.mean(l2_list) < best_l2:
            best_l2 = np.mean(l2_list)
            best_k = k

    current_lambdas[best_k] = 1.0
    remaining.remove(best_k)
    switch_order.append(best_k)

    # Full eval
    model_step, _ = build_model_with_lambdas(current_lambdas)
    l2_full = []
    agree_step = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            ls = model_step(tensors[idx])
        l2_full.append((ls - ref_logits[idx]).norm().item())
        if ls.argmax(1).item() == ref_preds[idx]:
            agree_step += 1
    cumulative_l2.append(np.mean(l2_full))
    cumulative_agree.append(agree_step)

    pct = (1 - np.mean(l2_full) / baseline_l2) * 100
    print(f"  Step {step+1:2d}: Switch block {best_k:2d} → L2={np.mean(l2_full):.4f} "
          f"({pct:+.1f}%)  agree={agree_step}/{N_IMAGES}")

print(f"\n  Switch order: {switch_order}")
print(f"  (blocks ordered by how much they help reach ground truth)")


# ============================================================================
# 4. CONTINUOUS λ: Optimize per-block λ to reach ground truth
# ============================================================================

print()
print("=" * 80)
print("4. CONTINUOUS λ: Optimize per-block interpolation")
print("=" * 80)
print()

# Try uniform λ across all blocks
uniform_lambdas_results = []
for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    model_u, _ = build_model_with_lambdas([lam] * N_BLOCKS)
    l2_list = []
    agree = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lu = model_u(tensors[idx])
        l2_list.append((lu - ref_logits[idx]).norm().item())
        if lu.argmax(1).item() == ref_preds[idx]:
            agree += 1
    mean_l2 = np.mean(l2_list)
    uniform_lambdas_results.append((lam, mean_l2, agree))
    print(f"  λ={lam:.1f}: L2={mean_l2:.4f}  agree={agree}/{N_IMAGES}")

# Find optimal uniform λ
best_uniform = min(uniform_lambdas_results, key=lambda x: x[1])
print(f"\n  Best uniform λ = {best_uniform[0]:.1f} → L2={best_uniform[1]:.4f}")


# ============================================================================
# 5. CONE INTERIOR: Random λ vectors — are they all valid?
# ============================================================================

print()
print("=" * 80)
print("5. CONE INTERIOR: Random paths through the cone")
print("=" * 80)
print()

# Sample random λ vectors and check if they all produce valid classifications
np.random.seed(42)
n_random = 50
random_results = []

for trial in range(n_random):
    lambdas_rand = np.random.uniform(0, 1, N_BLOCKS).tolist()
    model_r, _ = build_model_with_lambdas(lambdas_rand)
    agree = 0
    l2_list = []
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lr = model_r(tensors[idx])
        l2_list.append((lr - ref_logits[idx]).norm().item())
        if lr.argmax(1).item() == ref_preds[idx]:
            agree += 1
    random_results.append((agree, np.mean(l2_list), lambdas_rand))

agrees = [r[0] for r in random_results]
l2s = [r[1] for r in random_results]
print(f"  Random λ vectors ({n_random} trials):")
print(f"    Agreement range: {min(agrees)}-{max(agrees)} / {N_IMAGES}")
print(f"    Mean agreement: {np.mean(agrees):.1f} / {N_IMAGES}")
print(f"    L2 range: {min(l2s):.4f} - {max(l2s):.4f}")
print(f"    All valid (>50% agreement): {sum(1 for a in agrees if a > N_IMAGES//2)}/{n_random}")
print(f"    Perfect agreement: {sum(1 for a in agrees if a == N_IMAGES)}/{n_random}")


# ============================================================================
# 6. GREEDY λ OPTIMIZATION: per-block continuous λ
# ============================================================================

print()
print("=" * 80)
print("6. GREEDY λ OPTIMIZATION (per-block)")
print("=" * 80)
print()

# For each block, find the optimal λ while keeping others fixed
# Start from all-zero, optimize one block at a time

opt_lambdas = [0.0] * N_BLOCKS
for iteration in range(3):  # multiple passes
    for k in range(N_BLOCKS):
        best_lam = opt_lambdas[k]
        best_l2 = float('inf')
        for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            trial = opt_lambdas.copy()
            trial[k] = lam
            model_t, _ = build_model_with_lambdas(trial)
            l2_list = []
            for idx in range(min(10, N_IMAGES)):
                with torch.no_grad():
                    lt = model_t(tensors[idx])
                l2_list.append((lt - ref_logits[idx]).norm().item())
            if np.mean(l2_list) < best_l2:
                best_l2 = np.mean(l2_list)
                best_lam = lam
        opt_lambdas[k] = best_lam

    # Evaluate
    model_opt, _ = build_model_with_lambdas(opt_lambdas)
    l2_opt = []
    agree_opt = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lo = model_opt(tensors[idx])
        l2_opt.append((lo - ref_logits[idx]).norm().item())
        if lo.argmax(1).item() == ref_preds[idx]:
            agree_opt += 1

    pct = (1 - np.mean(l2_opt) / baseline_l2) * 100
    print(f"  Iteration {iteration+1}: L2={np.mean(l2_opt):.4f} ({pct:+.1f}%)  "
          f"agree={agree_opt}/{N_IMAGES}")

print(f"\n  Optimized λ per block:")
for k in range(N_BLOCKS):
    bar = "█" * int(opt_lambdas[k] * 20)
    print(f"    Block {k:2d}: λ={opt_lambdas[k]:.1f} {bar}")

# How much of the way to ground truth?
print(f"\n  Baseline (all Ideal):    L2 = {baseline_l2:.4f}")
print(f"  Optimized λ:             L2 = {np.mean(l2_opt):.4f} "
      f"({(1-np.mean(l2_opt)/baseline_l2)*100:.1f}% reduction)")
print(f"  Ground truth (all GELU): L2 = 0.0000")


# ============================================================================
# Visualization
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Cone geometry — opening angle vs depth
ax1 = fig.add_subplot(gs[0, 0])
means = [np.mean(cone_angles[s]) for s in stage_ids]
stds = [np.std(cone_angles[s]) for s in stage_ids]
ax1.fill_between(cum_blocks, [m - s for m, s in zip(means, stds)],
                 [m + s for m, s in zip(means, stds)], alpha=0.3, color='red')
ax1.plot(cum_blocks, means, 'ro-', linewidth=2, markersize=6)
ax1.set_xlabel('Cumulative blocks')
ax1.set_ylabel('Cone half-angle (degrees)')
ax1.set_title('Validity Cone Opening')
ax1.grid(True, alpha=0.3)

# Panel 2: Cone radius (RMS) vs depth — with ±1σ band
ax2 = fig.add_subplot(gs[0, 1])
r_means = [np.mean(cone_radii[s]) for s in stage_ids]
r_stds = [np.std(cone_radii[s]) for s in stage_ids]
ax2.fill_between(cum_blocks,
                 [max(m - s, 0) for m, s in zip(r_means, r_stds)],
                 [m + s for m, s in zip(r_means, r_stds)],
                 alpha=0.3, color='red', label='±1σ (the cone)')
ax2.plot(cum_blocks, r_means, 'ro-', linewidth=2, markersize=6, label='Mean radius')
ax2.set_xlabel('Cumulative blocks')
ax2.set_ylabel('Cone radius (RMS)')
ax2.set_title('Cone Radius = Error Growth')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: Per-block importance
ax3 = fig.add_subplot(gs[0, 2])
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, N_BLOCKS))
sort_idx = np.argsort(block_importance)
ax3.barh(range(N_BLOCKS), [block_importance[i] for i in sort_idx],
         color=[colors[i] for i in sort_idx])
ax3.set_yticks(range(N_BLOCKS))
ax3.set_yticklabels([f'Block {i}' for i in sort_idx], fontsize=7)
ax3.axvline(x=baseline_l2, color='black', linewidth=2, linestyle='--',
            label=f'Baseline={baseline_l2:.3f}')
ax3.set_xlabel('L2 to GELU (lower = more helpful)')
ax3.set_title('Block Importance\n(GELU at this block only)')
ax3.legend(fontsize=8)

# Panel 4: Cumulative greedy switching
ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(range(len(cumulative_l2)), cumulative_l2, 'ro-', linewidth=2, markersize=5)
ax4.fill_between(range(len(cumulative_l2)), cumulative_l2, 0, alpha=0.1, color='red')
ax4.set_xlabel('# blocks switched to GELU')
ax4.set_ylabel('L2 to ground truth')
ax4.set_title('Greedy Scan to Ground Truth')
ax4.grid(True, alpha=0.3)
# Annotate switch order
for i, k in enumerate(switch_order[:5]):
    ax4.annotate(f'blk {k}', (i+1, cumulative_l2[i+1]),
                 fontsize=7, textcoords="offset points", xytext=(5, 5))

# Panel 5: Uniform λ curve
ax5 = fig.add_subplot(gs[1, 0])
lams = [r[0] for r in uniform_lambdas_results]
l2s_u = [r[1] for r in uniform_lambdas_results]
agrees_u = [r[2] for r in uniform_lambdas_results]
ax5.plot(lams, l2s_u, 'ro-', linewidth=2, markersize=6, label='L2')
ax5_twin = ax5.twinx()
ax5_twin.plot(lams, agrees_u, 'bs--', linewidth=1.5, markersize=5, label='Agreement')
ax5.set_xlabel('λ (uniform across all blocks)')
ax5.set_ylabel('L2 to GELU', color='red')
ax5_twin.set_ylabel('Agreement', color='blue')
ax5.set_title('Uniform λ Interpolation')
ax5.grid(True, alpha=0.3)

# Panel 6: Random cone interior sampling
ax6 = fig.add_subplot(gs[1, 1])
ax6.scatter(l2s, agrees, c='purple', s=40, alpha=0.6, edgecolors='black', linewidths=0.5)
ax6.axhline(y=N_IMAGES, color='green', linewidth=1, linestyle='--', label='Perfect')
ax6.axhline(y=N_IMAGES * 0.5, color='red', linewidth=1, linestyle='--', label='50%')
ax6.set_xlabel('L2 to GELU')
ax6.set_ylabel(f'Agreement (/{N_IMAGES})')
ax6.set_title(f'Cone Interior: {n_random} Random Paths\nAll valid? {sum(1 for a in agrees if a > N_IMAGES//2)}/{n_random}')
ax6.legend(fontsize=8)

# Panel 7: Optimized λ profile
ax7 = fig.add_subplot(gs[1, 2])
ax7.bar(range(N_BLOCKS), opt_lambdas, color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
ax7.set_xlabel('Block')
ax7.set_ylabel('Optimal λ')
ax7.set_title(f'Greedy-Optimized λ Profile\nL2={np.mean(l2_opt):.4f}')
ax7.set_ylim(0, 1.1)
ax7.grid(True, alpha=0.3)

# Panel 8: The cone diagram (conceptual)
ax8 = fig.add_subplot(gs[1, 3])
ax8.set_xlim(-0.5, 18.5)
ax8.set_ylim(-0.15, 0.15)

# Draw the cone
cone_x = np.array([0] + list(cum_blocks))
cone_upper = np.array([0] + [np.mean(cone_radii[s]) + np.std(cone_radii[s]) for s in stage_ids])
cone_lower = -cone_upper
ax8.fill_between(cone_x, cone_lower, cone_upper, alpha=0.15, color='gold',
                 label='Validity cone')
ax8.plot(cone_x, cone_upper, 'orange', linewidth=1.5)
ax8.plot(cone_x, cone_lower, 'orange', linewidth=1.5)

# Draw paths
# GELU path (ground truth) = 0 error = straight line at y=0
ax8.plot([0, 18], [0, 0], 'g-', linewidth=2, label='GELU (ground truth)')

# Ideal gate path = growing error
ideal_y = [0] + [np.mean(cone_radii[s]) * 0.7 for s in stage_ids]
ax8.plot(cone_x, ideal_y, 'r-', linewidth=2, label='Ideal Gate')

# Random path
np.random.seed(7)
random_y = [0]
for s in stage_ids:
    random_y.append(np.random.uniform(-1, 1) * np.mean(cone_radii[s]))
ax8.plot(cone_x, random_y, 'b--', linewidth=1, alpha=0.5, label='Random path')

ax8.set_xlabel('Network depth (blocks)')
ax8.set_ylabel('Distance from ground truth')
ax8.set_title('The Validity Cone\n"Everything the flashlight touches is valid"')
ax8.legend(fontsize=7, loc='upper left')
ax8.grid(True, alpha=0.3)

# Panel 9-12: Agreement vs depth for cumulative switching
ax9 = fig.add_subplot(gs[2, 0])
ax9.plot(range(len(cumulative_agree)), cumulative_agree, 'go-', linewidth=2, markersize=5)
ax9.axhline(y=N_IMAGES, color='green', linewidth=1, linestyle='--', alpha=0.5)
ax9.set_xlabel('# blocks switched to GELU')
ax9.set_ylabel(f'Agreement (/{N_IMAGES})')
ax9.set_title('Agreement During Scan')
ax9.set_ylim(min(cumulative_agree) - 1, N_IMAGES + 1)
ax9.grid(True, alpha=0.3)

# Panel 10: L2 reduction per block switch
ax10 = fig.add_subplot(gs[2, 1])
l2_deltas = [cumulative_l2[i] - cumulative_l2[i+1] for i in range(len(cumulative_l2)-1)]
ax10.bar(range(len(l2_deltas)), l2_deltas, color='teal', alpha=0.7)
ax10.set_xlabel('Switch step')
ax10.set_ylabel('ΔL2 (improvement)')
ax10.set_title('L2 Improvement Per Block Switch')
# Label which block was switched
for i, k in enumerate(switch_order[:8]):
    ax10.annotate(f'blk {k}', (i, l2_deltas[i]), fontsize=7,
                  textcoords="offset points", xytext=(0, 5), ha='center')
ax10.grid(True, alpha=0.3)

# Panel 11: Cone opening rate
ax11 = fig.add_subplot(gs[2, 2])
# Plot angle/block as a function of depth
per_block_angle = [np.mean(cone_angles[s]) / max(cum_blocks[s], 1) for s in stage_ids]
ax11.bar(range(len(stage_ids)), per_block_angle, color='coral', alpha=0.7)
ax11.set_xlabel('Stage')
ax11.set_ylabel('Angle per block (degrees)')
ax11.set_title('Cone Opening Rate Per Block')
ax11.grid(True, alpha=0.3)

# Panel 12: Summary
ax12 = fig.add_subplot(gs[2, 3])
ax12.axis('off')
opt_pct = (1 - np.mean(l2_opt) / baseline_l2) * 100
summary = (
    f"THE VALIDITY CONE\n"
    f"{'─' * 30}\n\n"
    f"Cone radius at depth 18:\n"
    f"  {np.mean(cone_radii[7]):.5f}\n\n"
    f"Cone angle at depth 18:\n"
    f"  {np.mean(cone_angles[7]):.3f}°\n\n"
    f"All random paths valid?\n"
    f"  {sum(1 for a in agrees if a > N_IMAGES//2)}/{n_random}\n\n"
    f"Greedy scan closes gap:\n"
    f"  {baseline_l2:.4f} → {np.mean(l2_opt):.4f}\n"
    f"  ({opt_pct:.1f}% reduction)\n\n"
    f"Best uniform λ:\n"
    f"  λ={best_uniform[0]} → L2={best_uniform[1]:.4f}\n\n"
    f"Switch order (top 5):\n"
    f"  {switch_order[:5]}"
)
ax12.text(0.05, 0.95, summary, transform=ax12.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('The Validity Cone: Scanning for Ground Truth\n'
             '"Shine a flashlight into a dimension — everything it touches is valid"',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/tmp/validity_cone.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print()
print("Saved: /tmp/validity_cone.png")
