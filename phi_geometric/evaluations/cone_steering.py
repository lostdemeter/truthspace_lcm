#!/usr/bin/env python3
"""
Cone Steering: Controlling Where We Land in the Validity Cone

The cone IS the solution space. The network navigates through it using
depth (the 4th dimension) as the steering axis. The weights "remember"
a path that lands near a correct region.

Question: Can we add a CONTROL mechanism to steer WHERE in the cone
we land? And can we go BEYOND GELU — finding states that are even
better calibrated or more confident?

Experiments:
  1. Steering vectors: add per-block perturbations orthogonal to features
     → measure controllability (can we move in any direction?)
  2. Confidence steering: can we find a λ profile that INCREASES
     confidence beyond GELU?
  3. Calibration steering: can we find a path that improves calibration?
  4. Target steering: given a desired logit shift, can we find the
     per-block λ that achieves it?
  5. Beyond GELU: explore the cone PAST λ=1 (extrapolation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

PHI = (1 + np.sqrt(5)) / 2
SQRT_8_OVER_PI = np.sqrt(8.0 / np.pi)
C_GEOMETRIC = (4 - np.pi) / (6 * np.pi)


class SteerableGate(nn.Module):
    """Gate with continuous steering parameter λ.
    λ=0: Ideal Gate, λ=1: GELU, λ>1 or λ<0: extrapolation beyond the cone."""
    def __init__(self, lam=0.0):
        super().__init__()
        self.lam = lam
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
        ideal = x * torch.sigmoid(f)
        gelu = F.gelu(x)
        out = (1 - self.lam) * ideal + self.lam * gelu
        self.last_input = x.detach()
        self.last_output = out.detach()
        return out


def build_steerable_model(lambdas):
    """Build ConvNeXt with per-block steerable gates."""
    model = convnext_tiny(weights=weights_enum)
    model.eval()
    gates = []
    gate_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            lam = lambdas[gate_idx] if gate_idx < len(lambdas) else 0.0
            gate = SteerableGate(lam)
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

N_IMAGES = 30
N_BLOCKS = 18

# Precompute reference data
model_gelu = convnext_tiny(weights=weights_enum)
model_gelu.eval()

tensors = []
ref_logits = []
ref_confs = []
ref_preds = []

for idx in range(N_IMAGES):
    img = cv2.imread(images[idx])
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0)
    tensors.append(tensor)
    with torch.no_grad():
        logits = model_gelu(tensor)
    probs = F.softmax(logits, dim=1)
    ref_logits.append(logits)
    ref_confs.append(probs.max().item())
    ref_preds.append(logits.argmax(1).item())

# Ideal gate baseline
model_ideal, _ = build_steerable_model([0.0] * N_BLOCKS)
ideal_logits = []
ideal_confs = []
for idx in range(N_IMAGES):
    with torch.no_grad():
        logits = model_ideal(tensors[idx])
    ideal_logits.append(logits)
    ideal_confs.append(F.softmax(logits, dim=1).max().item())

print(f"Setup: {N_IMAGES} images, {N_BLOCKS} blocks")
print(f"GELU mean confidence: {np.mean(ref_confs):.4f}")
print(f"Ideal mean confidence: {np.mean(ideal_confs):.4f}")


# ============================================================================
# 1. BEYOND GELU: Extrapolate past λ=1
# ============================================================================

print()
print("=" * 80)
print("1. BEYOND GELU: Extrapolating the cone (λ > 1 and λ < 0)")
print("=" * 80)
print()

extrap_results = []
for lam in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0]:
    model_e, _ = build_steerable_model([lam] * N_BLOCKS)
    l2_list = []
    confs = []
    agree = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            le = model_e(tensors[idx])
        l2_list.append((le - ref_logits[idx]).norm().item())
        confs.append(F.softmax(le, dim=1).max().item())
        if le.argmax(1).item() == ref_preds[idx]:
            agree += 1
    extrap_results.append({
        'lam': lam, 'l2': np.mean(l2_list), 'conf': np.mean(confs),
        'agree': agree, 'conf_std': np.std(confs)
    })
    print(f"  λ={lam:+5.1f}: L2={np.mean(l2_list):.4f}  conf={np.mean(confs):.4f}±{np.std(confs):.4f}  "
          f"agree={agree}/{N_IMAGES}")

# Find the λ that maximizes confidence while maintaining agreement
valid_extrap = [r for r in extrap_results if r['agree'] == N_IMAGES]
if valid_extrap:
    best_conf = max(valid_extrap, key=lambda r: r['conf'])
    print(f"\n  Best confidence with perfect agreement: λ={best_conf['lam']:.1f} "
          f"conf={best_conf['conf']:.4f}")
    print(f"  GELU confidence: {np.mean(ref_confs):.4f}")
    print(f"  Difference: {(best_conf['conf'] - np.mean(ref_confs))*100:+.2f}%")


# ============================================================================
# 2. DIRECTIONAL STEERING: Which directions can we move in logit space?
# ============================================================================

print()
print("=" * 80)
print("2. DIRECTIONAL STEERING: Controllable dimensions")
print("=" * 80)
print()

# For a single image, vary λ at each block independently and measure
# the logit displacement vector
test_img = 0
ref_logit = ref_logits[test_img]
ideal_logit = ideal_logits[test_img]

# Baseline direction: Ideal → GELU
baseline_dir = (ref_logit - ideal_logit).flatten()
baseline_dir_norm = baseline_dir / (baseline_dir.norm() + 1e-12)

# Per-block steering directions: what logit direction does each block control?
block_directions = []
for k in range(N_BLOCKS):
    lambdas = [0.0] * N_BLOCKS
    lambdas[k] = 1.0
    model_k, _ = build_steerable_model(lambdas)
    with torch.no_grad():
        logits_k = model_k(tensors[test_img])

    direction = (logits_k - ideal_logit).flatten()
    block_directions.append(direction)

# Stack into matrix: [N_BLOCKS, 1000]
D = torch.stack(block_directions)

# SVD of the direction matrix — how many independent steering directions?
U_d, S_d, Vh_d = torch.linalg.svd(D, full_matrices=False)
S_d_np = S_d.numpy()
total_E = (S_d_np ** 2).sum()
cum_E = np.cumsum(S_d_np ** 2) / total_E

print(f"  Singular values of steering matrix (image {test_img}):")
for i in range(min(N_BLOCKS, 10)):
    print(f"    σ_{i} = {S_d_np[i]:.4f}  (cumulative energy: {cum_E[i]*100:.1f}%)")

effective_rank = np.searchsorted(cum_E, 0.95) + 1
print(f"\n  Effective rank (95% energy): {effective_rank} / {N_BLOCKS}")
print(f"  → We have {effective_rank} independent steering directions")

# How aligned are the per-block directions?
cos_matrix = torch.zeros(N_BLOCKS, N_BLOCKS)
for i in range(N_BLOCKS):
    for j in range(N_BLOCKS):
        cos_matrix[i, j] = F.cosine_similarity(
            D[i:i+1], D[j:j+1], dim=1).item()

mean_cos = (cos_matrix.sum() - cos_matrix.trace()) / (N_BLOCKS * (N_BLOCKS - 1))
print(f"  Mean pairwise cosine of block directions: {mean_cos.item():.4f}")
print(f"    (1.0 = all same direction, 0.0 = all independent)")

# φ-ratios in singular values?
ratios = S_d_np[:-1] / (S_d_np[1:] + 1e-12)
print(f"\n  S[i]/S[i+1] ratios: {', '.join(f'{r:.3f}' for r in ratios[:8])}")
print(f"  φ = {PHI:.3f}")


# ============================================================================
# 3. CONFIDENCE STEERING: Can we increase confidence beyond GELU?
# ============================================================================

print()
print("=" * 80)
print("3. CONFIDENCE STEERING: Optimize λ per block for max confidence")
print("=" * 80)
print()

# Greedy optimization: for each block, find λ that maximizes mean confidence
conf_lambdas = [0.5] * N_BLOCKS  # start from middle of cone
for iteration in range(3):
    for k in range(N_BLOCKS):
        best_lam = conf_lambdas[k]
        best_conf = -float('inf')
        for lam in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            trial = conf_lambdas.copy()
            trial[k] = lam
            model_t, _ = build_steerable_model(trial)
            confs = []
            all_agree = True
            for idx in range(min(15, N_IMAGES)):
                with torch.no_grad():
                    lt = model_t(tensors[idx])
                confs.append(F.softmax(lt, dim=1).max().item())
                if lt.argmax(1).item() != ref_preds[idx]:
                    all_agree = False
            mean_conf = np.mean(confs)
            if mean_conf > best_conf and all_agree:
                best_conf = mean_conf
                best_lam = lam
        conf_lambdas[k] = best_lam

    # Evaluate
    model_conf, _ = build_steerable_model(conf_lambdas)
    confs_eval = []
    agree_eval = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lc = model_conf(tensors[idx])
        confs_eval.append(F.softmax(lc, dim=1).max().item())
        if lc.argmax(1).item() == ref_preds[idx]:
            agree_eval += 1

    print(f"  Iter {iteration+1}: conf={np.mean(confs_eval):.4f}±{np.std(confs_eval):.4f}  "
          f"agree={agree_eval}/{N_IMAGES}")

print(f"\n  Confidence-optimized λ profile:")
for k in range(N_BLOCKS):
    bar = "█" * max(0, int(conf_lambdas[k] * 10))
    marker = " ← outside [0,1]" if conf_lambdas[k] < 0 or conf_lambdas[k] > 1 else ""
    print(f"    Block {k:2d}: λ={conf_lambdas[k]:+.1f} {bar}{marker}")

print(f"\n  GELU confidence:       {np.mean(ref_confs):.4f}")
print(f"  Ideal confidence:      {np.mean(ideal_confs):.4f}")
print(f"  Steered confidence:    {np.mean(confs_eval):.4f}")
print(f"  Improvement over GELU: {(np.mean(confs_eval) - np.mean(ref_confs))*100:+.3f}%")


# ============================================================================
# 4. TARGET STEERING: Navigate to a specific logit target
# ============================================================================

print()
print("=" * 80)
print("4. TARGET STEERING: Can we hit a specific target in the cone?")
print("=" * 80)
print()

# Target: for each image, try to maximize the ground-truth class logit
# while maintaining correct classification
# This is "sharpening" — making the network more decisive

target_lambdas = [0.5] * N_BLOCKS
for iteration in range(2):
    for k in range(N_BLOCKS):
        best_lam = target_lambdas[k]
        best_margin = -float('inf')
        for lam in [-0.3, 0.0, 0.3, 0.5, 0.7, 1.0, 1.3]:
            trial = target_lambdas.copy()
            trial[k] = lam
            model_t, _ = build_steerable_model(trial)
            margins = []
            for idx in range(min(10, N_IMAGES)):
                with torch.no_grad():
                    lt = model_t(tensors[idx])
                pred = lt.argmax(1).item()
                if pred == ref_preds[idx]:
                    # Margin: difference between top-1 and top-2
                    sorted_vals = lt.sort(dim=1, descending=True).values
                    margin = (sorted_vals[0, 0] - sorted_vals[0, 1]).item()
                    margins.append(margin)
            if len(margins) == min(10, N_IMAGES) and np.mean(margins) > best_margin:
                best_margin = np.mean(margins)
                best_lam = lam
        target_lambdas[k] = best_lam

    # Evaluate
    model_target, _ = build_steerable_model(target_lambdas)
    margins_eval = []
    agree_target = 0
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lt = model_target(tensors[idx])
        if lt.argmax(1).item() == ref_preds[idx]:
            agree_target += 1
        sorted_vals = lt.sort(dim=1, descending=True).values
        margins_eval.append((sorted_vals[0, 0] - sorted_vals[0, 1]).item())

    print(f"  Iter {iteration+1}: margin={np.mean(margins_eval):.3f}  agree={agree_target}/{N_IMAGES}")

# Compare margins
gelu_margins = []
ideal_margins = []
for idx in range(N_IMAGES):
    s_g = ref_logits[idx].sort(dim=1, descending=True).values
    gelu_margins.append((s_g[0, 0] - s_g[0, 1]).item())
    s_i = ideal_logits[idx].sort(dim=1, descending=True).values
    ideal_margins.append((s_i[0, 0] - s_i[0, 1]).item())

print(f"\n  GELU mean margin:    {np.mean(gelu_margins):.3f}")
print(f"  Ideal mean margin:   {np.mean(ideal_margins):.3f}")
print(f"  Steered mean margin: {np.mean(margins_eval):.3f}")
print(f"  Improvement: {(np.mean(margins_eval)/np.mean(gelu_margins) - 1)*100:+.2f}%")


# ============================================================================
# 5. CONE BOUNDARY: Where does validity break down?
# ============================================================================

print()
print("=" * 80)
print("5. CONE BOUNDARY: Where does agreement fail?")
print("=" * 80)
print()

# Sweep λ far beyond [0,1] to find the boundary
boundary_results = []
for lam in np.arange(-5.0, 6.0, 0.5):
    model_b, _ = build_steerable_model([lam] * N_BLOCKS)
    agree = 0
    confs = []
    for idx in range(N_IMAGES):
        with torch.no_grad():
            lb = model_b(tensors[idx])
        if lb.argmax(1).item() == ref_preds[idx]:
            agree += 1
        confs.append(F.softmax(lb, dim=1).max().item())
    boundary_results.append({
        'lam': lam, 'agree': agree, 'conf': np.mean(confs)
    })

print(f"  {'λ':<8} {'Agreement':<12} {'Confidence'}")
print(f"  {'-'*35}")
for r in boundary_results:
    marker = " ←" if r['agree'] < N_IMAGES else ""
    print(f"  {r['lam']:+5.1f}    {r['agree']}/{N_IMAGES}{'':>6}   {r['conf']:.4f}{marker}")

# Find the boundaries
valid_range = [r['lam'] for r in boundary_results if r['agree'] == N_IMAGES]
if valid_range:
    print(f"\n  Valid λ range: [{min(valid_range):.1f}, {max(valid_range):.1f}]")
    print(f"  Cone width: {max(valid_range) - min(valid_range):.1f}")
else:
    print(f"\n  No range with perfect agreement found")


# ============================================================================
# 6. PER-IMAGE STEERING: Different images need different steering?
# ============================================================================

print()
print("=" * 80)
print("6. PER-IMAGE STEERING: Does optimal λ vary per image?")
print("=" * 80)
print()

# For each image, find the λ that maximizes confidence
per_image_opt_lam = []
per_image_opt_conf = []
for idx in range(N_IMAGES):
    best_lam = 0.0
    best_conf = 0.0
    for lam in np.arange(-1.0, 3.1, 0.2):
        model_p, _ = build_steerable_model([lam] * N_BLOCKS)
        with torch.no_grad():
            lp = model_p(tensors[idx])
        if lp.argmax(1).item() == ref_preds[idx]:
            conf = F.softmax(lp, dim=1).max().item()
            if conf > best_conf:
                best_conf = conf
                best_lam = lam
    per_image_opt_lam.append(best_lam)
    per_image_opt_conf.append(best_conf)

print(f"  Per-image optimal λ (for max confidence):")
print(f"    mean={np.mean(per_image_opt_lam):.2f}, std={np.std(per_image_opt_lam):.2f}")
print(f"    min={np.min(per_image_opt_lam):.1f}, max={np.max(per_image_opt_lam):.1f}")
print(f"    Unique values: {sorted(set([f'{l:.1f}' for l in per_image_opt_lam]))}")
print()
print(f"  Mean confidence: GELU={np.mean(ref_confs):.4f}, "
      f"Steered={np.mean(per_image_opt_conf):.4f} "
      f"({(np.mean(per_image_opt_conf)-np.mean(ref_confs))*100:+.3f}%)")


# ============================================================================
# Visualization
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Extrapolation — L2 vs λ
ax1 = fig.add_subplot(gs[0, 0])
lams_e = [r['lam'] for r in extrap_results]
l2s_e = [r['l2'] for r in extrap_results]
agrees_e = [r['agree'] for r in extrap_results]
colors_e = ['green' if a == N_IMAGES else 'red' for a in agrees_e]
ax1.scatter(lams_e, l2s_e, c=colors_e, s=80, edgecolors='black', linewidths=0.5, zorder=5)
ax1.plot(lams_e, l2s_e, 'k-', alpha=0.3)
ax1.axvline(x=0, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='Ideal')
ax1.axvline(x=1, color='red', linewidth=1, linestyle='--', alpha=0.5, label='GELU')
ax1.axvspan(0, 1, alpha=0.05, color='yellow', label='Original cone')
ax1.set_xlabel('λ')
ax1.set_ylabel('L2 to GELU')
ax1.set_title('Extrapolation Beyond the Cone\n(green=valid, red=invalid)')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Panel 2: Confidence vs λ
ax2 = fig.add_subplot(gs[0, 1])
confs_e = [r['conf'] for r in extrap_results]
ax2.plot(lams_e, confs_e, 'ro-', linewidth=2, markersize=6)
ax2.axhline(y=np.mean(ref_confs), color='green', linewidth=1, linestyle='--',
            label=f'GELU conf={np.mean(ref_confs):.3f}')
ax2.axvspan(0, 1, alpha=0.05, color='yellow')
ax2.set_xlabel('λ')
ax2.set_ylabel('Mean confidence')
ax2.set_title('Confidence vs Steering λ')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: Steering direction SVD
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(range(len(S_d_np)), S_d_np, color='teal', alpha=0.7)
ax3.set_xlabel('Direction index')
ax3.set_ylabel('Singular value')
ax3.set_title(f'Steering Dimensions\n(effective rank={effective_rank})')
ax3.grid(True, alpha=0.3)

# Panel 4: Block direction cosine matrix
ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.imshow(cos_matrix.numpy(), cmap='RdBu_r', vmin=-0.5, vmax=1.0)
plt.colorbar(im4, ax=ax4, fraction=0.046)
ax4.set_xlabel('Block')
ax4.set_ylabel('Block')
ax4.set_title(f'Block Direction Similarity\nmean cos={mean_cos.item():.3f}')

# Panel 5: Confidence-optimized λ profile
ax5 = fig.add_subplot(gs[1, 0])
colors_c = ['blue' if 0 <= l <= 1 else 'red' for l in conf_lambdas]
ax5.bar(range(N_BLOCKS), conf_lambdas, color=colors_c, alpha=0.7, edgecolor='black', linewidth=0.5)
ax5.axhline(y=0, color='gray', linewidth=0.5)
ax5.axhline(y=1, color='gray', linewidth=0.5)
ax5.set_xlabel('Block')
ax5.set_ylabel('λ')
ax5.set_title(f'Confidence-Optimized λ\nconf={np.mean(confs_eval):.4f}')
ax5.grid(True, alpha=0.3)

# Panel 6: Target-steered margin profile
ax6 = fig.add_subplot(gs[1, 1])
colors_t = ['blue' if 0 <= l <= 1 else 'red' for l in target_lambdas]
ax6.bar(range(N_BLOCKS), target_lambdas, color=colors_t, alpha=0.7, edgecolor='black', linewidth=0.5)
ax6.axhline(y=0, color='gray', linewidth=0.5)
ax6.axhline(y=1, color='gray', linewidth=0.5)
ax6.set_xlabel('Block')
ax6.set_ylabel('λ')
ax6.set_title(f'Margin-Optimized λ\nmargin={np.mean(margins_eval):.3f}')
ax6.grid(True, alpha=0.3)

# Panel 7: Cone boundary
ax7 = fig.add_subplot(gs[1, 2])
lams_b = [r['lam'] for r in boundary_results]
agrees_b = [r['agree'] for r in boundary_results]
confs_b = [r['conf'] for r in boundary_results]
ax7.plot(lams_b, [a/N_IMAGES*100 for a in agrees_b], 'go-', linewidth=2, markersize=5, label='Agreement %')
ax7_twin = ax7.twinx()
ax7_twin.plot(lams_b, confs_b, 'r--', linewidth=1.5, alpha=0.7, label='Confidence')
# Shade valid region
if valid_range:
    ax7.axvspan(min(valid_range), max(valid_range), alpha=0.1, color='green')
ax7.set_xlabel('λ')
ax7.set_ylabel('Agreement %', color='green')
ax7_twin.set_ylabel('Confidence', color='red')
ax7.set_title('Cone Boundary')
ax7.grid(True, alpha=0.3)

# Panel 8: Per-image optimal λ histogram
ax8 = fig.add_subplot(gs[1, 3])
ax8.hist(per_image_opt_lam, bins=20, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
ax8.axvline(x=np.mean(per_image_opt_lam), color='red', linewidth=2,
            label=f'mean={np.mean(per_image_opt_lam):.2f}')
ax8.axvline(x=0, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='Ideal')
ax8.axvline(x=1, color='green', linewidth=1, linestyle='--', alpha=0.5, label='GELU')
ax8.set_xlabel('Optimal λ')
ax8.set_ylabel('Count')
ax8.set_title('Per-Image Optimal λ\n(for max confidence)')
ax8.legend(fontsize=8)

# Panel 9: Confidence comparison (per-image)
ax9 = fig.add_subplot(gs[2, 0])
x_idx = np.arange(N_IMAGES)
ax9.scatter(x_idx, ref_confs, c='green', s=30, label='GELU', zorder=5)
ax9.scatter(x_idx, ideal_confs, c='blue', s=30, label='Ideal', alpha=0.5)
ax9.scatter(x_idx, per_image_opt_conf, c='red', s=30, marker='*', label='Steered', zorder=5)
ax9.set_xlabel('Image')
ax9.set_ylabel('Confidence')
ax9.set_title('Per-Image Confidence Comparison')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

# Panel 10: Margin comparison
ax10 = fig.add_subplot(gs[2, 1])
ax10.bar(x_idx - 0.25, gelu_margins, 0.25, color='green', alpha=0.7, label='GELU')
ax10.bar(x_idx, ideal_margins, 0.25, color='blue', alpha=0.7, label='Ideal')
ax10.bar(x_idx + 0.25, margins_eval, 0.25, color='red', alpha=0.7, label='Steered')
ax10.set_xlabel('Image')
ax10.set_ylabel('Top-1 minus Top-2 margin')
ax10.set_title('Classification Margin')
ax10.legend(fontsize=7)

# Panel 11: The steerable cone diagram
ax11 = fig.add_subplot(gs[2, 2])
ax11.set_xlim(-1, 19)
ax11.set_ylim(-0.25, 0.25)

# Draw the extended cone (beyond λ=1)
from matplotlib.patches import FancyArrowPatch
cone_x = np.linspace(0, 18, 100)
# The cone expands linearly
radius = 0.002 * cone_x  # simplified
ax11.fill_between(cone_x, -radius * 30, radius * 30, alpha=0.08, color='gold')
# Inner valid region
ax11.fill_between(cone_x, -radius * 15, radius * 15, alpha=0.15, color='green',
                  label='Proven valid')

# Paths
ax11.plot([0, 18], [0, 0], 'g-', linewidth=3, label='GELU (λ=1)')
ax11.plot(cone_x, radius * 15, 'r-', linewidth=1.5, label='Ideal (λ=0)')
# Steered path
steered_y = np.sin(cone_x * 0.5) * radius * 10
ax11.plot(cone_x, steered_y, 'm-', linewidth=2, label='Steered path')

# Arrow showing control
ax11.annotate('', xy=(12, 0.02), xytext=(12, -0.02),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax11.text(12.5, 0, 'Steering\ncontrol', fontsize=8, color='purple', ha='left')

ax11.set_xlabel('Network depth (blocks)')
ax11.set_ylabel('Position in cone')
ax11.set_title('Steerable Validity Cone')
ax11.legend(fontsize=7, loc='upper left')
ax11.grid(True, alpha=0.3)

# Panel 12: Summary
ax12 = fig.add_subplot(gs[2, 3])
ax12.axis('off')
valid_width = (max(valid_range) - min(valid_range)) if valid_range else 0
summary = (
    f"CONE STEERING RESULTS\n"
    f"{'─' * 30}\n\n"
    f"Steering dimensions: {effective_rank}\n"
    f"Block direction cos: {mean_cos.item():.3f}\n\n"
    f"Cone boundary: [{min(valid_range):.1f}, {max(valid_range):.1f}]\n"
    f"Cone width: {valid_width:.1f}\n\n"
    f"Confidence:\n"
    f"  GELU:    {np.mean(ref_confs):.4f}\n"
    f"  Ideal:   {np.mean(ideal_confs):.4f}\n"
    f"  Steered: {np.mean(per_image_opt_conf):.4f}\n"
    f"  Δ: {(np.mean(per_image_opt_conf)-np.mean(ref_confs))*100:+.3f}%\n\n"
    f"Margin:\n"
    f"  GELU:    {np.mean(gelu_margins):.3f}\n"
    f"  Steered: {np.mean(margins_eval):.3f}\n"
    f"  Δ: {(np.mean(margins_eval)/np.mean(gelu_margins)-1)*100:+.2f}%\n\n"
    f"Per-image λ:\n"
    f"  mean={np.mean(per_image_opt_lam):.2f}\n"
    f"  std={np.std(per_image_opt_lam):.2f}"
)
ax12.text(0.05, 0.95, summary, transform=ax12.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Cone Steering: Controlling Where We Land\n'
             '"Use the 4th dimension to navigate and steer the output"',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/tmp/cone_steering.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print()
print("Saved: /tmp/cone_steering.png")
