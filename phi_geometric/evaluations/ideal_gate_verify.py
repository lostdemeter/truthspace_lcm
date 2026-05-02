#!/usr/bin/env python3
"""
Verify the Ideal Gate is ACTUALLY running, not just showing GELU.

1. Prove the gate modules were replaced (count GELU vs IdealGate)
2. Hook activations INSIDE the gate to show they differ
3. Side-by-side feature maps from GELU vs Ideal Gate
4. Show the actual gate output distribution differs (slightly)
5. Show images processed through EACH model independently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import copy
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


class IdealGate(nn.Module):
    """gate(x) = x · σ(√(8/π) · x · (1 + [(4-π)/(6π)] · x²))"""
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.call_count += 1
        self.last_input = x.detach()
        f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
        out = x * torch.sigmoid(f)
        self.last_output = out.detach()
        return out


class TrackedGELU(nn.Module):
    """Standard GELU but tracks calls."""
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.call_count += 1
        self.last_input = x.detach()
        out = F.gelu(x)
        self.last_output = out.detach()
        return out


def replace_gelu(model, gate_class):
    """Replace all GELU modules, return list of gate instances."""
    gates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            gate = gate_class()
            setattr(parent, parts[-1], gate)
            gates.append((name, gate))
    return gates


# ============================================================================
# Load models
# ============================================================================

weights_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
preprocess = weights_enum.transforms()
image_dir = '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017'
images = sorted(glob.glob(f'{image_dir}/*.jpg'))

model_gelu = convnext_tiny(weights=weights_enum)
model_gelu.eval()
gelu_gates = replace_gelu(model_gelu, TrackedGELU)

model_ideal = convnext_tiny(weights=weights_enum)
model_ideal.eval()
ideal_gates = replace_gelu(model_ideal, IdealGate)


# ============================================================================
# Step 1: Prove the replacement happened
# ============================================================================

print("=" * 80)
print("VERIFICATION: Gate Replacement")
print("=" * 80)
print()

# Count module types
gelu_count = sum(1 for _, m in model_gelu.named_modules() if isinstance(m, TrackedGELU))
ideal_count = sum(1 for _, m in model_ideal.named_modules() if isinstance(m, IdealGate))
native_gelu_in_ideal = sum(1 for _, m in model_ideal.named_modules() if isinstance(m, nn.GELU))

print(f"  model_gelu:  {gelu_count} TrackedGELU modules")
print(f"  model_ideal: {ideal_count} IdealGate modules")
print(f"  model_ideal: {native_gelu_in_ideal} remaining nn.GELU (should be 0)")
print(f"  Replaced locations (first 5):")
for name, gate in ideal_gates[:5]:
    print(f"    {name} -> IdealGate")
print(f"    ... ({len(ideal_gates)} total)")
print()


# ============================================================================
# Step 2: Run one image through both, verify gates fire
# ============================================================================

print("=" * 80)
print("VERIFICATION: Gates Actually Fire")
print("=" * 80)
print()

img = cv2.imread(images[0])
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
tensor = preprocess(pil_img).unsqueeze(0)

with torch.no_grad():
    logits_g = model_gelu(tensor)
    logits_i = model_ideal(tensor)

print(f"  TrackedGELU calls: {[g.call_count for _, g in gelu_gates[:5]]}... "
      f"(total {sum(g.call_count for _, g in gelu_gates)})")
print(f"  IdealGate calls:   {[g.call_count for _, g in ideal_gates[:5]]}... "
      f"(total {sum(g.call_count for _, g in ideal_gates)})")
print()

# Check outputs differ
print(f"  GELU logits[0,:5]:  {logits_g[0,:5].tolist()}")
print(f"  Ideal logits[0,:5]: {logits_i[0,:5].tolist()}")
print(f"  Same prediction: {logits_g.argmax().item()} vs {logits_i.argmax().item()}")
cos = F.cosine_similarity(logits_g, logits_i, dim=1).item()
print(f"  Cosine similarity: {cos:.6f}")
l2 = (logits_g - logits_i).norm().item()
print(f"  L2 distance: {l2:.6f} (should be >0 if gate is different)")
print()


# ============================================================================
# Step 3: Compare gate input/output distributions
# ============================================================================

print("=" * 80)
print("VERIFICATION: Gate Activations Differ")
print("=" * 80)
print()

# Pick the first gate in each model
gelu_gate = gelu_gates[0][1]
ideal_gate = ideal_gates[0][1]

gelu_in = gelu_gate.last_input
gelu_out = gelu_gate.last_output
ideal_in = ideal_gate.last_input
ideal_out = ideal_gate.last_output

print(f"  Gate: {gelu_gates[0][0]}")
print(f"  Input shapes: GELU={gelu_in.shape}, Ideal={ideal_in.shape}")
print(f"  Inputs identical? {torch.allclose(gelu_in, ideal_in)}")
print(f"    (Same because both get same input from previous layer)")
print()

# The outputs SHOULD differ slightly
out_diff = (gelu_out - ideal_out).abs()
print(f"  Output diff stats:")
print(f"    max:  {out_diff.max().item():.8f}")
print(f"    mean: {out_diff.mean().item():.8f}")
print(f"    Outputs identical? {torch.allclose(gelu_out, ideal_out)}")
print()

# Show what the gate actually does differently
x_sample = gelu_in.flatten()[:20]
gelu_applied = F.gelu(x_sample)
ideal_applied = x_sample * torch.sigmoid(SQRT_8_OVER_PI * x_sample * (1 + C_GEOMETRIC * x_sample * x_sample))
print(f"  Sample activations (first 10 values from real network):")
print(f"  {'Input':>10} {'GELU':>12} {'Ideal':>12} {'Diff':>12}")
for i in range(10):
    xi = x_sample[i].item()
    gi = gelu_applied[i].item()
    ii = ideal_applied[i].item()
    print(f"  {xi:>10.4f} {gi:>12.6f} {ii:>12.6f} {gi-ii:>12.8f}")


# ============================================================================
# Step 4: Visualization
# ============================================================================

N_SHOW = 8
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(5, N_SHOW, figure=fig, hspace=0.5, wspace=0.3)

# Row 0: Source images with predictions from BOTH models
all_imgs = []
all_logits_g = []
all_logits_i = []

# Hook into a mid-network stage for feature maps
stage5_gelu = {}
stage5_ideal = {}
stage1_gelu = {}
stage1_ideal = {}

h1 = model_gelu.features[1].register_forward_hook(
    lambda m, i, o: stage1_gelu.update({'feat': o.detach()}))
h2 = model_ideal.features[1].register_forward_hook(
    lambda m, i, o: stage1_ideal.update({'feat': o.detach()}))
h3 = model_gelu.features[5].register_forward_hook(
    lambda m, i, o: stage5_gelu.update({'feat': o.detach()}))
h4 = model_ideal.features[5].register_forward_hook(
    lambda m, i, o: stage5_ideal.update({'feat': o.detach()}))

stage1_feats_g = []
stage1_feats_i = []
stage5_feats_g = []
stage5_feats_i = []

with torch.no_grad():
    for idx in range(N_SHOW):
        img = cv2.imread(images[idx])
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = preprocess(pil_img).unsqueeze(0)
        all_imgs.append(cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))

        lg = model_gelu(t)
        li = model_ideal(t)
        all_logits_g.append(lg)
        all_logits_i.append(li)
        stage1_feats_g.append(stage1_gelu['feat'].clone())
        stage1_feats_i.append(stage1_ideal['feat'].clone())
        stage5_feats_g.append(stage5_gelu['feat'].clone())
        stage5_feats_i.append(stage5_ideal['feat'].clone())

for h in [h1, h2, h3, h4]:
    h.remove()

# Row 0: Images with dual predictions
for i in range(N_SHOW):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(all_imgs[i])
    pg = all_logits_g[i].argmax().item()
    pi = all_logits_i[i].argmax().item()
    cg = F.softmax(all_logits_g[i], dim=1).max().item()
    ci = F.softmax(all_logits_i[i], dim=1).max().item()
    l2 = (all_logits_g[i] - all_logits_i[i]).norm().item()
    match_str = "SAME" if pg == pi else "DIFF!"
    color = 'green' if pg == pi else 'red'
    ax.set_title(f'GELU:{pg} ({cg:.2f})\nIdeal:{pi} ({ci:.2f})\nL2={l2:.3f} [{match_str}]',
                 fontsize=7, color=color)
    ax.axis('off')

# Row 1: GELU feature maps (stage 1, channel mean)
for i in range(N_SHOW):
    ax = fig.add_subplot(gs[1, i])
    feat = stage1_feats_g[i].mean(dim=1).squeeze().numpy()
    ax.imshow(feat, cmap='viridis', aspect='auto')
    if i == 0:
        ax.set_ylabel('GELU\nStage 1', fontsize=9)
    ax.set_title(f'mean={feat.mean():.3f}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

# Row 2: Ideal Gate feature maps (stage 1, channel mean)
for i in range(N_SHOW):
    ax = fig.add_subplot(gs[2, i])
    feat = stage1_feats_i[i].mean(dim=1).squeeze().numpy()
    ax.imshow(feat, cmap='viridis', aspect='auto')
    if i == 0:
        ax.set_ylabel('Ideal Gate\nStage 1', fontsize=9)
    ax.set_title(f'mean={feat.mean():.3f}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

# Row 3: Difference maps (stage 1) — AMPLIFIED to make visible
for i in range(N_SHOW):
    ax = fig.add_subplot(gs[3, i])
    fg = stage1_feats_g[i].mean(dim=1).squeeze().numpy()
    fi = stage1_feats_i[i].mean(dim=1).squeeze().numpy()
    diff = fi - fg
    vmax = max(abs(diff.max()), abs(diff.min()), 1e-6)
    ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    if i == 0:
        ax.set_ylabel('Difference\n(Ideal - GELU)', fontsize=9)
    ax.set_title(f'max={np.abs(diff).max():.5f}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

# Row 4: Stage 5 difference maps (deeper — should be larger)
for i in range(N_SHOW):
    ax = fig.add_subplot(gs[4, i])
    fg = stage5_feats_g[i].mean(dim=1).squeeze().numpy()
    fi = stage5_feats_i[i].mean(dim=1).squeeze().numpy()
    diff = fi - fg
    vmax = max(abs(diff.max()), abs(diff.min()), 1e-6)
    ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    if i == 0:
        ax.set_ylabel('Difference\nStage 5 (deep)', fontsize=9)
    ax.set_title(f'max={np.abs(diff).max():.4f}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('PROOF: Ideal Gate IS Running (not just GELU)\n'
             'Row 1-2: Feature maps from EACH model independently\n'
             'Row 3-4: Differences exist but are tiny (gate is accurate, not identical)',
             fontsize=13, fontweight='bold', y=1.02)

plt.savefig('/tmp/ideal_gate_proof.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print()
print("Saved: /tmp/ideal_gate_proof.png")


# ============================================================================
# Figure 2: Activation distribution comparison
# ============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

# Run through several images to collect gate activations
all_gelu_acts = []
all_ideal_acts = []
gate_g = gelu_gates[5][1]  # pick a mid-network gate
gate_i = ideal_gates[5][1]

with torch.no_grad():
    for idx in range(20):
        img = cv2.imread(images[idx])
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = preprocess(pil_img).unsqueeze(0)
        _ = model_gelu(t)
        _ = model_ideal(t)
        all_gelu_acts.append(gate_g.last_output.flatten().numpy())
        all_ideal_acts.append(gate_i.last_output.flatten().numpy())

gelu_acts = np.concatenate(all_gelu_acts)
ideal_acts = np.concatenate(all_ideal_acts)

# Panel 1: Activation histograms overlaid
ax = axes[0, 0]
bins = np.linspace(min(gelu_acts.min(), ideal_acts.min()),
                   max(gelu_acts.max(), ideal_acts.max()), 200)
ax.hist(gelu_acts, bins=bins, alpha=0.5, color='blue', label='GELU', density=True)
ax.hist(ideal_acts, bins=bins, alpha=0.5, color='red', label='Ideal Gate', density=True)
ax.set_xlabel('Activation value')
ax.set_ylabel('Density')
ax.set_title(f'Gate output distributions\n(gate: {gelu_gates[5][0]})')
ax.legend()

# Panel 2: Scatter of GELU vs Ideal activations (subsample)
ax = axes[0, 1]
n_scatter = min(50000, len(gelu_acts))
idx_sub = np.random.choice(len(gelu_acts), n_scatter, replace=False)
ax.scatter(gelu_acts[idx_sub], ideal_acts[idx_sub], s=0.1, alpha=0.1, c='purple')
lims = [min(gelu_acts.min(), ideal_acts.min()), max(gelu_acts.max(), ideal_acts.max())]
ax.plot(lims, lims, 'k--', linewidth=0.5)
ax.set_xlabel('GELU activation')
ax.set_ylabel('Ideal Gate activation')
ax.set_title('GELU vs Ideal Gate activations\n(should be nearly identical, not exact)')
ax.set_aspect('equal')

# Panel 3: Difference distribution
ax = axes[0, 2]
diff_acts = ideal_acts - gelu_acts
ax.hist(diff_acts, bins=200, color='red', alpha=0.7, density=True)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel('Ideal - GELU')
ax.set_ylabel('Density')
ax.set_title(f'Activation differences\nmean={diff_acts.mean():.6f}, std={diff_acts.std():.6f}')

# Panel 4: The gate functions applied to the actual input range
ax = axes[1, 0]
gate_inputs = np.concatenate([g.last_input.flatten().numpy() for _, g in gelu_gates[:3]])
x_range = np.linspace(np.percentile(gate_inputs, 1), np.percentile(gate_inputs, 99), 1000)
x_t = torch.tensor(x_range, dtype=torch.float32)
gelu_curve = F.gelu(x_t).numpy()
ideal_curve = (x_t * torch.sigmoid(SQRT_8_OVER_PI * x_t * (1 + C_GEOMETRIC * x_t * x_t))).numpy()
ax.plot(x_range, gelu_curve, 'b-', linewidth=2, label='GELU')
ax.plot(x_range, ideal_curve, 'r--', linewidth=1.5, label='Ideal Gate')
ax.hist(gate_inputs, bins=100, alpha=0.15, color='gray', density=True,
        label='Input distribution')
ax.set_xlabel('x (actual network range)')
ax.set_ylabel('gate(x)')
ax.set_title('Gate curves over ACTUAL input range')
ax.legend(fontsize=8)

# Panel 5: Zoomed error over actual input range
ax = axes[1, 1]
err_curve = ideal_curve - gelu_curve
ax.plot(x_range, err_curve * 10000, 'r-', linewidth=1)
ax.fill_between(x_range, err_curve * 10000, 0, alpha=0.3, color='red')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('x (actual network range)')
ax.set_ylabel('Error × 10000')
ax.set_title(f'Error over actual range (×10000)\nmax={np.abs(err_curve).max():.6f}')

# Panel 6: Cumulative logit L2 distances across images
ax = axes[1, 2]
l2_dists = [(g - i).norm().item() for g, i in zip(all_logits_g, all_logits_i)]
ax.bar(range(len(l2_dists)), l2_dists, color='purple', alpha=0.7)
ax.axhline(y=np.mean(l2_dists), color='k', linewidth=1, linestyle='--',
           label=f'mean={np.mean(l2_dists):.3f}')
ax.set_xlabel('Image index')
ax.set_ylabel('L2 distance (logits)')
ax.set_title('Logit L2: GELU vs Ideal Gate\n(>0 proves different, small proves accurate)')
ax.legend()

fig2.suptitle('Ideal Gate Activation Analysis\n'
              'The gate IS different from GELU — but the differences are tiny (max error 0.00075)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/ideal_gate_activations.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: /tmp/ideal_gate_activations.png")

print("\nVerification complete.")
