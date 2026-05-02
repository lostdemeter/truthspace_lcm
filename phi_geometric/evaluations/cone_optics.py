#!/usr/bin/env python3
"""
Cone Optics: Treating the Validity Cone as Light

The validity cone IS a light cone. We can apply photographic and optical
techniques to it:

1. HDR ENSEMBLE: Multiple "exposures" at different λ, merged for sharper output
2. GAUSSIAN FOCUSING: Weight samples by φ-scaled Gaussians (dimensional downcasting)
3. APERTURE CONTROL: Vary how many samples to combine (f-stop = #samples)
4. BRACKET EXPOSURE: Sample at λ = {-2, -1, 0, 1, 2} and merge
5. DEPTH OF FIELD: Focus on specific blocks (shallow vs deep DOF)
6. POLARIZATION: Filter by direction in the cone (SVD components)

The dimensional downcasting connection:
- Traditional: 2D → 3D (upcast via radiance field, requires training)
- Our approach: ConeD → 1D (downcast via moment projection, pure math)
- Non-uniform Gaussians with φ-scaled widths act as the focusing lens
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
    def __init__(self, lam=0.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        f = SQRT_8_OVER_PI * x * (1.0 + C_GEOMETRIC * x * x)
        ideal = x * torch.sigmoid(f)
        gelu = F.gelu(x)
        return (1 - self.lam) * ideal + self.lam * gelu


def build_steerable_model(lambdas):
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


def get_logits(lambdas, tensor):
    """Get logits for a single image with given λ profile."""
    model, _ = build_steerable_model(lambdas)
    with torch.no_grad():
        return model(tensor)


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
ideal_logits = []
ideal_confs = []
model_ideal, _ = build_steerable_model([0.0] * N_BLOCKS)
for idx in range(N_IMAGES):
    with torch.no_grad():
        logits = model_ideal(tensors[idx])
    ideal_logits.append(logits)
    ideal_confs.append(F.softmax(logits, dim=1).max().item())

print(f"Setup: {N_IMAGES} images, {N_BLOCKS} blocks")
print(f"GELU mean confidence:  {np.mean(ref_confs):.6f}")
print(f"Ideal mean confidence: {np.mean(ideal_confs):.6f}")


# ============================================================================
# 1. HDR ENSEMBLE: Average multiple "exposures" at different λ
# ============================================================================

print()
print("=" * 80)
print("1. HDR ENSEMBLE: Multiple exposures merged for sharper output")
print("=" * 80)
print()

# Like HDR photography: take multiple shots at different exposures, merge
# Each λ is an "exposure setting"

def hdr_merge(lambdas_list, tensors, ref_preds, weights=None):
    """Merge logits from multiple λ profiles (exposures)."""
    if weights is None:
        weights = [1.0 / len(lambdas_list)] * len(lambdas_list)

    results = {'confs': [], 'agree': 0, 'margins': [], 'entropy': []}
    for idx in range(len(tensors)):
        merged_logits = torch.zeros(1, 1000)
        for lam_vec, w in zip(lambdas_list, weights):
            logits = get_logits(lam_vec, tensors[idx])
            merged_logits += w * logits
        probs = F.softmax(merged_logits, dim=1)
        results['confs'].append(probs.max().item())
        results['entropy'].append(-(probs * (probs + 1e-10).log()).sum().item())
        if merged_logits.argmax(1).item() == ref_preds[idx]:
            results['agree'] += 1
        sorted_vals = merged_logits.sort(dim=1, descending=True).values
        results['margins'].append((sorted_vals[0, 0] - sorted_vals[0, 1]).item())
    return results

# Exposure brackets — like photographing at -2EV, -1EV, 0EV, +1EV, +2EV
brackets = {
    '1-stop (λ=1 only)': [[1.0] * N_BLOCKS],
    '3-stop [-1,0,1]': [[-1.0]*N_BLOCKS, [0.0]*N_BLOCKS, [1.0]*N_BLOCKS],
    '5-stop [-2..2]': [[l]*N_BLOCKS for l in [-2, -1, 0, 1, 2]],
    '7-stop [-3..3]': [[l]*N_BLOCKS for l in [-3, -2, -1, 0, 1, 2, 3]],
    '9-stop [-4..4]': [[l]*N_BLOCKS for l in [-4, -3, -2, -1, 0, 1, 2, 3, 4]],
}

hdr_results = {}
for name, lam_list in brackets.items():
    r = hdr_merge(lam_list, tensors, ref_preds)
    hdr_results[name] = r
    print(f"  {name:<25}: conf={np.mean(r['confs']):.6f}  "
          f"margin={np.mean(r['margins']):.4f}  "
          f"entropy={np.mean(r['entropy']):.4f}  "
          f"agree={r['agree']}/{N_IMAGES}")

# Compare to single-shot baselines
print(f"\n  Single-shot baselines:")
print(f"    GELU (λ=1):  conf={np.mean(ref_confs):.6f}")
print(f"    Ideal (λ=0): conf={np.mean(ideal_confs):.6f}")


# ============================================================================
# 2. GAUSSIAN FOCUSING: φ-scaled Gaussian weights (dimensional downcasting)
# ============================================================================

print()
print("=" * 80)
print("2. GAUSSIAN FOCUSING: φ-scaled lens (dimensional downcasting)")
print("=" * 80)
print()

# Like a lens: Gaussian weights centered at different focal points
# The σ scales as φ^k (from the downcasting paper's moment hierarchy)

def gaussian_weight(lam, center, sigma):
    return np.exp(-0.5 * ((lam - center) / sigma) ** 2)

# Sample points across the cone
sample_lambdas = np.arange(-4, 5, 0.5)
n_samples = len(sample_lambdas)

# Pre-compute all single-λ logits for efficiency
print(f"  Pre-computing {n_samples} × {N_IMAGES} logits...")
all_logits = {}  # (lam_idx, img_idx) -> logits
for li, lam in enumerate(sample_lambdas):
    model_l, _ = build_steerable_model([lam] * N_BLOCKS)
    for idx in range(N_IMAGES):
        with torch.no_grad():
            all_logits[(li, idx)] = model_l(tensors[idx])

# Focal points and φ-scaled apertures
focal_points = [0.0, 0.5, 1.0]  # Focus on Ideal, midpoint, GELU
phi_sigmas = [PHI**k for k in range(-2, 4)]  # φ^-2 to φ^3

focus_results = {}
for center in focal_points:
    for sigma in phi_sigmas:
        # Compute Gaussian weights
        weights = np.array([gaussian_weight(l, center, sigma) for l in sample_lambdas])
        weights /= weights.sum()

        confs = []
        agree = 0
        margins = []
        for idx in range(N_IMAGES):
            merged = torch.zeros(1, 1000)
            for li in range(n_samples):
                merged += weights[li] * all_logits[(li, idx)]
            probs = F.softmax(merged, dim=1)
            confs.append(probs.max().item())
            if merged.argmax(1).item() == ref_preds[idx]:
                agree += 1
            sv = merged.sort(dim=1, descending=True).values
            margins.append((sv[0, 0] - sv[0, 1]).item())

        key = f"center={center:.1f}, σ=φ^{np.log(sigma)/np.log(PHI):.0f}={sigma:.3f}"
        focus_results[key] = {
            'center': center, 'sigma': sigma,
            'conf': np.mean(confs), 'agree': agree,
            'margin': np.mean(margins)
        }

print(f"\n  Gaussian focus results (center, σ → confidence, margin, agreement):")
print(f"  {'Setting':<40} {'Conf':>8} {'Margin':>8} {'Agree':>6}")
print(f"  {'-'*65}")
for key, r in focus_results.items():
    marker = " ★" if r['conf'] > np.mean(ref_confs) else ""
    print(f"  {key:<40} {r['conf']:.6f} {r['margin']:.4f} {r['agree']:>3}/{N_IMAGES}{marker}")

best_focus = max(focus_results.values(), key=lambda r: r['conf'] if r['agree'] == N_IMAGES else 0)
print(f"\n  Best focused: conf={best_focus['conf']:.6f} "
      f"(center={best_focus['center']:.1f}, σ={best_focus['sigma']:.3f})")
print(f"  GELU baseline: conf={np.mean(ref_confs):.6f}")
print(f"  Improvement: {(best_focus['conf'] - np.mean(ref_confs))*100:+.4f}%")


# ============================================================================
# 3. APERTURE CONTROL: f-stop = number of samples
# ============================================================================

print()
print("=" * 80)
print("3. APERTURE CONTROL: f-stop = number of samples")
print("=" * 80)
print()

# Like camera aperture: wider = more light (more samples), shallower DOF
# Narrower = less light (fewer samples), deeper DOF

aperture_results = []
for n_exp in [1, 2, 3, 5, 7, 9, 13, 17]:
    # Evenly spaced exposures centered on the cone
    if n_exp == 1:
        lam_list = [0.0]  # Single shot at Ideal (best single confidence)
    else:
        lam_list = np.linspace(-3, 3, n_exp).tolist()

    confs = []
    agree = 0
    for idx in range(N_IMAGES):
        merged = torch.zeros(1, 1000)
        for lam in lam_list:
            # Find nearest precomputed
            li = np.argmin(np.abs(sample_lambdas - lam))
            merged += all_logits[(li, idx)]
        merged /= len(lam_list)
        probs = F.softmax(merged, dim=1)
        confs.append(probs.max().item())
        if merged.argmax(1).item() == ref_preds[idx]:
            agree += 1

    aperture_results.append({
        'n': n_exp, 'conf': np.mean(confs), 'agree': agree
    })
    print(f"  f/{n_exp:<2} ({n_exp:>2} samples): conf={np.mean(confs):.6f}  agree={agree}/{N_IMAGES}")


# ============================================================================
# 4. POLARIZATION: Filter by SVD direction in the cone
# ============================================================================

print()
print("=" * 80)
print("4. POLARIZATION: Filter by principal cone directions")
print("=" * 80)
print()

# Collect logit vectors across λ for a single image to find the
# principal "polarization" directions of the cone
test_img = 0
logit_matrix = torch.stack([all_logits[(li, test_img)].flatten()
                           for li in range(n_samples)])
# Center
mean_logits = logit_matrix.mean(dim=0, keepdim=True)
centered = logit_matrix - mean_logits

# SVD — principal polarization directions
U_pol, S_pol, Vh_pol = torch.linalg.svd(centered, full_matrices=False)
S_pol_np = S_pol.numpy()
total_var = (S_pol_np ** 2).sum()
cum_var = np.cumsum(S_pol_np ** 2) / total_var

print(f"  Polarization analysis (image {test_img}):")
for i in range(min(8, len(S_pol_np))):
    print(f"    Component {i}: σ={S_pol_np[i]:.4f}  "
          f"(cumulative: {cum_var[i]*100:.1f}%)")

pol_rank = np.searchsorted(cum_var, 0.99) + 1
print(f"\n  99% variance in {pol_rank} components")
print(f"  → Cone light is {pol_rank}-dimensional")

# φ-ratios in polarization singular values?
pol_ratios = S_pol_np[:-1] / (S_pol_np[1:] + 1e-12)
print(f"\n  S[i]/S[i+1] ratios: {', '.join(f'{r:.3f}' for r in pol_ratios[:6])}")
print(f"  φ = {PHI:.3f}")

# Reconstruct with different numbers of polarization components
print(f"\n  Polarization filtering (keep K components):")
for K in [1, 2, 3, 5, 8, n_samples]:
    confs_k = []
    agree_k = 0
    for idx in range(N_IMAGES):
        lm = torch.stack([all_logits[(li, idx)].flatten() for li in range(n_samples)])
        mn = lm.mean(dim=0, keepdim=True)
        ct = lm - mn
        Uk, Sk, Vhk = torch.linalg.svd(ct, full_matrices=False)
        # Reconstruct with K components
        recon = mn + (Uk[:, :K] @ torch.diag(Sk[:K]) @ Vhk[:K]).mean(dim=0, keepdim=True)
        probs = F.softmax(recon, dim=1)
        confs_k.append(probs.max().item())
        if recon.argmax(1).item() == ref_preds[idx]:
            agree_k += 1

    k_label = "all" if K == n_samples else str(K)
    print(f"    K={k_label:<3}: conf={np.mean(confs_k):.6f}  agree={agree_k}/{N_IMAGES}")


# ============================================================================
# 5. DEPTH OF FIELD: Focus specific blocks while averaging others
# ============================================================================

print()
print("=" * 80)
print("5. DEPTH OF FIELD: Focus specific blocks")
print("=" * 80)
print()

# Like camera DOF: focus on certain blocks (sharp), blur others (average)
# Focused blocks get λ=optimal, blurred blocks get averaged

# Identify the "focus plane" — which blocks to keep sharp
# From Part 23: critical blocks are [12, 16, 10, 5, 13, 8]

# DOF experiment: vary which blocks are "in focus" (single λ)
# and which are "blurred" (averaged over multiple λ)
dof_results = []
for n_focused in [0, 1, 3, 6, 9, 12, 15, 18]:
    # Top n_focused blocks (by importance from Part 23) get λ=-1 (best confidence)
    # Remaining blocks get averaged over [-3, -1, 0, 1, 3]
    focus_blocks = [12, 16, 10, 5, 13, 8, 14, 7, 9, 3, 2, 17, 11, 4, 6, 1, 0, 15][:n_focused]

    confs = []
    agree = 0
    for idx in range(N_IMAGES):
        # Average over blur lambda values for unfocused blocks
        blur_lambdas = [-3, -1, 0, 1, 3]
        merged = torch.zeros(1, 1000)
        for blur_lam in blur_lambdas:
            lam_vec = [0.0] * N_BLOCKS
            for k in range(N_BLOCKS):
                if k in focus_blocks:
                    lam_vec[k] = -1.0  # focused: Ideal direction (max confidence)
                else:
                    lam_vec[k] = blur_lam  # blurred: varies
            logits = get_logits(lam_vec, tensors[idx])
            merged += logits
        merged /= len(blur_lambdas)

        probs = F.softmax(merged, dim=1)
        confs.append(probs.max().item())
        if merged.argmax(1).item() == ref_preds[idx]:
            agree += 1

    dof_results.append({
        'n_focused': n_focused, 'conf': np.mean(confs), 'agree': agree
    })
    print(f"  DOF {n_focused:>2} blocks focused: conf={np.mean(confs):.6f}  agree={agree}/{N_IMAGES}")


# ============================================================================
# 6. CHROMATIC ANALYSIS: Does each block contribute a different "color"?
# ============================================================================

print()
print("=" * 80)
print("6. CHROMATIC ANALYSIS: Block-by-block spectral decomposition")
print("=" * 80)
print()

# Each block's contribution to the logit change is like a spectral band
# Is the cone "white light" (uniform spectrum) or structured?

block_spectra = []
for k in range(N_BLOCKS):
    # Contribution of block k: logits with block k at λ=1 minus all-ideal
    lam_single = [0.0] * N_BLOCKS
    lam_single[k] = 1.0

    contributions = []
    for idx in range(min(10, N_IMAGES)):
        logits_k = get_logits(lam_single, tensors[idx])
        delta = (logits_k - ideal_logits[idx]).flatten()
        contributions.append(delta)

    stacked = torch.stack(contributions)
    # "Energy" in each logit dimension
    energy = (stacked ** 2).mean(dim=0).numpy()
    block_spectra.append(energy)

block_spectra = np.array(block_spectra)  # [N_BLOCKS, 1000]

# SVD of the spectral matrix
U_spec, S_spec, Vh_spec = np.linalg.svd(block_spectra, full_matrices=False)
total_spec_E = (S_spec ** 2).sum()
cum_spec_E = np.cumsum(S_spec ** 2) / total_spec_E

print(f"  Spectral decomposition of block contributions:")
for i in range(min(6, len(S_spec))):
    print(f"    Band {i}: σ={S_spec[i]:.6f}  (cumulative: {cum_spec_E[i]*100:.1f}%)")

spec_rank = np.searchsorted(cum_spec_E, 0.95) + 1
print(f"\n  95% of spectral energy in {spec_rank} bands")
print(f"  → The cone's 'light' has {spec_rank} spectral components")

# Is the spectrum φ-structured?
spec_ratios = S_spec[:-1] / (S_spec[1:] + 1e-12)
print(f"\n  σ[i]/σ[i+1] ratios: {', '.join(f'{r:.3f}' for r in spec_ratios[:6])}")
print(f"  φ = {PHI:.3f}, φ² = {PHI**2:.3f}")

# Which blocks have the most "spectral power"?
block_power = block_spectra.sum(axis=1)
block_order = np.argsort(block_power)[::-1]
print(f"\n  Block spectral power ranking:")
for i, k in enumerate(block_order[:6]):
    print(f"    #{i+1}: Block {k} → power={block_power[k]:.6f}")


# ============================================================================
# Visualization
# ============================================================================

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: HDR Bracket results
ax1 = fig.add_subplot(gs[0, 0])
bracket_names = list(hdr_results.keys())
bracket_confs = [np.mean(hdr_results[n]['confs']) for n in bracket_names]
bracket_colors = ['green' if c > np.mean(ref_confs) else 'gray' for c in bracket_confs]
bars1 = ax1.barh(range(len(bracket_names)), bracket_confs, color=bracket_colors,
                 edgecolor='black', linewidth=0.5)
ax1.axvline(x=np.mean(ref_confs), color='red', linestyle='--', linewidth=1.5,
            label=f'GELU={np.mean(ref_confs):.4f}')
ax1.set_yticks(range(len(bracket_names)))
ax1.set_yticklabels([n.split(' ')[0] for n in bracket_names], fontsize=8)
ax1.set_xlabel('Mean Confidence')
ax1.set_title('HDR Bracket Exposures')
ax1.legend(fontsize=8)

# Panel 2: Gaussian focus heatmap
ax2 = fig.add_subplot(gs[0, 1])
centers = sorted(set(r['center'] for r in focus_results.values()))
sigmas = sorted(set(r['sigma'] for r in focus_results.values()))
conf_grid = np.zeros((len(sigmas), len(centers)))
for r in focus_results.values():
    ci = centers.index(r['center'])
    si = sigmas.index(r['sigma'])
    conf_grid[si, ci] = r['conf']
im2 = ax2.imshow(conf_grid, cmap='RdYlGn', aspect='auto',
                 extent=[centers[0]-0.25, centers[-1]+0.25,
                         len(sigmas)-0.5, -0.5])
ax2.set_xticks(range(len(centers)))
ax2.set_xticklabels([f'{c:.1f}' for c in centers])
ax2.set_yticks(range(len(sigmas)))
ax2.set_yticklabels([f'φ^{np.log(s)/np.log(PHI):.0f}' for s in sigmas], fontsize=8)
ax2.set_xlabel('Focal center (λ)')
ax2.set_ylabel('Aperture σ')
ax2.set_title('Gaussian Focus Map\n(confidence by center × aperture)')
plt.colorbar(im2, ax=ax2, fraction=0.046)

# Panel 3: Aperture control
ax3 = fig.add_subplot(gs[0, 2])
ns = [r['n'] for r in aperture_results]
aconfs = [r['conf'] for r in aperture_results]
ax3.plot(ns, aconfs, 'bo-', linewidth=2, markersize=8)
ax3.axhline(y=np.mean(ref_confs), color='red', linestyle='--',
            label=f'GELU={np.mean(ref_confs):.4f}')
ax3.set_xlabel('Number of exposures (f-stop)')
ax3.set_ylabel('Mean Confidence')
ax3.set_title('Aperture Control\n(more samples = wider aperture)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Polarization SVD
ax4 = fig.add_subplot(gs[0, 3])
ax4.bar(range(min(12, len(S_pol_np))), S_pol_np[:12], color='purple', alpha=0.7)
ax4.set_xlabel('Polarization component')
ax4.set_ylabel('Singular value')
ax4.set_title(f'Cone Polarization\n(99% in {pol_rank} components)')
ax4.grid(True, alpha=0.3)

# Panel 5: Depth of field
ax5 = fig.add_subplot(gs[1, 0])
dof_ns = [r['n_focused'] for r in dof_results]
dof_confs = [r['conf'] for r in dof_results]
ax5.plot(dof_ns, dof_confs, 'go-', linewidth=2, markersize=8)
ax5.axhline(y=np.mean(ref_confs), color='red', linestyle='--',
            label=f'GELU={np.mean(ref_confs):.4f}')
ax5.set_xlabel('# blocks in focus')
ax5.set_ylabel('Mean Confidence')
ax5.set_title('Depth of Field\n(focused blocks vs blurred)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Panel 6: Block spectral power
ax6 = fig.add_subplot(gs[1, 1])
ax6.bar(range(N_BLOCKS), block_power, color='orange', alpha=0.7,
        edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Block')
ax6.set_ylabel('Spectral power')
ax6.set_title('Block "Color" Power\n(contribution to logit change)')
ax6.grid(True, alpha=0.3)

# Panel 7: Spectral SVD
ax7 = fig.add_subplot(gs[1, 2])
ax7.bar(range(min(10, len(S_spec))), S_spec[:10], color='teal', alpha=0.7)
ax7.set_xlabel('Spectral band')
ax7.set_ylabel('Singular value')
ax7.set_title(f'Cone Spectrum\n(95% in {spec_rank} bands)')
ax7.grid(True, alpha=0.3)

# Panel 8: Per-image HDR benefit
ax8 = fig.add_subplot(gs[1, 3])
# Compare single-shot GELU vs best HDR
best_hdr_key = max(hdr_results.keys(), key=lambda k: np.mean(hdr_results[k]['confs']))
best_hdr = hdr_results[best_hdr_key]
x_idx = np.arange(N_IMAGES)
ax8.scatter(x_idx, ref_confs, c='red', s=30, label='GELU (single)', zorder=5)
ax8.scatter(x_idx, best_hdr['confs'], c='green', s=30, marker='*', label=f'HDR ({best_hdr_key.split(" ")[0]})', zorder=5)
for i in x_idx:
    color = 'green' if best_hdr['confs'][i] > ref_confs[i] else 'red'
    ax8.plot([i, i], [ref_confs[i], best_hdr['confs'][i]], color=color, alpha=0.3)
ax8.set_xlabel('Image')
ax8.set_ylabel('Confidence')
ax8.set_title('Per-Image: Single vs HDR')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Panel 9: The Optics Diagram
ax9 = fig.add_subplot(gs[2, 0:2])
ax9.set_xlim(-1, 20)
ax9.set_ylim(-0.4, 0.4)

# Draw the cone
cone_x = np.linspace(0, 18, 200)
cone_r = 0.015 * cone_x
ax9.fill_between(cone_x, -cone_r, cone_r, alpha=0.08, color='gold', label='Cone')

# Light rays (different λ paths)
for lam_offset in np.linspace(-0.8, 0.8, 9):
    ray_y = lam_offset * cone_r
    alpha = 0.15 + 0.1 * np.exp(-lam_offset**2)
    ax9.plot(cone_x, ray_y, '-', color='orange', alpha=alpha, linewidth=0.5)

# Gaussian lens at a focal point
lens_x = 9
lens_height = 0.015 * lens_x * 1.2
ax9.plot([lens_x, lens_x], [-lens_height, lens_height], 'b-', linewidth=3, alpha=0.7)
# Lens curvature
theta = np.linspace(-np.pi/2, np.pi/2, 50)
lens_curve = 0.3 * np.cos(theta)
ax9.plot(lens_x + lens_curve, lens_height * np.sin(theta), 'b-', linewidth=2, alpha=0.5)
ax9.text(lens_x, -lens_height - 0.04, 'φ-Gaussian\nlens', ha='center', fontsize=8, color='blue')

# Focused output
focus_x = np.linspace(lens_x, 18, 100)
for lam_offset in np.linspace(-0.8, 0.8, 9):
    start_y = lam_offset * 0.015 * lens_x
    end_y = start_y * 0.2  # converging
    ray_y = np.linspace(start_y, end_y, len(focus_x))
    ax9.plot(focus_x, ray_y, '-', color='green', alpha=0.3, linewidth=0.5)

# Focal point
ax9.plot(18, 0, 'r*', markersize=15, zorder=10, label='Focused output')

ax9.set_xlabel('Network depth (blocks)')
ax9.set_ylabel('Position in cone')
ax9.set_title('Cone Optics: Gaussian Focusing\n"Treat the cone like light, focus it with a φ-lens"',
              fontsize=11)
ax9.legend(fontsize=8, loc='upper left')
ax9.grid(True, alpha=0.2)

# Panel 10: HDR vs Single comparison (margins)
ax10 = fig.add_subplot(gs[2, 2])
methods = ['GELU\n(single)', 'Ideal\n(single)', '3-stop\nHDR', '5-stop\nHDR', '9-stop\nHDR']
method_margins = [
    np.mean([ref_logits[i].sort(dim=1, descending=True).values[0, 0].item() -
             ref_logits[i].sort(dim=1, descending=True).values[0, 1].item() for i in range(N_IMAGES)]),
    np.mean([ideal_logits[i].sort(dim=1, descending=True).values[0, 0].item() -
             ideal_logits[i].sort(dim=1, descending=True).values[0, 1].item() for i in range(N_IMAGES)]),
    np.mean(hdr_results['3-stop [-1,0,1]']['margins']),
    np.mean(hdr_results['5-stop [-2..2]']['margins']),
    np.mean(hdr_results['9-stop [-4..4]']['margins']),
]
colors_m = ['red', 'blue', 'green', 'darkgreen', 'darkgreen']
ax10.bar(range(len(methods)), method_margins, color=colors_m, alpha=0.7,
         edgecolor='black', linewidth=0.5)
ax10.set_xticks(range(len(methods)))
ax10.set_xticklabels(methods, fontsize=8)
ax10.set_ylabel('Mean classification margin')
ax10.set_title('Classification Margin\nSingle Shot vs HDR')
ax10.grid(True, alpha=0.3)

# Panel 11: Summary
ax11 = fig.add_subplot(gs[2, 3])
ax11.axis('off')

best_hdr_conf = max(np.mean(r['confs']) for r in hdr_results.values())
best_focus_conf = best_focus['conf']
best_overall = max(best_hdr_conf, best_focus_conf)

summary = (
    f"CONE OPTICS RESULTS\n"
    f"{'─' * 30}\n\n"
    f"HDR Ensembles:\n"
    f"  Best bracket: {best_hdr_key.split(' ')[0]}\n"
    f"  Conf: {best_hdr_conf:.6f}\n\n"
    f"Gaussian Focus:\n"
    f"  Best: center={best_focus['center']:.1f}\n"
    f"        σ={best_focus['sigma']:.3f}\n"
    f"  Conf: {best_focus_conf:.6f}\n\n"
    f"Baselines:\n"
    f"  GELU:  {np.mean(ref_confs):.6f}\n"
    f"  Ideal: {np.mean(ideal_confs):.6f}\n\n"
    f"Best optical: {best_overall:.6f}\n"
    f"  Δ GELU: {(best_overall-np.mean(ref_confs))*100:+.4f}%\n\n"
    f"Cone dimensions:\n"
    f"  Polarization: {pol_rank}\n"
    f"  Spectral bands: {spec_rank}\n"
    f"  φ in ratios: "
    f"{'YES' if any(abs(r-PHI)<0.15 for r in pol_ratios[:4]) else 'weak'}"
)
ax11.text(0.05, 0.95, summary, transform=ax11.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Cone Optics: Treating the Validity Cone as Light\n'
             '"Focus it, bracket it, decompose its spectrum — it behaves like light"',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/tmp/cone_optics.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print()
print("Saved: /tmp/cone_optics.png")
