"""
Colorizer 4-State Gate Investigation (Doc 254)

Phase 17C found "dead" channels (pre_gelu < 0) contribute 31.6% of output energy.
Finding 57 found the transformer MLP gate has 4 states at ±log(φ) boundaries.

Question: Does the SAME 4-state structure appear in the colorizer's GELU gates?
If yes, the negative zero principle is architecture-universal.

4-State decomposition of pre-GELU activations:
  CONTRACT:   pre_gelu < -log(φ) ≈ -0.481
  PRESERVE-:  -log(φ) ≤ pre_gelu < 0
  PRESERVE+:  0 ≤ pre_gelu < +log(φ)
  EXPAND:     pre_gelu ≥ +log(φ)

Measures per block:
  1. What fraction of channels fall in each state?
  2. How much energy does each state contribute to PW2 output?
  3. Does PRESERVE carry disproportionate information (as in transformer)?
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
import json

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)  # ≈ 0.481

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

# ================================================================
# 4-State GELU Decomposition
# ================================================================
print()
print('=' * 80)
print('4-STATE GELU DECOMPOSITION (Doc 254: Negative Zero Cross-Architecture)')
print('=' * 80)
print()
print(f'Boundaries: ±log(φ) = ±{LOG_PHI:.4f}')
print()

N_IMGS = 10
block_stats = {}  # (stage, block) -> {state_name: {frac, energy_pct, ...}}

for img_idx in range(300, 300 + N_IMGS * 3):
    done = any(len(v.get('expand_frac', [])) >= N_IMGS for v in block_stats.values())
    if done:
        break
    im = cv2.imread(images[img_idx])
    if im is None:
        continue

    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean_t) / std_t

    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)

        for stage_idx in range(4):
            dim = dims[stage_idx]
            if stage_idx > 0:
                prefix = f'encoder.arch.downsample_layers.{stage_idx}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[stage_idx-1],),
                                 v16._get_weight(f'{prefix}.0.weight'),
                                 v16._get_weight(f'{prefix}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{prefix}.1.weight'),
                             v16._get_weight(f'{prefix}.1.bias'), stride=2)

            for block_idx in range(depths[stage_idx]):
                residual = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                W1 = v16._get_weight(f'{prefix}.pwconv1.weight')
                b1 = v16._get_weight(f'{prefix}.pwconv1.bias')
                W2 = v16._get_weight(f'{prefix}.pwconv2.weight')
                b2 = v16._get_weight(f'{prefix}.pwconv2.bias')

                pre_gelu = F.linear(xb, W1, b1)  # [1, H, W, 4C]
                post_gelu = geometric_gelu(pre_gelu)

                # 4-state masks (per-element, not per-channel)
                contract_mask = (pre_gelu < -LOG_PHI).float()
                preserve_n_mask = ((pre_gelu >= -LOG_PHI) & (pre_gelu < 0)).float()
                preserve_p_mask = ((pre_gelu >= 0) & (pre_gelu < LOG_PHI)).float()
                expand_mask = (pre_gelu >= LOG_PHI).float()

                # Separate post-GELU by state
                gelu_contract = post_gelu * contract_mask
                gelu_preserve_n = post_gelu * preserve_n_mask
                gelu_preserve_p = post_gelu * preserve_p_mask
                gelu_expand = post_gelu * expand_mask

                # PW2 output from each state
                pw2_contract = F.linear(gelu_contract, W2, None)
                pw2_preserve_n = F.linear(gelu_preserve_n, W2, None)
                pw2_preserve_p = F.linear(gelu_preserve_p, W2, None)
                pw2_expand = F.linear(gelu_expand, W2, None)
                pw2_total = F.linear(post_gelu, W2, b2)

                # Energy from each state
                e_contract = (pw2_contract ** 2).sum().item()
                e_preserve_n = (pw2_preserve_n ** 2).sum().item()
                e_preserve_p = (pw2_preserve_p ** 2).sum().item()
                e_expand = (pw2_expand ** 2).sum().item()
                e_total = e_contract + e_preserve_n + e_preserve_p + e_expand

                # Fractions
                n_elements = pre_gelu.numel()
                frac_contract = contract_mask.sum().item() / n_elements
                frac_preserve_n = preserve_n_mask.sum().item() / n_elements
                frac_preserve_p = preserve_p_mask.sum().item() / n_elements
                frac_expand = expand_mask.sum().item() / n_elements

                # Information density: energy_pct / frac (how much energy per channel)
                key = (stage_idx, block_idx)
                if key not in block_stats:
                    block_stats[key] = {
                        'contract_frac': [], 'preserve_n_frac': [],
                        'preserve_p_frac': [], 'expand_frac': [],
                        'contract_epct': [], 'preserve_n_epct': [],
                        'preserve_p_epct': [], 'expand_epct': [],
                    }

                s = block_stats[key]
                s['contract_frac'].append(frac_contract)
                s['preserve_n_frac'].append(frac_preserve_n)
                s['preserve_p_frac'].append(frac_preserve_p)
                s['expand_frac'].append(frac_expand)
                s['contract_epct'].append(e_contract / (e_total + 1e-10) * 100)
                s['preserve_n_epct'].append(e_preserve_n / (e_total + 1e-10) * 100)
                s['preserve_p_epct'].append(e_preserve_p / (e_total + 1e-10) * 100)
                s['expand_epct'].append(e_expand / (e_total + 1e-10) * 100)

                # Complete block
                xb = pw2_total
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# ================================================================
# Report
# ================================================================
print(f"{'Block':<8} {'CONTRACT':<12} {'PRESERVE-':<12} {'PRESERVE+':<12} {'EXPAND':<12} │ "
      f"{'C E%':<8} {'P- E%':<8} {'P+ E%':<8} {'X E%':<8} │ "
      f"{'C dens':<8} {'P- dens':<8} {'P+ dens':<8} {'X dens':<8}")
print("-" * 140)

# Accumulators for cross-block averages
all_fracs = {'contract': [], 'preserve_n': [], 'preserve_p': [], 'expand': []}
all_epcts = {'contract': [], 'preserve_n': [], 'preserve_p': [], 'expand': []}
all_dens = {'contract': [], 'preserve_n': [], 'preserve_p': [], 'expand': []}

for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        s = block_stats[key]

        fc = np.mean(s['contract_frac'])
        fn = np.mean(s['preserve_n_frac'])
        fp = np.mean(s['preserve_p_frac'])
        fx = np.mean(s['expand_frac'])

        ec = np.mean(s['contract_epct'])
        en = np.mean(s['preserve_n_epct'])
        ep = np.mean(s['preserve_p_epct'])
        ex = np.mean(s['expand_epct'])

        # Information density = energy% / fraction
        dc = ec / (fc * 100 + 1e-10)
        dn = en / (fn * 100 + 1e-10)
        dp = ep / (fp * 100 + 1e-10)
        dx = ex / (fx * 100 + 1e-10)

        all_fracs['contract'].append(fc)
        all_fracs['preserve_n'].append(fn)
        all_fracs['preserve_p'].append(fp)
        all_fracs['expand'].append(fx)
        all_epcts['contract'].append(ec)
        all_epcts['preserve_n'].append(en)
        all_epcts['preserve_p'].append(ep)
        all_epcts['expand'].append(ex)
        all_dens['contract'].append(dc)
        all_dens['preserve_n'].append(dn)
        all_dens['preserve_p'].append(dp)
        all_dens['expand'].append(dx)

        print(f"  {stage_idx}.{block_idx:<5} "
              f"{fc*100:>5.1f}%      {fn*100:>5.1f}%      {fp*100:>5.1f}%      {fx*100:>5.1f}%      │ "
              f"{ec:>5.1f}%  {en:>5.1f}%  {ep:>5.1f}%  {ex:>5.1f}%  │ "
              f"{dc:>6.2f}  {dn:>6.2f}  {dp:>6.2f}  {dx:>6.2f}")

print("-" * 140)

# Summary
print()
print("=" * 80)
print("SUMMARY: Cross-block averages")
print("=" * 80)
print()

states = ['contract', 'preserve_n', 'preserve_p', 'expand']
labels = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

print(f"{'State':<12} {'Fraction':<12} {'Energy%':<12} {'Info Density':<14} {'Density ratio'}")
print("-" * 62)

densities = {}
for state, label in zip(states, labels):
    frac = np.mean(all_fracs[state]) * 100
    epct = np.mean(all_epcts[state])
    dens = np.mean(all_dens[state])
    densities[state] = dens
    print(f"  {label:<10} {frac:>6.1f}%      {epct:>6.1f}%      {dens:>6.3f}")

print()

# The key question: does PRESERVE have higher info density than EXPAND/CONTRACT?
preserve_dens = (densities['preserve_n'] + densities['preserve_p']) / 2
boundary_dens = (densities['contract'] + densities['expand']) / 2
ratio = preserve_dens / (boundary_dens + 1e-10)

print(f"  PRESERVE avg density:  {preserve_dens:.3f}")
print(f"  BOUNDARY avg density:  {boundary_dens:.3f}")
print(f"  PRESERVE / BOUNDARY:   {ratio:.2f}×")
print()

if ratio > 1.5:
    print("  ★ PRESERVE carries disproportionate information — 4-state structure CONFIRMED")
    print("  ★ The negative zero principle is architecture-universal (transformer + ConvNeXt)")
elif ratio > 1.0:
    print("  → PRESERVE is slightly more informative — weak 4-state signal")
else:
    print("  → No 4-state signal — GELU gate may work differently from SiLU gate")

# Phase 17C comparison
print()
print("=" * 80)
print("COMPARISON WITH PHASE 17C (2-state decomposition)")
print("=" * 80)
print()
dead_epct = np.mean(all_epcts['contract']) + np.mean(all_epcts['preserve_n'])
alive_epct = np.mean(all_epcts['preserve_p']) + np.mean(all_epcts['expand'])
print(f"  Dead energy (CONTRACT + PRESERVE-):  {dead_epct:.1f}%  (Phase 17C found: 31.6%)")
print(f"  Alive energy (PRESERVE+ + EXPAND):   {alive_epct:.1f}%")
print()
print(f"  But within 'dead', CONTRACT vs PRESERVE- split:")
print(f"    CONTRACT:   {np.mean(all_epcts['contract']):.1f}%")
print(f"    PRESERVE-:  {np.mean(all_epcts['preserve_n']):.1f}%")
print(f"  And within 'alive', PRESERVE+ vs EXPAND split:")
print(f"    PRESERVE+:  {np.mean(all_epcts['preserve_p']):.1f}%")
print(f"    EXPAND:     {np.mean(all_epcts['expand']):.1f}%")

# Save results
results = {
    'boundary': float(LOG_PHI),
    'n_images': N_IMGS,
    'summary': {
        state: {
            'fraction_pct': float(np.mean(all_fracs[state]) * 100),
            'energy_pct': float(np.mean(all_epcts[state])),
            'info_density': float(np.mean(all_dens[state])),
        }
        for state in states
    },
    'preserve_boundary_ratio': float(ratio),
    'dead_energy_pct': float(dead_epct),
}

out_path = '/home/thorin/truthspace-lcm/experiments/model_reverse_engineering_v2/results/colorizer_4state.json'
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
