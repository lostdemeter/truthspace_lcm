#!/usr/bin/env python3
"""
V20 4th Dimension Trace

The insight: there's a 4th dimension (color/ab) latent in luminance structure.
The encoder NAVIGATES through possibility space, and each gate SELECTS
which possibilities survive. Layer by layer, possibilities narrow until
only truth remains.

The 5% desaturation means the φ-soft gate is slightly under-committing.
This traces WHERE the 4th dimension gets attenuated by comparing the
GELU and φ-soft gate selection patterns block by block.

Key measurements at each block:
  1. Gate value distributions (GELU vs φ-soft)
  2. Feature divergence (cosine sim between GELU and φ-soft features)
  3. Feature magnitude ratio (how much energy is being scaled)
  4. "Selection strength" — how decisive the gate is (variance of gate values)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sys
from pathlib import Path

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')

from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI

DIMS = [96, 192, 384, 768]
DEPTHS = [3, 3, 9, 3]


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def gelu_gate(z):
    """The gate VALUE of GELU: g(z) = 0.5 * (1 + erf(z/√2))"""
    return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

def phi_soft(x):
    return INV_PHI * x * torch.sigmoid(PHI * x)

def phi_soft_gate(z):
    """The gate VALUE of φ-soft: g(z) = (1/φ) * σ(φ·z)"""
    return INV_PHI * torch.sigmoid(PHI * z)


def trace_4th_dimension(n_images=5):
    """Trace the 4th dimension through GELU and φ-soft pipelines simultaneously."""

    weights_path = Path(__file__).parent / 'ddcolor_weights_static.npz'
    weights = np.load(weights_path)

    def _w(name):
        return torch.from_numpy(weights[name]).float()

    import glob
    images = sorted(glob.glob(
        '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

    # Accumulate stats across images
    block_stats = {}
    total_blocks = sum(DEPTHS)  # 18

    for img_idx in range(300, 300 + n_images * 2):
        im = cv2.imread(images[img_idx])
        if im is None:
            continue

        r = cv2.resize(im, (256, 256))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        with torch.no_grad():
            x = (t - mean) / std

            # Stem (shared)
            x = F.conv2d(x, _w('encoder.arch.downsample_layers.0.0.weight'),
                         _w('encoder.arch.downsample_layers.0.0.bias'), stride=4)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (96,),
                             _w('encoder.arch.downsample_layers.0.1.weight'),
                             _w('encoder.arch.downsample_layers.0.1.bias'))
            x = x.permute(0, 3, 1, 2)

            # Two parallel streams
            x_gelu = x.clone()
            x_phi = x.clone()

            block_count = 0
            for stage_idx in range(4):
                dim = DIMS[stage_idx]
                if stage_idx > 0:
                    prefix = f'encoder.arch.downsample_layers.{stage_idx}'
                    for x_stream in ['gelu', 'phi']:
                        xr = x_gelu if x_stream == 'gelu' else x_phi
                        xr = xr.permute(0, 2, 3, 1)
                        xr = F.layer_norm(xr, (DIMS[stage_idx-1],),
                                         _w(f'{prefix}.0.weight'),
                                         _w(f'{prefix}.0.bias'))
                        xr = xr.permute(0, 3, 1, 2)
                        xr = F.conv2d(xr, _w(f'{prefix}.1.weight'),
                                     _w(f'{prefix}.1.bias'), stride=2)
                        if x_stream == 'gelu':
                            x_gelu = xr
                        else:
                            x_phi = xr

                for block_idx in range(DEPTHS[stage_idx]):
                    prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
                    key = f'{stage_idx}.{block_idx}'

                    residual_gelu = x_gelu.clone()
                    residual_phi = x_phi.clone()

                    # DW conv (shared weights, but different inputs)
                    for x_stream in ['gelu', 'phi']:
                        xr = x_gelu if x_stream == 'gelu' else x_phi
                        xb = F.conv2d(xr, _w(f'{prefix}.dwconv.weight'),
                                     _w(f'{prefix}.dwconv.bias'),
                                     padding=3, groups=dim)
                        xb = xb.permute(0, 2, 3, 1)
                        xb = F.layer_norm(xb, (dim,),
                                         _w(f'{prefix}.norm.weight'),
                                         _w(f'{prefix}.norm.bias'))

                        # PW1
                        z = F.linear(xb,
                                    _w(f'{prefix}.pwconv1.weight'),
                                    _w(f'{prefix}.pwconv1.bias'))

                        if x_stream == 'gelu':
                            z_gelu = z
                        else:
                            z_phi = z

                    # Now z_gelu and z_phi are the pre-activation values
                    # They should be similar (since inputs diverge slowly)

                    # Compute GATE values (not the output, the GATE itself)
                    g_gelu_vals = gelu_gate(z_gelu)
                    g_phi_vals = phi_soft_gate(z_phi)

                    # Compute GATED outputs
                    post_gelu = gelu(z_gelu)
                    post_phi = phi_soft(z_phi)

                    # Selection strength: how decisive is the gate?
                    # High variance = decisive (some channels fully on, some fully off)
                    # Low variance = hedging (everything at middle values)
                    gelu_decisiveness = g_gelu_vals.var().item()
                    phi_decisiveness = g_phi_vals.var().item()

                    # Gate statistics
                    gelu_gate_mean = g_gelu_vals.mean().item()
                    phi_gate_mean = g_phi_vals.mean().item()
                    gelu_gate_max = g_gelu_vals.max().item()
                    phi_gate_max = g_phi_vals.max().item()

                    # How much of the input energy survives the gate?
                    # Energy ratio = ||gated_output||² / ||pre_gelu_input||²
                    z_energy = (z_gelu ** 2).mean().item()
                    gelu_energy = (post_gelu ** 2).mean().item()
                    phi_energy = (post_phi ** 2).mean().item()

                    gelu_survival = gelu_energy / (z_energy + 1e-10)
                    phi_survival = phi_energy / (z_energy + 1e-10)

                    # Feature divergence: cosine similarity of post-gate features
                    cos_sim = F.cosine_similarity(
                        post_gelu.reshape(-1).unsqueeze(0),
                        post_phi.reshape(-1).unsqueeze(0)
                    ).item()

                    # Magnitude ratio
                    mag_ratio = post_phi.abs().mean().item() / (post_gelu.abs().mean().item() + 1e-10)

                    # Fraction of channels where gate > 0.9 (fully committed)
                    gelu_committed = (g_gelu_vals > 0.9).float().mean().item()
                    phi_committed = (g_phi_vals > 0.9).float().mean().item()

                    # Fraction of channels where gate < 0.1 (fully rejected)
                    gelu_rejected = (g_gelu_vals < 0.1).float().mean().item()
                    phi_rejected = (g_phi_vals < 0.1).float().mean().item()

                    # PW2 and residual connection
                    for x_stream in ['gelu', 'phi']:
                        post = post_gelu if x_stream == 'gelu' else post_phi
                        xb = F.linear(post,
                                     _w(f'{prefix}.pwconv2.weight'),
                                     _w(f'{prefix}.pwconv2.bias'))
                        xb = xb.permute(0, 3, 1, 2)
                        gamma = _w(f'{prefix}.gamma')
                        if x_stream == 'gelu':
                            x_gelu = residual_gelu + gamma.view(1, -1, 1, 1) * xb
                        else:
                            x_phi = residual_phi + gamma.view(1, -1, 1, 1) * xb

                    # Post-block feature divergence
                    block_cos_sim = F.cosine_similarity(
                        x_gelu.reshape(-1).unsqueeze(0),
                        x_phi.reshape(-1).unsqueeze(0)
                    ).item()

                    block_mag_ratio = x_phi.abs().mean().item() / (x_gelu.abs().mean().item() + 1e-10)

                    if key not in block_stats:
                        block_stats[key] = {
                            'gelu_gate_mean': [], 'phi_gate_mean': [],
                            'gelu_gate_max': [], 'phi_gate_max': [],
                            'gelu_decisiveness': [], 'phi_decisiveness': [],
                            'gelu_survival': [], 'phi_survival': [],
                            'gate_cos_sim': [], 'gate_mag_ratio': [],
                            'block_cos_sim': [], 'block_mag_ratio': [],
                            'gelu_committed': [], 'phi_committed': [],
                            'gelu_rejected': [], 'phi_rejected': [],
                        }

                    block_stats[key]['gelu_gate_mean'].append(gelu_gate_mean)
                    block_stats[key]['phi_gate_mean'].append(phi_gate_mean)
                    block_stats[key]['gelu_gate_max'].append(gelu_gate_max)
                    block_stats[key]['phi_gate_max'].append(phi_gate_max)
                    block_stats[key]['gelu_decisiveness'].append(gelu_decisiveness)
                    block_stats[key]['phi_decisiveness'].append(phi_decisiveness)
                    block_stats[key]['gelu_survival'].append(gelu_survival)
                    block_stats[key]['phi_survival'].append(phi_survival)
                    block_stats[key]['gate_cos_sim'].append(cos_sim)
                    block_stats[key]['gate_mag_ratio'].append(mag_ratio)
                    block_stats[key]['block_cos_sim'].append(block_cos_sim)
                    block_stats[key]['block_mag_ratio'].append(block_mag_ratio)
                    block_stats[key]['gelu_committed'].append(gelu_committed)
                    block_stats[key]['phi_committed'].append(phi_committed)
                    block_stats[key]['gelu_rejected'].append(gelu_rejected)
                    block_stats[key]['phi_rejected'].append(phi_rejected)

                    block_count += 1

        if len(block_stats.get('0.0', {}).get('gelu_gate_mean', [])) >= n_images:
            break

    return block_stats


if __name__ == '__main__':
    print("=" * 90)
    print("4th DIMENSION TRACE — Where does the color possibility space narrow?")
    print("=" * 90)
    print()
    print("Concept: The 4th dimension (ab color) is latent in luminance structure.")
    print("Each gate SELECTS which color possibilities survive.")
    print("Tracing the selection pattern through 18 blocks...")
    print()

    stats = trace_4th_dimension(n_images=5)

    # Print header
    print(f"  {'Block':<8} {'GELU gate':<11} {'φ-soft gate':<11} "
          f"{'Gate Δ':<8} {'Committed%':<14} {'Rejected%':<14} "
          f"{'CosSimGate':<11} {'MagRatio':<9} {'BlockCosSim':<12} {'BlockMagR':<9}")
    print(f"  {'-'*115}")

    cumulative_mag = 1.0

    for stage_idx in range(4):
        for block_idx in range(DEPTHS[stage_idx]):
            key = f'{stage_idx}.{block_idx}'
            s = stats[key]

            gelu_g = np.mean(s['gelu_gate_mean'])
            phi_g = np.mean(s['phi_gate_mean'])
            gate_delta = phi_g - gelu_g

            gelu_c = np.mean(s['gelu_committed']) * 100
            phi_c = np.mean(s['phi_committed']) * 100
            gelu_r = np.mean(s['gelu_rejected']) * 100
            phi_r = np.mean(s['phi_rejected']) * 100

            cos_sim_gate = np.mean(s['gate_cos_sim'])
            mag_ratio = np.mean(s['gate_mag_ratio'])
            block_cos = np.mean(s['block_cos_sim'])
            block_mag = np.mean(s['block_mag_ratio'])

            cumulative_mag *= block_mag

            committed_str = f"{gelu_c:.0f}/{phi_c:.0f}"
            rejected_str = f"{gelu_r:.0f}/{phi_r:.0f}"

            print(f"  {key:<8} {gelu_g:<11.4f} {phi_g:<11.4f} "
                  f"{gate_delta:>+7.4f} {committed_str:<14} {rejected_str:<14} "
                  f"{cos_sim_gate:<11.6f} {mag_ratio:<9.4f} {block_cos:<12.6f} {block_mag:<9.4f}")

        if stage_idx < 3:
            print(f"  {'---':<8} {'--- stage ' + str(stage_idx) + ' → ' + str(stage_idx+1) + ' ---'}")

    print()
    print(f"  Cumulative magnitude ratio (φ-soft/GELU): {cumulative_mag:.4f}")
    print(f"  This means φ-soft output is {cumulative_mag*100:.1f}% of GELU output magnitude")
    print()

    # Analysis: where does the commitment gap matter most?
    print("=" * 90)
    print("ANALYSIS: Gate Commitment — the 4th dimension selection")
    print("=" * 90)
    print()
    print("  GELU reaches gate=1.0 for strong positives (FULL commitment)")
    print(f"  φ-soft caps at gate=1/φ={INV_PHI:.4f} (NEVER fully commits)")
    print()

    # Compute total committed fraction difference
    total_gelu_committed = 0
    total_phi_committed = 0
    total_gelu_survival = 0
    total_phi_survival = 0
    for key in stats:
        total_gelu_committed += np.mean(stats[key]['gelu_committed'])
        total_phi_committed += np.mean(stats[key]['phi_committed'])
        total_gelu_survival += np.mean(stats[key]['gelu_survival'])
        total_phi_survival += np.mean(stats[key]['phi_survival'])

    n = len(stats)
    print(f"  Average commitment (gate > 0.9):")
    print(f"    GELU:    {total_gelu_committed/n*100:.1f}% of activations")
    print(f"    φ-soft:  {total_phi_committed/n*100:.1f}% of activations")
    print()
    print(f"  Average energy survival (||gated||²/||input||²):")
    print(f"    GELU:    {total_gelu_survival/n:.4f}")
    print(f"    φ-soft:  {total_phi_survival/n:.4f}")
    print(f"    Ratio:   {total_phi_survival/total_gelu_survival:.4f}")
    print()

    # The key question: what if φ-soft committed fully?
    # gate_corrected = x * σ(φ·x) instead of (1/φ) * x * σ(φ·x)
    # That's SiLU with φ-curvature: lim_{x→∞} = x (full commitment)
    print("  HYPOTHESIS: The 5% color loss = the 1/φ scaling ceiling.")
    print(f"  φ-soft gate max = 1/φ = {INV_PHI:.4f}")
    print(f"  GELU gate max = 1.000")
    print(f"  For strong color signals (large positive z), φ-soft under-commits by {(1-INV_PHI)*100:.1f}%")
    print()
    print("  The 4th dimension is loudest in the COMMITTED channels —")
    print("  the ones where the gate says 'YES, this IS the color'.")
    print("  Capping at 0.618 means we're never fully saying YES.")
