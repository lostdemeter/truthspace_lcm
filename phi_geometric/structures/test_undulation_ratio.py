"""
Part 13: The 85/15 Ratio and Undulation Structure

Tests:
1. Is the warm/cool ratio exactly 1/φ⁴ vs 1-1/φ⁴?
2. Does this ratio appear at multiple scales in DDColor?
3. Do alive/dead zones alternate with φ-structured spacing?
4. Is the output energy distributed according to φ-powers?

The hypothesis: the signal passes through a φ-structured manifold.
The alternating alive/dead zones create an interference pattern.
The φ-lattice determines which "frequencies" pass through.
Dead zones are NODES of a standing wave, not empty space.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# φ-power ratios for reference
print("=" * 70)
print("φ-POWER RATIOS")
print("=" * 70)
print()
for n in range(1, 8):
    frac = 1/PHI**n
    print(f"  1/φ^{n} = {frac:.6f}  →  {frac*100:.1f}% / {(1-frac)*100:.1f}%")
print()
print(f"  DDColor vocabulary: 14% cool / 86% warm")
print(f"  1/φ⁴ = {1/PHI**4*100:.1f}% — matches cool fraction within {abs(14 - 1/PHI**4*100):.1f}%")
print(f"  Toy model pull: 11% — matches 1/φ⁵ ({1/PHI**5*100:.1f}%) within {abs(11 - 1/PHI**5*100):.1f}%")
print()

# ================================================================
# Load DDColor
# ================================================================
print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def gelu_derivative(z):
    """GELU'(z) = Φ(z) + z·φ(z) where Φ=CDF, φ=PDF of standard normal"""
    from scipy.stats import norm
    return norm.cdf(z) + z * norm.pdf(z)


# ================================================================
# PART 1: The 85/15 Ratio Across Scales
# ================================================================
print()
print("=" * 70)
print("PART 1: Does the 85/15 Ratio Appear at Multiple Scales?")
print("=" * 70)
print()

# Collect per-block statistics from DDColor
all_block_stats = {}

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        bias_key = f'{prefix}.pwconv1.bias'

        if bias_key not in v16.weights:
            continue

        bias = v16.weights[bias_key]
        gate = gelu_derivative(bias)

        # Multiple scales of the 85/15 test
        push = (gate > 0.5).sum() / len(gate)
        near_dead = (gate < 0.05).sum() / len(gate)  # deep CONTRACT
        alive_rate = (gate > 1/PHI**4).sum() / len(gate)  # above Level 2
        warm_rate = (gate > 1 - 1/PHI**4).sum() / len(gate)  # in EXPAND

        all_block_stats[(stage_idx, block_idx)] = {
            'gate': gate,
            'bias': bias,
            'push_frac': push,
            'near_dead_frac': near_dead,
            'alive_above_l2': alive_rate,
            'warm_expand': warm_rate,
            'n_channels': len(gate),
        }

# Report ratios at different thresholds
print("  Per-stage alive/dead ratios at different φ-level thresholds:")
print()

for threshold_name, threshold_val in [
    ("gate > 0.5 (push)", 0.5),
    ("gate > 1/φ (expand)", 1/PHI),
    ("gate > 1/φ² (preserve)", 1/PHI**2),
    ("gate > 1/φ³", 1/PHI**3),
    ("gate > 1/φ⁴ (Level 2)", 1/PHI**4),
    ("gate > 1/φ⁵", 1/PHI**5),
]:
    print(f"  Threshold: {threshold_name} = {threshold_val:.4f}")
    for stage_idx in range(4):
        stage_gates = []
        for block_idx in range(depths[stage_idx]):
            key = (stage_idx, block_idx)
            if key in all_block_stats:
                stage_gates.extend(all_block_stats[key]['gate'].tolist())
        stage_gates = np.array(stage_gates)
        alive = (stage_gates > threshold_val).sum() / len(stage_gates)
        print(f"    S{stage_idx}: {alive*100:5.1f}% alive / {(1-alive)*100:5.1f}% dead")
    print()


# ================================================================
# PART 2: Spatial Undulation — Alive/Dead Alternation in Gate Fields
# ================================================================
print()
print("=" * 70)
print("PART 2: Spatial Undulation — Do Gates Alternate with φ-Spacing?")
print("=" * 70)
print()

# Process images through V16 to get gate fields
N_IMGS = 3
target_blocks = [(0, 1), (1, 1), (2, 0), (2, 4), (3, 0)]

gate_fields = defaultdict(list)
pre_gelu_fields = defaultdict(list)

for img_idx in range(300, 300 + N_IMGS * 2):
    if len(gate_fields.get(target_blocks[0], [])) >= N_IMGS:
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
                key = (stage_idx, block_idx)

                dw_out = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                                 v16._get_weight(f'{prefix}.dwconv.bias'),
                                 padding=3, groups=dim)

                xb = dw_out.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))

                if key in target_blocks:
                    gate_binary = (pre_gelu > 0).float()[0].numpy()  # [H, W, 4C]
                    gate_fields[key].append(gate_binary)
                    pre_gelu_fields[key].append(pre_gelu[0].numpy())

                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# Analyze the spatial alternation pattern of alive/dead zones
print("  Alive/dead spatial alternation — run-length analysis:")
print("  (A 'run' is a consecutive sequence of alive or dead along a row)")
print()

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]  # [H, W, 4C]
    H, W, C4 = gate.shape

    # For each channel, compute run lengths along rows
    all_alive_runs = []
    all_dead_runs = []

    for ch in range(min(200, C4)):
        for row in range(H):
            pattern = gate[row, :, ch]
            # Compute run lengths
            changes = np.diff(pattern)
            change_pts = np.where(changes != 0)[0] + 1
            run_starts = np.concatenate([[0], change_pts])
            run_ends = np.concatenate([change_pts, [W]])
            run_lengths = run_ends - run_starts

            for i, start in enumerate(run_starts):
                rl = run_lengths[i]
                if pattern[start] > 0.5:
                    all_alive_runs.append(rl)
                else:
                    all_dead_runs.append(rl)

    if not all_alive_runs or not all_dead_runs:
        print(f"  Block {key[0]}.{key[1]}: insufficient data")
        continue

    alive_runs = np.array(all_alive_runs)
    dead_runs = np.array(all_dead_runs)

    alive_mean = alive_runs.mean()
    dead_mean = dead_runs.mean()

    # Key ratio: dead run length / alive run length
    ratio = dead_mean / alive_mean if alive_mean > 0 else float('inf')

    # Is the ratio near a φ-power?
    nearest_phi_power = None
    min_dist = float('inf')
    for n in range(-3, 4):
        d = abs(ratio - PHI**n)
        if d < min_dist:
            min_dist = d
            nearest_phi_power = n

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}, {C4} channels):")
    print(f"    Alive runs: mean={alive_mean:.2f}, median={np.median(alive_runs):.0f}")
    print(f"    Dead runs:  mean={dead_mean:.2f}, median={np.median(dead_runs):.0f}")
    print(f"    Dead/Alive ratio: {ratio:.3f}")
    print(f"    Nearest φ^n: φ^{nearest_phi_power} = {PHI**nearest_phi_power:.3f} "
          f"(dist={min_dist:.3f})")

    # Also check: what fraction of spatial positions are alive per channel?
    alive_frac_per_ch = gate.mean(axis=(0, 1))  # per channel
    mean_alive = alive_frac_per_ch.mean()

    # Is the alive fraction near 1/φ^n?
    for n in range(1, 7):
        if abs(mean_alive - 1/PHI**n) < 0.05:
            print(f"    Mean alive fraction: {mean_alive:.4f} ≈ 1/φ^{n} = {1/PHI**n:.4f}")
            break
    else:
        print(f"    Mean alive fraction: {mean_alive:.4f}")

    # Alive fraction distribution
    print(f"    Alive fraction per channel: "
          f"mean={alive_frac_per_ch.mean():.3f}, "
          f"std={alive_frac_per_ch.std():.3f}")

    # Does it match 1/φ⁴?
    near_phi4 = abs(mean_alive - 1/PHI**4) < 0.05
    near_phi3 = abs(mean_alive - 1/PHI**3) < 0.05
    near_phi2 = abs(mean_alive - 1/PHI**2) < 0.05
    if near_phi4:
        print(f"    → Matches 1/φ⁴ = {1/PHI**4:.4f} (Level 2 boundary!)")
    elif near_phi3:
        print(f"    → Matches 1/φ³ = {1/PHI**3:.4f}")
    elif near_phi2:
        print(f"    → Matches 1/φ² = {1/PHI**2:.4f}")
    print()


# ================================================================
# PART 3: Energy Distribution — Push/Pull Energy at φ-Levels
# ================================================================
print()
print("=" * 70)
print("PART 3: Output Energy by φ-Level — The Undulation Pattern")
print("=" * 70)
print()

print("  How much output energy comes from channels at each φ-level?")
print("  (Energy = contribution to PW2 output)")
print()

for key in target_blocks:
    if not pre_gelu_fields[key]:
        continue

    pre_gelu = pre_gelu_fields[key][0]  # [H, W, 4C]
    H, W, C4 = pre_gelu.shape

    # Get GELU output per channel
    gelu_out = pre_gelu * 0.5 * (1.0 + np.vectorize(
        lambda z: float(torch.erf(torch.tensor(z / np.sqrt(2.0))))
    )(pre_gelu))

    # Per-channel energy (mean absolute value of GELU output)
    ch_energy = np.mean(np.abs(gelu_out), axis=(0, 1))  # [4C]

    # Per-channel gate (mean GELU'(pre_gelu))
    ch_gate = np.mean((pre_gelu > 0).astype(float), axis=(0, 1))  # alive fraction

    # Bin channels by φ-level
    phi_levels = [
        ("dead (g<1/φ⁵)", 0, 1/PHI**5),
        ("1/φ⁵-1/φ⁴", 1/PHI**5, 1/PHI**4),
        ("1/φ⁴-1/φ³", 1/PHI**4, 1/PHI**3),
        ("1/φ³-1/φ²", 1/PHI**3, 1/PHI**2),
        ("1/φ²-0.5", 1/PHI**2, 0.5),
        ("0.5-1/φ", 0.5, 1/PHI),
        ("1/φ-1", 1/PHI, 1.0),
    ]

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}, {C4} channels):")
    total_energy = ch_energy.sum()

    for name, lo, hi in phi_levels:
        mask = (ch_gate >= lo) & (ch_gate < hi)
        n_ch = mask.sum()
        energy = ch_energy[mask].sum() if n_ch > 0 else 0
        pct_ch = n_ch / C4 * 100
        pct_energy = energy / total_energy * 100 if total_energy > 0 else 0

        if n_ch > 0:
            print(f"    {name:<15}: {n_ch:>4} ch ({pct_ch:>5.1f}%) → "
                  f"energy {pct_energy:>5.1f}%"
                  f"  (energy/channel: {energy/n_ch:.4f})")

    # The key ratio: energy from dead vs alive
    dead_mask = ch_gate < 0.5
    alive_mask = ch_gate >= 0.5
    dead_energy = ch_energy[dead_mask].sum()
    alive_energy = ch_energy[alive_mask].sum()

    print(f"    ---")
    print(f"    Dead energy:  {dead_energy/total_energy*100:.1f}%  "
          f"({dead_mask.sum()} channels)")
    print(f"    Alive energy: {alive_energy/total_energy*100:.1f}%  "
          f"({alive_mask.sum()} channels)")
    if alive_energy > 0:
        ratio = dead_energy / alive_energy
        print(f"    Dead/Alive energy ratio: {ratio:.3f}")
        # Phase 17C found 31.6% from dead — check consistency
    print()


# ================================================================
# PART 4: Spatial Frequency Spectrum — φ-Frequencies in Gate Pattern
# ================================================================
print()
print("=" * 70)
print("PART 4: Spatial Frequency Spectrum of Gate Pattern")
print("=" * 70)
print()

print("  If the gate field is a φ-structured interference pattern,")
print("  its spatial frequency spectrum should peak at φ-related frequencies.")
print()

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]  # [H, W, 4C]
    H, W, C4 = gate.shape

    # Compute 2D FFT of gate pattern (averaged over channels)
    gate_avg = gate.mean(axis=2)  # [H, W] — mean alive fraction per pixel

    fft_2d = np.fft.fft2(gate_avg - gate_avg.mean())
    power_2d = np.abs(fft_2d[:H//2, :W//2])**2

    # Radial power spectrum
    freqs_h = np.fft.fftfreq(H)[:H//2]
    freqs_w = np.fft.fftfreq(W)[:W//2]
    freq_grid_h, freq_grid_w = np.meshgrid(freqs_h, freqs_w, indexing='ij')
    radial_freq = np.sqrt(freq_grid_h**2 + freq_grid_w**2)

    # Bin by radial frequency
    n_bins = min(H, W) // 2
    bin_edges = np.linspace(0, radial_freq.max(), n_bins + 1)
    radial_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (radial_freq >= bin_edges[i]) & (radial_freq < bin_edges[i+1])
        if mask.sum() > 0:
            radial_power[i] = power_2d[mask].mean()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(radial_power, height=max(radial_power)*0.1, distance=2)

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}):")
    print(f"    Radial power spectrum peaks:")

    if len(peaks) > 0:
        for p in peaks[:6]:
            freq = bin_centers[p]
            power = radial_power[p]
            # Convert frequency to spatial wavelength
            wavelength = 1.0 / freq if freq > 0 else float('inf')

            # Is the wavelength near a φ-power of the dimension?
            for n in range(-2, 5):
                phi_wl = PHI**n
                if abs(wavelength - phi_wl) / phi_wl < 0.2:
                    print(f"      f={freq:.4f}, λ={wavelength:.2f} ≈ φ^{n}={phi_wl:.2f} "
                          f"(power={power:.1f})")
                    break
            else:
                print(f"      f={freq:.4f}, λ={wavelength:.2f} (power={power:.1f})")

        # Check ratios between adjacent peak frequencies
        if len(peaks) >= 2:
            peak_freqs = bin_centers[peaks[:6]]
            peak_freqs = peak_freqs[peak_freqs > 0]
            if len(peak_freqs) >= 2:
                ratios = peak_freqs[1:] / peak_freqs[:-1]
                print(f"    Frequency ratios between peaks: "
                      f"{[f'{r:.3f}' for r in ratios]}")
                phi_near = [r for r in ratios
                           if abs(r - PHI) / PHI < 0.15 or
                              abs(r - 1/PHI) / (1/PHI) < 0.15]
                if phi_near:
                    print(f"    φ-near ratios: {[f'{r:.3f}' for r in phi_near]}")
    else:
        print(f"    No significant peaks found")
    print()


# ================================================================
# PART 5: The Per-Channel Alive Fraction as Undulation
# ================================================================
print()
print("=" * 70)
print("PART 5: Per-Channel Alive Fraction Distribution")
print("=" * 70)
print()

print("  If channels undulate between alive and dead, the per-channel")
print("  alive fraction should cluster at φ-related values.")
print("  (A channel with alive_frac = 1/φ⁴ fires 14.6% of the time)")
print()

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]  # [H, W, 4C]
    H, W, C4 = gate.shape

    alive_fracs = gate.mean(axis=(0, 1))  # [4C]

    # Histogram of alive fractions
    print(f"  Block {key[0]}.{key[1]} ({C4} channels):")
    print(f"    Mean alive fraction: {alive_fracs.mean():.4f}")
    print(f"    Std:                 {alive_fracs.std():.4f}")

    # Count channels near each φ-related alive fraction
    phi_alive_targets = {
        '~0 (dead)': 0.02,
        '1/φ⁵ (9%)': 1/PHI**5,
        '1/φ⁴ (15%)': 1/PHI**4,
        '1/φ³ (24%)': 1/PHI**3,
        '1/φ² (38%)': 1/PHI**2,
        '0.5 (50%)': 0.5,
        '1/φ (62%)': 1/PHI,
        '1-1/φ³ (76%)': 1-1/PHI**3,
        '1-1/φ⁴ (85%)': 1-1/PHI**4,
        '~1 (always)': 0.98,
    }

    print(f"    Distribution near φ-targets (±0.03):")
    for name, target in phi_alive_targets.items():
        near = ((alive_fracs > target - 0.03) & (alive_fracs < target + 0.03)).sum()
        if near > 0:
            print(f"      {name:<18}: {near:>4} ({near/C4*100:>5.1f}%)")

    # The 85/15 test: what fraction of channels are in [12%, 18%] alive?
    # (i.e., near 1/φ⁴ = 14.6%)
    near_phi4 = ((alive_fracs > 0.12) & (alive_fracs < 0.18)).sum()
    print(f"    Channels with 12-18% alive rate (near 1/φ⁴): "
          f"{near_phi4} ({near_phi4/C4*100:.1f}%)")
    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 70)
print("SUMMARY: The Undulation Hypothesis")
print("=" * 70)
print()

print(f"  φ-power ratios for reference:")
print(f"    1/φ⁴ = {1/PHI**4:.4f} = 14.6%  ←  cool/pull fraction?")
print(f"    1/φ³ = {1/PHI**3:.4f} = 23.6%  ←  Phase 17D flip rate (13-21%)")
print(f"    1/φ² = {1/PHI**2:.4f} = 38.2%  ←  the PRESERVE boundary")
print(f"    1/φ  = {1/PHI:.4f} = 61.8%  ←  the scaffold gate")
print()
print(f"  The hypothesis: signals pass through a φ-structured manifold.")
print(f"  Alive/dead zones create an interference pattern.")
print(f"  The φ-lattice determines which frequencies can pass.")
print(f"  Dead zones are NODES that define the wave shape.")
