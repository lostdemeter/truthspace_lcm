"""
Phase 20C: Does the GELU Gate Field Have φ-Lattice Structure?

The chain: φ-basis DW conv → LayerNorm → PW1 → GELU gate

If the DW conv outputs have φ-structure (we proved R²=0.982 for the
kernel), does that structure propagate through LayerNorm and PW1 to
create φ-structured gate patterns?

Tests:
  1. Spatial autocorrelation of gate field — does it match φ-basis decay?
  2. Fourier spectrum of gate field — are there φ-frequency peaks?
  3. Gate field spatial PCA — do principal spatial modes have φ-structure?
  4. Direct comparison: gate field with φ-basis DW vs original DW
  5. Gate transition boundaries — do the alive/dead boundaries fall on
     φ-lattice positions?
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

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


# ================================================================
# Collect gate fields and DW outputs from real images
# ================================================================
print("\nCollecting gate fields and DW outputs...")

N_IMGS = 5
target_blocks = [(0, 1), (1, 1), (2, 0), (2, 4), (2, 8), (3, 0)]

gate_fields = defaultdict(list)       # key -> list of [H, W, 4C] binary patterns
dw_outputs = defaultdict(list)        # key -> list of [H, W, C] DW conv outputs
pre_gelu_fields = defaultdict(list)   # key -> list of [H, W, 4C] continuous values

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

                # DW conv output
                dw_out = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                                 v16._get_weight(f'{prefix}.dwconv.bias'),
                                 padding=3, groups=dim)

                if key in target_blocks:
                    dw_outputs[key].append(dw_out[0].permute(1, 2, 0).numpy())

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

print(f"  Collected {N_IMGS} images")


# ================================================================
# TEST 1: Spatial Autocorrelation of Gate Field vs φ-Decay
# ================================================================
print()
print('=' * 70)
print('TEST 1: Spatial Autocorrelation of Gate Field')
print('=' * 70)
print()

print("If φ-basis DW conv creates φ-structured features, the gate field's")
print("spatial autocorrelation should decay with φ-related structure.")
print()

print(f"{'Block':<8} {'H×W':<8} {'AC lag1':<10} {'AC lag2':<10} {'AC lag3':<10} "
      f"{'Ratio 1→2':<12} {'Ratio 2→3':<12} {'φ target':<10}")
print("-" * 82)

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]  # [H, W, 4C]
    H, W, C4 = gate.shape

    # Average gate activation per spatial position (mean over channels)
    gate_mean = gate.mean(axis=2)  # [H, W]

    # Spatial autocorrelation at different lags (horizontal)
    acs = []
    for lag in range(1, min(6, W)):
        shifted = gate_mean[:, lag:]
        original = gate_mean[:, :W-lag]
        ac = np.corrcoef(shifted.flatten(), original.flatten())[0, 1]
        acs.append(ac)

    if len(acs) >= 3:
        ratio_12 = acs[1] / (acs[0] + 1e-10)
        ratio_23 = acs[2] / (acs[1] + 1e-10)

        print(f"  {key[0]}.{key[1]:<5} {H}×{W:<5} {acs[0]:<10.4f} {acs[1]:<10.4f} "
              f"{acs[2]:<10.4f} {ratio_12:<12.4f} {ratio_23:<12.4f} {1/PHI:<10.4f}")


# ================================================================
# TEST 2: Fourier Spectrum of Gate Field
# ================================================================
print()
print('=' * 70)
print('TEST 2: Fourier Spectrum of Gate Field')
print('=' * 70)
print()

print("If φ-structure exists, we should see peaks at φ-related frequencies.")
print()

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]
    H, W, C4 = gate.shape
    gate_mean = gate.mean(axis=2)

    # 2D FFT
    fft = np.fft.fft2(gate_mean - gate_mean.mean())
    power = np.abs(np.fft.fftshift(fft))**2

    # Radial power spectrum
    cy, cx = H // 2, W // 2
    y, x_c = np.mgrid[:H, :W]
    r = np.sqrt((y - cy)**2 + (x_c - cx)**2).astype(int)
    max_r = min(cy, cx)

    radial = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        mask = (r == ri)
        if mask.any():
            radial[ri] = power[mask].mean()
            counts[ri] = mask.sum()

    # Normalize
    radial /= (radial.max() + 1e-10)

    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(radial[1:], height=0.1, prominence=0.05)
    peak_freqs = peaks + 1  # offset by 1 since we skipped DC

    # φ-related frequencies
    phi_freqs = [int(round(PHI**k)) for k in range(1, 8) if PHI**k < max_r]
    phi_freqs_set = set(phi_freqs)

    # Check overlap
    phi_hits = [f for f in peak_freqs if f in phi_freqs_set or
                any(abs(f - pf) <= 1 for pf in phi_freqs)]

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}):")
    print(f"    Power spectrum peaks at frequencies: {list(peak_freqs[:8])}")
    print(f"    φ-lattice frequencies: {phi_freqs}")
    print(f"    φ-hits (within ±1): {phi_hits if phi_hits else 'NONE'}")
    print(f"    Hit rate: {len(phi_hits)}/{len(peak_freqs)} peaks "
          f"({len(phi_hits)/max(1,len(peak_freqs))*100:.0f}%)")
    print()


# ================================================================
# TEST 3: Gate Transition Boundaries — φ-Lattice Positions?
# ================================================================
print()
print('=' * 70)
print('TEST 3: Gate Transition Boundaries')
print('=' * 70)
print()

print("The alive/dead boundary is where pre_gelu ≈ 0. Do these")
print("boundaries align with φ-lattice positions in the spatial field?")
print()

for key in target_blocks:
    if not pre_gelu_fields[key]:
        continue

    pre_gelu = pre_gelu_fields[key][0]  # [H, W, 4C]
    H, W, C4 = pre_gelu.shape

    # For each channel, find the spatial positions where the gate transitions
    # (where |pre_gelu| is smallest = near the decision boundary)
    n_channels_to_check = min(50, C4)

    boundary_positions_h = []
    boundary_positions_w = []

    for ch in range(n_channels_to_check):
        channel_data = pre_gelu[:, :, ch]
        # Find positions near zero (within 10th percentile of |value|)
        abs_vals = np.abs(channel_data)
        threshold = np.percentile(abs_vals, 10)
        near_zero = abs_vals < threshold

        # Get boundary positions
        ys, xs = np.where(near_zero)
        if len(ys) > 0:
            boundary_positions_h.extend(ys.tolist())
            boundary_positions_w.extend(xs.tolist())

    if boundary_positions_h:
        bh = np.array(boundary_positions_h)
        bw = np.array(boundary_positions_w)

        # Normalize to [0, 1]
        bh_norm = bh / H
        bw_norm = bw / W

        # φ-lattice positions in [0, 1]: {n/φ^k mod 1} for various n, k
        phi_lattice = set()
        for k in range(1, 6):
            for n in range(int(PHI**k) + 1):
                val = (n / PHI**k) % 1.0
                phi_lattice.add(round(val, 4))
                phi_lattice.add(round(1.0 - val, 4))
        phi_lattice = sorted(phi_lattice)

        # Check: are boundary positions closer to φ-lattice than random?
        def min_dist_to_lattice(positions, lattice):
            dists = []
            for p in positions:
                d = min(abs(p - l) for l in lattice)
                dists.append(d)
            return np.mean(dists)

        # Actual boundary distances to φ-lattice
        actual_dist_h = min_dist_to_lattice(bh_norm[:500], phi_lattice)
        actual_dist_w = min_dist_to_lattice(bw_norm[:500], phi_lattice)

        # Random baseline
        random_h = np.random.uniform(0, 1, 500)
        random_dist = min_dist_to_lattice(random_h, phi_lattice)

        print(f"  Block {key[0]}.{key[1]} ({H}×{W}):")
        print(f"    # boundary positions: {len(bh)}")
        print(f"    Mean dist to φ-lattice (H): {actual_dist_h:.6f}")
        print(f"    Mean dist to φ-lattice (W): {actual_dist_w:.6f}")
        print(f"    Random baseline:             {random_dist:.6f}")
        print(f"    Ratio (actual/random H): {actual_dist_h/random_dist:.4f}")
        print(f"    Ratio (actual/random W): {actual_dist_w/random_dist:.4f}")
        closer = actual_dist_h < random_dist * 0.9 or actual_dist_w < random_dist * 0.9
        print(f"    → {'φ-LATTICE ALIGNMENT' if closer else 'No significant φ alignment'}")
        print()


# ================================================================
# TEST 4: DW Output → Gate Field Correlation
# ================================================================
print()
print('=' * 70)
print('TEST 4: DW Conv Spatial Structure → Gate Field Structure')
print('=' * 70)
print()

print("Does the spatial structure of the DW conv output predict the")
print("spatial structure of the gate field?")
print()

for key in target_blocks:
    if not dw_outputs[key] or not gate_fields[key]:
        continue

    dw = dw_outputs[key][0]    # [H, W, C]
    gate = gate_fields[key][0]  # [H, W, 4C]
    H, W = dw.shape[:2]

    # DW spatial energy: how much total DW activation per pixel
    dw_energy = np.sum(dw**2, axis=2)  # [H, W]

    # Gate activation rate per pixel: what fraction of channels are alive
    gate_rate = gate.mean(axis=2)  # [H, W]

    # Correlation between DW energy and gate activation
    corr = np.corrcoef(dw_energy.flatten(), gate_rate.flatten())[0, 1]

    # DW spatial autocorrelation
    dw_ac = np.corrcoef(dw_energy[:, 1:].flatten(), dw_energy[:, :-1].flatten())[0, 1]
    gate_ac = np.corrcoef(gate_rate[:, 1:].flatten(), gate_rate[:, :-1].flatten())[0, 1]

    print(f"  Block {key[0]}.{key[1]}:")
    print(f"    DW energy ↔ gate rate correlation: {corr:.4f}")
    print(f"    DW spatial autocorrelation:         {dw_ac:.4f}")
    print(f"    Gate spatial autocorrelation:        {gate_ac:.4f}")
    print(f"    AC ratio (gate/DW):                 {gate_ac/dw_ac:.4f}")
    print(f"    → {'DW DRIVES GATE STRUCTURE' if abs(corr) > 0.3 else 'Weak coupling'}")
    print()


# ================================================================
# TEST 5: φ-Ratio in Gate Field Spatial Frequencies
# ================================================================
print()
print('=' * 70)
print('TEST 5: φ-Ratio Between Adjacent Gate Field Modes')
print('=' * 70)
print()

print("If the gate field has φ-structure, ratios between adjacent")
print("spatial frequency modes should be near φ or 1/φ.")
print()

for key in target_blocks:
    if not gate_fields[key]:
        continue

    gate = gate_fields[key][0]
    H, W, C4 = gate.shape

    # Per-channel 1D FFT along horizontal axis, averaged over channels and rows
    all_power = []
    for ch in range(min(100, C4)):
        for row in range(H):
            signal = gate[row, :, ch]
            fft_1d = np.fft.fft(signal - signal.mean())
            power = np.abs(fft_1d[:W//2])**2
            all_power.append(power)

    avg_power = np.mean(all_power, axis=0)
    avg_power[0] = 0  # Remove DC

    # Find the dominant frequencies
    sorted_idx = np.argsort(avg_power)[::-1]
    top_freqs = sorted_idx[:6]
    top_freqs_sorted = sorted(top_freqs[top_freqs > 0])

    # Compute ratios between adjacent dominant frequencies
    ratios = []
    for i in range(len(top_freqs_sorted) - 1):
        r = top_freqs_sorted[i+1] / top_freqs_sorted[i]
        ratios.append(r)

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}):")
    print(f"    Top spatial frequencies: {list(top_freqs_sorted)}")
    print(f"    Ratios between adjacent: {[f'{r:.3f}' for r in ratios]}")
    print(f"    φ = {PHI:.3f}, 1/φ = {1/PHI:.3f}")

    # Check if any ratio is within 10% of φ or 1/φ
    phi_near = [r for r in ratios if abs(r - PHI) / PHI < 0.15 or
                abs(r - 1/PHI) / (1/PHI) < 0.15]
    print(f"    φ-near ratios (within 15%): {[f'{r:.3f}' for r in phi_near]}")
    print(f"    → {'φ-RATIO DETECTED' if phi_near else 'No φ-ratio'}")
    print()


# ================================================================
# TEST 6: The Critical Test — φ-Structured Gate vs Random Gate
# ================================================================
print()
print('=' * 70)
print('TEST 6: Information Content at φ-Lattice Positions')
print('=' * 70)
print()

print("If φ-structure matters, the gate field values at φ-lattice")
print("positions should carry MORE information than random positions.")
print("(Like Boom positions in attention — sparse but high-information)")
print()

for key in target_blocks:
    if not pre_gelu_fields[key]:
        continue

    pre_gelu = pre_gelu_fields[key][0]  # [H, W, 4C]
    H, W, C4 = pre_gelu.shape

    # Define φ-lattice positions (Fibonacci-spaced)
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    phi_rows = sorted(set([f % H for f in fib if f < H * 2]))
    phi_cols = sorted(set([f % W for f in fib if f < W * 2]))

    # Gate values at φ-lattice positions
    phi_gate_values = []
    for r in phi_rows:
        for c in phi_cols:
            if r < H and c < W:
                phi_gate_values.append(pre_gelu[r, c, :])

    # Gate values at random positions (same count)
    n_phi = len(phi_gate_values)
    rand_rows = np.random.randint(0, H, n_phi)
    rand_cols = np.random.randint(0, W, n_phi)
    rand_gate_values = [pre_gelu[r, c, :] for r, c in zip(rand_rows, rand_cols)]

    phi_vals = np.array(phi_gate_values)
    rand_vals = np.array(rand_gate_values)

    # Information content: variance of gate values (higher = more information)
    phi_var = np.var(phi_vals, axis=0).mean()
    rand_var = np.var(rand_vals, axis=0).mean()

    # Gate transition rate (how often the gate flips between adjacent φ positions)
    phi_transitions = 0
    phi_total = 0
    for i in range(len(phi_gate_values) - 1):
        diff = np.abs((phi_gate_values[i] > 0).astype(float) -
                      (phi_gate_values[i+1] > 0).astype(float))
        phi_transitions += diff.mean()
        phi_total += 1
    phi_transition_rate = phi_transitions / max(1, phi_total)

    rand_transitions = 0
    rand_total = 0
    for i in range(len(rand_gate_values) - 1):
        diff = np.abs((rand_gate_values[i] > 0).astype(float) -
                      (rand_gate_values[i+1] > 0).astype(float))
        rand_transitions += diff.mean()
        rand_total += 1
    rand_transition_rate = rand_transitions / max(1, rand_total)

    # Absolute magnitude at boundaries (proximity to decision boundary)
    phi_boundary_proximity = np.mean(np.abs(phi_vals))
    rand_boundary_proximity = np.mean(np.abs(rand_vals))

    print(f"  Block {key[0]}.{key[1]} ({H}×{W}, {n_phi} φ-positions):")
    print(f"    Variance: φ-lattice={phi_var:.4f}  random={rand_var:.4f}  "
          f"ratio={phi_var/rand_var:.3f}")
    print(f"    Transition rate: φ={phi_transition_rate:.4f}  "
          f"random={rand_transition_rate:.4f}  "
          f"ratio={phi_transition_rate/rand_transition_rate:.3f}")
    print(f"    Magnitude: φ={phi_boundary_proximity:.4f}  "
          f"random={rand_boundary_proximity:.4f}  "
          f"ratio={phi_boundary_proximity/rand_boundary_proximity:.3f}")
    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 20C SUMMARY: φ-Lattice Structure in the Gate Field')
print('=' * 70)
print()
print("The gate field (GELU binary decision boundary) is the")
print("'holographic plate' that encodes per-image information.")
print()
print("If it has φ-structure, then φ-space CAN navigate this plate.")
print("If not, the hologram is structured by something else.")
