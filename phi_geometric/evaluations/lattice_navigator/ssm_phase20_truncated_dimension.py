"""
Phase 20: The Truncated Dimension — Gödel's Missing Sign

Hypothesis: DDColor and Ground Truth are projections of the same
higher-dimensional state. The GELU gate truncates a dimension,
collapsing the representation like abs() collapses sign.

If the residual (GT - DDColor) has structure in the GELU-truncated
dimensions, then recovering those dimensions gives us ground truth.

The 4-bit cliff (Phase 17E) and binary code (Phase 17D) suggest
we're at a quantum information boundary. Like polarized light,
the order of operations and the full state (not just the projection)
determines the outcome.

Tests:
  A) Does the GT-DDColor residual have structure? (not random noise)
  B) Does the residual correlate with GELU leakage patterns?
  C) Does the mean Jacobian's improvement point TOWARD ground truth?
  D) Is there a rotation in the truncated space that maps DDColor → GT?
  E) What is the effective dimensionality of the residual?
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
# Step 1: Compute DDColor output and Ground Truth
# ================================================================
print("\nComputing DDColor outputs and ground truth...")

N_TEST = 20
test_pairs = []  # (gray_tensor, gt_ab, ddcolor_ab)

for idx in range(300, 360):
    if len(test_pairs) >= N_TEST:
        break
    im = cv2.imread(images[idx])
    if im is None:
        continue
    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # Ground truth ab channels
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0  # [256, 256, 2]

    # DDColor output (needs 3-channel input)
    result = v16.colorize(g3)
    if result is None:
        continue
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
    ddcolor_ab = result_lab[:, :, 1:].astype(float) - 128.0  # [256, 256, 2]

    test_pairs.append((t, gt_ab, ddcolor_ab))

print(f"  Collected {len(test_pairs)} image pairs")


# ================================================================
# TEST A: Structure of the Residual
# ================================================================
print()
print('=' * 70)
print('TEST A: Is the GT-DDColor Residual Structured?')
print('=' * 70)
print()

residuals = []
for _, gt_ab, dd_ab in test_pairs:
    residual = gt_ab - dd_ab  # [256, 256, 2]
    residuals.append(residual)

# Stack all residuals
all_residuals = np.stack(residuals)  # [N, 256, 256, 2]

# Per-image RMSE
rmses = [np.sqrt(np.mean(r**2)) for r in residuals]
print(f"  Mean RMSE: {np.mean(rmses):.2f}")
print(f"  Std RMSE:  {np.std(rmses):.2f}")

# Spatial structure: is the residual spatially smooth or random noise?
# Measure spatial autocorrelation
spatial_corrs = []
for r in residuals:
    # Shift by 1 pixel in each direction and measure correlation
    r_flat = r[:, :, 0]  # Just channel a
    shifted = r_flat[1:, :] 
    original = r_flat[:-1, :]
    corr = np.corrcoef(shifted.flatten(), original.flatten())[0, 1]
    spatial_corrs.append(corr)

mean_spatial = np.mean(spatial_corrs)
print(f"  Spatial autocorrelation (1-pixel shift): {mean_spatial:.4f}")
print(f"  → {'STRUCTURED (smooth)' if mean_spatial > 0.5 else 'NOISY (random)'}")

# Cross-image correlation: do different images have similar residual patterns?
cross_corrs = []
for i in range(len(residuals)):
    for j in range(i+1, min(i+5, len(residuals))):
        c = np.corrcoef(residuals[i].flatten(), residuals[j].flatten())[0, 1]
        cross_corrs.append(c)

print(f"  Cross-image residual correlation: {np.mean(cross_corrs):.4f}")
print(f"  → {'SYSTEMATIC bias' if abs(np.mean(cross_corrs)) > 0.1 else 'Image-specific'}")

# PCA of residuals: what is the effective dimensionality?
# Reshape: [N, 256*256*2]
resid_flat = all_residuals.reshape(len(residuals), -1)
# SVD
U_r, S_r, Vt_r = np.linalg.svd(resid_flat, full_matrices=False)
cumvar_r = np.cumsum(S_r**2) / np.sum(S_r**2)
rank50 = np.searchsorted(cumvar_r, 0.50) + 1
rank90 = np.searchsorted(cumvar_r, 0.90) + 1

print(f"  Residual PCA: 50% variance in {rank50}/{len(S_r)} dims")
print(f"  Residual PCA: 90% variance in {rank90}/{len(S_r)} dims")
print(f"  → {'LOW-DIMENSIONAL' if rank90 < len(S_r) * 0.5 else 'HIGH-DIMENSIONAL'}")


# ================================================================
# TEST B: Residual vs GELU Leakage Pattern
# ================================================================
print()
print('=' * 70)
print('TEST B: Does the Residual Correlate with GELU Leakage?')
print('=' * 70)
print()

# For each image, run through the encoder and capture:
# 1. The GELU leakage pattern (dead channel contributions)
# 2. The full MLP output
# 3. The alive-only MLP output
# Then see if (full - alive_only) correlates with the GT residual

target_blocks = [(2, 0), (2, 4), (2, 8), (3, 0)]

for pair_idx in range(min(3, len(test_pairs))):
    t, gt_ab, dd_ab = test_pairs[pair_idx]
    residual = gt_ab - dd_ab

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean_t) / std_t

    dead_contributions = {}

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
                residual_x = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
                key = (stage_idx, block_idx)

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                W2 = v16._get_weight(f'{prefix}.pwconv2.weight')
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)

                if key in target_blocks:
                    alive_mask = (pre_gelu > 0).float()
                    dead_mask = 1.0 - alive_mask

                    pw2_dead = F.linear(post_gelu * dead_mask, W2, None)
                    pw2_alive = F.linear(post_gelu * alive_mask, W2, None)

                    dead_contributions[key] = {
                        'dead_output': pw2_dead[0].numpy(),  # [H, W, C]
                        'alive_output': pw2_alive[0].numpy(),
                        'dead_energy': (pw2_dead**2).mean().item(),
                        'alive_energy': (pw2_alive**2).mean().item(),
                    }

                xb = F.linear(post_gelu, W2,
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual_x + gamma.view(1, -1, 1, 1) * xb

    if pair_idx == 0:
        print(f"  Image {pair_idx}: residual RMSE = {np.sqrt(np.mean(residual**2)):.2f}")
        print(f"  {'Block':<8} {'Dead E':<10} {'Alive E':<10} {'Dead/Total':<12}")
        print(f"  " + "-" * 40)
        for key in target_blocks:
            d = dead_contributions[key]
            total_e = d['dead_energy'] + d['alive_energy']
            print(f"  {key[0]}.{key[1]:<5} {d['dead_energy']:<10.4f} "
                  f"{d['alive_energy']:<10.4f} {d['dead_energy']/total_e*100:<11.1f}%")


# ================================================================
# TEST C: Does the Jacobian Improvement Point Toward GT?
# ================================================================
print()
print('=' * 70)
print('TEST C: Does the Mean Jacobian Move Closer to Ground Truth?')
print('=' * 70)
print()

# The mean Jacobian improved RMSE by -1.64%. Does this improvement
# come from moving the output TOWARD ground truth?
# Or is it just general denoising that happens to reduce RMSE?

# Run both versions and compute directional alignment
from collections import defaultdict
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

# We need the mean Jacobian data from Phase 18C
# Recompute quickly
print("  Computing mean Jacobians for directional test...")

def gelu_derivative(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    pdf = torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

# Quick calibration
mean_gelu_d = {}
cal_count = defaultdict(int)
for img_idx in range(200, 215):
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
                residual_x = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
                key = (stage_idx, block_idx)

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                g_d = gelu_derivative(pre_gelu).mean(dim=(0, 1, 2)).numpy()

                if key not in mean_gelu_d:
                    mean_gelu_d[key] = g_d
                else:
                    mean_gelu_d[key] += g_d
                cal_count[key] += 1

                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual_x + gamma.view(1, -1, 1, 1) * xb

for key in mean_gelu_d:
    mean_gelu_d[key] /= cal_count[key]

# Compute Jacobians
jacobians = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
        b1 = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
        b2 = v16._get_weight(f'{prefix}.pwconv2.bias').numpy()

        g_mean = mean_gelu_d[key]
        J = (W2 * g_mean[np.newaxis, :]) @ W1
        gelu_at_bias = geometric_gelu(torch.from_numpy(b1).float()).numpy()
        bias_out = W2 @ gelu_at_bias + b2

        jacobians[key] = {'J': J, 'bias': bias_out}

# Now: for each test image, compute original output, Jacobian output,
# and ground truth. Check if Jacobian moves toward GT.
print()
gt_closer_count = 0
total_count = 0

for pair_idx in range(len(test_pairs)):
    t, gt_ab, dd_ab = test_pairs[pair_idx]
    residual_to_gt = gt_ab - dd_ab  # Direction from DDColor to GT

    # Original DDColor RMSE
    orig_rmse = np.sqrt(np.mean(residual_to_gt**2))

    # Jacobian version
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

        features = []
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
                residual_x = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
                key = (stage_idx, block_idx)

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                # JACOBIAN version
                J_t = torch.from_numpy(jacobians[key]['J'].astype(np.float32))
                b_t = torch.from_numpy(jacobians[key]['bias'].astype(np.float32))
                xb = F.linear(xb, J_t, b_t)

                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual_x + gamma.view(1, -1, 1, 1) * xb

            x_normed = x.permute(0, 2, 3, 1)
            x_normed = F.layer_norm(x_normed, (dim,),
                                    v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                    v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
            features.append(x_normed.permute(0, 3, 1, 2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

        cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
        color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
        color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out3)

        coarse_input = torch.cat([color_out, (t - mean_t) / std_t], dim=1)
        pred = F.conv2d(coarse_input,
                        v16._get_weight('refine_net.0.0.weight'),
                        v16._get_weight('refine_net.0.0.bias'))

    jac_ab = pred[0, :2].permute(1, 2, 0).numpy()
    jac_ab = cv2.resize(jac_ab, (gt_ab.shape[1], gt_ab.shape[0]))

    jac_rmse = np.sqrt(np.mean((jac_ab - gt_ab)**2))

    # Did Jacobian move closer to GT?
    closer = jac_rmse < orig_rmse
    if closer:
        gt_closer_count += 1
    total_count += 1

    # Directional analysis: is the Jacobian shift in the direction of GT?
    jac_shift = jac_ab - dd_ab  # How Jacobian changed from original
    # Cosine between (Jac shift) and (GT direction)
    cos_sim = np.sum(jac_shift * residual_to_gt) / (
        np.linalg.norm(jac_shift) * np.linalg.norm(residual_to_gt) + 1e-10)

    if pair_idx < 5:
        print(f"  Image {pair_idx}: orig={orig_rmse:.2f} jac={jac_rmse:.2f} "
              f"{'CLOSER' if closer else 'farther'}  "
              f"cos(shift, GT_direction)={cos_sim:.4f}")

print(f"\n  Jacobian closer to GT: {gt_closer_count}/{total_count} "
      f"({gt_closer_count/total_count*100:.0f}%)")
print(f"  → {'JACOBIAN MOVES TOWARD GT' if gt_closer_count > total_count * 0.5 else 'NOT DIRECTIONAL'}")


# ================================================================
# TEST D: Dimensionality of the DDColor→GT Gap
# ================================================================
print()
print('=' * 70)
print('TEST D: The Gap Between DDColor and Ground Truth')
print('=' * 70)
print()

# For each image, the residual (GT - DDColor) is a vector in color space.
# How many dimensions does this gap span across images?
# If low-dimensional, it's a systematic truncation.
# If high-dimensional, it's image-specific noise.

print("  Residual dimensionality analysis:")
print(f"  Total images: {len(residuals)}")
print(f"  Residual size per image: {residuals[0].shape} = {residuals[0].size} values")
print()

# Mean residual (systematic bias)
mean_residual = np.mean(residuals, axis=0)
mean_residual_energy = np.sum(mean_residual**2)
total_residual_energy = np.mean([np.sum(r**2) for r in residuals])

print(f"  Mean residual energy: {mean_residual_energy:.2f}")
print(f"  Total residual energy (avg): {total_residual_energy:.2f}")
print(f"  Systematic fraction: {mean_residual_energy/total_residual_energy*100:.1f}%")
print(f"  → {'SYSTEMATIC BIAS' if mean_residual_energy/total_residual_energy > 0.1 else 'MOSTLY RANDOM'}")

# Mean residual in a/b channels
mean_a = np.mean(mean_residual[:, :, 0])
mean_b = np.mean(mean_residual[:, :, 1])
print(f"\n  Mean bias: a={mean_a:.2f}, b={mean_b:.2f}")
print(f"  → DDColor systematically {'shifts' if abs(mean_a) > 1 or abs(mean_b) > 1 else 'is close in'} color space")

# Per-pixel residual statistics
resid_magnitudes = [np.sqrt(r[:,:,0]**2 + r[:,:,1]**2) for r in residuals]
mean_mag = np.mean([np.mean(m) for m in resid_magnitudes])
max_mag = np.mean([np.max(m) for m in resid_magnitudes])
print(f"\n  Mean per-pixel residual magnitude: {mean_mag:.2f}")
print(f"  Mean max per-pixel residual: {max_mag:.2f}")

# Where are the largest errors? Spatially?
# Average the magnitude maps
mean_mag_map = np.mean(resid_magnitudes, axis=0)
# Find the regions with highest error
top_pct = np.percentile(mean_mag_map, 90)
high_error_frac = np.mean(mean_mag_map > top_pct)
print(f"  High-error regions (>90th pct): {high_error_frac*100:.1f}% of pixels")
print(f"  90th percentile magnitude: {top_pct:.2f}")


# ================================================================
# TEST E: φ-Structure in the Residual
# ================================================================
print()
print('=' * 70)
print('TEST E: Is There φ-Structure in the Residual?')
print('=' * 70)
print()

# If φ-space can navigate between DDColor and GT, the residual
# should have structure related to φ

# Test: does the residual magnitude follow a φ-lattice distribution?
all_resid_values = np.concatenate([r.flatten() for r in residuals])

# Histogram of residual values
hist, bin_edges = np.histogram(all_resid_values, bins=200, range=(-50, 50))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Check if peaks align with φ powers
phi_powers = [PHI**k for k in range(-5, 6)]
neg_phi_powers = [-p for p in phi_powers]
all_phi = sorted(neg_phi_powers + phi_powers)

# Find histogram peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1, distance=5)
peak_positions = bin_centers[peaks]

print(f"  Residual distribution peaks at: {peak_positions[:10]}")
print(f"  φ-lattice points near zero: {[f'{p:.3f}' for p in all_phi if abs(p) < 5]}")

# Check if the residual's spectral structure has φ-related features
# FFT of mean residual map
fft_a = np.fft.fft2(mean_residual[:, :, 0])
power_spectrum = np.abs(fft_a)**2
# Radial average
H, W = power_spectrum.shape
y, x_coord = np.mgrid[:H, :W]
r = np.sqrt((y - H//2)**2 + (x_coord - W//2)**2).astype(int)
radial_prof = np.bincount(r.flatten(), weights=np.fft.fftshift(power_spectrum).flatten())
radial_count = np.bincount(r.flatten())
radial_mean = radial_prof / (radial_count + 1e-10)

# Check if power spectrum peaks at φ-related frequencies
print(f"\n  Power spectrum: peak at frequency {np.argmax(radial_mean[1:])+1}")
print(f"  φ-related frequencies: {[int(PHI**k) for k in range(1, 8) if PHI**k < H//2]}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 20 SUMMARY: The Truncated Dimension')
print('=' * 70)
print()
print("Gödel's insight: 'this is false' is only paradoxical if you")
print("truncate the dimension that makes it true.")
print()
print("DDColor truncates via GELU (binary gate → sign information).")
print("The residual between DDColor and GT is the truncated dimension.")
print()
print("Key findings:")
print("  A) Is the residual structured?")
print("  B) Does it correlate with GELU leakage?")
print("  C) Does the Jacobian move toward GT?")
print("  D) What dimensionality is the gap?")
print("  E) Is there φ-structure?")
