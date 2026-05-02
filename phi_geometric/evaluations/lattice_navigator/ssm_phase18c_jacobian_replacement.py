"""
Phase 18C: Jacobian Replacement — Using the Composed Transform Directly

Phase 18B showed: the Jacobian J(z) = W2 @ diag(GELU'(z)) @ W1 has
effective rank 16-25% of C. The spectral profile is universal (0.994+).

NEW IDEA: Instead of storing W1 [4C, C] and W2 [C, 4C] separately,
store the mean Jacobian J_mean [C, C] directly. This is:
  J_mean = W2 @ diag(E[GELU'(z)]) @ W1

Advantages:
  - J_mean is [C, C] instead of W1 [4C, C] + W2 [C, 4C]
  - J_mean has rank ~124 at stage 3 (vs 467+564 for W1+W2)
  - Captures the COMPOSED geometry directly

Tests:
  A) Replace MLP with mean Jacobian (linear approximation)
  B) Low-rank Jacobian (keep only top-K SVs)
  C) Per-image Jacobian (compute at runtime from a few features)
  D) Jacobian + bias correction (affine approximation)
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

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def gelu_derivative(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    pdf = torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


# ================================================================
# Step 1: Compute mean Jacobian + bias at each block
# ================================================================
print("\nComputing mean Jacobians from calibration images...")

N_CAL = 15
mean_gelu_deriv = defaultdict(lambda: None)
mean_gelu_output = defaultdict(lambda: None)
mean_input = defaultdict(lambda: None)
count = defaultdict(int)

for img_idx in range(200, 200 + N_CAL * 2):
    if count.get((0, 0), 0) >= N_CAL:
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

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                W1 = v16._get_weight(f'{prefix}.pwconv1.weight')
                b1 = v16._get_weight(f'{prefix}.pwconv1.bias')

                pre_gelu = F.linear(xb, W1, b1)
                g_prime = gelu_derivative(pre_gelu)  # [1, H, W, 4C]
                post_gelu = geometric_gelu(pre_gelu)

                # Accumulate mean GELU derivative per channel
                g_mean = g_prime.mean(dim=(0, 1, 2)).numpy()  # [4C]
                gelu_out_mean = post_gelu.mean(dim=(0, 1, 2)).numpy()  # [4C]
                inp_mean = xb.mean(dim=(0, 1, 2)).numpy()  # [C]

                if mean_gelu_deriv[key] is None:
                    mean_gelu_deriv[key] = g_mean
                    mean_gelu_output[key] = gelu_out_mean
                    mean_input[key] = inp_mean
                else:
                    mean_gelu_deriv[key] += g_mean
                    mean_gelu_output[key] += gelu_out_mean
                    mean_input[key] += inp_mean
                count[key] += 1

                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# Normalize
for key in mean_gelu_deriv:
    if mean_gelu_deriv[key] is not None:
        mean_gelu_deriv[key] /= count[key]
        mean_gelu_output[key] /= count[key]
        mean_input[key] /= count[key]

# Compute mean Jacobian: J_mean = W2 @ diag(g'_mean) @ W1
# Also compute the bias: y_0 = W2 @ GELU(b1) + b2 (output when input is zero)
jacobians = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        dim = dims[stage_idx]

        W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
        b1 = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
        b2 = v16._get_weight(f'{prefix}.pwconv2.bias').numpy()

        g_mean = mean_gelu_deriv[key]

        # Mean Jacobian: [C, C]
        J_mean = (W2 * g_mean[np.newaxis, :]) @ W1

        # Bias correction: output when z=mean_input
        gelu_mean = mean_gelu_output[key]
        y_at_mean = W2 @ gelu_mean + b2  # [C]
        jac_at_mean = J_mean @ mean_input[key]  # [C]
        bias_correction = y_at_mean - jac_at_mean  # [C]

        # SVD of J_mean
        U_J, S_J, Vt_J = np.linalg.svd(J_mean, full_matrices=False)

        jacobians[key] = {
            'J_mean': J_mean,
            'bias': bias_correction,
            'U': U_J, 'S': S_J, 'Vt': Vt_J,
            'W1_params': W1.size + b1.size,
            'W2_params': W2.size + b2.size,
            'J_params': J_mean.size + bias_correction.size,
        }

print(f"  Calibrated on {N_CAL} images")


# ================================================================
# Step 2: Test Jacobian replacement
# ================================================================

N_TEST = 20
test_data = []
for idx in range(300, 360):
    if len(test_data) >= N_TEST:
        break
    im = cv2.imread(images[idx])
    if im is None:
        continue
    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append((t, gt_ab))


def forward_with_jacobian(img_tensor, mode='full', rank_frac=1.0):
    """
    Replace the MLP (W1 + GELU + W2) with its mean Jacobian.

    Modes:
      'full': full MLP (baseline)
      'jacobian': y = J_mean @ z + bias
      'jacobian_lowrank': y = U[:,:K] @ diag(S[:K]) @ Vt[:K,:] @ z + bias
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

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
            residual = x
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            key = (stage_idx, block_idx)

            xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                         v16._get_weight(f'{prefix}.dwconv.bias'),
                         padding=3, groups=dim)
            xb = xb.permute(0, 2, 3, 1)
            xb = F.layer_norm(xb, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))

            if mode == 'full':
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))

            elif mode == 'jacobian':
                J = jacobians[key]
                J_t = torch.from_numpy(J['J_mean'].astype(np.float32))
                b_t = torch.from_numpy(J['bias'].astype(np.float32))
                xb = F.linear(xb, J_t, b_t)

            elif mode == 'jacobian_lowrank':
                J = jacobians[key]
                K = max(1, int(len(J['S']) * rank_frac))
                # Low-rank: U[:,:K] @ diag(S[:K]) @ Vt[:K,:]
                US = J['U'][:, :K] * J['S'][:K]
                J_lr = US @ J['Vt'][:K, :]
                J_t = torch.from_numpy(J_lr.astype(np.float32))
                b_t = torch.from_numpy(J['bias'].astype(np.float32))
                xb = F.linear(xb, J_t, b_t)

            xb = xb.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * xb

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

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


def eval_variant(label, mode, rank_frac=1.0):
    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = forward_with_jacobian(t, mode, rank_frac)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    return np.mean(rmses)


# ================================================================
# Run all tests
# ================================================================
print()
print('=' * 70)
print('JACOBIAN REPLACEMENT: RMSE Tests')
print('=' * 70)
print()

baseline = eval_variant("Baseline (full MLP)", 'full')
print(f"  Baseline (full MLP):          RMSE = {baseline:.3f}")

jac_full = eval_variant("Mean Jacobian (full rank)", 'jacobian')
print(f"  Mean Jacobian (full rank):    RMSE = {jac_full:.3f} ({(jac_full-baseline)/baseline*100:+.2f}%)")

# Count params
total_pw = sum(j['W1_params'] + j['W2_params'] for j in jacobians.values())
total_jac = sum(j['J_params'] for j in jacobians.values())
print(f"  PW params: {total_pw:,} → Jacobian params: {total_jac:,} "
      f"({total_jac/total_pw*100:.1f}%)")

print()
print("  Low-rank Jacobian:")
print(f"  {'Rank %':<10} {'Rank dims':<12} {'Params':<14} {'RMSE':<10} {'Δ%':<10}")
print(f"  " + "-" * 56)

for rank_frac in [1.0, 0.75, 0.50, 0.25, 0.10]:
    rmse = eval_variant(f"LR {rank_frac:.0%}", 'jacobian_lowrank', rank_frac)

    # Count low-rank params: per block, U[:K]*C + S[:K] + Vt[:K]*C + bias
    lr_params = 0
    for key, J in jacobians.items():
        dim = dims[key[0]]
        K = max(1, int(len(J['S']) * rank_frac))
        lr_params += K * dim + K + K * dim + dim  # U_K, S_K, Vt_K, bias
    reduction = lr_params / total_pw * 100

    delta = (rmse - baseline) / baseline * 100
    print(f"  {rank_frac:<10.0%} {'~' + str(int(384*rank_frac)):<12} "
          f"{lr_params:>12,}  {rmse:<10.3f} {delta:+.2f}%")


# ================================================================
# Compare all approaches
# ================================================================
print()
print('=' * 70)
print('COMPARISON: All PW Compression Methods')
print('=' * 70)
print()

print(f"{'Method':<40} {'Params':<14} {'% of orig':<10} {'RMSE':<10} {'Δ%':<10}")
print("-" * 84)

# Original PW
print(f"  {'Original PW (W1+W2)':<38} {total_pw:>12,} {'100%':<10} {baseline:<10.3f} —")

# Full Jacobian
print(f"  {'Mean Jacobian (full)':<38} {total_jac:>12,} {total_jac/total_pw*100:<9.1f}% "
      f"{jac_full:<10.3f} {(jac_full-baseline)/baseline*100:+.2f}%")

# Low-rank Jacobian best
for rf in [0.50, 0.25]:
    rmse = eval_variant(f"LR {rf:.0%}", 'jacobian_lowrank', rf)
    lr_params = 0
    for key, J in jacobians.items():
        dim = dims[key[0]]
        K = max(1, int(len(J['S']) * rf))
        lr_params += K * dim * 2 + K + dim
    print(f"  {'Jacobian rank ' + f'{rf:.0%}':<38} {lr_params:>12,} "
          f"{lr_params/total_pw*100:<9.1f}% {rmse:<10.3f} {(rmse-baseline)/baseline*100:+.2f}%")


print()
print('=' * 70)
print('PHASE 18C SUMMARY')
print('=' * 70)
