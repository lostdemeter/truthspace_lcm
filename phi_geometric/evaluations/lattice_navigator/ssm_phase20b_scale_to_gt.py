"""
Phase 20B: The Magnitude as the Truncated Dimension

Phase 20 showed: the Jacobian shift points TOWARD ground truth
(cosine 0.19-0.75) but with wrong MAGNITUDE.

This is the Gödel truncation: abs(-2)=abs(2) loses the sign.
GELU truncation: the direction is preserved but the magnitude
(how far to go in that direction) is input-dependent and lost
in linearization.

Tests:
  A) Oracle scaling: for each image, find the optimal scale factor
     that minimizes distance to GT. If this is small and consistent,
     it's a single "missing dimension."
  B) Per-channel oracle: does scaling vary by channel? If so, the
     truncated dimension has structure.
  C) The interpolation path: DDColor → GT as a function of scale.
     Is the path linear? Or does it curve through a higher dimension?
  D) Can we predict the optimal scale from the input features?
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
from collections import defaultdict
from scipy.optimize import minimize_scalar

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

def gelu_derivative(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    pdf = torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


# ================================================================
# Calibrate mean Jacobians
# ================================================================
print("Calibrating mean Jacobians...")

mean_gelu_d = {}
mean_gelu_out = {}
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
                g_o = geometric_gelu(pre_gelu).mean(dim=(0, 1, 2)).numpy()

                if key not in mean_gelu_d:
                    mean_gelu_d[key] = g_d
                    mean_gelu_out[key] = g_o
                else:
                    mean_gelu_d[key] += g_d
                    mean_gelu_out[key] += g_o
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
    mean_gelu_out[key] /= cal_count[key]

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
        J = (W2 * mean_gelu_d[key][np.newaxis, :]) @ W1
        bias = W2 @ mean_gelu_out[key] + b2
        jacobians[key] = {'J': J, 'bias': bias}

print("  Done")


# ================================================================
# Forward functions
# ================================================================
def forward_encoder(img_tensor, mode='original', jac_scale=1.0):
    """Run encoder with original or Jacobian PW."""
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

            if mode == 'original':
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
            elif mode == 'jacobian':
                J_t = torch.from_numpy(jacobians[key]['J'].astype(np.float32))
                b_t = torch.from_numpy(jacobians[key]['bias'].astype(np.float32))
                xb = F.linear(xb, J_t * jac_scale, b_t)

            xb = xb.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual_x + gamma.view(1, -1, 1, 1) * xb

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))

    return features


def features_to_color(features, img_tensor):
    """Run UNet + color decode."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    out0 = v16._geometric_unet_block(features[3], features[2], 0)
    out1 = v16._geometric_unet_block(out0, features[1], 1)
    out2 = v16._geometric_unet_block(out1, features[0], 2)
    out3 = v16._geometric_last_shuf(out2)

    cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
    color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out3)

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    pred = F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))
    return pred[0, :2].permute(1, 2, 0).numpy()


# ================================================================
# Collect test data
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


# ================================================================
# TEST A: Interpolation path — DDColor → GT via scale
# ================================================================
print()
print('=' * 70)
print('TEST A: Scale Factor Sweep — Original vs Jacobian')
print('=' * 70)
print()

# For the original model, we can't scale. But for the Jacobian,
# we can scale J_mean to control the magnitude.
# Also test: interpolating between DDColor output and GT directly.

print("  Testing Jacobian scale factors:")
print(f"  {'Scale':<8} {'RMSE':<10} {'vs baseline':<12}")
print(f"  " + "-" * 30)

for scale in [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0]:
    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            features = forward_encoder(t, 'jacobian', jac_scale=scale)
            pred_ab = features_to_color(features, t)
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    mean_rmse = np.mean(rmses)
    print(f"  {scale:<8.2f} {mean_rmse:<10.3f}")


# ================================================================
# TEST B: Oracle per-image optimal scale
# ================================================================
print()
print('=' * 70)
print('TEST B: Oracle Per-Image Optimal Scale')
print('=' * 70)
print()

print("  For each image, find the scale that minimizes RMSE to GT:")
print(f"  {'Image':<8} {'Orig RMSE':<12} {'Best scale':<12} {'Best RMSE':<12} "
      f"{'Improvement':<12}")
print(f"  " + "-" * 56)

oracle_scales = []
for img_idx, (t, gt_ab) in enumerate(test_data[:10]):
    with torch.no_grad():
        # Original RMSE
        features_orig = forward_encoder(t, 'original')
        pred_orig = features_to_color(features_orig, t)
    pred_orig_r = cv2.resize(pred_orig, (gt_ab.shape[1], gt_ab.shape[0]))
    orig_rmse = np.sqrt(np.mean((pred_orig_r - gt_ab)**2))

    # Find optimal Jacobian scale
    def rmse_at_scale(s):
        with torch.no_grad():
            features_j = forward_encoder(t, 'jacobian', jac_scale=s)
            pred_j = features_to_color(features_j, t)
        pred_j_r = cv2.resize(pred_j, (gt_ab.shape[1], gt_ab.shape[0]))
        return np.sqrt(np.mean((pred_j_r - gt_ab)**2))

    result = minimize_scalar(rmse_at_scale, bounds=(0.01, 5.0), method='bounded')
    best_scale = result.x
    best_rmse = result.fun
    oracle_scales.append(best_scale)

    improvement = (orig_rmse - best_rmse) / orig_rmse * 100
    print(f"  {img_idx:<8} {orig_rmse:<12.3f} {best_scale:<12.3f} {best_rmse:<12.3f} "
          f"{improvement:+.1f}%")

print(f"\n  Mean oracle scale: {np.mean(oracle_scales):.3f} ± {np.std(oracle_scales):.3f}")
print(f"  Scale range: {np.min(oracle_scales):.3f} to {np.max(oracle_scales):.3f}")

# Is the optimal scale near a φ-related value?
phi_candidates = [1/PHI**2, 1/PHI, 1.0, PHI, PHI**2, 2*np.sqrt(2/np.pi)]
print(f"\n  φ-related candidates:")
for p in phi_candidates:
    dist = abs(np.mean(oracle_scales) - p)
    print(f"    {p:.4f} — distance from mean: {dist:.4f}")


# ================================================================
# TEST C: The interpolation geometry
# ================================================================
print()
print('=' * 70)
print('TEST C: Interpolation Geometry — Is the Path Linear?')
print('=' * 70)
print()

# For one image, compute RMSE along the interpolation path
# from DDColor output to GT. If linear, the path is a straight line.
# If curved, there's a "shortcut" through a higher dimension.

t, gt_ab = test_data[0]
with torch.no_grad():
    features_orig = forward_encoder(t, 'original')
    pred_orig = features_to_color(features_orig, t)
pred_orig_r = cv2.resize(pred_orig, (gt_ab.shape[1], gt_ab.shape[0]))

print("  Interpolation: output(α) = (1-α)*DDColor + α*GT")
print(f"  {'α':<8} {'RMSE':<10} {'Expected (linear)':<18} {'Curvature':<10}")
print(f"  " + "-" * 46)

orig_rmse = np.sqrt(np.mean((pred_orig_r - gt_ab)**2))
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    interp = (1 - alpha) * pred_orig_r + alpha * gt_ab
    actual_rmse = np.sqrt(np.mean((interp - gt_ab)**2))
    expected_rmse = (1 - alpha) * orig_rmse  # Linear expectation
    curvature = actual_rmse - expected_rmse
    print(f"  {alpha:<8.1f} {actual_rmse:<10.3f} {expected_rmse:<18.3f} {curvature:<+10.3f}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 20B SUMMARY: The Truncated Dimension')
print('=' * 70)
print()
print("The Jacobian shift points TOWARD GT (cos 0.19-0.75).")
print("The question is: can we find the right MAGNITUDE?")
print()
print("If the optimal scale is consistent → single truncated dimension")
print("If the optimal scale varies wildly → multiple truncated dimensions")
print("If the optimal scale is near φ → the dimension IS φ-structured")
