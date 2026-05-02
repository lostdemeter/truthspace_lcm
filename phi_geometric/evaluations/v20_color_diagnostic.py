#!/usr/bin/env python3
"""
V20 Color Diagnostic — find exactly where color dies.

Tests each component independently to isolate the desaturation:
  A) V16 baseline (full everything)
  B) Jacobian full-rank + full UNet (is linearization the issue?)
  C) Jacobian rank 25% + full UNet (is low-rank the issue?)
  D) Full encoder + UNet rank 50% (is UNet low-rank the issue?)
  E) V20 assembly (Jacobian r25% + UNet r50%)
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sys
import glob
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

DIMS = [96, 192, 384, 768]
DEPTHS = [3, 3, 9, 3]


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def gelu_derivative(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    pdf = torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


print("Loading V16...")
v16 = V16GeometricColorizer()

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

# Load color matrix
cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
color_matrix = torch.from_numpy(cm_data['color_matrix']).float()

# Calibrate Jacobians
print("Calibrating Jacobians...")
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
            dim = DIMS[stage_idx]
            if stage_idx > 0:
                prefix = f'encoder.arch.downsample_layers.{stage_idx}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (DIMS[stage_idx-1],),
                                 v16._get_weight(f'{prefix}.0.weight'),
                                 v16._get_weight(f'{prefix}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{prefix}.1.weight'),
                             v16._get_weight(f'{prefix}.1.bias'), stride=2)

            for block_idx in range(DEPTHS[stage_idx]):
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

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                g_prime = gelu_derivative(pre_gelu)
                post_gelu = geometric_gelu(pre_gelu)

                g_mean = g_prime.mean(dim=(0, 1, 2)).numpy()
                gelu_out_mean = post_gelu.mean(dim=(0, 1, 2)).numpy()
                inp_mean = xb.mean(dim=(0, 1, 2)).numpy()

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

# Compute Jacobians
weights = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_weights_static.npz')
jacobians = {}
for key in mean_gelu_deriv:
    if mean_gelu_deriv[key] is not None:
        mean_gelu_deriv[key] /= count[key]
        mean_gelu_output[key] /= count[key]
        mean_input[key] /= count[key]

for stage_idx in range(4):
    for block_idx in range(DEPTHS[stage_idx]):
        key = (stage_idx, block_idx)
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        W1 = weights[f'{prefix}.pwconv1.weight']
        W2 = weights[f'{prefix}.pwconv2.weight']
        b1 = weights[f'{prefix}.pwconv1.bias']
        b2 = weights[f'{prefix}.pwconv2.bias']
        g_mean = mean_gelu_deriv[key]
        J_mean = (W2 * g_mean[np.newaxis, :]) @ W1
        gelu_mean = mean_gelu_output[key]
        y_at_mean = W2 @ gelu_mean + b2
        jac_at_mean = J_mean @ mean_input[key]
        bias_correction = y_at_mean - jac_at_mean
        U_J, S_J, Vt_J = np.linalg.svd(J_mean, full_matrices=False)
        jacobians[key] = {
            'J_mean': J_mean, 'bias': bias_correction,
            'U': U_J, 'S': S_J, 'Vt': Vt_J,
        }

print(f"  Calibrated {len(jacobians)} blocks")


def forward_diagnostic(img_tensor, encoder_mode='full', encoder_rank=1.0,
                       unet_mode='full', unet_rank=1.0):
    """Forward pass with configurable encoder and UNet."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    x_input = x.clone()

    # Stem
    x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                 v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, (96,),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
    x = x.permute(0, 3, 1, 2)

    features = []
    for stage_idx in range(4):
        dim = DIMS[stage_idx]
        if stage_idx > 0:
            prefix = f'encoder.arch.downsample_layers.{stage_idx}'
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (DIMS[stage_idx-1],),
                             v16._get_weight(f'{prefix}.0.weight'),
                             v16._get_weight(f'{prefix}.0.bias'))
            x = x.permute(0, 3, 1, 2)
            x = F.conv2d(x, v16._get_weight(f'{prefix}.1.weight'),
                         v16._get_weight(f'{prefix}.1.bias'), stride=2)

        for block_idx in range(DEPTHS[stage_idx]):
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

            if encoder_mode == 'full':
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
            elif encoder_mode == 'jacobian':
                J = jacobians[key]
                K = max(1, int(len(J['S']) * encoder_rank))
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

    # UNet decoder
    def get_unet_weight(wkey, rank_frac):
        w = v16._get_weight(wkey)
        if rank_frac >= 1.0 or w is None:
            return w
        shape = w.shape
        if len(shape) == 4:
            C_out = shape[0]
            w_2d = w.reshape(C_out, -1)
            U, S, Vt = torch.linalg.svd(w_2d, full_matrices=False)
            K = max(1, int(len(S) * rank_frac))
            w_lr = (U[:, :K] * S[:K]) @ Vt[:K]
            return w_lr.reshape(shape)
        return w

    r = unet_rank if unet_mode == 'lowrank' else 1.0

    cur = features[3]
    for layer_idx in range(3):
        prefix = f'decoder.layers.{layer_idx}'

        w_up = get_unet_weight(f'{prefix}.shuf.conv.0.weight', r)
        up = F.conv2d(cur, w_up, bias=None)
        up = F.batch_norm(up,
                         v16._get_weight(f'{prefix}.shuf.conv.1.running_mean'),
                         v16._get_weight(f'{prefix}.shuf.conv.1.running_var'),
                         v16._get_weight(f'{prefix}.shuf.conv.1.weight'),
                         v16._get_weight(f'{prefix}.shuf.conv.1.bias'),
                         training=False)
        up = F.relu(up)
        up = F.pixel_shuffle(up, 2)
        up = F.pad(up, (1, 0, 1, 0), mode='replicate')
        up = F.avg_pool2d(up, kernel_size=2, stride=1)

        skip = F.batch_norm(features[2 - layer_idx],
                           v16._get_weight(f'{prefix}.bn.running_mean'),
                           v16._get_weight(f'{prefix}.bn.running_var'),
                           v16._get_weight(f'{prefix}.bn.weight'),
                           v16._get_weight(f'{prefix}.bn.bias'),
                           training=False)

        cat = F.relu(torch.cat([up, skip], dim=1))

        merge_w = get_unet_weight(f'{prefix}.conv.0.weight', r)
        cur = F.conv2d(cat, merge_w, None, padding=1)
        cur = F.relu(cur)
        cur = F.batch_norm(cur,
                          v16._get_weight(f'{prefix}.conv.2.running_mean'),
                          v16._get_weight(f'{prefix}.conv.2.running_var'),
                          v16._get_weight(f'{prefix}.conv.2.weight'),
                          v16._get_weight(f'{prefix}.conv.2.bias'),
                          training=False)

    last_w = get_unet_weight('decoder.last_shuf.conv.0.weight', r)
    last_b = v16._get_weight('decoder.last_shuf.conv.0.bias')
    out = F.conv2d(cur, last_w, last_b)
    out = F.relu(out)
    out = F.pixel_shuffle(out, 4)
    out = F.pad(out, (1, 0, 1, 0), mode='replicate')
    out = F.avg_pool2d(out, kernel_size=2, stride=1)

    # Color decode (V17 color matrix)
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out)

    # Refine net
    coarse_input = torch.cat([color_out, x_input], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


# Test image
im = cv2.imread(images[300])
r = cv2.resize(im, (256, 256))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
gt_ab = lab[:, :, 1:].astype(float) - 128.0

configs = [
    ("A: V16 full",           'full',     1.0,  'full',    1.0),
    ("B: Jac full + UNet full", 'jacobian', 1.0,  'full',    1.0),
    ("C: Jac r25% + UNet full", 'jacobian', 0.25, 'full',    1.0),
    ("D: Full enc + UNet r50%", 'full',     1.0,  'lowrank', 0.50),
    ("E: Jac r25% + UNet r50%", 'jacobian', 0.25, 'lowrank', 0.50),
    ("F: Jac r50% + UNet full", 'jacobian', 0.50, 'full',    1.0),
    ("G: Jac r75% + UNet full", 'jacobian', 0.75, 'full',    1.0),
]

print()
print("=" * 80)
print("COLOR DIAGNOSTIC — ab channel magnitude analysis")
print("=" * 80)
print()
print(f"  {'Config':<30} {'RMSE':<8} {'|ab| mean':<10} {'|ab| max':<10} {'ab std':<8}")
print(f"  {'-'*66}")

for label, enc_mode, enc_rank, unet_mode, unet_rank in configs:
    with torch.no_grad():
        pred = forward_diagnostic(t, enc_mode, enc_rank, unet_mode, unet_rank)
    pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    rmse = np.sqrt(np.mean((pred_r - gt_ab)**2))
    ab_mean = np.mean(np.abs(pred_r))
    ab_max = np.max(np.abs(pred_r))
    ab_std = np.std(pred_r)

    print(f"  {label:<30} {rmse:<8.2f} {ab_mean:<10.2f} {ab_max:<10.2f} {ab_std:<8.2f}")

print()
print(f"  {'Ground truth':<30} {'—':<8} {np.mean(np.abs(gt_ab)):<10.2f} {np.max(np.abs(gt_ab)):<10.2f} {np.std(gt_ab):<8.2f}")

# Save visual comparison for the diagnostic configs
SZ = 256
COLS = len(configs) + 2  # +gray +GT
grid = np.ones((SZ + 40, COLS * (SZ + 4) + 4, 3), dtype=np.uint8) * 255

def ab_to_bgr(gray_3ch, ab_pred):
    img_lab = cv2.cvtColor(gray_3ch, cv2.COLOR_BGR2Lab)
    L = img_lab[:, :, 0]
    ab_np = ab_pred[0, :2].permute(1, 2, 0).numpy()
    ab_np = cv2.resize(ab_np, (gray_3ch.shape[1], gray_3ch.shape[0]))
    ab_scaled = np.clip(ab_np + 128, 0, 255).astype(np.uint8)
    out_lab = np.stack([L, ab_scaled[:, :, 0], ab_scaled[:, :, 1]], axis=-1)
    return cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)

col = 0
# Gray
grid[30:30+SZ, 4:4+SZ] = cv2.resize(g3, (SZ, SZ))
cv2.putText(grid, "Gray", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
col += 1

# GT
grid[30:30+SZ, 4+(SZ+4):4+(SZ+4)+SZ] = cv2.resize(r, (SZ, SZ))
cv2.putText(grid, "GT", (4+(SZ+4)+2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
col += 1

for label, enc_mode, enc_rank, unet_mode, unet_rank in configs:
    with torch.no_grad():
        pred = forward_diagnostic(t, enc_mode, enc_rank, unet_mode, unet_rank)
    bgr = ab_to_bgr(g3, pred)
    x0 = 4 + col * (SZ + 4)
    grid[30:30+SZ, x0:x0+SZ] = cv2.resize(bgr, (SZ, SZ))
    short = label.split(":")[0] + ":" + label.split(":")[1][:15]
    cv2.putText(grid, short, (x0+2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    col += 1

out_path = '/tmp/v20_color_diagnostic.png'
cv2.imwrite(out_path, grid)
print(f"  Diagnostic grid saved: {out_path}")
