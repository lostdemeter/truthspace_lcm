"""
Phase 17B: Dead Channel Pruning — The Main Attack

Phase 17 discovered: 54.7% of expanded channels are ALWAYS DEAD.
They never produce positive pre-GELU activations across any image.

This means:
  - Those rows of PW1 and columns of PW2 are dead weight
  - Pruning them should have ZERO impact (they never contribute)
  - But let's verify with RMSE on real images

We also test aggressive pruning thresholds:
  - <5% survival: definitely dead (54.7%)
  - <10% survival: mostly dead
  - <20% survival: rarely used
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
import time

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

# ================================================================
# Step 1: Compute channel survival rates from calibration images
# ================================================================
print("\nComputing channel survival rates...")

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

N_CAL = 15  # Calibration images
channel_survivals = {}  # (stage, block) -> [4C] survival rates

for img_idx in range(200, 200 + N_CAL * 2):
    if sum(1 for v in channel_survivals.values() if len(v) > 0) > 0 and \
       len(list(channel_survivals.values())[0]) >= N_CAL:
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

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))

                # Record per-channel survival
                gate_mask = (pre_gelu > 0).float()
                ch_surv = gate_mask.mean(dim=(0, 1, 2)).numpy()

                key = (stage_idx, block_idx)
                if key not in channel_survivals:
                    channel_survivals[key] = []
                channel_survivals[key].append(ch_surv)

                # Complete block
                post_gelu = pre_gelu * 0.5 * (1.0 + torch.erf(pre_gelu / np.sqrt(2.0)))
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# Average survival rates
avg_survival = {}
for key, surv_list in channel_survivals.items():
    avg_survival[key] = np.mean(surv_list, axis=0)

print(f"  Calibrated on {N_CAL} images")


# ================================================================
# Step 2: Build pruned weight variants
# ================================================================

def build_pruned_weights(threshold):
    """Build pruned PW weights for a given survival threshold."""
    pruned = {}
    stats = {'original': 0, 'pruned': 0, 'channels_kept': 0, 'channels_total': 0}

    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            key = (stage_idx, block_idx)
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

            ch_surv = avg_survival[key]
            alive = ch_surv >= threshold

            W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            b1 = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()
            W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()

            stats['original'] += W1.size + b1.size + W2.size
            stats['channels_total'] += len(alive)
            stats['channels_kept'] += alive.sum()

            # Prune: keep only alive rows of W1 and alive columns of W2
            W1_pruned = W1[alive]  # [n_alive, C]
            b1_pruned = b1[alive]  # [n_alive]
            W2_pruned = W2[:, alive]  # [C, n_alive]

            stats['pruned'] += W1_pruned.size + b1_pruned.size + W2_pruned.size

            pruned[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W1_pruned).float()
            pruned[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b1_pruned).float()
            pruned[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W2_pruned).float()
            pruned[f'{prefix}.alive_mask'] = torch.from_numpy(alive)

    return pruned, stats


# ================================================================
# Step 3: Forward pass with pruned weights
# ================================================================

def forward_pruned(img_tensor, pruned_weights):
    """Run V16 forward with pruned PW weights."""
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

            xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                         v16._get_weight(f'{prefix}.dwconv.bias'),
                         padding=3, groups=dim)
            xb = xb.permute(0, 2, 3, 1)
            xb = F.layer_norm(xb, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))

            # PRUNED PW1: only alive channels
            pw1_key = f'{prefix}.pwconv1.weight'
            if pw1_key in pruned_weights:
                W1_p = pruned_weights[pw1_key]
                b1_p = pruned_weights[f'{prefix}.pwconv1.bias']
                W2_p = pruned_weights[f'{prefix}.pwconv2.weight']

                pre_gelu = F.linear(xb, W1_p, b1_p)
            else:
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))

            post_gelu = pre_gelu * 0.5 * (1.0 + torch.erf(pre_gelu / np.sqrt(2.0)))

            if pw1_key in pruned_weights:
                xb = F.linear(post_gelu, W2_p,
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
            else:
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))

            xb = xb.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * xb

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))

    # UNet
    out0 = v16._geometric_unet_block(features[3], features[2], 0)
    out1 = v16._geometric_unet_block(out0, features[1], 1)
    out2 = v16._geometric_unet_block(out1, features[0], 2)
    out3 = v16._geometric_last_shuf(out2)

    # Color (V17 style - single matmul)
    cm_path = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz'
    cm_data = np.load(cm_path)
    color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out3)

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


# ================================================================
# Step 4: Test pruning at multiple thresholds
# ================================================================
print()
print('=' * 70)
print('DEAD CHANNEL PRUNING: RMSE Impact')
print('=' * 70)
print()

# Test images
N_TEST = 30
SZ = 256
test_data = []
for idx in range(300, 400):
    if len(test_data) >= N_TEST:
        break
    im = cv2.imread(images[idx])
    if im is None:
        continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append((t, gt_ab))

# Baseline (no pruning, but with V17 color matrix)
baseline_rmses = []
for t, gt_ab in test_data:
    with torch.no_grad():
        pred = forward_pruned(t, {})
    pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    baseline_rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
baseline_mean = np.mean(baseline_rmses)

print(f"Baseline (V17, no pruning): RMSE = {baseline_mean:.3f}")
print()

print(f"{'Threshold':<12} {'Ch kept':<12} {'Ch pruned':<12} {'PW params':<14} "
      f"{'Reduction':<12} {'RMSE':<10} {'Δ%':<10}")
print("-" * 82)

for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    pruned, stats = build_pruned_weights(threshold)

    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = forward_pruned(t, pruned)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))

    mean_rmse = np.mean(rmses)
    delta = (mean_rmse - baseline_mean) / baseline_mean * 100
    kept = stats['channels_kept']
    total = stats['channels_total']
    reduction = (1 - stats['pruned'] / stats['original']) * 100

    print(f"  <{threshold:<9.0%} {kept:<12} {total-kept:<12} "
          f"{stats['pruned']:>12,}  {reduction:>10.1f}%  "
          f"{mean_rmse:<10.3f} {delta:+.2f}%")


# ================================================================
# Step 5: Combined — dead channel pruning + low-rank
# ================================================================
print()
print('=' * 70)
print('COMBINED: Dead Channel Pruning + Low-Rank Approximation')
print('=' * 70)
print()

# Best threshold from above: use 5% (the safe one)
best_threshold = 0.05
pruned_05, stats_05 = build_pruned_weights(best_threshold)

# Now apply low-rank to the PRUNED weights
for rank_frac in [1.0, 0.75, 0.50, 0.25]:
    combined = {}
    combined_params = 0
    original_params = 0

    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            pw1_key = f'{prefix}.pwconv1.weight'

            if pw1_key in pruned_05:
                W1 = pruned_05[pw1_key].numpy()
                b1 = pruned_05[f'{prefix}.pwconv1.bias'].numpy()
                W2 = pruned_05[f'{prefix}.pwconv2.weight'].numpy()

                original_params += v16._get_weight(f'{prefix}.pwconv1.weight').numel()
                original_params += v16._get_weight(f'{prefix}.pwconv1.bias').numel()
                original_params += v16._get_weight(f'{prefix}.pwconv2.weight').numel()

                if rank_frac < 1.0:
                    # Low-rank PW1
                    U1, S1, V1t = np.linalg.svd(W1, full_matrices=False)
                    k1 = max(1, int(len(S1) * rank_frac))
                    W1_lr = (U1[:, :k1] * S1[:k1]) @ V1t[:k1]

                    # Low-rank PW2
                    U2, S2, V2t = np.linalg.svd(W2, full_matrices=False)
                    k2 = max(1, int(len(S2) * rank_frac))
                    W2_lr = (U2[:, :k2] * S2[:k2]) @ V2t[:k2]

                    combined[pw1_key] = torch.from_numpy(W1_lr.astype(np.float32))
                    combined[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W2_lr.astype(np.float32))

                    # Count params: for factored storage, need U*S + V per matrix
                    combined_params += k1 * (W1.shape[0] + W1.shape[1])
                    combined_params += k2 * (W2.shape[0] + W2.shape[1])
                else:
                    combined[pw1_key] = pruned_05[pw1_key]
                    combined[f'{prefix}.pwconv2.weight'] = pruned_05[f'{prefix}.pwconv2.weight']
                    combined_params += W1.size + W2.size

                combined[f'{prefix}.pwconv1.bias'] = pruned_05[f'{prefix}.pwconv1.bias']
                combined_params += b1.size

    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = forward_pruned(t, combined)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))

    mean_rmse = np.mean(rmses)
    delta = (mean_rmse - baseline_mean) / baseline_mean * 100
    pw_reduction = (1 - combined_params / original_params) * 100

    print(f"  Prune <5% + rank {rank_frac:.0%}: "
          f"PW params {combined_params:>12,} ({pw_reduction:+.1f}% vs original)  "
          f"RMSE {mean_rmse:.3f} ({delta:+.2f}%)")


print()
print('=' * 70)
print('CONCLUSION')
print('=' * 70)
