"""
Phase 19: UNet Decoder Analysis

The UNet decoder has 12,410,496 params (22.6% of DDColor), completely untouched.

Components:
  - Merge conv 3×3: 8,188,928 params (3 layers, largest component)
  - Upsample 1×1 + pixshuffle: 3,166,208 params (3 layers)
  - Last pixel shuffle 1×1: 1,052,672 params
  - Skip batchnorms: 2,688 params (trivial)

Strategy: test low-rank SVD on all weight matrices, then test RMSE impact.
The 3×3 convs can be reshaped to [C_out, C_in*9] for SVD analysis.
The 1×1 convs are just [C_out, C_in] matrices.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

weights = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_weights_static.npz')


# ================================================================
# Step 1: SVD Analysis of all UNet weight matrices
# ================================================================
print()
print('=' * 70)
print('STEP 1: SVD Analysis of UNet Weight Matrices')
print('=' * 70)
print()

unet_weights = {}

# Merge convs (3×3)
for layer in range(3):
    key = f'decoder.layers.{layer}.conv.0.weight'
    w = weights[key]  # [C_out, C_in, 3, 3]
    C_out, C_in = w.shape[:2]
    # Reshape to [C_out, C_in*9] for SVD
    w_2d = w.reshape(C_out, -1)
    S = np.linalg.svdvals(w_2d)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    r90 = np.searchsorted(cumvar, 0.90) + 1
    r95 = np.searchsorted(cumvar, 0.95) + 1
    r99 = np.searchsorted(cumvar, 0.99) + 1

    unet_weights[f'merge_{layer}'] = {
        'key': key, 'shape': w.shape, 'params': w.size,
        'S': S, 'r90': r90, 'r95': r95, 'r99': r99,
        'full_rank': min(C_out, C_in * 9)
    }

    print(f"  Merge conv {layer}: {w.shape} = {w.size:,} params")
    print(f"    SVD rank: 90%={r90}, 95%={r95}, 99%={r99} (full={min(C_out, C_in*9)})")

# Upsample convs (1×1)
for layer in range(3):
    key = f'decoder.layers.{layer}.shuf.conv.0.weight'
    w = weights[key]  # [C_out, C_in, 1, 1]
    w_2d = w.reshape(w.shape[0], w.shape[1])
    S = np.linalg.svdvals(w_2d)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    r90 = np.searchsorted(cumvar, 0.90) + 1
    r95 = np.searchsorted(cumvar, 0.95) + 1

    unet_weights[f'upsample_{layer}'] = {
        'key': key, 'shape': w.shape, 'params': w.size,
        'S': S, 'r90': r90, 'r95': r95,
        'full_rank': min(w.shape[0], w.shape[1])
    }

    print(f"  Upsample conv {layer}: {w.shape} = {w.size:,} params")
    print(f"    SVD rank: 90%={r90}, 95%={r95} (full={min(w.shape[0], w.shape[1])})")

# Last pixel shuffle
key = 'decoder.last_shuf.conv.0.weight'
w = weights[key]  # [4096, 256, 1, 1]
w_2d = w.reshape(w.shape[0], w.shape[1])
S = np.linalg.svdvals(w_2d)
cumvar = np.cumsum(S**2) / np.sum(S**2)
r90 = np.searchsorted(cumvar, 0.90) + 1
r95 = np.searchsorted(cumvar, 0.95) + 1

unet_weights['last_shuf'] = {
    'key': key, 'shape': w.shape, 'params': w.size,
    'S': S, 'r90': r90, 'r95': r95,
    'full_rank': min(w.shape[0], w.shape[1])
}

print(f"  Last pixel shuffle: {w.shape} = {w.size:,} params")
print(f"    SVD rank: 90%={r90}, 95%={r95} (full={min(w.shape[0], w.shape[1])})")


# ================================================================
# Step 2: Low-rank replacement RMSE test
# ================================================================
print()
print('=' * 70)
print('STEP 2: Low-Rank UNet Replacement — RMSE Tests')
print('=' * 70)
print()

def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def full_forward(img_tensor, unet_rank_frac=1.0):
    """Full V16+V17 forward with optional UNet low-rank."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    # Encoder (full, original weights)
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
            pre_gelu = F.linear(xb,
                                v16._get_weight(f'{prefix}.pwconv1.weight'),
                                v16._get_weight(f'{prefix}.pwconv1.bias'))
            post_gelu = geometric_gelu(pre_gelu)
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

    # UNet decoder with optional low-rank
    def get_lr_weight(key, rank_frac):
        w = v16._get_weight(key)
        if rank_frac >= 1.0 or w is None:
            return w
        shape = w.shape
        if len(shape) == 4:
            C_out, C_in, kH, kW = shape
            w_2d = w.reshape(C_out, -1)
            U, S, Vt = torch.linalg.svd(w_2d, full_matrices=False)
            K = max(1, int(len(S) * rank_frac))
            w_lr = (U[:, :K] * S[:K]) @ Vt[:K]
            return w_lr.reshape(shape)
        return w

    def pixel_shuffle_icnr(inp, prefix_ps):
        w = get_lr_weight(f'{prefix_ps}.conv.0.weight', unet_rank_frac)
        out_ps = F.conv2d(inp, w, bias=None)
        out_ps = F.batch_norm(out_ps,
                             v16._get_weight(f'{prefix_ps}.conv.1.running_mean'),
                             v16._get_weight(f'{prefix_ps}.conv.1.running_var'),
                             v16._get_weight(f'{prefix_ps}.conv.1.weight'),
                             v16._get_weight(f'{prefix_ps}.conv.1.bias'),
                             training=False)
        out_ps = F.relu(out_ps)
        out_ps = F.pixel_shuffle(out_ps, 2)
        out_ps = F.pad(out_ps, (1, 0, 1, 0), mode='replicate')
        return F.avg_pool2d(out_ps, kernel_size=2, stride=1)

    cur = features[3]  # Start from deepest encoder feature
    for layer_idx in range(3):
        prefix = f'decoder.layers.{layer_idx}'

        # Upsample via pixel shuffle (ICNR style)
        up = pixel_shuffle_icnr(cur, f'{prefix}.shuf')

        # Skip connection with batch norm
        skip = F.batch_norm(features[2 - layer_idx],
                           v16._get_weight(f'{prefix}.bn.running_mean'),
                           v16._get_weight(f'{prefix}.bn.running_var'),
                           v16._get_weight(f'{prefix}.bn.weight'),
                           v16._get_weight(f'{prefix}.bn.bias'),
                           training=False)

        # ReLU on concatenation, THEN merge conv
        cat = F.relu(torch.cat([up, skip], dim=1))

        # Merge conv (3×3)
        merge_w = get_lr_weight(f'{prefix}.conv.0.weight', unet_rank_frac)
        cur = F.conv2d(cat, merge_w, None, padding=1)
        cur = F.relu(cur)
        cur = F.batch_norm(cur,
                          v16._get_weight(f'{prefix}.conv.2.running_mean'),
                          v16._get_weight(f'{prefix}.conv.2.running_var'),
                          v16._get_weight(f'{prefix}.conv.2.weight'),
                          v16._get_weight(f'{prefix}.conv.2.bias'),
                          training=False)
    out = cur

    # Last pixel shuffle
    last_w = get_lr_weight('decoder.last_shuf.conv.0.weight', unet_rank_frac)
    last_b = v16._get_weight('decoder.last_shuf.conv.0.bias')
    out = F.conv2d(out, last_w, last_b)
    out = F.relu(out)
    out = F.pixel_shuffle(out, 4)
    out = F.pad(out, (1, 0, 1, 0), mode='replicate')
    out = F.avg_pool2d(out, kernel_size=2, stride=1)

    # Color decode (V17 style)
    cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
    color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out)

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


# Test data
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


def eval_rank(rank_frac):
    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = full_forward(t, unet_rank_frac=rank_frac)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    return np.mean(rmses)


baseline = eval_rank(1.0)
print(f"  Baseline (full UNet):  RMSE = {baseline:.3f}")
print()

total_unet_params = sum(w['params'] for w in unet_weights.values())

print(f"{'Rank %':<10} {'RMSE':<10} {'Δ%':<10} {'Est. UNet params':<18}")
print("-" * 48)
for rank_frac in [1.0, 0.75, 0.50, 0.25, 0.10]:
    rmse = eval_rank(rank_frac)
    delta = (rmse - baseline) / baseline * 100

    # Estimate params
    est_params = 0
    for name, info in unet_weights.items():
        w = weights[info['key']]
        shape = w.shape
        if len(shape) == 4:
            C_out, C_in, kH, kW = shape
            full_2d = C_in * kH * kW
            full_rank = min(C_out, full_2d)
            K = max(1, int(full_rank * rank_frac))
            est_params += K * (C_out + full_2d)  # U*K + K*Vt
        else:
            est_params += w.size

    print(f"  {rank_frac:<8.0%} {rmse:<10.3f} {delta:+8.2f}%  {est_params:>16,}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 19 SUMMARY: UNet Decoder')
print('=' * 70)
print()
print(f"Total UNet params: {total_unet_params:,}")
print()
for name, info in sorted(unet_weights.items()):
    print(f"  {name:<15} {str(info['shape']):<25} {info['params']:>10,}  "
          f"rank@90%={info['r90']}/{info['full_rank']}")
