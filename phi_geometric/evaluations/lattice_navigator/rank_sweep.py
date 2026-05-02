"""
Rank Sweep: Find the minimum rank where the low-rank encoder preserves color.

The rank-90 attempt failed (feature correlation 0.16). 
Question: at what rank does the approximation work?
Options: 90%, 95%, 99%, 99.9%, full
"""
import numpy as np
import cv2
import sys
import os
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256

print('=== RANK SWEEP: Finding the Threshold ===\n')

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


def build_lowrank_kernels(v16, variance_target):
    """Build low-rank kernel approximations at a given variance target."""
    lowrank = {}

    # Stem
    stem_w = v16._get_weight('encoder.arch.downsample_layers.0.0.weight').numpy()
    W_2d = stem_w.reshape(96, -1)
    U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    k = np.searchsorted(cumvar, variance_target) + 1
    k = min(k, len(S))
    approx = (U[:, :k] * S[:k]) @ Vt[:k]
    lowrank['stem'] = torch.from_numpy(approx.reshape(stem_w.shape)).float()
    stem_rank = k

    # DW convs
    total_rank = 0
    n_blocks = 0
    for stage_idx in range(4):
        dim = dims[stage_idx]
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
            kernels_flat = dw_w.squeeze(1).reshape(dim, -1)
            U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
            cumvar = np.cumsum(S**2) / (S**2).sum()
            k = np.searchsorted(cumvar, variance_target) + 1
            k = min(k, len(S))
            approx = (U[:, :k] * S[:k]) @ Vt[:k]
            lowrank[f'{prefix}.dwconv.weight'] = torch.from_numpy(approx.reshape(dw_w.shape)).float()
            total_rank += k
            n_blocks += 1

    return lowrank, stem_rank, total_rank / n_blocks


def run_encoder_lowrank(v16, x, lowrank_kernels):
    """Run encoder with low-rank kernel replacements."""
    features = []
    # Stem
    x = F.conv2d(x, lowrank_kernels['stem'],
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
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            residual = x
            dw_key = f'{prefix}.dwconv.weight'
            x = F.conv2d(x, lowrank_kernels[dw_key],
                         v16._get_weight(f'{prefix}.dwconv.bias'), padding=3, groups=dim)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))
            x = F.linear(x, v16._get_weight(f'{prefix}.pwconv1.weight'),
                         v16._get_weight(f'{prefix}.pwconv1.bias'))
            x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
            x = F.linear(x, v16._get_weight(f'{prefix}.pwconv2.weight'),
                         v16._get_weight(f'{prefix}.pwconv2.bias'))
            x = x.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * x

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))
    return features


def get_features(v16, img_tensor, lowrank=None):
    """Get 256-dim features from encoder + UNet."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    with torch.no_grad():
        if lowrank:
            features = run_encoder_lowrank(v16, x, lowrank)
        else:
            features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()


# Use 5 test images for the sweep
test_indices = [55, 60, 65, 70, 75]

# First, get full-rank features and build color basis
print('Building color basis from full encoder...')
all_full = []
all_gt = []

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features(v16, img_tensor, lowrank=None)
    flat = enc.reshape(256, -1).T
    sample = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_full.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_full = np.vstack(all_full)
all_gt = np.vstack(all_gt)
enc_mean = all_full.mean(axis=0)

C = (all_full - enc_mean).T @ all_gt / len(all_full)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dir_1 = U_color[:, 0]
color_dir_2 = U_color[:, 1]

# Store full features per image for correlation
full_features_per_img = {}
for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features(v16, img_tensor, lowrank=None)
    flat = (enc.reshape(256, -1).T - enc_mean)
    f1 = (flat @ color_dir_1).reshape(SZ, SZ)
    f2 = (flat @ color_dir_2).reshape(SZ, SZ)
    full_features_per_img[idx] = (f1, f2)

# Now sweep variance targets
print(f'\n{"Var%":>6} {"Stem":>5} {"AvgDW":>6} {"FeatCorr":>9} {"F1_r":>7} {"F2_r":>7}')
print('-' * 50)

for var_target in [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999, 1.0]:
    if var_target >= 1.0:
        # Full rank = no approximation
        lr_kernels = None
        stem_rank = 48
        avg_dw_rank = 49.0
    else:
        lr_kernels, stem_rank, avg_dw_rank = build_lowrank_kernels(v16, var_target)

    # Test on same images
    feat_corrs = []
    f1_corrs = []
    f2_corrs = []

    for idx in test_indices:
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        enc = get_features(v16, img_tensor, lowrank=lr_kernels)
        flat = (enc.reshape(256, -1).T - enc_mean)
        f1_lr = (flat @ color_dir_1).reshape(SZ, SZ)
        f2_lr = (flat @ color_dir_2).reshape(SZ, SZ)

        f1_full, f2_full = full_features_per_img[idx]

        # Per-channel feature correlation (sample a few channels)
        flat_full_raw = (get_features(v16, img_tensor, lowrank=None) if lr_kernels else enc).reshape(256, -1).T
        flat_lr_raw = enc.reshape(256, -1).T
        # Quick: correlate the 2D fields
        r1 = np.corrcoef(f1_full.flatten(), f1_lr.flatten())[0, 1]
        r2 = np.corrcoef(f2_full.flatten(), f2_lr.flatten())[0, 1]

        # Mean feature correlation (sample 20 channels)
        ch_corrs = []
        for ch in range(0, 256, 13):
            r_ch = np.corrcoef(flat_full_raw[:, ch], flat_lr_raw[:, ch])[0, 1]
            if not np.isnan(r_ch):
                ch_corrs.append(r_ch)
        feat_corrs.append(np.mean(ch_corrs) if ch_corrs else 0)
        f1_corrs.append(r1 if not np.isnan(r1) else 0)
        f2_corrs.append(r2 if not np.isnan(r2) else 0)

    mean_fc = np.mean(feat_corrs)
    mean_f1 = np.mean(f1_corrs)
    mean_f2 = np.mean(f2_corrs)

    label = f'{var_target*100:6.2f}%' if var_target < 1.0 else '  FULL'
    print(f'{label} {stem_rank:5d} {avg_dw_rank:6.1f} {mean_fc:9.4f} {mean_f1:7.4f} {mean_f2:7.4f}')

print('\nDone!')
