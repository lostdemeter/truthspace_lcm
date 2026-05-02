"""
Geometric Encoder at Rank-99%: The Working Threshold.

Phase transition discovered:
  95% variance → correlation 0.21 (broken)
  99% variance → correlation 0.98 (works!)
  99.9% variance → correlation 0.999

At 99%, DW convs need avg rank 23 (not 5). Still ~50% param reduction per kernel.
Stem needs rank 36 (not 13).

This script:
1. Builds rank-99% encoder
2. Full held-out comparison: LowRank vs Full vs DDColor
3. Analyzes the critical 95→99% modes
4. Visual comparison output
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

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]


def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def build_lowrank_kernels(v16, variance_target):
    """Build low-rank kernel approximations at given variance target."""
    lowrank = {}
    ranks = {}

    # Stem
    stem_w = v16._get_weight('encoder.arch.downsample_layers.0.0.weight').numpy()
    W_2d = stem_w.reshape(96, -1)
    U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    k = np.searchsorted(cumvar, variance_target) + 1
    k = min(k, len(S))
    approx = (U[:, :k] * S[:k]) @ Vt[:k]
    lowrank['stem'] = torch.from_numpy(approx.reshape(stem_w.shape)).float()
    ranks['stem'] = k

    # DW convs
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
            ranks[f'S{stage_idx}.B{block_idx}'] = k

    return lowrank, ranks


def run_encoder_lowrank(v16, x, lowrank_kernels):
    """Run encoder with low-rank kernel replacements."""
    features = []
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


print('=== GEOMETRIC ENCODER: Rank-99% ===\n')

v16 = V16GeometricColorizer()
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/rank99_encoder'
os.makedirs(out_dir, exist_ok=True)
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ============================================================
# PART 1: Build rank-99% encoder
# ============================================================
print('=== PART 1: Building Rank-99% Encoder ===\n')

lr_99, ranks_99 = build_lowrank_kernels(v16, 0.99)

print(f'Ranks at 99% variance:')
print(f'  Stem: {ranks_99["stem"]}/48')
total_orig = 48  # stem max rank
total_reduced = ranks_99['stem']
for stage_idx in range(4):
    dim = dims[stage_idx]
    stage_ranks = []
    for block_idx in range(depths[stage_idx]):
        key = f'S{stage_idx}.B{block_idx}'
        r = ranks_99[key]
        stage_ranks.append(r)
        max_r = min(dim, 49)
        total_orig += max_r
        total_reduced += r
    print(f'  Stage {stage_idx} ({dim}ch): ranks={stage_ranks}, avg={np.mean(stage_ranks):.1f}/{min(dim,49)}')

print(f'\nTotal rank budget: {total_reduced}/{total_orig} ({total_reduced/total_orig*100:.1f}%)')


# ============================================================
# PART 2: Build color basis from training images
# ============================================================
print('\n=== PART 2: Color Basis ===\n')

train_indices = list(range(50, 75))
all_enc = []
all_gt = []

for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features(v16, img_tensor, lowrank=None)
    flat = enc.reshape(256, -1).T
    sample = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_enc.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc = np.vstack(all_enc)
all_gt = np.vstack(all_gt)
enc_mean = all_enc.mean(axis=0)

C = (all_enc - enc_mean).T @ all_gt / len(all_enc)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dir_1 = U_color[:, 0]
color_dir_2 = U_color[:, 1]

# Fit linear: 2D → color
from numpy.linalg import lstsq
proj_1 = (all_enc - enc_mean) @ color_dir_1
proj_2 = (all_enc - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt[:, 1], rcond=None)

print(f'Color directions: S=[{S_color[0]:.4f}, {S_color[1]:.4f}], ratio={S_color[0]/S_color[1]:.4f}')


# ============================================================
# PART 3: Held-out comparison
# ============================================================
print('\n=== PART 3: Held-Out Comparison ===\n')

test_indices = list(range(80, 100))
results = []

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # Full encoder → 2D → color
    enc_full = get_features(v16, img_tensor, lowrank=None)
    flat_full = (enc_full.reshape(256, -1).T - enc_mean)
    fields_full = np.column_stack([flat_full @ color_dir_1, flat_full @ color_dir_2, np.ones(SZ*SZ)])
    ab_full = np.stack([
        np.clip(fields_full @ W_a, -50, 50).reshape(SZ, SZ),
        np.clip(fields_full @ W_b, -50, 50).reshape(SZ, SZ)
    ], axis=2)

    # Rank-99% encoder → 2D → color
    enc_lr = get_features(v16, img_tensor, lowrank=lr_99)
    flat_lr = (enc_lr.reshape(256, -1).T - enc_mean)
    fields_lr = np.column_stack([flat_lr @ color_dir_1, flat_lr @ color_dir_2, np.ones(SZ*SZ)])
    ab_lr = np.stack([
        np.clip(fields_lr @ W_a, -50, 50).reshape(SZ, SZ),
        np.clip(fields_lr @ W_b, -50, 50).reshape(SZ, SZ)
    ], axis=2)

    # DDColor full pipeline
    with torch.no_grad():
        ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    # Errors
    err_full = np.sqrt(np.mean((ab_full - ab_gt)**2))
    err_lr = np.sqrt(np.mean((ab_lr - ab_gt)**2))
    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_z = np.sqrt(np.mean(ab_gt**2))

    # Field correlations
    f1_full = (flat_full @ color_dir_1).reshape(SZ, SZ)
    f1_lr = (flat_lr @ color_dir_1).reshape(SZ, SZ)
    f2_full = (flat_full @ color_dir_2).reshape(SZ, SZ)
    f2_lr = (flat_lr @ color_dir_2).reshape(SZ, SZ)
    r1 = np.corrcoef(f1_full.flatten(), f1_lr.flatten())[0, 1]
    r2 = np.corrcoef(f2_full.flatten(), f2_lr.flatten())[0, 1]

    gap_full = (1 - err_full/err_z) * 100
    gap_lr = (1 - err_lr/err_z) * 100
    gap_dd = (1 - err_dd/err_z) * 100

    winner = 'LR' if err_lr < err_dd else 'DD'

    print(f'  {name}: Full={err_full:.1f}({gap_full:+.0f}%) LR99={err_lr:.1f}({gap_lr:+.0f}%) '
          f'DD={err_dd:.1f}({gap_dd:+.0f}%) [{winner}] r=[{r1:.4f},{r2:.4f}]')

    results.append({
        'name': name, 'err_full': err_full, 'err_lr': err_lr,
        'err_dd': err_dd, 'err_z': err_z, 'r1': r1, 'r2': r2
    })

    # Save visual comparison
    bgr_full = ab_to_bgr(ab_full, L)
    bgr_lr = ab_to_bgr(ab_lr, L)
    bgr_dd = ab_to_bgr(ab_dd, L)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    imgs = [
        (gray_bgr, 'Gray'),
        (bgr_lr, f'Rank99 e={err_lr:.1f}'),
        (bgr_full, f'Full e={err_full:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'r99_{name}.jpg'), strip)


# ============================================================
# PART 4: Analyze the critical 95→99% modes
# ============================================================
print('\n=== PART 4: The Critical Modes (95→99% variance) ===\n')

for stage_idx in [0, 2, 3]:
    dim = dims[stage_idx]
    prefix = f'encoder.arch.stages.{stage_idx}.0'
    dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
    kernels_flat = dw_w.squeeze(1).reshape(dim, -1)
    U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()

    rank95 = np.searchsorted(cumvar, 0.95) + 1
    rank99 = np.searchsorted(cumvar, 0.99) + 1

    print(f'Stage {stage_idx} Block 0 ({dim}ch):')
    print(f'  Rank 95%: {rank95}, Rank 99%: {rank99}')
    print(f'  Critical band: modes {rank95}→{rank99} ({rank99-rank95} modes)')

    # These critical modes: what do they look like?
    critical_bases = Vt[rank95:rank99].reshape(-1, 7, 7)
    print(f'  Critical basis shapes: {critical_bases.shape}')

    # Compare: singular values in bands
    s_90 = S[:np.searchsorted(cumvar, 0.90)+1]
    s_95 = S[np.searchsorted(cumvar, 0.90)+1:rank95]
    s_99 = S[rank95:rank99]

    print(f'  S band means: top90={s_90.mean():.4f}, 90-95={s_95.mean():.4f}, 95-99={s_99.mean():.4f}')

    # φ analysis of critical band
    for i in range(min(3, len(s_99)-1)):
        ratio = s_99[i] / s_99[i+1] if s_99[i+1] > 1e-10 else float('inf')
        phi_err = abs(ratio - PHI) / PHI * 100
        print(f'  Critical S[{i}]/S[{i+1}] = {ratio:.4f} ({phi_err:.1f}% from φ)')
    print()


# ============================================================
# SUMMARY
# ============================================================
print('=== SUMMARY ===\n')
if results:
    mean_full = np.mean([r['err_full'] for r in results])
    mean_lr = np.mean([r['err_lr'] for r in results])
    mean_dd = np.mean([r['err_dd'] for r in results])
    mean_z = np.mean([r['err_z'] for r in results])
    mean_r1 = np.mean([r['r1'] for r in results])
    mean_r2 = np.mean([r['r2'] for r in results])
    lr_wins = sum(1 for r in results if r['err_lr'] < r['err_dd'])

    print(f'Held-out ({len(results)} images):')
    print(f'  Zero:      {mean_z:.2f}')
    print(f'  Rank-99%:  {mean_lr:.2f} (gap={(1-mean_lr/mean_z)*100:.1f}%)')
    print(f'  Full 2D:   {mean_full:.2f} (gap={(1-mean_full/mean_z)*100:.1f}%)')
    print(f'  DDColor:   {mean_dd:.2f} (gap={(1-mean_dd/mean_z)*100:.1f}%)')
    print(f'  Rank99 wins: {lr_wins}/{len(results)}')
    print(f'  Field correlation: F1={mean_r1:.4f}, F2={mean_r2:.4f}')
    print()
    print(f'Compression: {total_reduced}/{total_orig} rank budget ({total_reduced/total_orig*100:.1f}%)')
    print(f'The rank-99% encoder preserves the color field while using ~50% of the spatial basis rank.')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
