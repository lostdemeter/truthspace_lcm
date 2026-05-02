"""
Geometric Encoder: Replace ConvNeXt with low-rank SVD approximations.

Strategy:
  - Stem (96 kernels): replace with rank-13 approximation
  - Depthwise 7x7 (96-768 kernels): replace with rank-k (k=rank90 from analysis)
  - Keep ALL pointwise convs, layer norms, gamma unchanged
  - This isolates: "do the spatial filters need full rank, or are 3-7 bases enough?"

If color field is preserved → spatial ops are truly low-rank geometric.
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


def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


print('=== GEOMETRIC ENCODER: Low-Rank SVD Replacement ===\n')

v16 = V16GeometricColorizer()

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/svd_encoder'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Build low-rank kernel approximations
# ============================================================
print('=== PART 1: Computing Low-Rank Approximations ===\n')

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

# Store low-rank kernels
lowrank_kernels = {}

# Stem: [96, 3, 4, 4] → rank-13
stem_w = v16._get_weight('encoder.arch.downsample_layers.0.0.weight').numpy()
W_2d = stem_w.reshape(96, -1)  # [96, 48]
U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)
cumvar = np.cumsum(S**2) / (S**2).sum()

for target_rank in [5, 8, 13, 16, 96]:
    k = min(target_rank, len(S))
    approx = (U[:, :k] * S[:k]) @ Vt[:k]
    err = np.sqrt(np.mean((W_2d - approx)**2)) / np.sqrt(np.mean(W_2d**2))
    var_kept = cumvar[k-1] if k <= len(cumvar) else 1.0
    print(f'  Stem rank-{k:2d}: relative err={err:.4f}, variance={var_kept*100:.1f}%')

stem_rank = 13
approx_stem = (U[:, :stem_rank] * S[:stem_rank]) @ Vt[:stem_rank]
lowrank_kernels['stem'] = torch.from_numpy(approx_stem.reshape(stem_w.shape)).float()
print(f'  → Using rank-{stem_rank}')

# Depthwise convolutions: [dim, 1, 7, 7] → rank-k per block
total_orig_params = 0
total_reduced_params = 0

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()  # [dim, 1, 7, 7]
        kernels = dw_w.squeeze(1)  # [dim, 7, 7]
        kernels_flat = kernels.reshape(dim, -1)  # [dim, 49]

        U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        rank90 = np.searchsorted(cumvar, 0.90) + 1
        rank95 = np.searchsorted(cumvar, 0.95) + 1

        # Use rank90 as our target
        k = rank90
        approx = (U[:, :k] * S[:k]) @ Vt[:k]
        approx_w = approx.reshape(dw_w.shape)
        err = np.sqrt(np.mean((kernels_flat - approx)**2)) / np.sqrt(np.mean(kernels_flat**2))

        lowrank_kernels[f'{prefix}.dwconv.weight'] = torch.from_numpy(approx_w).float()

        orig_params = dim * 49
        reduced_params = k * (dim + 49)  # U[:,:k] and Vt[:k] storage
        total_orig_params += orig_params
        total_reduced_params += reduced_params

        print(f'  S{stage_idx}.B{block_idx} ({dim:3d}ch): rank90={k:2d}, '
              f'err={err:.4f}, var={cumvar[k-1]*100:.1f}%, '
              f'params {orig_params}→{reduced_params} ({reduced_params/orig_params*100:.0f}%)')

print(f'\nTotal DW conv params: {total_orig_params:,} → {total_reduced_params:,} '
      f'({total_reduced_params/total_orig_params*100:.1f}%)')


# ============================================================
# PART 2: Run both encoders and compare features
# ============================================================
print('\n=== PART 2: Comparing Full vs Low-Rank Encoder ===\n')


def run_encoder_lowrank(v16, x, lowrank_kernels):
    """Run encoder with low-rank kernel replacements."""
    dims = [96, 192, 384, 768]
    depths = [3, 3, 9, 3]
    features = []

    # Stem (low-rank)
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

            # Low-rank depthwise conv
            dw_key = f'{prefix}.dwconv.weight'
            x = F.conv2d(x, lowrank_kernels[dw_key],
                         v16._get_weight(f'{prefix}.dwconv.bias'), padding=3, groups=dim)

            # Everything else unchanged
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))
            x = F.linear(x, v16._get_weight(f'{prefix}.pwconv1.weight'),
                         v16._get_weight(f'{prefix}.pwconv1.bias'))
            x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))  # GELU
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


def run_full_pipeline(v16, img_tensor, lowrank_kernels=None):
    """Run full encoder → UNet → 256d features. If lowrank_kernels provided, use them."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    with torch.no_grad():
        if lowrank_kernels is not None:
            features = run_encoder_lowrank(v16, x, lowrank_kernels)
        else:
            features = v16._geometric_encoder(x)

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

    return out3.squeeze(0).detach().numpy()  # [256, H, W]


# Build color basis from full encoder
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
train_indices = list(range(50, 65))

all_enc_full = []
all_enc_lr = []
all_gt_ab = []

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

    enc_full = run_full_pipeline(v16, img_tensor, lowrank_kernels=None)
    enc_lr = run_full_pipeline(v16, img_tensor, lowrank_kernels=lowrank_kernels)

    flat_full = enc_full.reshape(256, -1).T
    flat_lr = enc_lr.reshape(256, -1).T

    sample = np.random.choice(len(flat_full), min(2000, len(flat_full)), replace=False)
    all_enc_full.append(flat_full[sample])
    all_enc_lr.append(flat_lr[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt_ab.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc_full = np.vstack(all_enc_full)
all_enc_lr = np.vstack(all_enc_lr)
all_gt_ab = np.vstack(all_gt_ab)

# Feature-level comparison
corr_per_ch = []
for ch in range(256):
    r = np.corrcoef(all_enc_full[:, ch], all_enc_lr[:, ch])[0, 1]
    corr_per_ch.append(r)

corr_arr = np.array(corr_per_ch)
print(f'Feature correlation (full vs low-rank):')
print(f'  Mean:   {np.nanmean(corr_arr):.6f}')
print(f'  Median: {np.nanmedian(corr_arr):.6f}')
print(f'  Min:    {np.nanmin(corr_arr):.6f}')
print(f'  >0.99:  {np.sum(corr_arr > 0.99)}/256')
print(f'  >0.95:  {np.sum(corr_arr > 0.95)}/256')
print(f'  >0.90:  {np.sum(corr_arr > 0.90)}/256')

# Overall feature distance
rmse = np.sqrt(np.mean((all_enc_full - all_enc_lr)**2))
full_std = np.std(all_enc_full)
print(f'  RMSE:   {rmse:.4f} (feature std={full_std:.4f}, ratio={rmse/full_std:.4f})')


# ============================================================
# PART 3: Color prediction comparison
# ============================================================
print('\n=== PART 3: Color Prediction ===\n')

# Build color basis from full encoder
enc_mean = all_enc_full.mean(axis=0)
C_full = (all_enc_full - enc_mean).T @ all_gt_ab / len(all_enc_full)
U_color, S_color, Vt_color = np.linalg.svd(C_full, full_matrices=False)
color_dir_1 = U_color[:, 0]
color_dir_2 = U_color[:, 1]

# Linear model: 2D projection → color
proj_full_1 = (all_enc_full - enc_mean) @ color_dir_1
proj_full_2 = (all_enc_full - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_full_1, proj_full_2, np.ones(len(proj_full_1))])
from numpy.linalg import lstsq
W_a, _, _, _ = lstsq(X_2d, all_gt_ab[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt_ab[:, 1], rcond=None)

# Color from full encoder features
pred_full_a = X_2d @ W_a
pred_full_b = X_2d @ W_b
err_full = np.sqrt(np.mean((all_gt_ab[:,0] - pred_full_a)**2 + (all_gt_ab[:,1] - pred_full_b)**2))

# Color from low-rank encoder features (using same color basis)
proj_lr_1 = (all_enc_lr - enc_mean) @ color_dir_1
proj_lr_2 = (all_enc_lr - enc_mean) @ color_dir_2
X_2d_lr = np.column_stack([proj_lr_1, proj_lr_2, np.ones(len(proj_lr_1))])
pred_lr_a = X_2d_lr @ W_a
pred_lr_b = X_2d_lr @ W_b
err_lr = np.sqrt(np.mean((all_gt_ab[:,0] - pred_lr_a)**2 + (all_gt_ab[:,1] - pred_lr_b)**2))

err_zero = np.sqrt(np.mean(all_gt_ab**2))

print(f'Color prediction from 2D projection:')
print(f'  Full encoder:    err={err_full:.2f} (gap={(1-err_full/err_zero)*100:.1f}%)')
print(f'  Low-rank encoder: err={err_lr:.2f} (gap={(1-err_lr/err_zero)*100:.1f}%)')
print(f'  Zero (gray):     err={err_zero:.2f}')

# 2D field correlation
corr_f1 = np.corrcoef(proj_full_1, proj_lr_1)[0, 1]
corr_f2 = np.corrcoef(proj_full_2, proj_lr_2)[0, 1]
print(f'  Field correlations: F1 r={corr_f1:.6f}, F2 r={corr_f2:.6f}')


# ============================================================
# PART 4: Held-out image comparison with visual output
# ============================================================
print('\n=== PART 4: Held-Out Image Test ===\n')

test_indices = list(range(80, 95))
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

    # Full encoder features
    enc_full = run_full_pipeline(v16, img_tensor, lowrank_kernels=None)
    flat_full = (enc_full.reshape(256, -1).T - enc_mean)
    f1_full = (flat_full @ color_dir_1).reshape(SZ, SZ)
    f2_full = (flat_full @ color_dir_2).reshape(SZ, SZ)

    # Low-rank encoder features
    enc_lr = run_full_pipeline(v16, img_tensor, lowrank_kernels=lowrank_kernels)
    flat_lr = (enc_lr.reshape(256, -1).T - enc_mean)
    f1_lr = (flat_lr @ color_dir_1).reshape(SZ, SZ)
    f2_lr = (flat_lr @ color_dir_2).reshape(SZ, SZ)

    # DDColor full pipeline
    with torch.no_grad():
        ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    # Color from full 2D
    fields_full = np.column_stack([f1_full.flatten(), f2_full.flatten(), np.ones(SZ*SZ)])
    a_full = np.clip(fields_full @ W_a, -50, 50).reshape(SZ, SZ)
    b_full = np.clip(fields_full @ W_b, -50, 50).reshape(SZ, SZ)
    ab_full_color = np.stack([a_full, b_full], axis=2)

    # Color from low-rank 2D
    fields_lr = np.column_stack([f1_lr.flatten(), f2_lr.flatten(), np.ones(SZ*SZ)])
    a_lr = np.clip(fields_lr @ W_a, -50, 50).reshape(SZ, SZ)
    b_lr = np.clip(fields_lr @ W_b, -50, 50).reshape(SZ, SZ)
    ab_lr_color = np.stack([a_lr, b_lr], axis=2)

    # Errors
    err_full_img = np.sqrt(np.mean((ab_full_color - ab_gt)**2))
    err_lr_img = np.sqrt(np.mean((ab_lr_color - ab_gt)**2))
    err_dd_img = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_z = np.sqrt(np.mean(ab_gt**2))

    # Field correlation for this image
    r2_f1 = np.corrcoef(f1_full.flatten(), f1_lr.flatten())[0, 1]
    r2_f2 = np.corrcoef(f2_full.flatten(), f2_lr.flatten())[0, 1]

    print(f'  {name}: Full2D={err_full_img:.1f} LR2D={err_lr_img:.1f} DD={err_dd_img:.1f} '
          f'Zero={err_z:.1f} | Field r=[{r2_f1:.4f}, {r2_f2:.4f}]')

    results.append({
        'name': name, 'err_full': err_full_img, 'err_lr': err_lr_img,
        'err_dd': err_dd_img, 'err_z': err_z, 'r_f1': r2_f1, 'r_f2': r2_f2
    })

    # Visualization
    bgr_full = ab_to_bgr(ab_full_color, L)
    bgr_lr = ab_to_bgr(ab_lr_color, L)
    bgr_dd = ab_to_bgr(ab_dd, L)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    imgs = [
        (gray_bgr, 'Gray'),
        (bgr_lr, f'LowRank e={err_lr_img:.1f}'),
        (bgr_full, f'Full2D e={err_full_img:.1f}'),
        (bgr_dd, f'DDColor e={err_dd_img:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'svd_{name}.jpg'), strip)


# ============================================================
# SUMMARY
# ============================================================
print('\n=== SUMMARY ===\n')
if results:
    mean_full = np.mean([r['err_full'] for r in results])
    mean_lr = np.mean([r['err_lr'] for r in results])
    mean_dd = np.mean([r['err_dd'] for r in results])
    mean_z = np.mean([r['err_z'] for r in results])
    mean_r1 = np.mean([r['r_f1'] for r in results])
    mean_r2 = np.mean([r['r_f2'] for r in results])

    print(f'Held-out results ({len(results)} images):')
    print(f'  Zero:       {mean_z:.2f}')
    print(f'  LowRank 2D: {mean_lr:.2f} (gap={(1-mean_lr/mean_z)*100:.1f}%)')
    print(f'  Full 2D:    {mean_full:.2f} (gap={(1-mean_full/mean_z)*100:.1f}%)')
    print(f'  DDColor:    {mean_dd:.2f} (gap={(1-mean_dd/mean_z)*100:.1f}%)')
    print(f'  Field correlation: F1 r={mean_r1:.6f}, F2 r={mean_r2:.6f}')
    print()
    print(f'The low-rank encoder uses:')
    print(f'  - Stem: rank-{stem_rank} (13 basis filters instead of 96)')
    print(f'  - DW convs: rank-3 to rank-7 (instead of 96-768)')
    print(f'  - All pointwise convs, norms, gammas UNCHANGED')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
