"""
SSM Optimal Compression: Per-Stage Variance-Targeted Truncation

From the rank sweep, optimal per-stage variance targets:
  Stage 0: 70% variance (rank_ratio=0.35) — 96ch, 3 blocks
  Stage 1: 70% variance (rank_ratio=0.36) — 192ch, 3 blocks
  Stage 2: 70% variance (rank_ratio=0.36) — 384ch, 9 blocks
  Stage 3: 50% variance (rank_ratio=0.23) — 768ch, 3 blocks

This script:
1. Tests the combined optimal compression on 30+ images
2. Computes exact parameter savings
3. Tests aggressive variants (Stage 3 at 30%, 40%)
4. Tests boosted variants (S0-S1 at 80%, S2 at 70%, S3 at 50%)
5. What's the MINIMUM per-stage that still beats zero?
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

print('=== SSM OPTIMAL COMPRESSION ===\n')
v16 = V16GeometricColorizer()

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_mutated(v16, x, mutations):
    features = []
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
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            residual = x
            x = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                         v16._get_weight(f'{prefix}.dwconv.bias'), padding=3, groups=dim)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))

            pw1_w = mutations.get(f'{prefix}.pwconv1.weight', v16._get_weight(f'{prefix}.pwconv1.weight'))
            pw1_b = mutations.get(f'{prefix}.pwconv1.bias', v16._get_weight(f'{prefix}.pwconv1.bias'))
            pw2_w = mutations.get(f'{prefix}.pwconv2.weight', v16._get_weight(f'{prefix}.pwconv2.weight'))
            pw2_b = mutations.get(f'{prefix}.pwconv2.bias', v16._get_weight(f'{prefix}.pwconv2.bias'))

            x = F.linear(x, pw1_w, pw1_b)
            x = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
            x = F.linear(x, pw2_w, pw2_b)
            x = x.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * x

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))
    return features


def get_features_mutated(v16, img_tensor, mutations):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    with torch.no_grad():
        features = run_encoder_mutated(v16, x, mutations)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()


# Build color basis
print('Building color basis...')
from numpy.linalg import lstsq

train_indices = list(range(50, 70))
all_enc = []
all_gt = []
for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    if np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean() < 2: continue
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features_mutated(v16, img_tensor, {})
    flat = enc.reshape(256, -1).T
    sample = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_enc.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc_arr = np.vstack(all_enc)
all_gt_arr = np.vstack(all_gt)
enc_mean = all_enc_arr.mean(axis=0)
C = (all_enc_arr - enc_mean).T @ all_gt_arr / len(all_enc_arr)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dir_1, color_dir_2 = U_color[:, 0], U_color[:, 1]

proj_1 = (all_enc_arr - enc_mean) @ color_dir_1
proj_2 = (all_enc_arr - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt_arr[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt_arr[:, 1], rcond=None)


def predict_color(enc_features):
    flat = (enc_features.reshape(256, -1).T - enc_mean)
    fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
    return np.stack([
        np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
        np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
    ], axis=2)


def evaluate_images(mutations, indices=None, n_max=30):
    if indices is None:
        indices = list(range(80, 160))
    gaps = []
    for idx in indices:
        if len(gaps) >= n_max: break
        if idx >= len(all_imgs): break
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
        err_z = np.sqrt(np.mean(ab_gt**2))
        if err_z < 2: continue
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        enc = get_features_mutated(v16, img_tensor, mutations)
        ab_pred = predict_color(enc)
        err = np.sqrt(np.mean((ab_pred - ab_gt)**2))
        gaps.append((1 - err/err_z) * 100)
    return gaps


def build_lr_mutations(var_targets):
    """Build mutations dictionary for per-stage variance targets.
    var_targets: dict mapping stage_idx -> variance target (0-1)
    Returns: mutations dict, param stats
    """
    muts = {}
    stats = {'total_orig': 0, 'total_compressed': 0, 'per_stage': {}}

    for stage_idx in range(4):
        dim = dims[stage_idx]
        var_target = var_targets.get(stage_idx, 1.0)  # 1.0 = full rank
        stage_orig = 0
        stage_comp = 0

        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            for pw_name in ['pwconv1', 'pwconv2']:
                w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
                b = v16._get_weight(f'{prefix}.{pw_name}.bias').numpy()

                orig_params = w.size + b.size
                stage_orig += orig_params
                stats['total_orig'] += orig_params

                if var_target < 1.0:
                    U, S, Vt = np.linalg.svd(w, full_matrices=False)
                    cumvar = np.cumsum(S**2) / (S**2).sum()
                    k = np.searchsorted(cumvar, var_target) + 1
                    k = min(k, len(S))
                    approx = (U[:, :k] * S[:k]) @ Vt[:k]
                    muts[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()
                    # Stored as U[:,:k], S[:k], Vt[:k,:], bias
                    comp_params = w.shape[0] * k + k + k * w.shape[1] + b.size
                else:
                    comp_params = orig_params

                stage_comp += comp_params
                stats['total_compressed'] += comp_params

        stats['per_stage'][stage_idx] = {
            'orig': stage_orig, 'compressed': stage_comp,
            'ratio': stage_comp / stage_orig if stage_orig > 0 else 1.0,
            'var_target': var_target
        }

    return muts, stats


# ================================================================
# PART 1: Test multiple compression strategies on 30 images
# ================================================================
print()
print('=' * 70)
print('PART 1: COMPRESSION STRATEGY COMPARISON (30 images)')
print('=' * 70)
print()

strategies = {
    'Full encoder (baseline)': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    'LR90 all stages': {0: 0.90, 1: 0.90, 2: 0.90, 3: 0.90},
    'Optimal A: 70/70/70/50': {0: 0.70, 1: 0.70, 2: 0.70, 3: 0.50},
    'Optimal B: 80/80/70/50': {0: 0.80, 1: 0.80, 2: 0.70, 3: 0.50},
    'Aggressive: 70/70/70/30': {0: 0.70, 1: 0.70, 2: 0.70, 3: 0.30},
    'Ultra-aggressive: 50/50/50/30': {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.30},
    'Conservative: 80/80/80/70': {0: 0.80, 1: 0.80, 2: 0.80, 3: 0.70},
    'Stage3-only: 100/100/100/50': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.50},
}

print(f'{"Strategy":<35} {"Gap%":<10} {"Std":<8} {"Params":<12} {"Ratio":<8}')
print('-' * 75)

results = {}
for name, var_targets in strategies.items():
    muts, stats = build_lr_mutations(var_targets)
    gaps = evaluate_images(muts)
    results[name] = {
        'mean': np.mean(gaps), 'std': np.std(gaps),
        'params': stats['total_compressed'], 'orig': stats['total_orig'],
        'per_stage': stats['per_stage']
    }
    ratio = stats['total_compressed'] / stats['total_orig']
    print(f'  {name:<33} {np.mean(gaps):+5.1f}%    {np.std(gaps):5.1f}   '
          f'{stats["total_compressed"]:>10,}  {ratio:.2f}')

# Also test zero baseline
gaps_zero_muts = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw_name}.weight')
            gaps_zero_muts[f'{prefix}.{pw_name}.weight'] = torch.zeros_like(w)
gaps_zero = evaluate_images(gaps_zero_muts)
print(f'  {"Zero spectrometer":<33} {np.mean(gaps_zero):+5.1f}%    {np.std(gaps_zero):5.1f}')


# ================================================================
# PART 2: Per-stage parameter breakdown for best strategies
# ================================================================
print()
print('=' * 70)
print('PART 2: PARAMETER BREAKDOWN FOR BEST STRATEGIES')
print('=' * 70)
print()

for name in ['Optimal A: 70/70/70/50', 'Optimal B: 80/80/70/50', 'Conservative: 80/80/80/70']:
    r = results[name]
    print(f'{name}:')
    print(f'  Overall: {r["mean"]:+.1f}% gap, {r["params"]:,} params ({r["params"]/r["orig"]*100:.1f}%)')
    for s in range(4):
        ps = r['per_stage'][s]
        print(f'  Stage {s} ({dims[s]}ch, {depths[s]} blocks): '
              f'{ps["orig"]:>10,} → {ps["compressed"]:>10,} ({ps["ratio"]:.2f})')
    print()


# ================================================================
# PART 3: What's the absolute minimum per stage?
# ================================================================
print()
print('=' * 70)
print('PART 3: MINIMUM PER-STAGE VARIANCE TO STAY POSITIVE')
print('=' * 70)
print('Each stage at minimum variance, all others at 90%')
print()

for target_stage in range(4):
    print(f'  Stage {target_stage} ({dims[target_stage]}ch):')
    for var_target in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        var_targets = {0: 0.90, 1: 0.90, 2: 0.90, 3: 0.90}
        var_targets[target_stage] = var_target
        muts, stats = build_lr_mutations(var_targets)
        gaps = evaluate_images(muts, n_max=12)
        print(f'    var={var_target*100:3.0f}%: gap={np.mean(gaps):+6.1f}% ± {np.std(gaps):5.1f}%')
    print()


# ================================================================
# PART 4: Rank counts for optimal strategy
# ================================================================
print()
print('=' * 70)
print('PART 4: EXACT RANK COUNTS FOR OPTIMAL STRATEGIES')
print('=' * 70)
print()

for strat_name, var_targets in [
    ('Optimal A: 70/70/70/50', {0: 0.70, 1: 0.70, 2: 0.70, 3: 0.50}),
    ('Optimal B: 80/80/70/50', {0: 0.80, 1: 0.80, 2: 0.70, 3: 0.50}),
]:
    print(f'{strat_name}:')
    for stage_idx in range(4):
        dim = dims[stage_idx]
        var_target = var_targets[stage_idx]
        ranks_pw1 = []
        ranks_pw2 = []
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            for pw_name, rank_list in [('pwconv1', ranks_pw1), ('pwconv2', ranks_pw2)]:
                w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
                U, S, Vt = np.linalg.svd(w, full_matrices=False)
                cumvar = np.cumsum(S**2) / (S**2).sum()
                k = np.searchsorted(cumvar, var_target) + 1
                rank_list.append(k)

        full_rank = min(dim, dim * 4)
        print(f'  Stage {stage_idx} ({dim}ch): var={var_target*100:.0f}%, '
              f'pw1 ranks={ranks_pw1} (of {full_rank}), '
              f'pw2 ranks={ranks_pw2} (of {full_rank})')
    print()


# ================================================================
# PART 5: Per-block importance within Stage 2 (9 blocks)
# ================================================================
print()
print('=' * 70)
print('PART 5: STAGE 2 PER-BLOCK IMPORTANCE (9 blocks)')
print('=' * 70)
print('Which blocks in Stage 2 matter most?')
print()

for block_idx in range(9):
    muts = {}
    prefix = f'encoder.arch.stages.2.{block_idx}'
    for pw_name in ['pwconv1', 'pwconv2']:
        w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
        muts[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(
            np.random.randn(*w.shape) * np.std(w)).float()
    gaps = evaluate_images(muts, n_max=10)
    print(f'  Block {block_idx}: gap={np.mean(gaps):+5.1f}% ± {np.std(gaps):4.1f}% '
          f'(Δ={np.mean(gaps) - results["Full encoder (baseline)"]["mean"]:+.1f}%)')


# ================================================================
# GRAND SUMMARY
# ================================================================
print()
print('=' * 70)
print('GRAND SUMMARY — OPTIMAL COMPRESSION')
print('=' * 70)

full_gap = results['Full encoder (baseline)']['mean']
full_params = results['Full encoder (baseline)']['params']

print(f'\n  Full encoder: {full_gap:+.1f}% gap, {full_params:,} params')
print(f'  Zero spectrometer: {np.mean(gaps_zero):+.1f}% gap')
print()

best_name = None
best_ratio = 1.0
for name, r in results.items():
    if name == 'Full encoder (baseline)': continue
    ratio = r['params'] / r['orig']
    # Must be within 3% gap of full
    if r['mean'] >= full_gap - 3.0 and ratio < best_ratio:
        best_ratio = ratio
        best_name = name

if best_name:
    r = results[best_name]
    print(f'  BEST compression within 3% of full: {best_name}')
    print(f'    Gap: {r["mean"]:+.1f}%, Params: {r["params"]:,} ({r["params"]/r["orig"]*100:.1f}%)')
    print(f'    Savings: {r["orig"] - r["params"]:,} params removed')
    print(f'    Compression: {r["orig"]/r["params"]:.1f}× smaller')
else:
    print('  No compression strategy within 3% of full encoder')

print()
print('Done!')
