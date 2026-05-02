"""
Spectrometer Mutations: Probing the Encoder's Semantic Database

Now that we understand the SSM structure, we probe the ACTUAL encoder:
1. Randomize the spectrometer (keep spatial filters) — what breaks?
2. Swap spectrometers between stages — are they interchangeable?
3. Interpolate between real and random — where does it break?
4. Transpose the spectrometer (reverse expand/compress) — symmetry test
5. Low-rank approximate the spectrometer — is it compressible after all?
6. Zero out the spectrometer — what does pure spatial filtering give?
"""
import numpy as np
import cv2
import sys
import os
import glob
import torch
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=== SPECTROMETER MUTATIONS ===\n')
v16 = V16GeometricColorizer()

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/spectrometer_mutations'
os.makedirs(out_dir, exist_ok=True)


def run_encoder_mutated(v16, x, mutations):
    """Run encoder with mutated pointwise convolutions."""
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

            # Get mutated or original pw weights
            key1 = f'{prefix}.pwconv1.weight'
            key2 = f'{prefix}.pwconv2.weight'
            pw1_w = mutations.get(key1, v16._get_weight(key1))
            pw1_b = v16._get_weight(f'{prefix}.pwconv1.bias')
            pw2_w = mutations.get(key2, v16._get_weight(key2))
            pw2_b = v16._get_weight(f'{prefix}.pwconv2.bias')

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


def get_features_full(v16, img_tensor):
    return get_features_mutated(v16, img_tensor, {})


# Build color basis from training images
print('Building color basis...')
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
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features_full(v16, img_tensor)
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

from numpy.linalg import lstsq
proj_1 = (all_enc - enc_mean) @ color_dir_1
proj_2 = (all_enc - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt[:, 1], rcond=None)

def predict_color(enc_features):
    flat = (enc_features.reshape(256, -1).T - enc_mean)
    fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
    ab = np.stack([
        np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
        np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
    ], axis=2)
    return ab

def field_corr(enc1, enc2):
    f1a = ((enc1.reshape(256, -1).T - enc_mean) @ color_dir_1)
    f1b = ((enc2.reshape(256, -1).T - enc_mean) @ color_dir_1)
    f2a = ((enc1.reshape(256, -1).T - enc_mean) @ color_dir_2)
    f2b = ((enc2.reshape(256, -1).T - enc_mean) @ color_dir_2)
    r1 = np.corrcoef(f1a, f1b)[0, 1] if np.std(f1a) > 1e-8 and np.std(f1b) > 1e-8 else 0
    r2 = np.corrcoef(f2a, f2b)[0, 1] if np.std(f2a) > 1e-8 and np.std(f2b) > 1e-8 else 0
    return r1, r2


# ================================================================
# Use a single test image for all mutations
# ================================================================
test_idx = 85
im = cv2.imread(all_imgs[test_idx])
r_img = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
lab_gt = cv2.cvtColor(r_img, cv2.COLOR_BGR2Lab)
L = lab_gt[:,:,0]
ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
err_z = np.sqrt(np.mean(ab_gt**2))

# Full encoder baseline
enc_full = get_features_full(v16, img_tensor)
ab_full = predict_color(enc_full)
err_full = np.sqrt(np.mean((ab_full - ab_gt)**2))
print(f'\nBaseline: Full encoder error = {err_full:.2f}, Zero = {err_z:.2f}')


# ================================================================
# MUTATION 1: Randomize ALL spectrometers
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 1: RANDOMIZE ALL SPECTROMETERS')
print('=' * 70)
print('Replace all pw1/pw2 with random matrices (same shape, same scale)')

mutations_random = {}
for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()

        # Random with same scale
        r1 = np.random.randn(*pw1.shape) * np.std(pw1)
        r2 = np.random.randn(*pw2.shape) * np.std(pw2)
        mutations_random[f'{prefix}.pwconv1.weight'] = torch.from_numpy(r1).float()
        mutations_random[f'{prefix}.pwconv2.weight'] = torch.from_numpy(r2).float()

enc_rand = get_features_mutated(v16, img_tensor, mutations_random)
r1, r2 = field_corr(enc_full, enc_rand)
ab_rand = predict_color(enc_rand)
err_rand = np.sqrt(np.mean((ab_rand - ab_gt)**2))
print(f'  Error: {err_rand:.2f} (full={err_full:.2f}, zero={err_z:.2f})')
print(f'  Field corr: [{r1:.4f}, {r2:.4f}]')
print(f'  → Random spectrometer {"works" if err_rand < err_z else "BREAKS"}')


# ================================================================
# MUTATION 2: Zero out all spectrometers (pure spatial + residual)
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 2: ZERO ALL SPECTROMETERS')
print('=' * 70)
print('Set all pw1/pw2 to zero — only spatial filters + residual connections')

mutations_zero = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight')
        mutations_zero[f'{prefix}.pwconv1.weight'] = torch.zeros_like(pw1)
        mutations_zero[f'{prefix}.pwconv2.weight'] = torch.zeros_like(pw2)

enc_zero_pw = get_features_mutated(v16, img_tensor, mutations_zero)
r1, r2 = field_corr(enc_full, enc_zero_pw)
ab_zero_pw = predict_color(enc_zero_pw)
err_zero_pw = np.sqrt(np.mean((ab_zero_pw - ab_gt)**2))
print(f'  Error: {err_zero_pw:.2f} (full={err_full:.2f}, zero={err_z:.2f})')
print(f'  Field corr: [{r1:.4f}, {r2:.4f}]')
print(f'  → Pure spatial filtering: gap={(1-err_zero_pw/err_z)*100:.1f}%')


# ================================================================
# MUTATION 3: Interpolate real → random
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 3: INTERPOLATE REAL → RANDOM')
print('=' * 70)
print('At what mixing ratio does the spectrometer break?')

for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    mutations_interp = {}
    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            pw1_real = v16._get_weight(f'{prefix}.pwconv1.weight')
            pw2_real = v16._get_weight(f'{prefix}.pwconv2.weight')
            pw1_rand = mutations_random[f'{prefix}.pwconv1.weight']
            pw2_rand = mutations_random[f'{prefix}.pwconv2.weight']

            mutations_interp[f'{prefix}.pwconv1.weight'] = (1-alpha) * pw1_real + alpha * pw1_rand
            mutations_interp[f'{prefix}.pwconv2.weight'] = (1-alpha) * pw2_real + alpha * pw2_rand

    enc_interp = get_features_mutated(v16, img_tensor, mutations_interp)
    r1, r2 = field_corr(enc_full, enc_interp)
    ab_interp = predict_color(enc_interp)
    err_interp = np.sqrt(np.mean((ab_interp - ab_gt)**2))
    gap = (1 - err_interp/err_z) * 100
    print(f'  α={alpha:.1f}: err={err_interp:.2f}, gap={gap:+.1f}%, '
          f'field_r=[{r1:.3f},{r2:.3f}]')


# ================================================================
# MUTATION 4: Mutate only ONE stage
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 4: RANDOMIZE ONE STAGE AT A TIME')
print('=' * 70)
print('Which stage\'s spectrometer matters most?')

for target_stage in range(4):
    mutations_one = {}
    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            if stage_idx == target_stage:
                mutations_one[f'{prefix}.pwconv1.weight'] = mutations_random[f'{prefix}.pwconv1.weight']
                mutations_one[f'{prefix}.pwconv2.weight'] = mutations_random[f'{prefix}.pwconv2.weight']

    enc_one = get_features_mutated(v16, img_tensor, mutations_one)
    r1, r2 = field_corr(enc_full, enc_one)
    ab_one = predict_color(enc_one)
    err_one = np.sqrt(np.mean((ab_one - ab_gt)**2))
    gap = (1 - err_one/err_z) * 100
    print(f'  Randomize Stage {target_stage} ({dims[target_stage]:3d}ch, {depths[target_stage]} blocks): '
          f'err={err_one:.2f}, gap={gap:+.1f}%, r=[{r1:.3f},{r2:.3f}]')


# ================================================================
# MUTATION 5: Transpose the spectrometer
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 5: TRANSPOSE — Swap expand/compress roles')
print('=' * 70)
print('If pw1 expands and pw2 compresses, what if we swap them?')

mutations_transpose = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight')
        # Swap: use pw2^T as the new expand, pw1^T as the new compress
        mutations_transpose[f'{prefix}.pwconv1.weight'] = pw2.T.contiguous()
        mutations_transpose[f'{prefix}.pwconv2.weight'] = pw1.T.contiguous()

enc_trans = get_features_mutated(v16, img_tensor, mutations_transpose)
r1, r2 = field_corr(enc_full, enc_trans)
ab_trans = predict_color(enc_trans)
err_trans = np.sqrt(np.mean((ab_trans - ab_gt)**2))
gap = (1 - err_trans/err_z) * 100
print(f'  Transposed: err={err_trans:.2f}, gap={gap:+.1f}%, r=[{r1:.3f},{r2:.3f}]')
print(f'  Original:   err={err_full:.2f}, gap={(1-err_full/err_z)*100:+.1f}%')


# ================================================================
# MUTATION 6: Low-rank approximate the spectrometer
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 6: LOW-RANK SPECTROMETER')
print('=' * 70)
print('Can we compress the spectrometer like we compressed the spatial filters?')

for var_target in [0.50, 0.80, 0.90, 0.95, 0.99]:
    mutations_lr = {}
    total_orig = 0
    total_reduced = 0

    for stage_idx in range(4):
        dim = dims[stage_idx]
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            for pw_name in ['pwconv1', 'pwconv2']:
                w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
                U, S, Vt = np.linalg.svd(w, full_matrices=False)
                cumvar = np.cumsum(S**2) / (S**2).sum()
                k = np.searchsorted(cumvar, var_target) + 1
                k = min(k, len(S))
                approx = (U[:, :k] * S[:k]) @ Vt[:k]
                mutations_lr[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()
                total_orig += len(S)
                total_reduced += k

    enc_lr = get_features_mutated(v16, img_tensor, mutations_lr)
    r1, r2 = field_corr(enc_full, enc_lr)
    ab_lr = predict_color(enc_lr)
    err_lr = np.sqrt(np.mean((ab_lr - ab_gt)**2))
    gap = (1 - err_lr/err_z) * 100
    ratio = total_reduced / total_orig
    print(f'  var={var_target*100:.0f}%: err={err_lr:.2f}, gap={gap:+.1f}%, '
          f'r=[{r1:.3f},{r2:.3f}], rank_ratio={ratio:.2f}')


# ================================================================
# MUTATION 7: Scale the spectrometer output
# ================================================================
print('\n' + '=' * 70)
print('MUTATION 7: AMPLIFY/ATTENUATE THE SPECTROMETER')
print('=' * 70)
print('Scale pw2 output — does the spectrometer have a "volume" control?')

for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
    mutations_scale = {}
    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            pw2 = v16._get_weight(f'{prefix}.pwconv2.weight')
            mutations_scale[f'{prefix}.pwconv2.weight'] = pw2 * scale

    enc_scale = get_features_mutated(v16, img_tensor, mutations_scale)
    r1, r2 = field_corr(enc_full, enc_scale)
    ab_scale = predict_color(enc_scale)
    err_scale = np.sqrt(np.mean((ab_scale - ab_gt)**2))
    gap = (1 - err_scale/err_z) * 100

    # Check if output is finite
    if np.isnan(err_scale) or np.isinf(err_scale):
        print(f'  scale={scale:.2f}: DIVERGED')
    else:
        print(f'  scale={scale:.2f}: err={err_scale:.2f}, gap={gap:+.1f}%, '
              f'r=[{r1:.3f},{r2:.3f}]')


# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 70)
print('SUMMARY: SPECTROMETER MUTATION RESULTS')
print('=' * 70)
print(f"""
Baseline: Full encoder = {err_full:.2f}, Zero = {err_z:.2f}

Key findings from mutations:
""")

print('Done!')
