"""
Two Frontier Questions:

Q1: Are the pointwise convolutions (channel mixing) φ-structured?
    Each ConvNeXt block has:
      pwconv1: dim → 4*dim (expand)
      pwconv2: 4*dim → dim (compress)
    These are the channel mixing operations. Analyze their SVD structure.

Q2: Can we derive the ~23 DW basis filters from first principles?
    Extract the SVD basis vectors of each depthwise conv bank.
    Compare to canonical geometric filters (Gabor, DoG, Laplacian, etc.)
    Test if a handcrafted basis can replace the extracted one.
"""
import numpy as np
import cv2
import sys
import os
import glob
import torch
import torch.nn.functional as F
from scipy import signal

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

print('=== FRONTIER ANALYSIS: Pointwise + First-Principles Basis ===\n')

v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/frontier_analysis'
os.makedirs(out_dir, exist_ok=True)


# ================================================================
# Q1: POINTWISE CONVOLUTION ANALYSIS
# ================================================================
print('=' * 60)
print('Q1: ARE POINTWISE CONVOLUTIONS φ-STRUCTURED?')
print('=' * 60)

print('\n--- pwconv1 (expand: dim → 4*dim) ---\n')
print(f'{"Layer":<20} {"Shape":<20} {"Rank90":>7} {"Rank95":>7} {"Rank99":>7} '
      f'{"S0/S1":>8} {"φ_err":>7} {"S1/S2":>8} {"φ_err":>7}')
print('-' * 105)

pw1_phi_ratios = []
pw1_ranks = []

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()  # [4*dim, dim]

        U, S, Vt = np.linalg.svd(w, full_matrices=False)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        r90 = np.searchsorted(cumvar, 0.90) + 1
        r95 = np.searchsorted(cumvar, 0.95) + 1
        r99 = np.searchsorted(cumvar, 0.99) + 1

        ratio_01 = S[0] / S[1] if S[1] > 1e-10 else float('inf')
        ratio_12 = S[1] / S[2] if S[2] > 1e-10 else float('inf')
        err_01 = abs(ratio_01 - PHI) / PHI * 100
        err_12 = abs(ratio_12 - PHI) / PHI * 100

        pw1_phi_ratios.append(ratio_01)
        pw1_ranks.append(r90)

        name = f'S{stage_idx}.B{block_idx}'
        shape = f'{w.shape}'
        print(f'{name:<20} {shape:<20} {r90:7d} {r95:7d} {r99:7d} '
              f'{ratio_01:8.4f} {err_01:6.1f}% {ratio_12:8.4f} {err_12:6.1f}%')


print(f'\npwconv1 summary:')
print(f'  Mean S[0]/S[1]: {np.mean(pw1_phi_ratios):.4f} (φ err: {abs(np.mean(pw1_phi_ratios)-PHI)/PHI*100:.1f}%)')
print(f'  Mean rank90: {np.mean(pw1_ranks):.1f}')

print('\n--- pwconv2 (compress: 4*dim → dim) ---\n')
print(f'{"Layer":<20} {"Shape":<20} {"Rank90":>7} {"Rank95":>7} {"Rank99":>7} '
      f'{"S0/S1":>8} {"φ_err":>7} {"S1/S2":>8} {"φ_err":>7}')
print('-' * 105)

pw2_phi_ratios = []
pw2_ranks = []

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        w = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()  # [dim, 4*dim]

        U, S, Vt = np.linalg.svd(w, full_matrices=False)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        r90 = np.searchsorted(cumvar, 0.90) + 1
        r95 = np.searchsorted(cumvar, 0.95) + 1
        r99 = np.searchsorted(cumvar, 0.99) + 1

        ratio_01 = S[0] / S[1] if S[1] > 1e-10 else float('inf')
        ratio_12 = S[1] / S[2] if S[2] > 1e-10 else float('inf')
        err_01 = abs(ratio_01 - PHI) / PHI * 100
        err_12 = abs(ratio_12 - PHI) / PHI * 100

        pw2_phi_ratios.append(ratio_01)
        pw2_ranks.append(r90)

        name = f'S{stage_idx}.B{block_idx}'
        shape = f'{w.shape}'
        print(f'{name:<20} {shape:<20} {r90:7d} {r95:7d} {r99:7d} '
              f'{ratio_01:8.4f} {err_01:6.1f}% {ratio_12:8.4f} {err_12:6.1f}%')

print(f'\npwconv2 summary:')
print(f'  Mean S[0]/S[1]: {np.mean(pw2_phi_ratios):.4f} (φ err: {abs(np.mean(pw2_phi_ratios)-PHI)/PHI*100:.1f}%)')
print(f'  Mean rank90: {np.mean(pw2_ranks):.1f}')

# Also check gamma (layer scale) values
print('\n--- Layer Scale (γ) Analysis ---\n')
all_gammas = []
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        gamma = v16._get_weight(f'{prefix}.gamma').numpy()
        all_gammas.append(gamma)

        # Check if gamma magnitudes follow φ pattern
        g_sorted = np.sort(np.abs(gamma))[::-1]
        if len(g_sorted) > 1 and g_sorted[1] > 1e-10:
            top_ratio = g_sorted[0] / g_sorted[1]

all_gammas_flat = np.concatenate(all_gammas)
print(f'All γ values: mean={all_gammas_flat.mean():.4f}, std={all_gammas_flat.std():.4f}')
print(f'  Range: [{all_gammas_flat.min():.4f}, {all_gammas_flat.max():.4f}]')
print(f'  |γ| > 1: {np.sum(np.abs(all_gammas_flat) > 1)} / {len(all_gammas_flat)}')

# Check ratio of γ magnitudes between successive stages
for stage_idx in range(3):
    g1 = np.mean(np.abs(all_gammas[sum(depths[:stage_idx]):sum(depths[:stage_idx+1])]))
    g2 = np.mean(np.abs(all_gammas[sum(depths[:stage_idx+1]):sum(depths[:stage_idx+2])]))
    if g1 > 1e-10:
        ratio = g2 / g1
        phi_err = abs(ratio - PHI) / PHI * 100
        print(f'  |γ| stage{stage_idx+1}/stage{stage_idx} = {ratio:.4f} (φ err: {phi_err:.1f}%)')

# Downsample layer analysis
print('\n--- Downsample Convolutions (stride-2) ---\n')
for stage_idx in range(1, 4):
    prefix = f'encoder.arch.downsample_layers.{stage_idx}'
    w = v16._get_weight(f'{prefix}.1.weight').numpy()  # [dim_out, dim_in, 2, 2]
    W_2d = w.reshape(w.shape[0], -1)
    U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    r90 = np.searchsorted(cumvar, 0.90) + 1
    r99 = np.searchsorted(cumvar, 0.99) + 1

    ratio_01 = S[0] / S[1] if S[1] > 1e-10 else float('inf')
    err_01 = abs(ratio_01 - PHI) / PHI * 100

    print(f'  DS{stage_idx} {w.shape}: rank90={r90}/{min(w.shape[0], W_2d.shape[1])}, '
          f'rank99={r99}, S[0]/S[1]={ratio_01:.4f} (φ err: {err_01:.1f}%)')


# ================================================================
# Q2: FIRST-PRINCIPLES BASIS FILTERS
# ================================================================
print('\n' + '=' * 60)
print('Q2: CAN WE DERIVE THE BASIS FROM FIRST PRINCIPLES?')
print('=' * 60)

# Extract the actual SVD basis vectors from the learned depthwise convolutions
# Then compare them to canonical geometric filters

def make_canonical_7x7_basis(n_filters=49):
    """Generate canonical 7×7 geometric filter basis."""
    filters = {}
    y, x = np.mgrid[-3:4, -3:4].astype(float)
    r = np.sqrt(x**2 + y**2)

    # 1. DC (mean)
    filters['DC'] = np.ones((7, 7)) / 49.0

    # 2. Gaussian blobs at different scales
    for sigma in [1.0, 2.0, 3.0]:
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        filters[f'gauss_s{sigma}'] = g / g.sum()

    # 3. Difference of Gaussians (DoG) — blob detectors
    for s1, s2 in [(1.0, 2.0), (1.5, 3.0), (0.8, 1.6)]:
        g1 = np.exp(-(x**2 + y**2) / (2 * s1**2))
        g2 = np.exp(-(x**2 + y**2) / (2 * s2**2))
        dog = g1/g1.sum() - g2/g2.sum()
        filters[f'DoG_{s1}_{s2}'] = dog

    # 4. Laplacian of Gaussian
    for sigma in [1.0, 2.0]:
        log = (x**2 + y**2 - 2*sigma**2) / sigma**4 * np.exp(-(x**2+y**2)/(2*sigma**2))
        filters[f'LoG_s{sigma}'] = log

    # 5. Oriented first derivatives (edge detectors) at multiple orientations
    for angle_deg in [0, 45, 90, 135]:
        theta = np.radians(angle_deg)
        dx = x * np.cos(theta) + y * np.sin(theta)
        g = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
        filters[f'edge_{angle_deg}'] = dx * g

    # 6. Oriented second derivatives at multiple orientations
    for angle_deg in [0, 45, 90, 135]:
        theta = np.radians(angle_deg)
        dx = x * np.cos(theta) + y * np.sin(theta)
        g = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
        filters[f'edge2_{angle_deg}'] = (dx**2 - 2.0**2) * g

    # 7. Gabor filters at multiple orientations and frequencies
    for angle_deg in [0, 45, 90, 135]:
        theta = np.radians(angle_deg)
        for freq in [0.3, 0.5]:
            dx = x * np.cos(theta) + y * np.sin(theta)
            dy = -x * np.sin(theta) + y * np.cos(theta)
            g = np.exp(-(dx**2 + dy**2) / (2 * 2.0**2))
            gabor = g * np.cos(2 * np.pi * freq * dx)
            filters[f'gabor_{angle_deg}_{freq}'] = gabor

    # 8. Center-surround at different scales
    for r_inner in [1, 2]:
        cs = np.zeros((7, 7))
        cs[r <= r_inner] = 1.0
        cs[r > r_inner] = -1.0 * (r[r <= r_inner].size / max(1, r[r > r_inner].size))
        filters[f'cs_r{r_inner}'] = cs

    # 9. Quadrant filters
    filters['quad_tl'] = np.where((x <= 0) & (y <= 0), 1.0, -1.0/3)
    filters['quad_tr'] = np.where((x >= 0) & (y <= 0), 1.0, -1.0/3)
    filters['quad_bl'] = np.where((x <= 0) & (y >= 0), 1.0, -1.0/3)
    filters['quad_br'] = np.where((x >= 0) & (y >= 0), 1.0, -1.0/3)

    # Normalize all
    for k, v in filters.items():
        norm = np.sqrt(np.sum(v**2))
        if norm > 1e-8:
            filters[k] = v / norm

    return filters


canonical = make_canonical_7x7_basis()
print(f'\nCanonical basis: {len(canonical)} filters')

# For each stage, extract SVD bases and compare to canonical
print('\n--- SVD Basis vs Canonical Basis ---\n')

for stage_idx in [0, 2, 3]:
    dim = dims[stage_idx]
    # Use first block
    prefix = f'encoder.arch.stages.{stage_idx}.0'
    dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
    kernels_flat = dw_w.squeeze(1).reshape(dim, -1)
    U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    rank99 = np.searchsorted(cumvar, 0.99) + 1

    print(f'Stage {stage_idx} Block 0 ({dim}ch, rank99={rank99}):')

    # For each SVD basis vector, find best matching canonical filter
    canonical_names = list(canonical.keys())
    canonical_mat = np.array([canonical[k].flatten() for k in canonical_names])
    # Normalize canonical
    canonical_norms = np.sqrt(np.sum(canonical_mat**2, axis=1, keepdims=True))
    canonical_mat_n = canonical_mat / (canonical_norms + 1e-8)

    total_explained = 0
    for mode_idx in range(min(rank99, 30)):
        basis = Vt[mode_idx]  # [49]
        basis_n = basis / (np.sqrt(np.sum(basis**2)) + 1e-8)

        # Correlation with each canonical filter
        corrs = canonical_mat_n @ basis_n
        best_idx = np.argmax(np.abs(corrs))
        best_corr = corrs[best_idx]
        best_name = canonical_names[best_idx]

        var_pct = (S[mode_idx]**2 / (S**2).sum()) * 100
        total_explained += var_pct

        marker = ''
        if abs(best_corr) > 0.9:
            marker = ' ★★★'
        elif abs(best_corr) > 0.7:
            marker = ' ★★'
        elif abs(best_corr) > 0.5:
            marker = ' ★'

        if mode_idx < 10 or abs(best_corr) > 0.7:
            print(f'  Mode {mode_idx:2d} ({var_pct:5.2f}%, cum={total_explained:5.1f}%): '
                  f'best={best_name:<18s} r={best_corr:+.3f}{marker}')

    # Overall: what fraction of basis vectors have canonical matches?
    n_good = 0
    for mode_idx in range(rank99):
        basis_n = Vt[mode_idx] / (np.sqrt(np.sum(Vt[mode_idx]**2)) + 1e-8)
        corrs = canonical_mat_n @ basis_n
        if np.max(np.abs(corrs)) > 0.7:
            n_good += 1

    print(f'  Canonical match (|r|>0.7): {n_good}/{rank99} ({n_good/rank99*100:.0f}%)')

    # Try to reconstruct kernels using ONLY canonical basis
    # Project each kernel onto canonical basis, reconstruct, measure error
    kernels = dw_w.squeeze(1).reshape(dim, 49)
    # Project: coeffs = kernels @ canonical_mat.T (pseudo-inverse)
    # Using least squares for each kernel
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(canonical_mat.T, kernels.T, rcond=None)
    # coeffs: [n_canonical, dim]
    reconstructed = (canonical_mat.T @ coeffs).T  # [dim, 49]

    recon_err = np.sqrt(np.mean((kernels - reconstructed)**2))
    orig_norm = np.sqrt(np.mean(kernels**2))
    rel_err = recon_err / orig_norm

    # How much variance is captured?
    var_captured = 1 - np.sum((kernels - reconstructed)**2) / np.sum((kernels - kernels.mean(axis=0))**2)

    print(f'  Canonical reconstruction: rel_err={rel_err:.4f}, var_captured={var_captured*100:.1f}%')
    print()

# ================================================================
# Q2b: Test canonical basis encoder
# ================================================================
print('=' * 60)
print('Q2b: CANONICAL BASIS ENCODER TEST')
print('=' * 60)
print()

# Build lowrank kernels using canonical basis projection instead of SVD
def build_canonical_kernels(v16, canonical_filters):
    """Replace DW convs with canonical basis reconstruction."""
    canonical_names = list(canonical_filters.keys())
    canonical_mat = np.array([canonical_filters[k].flatten() for k in canonical_names])

    lowrank = {}

    # Stem: keep as-is (it's 4x4, canonical is 7x7)
    lowrank['stem'] = v16._get_weight('encoder.arch.downsample_layers.0.0.weight')

    for stage_idx in range(4):
        dim = dims[stage_idx]
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
            kernels = dw_w.squeeze(1).reshape(dim, 49)

            # Project onto canonical basis
            from numpy.linalg import lstsq
            coeffs, _, _, _ = lstsq(canonical_mat.T, kernels.T, rcond=None)
            reconstructed = (canonical_mat.T @ coeffs).T
            reconstructed = reconstructed.reshape(dw_w.shape)

            lowrank[f'{prefix}.dwconv.weight'] = torch.from_numpy(reconstructed).float()

    return lowrank


def run_encoder_lowrank(v16, x, lowrank_kernels):
    """Run encoder with kernel replacements."""
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


SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

# Build canonical kernel encoder
canon_kernels = build_canonical_kernels(v16, canonical)

# Build color basis from full encoder
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

from numpy.linalg import lstsq
proj_1 = (all_enc - enc_mean) @ color_dir_1
proj_2 = (all_enc - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt[:, 1], rcond=None)

# Test on held-out images
print('Testing canonical basis encoder vs full encoder...\n')
test_indices = list(range(80, 95))
results = []

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # Full encoder
    enc_full = get_features(v16, img_tensor, lowrank=None)
    flat_full = (enc_full.reshape(256, -1).T - enc_mean)
    f1_full = (flat_full @ color_dir_1).reshape(SZ, SZ)
    f2_full = (flat_full @ color_dir_2).reshape(SZ, SZ)

    # Canonical basis encoder
    enc_canon = get_features(v16, img_tensor, lowrank=canon_kernels)
    flat_canon = (enc_canon.reshape(256, -1).T - enc_mean)
    f1_canon = (flat_canon @ color_dir_1).reshape(SZ, SZ)
    f2_canon = (flat_canon @ color_dir_2).reshape(SZ, SZ)

    # Field correlations
    r1 = np.corrcoef(f1_full.flatten(), f1_canon.flatten())[0, 1]
    r2 = np.corrcoef(f2_full.flatten(), f2_canon.flatten())[0, 1]

    # Color errors
    err_z = np.sqrt(np.mean(ab_gt**2))

    fields_full = np.column_stack([f1_full.flatten(), f2_full.flatten(), np.ones(SZ*SZ)])
    ab_full = np.stack([np.clip(fields_full @ W_a, -50, 50).reshape(SZ, SZ),
                        np.clip(fields_full @ W_b, -50, 50).reshape(SZ, SZ)], axis=2)
    err_full = np.sqrt(np.mean((ab_full - ab_gt)**2))

    fields_canon = np.column_stack([f1_canon.flatten(), f2_canon.flatten(), np.ones(SZ*SZ)])
    ab_canon = np.stack([np.clip(fields_canon @ W_a, -50, 50).reshape(SZ, SZ),
                         np.clip(fields_canon @ W_b, -50, 50).reshape(SZ, SZ)], axis=2)
    err_canon = np.sqrt(np.mean((ab_canon - ab_gt)**2))

    print(f'  {name}: Full={err_full:.1f} Canon={err_canon:.1f} Zero={err_z:.1f} | r=[{r1:.4f},{r2:.4f}]')
    results.append({'err_full': err_full, 'err_canon': err_canon, 'err_z': err_z, 'r1': r1, 'r2': r2})


# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)

print('\nQ1: Pointwise Convolution φ-Structure:')
print(f'  pwconv1 mean S[0]/S[1]: {np.mean(pw1_phi_ratios):.4f} (φ err: {abs(np.mean(pw1_phi_ratios)-PHI)/PHI*100:.1f}%)')
print(f'  pwconv2 mean S[0]/S[1]: {np.mean(pw2_phi_ratios):.4f} (φ err: {abs(np.mean(pw2_phi_ratios)-PHI)/PHI*100:.1f}%)')
print(f'  pwconv1 mean rank90: {np.mean(pw1_ranks):.1f}')
print(f'  pwconv2 mean rank90: {np.mean(pw2_ranks):.1f}')

if results:
    mean_full = np.mean([r['err_full'] for r in results])
    mean_canon = np.mean([r['err_canon'] for r in results])
    mean_z = np.mean([r['err_z'] for r in results])
    mean_r1 = np.mean([r['r1'] for r in results])
    mean_r2 = np.mean([r['r2'] for r in results])

    print(f'\nQ2: Canonical Basis Encoder:')
    print(f'  Zero:      {mean_z:.2f}')
    print(f'  Canonical: {mean_canon:.2f} (gap={(1-mean_canon/mean_z)*100:.1f}%)')
    print(f'  Full:      {mean_full:.2f} (gap={(1-mean_full/mean_z)*100:.1f}%)')
    print(f'  Field correlation: F1={mean_r1:.4f}, F2={mean_r2:.4f}')
    print(f'  The canonical basis uses {len(canonical)} handcrafted geometric filters')
    print(f'  (Gabor, DoG, LoG, oriented edges, center-surround, quadrants)')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
