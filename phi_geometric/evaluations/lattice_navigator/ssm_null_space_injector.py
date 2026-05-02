"""
SSM Null-Space Injector: First-Principles with Correct Architecture

Discovery: W₂ is 65% in null(W₁). The SSM is a conditional orthogonal 
injector, not an autoencoder.

Previous first-principles used W₂ = W₁.T (pseudoinverse assumption).
This is WRONG — the real W₂ operates in a different subspace.

New approach: construct W₂ to be ORTHOGONAL to W₁'s column space.

Tests:
1. W₂ in null(W₁) — pure orthogonal injection
2. W₂ mixed: 35% range + 65% null (matching real ratio)
3. W₂ with φ-structured SVD in both spaces
4. Compare all approaches
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from numpy.linalg import lstsq

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=' * 70)
print('SSM NULL-SPACE INJECTOR: Correct First-Principles Architecture')
print('=' * 70)

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ================================================================
# INFRASTRUCTURE (reused from previous scripts)
# ================================================================

def run_encoder_custom(v16, img_tensor, weight_mutations=None, gate_fn=None):
    if weight_mutations is None:
        weight_mutations = {}
    if gate_fn is None:
        gate_fn = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    features = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                pre = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{pre}.0.weight'), v16._get_weight(f'{pre}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{pre}.1.weight'), v16._get_weight(f'{pre}.1.bias'), stride=2)
            for bi in range(depths[si]):
                pre = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{pre}.dwconv.weight'), v16._get_weight(f'{pre}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{pre}.norm.weight'), v16._get_weight(f'{pre}.norm.bias'))
                
                pw1_w = weight_mutations.get(f'{pre}.pwconv1.weight', v16._get_weight(f'{pre}.pwconv1.weight'))
                pw1_b = weight_mutations.get(f'{pre}.pwconv1.bias', v16._get_weight(f'{pre}.pwconv1.bias'))
                pw2_w = weight_mutations.get(f'{pre}.pwconv2.weight', v16._get_weight(f'{pre}.pwconv2.weight'))
                pw2_b = weight_mutations.get(f'{pre}.pwconv2.bias', v16._get_weight(f'{pre}.pwconv2.bias'))
                
                x = F.linear(x, pw1_w, pw1_b)
                x = gate_fn(x)
                x = F.linear(x, pw2_w, pw2_b)
                x = x.permute(0, 3, 1, 2)
                x = res + v16._get_weight(f'{pre}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0, 2, 3, 1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0, 3, 1, 2))
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()


print('\nBuilding color basis...')
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
    t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = run_encoder_custom(v16, t)
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


def evaluate(weight_mutations=None, gate_fn=None, n_images=12):
    if gate_fn is None:
        gate_fn = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    gaps = []
    for idx in range(80, 200):
        if len(gaps) >= n_images: break
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
        t = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        enc = run_encoder_custom(v16, t, weight_mutations, gate_fn)
        flat = (enc.reshape(256, -1).T - enc_mean)
        fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
        ab_pred = np.stack([
            np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
            np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
        ], axis=2)
        err = np.sqrt(np.mean((ab_pred - ab_gt)**2))
        gaps.append((1 - err/err_z) * 100)
    return np.mean(gaps), np.std(gaps)


# ================================================================
# FIRST-PRINCIPLES METHODS
# ================================================================

def make_mutations(method, seed=42):
    """Create weight mutations for ALL spectrometers."""
    np.random.seed(seed)
    muts = {}
    
    bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
    bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
    
    for stage_idx in range(4):
        dim = dims[stage_idx]
        dim_expand = dim * 4
        
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            
            if method == 'old_ortho_transpose':
                # Previous best: W₂ = W₁.T (WRONG assumption)
                M = np.random.randn(dim_expand, dim)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                W_expand = U @ Vt
                W_compress = W_expand.T
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'phi_svd_transpose':
                # Previous best: φ-structured SVD + W₂ = W₁.T
                k = min(dim, dim_expand)
                singular_values = np.array([PHI ** (-i * 0.5 / k) for i in range(k)])
                singular_values *= np.sqrt(k)
                U_rand = np.linalg.qr(np.random.randn(dim_expand, k))[0]
                V_rand = np.linalg.qr(np.random.randn(dim, k))[0]
                W_expand = U_rand * singular_values @ V_rand.T
                W_compress = V_rand * (1.0 / (singular_values + 1e-6)) @ U_rand.T
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'null_space_injector':
                # NEW: W₂ is constructed in null(W₁)
                # W₁ is [dim_expand, dim]
                # null(W₁.T) has dim (dim_expand - dim) dimensions
                M = np.random.randn(dim_expand, dim)
                U_full, _, Vt = np.linalg.svd(M, full_matrices=True)
                # U_full[:, :dim] = range(W₁), U_full[:, dim:] = null(W₁.T)
                W_expand = U_full[:, :dim] @ Vt  # [dim_expand, dim]
                
                # W₂ should be [dim, dim_expand]
                # Its rows should live in the null space of W₁.T
                # null(W₁.T) = span of U_full[:, dim:]
                U_null = U_full[:, dim:]  # [dim_expand, dim_expand - dim]
                
                # Generate random directions in null space
                R = np.random.randn(dim, dim_expand - dim)
                W_compress = R @ U_null.T  # [dim, dim_expand] — each row in null(W₁.T)
                
                # Scale to match real encoder norms
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'mixed_injector_65_35':
                # NEW: W₂ = 35% range(W₁) + 65% null(W₁) (matching real ratio)
                M = np.random.randn(dim_expand, dim)
                U_full, _, Vt = np.linalg.svd(M, full_matrices=True)
                W_expand = U_full[:, :dim] @ Vt
                
                U_range = U_full[:, :dim]   # [dim_expand, dim]
                U_null = U_full[:, dim:]     # [dim_expand, dim_expand - dim]
                
                # Range component
                R_range = np.random.randn(dim, dim)
                W_range = R_range @ U_range.T  # [dim, dim_expand]
                
                # Null component  
                R_null = np.random.randn(dim, dim_expand - dim)
                W_null = R_null @ U_null.T  # [dim, dim_expand]
                
                # Mix: 35% range + 65% null (by energy)
                W_range *= 0.35 / (np.linalg.norm(W_range) + 1e-8)
                W_null *= 0.65 / (np.linalg.norm(W_null) + 1e-8)
                W_compress = W_range + W_null
                
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'phi_svd_null_inject':
                # NEW: φ-structured W₁ + null-space W₂
                k = min(dim, dim_expand)
                singular_values = np.array([PHI ** (-i * 0.5 / k) for i in range(k)])
                singular_values *= np.sqrt(k)
                
                # Full SVD for null space
                U_full = np.linalg.qr(np.random.randn(dim_expand, dim_expand))[0]
                V_rand = np.linalg.qr(np.random.randn(dim, dim))[0]
                
                W_expand = (U_full[:, :dim] * singular_values) @ V_rand.T
                
                U_null = U_full[:, dim:]
                R_null = np.random.randn(dim, dim_expand - dim)
                W_compress = R_null @ U_null.T
                
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'phi_svd_mixed_65_35':
                # NEW: φ-structured W₁ + 65/35 mixed W₂
                k = min(dim, dim_expand)
                singular_values = np.array([PHI ** (-i * 0.5 / k) for i in range(k)])
                singular_values *= np.sqrt(k)
                
                U_full = np.linalg.qr(np.random.randn(dim_expand, dim_expand))[0]
                V_rand = np.linalg.qr(np.random.randn(dim, dim))[0]
                
                W_expand = (U_full[:, :dim] * singular_values) @ V_rand.T
                
                U_range = U_full[:, :dim]
                U_null = U_full[:, dim:]
                
                R_range = np.random.randn(dim, dim)
                W_range = R_range @ U_range.T
                R_null = np.random.randn(dim, dim_expand - dim)
                W_null = R_null @ U_null.T
                
                W_range *= 0.35 / (np.linalg.norm(W_range) + 1e-8)
                W_null *= 0.65 / (np.linalg.norm(W_null) + 1e-8)
                W_compress = W_range + W_null
                
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'anti_corr_bias':
                # NEW: W₁ + anti-correlated bias (stronger dirs = more negative bias)
                k = min(dim, dim_expand)
                singular_values = np.array([PHI ** (-i * 0.5 / k) for i in range(k)])
                singular_values *= np.sqrt(k)
                
                U_full = np.linalg.qr(np.random.randn(dim_expand, dim_expand))[0]
                V_rand = np.linalg.qr(np.random.randn(dim, dim))[0]
                
                W_expand = (U_full[:, :dim] * singular_values) @ V_rand.T
                
                # Anti-correlated bias: channels with higher W₁ row norm get more negative bias
                row_norms = np.linalg.norm(W_expand, axis=1)
                # Normalize and anti-correlate
                norm_ranks = np.argsort(np.argsort(-row_norms))  # rank by descending norm
                # Map ranks to bias: rank 0 (strongest) gets most negative
                bias_base = bias_means[stage_idx] + bias_stds[stage_idx] * np.random.randn(dim_expand)
                # Add anti-correlation: top channels get extra negative
                anticorr_strength = 0.5  # How much anti-correlation
                bias_adjustment = -anticorr_strength * (1 - 2 * norm_ranks / dim_expand)
                b_expand = bias_base + bias_adjustment
                
                # null-space W₂
                U_null = U_full[:, dim:]
                R_null = np.random.randn(dim, dim_expand - dim)
                W_compress = R_null @ U_null.T
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                
            elif method == 'random_independent':
                # Control: W₁ and W₂ are independently random (no structure)
                W_expand = np.random.randn(dim_expand, dim) * 0.02
                W_compress = np.random.randn(dim, dim_expand) * 0.02
                real_pw1_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv1.weight').numpy())
                real_pw2_norm = np.linalg.norm(v16._get_weight(f'{prefix}.pwconv2.weight').numpy())
                W_expand *= real_pw1_norm / np.linalg.norm(W_expand)
                W_compress *= real_pw2_norm / np.linalg.norm(W_compress)
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
            
            else:
                raise ValueError(f'Unknown method: {method}')
            
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_expand).float()
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_compress).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b_expand).float()
    
    return muts


# ================================================================
# COMPARISON
# ================================================================
print()
print('=' * 70)
print('COMPARISON: First-Principles Methods (12 images each)')
print('=' * 70)
print()

methods = [
    ('old_ortho_transpose', 'Ortho + W₂=W₁ᵀ (old baseline)'),
    ('phi_svd_transpose', 'φ-SVD + W₂=pinv(W₁) (old best)'),
    ('null_space_injector', 'Ortho + W₂ in null(W₁)'),
    ('mixed_injector_65_35', 'Ortho + W₂ mixed 65/35'),
    ('phi_svd_null_inject', 'φ-SVD + W₂ in null(W₁)'),
    ('phi_svd_mixed_65_35', 'φ-SVD + W₂ mixed 65/35'),
    ('anti_corr_bias', 'φ-SVD + null W₂ + anti-corr bias'),
    ('random_independent', 'Random independent (control)'),
]

print(f'{"Method":<45} {"Gap%":<12} {"Std":<8}')
print('-' * 65)

# Also get baseline
mean_base, std_base = evaluate()
print(f'  {"Full encoder (baseline)":<43} {mean_base:+5.1f}%    {std_base:4.1f}%')
print()

results = {}
for method_key, method_name in methods:
    muts = make_mutations(method_key)
    mean_g, std_g = evaluate(muts)
    results[method_key] = mean_g
    marker = ''
    if mean_g == max(results.values()):
        marker = ' ★'
    print(f'  {method_name:<43} {mean_g:+5.1f}%    {std_g:4.1f}%{marker}')

print()

# Find the best
best_key = max(results, key=results.get)
best_name = dict(methods)[best_key]
print(f'Best first-principles: {best_name} ({results[best_key]:+.1f}%)')
print(f'Gap from full encoder: {mean_base - results[best_key]:.1f}%')


# ================================================================
# VERIFY: Null-space properties of mutations
# ================================================================
print()
print('=' * 70)
print('VERIFICATION: Structural Properties of Each Method')
print('=' * 70)
print()

for method_key, method_name in methods[:6]:
    muts = make_mutations(method_key)
    pre = 'encoder.arch.stages.0.0'
    pw1 = muts[f'{pre}.pwconv1.weight'].numpy()
    pw2 = muts[f'{pre}.pwconv2.weight'].numpy()
    
    # Null-space fraction
    U1, _, _ = np.linalg.svd(pw1, full_matrices=True)
    U_range = U1[:, :96]
    U_null = U1[:, 96:]
    
    pw2_range = pw2 @ U_range @ U_range.T
    pw2_null = pw2 @ U_null @ U_null.T
    
    frac_null = np.linalg.norm(pw2_null)**2 / np.linalg.norm(pw2)**2
    cos_w2_w1t = np.dot(pw2.ravel(), pw1.T.ravel()) / (np.linalg.norm(pw2) * np.linalg.norm(pw1))
    
    print(f'  {method_name:<43}: null_frac={100*frac_null:.1f}%, cos(W₂,W₁ᵀ)={cos_w2_w1t:.4f}')

# Real encoder for comparison
pre = 'encoder.arch.stages.0.0'
pw1_real = v16._get_weight(f'{pre}.pwconv1.weight').numpy()
pw2_real = v16._get_weight(f'{pre}.pwconv2.weight').numpy()
U1_real, _, _ = np.linalg.svd(pw1_real, full_matrices=True)
pw2_range_r = pw2_real @ U1_real[:, :96] @ U1_real[:, :96].T
pw2_null_r = pw2_real @ U1_real[:, 96:] @ U1_real[:, 96:].T
frac_null_r = np.linalg.norm(pw2_null_r)**2 / np.linalg.norm(pw2_real)**2
cos_real = np.dot(pw2_real.ravel(), pw1_real.T.ravel()) / (np.linalg.norm(pw2_real) * np.linalg.norm(pw1_real))
print(f'  {"REAL ENCODER":<43}: null_frac={100*frac_null_r:.1f}%, cos(W₂,W₁ᵀ)={cos_real:.4f}')


print()
print('Done!')
