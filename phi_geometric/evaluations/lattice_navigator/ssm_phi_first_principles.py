"""
SSM φ-Gate First Principles: Can φ-gate improve first-principles construction?

Previous findings:
  - First-principles spectrometers all negative when replacing ALL blocks
  - SVD-guided best at -6.4%, ortho_biased -8.6%
  - BUT those all used GELU
  
Now we know:
  - φ-SiLU matches GELU for the REAL encoder
  - The gate curvature matters (phase transition at α≈1.5)
  - The bias controls selectivity

Questions:
1. Does φ-gate help first-principles spectrometers?
2. What if we match BOTH the gate AND the bias distribution?
3. Can we build a working spectrometer from pure geometry?
4. What's the minimum learned content needed?
"""
import numpy as np
import cv2
import sys
import os
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
print('SSM φ-GATE FIRST PRINCIPLES')
print('=' * 70)

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ================================================================
# INFRASTRUCTURE
# ================================================================

def run_encoder_custom(v16, img_tensor, weight_mutations=None, gate_fn=None):
    """Run encoder with optional weight mutations AND custom gate."""
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


# Build color basis
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


def evaluate(weight_mutations=None, gate_fn=None, n_images=15):
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


# Gate functions
def gate_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def gate_phi_silu(x):
    return x * torch.sigmoid(PHI * x)

def gate_optimal_silu(x):
    return x * torch.sigmoid(1.74 * x)


# ================================================================
# PART 1: BASELINES — Real weights with different gates
# ================================================================
print()
print('=' * 70)
print('PART 1: BASELINES — Real Weights + Different Gates')
print('=' * 70)
print()

print('Real encoder weights with different gate functions:')
for name, gate in [('GELU', gate_gelu), ('φ-SiLU', gate_phi_silu), ('Optimal-SiLU (1.74)', gate_optimal_silu)]:
    mean_g, std_g = evaluate(gate_fn=gate)
    print(f'  {name:<25}: {mean_g:+5.1f}% ± {std_g:4.1f}%')


# ================================================================
# PART 2: FIRST-PRINCIPLES + GELU vs FIRST-PRINCIPLES + φ-GATE
# ================================================================
print()
print('=' * 70)
print('PART 2: FIRST-PRINCIPLES SPECTROMETERS — GELU vs φ-Gate')
print('=' * 70)
print()

def make_spectrometer_mutations(method, seed=42):
    """Create weight mutations for ALL spectrometers using a given method."""
    np.random.seed(seed)
    muts = {}
    
    for stage_idx in range(4):
        dim = dims[stage_idx]
        dim_expand = dim * 4
        
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            
            if method == 'random_orthogonal':
                M = np.random.randn(dim_expand, dim)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                W_expand = U @ Vt
                W_compress = W_expand.T
                b_expand = np.zeros(dim_expand)
                
            elif method == 'ortho_biased':
                M = np.random.randn(dim_expand, dim)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                W_expand = U @ Vt
                W_compress = W_expand.T
                # Match encoder bias: mostly negative
                b_expand = np.random.randn(dim_expand) * 0.9 - 1.0
                
            elif method == 'ortho_stage_biased':
                # Match the per-stage bias distribution from the real encoder
                M = np.random.randn(dim_expand, dim)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                W_expand = U @ Vt
                W_compress = W_expand.T
                # Stage-specific bias matching
                bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
                bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'svd_guided_stage_biased':
                # Use the REAL encoder's singular value distribution + stage bias
                real_pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
                _, S_real, _ = np.linalg.svd(real_pw1, full_matrices=False)
                # Random orthogonal bases
                k = min(dim, dim_expand)
                U = np.linalg.qr(np.random.randn(dim_expand, k))[0]
                V = np.linalg.qr(np.random.randn(dim, k))[0]
                W_expand = U * S_real @ V.T
                W_compress = V * (1.0 / (S_real + 1e-6)) @ U.T
                # Stage-specific bias
                bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
                bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
                
            elif method == 'phi_structured_biased':
                # φ-decay singular values + orthogonal bases + stage bias
                k = min(dim, dim_expand)
                # Use φ^(-i/k) decay — slower than φ^(-i/2), matches observed Zipf
                singular_values = np.array([PHI ** (-i * 0.5 / k) for i in range(k)])
                singular_values *= np.sqrt(k)  # Scale to match real encoder norms
                U = np.linalg.qr(np.random.randn(dim_expand, k))[0]
                V = np.linalg.qr(np.random.randn(dim, k))[0]
                W_expand = U * singular_values @ V.T
                W_compress = V * (1.0 / (singular_values + 1e-6)) @ U.T
                bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
                bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
                b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
            
            else:
                raise ValueError(f'Unknown method: {method}')
            
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_expand).float()
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_compress).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b_expand).float()
    
    return muts


methods = [
    'random_orthogonal',
    'ortho_biased',
    'ortho_stage_biased',
    'svd_guided_stage_biased',
    'phi_structured_biased',
]

print(f'{"Method":<30} {"GELU":<15} {"φ-SiLU":<15} {"Opt-SiLU(1.74)":<15}')
print('-' * 75)

for method in methods:
    muts = make_spectrometer_mutations(method)
    results = []
    for gate_name, gate_fn in [('GELU', gate_gelu), ('φ-SiLU', gate_phi_silu), ('Opt-SiLU', gate_optimal_silu)]:
        mean_g, std_g = evaluate(muts, gate_fn, n_images=10)
        results.append(f'{mean_g:+5.1f}±{std_g:4.1f}')
    print(f'  {method:<28} {results[0]:<15} {results[1]:<15} {results[2]:<15}')


# ================================================================
# PART 3: HYBRID — Real weights for critical stages, first-principles for rest
# ================================================================
print()
print('=' * 70)
print('PART 3: HYBRID — Real weights for critical stages + first-principles')
print('=' * 70)
print()

# From our findings:
# - Stage 2 is most sensitive to compression
# - Stage 0 is extremely resilient
# - Stage 3 is most compressible

# Strategy: keep Stages 1-2 real, replace Stage 0 and 3 with first-principles
print('Hybrid strategies (10 images each):')
print()

strategies = {
    'All real (baseline)': {'replace': []},
    'Replace S0 only': {'replace': [0]},
    'Replace S3 only': {'replace': [3]},
    'Replace S0+S3': {'replace': [0, 3]},
    'Replace S0+S2+S3 (keep S1 only)': {'replace': [0, 2, 3]},
    'Replace all': {'replace': [0, 1, 2, 3]},
}

for strat_name, strat in strategies.items():
    muts = {}
    np.random.seed(42)
    for stage_idx in strat['replace']:
        dim = dims[stage_idx]
        dim_expand = dim * 4
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            # Use best method: ortho + stage-matched bias
            M = np.random.randn(dim_expand, dim)
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            W_expand = U @ Vt
            W_compress = W_expand.T
            bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
            bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
            b_expand = np.random.randn(dim_expand) * bias_stds[stage_idx] + bias_means[stage_idx]
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_expand).float()
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_compress).float()
            muts[f'{prefix}.pwconv1.bias'] = torch.from_numpy(b_expand).float()
    
    mean_g, std_g = evaluate(muts, gate_gelu, n_images=10)
    print(f'  {strat_name:<40}: {mean_g:+5.1f}% ± {std_g:4.1f}%')


# ================================================================
# PART 4: LR90 + φ-GATE — Best of both worlds?
# ================================================================
print()
print('=' * 70)
print('PART 4: LR90 + φ-GATE — Does φ-gate improve truncated encoder?')
print('=' * 70)
print()

# Build LR90 mutations
lr90_muts = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            cumvar = np.cumsum(S**2) / (S**2).sum()
            k = np.searchsorted(cumvar, 0.90) + 1
            approx = (U[:, :k] * S[:k]) @ Vt[:k]
            lr90_muts[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()

print('LR90 truncated encoder with different gates:')
for name, gate in [('GELU', gate_gelu), ('φ-SiLU', gate_phi_silu), ('Opt-SiLU (1.74)', gate_optimal_silu)]:
    mean_g, std_g = evaluate(lr90_muts, gate)
    print(f'  LR90 + {name:<25}: {mean_g:+5.1f}% ± {std_g:4.1f}%')

# Also test more aggressive truncation with φ-gate
for var_target in [0.70, 0.80, 0.90]:
    muts = {}
    for stage_idx in range(4):
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            for pw_name in ['pwconv1', 'pwconv2']:
                w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
                U, S, Vt = np.linalg.svd(w, full_matrices=False)
                cumvar = np.cumsum(S**2) / (S**2).sum()
                k = np.searchsorted(cumvar, var_target) + 1
                approx = (U[:, :k] * S[:k]) @ Vt[:k]
                muts[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()
    
    mean_gelu, _ = evaluate(muts, gate_gelu, n_images=10)
    mean_phi, _ = evaluate(muts, gate_phi_silu, n_images=10)
    mean_opt, _ = evaluate(muts, gate_optimal_silu, n_images=10)
    print(f'  LR{int(var_target*100)} + GELU: {mean_gelu:+5.1f}%, '
          f'φ-SiLU: {mean_phi:+5.1f}%, Opt: {mean_opt:+5.1f}%')


# ================================================================
# PART 5: THE MINIMUM VIABLE SPECTROMETER
# ================================================================
print()
print('=' * 70)
print('PART 5: MINIMUM VIABLE SPECTROMETER')
print('=' * 70)
print()

# What's the absolute minimum we need?
# From our findings:
# - Orthogonal expand + transpose compress works reasonably
# - Stage-matched bias is important
# - φ-gate ≈ GELU
# - LR90 of real weights matches full encoder
#
# Try: keep ONLY the top-k SVD modes of the real expand matrix,
# replace everything else with orthogonal fill

print('Real top-k SVD modes + orthogonal fill for rest (GELU, 10 images):')
for n_keep in [1, 3, 5, 10, 20, 50]:
    muts = {}
    np.random.seed(42)
    for stage_idx in range(4):
        dim = dims[stage_idx]
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            
            # Get real expand matrix
            pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
            U, S, Vt = np.linalg.svd(pw1, full_matrices=False)
            
            k = min(n_keep, len(S))
            # Keep top-k real modes
            W_core = (U[:, :k] * S[:k]) @ Vt[:k]
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_core).float()
            
            # Same for compress
            U2, S2, Vt2 = np.linalg.svd(pw2, full_matrices=False)
            W_core2 = (U2[:, :k] * S2[:k]) @ Vt2[:k]
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_core2).float()
    
    mean_g, std_g = evaluate(muts, gate_gelu, n_images=10)
    # Count params
    total_params = 0
    total_orig = 0
    for stage_idx in range(4):
        dim = dims[stage_idx]
        dim_exp = dim * 4
        k = min(n_keep, dim)
        # Each truncated matrix: rows×k + k + k×cols params
        params_per_block = (dim_exp * k + k + k * dim) * 2  # pw1 + pw2
        total_params += params_per_block * depths[stage_idx]
        total_orig += (dim_exp * dim + dim) * 2 * depths[stage_idx]
    
    ratio = total_params / total_orig
    print(f'  top-{n_keep:<3}: gap={mean_g:+5.1f}% ± {std_g:4.1f}%, params={ratio*100:.1f}%')


# ================================================================
# GRAND SUMMARY
# ================================================================
print()
print('=' * 70)
print('GRAND SUMMARY')
print('=' * 70)
print()

# Run final comprehensive comparison
print('Final comparison (15 images):')
configs = [
    ('Full encoder + GELU', {}, gate_gelu),
    ('Full encoder + φ-SiLU', {}, gate_phi_silu),
    ('LR90 + GELU', lr90_muts, gate_gelu),
    ('LR90 + φ-SiLU', lr90_muts, gate_phi_silu),
    ('Ortho+bias (all) + GELU', make_spectrometer_mutations('ortho_stage_biased'), gate_gelu),
    ('Ortho+bias (all) + φ-SiLU', make_spectrometer_mutations('ortho_stage_biased'), gate_phi_silu),
]

for name, muts, gate in configs:
    mean_g, std_g = evaluate(muts, gate)
    print(f'  {name:<35}: {mean_g:+5.1f}% ± {std_g:4.1f}%')

print()
print('Done!')
