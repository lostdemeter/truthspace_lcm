"""
TRUE Reverse Navigation: Examine the encoder's actual operations.

The encoder is:
  Stem (3→96, stride 4) → Stage0 (96d, 3 blocks) → Stage1 (192d, 3 blocks)
  → Stage2 (384d, 9 blocks) → Stage3 (768d, 3 blocks) → UNet → 256d features

Each ConvNeXt block:
  Depthwise 7×7 conv → LayerNorm → Pointwise expand → GELU → Pointwise compress

Questions:
1. What do the stem kernels look like? (first thing that touches pixels)
2. Are the depthwise 7×7 kernels geometric? (Gabor, edge, DoG?)
3. What is the SVD/φ structure of each layer's weights?
4. Layer by layer, how much of the 2D color field does each stage create?
"""
import numpy as np
import cv2
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

print('=== TRUE REVERSE: Kernel Analysis ===\n')

v16 = V16GeometricColorizer()

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/kernel_analysis'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Stem kernels — the FIRST operation on pixels
# ============================================================
print('=== PART 1: Stem Kernels (3→96, stride 4) ===\n')

stem_w = v16._get_weight('encoder.arch.downsample_layers.0.0.weight').numpy()
stem_b = v16._get_weight('encoder.arch.downsample_layers.0.0.bias').numpy()
print(f'Stem weight shape: {stem_w.shape}')  # [96, 3, 4, 4]
print(f'Stem bias shape: {stem_b.shape}')

# The stem takes 3 grayscale channels (identical for gray input) and produces 96 features
# Each of the 96 output channels has a 3×4×4 kernel
# For grayscale input, the 3 input channels are identical, so effective kernel = sum over channel dim

# Visualize the 96 kernels (sum over input channels for grayscale interpretation)
gray_kernels = stem_w.sum(axis=1)  # [96, 4, 4] - effective gray filter
print(f'Effective grayscale kernels: {gray_kernels.shape}')

# SVD of the kernel bank
kernels_flat = gray_kernels.reshape(96, -1)  # [96, 16]
U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
cumvar = np.cumsum(S**2) / (S**2).sum()

print(f'\nStem kernel SVD:')
print(f'  Effective rank for 90% variance: {np.searchsorted(cumvar, 0.9)+1}')
print(f'  Effective rank for 95% variance: {np.searchsorted(cumvar, 0.95)+1}')
print(f'  Effective rank for 99% variance: {np.searchsorted(cumvar, 0.99)+1}')

print(f'\n  Singular value ratios:')
for i in range(min(8, len(S)-1)):
    ratio = S[i] / S[i+1] if S[i+1] > 1e-10 else float('inf')
    phi_err = abs(ratio - PHI) / PHI * 100
    marker = ' ← φ!' if phi_err < 15 else ''
    print(f'    S[{i}]/S[{i+1}] = {ratio:.4f} ({phi_err:.1f}% from φ){marker}')

# Classify the 96 kernels — what geometric operation does each one perform?
# Compare each kernel to known filter types
print(f'\nKernel classification:')

# Generate reference filters
def make_reference_filters():
    """Create canonical 4×4 geometric filters."""
    refs = {}
    
    # Mean (DC)
    refs['mean'] = np.ones((4, 4)) / 16.0
    
    # Horizontal edge
    refs['h_edge'] = np.array([[-1,-1,-1,-1], [-1,-1,-1,-1], [1,1,1,1], [1,1,1,1]]) / 8.0
    
    # Vertical edge
    refs['v_edge'] = np.array([[-1,-1,1,1], [-1,-1,1,1], [-1,-1,1,1], [-1,-1,1,1]]) / 8.0
    
    # Diagonal edges
    refs['d1_edge'] = np.array([[-1,-1,-1,1], [-1,-1,1,1], [-1,1,1,1], [1,1,1,-1]]) / 8.0
    refs['d2_edge'] = np.array([[1,-1,-1,-1], [1,1,-1,-1], [1,1,1,-1], [-1,1,1,1]]) / 8.0
    
    # Center-surround (blob)
    refs['blob'] = np.array([[-1,-1,-1,-1], [-1,2,2,-1], [-1,2,2,-1], [-1,-1,-1,-1]]) / 8.0
    
    # Corner detectors
    refs['corner_tl'] = np.array([[2,2,-1,-1], [2,2,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]]) / 8.0
    refs['corner_br'] = np.array([[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,2,2], [-1,-1,2,2]]) / 8.0
    
    # Normalize
    for k, v in refs.items():
        norm = np.sqrt(np.sum(v**2))
        if norm > 1e-8:
            refs[k] = v / norm
    
    return refs

ref_filters = make_reference_filters()

classifications = {}
for i in range(96):
    k = gray_kernels[i]
    k_norm = k / (np.sqrt(np.sum(k**2)) + 1e-8)
    
    best_corr = 0
    best_type = 'unknown'
    for name, ref in ref_filters.items():
        corr = abs(np.sum(k_norm * ref))
        if corr > best_corr:
            best_corr = corr
            best_type = name
    
    if best_type not in classifications:
        classifications[best_type] = []
    classifications[best_type].append((i, best_corr))

for ftype in sorted(classifications.keys()):
    items = classifications[ftype]
    mean_corr = np.mean([c for _, c in items])
    print(f'  {ftype:<12}: {len(items):2d} kernels (mean corr={mean_corr:.3f})')

# Visualize the top kernels
n_show = min(96, 96)
grid_w = 12
grid_h = (n_show + grid_w - 1) // grid_w
cell_size = 32

vis = np.ones((grid_h * (cell_size + 2), grid_w * (cell_size + 2)), dtype=np.uint8) * 128
for i in range(n_show):
    row, col = i // grid_w, i % grid_w
    k = gray_kernels[i]
    # Normalize to [0, 255]
    vmin, vmax = k.min(), k.max()
    if vmax - vmin < 1e-8: vmax = vmin + 1
    k_vis = ((k - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    k_vis = cv2.resize(k_vis, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
    y, x = row * (cell_size + 2), col * (cell_size + 2)
    vis[y:y+cell_size, x:x+cell_size] = k_vis

cv2.imwrite(os.path.join(out_dir, 'stem_kernels.jpg'), vis)
print(f'\nStem kernel grid saved.')


# ============================================================
# PART 2: Depthwise 7×7 convolutions in each stage
# ============================================================
print('\n=== PART 2: Depthwise 7×7 Convolutions ===\n')

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

for stage_idx in range(4):
    dim = dims[stage_idx]
    print(f'Stage {stage_idx} ({dim} channels, {depths[stage_idx]} blocks):')
    
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
        # Shape: [dim, 1, 7, 7] (depthwise)
        
        # Each channel has its own 7×7 kernel
        kernels = dw_w.squeeze(1)  # [dim, 7, 7]
        kernels_flat = kernels.reshape(dim, -1)  # [dim, 49]
        
        U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        rank90 = np.searchsorted(cumvar, 0.9) + 1
        rank95 = np.searchsorted(cumvar, 0.95) + 1
        
        # φ structure
        ratios = []
        for i in range(min(5, len(S)-1)):
            if S[i+1] > 1e-10:
                ratios.append(S[i] / S[i+1])
        
        # Compare to Gabor bank
        # The top SVD basis vectors ARE the dominant filter types
        top_basis = Vt[:5].reshape(5, 7, 7)
        
        # Mean kernel properties
        mean_kernel = kernels.mean(axis=0)
        kernel_energy = np.sqrt(np.mean(kernels**2, axis=(1,2)))
        
        # Layer scale
        gamma = v16._get_weight(f'{prefix}.gamma').numpy()
        gamma_range = f'[{gamma.min():.4f}, {gamma.max():.4f}]'
        
        r_str = ', '.join(f'{r:.3f}' for r in ratios[:3])
        print(f'  Block {block_idx}: rank90={rank90:2d}/{dim} rank95={rank95:2d}/{dim} | '
              f'S ratios=[{r_str}] | γ={gamma_range}')
    
    # Visualize first block's top SVD basis vectors
    prefix = f'encoder.arch.stages.{stage_idx}.0'
    dw_w = v16._get_weight(f'{prefix}.dwconv.weight').numpy()
    kernels = dw_w.squeeze(1)
    kernels_flat = kernels.reshape(dim, -1)
    U, S, Vt = np.linalg.svd(kernels_flat, full_matrices=False)
    
    # Show top 8 basis filters
    n_basis = min(8, len(S))
    cell = 56
    basis_vis = np.ones((cell + 2, n_basis * (cell + 2)), dtype=np.uint8) * 128
    for i in range(n_basis):
        basis = Vt[i].reshape(7, 7)
        vmin, vmax = basis.min(), basis.max()
        if vmax - vmin < 1e-8: vmax = vmin + 1
        bv = ((basis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        bv = cv2.resize(bv, (cell, cell), interpolation=cv2.INTER_NEAREST)
        x = i * (cell + 2)
        basis_vis[0:cell, x:x+cell] = bv
    
    cv2.imwrite(os.path.join(out_dir, f'stage{stage_idx}_basis.jpg'), basis_vis)
    print()


# ============================================================
# PART 3: Layer-by-layer contribution to color field
# How much of the 2D color field does each stage create?
# ============================================================
print('=== PART 3: Layer-by-Layer Color Field Contribution ===\n')

import glob

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
SZ = 256

# First, get the color basis from the full encoder
train_indices = list(range(50, 65))
all_enc_pix = []
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

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

    enc = out3.squeeze(0).detach().numpy()
    flat = enc.reshape(256, -1).T
    sample = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_enc_pix.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt_ab.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc_pix = np.vstack(all_enc_pix)
all_gt_ab = np.vstack(all_gt_ab)
enc_mean = all_enc_pix.mean(axis=0)

C = (all_enc_pix - enc_mean).T @ all_gt_ab / len(all_enc_pix)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dir_1 = U_color[:, 0]
color_dir_2 = U_color[:, 1]

# Now trace through the encoder stage by stage and see what each stage contributes
# Use a test image
test_im = cv2.imread(all_imgs[55])
test_r = cv2.resize(test_im, (SZ, SZ))
test_gray = cv2.cvtColor(test_r, cv2.COLOR_BGR2GRAY)
test_lab = cv2.cvtColor(test_r, cv2.COLOR_BGR2Lab)
test_ab = test_lab[:,:,1:].astype(float) - 128.0
test_gbgr = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)
test_tensor = torch.from_numpy(test_gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x = (test_tensor - mean_t) / std_t

# Run stage by stage
with torch.no_grad():
    # Stem
    x_stem = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                      v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
    x_stem = x_stem.permute(0, 2, 3, 1)
    x_stem = F.layer_norm(x_stem, (96,),
                          v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                          v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
    x_stem = x_stem.permute(0, 3, 1, 2)

    print(f'After stem: shape={list(x_stem.shape)}')
    stem_flat = x_stem.squeeze(0).detach().numpy().reshape(96, -1).T  # [64*64, 96]

    # SVD of stem features
    stem_mean = stem_flat.mean(axis=0)
    U_s, S_s, Vt_s = np.linalg.svd(stem_flat - stem_mean, full_matrices=False)
    cumvar_s = np.cumsum(S_s**2) / (S_s**2).sum()
    rank90_stem = np.searchsorted(cumvar_s, 0.9) + 1
    print(f'  Stem features: rank90={rank90_stem}/96')
    for i in range(min(5, len(S_s)-1)):
        ratio = S_s[i] / S_s[i+1]
        phi_err = abs(ratio - PHI) / PHI * 100
        marker = ' ← φ!' if phi_err < 15 else ''
        print(f'  S[{i}]/S[{i+1}] = {ratio:.4f} ({phi_err:.1f}%){marker}')

    # Run through stages and track features at each level
    x_running = x_stem.clone()
    stage_features = []

    for stage_idx in range(4):
        dim = dims[stage_idx]
        if stage_idx > 0:
            prefix = f'encoder.arch.downsample_layers.{stage_idx}'
            x_running = x_running.permute(0, 2, 3, 1)
            x_running = F.layer_norm(x_running, (dims[stage_idx-1],),
                                     v16._get_weight(f'{prefix}.0.weight'),
                                     v16._get_weight(f'{prefix}.0.bias'))
            x_running = x_running.permute(0, 3, 1, 2)
            x_running = F.conv2d(x_running, v16._get_weight(f'{prefix}.1.weight'),
                                 v16._get_weight(f'{prefix}.1.bias'), stride=2)

        for block_idx in range(depths[stage_idx]):
            x_running = v16._geometric_convnext_block(
                x_running, f'encoder.arch.stages.{stage_idx}.{block_idx}', dim)

        # Norm
        x_normed = x_running.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        x_normed = x_normed.permute(0, 3, 1, 2)
        stage_features.append(x_normed)

        feat_np = x_normed.squeeze(0).detach().numpy()
        h_f, w_f = feat_np.shape[1], feat_np.shape[2]
        flat = feat_np.reshape(dim, -1).T

        # SVD
        fm = flat.mean(axis=0)
        Us, Ss, Vts = np.linalg.svd(flat - fm, full_matrices=False)
        cumvars = np.cumsum(Ss**2) / (Ss**2).sum()
        r90 = np.searchsorted(cumvars, 0.9) + 1

        # φ ratios
        ratios = []
        for i in range(min(5, len(Ss)-1)):
            if Ss[i+1] > 1e-10:
                ratios.append(Ss[i] / Ss[i+1])
        r_str = ', '.join(f'{r:.3f}' for r in ratios[:3])

        print(f'\n  Stage {stage_idx} ({dim}d, {h_f}×{w_f}): rank90={r90}/{dim}, S ratios=[{r_str}]')


print(f'\nOutput saved to: {out_dir}/')
print('Done!')
