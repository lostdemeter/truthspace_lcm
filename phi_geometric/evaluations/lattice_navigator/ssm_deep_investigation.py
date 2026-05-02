"""
SSM Deep Investigation: Reverse Engineering the Artifact

PHASE 1: Multi-image validation
  - Confirm Stage 3 disposability across 20+ images
  - Confirm low-rank improvement across 20+ images
  - Per-stage importance map

PHASE 2: Stage 0 Deep Dive
  - Extract Stage 0's expand/compress matrices
  - SVD of expand matrix: what "questions" does it ask?
  - Which expanded dimensions survive GELU? (the vocabulary)
  - Cross-image stability of the vocabulary
  - What input features trigger each vocabulary entry?

PHASE 3: First-Principles Spectrometer
  - Random projection spectrometer (no learning at all)
  - φ-structured spectrometer (golden ratio singular values)
  - Hadamard spectrometer (orthogonal, deterministic)
  - DCT spectrometer (frequency domain)
  - Identity-expand spectrometer (simple replication + noise)
  - Compare all against real spectrometer

PHASE 4: Compressed 15% Encoder
  - Build: rank-5 spatial + rank-90% S0-S2 + random S3
  - Test end-to-end on held-out images
"""
import numpy as np
import cv2
import sys
import os
import glob
import torch
import torch.nn.functional as F
from scipy.special import erf
from scipy.linalg import hadamard
from scipy.fft import dct

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

print('=== SSM DEEP INVESTIGATION ===\n')
v16 = V16GeometricColorizer()

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/ssm_investigation'
os.makedirs(out_dir, exist_ok=True)


# ================================================================
# SHARED INFRASTRUCTURE
# ================================================================

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


# Build color basis
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

from numpy.linalg import lstsq
proj_1 = (all_enc_arr - enc_mean) @ color_dir_1
proj_2 = (all_enc_arr - enc_mean) @ color_dir_2
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt_arr[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt_arr[:, 1], rcond=None)

def predict_color(enc_features):
    flat = (enc_features.reshape(256, -1).T - enc_mean)
    fields = np.column_stack([flat @ color_dir_1, flat @ color_dir_2, np.ones(SZ*SZ)])
    ab = np.stack([
        np.clip(fields @ W_a, -50, 50).reshape(SZ, SZ),
        np.clip(fields @ W_b, -50, 50).reshape(SZ, SZ)
    ], axis=2)
    return ab

def evaluate_image(img_path, mutations):
    im = cv2.imread(img_path)
    if im is None: return None, None, None
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    err_z = np.sqrt(np.mean(ab_gt**2))
    if err_z < 2: return None, None, None
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_features_mutated(v16, img_tensor, mutations)
    ab_pred = predict_color(enc)
    err = np.sqrt(np.mean((ab_pred - ab_gt)**2))
    return err, err_z, (1 - err/err_z) * 100


# Prepare mutation sets
print('Preparing mutation sets...\n')

# Random spectrometers (fixed seed for consistency)
np.random.seed(42)
mutations_random_s3 = {}
for block_idx in range(depths[3]):
    prefix = f'encoder.arch.stages.3.{block_idx}'
    pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
    pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
    mutations_random_s3[f'{prefix}.pwconv1.weight'] = torch.from_numpy(
        np.random.randn(*pw1.shape) * np.std(pw1)).float()
    mutations_random_s3[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
        np.random.randn(*pw2.shape) * np.std(pw2)).float()

# Per-stage randomization
stage_random_mutations = {}
for target_stage in range(4):
    muts = {}
    for block_idx in range(depths[target_stage]):
        prefix = f'encoder.arch.stages.{target_stage}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
        muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(
            np.random.randn(*pw1.shape) * np.std(pw1)).float()
        muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
            np.random.randn(*pw2.shape) * np.std(pw2)).float()
    stage_random_mutations[target_stage] = muts

# Low-rank 90% spectrometer
mutations_lr90 = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            cumvar = np.cumsum(S**2) / (S**2).sum()
            k = np.searchsorted(cumvar, 0.90) + 1
            approx = (U[:, :k] * S[:k]) @ Vt[:k]
            mutations_lr90[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()

# Low-rank 95%
mutations_lr95 = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            cumvar = np.cumsum(S**2) / (S**2).sum()
            k = np.searchsorted(cumvar, 0.95) + 1
            approx = (U[:, :k] * S[:k]) @ Vt[:k]
            mutations_lr95[f'{prefix}.{pw_name}.weight'] = torch.from_numpy(approx).float()

# Compressed encoder: rank-5 spatial + LR90 S0-S2 + random S3
mutations_compressed = {}
# LR90 for stages 0-2
for stage_idx in range(3):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            key = f'{prefix}.{pw_name}.weight'
            mutations_compressed[key] = mutations_lr90[key]
# Random for stage 3
for key, val in mutations_random_s3.items():
    mutations_compressed[key] = val


# ================================================================
# PHASE 1: MULTI-IMAGE VALIDATION
# ================================================================
print('=' * 70)
print('PHASE 1: MULTI-IMAGE VALIDATION (20 held-out images)')
print('=' * 70)
print()

test_indices = list(range(80, 110))
results = {
    'full': [], 'random_s3': [], 'lr90': [], 'lr95': [],
    'compressed': [], 'zero': [],
    'rand_s0': [], 'rand_s1': [], 'rand_s2': [], 'rand_s3_only': []
}

mutations_zero = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight')
        mutations_zero[f'{prefix}.pwconv1.weight'] = torch.zeros_like(pw1)
        mutations_zero[f'{prefix}.pwconv2.weight'] = torch.zeros_like(pw2)

n_tested = 0
for idx in test_indices:
    if n_tested >= 20: break

    err_full, err_z, gap_full = evaluate_image(all_imgs[idx], {})
    if err_full is None: continue

    _, _, gap_rs3 = evaluate_image(all_imgs[idx], mutations_random_s3)
    _, _, gap_lr90 = evaluate_image(all_imgs[idx], mutations_lr90)
    _, _, gap_lr95 = evaluate_image(all_imgs[idx], mutations_lr95)
    _, _, gap_comp = evaluate_image(all_imgs[idx], mutations_compressed)
    _, _, gap_zero = evaluate_image(all_imgs[idx], mutations_zero)

    # Per-stage
    _, _, gap_s0 = evaluate_image(all_imgs[idx], stage_random_mutations[0])
    _, _, gap_s1 = evaluate_image(all_imgs[idx], stage_random_mutations[1])
    _, _, gap_s2 = evaluate_image(all_imgs[idx], stage_random_mutations[2])
    _, _, gap_s3o = evaluate_image(all_imgs[idx], stage_random_mutations[3])

    results['full'].append(gap_full)
    results['random_s3'].append(gap_rs3)
    results['lr90'].append(gap_lr90)
    results['lr95'].append(gap_lr95)
    results['compressed'].append(gap_comp)
    results['zero'].append(gap_zero)
    results['rand_s0'].append(gap_s0)
    results['rand_s1'].append(gap_s1)
    results['rand_s2'].append(gap_s2)
    results['rand_s3_only'].append(gap_s3o)

    n_tested += 1
    print(f'  Image {n_tested:2d}: full={gap_full:+5.1f}% lr90={gap_lr90:+5.1f}% '
          f'lr95={gap_lr95:+5.1f}% comp={gap_comp:+5.1f}% rand_s3={gap_rs3:+5.1f}%')

print(f'\n  Summary over {n_tested} images:')
print(f'  {"Variant":<25} {"Mean Gap%":>10} {"Std":>8} {"vs Full":>10}')
print(f'  {"-"*25} {"-"*10} {"-"*8} {"-"*10}')
for name, label in [('full', 'Full encoder'), ('lr90', 'Low-rank 90%'),
                     ('lr95', 'Low-rank 95%'), ('random_s3', 'Random Stage 3'),
                     ('compressed', 'Compressed (15%)'),
                     ('zero', 'Zero spectrometer')]:
    vals = results[name]
    m, s = np.mean(vals), np.std(vals)
    diff = m - np.mean(results['full'])
    print(f'  {label:<25} {m:>+10.1f} {s:>8.1f} {diff:>+10.1f}')

print(f'\n  Per-stage importance (gap when that stage is randomized):')
for si, name in enumerate(['rand_s0', 'rand_s1', 'rand_s2', 'rand_s3_only']):
    vals = results[name]
    m = np.mean(vals)
    diff = m - np.mean(results['full'])
    print(f'    Stage {si} ({dims[si]:3d}ch): gap={m:+.1f}%, Δ={diff:+.1f}%')


# ================================================================
# PHASE 2: STAGE 0 DEEP DIVE — What's in the vocabulary?
# ================================================================
print()
print('=' * 70)
print('PHASE 2: STAGE 0 DEEP DIVE — The Critical Vocabulary')
print('=' * 70)
print()

# Extract Stage 0's spectrometer matrices
for block_idx in range(depths[0]):
    prefix = f'encoder.arch.stages.0.{block_idx}'
    pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()  # [384, 96]
    pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()  # [96, 384]
    b1 = v16._get_weight(f'{prefix}.pwconv1.bias').numpy()     # [384]

    print(f'Block {block_idx}: pw1={pw1.shape}, pw2={pw2.shape}')

    # SVD of expand matrix
    U1, S1, Vt1 = np.linalg.svd(pw1, full_matrices=False)
    cumvar1 = np.cumsum(S1**2) / (S1**2).sum()
    r50 = np.searchsorted(cumvar1, 0.50) + 1
    r90 = np.searchsorted(cumvar1, 0.90) + 1
    r99 = np.searchsorted(cumvar1, 0.99) + 1
    print(f'  Expand SVD: rank50={r50}, rank90={r90}, rank99={r99} (of {len(S1)})')
    print(f'  S[0]/S[1] = {S1[0]/S1[1]:.4f} (φ={PHI:.4f}, Δ={abs(S1[0]/S1[1]-PHI)/PHI*100:.1f}%)')

    # Bias analysis — how many dimensions start negative (initially suppressed)?
    n_neg = np.sum(b1 < 0)
    n_pos = np.sum(b1 >= 0)
    print(f'  Expand bias: {n_neg} negative ({n_neg/len(b1)*100:.0f}%), '
          f'{n_pos} positive ({n_pos/len(b1)*100:.0f}%)')
    print(f'  Bias mean={b1.mean():.4f}, std={b1.std():.4f}, '
          f'min={b1.min():.4f}, max={b1.max():.4f}')

    # Which expanded dimensions have the largest compress weights?
    # (Most important for the output)
    compress_importance = np.linalg.norm(pw2, axis=0)  # [384] — norm of each column
    expand_importance = np.linalg.norm(pw1, axis=1)    # [384] — norm of each row

    # Combined importance: expand_norm * compress_norm * (bias > threshold probability)
    # Approximate: for standard normal input, P(W·x + b > 0) depends on b/||W||
    effective_threshold = -b1 / (expand_importance + 1e-8)
    p_active = 0.5 * (1 + np.vectorize(lambda v: float(erf(v / np.sqrt(2.0))))(
        -effective_threshold))  # P(activation > 0)

    combined = expand_importance * compress_importance * p_active
    top_k = 20
    top_dims = np.argsort(combined)[-top_k:][::-1]
    print(f'  Top {top_k} vocabulary entries (by combined importance):')
    for rank, d in enumerate(top_dims[:10]):
        print(f'    #{rank}: dim={d:3d}, expand_norm={expand_importance[d]:.3f}, '
              f'compress_norm={compress_importance[d]:.3f}, '
              f'P(active)={p_active[d]:.3f}, bias={b1[d]:.3f}')

    # Net transform analysis
    net = pw2 @ pw1  # [96, 96]
    eigvals = np.linalg.eigvals(net)
    n_complex = np.sum(np.abs(eigvals.imag) > 0.01)
    n_neg_eig = np.sum(eigvals.real < -0.01)
    print(f'  Net transform: {n_complex}/{len(eigvals)} complex, '
          f'{n_neg_eig}/{len(eigvals)} negative')

    # Compare expand matrix to known structures
    # Check: is it close to a Hadamard-like orthogonal structure?
    UU = U1[:, :96]  # First 96 left singular vectors
    gram = UU.T @ UU
    off_diag = gram - np.eye(96)
    ortho_err = np.sqrt(np.mean(off_diag**2))
    print(f'  Expand left-SVs orthogonality error: {ortho_err:.4f} (0=perfect)')
    print()


# ================================================================
# PHASE 2b: Cross-image activation stability of Stage 0
# ================================================================
print('Stage 0 activation stability across images:')
activation_patterns_per_image = []

for idx in [80, 85, 90, 95, 100]:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # Run through stem + stage 0 to get pre-GELU activations
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean_t) / std_t

    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))

        # Through block 0
        prefix = 'encoder.arch.stages.0.0'
        x_dw = x.permute(0, 3, 1, 2)
        residual = x_dw
        x_dw = F.conv2d(x_dw, v16._get_weight(f'{prefix}.dwconv.weight'),
                        v16._get_weight(f'{prefix}.dwconv.bias'), padding=3, groups=96)
        x_ln = x_dw.permute(0, 2, 3, 1)
        x_ln = F.layer_norm(x_ln, (96,),
                            v16._get_weight(f'{prefix}.norm.weight'),
                            v16._get_weight(f'{prefix}.norm.bias'))

        # Get pre-GELU activations
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight')
        pw1_b = v16._get_weight(f'{prefix}.pwconv1.bias')
        h = F.linear(x_ln, pw1, pw1_b)  # [1, H, W, 384]

        h_np = h.squeeze(0).numpy()  # [H, W, 384]
        active = (h_np > 0).astype(float)

        # Per-channel activation fraction across all pixels
        chan_frac = active.reshape(-1, 384).mean(axis=0)  # [384]
        activation_patterns_per_image.append(chan_frac)

        spatial_active_frac = active.reshape(-1, 384).mean(axis=1)  # [H*W]
        print(f'  Image {idx}: mean_active={chan_frac.mean()*100:.1f}%, '
              f'always_on={np.sum(chan_frac > 0.95)}, '
              f'always_off={np.sum(chan_frac < 0.05)}, '
              f'per_pixel_active={spatial_active_frac.mean()*100:.1f}%')

# Cross-image stability
if len(activation_patterns_per_image) >= 2:
    patterns = np.array(activation_patterns_per_image)
    cross_corrs = []
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            r = np.corrcoef(patterns[i], patterns[j])[0, 1]
            cross_corrs.append(r)
    print(f'  Cross-image channel activation correlation: {np.mean(cross_corrs):.4f} ± {np.std(cross_corrs):.4f}')
    print(f'  → {"STABLE" if np.mean(cross_corrs) > 0.8 else "VARIABLE"} across images')


# ================================================================
# PHASE 3: FIRST-PRINCIPLES SPECTROMETER
# ================================================================
print()
print('=' * 70)
print('PHASE 3: FIRST-PRINCIPLES SPECTROMETER — Can we build one without learning?')
print('=' * 70)
print()

def make_spectrometer(dim_in, dim_expand, method):
    """Create expand/compress matrices using various strategies."""
    if method == 'random_gaussian':
        W_expand = np.random.randn(dim_expand, dim_in) * np.sqrt(2.0 / dim_in)
        W_compress = np.random.randn(dim_in, dim_expand) * np.sqrt(2.0 / dim_expand)
        b_expand = np.zeros(dim_expand)
        return W_expand, W_compress, b_expand

    elif method == 'random_orthogonal':
        # Random orthogonal expand, pseudoinverse compress
        M = np.random.randn(dim_expand, dim_in)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        W_expand = U @ Vt  # Orthogonal rows
        W_compress = W_expand.T  # Pseudoinverse
        b_expand = np.zeros(dim_expand)
        return W_expand, W_compress, b_expand

    elif method == 'hadamard':
        # Pad to nearest power of 2, use Hadamard rows
        n = 1
        while n < dim_expand:
            n *= 2
        H = hadamard(n) / np.sqrt(n)
        # Select rows for expand
        W_expand = H[:dim_expand, :dim_in]
        W_compress = H[:dim_in, :dim_expand]
        b_expand = np.zeros(dim_expand)
        return W_expand, W_compress, b_expand

    elif method == 'dct':
        # DCT basis — frequency-domain decomposition
        W_expand = np.zeros((dim_expand, dim_in))
        for i in range(dim_expand):
            for j in range(dim_in):
                W_expand[i, j] = np.cos(np.pi * (2*j+1) * i / (2*dim_in))
        W_expand *= np.sqrt(2.0 / dim_in)
        W_compress = W_expand.T  # [dim_in, dim_expand] — pseudoinverse
        b_expand = np.zeros(dim_expand)
        return W_expand, W_compress, b_expand

    elif method == 'phi_structured':
        # Golden ratio structured singular values
        k = min(dim_in, dim_expand)
        singular_values = np.array([PHI ** (-i/2) for i in range(k)])
        # Random orthogonal bases
        U = np.linalg.qr(np.random.randn(dim_expand, k))[0]
        V = np.linalg.qr(np.random.randn(dim_in, k))[0]
        W_expand = U * singular_values @ V.T
        W_compress = V * (1.0 / singular_values) @ U.T
        b_expand = np.zeros(dim_expand)
        return W_expand, W_compress, b_expand

    elif method == 'phi_sparse':
        # φ-structured with negative bias to enforce sparsity
        k = min(dim_in, dim_expand)
        singular_values = np.array([PHI ** (-i/2) for i in range(k)])
        U = np.linalg.qr(np.random.randn(dim_expand, k))[0]
        V = np.linalg.qr(np.random.randn(dim_in, k))[0]
        W_expand = U * singular_values @ V.T
        W_compress = V * (1.0 / singular_values) @ U.T
        # Negative bias — enforce ~30% activation like the real encoder
        b_expand = np.ones(dim_expand) * (-0.5)
        return W_expand, W_compress, b_expand

    elif method == 'encoder_real':
        # Extract the actual encoder's Stage 0 Block 0 spectrometer
        pw1 = v16._get_weight('encoder.arch.stages.0.0.pwconv1.weight').numpy()
        pw2 = v16._get_weight('encoder.arch.stages.0.0.pwconv2.weight').numpy()
        b1 = v16._get_weight('encoder.arch.stages.0.0.pwconv1.bias').numpy()
        return pw1, pw2, b1


# Test each spectrometer by replacing Stage 0 Block 0
print('Replacing Stage 0 Block 0 spectrometer with first-principles versions:')
print(f'(Keeping all other blocks unchanged)')
print()

np.random.seed(123)  # Fixed seed for reproducibility
methods = ['encoder_real', 'random_gaussian', 'random_orthogonal',
           'hadamard', 'dct', 'phi_structured', 'phi_sparse']

for method in methods:
    W_exp, W_comp, b_exp = make_spectrometer(96, 384, method)

    muts = {}
    prefix = 'encoder.arch.stages.0.0'
    muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_exp).float()
    muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_comp).float()

    # Also override bias for non-real methods
    # (we'd need to modify the encoder function for this, so for now
    #  we just modify the weight matrices and use original biases)

    # Evaluate on 5 images
    gaps = []
    for test_idx in [80, 85, 90, 95, 100]:
        err, err_z, gap = evaluate_image(all_imgs[test_idx], muts)
        if err is not None:
            gaps.append(gap)

    if gaps:
        print(f'  {method:<22}: gap={np.mean(gaps):+5.1f}% ± {np.std(gaps):4.1f}%')
    else:
        print(f'  {method:<22}: FAILED')


# ================================================================
# PHASE 3b: Replace ALL Stage 0 blocks
# ================================================================
print()
print('Replacing ALL Stage 0 blocks (3 blocks) with first-principles:')
print()

for method in methods:
    muts = {}
    for block_idx in range(depths[0]):
        prefix = f'encoder.arch.stages.0.{block_idx}'
        W_exp, W_comp, b_exp = make_spectrometer(96, 384, method)
        muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_exp).float()
        muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_comp).float()

    gaps = []
    for test_idx in [80, 85, 90, 95, 100]:
        err, err_z, gap = evaluate_image(all_imgs[test_idx], muts)
        if err is not None:
            gaps.append(gap)

    if gaps:
        print(f'  {method:<22}: gap={np.mean(gaps):+5.1f}% ± {np.std(gaps):4.1f}%')


# ================================================================
# PHASE 3c: Replace ALL stages with first-principles spectrometers
# ================================================================
print()
print('Replacing ALL spectrometers (all stages, all blocks) with first-principles:')
print()

for method in methods:
    if method == 'encoder_real': continue  # Only works for stage 0 dim

    muts = {}
    for stage_idx in range(4):
        dim = dims[stage_idx]
        dim_expand = dim * 4
        for block_idx in range(depths[stage_idx]):
            prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
            W_exp, W_comp, b_exp = make_spectrometer(dim, dim_expand, method)
            muts[f'{prefix}.pwconv1.weight'] = torch.from_numpy(W_exp).float()
            muts[f'{prefix}.pwconv2.weight'] = torch.from_numpy(W_comp).float()

    gaps = []
    for test_idx in [80, 85, 90, 95, 100]:
        err, err_z, gap = evaluate_image(all_imgs[test_idx], muts)
        if err is not None:
            gaps.append(gap)

    if gaps:
        print(f'  {method:<22}: gap={np.mean(gaps):+5.1f}% ± {np.std(gaps):4.1f}%')


# ================================================================
# PHASE 4: COMPRESSED 15% ENCODER
# ================================================================
print()
print('=' * 70)
print('PHASE 4: COMPRESSED 15% ENCODER')
print('=' * 70)
print('Config: LR90 spectrometer (stages 0-2) + random spectrometer (stage 3)')
print()

gaps_compressed = []
gaps_full = []
gaps_zero = []

test_indices_p4 = list(range(80, 130))
n_tested = 0
for idx in test_indices_p4:
    if n_tested >= 30: break
    err_full, err_z, gap_full = evaluate_image(all_imgs[idx], {})
    if err_full is None: continue
    _, _, gap_comp = evaluate_image(all_imgs[idx], mutations_compressed)
    _, _, gap_zero = evaluate_image(all_imgs[idx], mutations_zero)

    gaps_compressed.append(gap_comp)
    gaps_full.append(gap_full)
    gaps_zero.append(gap_zero)
    n_tested += 1

print(f'  Over {n_tested} images:')
print(f'  {"Model":<30} {"Mean Gap%":>10} {"Std":>8}')
print(f'  {"-"*30} {"-"*10} {"-"*8}')
print(f'  {"Full encoder (100% params)":<30} {np.mean(gaps_full):>+10.1f} {np.std(gaps_full):>8.1f}')
print(f'  {"Compressed (~15% params)":<30} {np.mean(gaps_compressed):>+10.1f} {np.std(gaps_compressed):>8.1f}')
print(f'  {"Zero spectrometer":<30} {np.mean(gaps_zero):>+10.1f} {np.std(gaps_zero):>8.1f}')

# Count actual parameters
total_orig = 0
total_compressed = 0
for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw_name in ['pwconv1', 'pwconv2']:
            w = v16._get_weight(f'{prefix}.{pw_name}.weight').numpy()
            orig_params = w.size
            total_orig += orig_params

            if stage_idx < 3:
                # LR90
                U, S, Vt = np.linalg.svd(w, full_matrices=False)
                cumvar = np.cumsum(S**2) / (S**2).sum()
                k = np.searchsorted(cumvar, 0.90) + 1
                # Stored as U[:,:k], S[:k], Vt[:k,:] = k*(rows+cols+1)
                compressed_params = k * (w.shape[0] + w.shape[1] + 1)
                total_compressed += compressed_params
            else:
                # Random — no stored params (generated from seed)
                total_compressed += 0  # Just need the seed + shape

print(f'\n  Parameter count:')
print(f'    Original spectrometer: {total_orig:,} params')
print(f'    Compressed (LR90 S0-2 + random S3): {total_compressed:,} params')
print(f'    Compression ratio: {total_compressed/total_orig*100:.1f}%')


# ================================================================
# PHASE 5: THE HYBRID — Real S0 + Random everything else
# ================================================================
print()
print('=' * 70)
print('PHASE 5: HYBRID ENCODER — Real Stage 0 + Random Stages 1-3')
print('=' * 70)
print()

mutations_hybrid = {}
for stage_idx in range(1, 4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        pw1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        pw2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
        mutations_hybrid[f'{prefix}.pwconv1.weight'] = torch.from_numpy(
            np.random.randn(*pw1.shape) * np.std(pw1)).float()
        mutations_hybrid[f'{prefix}.pwconv2.weight'] = torch.from_numpy(
            np.random.randn(*pw2.shape) * np.std(pw2)).float()

gaps_hybrid = []
n_tested = 0
for idx in test_indices_p4:
    if n_tested >= 20: break
    err, err_z, gap = evaluate_image(all_imgs[idx], mutations_hybrid)
    if err is None: continue
    gaps_hybrid.append(gap)
    n_tested += 1

print(f'  Real S0 + Random S1-S3: gap={np.mean(gaps_hybrid):+.1f}% ± {np.std(gaps_hybrid):.1f}%')
print(f'  Full encoder:           gap={np.mean(gaps_full[:20]):+.1f}% ± {np.std(gaps_full[:20]):.1f}%')
print(f'  → Stage 0 alone carries {np.mean(gaps_hybrid)/np.mean(gaps_full[:20])*100:.0f}% of spectrometer value')


# ================================================================
# GRAND SUMMARY
# ================================================================
print()
print('=' * 70)
print('GRAND SUMMARY')
print('=' * 70)
print(f"""
Multi-image validation ({n_tested} images):
  Full encoder:        {np.mean(results['full']):+.1f}% gap
  Low-rank 90%:        {np.mean(results['lr90']):+.1f}% gap
  Low-rank 95%:        {np.mean(results['lr95']):+.1f}% gap
  Compressed (15%):    {np.mean(gaps_compressed):+.1f}% gap
  Real S0 + Random:    {np.mean(gaps_hybrid):+.1f}% gap
  Zero spectrometer:   {np.mean(results['zero']):+.1f}% gap

Stage importance (gap when randomized):
  Stage 0: {np.mean(results['rand_s0']):+.1f}%  ← CRITICAL
  Stage 1: {np.mean(results['rand_s1']):+.1f}%
  Stage 2: {np.mean(results['rand_s2']):+.1f}%
  Stage 3: {np.mean(results['rand_s3_only']):+.1f}%  ← DISPOSABLE

Compression: {total_compressed:,} / {total_orig:,} = {total_compressed/total_orig*100:.1f}% of spectrometer params
""")
print('Done!')
