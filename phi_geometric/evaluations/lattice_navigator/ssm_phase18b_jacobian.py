"""
Phase 18B: The Composed Transform — Jacobian Analysis

The insight: we've been decomposing PW1 and PW2 independently via SVD.
But the actual computation is:
    y = W2 @ GELU(W1 @ z + b)

This is nonlinear. The Jacobian at input z is:
    J(z) = W2 @ diag(GELU'(W1@z + b)) @ W1

This is the EFFECTIVE linear transform at operating point z.
It's input-dependent through the GELU derivative.

Key questions:
  1. Does J(z) have consistent structure across different inputs?
  2. What's the effective rank of J vs the rank of W1 or W2 alone?
  3. Is the Jacobian more or less compressible than the individual matrices?
  4. Does the Jacobian reveal the "right sequence" — the composed geometry?
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def geometric_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def gelu_derivative(x):
    """GELU'(x) = Φ(x) + x·φ(x) where Φ is CDF, φ is PDF of N(0,1)."""
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    pdf = torch.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


# ================================================================
# Collect Jacobians at real operating points
# ================================================================
print("\nComputing Jacobians at real operating points...")

N_IMGS = 5
target_blocks = [(0, 1), (1, 1), (2, 0), (2, 4), (2, 8), (3, 0)]

# For each block, we'll compute J at multiple spatial positions
jacobians_data = defaultdict(list)

for img_idx in range(300, 300 + N_IMGS * 2):
    if len(jacobians_data.get(target_blocks[0], [])) >= N_IMGS:
        break
    im = cv2.imread(images[img_idx])
    if im is None:
        continue

    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean_t) / std_t

    with torch.no_grad():
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
                residual = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
                key = (stage_idx, block_idx)

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                W1 = v16._get_weight(f'{prefix}.pwconv1.weight')
                b1 = v16._get_weight(f'{prefix}.pwconv1.bias')
                W2 = v16._get_weight(f'{prefix}.pwconv2.weight')

                pre_gelu = F.linear(xb, W1, b1)  # [1, H, W, 4C]

                if key in target_blocks:
                    # Compute GELU derivative at each spatial position
                    gelu_d = gelu_derivative(pre_gelu)  # [1, H, W, 4C]

                    # Sample spatial positions (computing full Jacobian for every pixel is expensive)
                    H, W_sp = pre_gelu.shape[1], pre_gelu.shape[2]
                    n_sample = min(64, H * W_sp)
                    spatial_idx = np.random.choice(H * W_sp, n_sample, replace=False)

                    gelu_d_flat = gelu_d[0].reshape(-1, gelu_d.shape[-1])  # [H*W, 4C]

                    # Compute Jacobian: J = W2 @ diag(g') @ W1 for each sampled position
                    # J shape: [C, C] per position
                    W1_np = W1.numpy()
                    W2_np = W2.numpy()

                    # Mean Jacobian (averaged over spatial positions)
                    J_sum = np.zeros((dim, dim))
                    J_svs_list = []  # SVs of J at each position

                    for idx in spatial_idx:
                        g_prime = gelu_d_flat[idx].numpy()  # [4C]
                        # J = W2 @ diag(g') @ W1 = (W2 * g') @ W1
                        J = (W2_np * g_prime[np.newaxis, :]) @ W1_np  # [C, C]
                        J_sum += J
                        svs = np.linalg.svdvals(J)
                        J_svs_list.append(svs)

                    J_mean = J_sum / n_sample
                    J_mean_svs = np.linalg.svdvals(J_mean)

                    # Also compute W2 @ W1 directly (the Jacobian if GELU were identity)
                    J_linear = W2_np @ W1_np
                    J_linear_svs = np.linalg.svdvals(J_linear)

                    jacobians_data[key].append({
                        'J_mean': J_mean,
                        'J_mean_svs': J_mean_svs,
                        'J_pointwise_svs': J_svs_list,
                        'J_linear_svs': J_linear_svs,
                    })

                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu, W2,
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb


# ================================================================
# Analysis 1: Jacobian rank vs individual matrix rank
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 1: Effective Rank — Jacobian vs Individual Matrices')
print('=' * 70)
print()

print(f"{'Block':<8} {'C':<6} {'W1 rank@90':<12} {'W2 rank@90':<12} "
      f"{'J_linear r90':<14} {'J_mean r90':<12} {'J_point r90':<12}")
print("-" * 76)

for key in target_blocks:
    if not jacobians_data[key]:
        continue

    dim = dims[key[0]]
    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'

    # Individual matrix ranks
    W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
    W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
    s1 = np.linalg.svdvals(W1)
    s2 = np.linalg.svdvals(W2)
    cv1 = np.cumsum(s1**2) / np.sum(s1**2)
    cv2_ = np.cumsum(s2**2) / np.sum(s2**2)
    r90_w1 = np.searchsorted(cv1, 0.90) + 1
    r90_w2 = np.searchsorted(cv2_, 0.90) + 1

    # First image's data
    d = jacobians_data[key][0]

    # Linear Jacobian (no GELU)
    cvl = np.cumsum(d['J_linear_svs']**2) / np.sum(d['J_linear_svs']**2)
    r90_linear = np.searchsorted(cvl, 0.90) + 1

    # Mean Jacobian (averaged GELU derivative)
    cvm = np.cumsum(d['J_mean_svs']**2) / np.sum(d['J_mean_svs']**2)
    r90_mean = np.searchsorted(cvm, 0.90) + 1

    # Average pointwise Jacobian rank
    r90_points = []
    for svs in d['J_pointwise_svs']:
        cvp = np.cumsum(svs**2) / np.sum(svs**2)
        r90_points.append(np.searchsorted(cvp, 0.90) + 1)
    r90_point = np.mean(r90_points)

    print(f"  {key[0]}.{key[1]:<5} {dim:<6} {r90_w1:<12} {r90_w2:<12} "
          f"{r90_linear:<14} {r90_mean:<12} {r90_point:<12.1f}")


# ================================================================
# Analysis 2: Cross-input Jacobian consistency
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 2: Jacobian Consistency Across Inputs')
print('=' * 70)
print()

print("If the Jacobian is consistent across images, the composed transform")
print("has a STABLE structure independent of the specific input.")
print()

print(f"{'Block':<8} {'J_mean cross-img corr':<24} {'SV profile corr':<18}")
print("-" * 50)

for key in target_blocks:
    if len(jacobians_data[key]) < 2:
        continue

    # Compare J_mean across images
    J_mats = [d['J_mean'] for d in jacobians_data[key]]
    corrs = []
    sv_corrs = []
    for i in range(len(J_mats)):
        for j in range(i+1, len(J_mats)):
            c = np.corrcoef(J_mats[i].flatten(), J_mats[j].flatten())[0, 1]
            corrs.append(c)
            # SV profile correlation
            sv_c = np.corrcoef(
                jacobians_data[key][i]['J_mean_svs'],
                jacobians_data[key][j]['J_mean_svs']
            )[0, 1]
            sv_corrs.append(sv_c)

    print(f"  {key[0]}.{key[1]:<5} {np.mean(corrs):<24.4f} {np.mean(sv_corrs):<18.4f}")


# ================================================================
# Analysis 3: The GELU derivative concentrates the Jacobian
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 3: GELU Derivative as Focusing Lens')
print('=' * 70)
print()

print("Does GELU focus the Jacobian (reduce rank) or scatter it?")
print("Compare: J_linear = W2 @ W1 vs J_gelu = W2 @ diag(g') @ W1")
print()

print(f"{'Block':<8} {'J_linear rank90':<16} {'J_gelu rank90':<16} "
      f"{'J_linear top1%':<16} {'J_gelu top1%':<16} {'GELU focuses?':<14}")
print("-" * 76)

for key in target_blocks:
    if not jacobians_data[key]:
        continue

    d = jacobians_data[key][0]
    dim = dims[key[0]]

    # Linear: how much energy in top 1% of SVs
    n_top = max(1, dim // 100)
    e_linear_top = np.sum(d['J_linear_svs'][:n_top]**2) / np.sum(d['J_linear_svs']**2)
    e_gelu_top = np.sum(d['J_mean_svs'][:n_top]**2) / np.sum(d['J_mean_svs']**2)

    cvl = np.cumsum(d['J_linear_svs']**2) / np.sum(d['J_linear_svs']**2)
    cvm = np.cumsum(d['J_mean_svs']**2) / np.sum(d['J_mean_svs']**2)
    r90_linear = np.searchsorted(cvl, 0.90) + 1
    r90_gelu = np.searchsorted(cvm, 0.90) + 1

    focuses = "YES" if r90_gelu < r90_linear else "NO"

    print(f"  {key[0]}.{key[1]:<5} {r90_linear:<16} {r90_gelu:<16} "
          f"{e_linear_top:<16.4f} {e_gelu_top:<16.4f} {focuses:<14}")


# ================================================================
# Analysis 4: Jacobian PCA alignment
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 4: Jacobian Directions vs Individual Matrix Directions')
print('=' * 70)
print()

print("The Jacobian's SVD gives us the COMPOSED directions.")
print("How much do they overlap with PW1's V or PW2's U?")
print()

print(f"{'Block':<8} {'J_U vs W2_U top5':<20} {'J_V vs W1_V top5':<20} "
      f"{'J_U vs J_V (symmetry)':<22}")
print("-" * 70)

for key in target_blocks:
    if not jacobians_data[key]:
        continue

    dim = dims[key[0]]
    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'
    d = jacobians_data[key][0]

    # SVD of mean Jacobian
    U_J, S_J, Vt_J = np.linalg.svd(d['J_mean'], full_matrices=False)

    # SVD of W1 and W2
    W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
    W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
    U_W2, _, _ = np.linalg.svd(W2, full_matrices=False)
    _, _, Vt_W1 = np.linalg.svd(W1, full_matrices=False)

    k = min(5, dim)

    # Jacobian U vs W2 U
    M1 = U_J[:, :k].T @ U_W2[:, :k]
    align_u = np.mean(np.linalg.svdvals(M1))

    # Jacobian V vs W1 V
    M2 = Vt_J[:k, :] @ Vt_W1[:k, :].T
    align_v = np.mean(np.linalg.svdvals(M2))

    # Symmetry: is J close to symmetric (J ≈ J.T)?
    # If so, U_J ≈ V_J and the transform is self-adjoint
    M3 = U_J[:, :k].T @ Vt_J[:k, :].T
    symmetry = np.mean(np.linalg.svdvals(M3))

    print(f"  {key[0]}.{key[1]:<5} {align_u:<20.4f} {align_v:<20.4f} {symmetry:<22.4f}")


# ================================================================
# Analysis 5: The residual perspective
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 5: Full Block Transform (including residual)')
print('=' * 70)
print()

print("The full block is: x_out = x + gamma * J(x) @ x")
print("The effective transform is: x_out = (I + gamma * J) @ x")
print("This is a PERTURBATION of identity. How large is the perturbation?")
print()

print(f"{'Block':<8} {'gamma mean':<12} {'||gamma*J||_F':<14} "
      f"{'||I||_F':<10} {'Perturbation %':<16} {'Effective type':<16}")
print("-" * 76)

for key in target_blocks:
    if not jacobians_data[key]:
        continue

    dim = dims[key[0]]
    prefix = f'encoder.arch.stages.{key[0]}.{key[1]}'
    d = jacobians_data[key][0]

    gamma = v16._get_weight(f'{prefix}.gamma').numpy()
    gamma_mean = np.mean(np.abs(gamma))

    # J_mean is [C, C]. gamma is [C]. The full perturbation is diag(gamma) @ J
    gamma_J = gamma[:, np.newaxis] * d['J_mean']  # [C, C]

    norm_gJ = np.linalg.norm(gamma_J, 'fro')
    norm_I = np.sqrt(dim)  # ||I||_F = sqrt(C)

    perturbation = norm_gJ / norm_I * 100

    # Spectral analysis of I + gamma*J
    full_transform = np.eye(dim) + gamma_J
    svs_full = np.linalg.svdvals(full_transform)

    # How close to identity? Look at condition number and SV spread
    sv_mean = np.mean(svs_full)
    sv_std = np.std(svs_full)

    eff_type = "near-identity" if perturbation < 30 else "significant"

    print(f"  {key[0]}.{key[1]:<5} {gamma_mean:<12.4f} {norm_gJ:<14.4f} "
          f"{norm_I:<10.2f} {perturbation:<16.1f} {eff_type:<16}")

    # Print SV distribution of full transform
    print(f"          SVs of (I + gamma*J): mean={sv_mean:.3f} std={sv_std:.3f} "
          f"min={svs_full[-1]:.3f} max={svs_full[0]:.3f}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 18B SUMMARY')
print('=' * 70)
print()
print("The Jacobian J(z) = W2 @ diag(GELU'(z)) @ W1 captures the COMPOSED")
print("transform at each operating point. Key findings:")
print()
print("1. Does GELU focus the Jacobian?")
print("2. Is the Jacobian consistent across inputs?")
print("3. Are the Jacobian directions different from W1/W2 individually?")
print("4. Is the full block (I + gamma*J) a near-identity perturbation?")
