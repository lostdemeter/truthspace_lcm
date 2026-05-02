"""
Phase 18: Sequential Bootstrap — Deriving PW Directions from Feature Statistics

The insight: we can't decompose PW independently because:
  1. The system is nonlinear (GELU, LayerNorm, residual)
  2. Each block depends on ALL previous blocks' outputs
  3. SVD imposes a linear basis on a nonlinear chain
  4. "Our own manipulations change meaning as we work"

NEW APPROACH: Sequential bootstrap.
  - At each block, observe the actual input features from calibration images
  - Derive the PW directions from the STATISTICS of those features
  - Use the derived PW for that block, then propagate to the next
  - Build the whole chain sequentially, one block at a time

Tests:
  A) PCA alignment: do PCA eigenvectors of input features match PW1's V?
  B) Covariance-derived PW: use feature PCA as PW1 directions, measure RMSE
  C) Sequential vs parallel: derive PW block-by-block (sequential) vs all-at-once
  D) How many calibration images are needed?
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


# ================================================================
# Step 1: Collect input features at every block (full V16 forward)
# ================================================================
print("\nCollecting per-block input features from calibration images...")

N_CAL = 10
block_inputs = defaultdict(list)  # (stage, block) -> list of [H*W, C] feature arrays

for img_idx in range(200, 200 + N_CAL * 2):
    if len(block_inputs.get((0, 0), [])) >= N_CAL:
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

                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                # CAPTURE: this is the input to PW1
                # xb is [1, H, W, C] — flatten to [H*W, C]
                key = (stage_idx, block_idx)
                block_inputs[key].append(xb[0].reshape(-1, dim).numpy())

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

print(f"  Collected features from {N_CAL} images at all 18 blocks")


# ================================================================
# TEST A: PCA Alignment with Learned PW1 Directions
# ================================================================
print()
print('=' * 70)
print('TEST A: PCA of Input Features vs Learned PW1 Directions')
print('=' * 70)
print()

# For each block:
#   - Compute PCA of input features → eigenvectors of covariance
#   - Compare with PW1's right singular vectors (V from SVD of W1)
#   - Measure alignment via subspace angle / canonical correlations

print(f"{'Block':<8} {'C':<6} {'PCA top-1 align':<16} {'PCA top-5 align':<16} "
      f"{'PCA top-C align':<16} {'Feature rank@90%':<16}")
print("-" * 78)

pca_directions = {}  # Store for later use

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

        # Stack all feature vectors: [N_total, C]
        features = np.vstack(block_inputs[key])

        # Feature covariance and PCA
        feat_centered = features - features.mean(axis=0)
        cov = feat_centered.T @ feat_centered / len(feat_centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]  # [C, C] columns are eigenvectors

        pca_directions[key] = eigvecs

        # Feature effective rank
        cumvar_feat = np.cumsum(eigvals) / np.sum(eigvals)
        feat_rank90 = np.searchsorted(cumvar_feat, 0.90) + 1

        # Learned PW1 right singular vectors
        W1 = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
        _, _, Vt = np.linalg.svd(W1, full_matrices=False)
        V_learned = Vt.T  # [C, rank] — columns are right singular vectors

        # Alignment: for top-K PCA vectors, what fraction of their variance
        # is captured by the learned PW1 V subspace?
        # Use canonical correlation between PCA subspace and V subspace

        def subspace_alignment(A, B, k):
            """Alignment between top-k columns of A and top-k columns of B."""
            Ak = A[:, :k]
            Bk = B[:, :k]
            # Canonical correlations = singular values of Ak.T @ Bk
            M = Ak.T @ Bk
            svals = np.linalg.svdvals(M)
            # Mean canonical correlation
            return np.mean(svals)

        align1 = subspace_alignment(eigvecs, V_learned, 1)
        align5 = subspace_alignment(eigvecs, V_learned, min(5, dim))
        alignC = subspace_alignment(eigvecs, V_learned, dim)

        print(f"  {stage_idx}.{block_idx:<5} {dim:<6} {align1:<16.4f} "
              f"{align5:<16.4f} {alignC:<16.4f} {feat_rank90:<16}")


# ================================================================
# TEST B: Use Feature PCA as PW1 Directions — RMSE Test
# ================================================================
print()
print('=' * 70)
print('TEST B: Replace PW1 Directions with Feature PCA')
print('=' * 70)
print()

# Prepare test data
N_TEST = 20
test_data = []
for idx in range(300, 360):
    if len(test_data) >= N_TEST:
        break
    im = cv2.imread(images[idx])
    if im is None:
        continue
    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append((t, gt_ab))


def forward_with_pca_pw1(img_tensor, mode='pca_v_only'):
    """
    Replace PW1's right singular vectors (V) with feature PCA directions.

    Modes:
      'pca_v_only': Replace V in W1 = U @ S @ Vt with PCA eigenvectors
      'pca_full': Replace entire W1 with PCA-based projection
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                 v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, (96,),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
    x = x.permute(0, 3, 1, 2)

    features = []
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

            W1_orig = v16._get_weight(f'{prefix}.pwconv1.weight').numpy()
            b1 = v16._get_weight(f'{prefix}.pwconv1.bias')

            if mode == 'pca_v_only':
                # Replace V in W1 = U @ S @ Vt with PCA directions
                U1, S1, Vt1 = np.linalg.svd(W1_orig, full_matrices=False)
                pca_V = pca_directions[key]  # [C, C]
                # Use PCA as new V: W1_new = U @ S @ PCA_V.T
                # PCA_V columns are [C, C], we need [rank, C] for Vt
                rank = len(S1)
                Vt_new = pca_V[:, :rank].T  # [rank, C]
                W1_new = (U1 * S1) @ Vt_new
                W1_mod = torch.from_numpy(W1_new.astype(np.float32))

            elif mode == 'pca_full':
                # PW1 = project onto top-4C PCA directions with learned scaling
                pca_V = pca_directions[key]  # [C, C]
                U1, S1, Vt1 = np.linalg.svd(W1_orig, full_matrices=False)
                # Expand: project input onto C PCA dirs, then scale to 4C
                # W1_new has same U and S but uses PCA for V
                Vt_new = pca_V[:, :len(S1)].T
                W1_new = (U1 * S1) @ Vt_new
                W1_mod = torch.from_numpy(W1_new.astype(np.float32))

            pre_gelu = F.linear(xb, W1_mod, b1)
            post_gelu = geometric_gelu(pre_gelu)

            xb = F.linear(post_gelu,
                         v16._get_weight(f'{prefix}.pwconv2.weight'),
                         v16._get_weight(f'{prefix}.pwconv2.bias'))
            xb = xb.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * xb

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))

    out0 = v16._geometric_unet_block(features[3], features[2], 0)
    out1 = v16._geometric_unet_block(out0, features[1], 1)
    out2 = v16._geometric_unet_block(out1, features[0], 2)
    out3 = v16._geometric_last_shuf(out2)

    cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
    color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out3)

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


# Simpler: just test both modes
def eval_mode(mode_name, forward_fn):
    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = forward_fn(t)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    mean_rmse = np.mean(rmses)
    print(f"  {mode_name:<40} RMSE = {mean_rmse:.3f}")
    return mean_rmse

def normal_forward(img_tensor):
    """Standard forward with original weights."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                 v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, (96,),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
    x = x.permute(0, 3, 1, 2)

    features = []
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
            xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                         v16._get_weight(f'{prefix}.dwconv.bias'),
                         padding=3, groups=dim)
            xb = xb.permute(0, 2, 3, 1)
            xb = F.layer_norm(xb, (dim,),
                             v16._get_weight(f'{prefix}.norm.weight'),
                             v16._get_weight(f'{prefix}.norm.bias'))
            pre_gelu = F.linear(xb,
                                v16._get_weight(f'{prefix}.pwconv1.weight'),
                                v16._get_weight(f'{prefix}.pwconv1.bias'))
            post_gelu = geometric_gelu(pre_gelu)
            xb = F.linear(post_gelu,
                         v16._get_weight(f'{prefix}.pwconv2.weight'),
                         v16._get_weight(f'{prefix}.pwconv2.bias'))
            xb = xb.permute(0, 3, 1, 2)
            gamma = v16._get_weight(f'{prefix}.gamma')
            x = residual + gamma.view(1, -1, 1, 1) * xb

        x_normed = x.permute(0, 2, 3, 1)
        x_normed = F.layer_norm(x_normed, (dim,),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.weight'),
                                v16._get_weight(f'encoder.arch.norm{stage_idx}.bias'))
        features.append(x_normed.permute(0, 3, 1, 2))

    out0 = v16._geometric_unet_block(features[3], features[2], 0)
    out1 = v16._geometric_unet_block(out0, features[1], 1)
    out2 = v16._geometric_unet_block(out1, features[0], 2)
    out3 = v16._geometric_last_shuf(out2)

    cm_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/v17_color_matrix.npz')
    color_matrix = torch.from_numpy(cm_data['color_matrix']).float()
    color_out = torch.einsum('bqc,bchw->bqhw', color_matrix, out3)

    coarse_input = torch.cat([color_out, (img_tensor - mean) / std], dim=1)
    return F.conv2d(coarse_input,
                    v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


baseline = eval_mode("Baseline (original PW1)", normal_forward)

pca_v = eval_mode("PCA V directions (keep U, S)",
                   lambda t: forward_with_pca_pw1(t, 'pca_v_only'))

delta = (pca_v - baseline) / baseline * 100
print(f"\n  PCA V replacement: {delta:+.2f}% vs baseline")


# ================================================================
# TEST C: What if we also replace PW2's U with output PCA?
# ================================================================
print()
print('=' * 70)
print('TEST C: Replace BOTH PW1-V and PW2-U with Feature PCA')
print('=' * 70)
print()

# For PW2, we need the OUTPUT feature statistics
# But the output of PW2 goes through scale + residual → becomes INPUT to next block
# So the PW2 output directions should align with the RESIDUAL space
# which is the SAME as the input space (since residual = x + scale * PW2_out)

# Let's also collect PW2 output features (post-GELU → PW2 output, before residual)
block_pw2_outputs = defaultdict(list)

for img_idx in range(200, 200 + N_CAL * 2):
    if len(block_pw2_outputs.get((0, 0), [])) >= N_CAL:
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

                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))
                post_gelu = geometric_gelu(pre_gelu)
                pw2_out = F.linear(post_gelu,
                                  v16._get_weight(f'{prefix}.pwconv2.weight'),
                                  v16._get_weight(f'{prefix}.pwconv2.bias'))

                # Capture PW2 output [1, H, W, C] → [H*W, C]
                block_pw2_outputs[key].append(pw2_out[0].reshape(-1, dim).numpy())

                xb = pw2_out.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# Compute PCA of PW2 outputs
pw2_pca_directions = {}
for key in block_pw2_outputs:
    outputs = np.vstack(block_pw2_outputs[key])
    out_centered = outputs - outputs.mean(axis=0)
    cov = out_centered.T @ out_centered / len(out_centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    pw2_pca_directions[key] = eigvecs[:, idx]

# Now check: do PW2's LEFT singular vectors align with PW2 output PCA?
print(f"{'Block':<8} {'PW2 U vs output PCA top-5':<30} {'PW2 U vs INPUT PCA top-5':<30}")
print("-" * 68)

for stage_idx in range(4):
    dim = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

        W2 = v16._get_weight(f'{prefix}.pwconv2.weight').numpy()
        U2, _, _ = np.linalg.svd(W2, full_matrices=False)
        # U2: [C, rank]

        k = min(5, dim)

        # Align with OUTPUT PCA
        out_pca = pw2_pca_directions[key][:, :k]
        M_out = U2[:, :k].T @ out_pca
        align_out = np.mean(np.linalg.svdvals(M_out))

        # Align with INPUT PCA (same space — C dims)
        in_pca = pca_directions[key][:, :k]
        M_in = U2[:, :k].T @ in_pca
        align_in = np.mean(np.linalg.svdvals(M_in))

        print(f"  {stage_idx}.{block_idx:<5} {align_out:<30.4f} {align_in:<30.4f}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 18 SUMMARY')
print('=' * 70)
print()
print("PCA alignment tells us whether the learned PW directions")
print("are determined by the feature statistics (data geometry)")
print("or encode something beyond what the data shape reveals.")
