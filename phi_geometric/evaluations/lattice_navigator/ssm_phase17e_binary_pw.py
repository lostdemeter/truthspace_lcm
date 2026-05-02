"""
Phase 17E: Binary PW1 — Testing the Hyperplane Hypothesis

If the gate SIGN PATTERN is the information (Phase 17D), then:
  - PW1 rows are hyperplane normals
  - We only need the sign of PW1 @ x to be correct
  - The magnitude doesn't matter (GELU leakage adjusts)

Tests:
  A) Binarize PW1 (+1/-1 weights) — preserve only hyperplane orientation
  B) Ternary PW1 (+1/0/-1) — add sparsity
  C) Low-bit PW1 (4-bit, 2-bit) — progressive quantization
  D) Random hyperplanes with CORRECT bias — does geometry alone work?
  E) Shared hyperplanes across blocks — reuse the same normals

Each test: measure RMSE and gate pattern accuracy.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob

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


def forward_with_modified_pw1(img_tensor, pw1_modifier):
    """Run encoder with modified PW1 weights, V17-style color decoding."""
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

            W1_orig = v16._get_weight(f'{prefix}.pwconv1.weight')
            b1 = v16._get_weight(f'{prefix}.pwconv1.bias')

            # Apply modifier to PW1
            W1_mod = pw1_modifier(W1_orig, stage_idx, block_idx)

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


def eval_modifier(pw1_modifier, label):
    """Evaluate a PW1 modifier on test set."""
    rmses = []
    for t, gt_ab in test_data:
        with torch.no_grad():
            pred = forward_with_modified_pw1(t, pw1_modifier)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_r - gt_ab)**2)))
    return np.mean(rmses)


# ================================================================
# Baseline
# ================================================================
print()
print('=' * 70)
print('PW1 MODIFICATION TESTS')
print('=' * 70)
print()

baseline = eval_modifier(lambda W, s, b: W, "baseline")
print(f"Baseline: RMSE = {baseline:.3f}")
print()

# ================================================================
# Test A: Binarize PW1
# ================================================================
print("A) Binary PW1 (sign only, uniform magnitude)")

def binary_pw1(W, s, b):
    """Replace weights with sign(W) * scale."""
    # Scale to preserve norm: ||binary|| ≈ ||original||
    scale = W.norm() / np.sqrt(W.numel())
    return torch.sign(W) * scale

rmse_binary = eval_modifier(binary_pw1, "binary")
print(f"   RMSE = {rmse_binary:.3f} ({(rmse_binary-baseline)/baseline*100:+.2f}%)")

# Row-wise scaling (preserve per-row norm)
def binary_pw1_rowscale(W, s, b):
    row_norms = W.norm(dim=1, keepdim=True)
    n_cols = W.shape[1]
    return torch.sign(W) * row_norms / np.sqrt(n_cols)

rmse_binary_row = eval_modifier(binary_pw1_rowscale, "binary_rowscale")
print(f"   Row-scaled: RMSE = {rmse_binary_row:.3f} ({(rmse_binary_row-baseline)/baseline*100:+.2f}%)")

# ================================================================
# Test B: Ternary PW1 (+1, 0, -1)
# ================================================================
print("\nB) Ternary PW1 (+1/0/-1, threshold = mean(|W|))")

def ternary_pw1(W, s, b):
    threshold = W.abs().mean()
    ternary = torch.zeros_like(W)
    ternary[W > threshold] = 1.0
    ternary[W < -threshold] = -1.0
    # Scale
    scale = W.norm() / (ternary.norm() + 1e-10)
    return ternary * scale

rmse_ternary = eval_modifier(ternary_pw1, "ternary")
print(f"   RMSE = {rmse_ternary:.3f} ({(rmse_ternary-baseline)/baseline*100:+.2f}%)")


# ================================================================
# Test C: Low-bit quantization
# ================================================================
print("\nC) Low-bit quantization of PW1")

def quantize_pw1(bits):
    def modifier(W, s, b):
        n_levels = 2**bits
        w_min, w_max = W.min(), W.max()
        scale = (w_max - w_min) / (n_levels - 1)
        W_q = torch.round((W - w_min) / scale) * scale + w_min
        return W_q
    return modifier

for bits in [8, 4, 2, 1]:
    rmse_q = eval_modifier(quantize_pw1(bits), f"{bits}-bit")
    params_bits = sum(
        v16._get_weight(f'encoder.arch.stages.{s}.{b}.pwconv1.weight').numel()
        for s in range(4) for b in range(depths[s])
    ) * bits / 32
    print(f"   {bits}-bit: RMSE = {rmse_q:.3f} ({(rmse_q-baseline)/baseline*100:+.2f}%)  "
          f"[{params_bits/1e6:.1f}M equivalent params]")


# ================================================================
# Test D: Random hyperplanes with correct bias
# ================================================================
print("\nD) Random hyperplanes (preserving bias + PW2)")

def random_pw1(W, s, b):
    """Random orthogonal matrix with same norm."""
    m, n = W.shape
    # Random Gaussian, then scale rows to match original norms
    R = torch.randn_like(W)
    row_norms_orig = W.norm(dim=1, keepdim=True)
    row_norms_rand = R.norm(dim=1, keepdim=True)
    return R * row_norms_orig / (row_norms_rand + 1e-10)

rmse_random = eval_modifier(random_pw1, "random")
print(f"   RMSE = {rmse_random:.3f} ({(rmse_random-baseline)/baseline*100:+.2f}%)")


# ================================================================
# Test E: PW1 = scaled version of PW2.T
# ================================================================
print("\nE) PW1 from PW2 transpose (ENCODE=DECODE literal)")

def pw1_from_pw2(W, s, b):
    """Use PW2.T as PW1 (scaled to match norm)."""
    prefix = f'encoder.arch.stages.{s}.{b}'
    W2 = v16._get_weight(f'{prefix}.pwconv2.weight')
    # W1 is [4C, C], W2 is [C, 4C], so W2.T is [4C, C]
    W_new = W2.T.clone()
    scale = W.norm() / (W_new.norm() + 1e-10)
    return W_new * scale

rmse_encode_decode = eval_modifier(pw1_from_pw2, "PW2.T")
print(f"   RMSE = {rmse_encode_decode:.3f} ({(rmse_encode_decode-baseline)/baseline*100:+.2f}%)")


# ================================================================
# Test F: Top-K principal components only
# ================================================================
print("\nF) PW1 from top-K SVD components (rank reduction)")

for rank_frac in [0.75, 0.50, 0.25, 0.10]:
    def lowrank_pw1(W, s, b, rf=rank_frac):
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(1, int(len(S) * rf))
        return (U[:, :k] * S[:k]) @ Vt[:k]

    rmse_lr = eval_modifier(lowrank_pw1, f"rank-{rank_frac:.0%}")
    print(f"   Rank {rank_frac:.0%}: RMSE = {rmse_lr:.3f} ({(rmse_lr-baseline)/baseline*100:+.2f}%)")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print(f"{'Method':<35} {'RMSE':<10} {'Δ%':<10} {'PW1 bits/param':<15}")
print("-" * 70)
print(f"  {'Baseline (32-bit)':<33} {baseline:<10.3f} {'—':<10} {'32':<15}")
print(f"  {'8-bit quantized':<33} {eval_modifier(quantize_pw1(8), '8b'):<10.3f} "
      f"{'':>10} {'8':<15}")
print(f"  {'4-bit quantized':<33} {eval_modifier(quantize_pw1(4), '4b'):<10.3f} "
      f"{'':>10} {'4':<15}")
print(f"  {'Binary (sign × row_norm)':<33} {rmse_binary_row:<10.3f} "
      f"{(rmse_binary_row-baseline)/baseline*100:+8.2f}%  {'1 + scale':<15}")
print(f"  {'Ternary (+1/0/-1)':<33} {rmse_ternary:<10.3f} "
      f"{(rmse_ternary-baseline)/baseline*100:+8.2f}%  {'1.58':<15}")
print(f"  {'Rank 50%':<33} {eval_modifier(lambda W,s,b: (lambda U,S,Vt: (U[:,:max(1,len(S)//2)]*S[:max(1,len(S)//2)])@Vt[:max(1,len(S)//2)])(*torch.linalg.svd(W,full_matrices=False)), 'r50'):<10.3f}")
print(f"  {'PW2.T (encode=decode)':<33} {rmse_encode_decode:<10.3f} "
      f"{(rmse_encode_decode-baseline)/baseline*100:+8.2f}%  {'shared':<15}")
print(f"  {'Random hyperplanes':<33} {rmse_random:<10.3f} "
      f"{(rmse_random-baseline)/baseline*100:+8.2f}%  {'random':<15}")
