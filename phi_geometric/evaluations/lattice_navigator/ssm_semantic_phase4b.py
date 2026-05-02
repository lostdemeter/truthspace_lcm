"""
Phase 4B: Why does single-block ablation have zero effect?

Phase 4 found: ablating ALL 768 neurons in stage1.block2 = no RMSE change.
The semantic code is distributed across ALL blocks, not just one.

This script measures:
  1. γ magnitude and MLP output magnitude per block
  2. MLP contribution vs residual magnitude per block
  3. Multi-block ablation (ablate entire stages)
  4. Where does the 28.2% gap actually live?
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ================================================================
# STEP 1: γ VALUES AND MLP CONTRIBUTIONS ACROSS ALL BLOCKS
# ================================================================
print('=' * 70)
print('STEP 1: γ VALUES AND MLP CONTRIBUTION ACROSS ALL BLOCKS')
print('=' * 70)
print()

gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
m_t = torch.tensor([.485,.456,.406]).view(1,3,1,1)
s_t = torch.tensor([.229,.224,.225]).view(1,3,1,1)

# Load test image
im = cv2.imread(all_imgs[50])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.

x = (t - m_t) / s_t
block_info = []

with torch.no_grad():
    x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                 v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
    x = x.permute(0,2,3,1)
    x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
    x = x.permute(0,3,1,2)

    for si in range(4):
        d = dims[si]
        if si > 0:
            p = f'encoder.arch.downsample_layers.{si}'
            x = x.permute(0,2,3,1)
            x = F.layer_norm(x, (dims[si-1],),
                             v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
            x = x.permute(0,3,1,2)
            x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)

        for bi in range(depths[si]):
            p = f'encoder.arch.stages.{si}.{bi}'
            res = x

            x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                         v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
            x = x.permute(0,2,3,1)
            x = F.layer_norm(x, (d,),
                             v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
            post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                      v16._get_weight(f'{p}.pwconv1.bias')))
            mlp_out = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                               v16._get_weight(f'{p}.pwconv2.bias'))
            x = mlp_out.permute(0,3,1,2)
            gamma = v16._get_weight(f'{p}.gamma')

            # Measure contributions
            gamma_vals = gamma.detach().numpy()
            res_mag = torch.norm(res).item()
            mlp_mag = torch.norm(x).item()
            gamma_scaled_mag = torch.norm(gamma.view(1,-1,1,1) * x).item()
            contribution_ratio = gamma_scaled_mag / (res_mag + 1e-8)

            block_info.append({
                'stage': si, 'block': bi,
                'gamma_mean': np.abs(gamma_vals).mean(),
                'gamma_max': np.abs(gamma_vals).max(),
                'gamma_min': np.abs(gamma_vals).min(),
                'residual_mag': res_mag,
                'mlp_mag': mlp_mag,
                'gamma_scaled_mag': gamma_scaled_mag,
                'contribution_ratio': contribution_ratio,
            })

            x = res + gamma.view(1,-1,1,1) * x

print(f'{"Stage.Block":<12} {"γ_mean":<8} {"γ_max":<8} {"Res_mag":<10} {"MLP_mag":<10} '
      f'{"γ×MLP_mag":<10} {"Ratio":<8}')
print('-' * 70)
for info in block_info:
    print(f'  {info["stage"]}.{info["block"]:<9} {info["gamma_mean"]:<8.4f} {info["gamma_max"]:<8.4f} '
          f'{info["residual_mag"]:<10.1f} {info["mlp_mag"]:<10.1f} '
          f'{info["gamma_scaled_mag"]:<10.1f} {info["contribution_ratio"]:<8.4f}')


# ================================================================
# STEP 2: MULTI-BLOCK ABLATION — Ablate entire stages
# ================================================================
print()
print('=' * 70)
print('STEP 2: MULTI-BLOCK ABLATION — Entire Stages')
print('=' * 70)
print()

def run_full_encoder_with_stage_ablation(v16, img_tensor, ablate_stages=None):
    """Run encoder with entire stage MLP ablation (skip γ×MLP for ablated stages)."""
    if ablate_stages is None: ablate_stages = set()
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)

                if si in ablate_stages:
                    x = res  # Skip MLP contribution entirely
                else:
                    x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()

# Test images
test_images = []
for img_idx in range(300, 400):
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    test_images.append({'tensor': t, 'ab': ab, 'idx': img_idx})
    if len(test_images) >= 10: break

def eval_stage_ablation(test_images, ablate_stages):
    rmses = []
    for ti in test_images:
        pred = run_full_encoder_with_stage_ablation(v16, ti['tensor'], ablate_stages)
        ab_pred_size = cv2.resize(ti['ab'], (pred.shape[2], pred.shape[1]))
        rmse = np.sqrt(np.mean((pred[0] - ab_pred_size[:,:,0])**2 +
                               (pred[1] - ab_pred_size[:,:,1])**2))
        rmses.append(rmse)
    return np.mean(rmses)

baseline = eval_stage_ablation(test_images, set())
print(f'  Baseline (no ablation):        RMSE = {baseline:.2f}')

for stages in [{0}, {1}, {2}, {3}, {0,1}, {2,3}, {0,1,2,3}]:
    rmse = eval_stage_ablation(test_images, stages)
    delta = (rmse - baseline) / baseline * 100
    stage_str = '+'.join(str(s) for s in sorted(stages))
    blocks_ablated = sum(depths[s] for s in stages)
    print(f'  Ablate stage {stage_str:<7} ({blocks_ablated:2d} blocks): RMSE = {rmse:.2f} ({delta:+.1f}%)')


# ================================================================
# STEP 3: PER-BLOCK ABLATION — Which individual blocks matter?
# ================================================================
print()
print('=' * 70)
print('STEP 3: PER-BLOCK ABLATION — Which Blocks Matter?')
print('=' * 70)
print()

def run_full_encoder_with_block_ablation(v16, img_tensor, ablate_blocks=None):
    """Run encoder with specific block MLP ablation.
    ablate_blocks: set of (stage, block) tuples."""
    if ablate_blocks is None: ablate_blocks = set()
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)

                if (si, bi) in ablate_blocks:
                    x = res
                else:
                    x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()

def eval_block_ablation(test_images, ablate_blocks):
    rmses = []
    for ti in test_images:
        pred = run_full_encoder_with_block_ablation(v16, ti['tensor'], ablate_blocks)
        ab_pred_size = cv2.resize(ti['ab'], (pred.shape[2], pred.shape[1]))
        rmse = np.sqrt(np.mean((pred[0] - ab_pred_size[:,:,0])**2 +
                               (pred[1] - ab_pred_size[:,:,1])**2))
        rmses.append(rmse)
    return np.mean(rmses)

print(f'{"Block":<12} {"RMSE":<10} {"Delta":<10} {"γ_mean":<10}')
print('-' * 42)
bi_global = 0
for si in range(4):
    for bi in range(depths[si]):
        gamma = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.gamma').detach().numpy()
        rmse = eval_block_ablation(test_images, {(si, bi)})
        delta = (rmse - baseline) / baseline * 100
        print(f'  {si}.{bi:<9} {rmse:<10.2f} {delta:+8.1f}%  {np.abs(gamma).mean():.4f}')
        bi_global += 1


# ================================================================
# STEP 4: ALSO ABLATE DWCONV — What about spatial processing?
# ================================================================
print()
print('=' * 70)
print('STEP 4: DWCONV vs MLP — What carries the signal?')
print('=' * 70)
print()

def run_encoder_ablate_dwconv(v16, img_tensor, ablate_dwconv_stages=None):
    """Run encoder skipping dwconv for specified stages (pass residual directly to norm→MLP)."""
    if ablate_dwconv_stages is None: ablate_dwconv_stages = set()
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x

                if si not in ablate_dwconv_stages:
                    x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                                 v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                # If ablated, skip dwconv — x is still the residual in NCHW
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()

def eval_dwconv_ablation(test_images, ablate_stages):
    rmses = []
    for ti in test_images:
        pred = run_encoder_ablate_dwconv(v16, ti['tensor'], ablate_stages)
        ab_pred_size = cv2.resize(ti['ab'], (pred.shape[2], pred.shape[1]))
        rmse = np.sqrt(np.mean((pred[0] - ab_pred_size[:,:,0])**2 +
                               (pred[1] - ab_pred_size[:,:,1])**2))
        rmses.append(rmse)
    return np.mean(rmses)

print(f'  Baseline:                  RMSE = {baseline:.2f}')
for stages in [{0}, {1}, {2}, {3}, {0,1,2,3}]:
    rmse = eval_dwconv_ablation(test_images, stages)
    delta = (rmse - baseline) / baseline * 100
    stage_str = '+'.join(str(s) for s in sorted(stages))
    print(f'  Ablate dwconv stage {stage_str:<7}: RMSE = {rmse:.2f} ({delta:+.1f}%)')


# ================================================================
# STEP 5: WHERE IS THE 28.2% GAP? — Decoder ablation
# ================================================================
print()
print('=' * 70)
print('STEP 5: WHERE IS THE GAP? — Which stages feed the decoder?')
print('=' * 70)
print()

def run_with_zeroed_features(v16, img_tensor, zero_stages=None):
    """Run encoder but zero out specific stage features before U-Net."""
    if zero_stages is None: zero_stages = set()
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            feat = xn.permute(0,3,1,2)
            if si in zero_stages:
                feat = torch.zeros_like(feat)
            features.append(feat)

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()

def eval_zeroed_features(test_images, zero_stages):
    rmses = []
    for ti in test_images:
        pred = run_with_zeroed_features(v16, ti['tensor'], zero_stages)
        ab_pred_size = cv2.resize(ti['ab'], (pred.shape[2], pred.shape[1]))
        rmse = np.sqrt(np.mean((pred[0] - ab_pred_size[:,:,0])**2 +
                               (pred[1] - ab_pred_size[:,:,1])**2))
        rmses.append(rmse)
    return np.mean(rmses)

print(f'  Baseline:                    RMSE = {baseline:.2f}')
for stages in [{0}, {1}, {2}, {3}, {3,2}, {3,2,1}, {0,1,2,3}]:
    rmse = eval_zeroed_features(test_images, stages)
    delta = (rmse - baseline) / baseline * 100
    stage_str = '+'.join(str(s) for s in sorted(stages))
    print(f'  Zero stage {stage_str:<7} features: RMSE = {rmse:.2f} ({delta:+.1f}%)')


print()
print('=' * 70)
print('SUMMARY — Phase 4B')
print('=' * 70)
print()
print('The ablation reveals WHERE the semantic content lives.')
