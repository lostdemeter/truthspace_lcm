"""
Phase 5C: Can We Improve the Model by Ablating Harmful Blocks?

Blocks 1.1 and 1.2 are slightly harmful (-1.3%, -1.9% when ablated).
Block 2.2 is also harmful (-1.9%).

Test: ablate these blocks across many images and measure consistent improvement.
Also test combined ablation of all harmful blocks.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]
v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def full_forward_skip_blocks(v16, img_tensor, skip_blocks=None):
    """Full pipeline with blocks skipped (residual only, no MLP)."""
    if skip_blocks is None: skip_blocks = set()
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_input = (img_tensor - mean) / std
    x = x_input.clone()
    features = []

    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)

        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)

            for bi in range(depths[si]):
                if (si, bi) in skip_blocks:
                    continue  # pure skip — no MLP contribution
                x = v16._geometric_convnext_block(x, f'encoder.arch.stages.{si}.{bi}', d)

            xn = x.permute(0, 2, 3, 1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0, 3, 1, 2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
        color_out = v16._geometric_color_decoder([out0, out1, out2], out3)
        coarse_input = torch.cat([color_out, x_input], dim=1)
        final = F.conv2d(coarse_input, v16._get_weight('refine_net.0.0.weight'),
                         v16._get_weight('refine_net.0.0.bias'))
    return final


# Load test images
print('Loading test images...')
test_data = []
for idx in range(200, 400):
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append((t, gt_ab))
    if len(test_data) >= 50: break

print(f'Loaded {len(test_data)} test images')


def eval_config(skip_blocks=None):
    """Evaluate model with specific blocks skipped."""
    rmses = []
    for t, gt_ab in test_data:
        pred = full_forward_skip_blocks(v16, t, skip_blocks)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_ab_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
        rmses.append(np.sqrt(np.mean((pred_ab_r - gt_ab)**2)))
    return np.array(rmses)


# ================================================================
# Test configurations
# ================================================================
print()
print('=' * 70)
print('ABLATION OF HARMFUL BLOCKS (50 images)')
print('=' * 70)
print()

configs = {
    'Baseline (no ablation)': set(),
    'Skip block 1.1': {(1, 1)},
    'Skip block 1.2': {(1, 2)},
    'Skip block 2.2': {(2, 2)},
    'Skip blocks 1.1+1.2': {(1, 1), (1, 2)},
    'Skip blocks 1.1+1.2+2.2': {(1, 1), (1, 2), (2, 2)},
    'Skip block 2.5 (near-zero impact)': {(2, 5)},
    'Skip block 2.8 (most critical)': {(2, 8)},
}

baseline_rmses = None
print(f'{"Configuration":<40} {"Mean RMSE":<10} {"Δ%":<10} {"Improved":<10} {"Wilcoxon p":<10}')
print('-' * 80)

results = {}
for name, skip in configs.items():
    rmses = eval_config(skip)
    results[name] = rmses
    if baseline_rmses is None:
        baseline_rmses = rmses
        print(f'  {name:<38} {rmses.mean():<10.3f}')
    else:
        delta_pct = (rmses.mean() - baseline_rmses.mean()) / baseline_rmses.mean() * 100
        n_improved = np.sum(rmses < baseline_rmses)
        # Simple sign test p-value
        from scipy.stats import wilcoxon
        try:
            stat, p_val = wilcoxon(baseline_rmses, rmses)
        except:
            p_val = 1.0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f'  {name:<38} {rmses.mean():<10.3f} {delta_pct:+8.2f}%  {n_improved:3d}/50     '
              f'{p_val:.4f} {sig}')


print()
print('=' * 70)
print('PER-IMAGE ANALYSIS: Best configuration per image')
print('=' * 70)
print()

# For each image, which config is best?
config_wins = {name: 0 for name in results}
for i in range(len(test_data)):
    best_name = min(results.keys(), key=lambda n: results[n][i])
    config_wins[best_name] += 1

for name, wins in sorted(config_wins.items(), key=lambda x: -x[1]):
    print(f'  {name:<40} wins: {wins}/50')
