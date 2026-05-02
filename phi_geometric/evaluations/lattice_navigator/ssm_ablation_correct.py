"""
Correct Ablation — Using the FULL V16 pipeline

Previous ablation experiments had a critical bug: they only ran the encoder
+ UNet decoder and treated 256-channel UNet features as color predictions.

This script runs the FULL pipeline (encoder → UNet → color decoder → refine net)
and ablates neurons at specific encoder blocks to measure actual color impact.
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


def full_forward_with_ablation(v16, img_tensor, ablate_blocks=None, ablate_neurons=None):
    """
    Full V16 forward pass with targeted ablation.
    ablate_blocks: set of (stage, block) tuples — skip γ×MLP for these blocks
    ablate_neurons: dict of {(stage, block): set_of_neuron_indices} — zero specific neurons
    Returns: color prediction [1, 2, H, W]
    """
    if ablate_blocks is None: ablate_blocks = set()
    if ablate_neurons is None: ablate_neurons = {}

    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_input = (img_tensor - mean) / std
    x = x_input.clone()

    features = []
    with torch.no_grad():
        # Stem
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
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))

                # Neuron-level ablation
                key = (si, bi)
                if key in ablate_neurons:
                    mask = torch.ones(post_gelu.shape[-1])
                    for n in ablate_neurons[key]:
                        mask[n] = 0
                    post_gelu = post_gelu * mask

                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0, 3, 1, 2)

                # Block-level ablation
                if key in ablate_blocks:
                    x = res  # Skip MLP entirely
                else:
                    x = res + v16._get_weight(f'{p}.gamma').view(1, -1, 1, 1) * x

            xn = x.permute(0, 2, 3, 1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0, 3, 1, 2))

        # UNet decoder
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

        # Color decoder (9 transformer layers)
        color_out = v16._geometric_color_decoder([out0, out1, out2], out3)

        # Refine net
        coarse_input = torch.cat([color_out, x_input], dim=1)
        final = F.conv2d(coarse_input,
                         v16._get_weight('refine_net.0.0.weight'),
                         v16._get_weight('refine_net.0.0.bias'))

    return final


# ================================================================
# STEP 1: BASELINE — Full pipeline, no ablation
# ================================================================
print()
print('=' * 70)
print('STEP 1: BASELINE — Full Pipeline')
print('=' * 70)
print()

test_images = []
for img_idx in range(300, 330):
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_images.append({'tensor': t, 'gt_ab': gt_ab, 'idx': img_idx})
    if len(test_images) >= 10: break

def eval_ablation(test_images, ablate_blocks=None, ablate_neurons=None):
    """Evaluate color RMSE with ablation using FULL pipeline."""
    rmses = []
    for ti in test_images:
        pred = full_forward_with_ablation(v16, ti['tensor'], ablate_blocks, ablate_neurons)
        pred_ab = pred[0, :2].permute(1, 2, 0).numpy()
        pred_ab_r = cv2.resize(pred_ab, (ti['gt_ab'].shape[1], ti['gt_ab'].shape[0]))
        rmse = np.sqrt(np.mean((pred_ab_r - ti['gt_ab'])**2))
        rmses.append(rmse)
    return np.mean(rmses)

baseline = eval_ablation(test_images)
neutral_rmse = np.mean([np.sqrt(np.mean(ti['gt_ab']**2)) for ti in test_images])
print(f'  Baseline RMSE (full pipeline): {baseline:.2f}')
print(f'  Neutral RMSE:                  {neutral_rmse:.2f}')
print(f'  Model vs neutral:              {(1 - baseline/neutral_rmse)*100:+.1f}%')

# Quick sanity: check that predictions are NOT near-zero
pred = full_forward_with_ablation(v16, test_images[0]['tensor'])
pred_np = pred[0].permute(1, 2, 0).numpy()
print(f'\n  Sanity check — first image prediction:')
print(f'    a*: mean={pred_np[:,:,0].mean():+.2f}  std={pred_np[:,:,0].std():.2f}')
print(f'    b*: mean={pred_np[:,:,1].mean():+.2f}  std={pred_np[:,:,1].std():.2f}')


# ================================================================
# STEP 2: BLOCK-LEVEL ABLATION — Which blocks matter?
# ================================================================
print()
print('=' * 70)
print('STEP 2: PER-BLOCK ABLATION (Skip MLP contribution)')
print('=' * 70)
print()

print(f'{"Block":<10} {"RMSE":<10} {"Δ from base":<12} {"Δ%":<10}')
print('-' * 42)
for si in range(4):
    for bi in range(depths[si]):
        rmse = eval_ablation(test_images, ablate_blocks={(si, bi)})
        delta = rmse - baseline
        delta_pct = (rmse - baseline) / baseline * 100
        marker = ' ***' if abs(delta_pct) > 1.0 else ''
        print(f'  {si}.{bi:<7} {rmse:<10.2f} {delta:+10.2f}   {delta_pct:+8.1f}%{marker}')


# ================================================================
# STEP 3: STAGE-LEVEL ABLATION — Ablate all blocks in a stage
# ================================================================
print()
print('=' * 70)
print('STEP 3: STAGE-LEVEL ABLATION')
print('=' * 70)
print()

for stages in [{0}, {1}, {2}, {3}, {0,1}, {2,3}, {0,1,2,3}]:
    blocks = set()
    for si in stages:
        for bi in range(depths[si]):
            blocks.add((si, bi))
    rmse = eval_ablation(test_images, ablate_blocks=blocks)
    delta_pct = (rmse - baseline) / baseline * 100
    stage_str = '+'.join(str(s) for s in sorted(stages))
    n_blocks = len(blocks)
    print(f'  Stage {stage_str:<10} ({n_blocks:2d} blocks): RMSE={rmse:.2f}  ({delta_pct:+.1f}%)')


# ================================================================
# STEP 4: NEURON-LEVEL ABLATION IN KEY BLOCKS
# ================================================================
print()
print('=' * 70)
print('STEP 4: NEURON-LEVEL ABLATION')
print('=' * 70)
print()

# Find the most impactful block from Step 2 results, then ablate neurons within it
# Test stage 1 block 2 (the one we analyzed in Phases 1-3)
target = (1, 2)
n_neurons = dims[target[0]] * 4  # 192 * 4 = 768

print(f'Ablating neurons in block {target[0]}.{target[1]} ({n_neurons} neurons):')
for frac in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    n = int(n_neurons * frac)
    np.random.seed(42)
    neurons = set(np.random.choice(n_neurons, n, replace=False))
    rmse = eval_ablation(test_images, ablate_neurons={target: neurons})
    delta_pct = (rmse - baseline) / baseline * 100
    print(f'  Ablate {frac:.0%} ({n:3d}): RMSE={rmse:.2f}  ({delta_pct:+.1f}%)')

# Also test the deepest blocks
for target in [(2, 8), (3, 2)]:
    n_neurons = dims[target[0]] * 4
    print(f'\nAblating neurons in block {target[0]}.{target[1]} ({n_neurons} neurons):')
    for frac in [0.5, 1.0]:
        n = int(n_neurons * frac)
        np.random.seed(42)
        neurons = set(np.random.choice(n_neurons, n, replace=False))
        rmse = eval_ablation(test_images, ablate_neurons={target: neurons})
        delta_pct = (rmse - baseline) / baseline * 100
        print(f'  Ablate {frac:.0%} ({n:4d}): RMSE={rmse:.2f}  ({delta_pct:+.1f}%)')


# ================================================================
# STEP 5: MASSIVE ABLATION — Multiple blocks at once
# ================================================================
print()
print('=' * 70)
print('STEP 5: MASSIVE ABLATION — All neurons in multiple blocks')
print('=' * 70)
print()

# Ablate ALL neurons in ALL blocks of a stage
for stage_set in [{0}, {1}, {2}, {3}, {0,1,2,3}]:
    ablate_neurons = {}
    total = 0
    for si in stage_set:
        for bi in range(depths[si]):
            n_neurons = dims[si] * 4
            ablate_neurons[(si, bi)] = set(range(n_neurons))
            total += n_neurons
    rmse = eval_ablation(test_images, ablate_neurons=ablate_neurons)
    delta_pct = (rmse - baseline) / baseline * 100
    stage_str = '+'.join(str(s) for s in sorted(stage_set))
    print(f'  All neurons stage {stage_str:<10} ({total:5d} total): RMSE={rmse:.2f}  ({delta_pct:+.1f}%)')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print(f'  Baseline RMSE: {baseline:.2f}')
print(f'  Neutral RMSE:  {neutral_rmse:.2f}')
print(f'  Full pipeline now used (encoder → UNet → color decoder → refine net)')
print(f'  This is the CORRECT measurement of encoder impact on color prediction.')
