"""
Phase 5B: Do Resonant Neurons Predict Functional Importance?

Phase 3 identified "resonant neurons" — neurons whose W₁ row (read direction)
and W₂ column (write direction) both align with a semantic category centroid.

Phase 5A found that block 2.8 is the most critical (+63.9% when ablated).

This script tests: if we identify resonant neurons in block 2.8 using
purely geometric criteria, do they overlap with the functionally important
neurons (measured by ablation impact on RMSE)?

If yes: geometry predicts function. The φ-angular lattice IS the code.
If no: geometry and function are decoupled. Structure ≠ information.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

# Color categories
COLOR_CATEGORIES = {
    'red': ((0, 30), (150, 180)), 'orange': ((5, 25),), 'yellow': ((20, 40),),
    'green': ((35, 85),), 'cyan': ((80, 100),), 'blue': ((100, 130),),
    'magenta': ((140, 170),), 'neutral': 'low_sat',
}

def classify_patch(hsv_patch):
    h_mean = hsv_patch[:, :, 0].mean()
    s_mean = hsv_patch[:, :, 1].mean()
    if s_mean < 40: return 'neutral'
    for cat, ranges in COLOR_CATEGORIES.items():
        if cat == 'neutral': continue
        for lo, hi in ranges:
            if lo <= h_mean <= hi: return cat
    return 'neutral'


# ================================================================
# STEP 1: Build semantic centroids from image features
# ================================================================
print('=' * 70)
print('STEP 1: Build semantic centroids')
print('=' * 70)
print()

N_IMAGES = 80
image_data = []
for idx in range(150, 150 + N_IMAGES * 3):
    if len(image_data) >= N_IMAGES: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    cat = classify_patch(hsv)
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    image_data.append({'tensor': t, 'category': cat, 'idx': idx, 'gt_ab': gt_ab})

cats = defaultdict(int)
for d in image_data: cats[d['category']] += 1
print(f'Loaded {len(image_data)} images. Categories: {dict(cats)}')


def extract_block_activations(v16, img_tensor, target_block):
    """Extract post-GELU activations at a specific block."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

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
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{p}.norm.weight'),
                                 v16._get_weight(f'{p}.norm.bias'))
                pre_gelu = F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                    v16._get_weight(f'{p}.pwconv1.bias'))
                post_gelu = gate(pre_gelu)

                if (si, bi) == target_block:
                    # Return: post_gelu [1, H, W, expanded_dim], residual input [1, C, H, W]
                    return post_gelu.squeeze(0).numpy(), res.squeeze(0).permute(1, 2, 0).numpy()

                mlp_out = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                                   v16._get_weight(f'{p}.pwconv2.bias'))
                x = mlp_out.permute(0, 3, 1, 2)
                x = res + v16._get_weight(f'{p}.gamma').view(1, -1, 1, 1) * x

    return None, None


# Analyze blocks: 2.8 (most critical), 1.0 (second), 1.2 (harmful - control)
TARGET_BLOCKS = [(2, 8), (1, 0), (1, 2)]

for target_block in TARGET_BLOCKS:
    si, bi = target_block
    d_in = dims[si]
    d_exp = dims[si] * 4  # expanded dimension
    print(f'\n{"="*70}')
    print(f'BLOCK {si}.{bi} (dim_in={d_in}, dim_exp={d_exp})')
    print(f'{"="*70}')

    # ================================================================
    # STEP 2: Identify resonant neurons using Phase 3 methodology
    # ================================================================
    print(f'\nSTEP 2: Identify resonant neurons for block {si}.{bi}')

    # Get W1, W2, gamma
    w1 = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.pwconv1.weight').numpy()  # [d_exp, d_in]
    w2 = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.pwconv2.weight').numpy()  # [d_in, d_exp]
    gamma = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.gamma').numpy()  # [d_in]

    # Build semantic centroids from activations
    cat_activations = defaultdict(list)
    for img_d in image_data:
        post_gelu, res_input = extract_block_activations(v16, img_d['tensor'], target_block)
        if post_gelu is None: continue
        # Residual = input feature - image mean
        img_mean = res_input.mean(axis=(0, 1), keepdims=True)
        residual = res_input - img_mean
        cat_activations[img_d['category']].append(residual.mean(axis=(0, 1)))

    centroids = {}
    for cat, feats in cat_activations.items():
        if len(feats) >= 3:
            centroids[cat] = np.mean(feats, axis=0)

    print(f'  Categories with centroids: {list(centroids.keys())}')

    # For each neuron, compute:
    # - read_align: max |cos(w1_row, centroid)| across categories
    # - write_align: max |cos(w2_col * gamma, centroid)| across categories
    # - resonance: both read AND write align with same category

    neuron_stats = []
    for n in range(d_exp):
        w1_row = w1[n]  # [d_in]
        w2_col = w2[:, n] * gamma  # [d_in] (scaled by gamma)

        read_aligns = {}
        write_aligns = {}
        for cat, cent in centroids.items():
            r_cos = np.dot(w1_row, cent) / (np.linalg.norm(w1_row) * np.linalg.norm(cent) + 1e-10)
            w_cos = np.dot(w2_col, cent) / (np.linalg.norm(w2_col) * np.linalg.norm(cent) + 1e-10)
            read_aligns[cat] = r_cos
            write_aligns[cat] = w_cos

        best_read_cat = max(read_aligns, key=lambda c: abs(read_aligns[c]))
        best_write_cat = max(write_aligns, key=lambda c: abs(write_aligns[c]))
        max_read = abs(read_aligns[best_read_cat])
        max_write = abs(write_aligns[best_write_cat])

        # Read-write orthogonality
        rw_cos = np.dot(w1_row, w2_col) / (np.linalg.norm(w1_row) * np.linalg.norm(w2_col) + 1e-10)

        neuron_stats.append({
            'neuron': n,
            'max_read_align': max_read,
            'max_write_align': max_write,
            'best_read_cat': best_read_cat,
            'best_write_cat': best_write_cat,
            'resonant': best_read_cat == best_write_cat and max_read > 0.1 and max_write > 0.1,
            'rw_cos': rw_cos,
            'combined_align': max_read * max_write,
            'w1_norm': np.linalg.norm(w1_row),
            'w2_norm': np.linalg.norm(w2_col),
        })

    resonant = [ns for ns in neuron_stats if ns['resonant']]
    print(f'  Resonant neurons: {len(resonant)}/{d_exp} '
          f'({len(resonant)/d_exp*100:.1f}%)')

    # Top 10 by combined alignment
    sorted_by_combined = sorted(neuron_stats, key=lambda x: x['combined_align'], reverse=True)
    print(f'  Top 10 by combined read×write alignment:')
    for ns in sorted_by_combined[:10]:
        r_cat = ns['best_read_cat']
        w_cat = ns['best_write_cat']
        print(f'    n{ns["neuron"]:4d}: read={ns["max_read_align"]:.3f}({r_cat:<8s}) '
              f'write={ns["max_write_align"]:.3f}({w_cat:<8s}) '
              f'rw_cos={ns["rw_cos"]:+.3f} {"RESONANT" if ns["resonant"] else ""}')

    # ================================================================
    # STEP 3: Measure functional importance per neuron
    # ================================================================
    print(f'\nSTEP 3: Measure functional importance per neuron (block {si}.{bi})')

    # Use 5 test images
    test_imgs = []
    for idx in [300, 305, 310, 315, 320]:
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        gt_ab = lab[:, :, 1:].astype(float) - 128.0
        test_imgs.append((t, gt_ab))

    def eval_ablation(v16, test_imgs, target_block, neurons_to_ablate):
        """Evaluate RMSE with specific neurons ablated in target block."""
        gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        rmses = []
        for img_t, gt_ab in test_imgs:
            x_input = (img_t - mean_t) / std_t
            x = x_input.clone()
            features = []

            with torch.no_grad():
                x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                             v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                                 v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
                x = x.permute(0, 3, 1, 2)

                for s in range(4):
                    dd = dims[s]
                    if s > 0:
                        p = f'encoder.arch.downsample_layers.{s}'
                        x = x.permute(0, 2, 3, 1)
                        x = F.layer_norm(x, (dims[s-1],),
                                         v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                        x = x.permute(0, 3, 1, 2)
                        x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'),
                                     v16._get_weight(f'{p}.1.bias'), stride=2)

                    for b in range(depths[s]):
                        p = f'encoder.arch.stages.{s}.{b}'
                        res = x
                        x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                                     v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=dd)
                        x = x.permute(0, 2, 3, 1)
                        x = F.layer_norm(x, (dd,), v16._get_weight(f'{p}.norm.weight'),
                                         v16._get_weight(f'{p}.norm.bias'))
                        post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                                  v16._get_weight(f'{p}.pwconv1.bias')))

                        if (s, b) == target_block and neurons_to_ablate is not None:
                            mask = torch.ones(post_gelu.shape[-1])
                            for n in neurons_to_ablate:
                                mask[n] = 0
                            post_gelu = post_gelu * mask

                        x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                                     v16._get_weight(f'{p}.pwconv2.bias'))
                        x = x.permute(0, 3, 1, 2)
                        x = res + v16._get_weight(f'{p}.gamma').view(1, -1, 1, 1) * x

                    xn = x.permute(0, 2, 3, 1)
                    xn = F.layer_norm(xn, (dd,), v16._get_weight(f'encoder.arch.norm{s}.weight'),
                                      v16._get_weight(f'encoder.arch.norm{s}.bias'))
                    features.append(xn.permute(0, 3, 1, 2))

                out0 = v16._geometric_unet_block(features[3], features[2], 0)
                out1 = v16._geometric_unet_block(out0, features[1], 1)
                out2 = v16._geometric_unet_block(out1, features[0], 2)
                out3 = v16._geometric_last_shuf(out2)
                color_out = v16._geometric_color_decoder([out0, out1, out2], out3)
                coarse_input = torch.cat([color_out, x_input], dim=1)
                final = F.conv2d(coarse_input, v16._get_weight('refine_net.0.0.weight'),
                                 v16._get_weight('refine_net.0.0.bias'))

            pred_ab = final[0, :2].permute(1, 2, 0).numpy()
            pred_ab_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
            rmses.append(np.sqrt(np.mean((pred_ab_r - gt_ab)**2)))

        return np.mean(rmses)

    baseline_rmse = eval_ablation(v16, test_imgs, target_block, None)
    print(f'  Baseline RMSE: {baseline_rmse:.3f}')

    # Measure importance of neuron groups
    # Sort neurons by different geometric criteria
    n_neurons = d_exp
    n_groups = 5  # quintiles
    group_size = n_neurons // n_groups

    # Strategy 1: Ablate by combined alignment (geometric prediction)
    sorted_combined = sorted(neuron_stats, key=lambda x: x['combined_align'], reverse=True)
    combined_order = [ns['neuron'] for ns in sorted_combined]

    # Strategy 2: Ablate by W1 norm (magnitude prediction)
    sorted_w1norm = sorted(neuron_stats, key=lambda x: x['w1_norm'], reverse=True)
    w1norm_order = [ns['neuron'] for ns in sorted_w1norm]

    # Strategy 3: Random order
    np.random.seed(42)
    random_order = np.random.permutation(n_neurons).tolist()

    # Strategy 4: Ablate resonant neurons first
    resonant_ids = set(ns['neuron'] for ns in resonant)
    non_resonant_ids = [n for n in range(n_neurons) if n not in resonant_ids]
    np.random.shuffle(non_resonant_ids)
    resonant_first = list(resonant_ids) + non_resonant_ids

    strategies = {
        'Combined align (top→bottom)': combined_order,
        'W1 norm (top→bottom)': w1norm_order,
        'Random': random_order,
        'Resonant first': resonant_first,
    }

    print(f'\n  Ablation by strategy (ablating top-N neurons):')
    print(f'  {"Strategy":<35} ', end='')
    fracs = [0.05, 0.10, 0.25, 0.50]
    for fr in fracs:
        print(f'{fr*100:3.0f}%    ', end='')
    print()
    print(f'  {"-"*67}')

    strategy_results = {}
    for strat_name, order in strategies.items():
        results = []
        for frac in fracs:
            n_ablate = max(1, int(n_neurons * frac))
            neurons = set(order[:n_ablate])
            rmse = eval_ablation(v16, test_imgs, target_block, neurons)
            delta_pct = (rmse - baseline_rmse) / baseline_rmse * 100
            results.append(delta_pct)

        strategy_results[strat_name] = results
        print(f'  {strat_name:<35} ', end='')
        for dp in results:
            print(f'{dp:+6.1f}%  ', end='')
        print()

    # ================================================================
    # STEP 4: Reverse test — keep only geometric neurons, ablate rest
    # ================================================================
    print(f'\n  Reverse test (KEEP top-N, ablate rest):')
    print(f'  {"Strategy":<35} ', end='')
    for fr in fracs:
        print(f'{fr*100:3.0f}%    ', end='')
    print()
    print(f'  {"-"*67}')

    for strat_name, order in strategies.items():
        results = []
        for frac in fracs:
            n_keep = max(1, int(n_neurons * frac))
            keep_set = set(order[:n_keep])
            ablate_set = set(range(n_neurons)) - keep_set
            rmse = eval_ablation(v16, test_imgs, target_block, ablate_set)
            delta_pct = (rmse - baseline_rmse) / baseline_rmse * 100
            results.append(delta_pct)

        print(f'  {strat_name:<35} ', end='')
        for dp in results:
            print(f'{dp:+6.1f}%  ', end='')
        print()


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('FINAL SUMMARY')
print('=' * 70)
print()
print('Question: Do resonant neurons (identified from geometry) predict')
print('which neurons are functionally important (measured by ablation)?')
print()
print('If geometric strategies cause MORE damage when ablated (or preserve')
print('MORE function when kept) than random, then geometry predicts function.')
