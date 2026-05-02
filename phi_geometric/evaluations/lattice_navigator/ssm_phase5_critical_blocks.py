"""
Phase 5: Geometric Analysis of Functionally Critical Blocks

Previous phases analyzed block 1.2 (768 neurons) — which turns out to be
slightly HARMFUL (-1.9% when ablated). Now we analyze the blocks that
actually matter:

  Block 2.8: +63.9% RMSE when ablated (MOST critical)
  Block 1.0: +42.7%
  Block 0.1: +36.3%
  Block 3.2: +27.8%

Questions:
  1. Does the φ-angular lattice appear in these critical blocks?
  2. Do critical blocks have different geometric signatures than harmful ones?
  3. Can we predict block importance from geometric features alone?
  4. Do resonant neurons in critical blocks overlap with high-impact neurons?
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

# Block importance from correct ablation
BLOCK_IMPORTANCE = {
    (0, 0): 13.2, (0, 1): 36.3, (0, 2): 4.2,
    (1, 0): 42.7, (1, 1): -1.3, (1, 2): -1.9,
    (2, 0): 3.9, (2, 1): 1.6, (2, 2): -1.9, (2, 3): 5.8,
    (2, 4): 7.6, (2, 5): 0.4, (2, 6): 1.6, (2, 7): 6.4,
    (2, 8): 63.9,
    (3, 0): 8.2, (3, 1): 21.5, (3, 2): 27.8,
}

# Color categories for semantic analysis
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
# STEP 1: EXTRACT FEATURES AT EACH BLOCK FOR MULTIPLE IMAGES
# ================================================================
print('=' * 70)
print('STEP 1: Extract features per block across images')
print('=' * 70)
print()

def extract_block_features(v16, img_tensor):
    """Run encoder and capture output features at each block."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    block_features = {}
    block_mlp_outputs = {}

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
                mlp_out = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                                   v16._get_weight(f'{p}.pwconv2.bias'))
                mlp_out_scaled = mlp_out.permute(0, 3, 1, 2)
                gamma = v16._get_weight(f'{p}.gamma').view(1, -1, 1, 1)
                mlp_contribution = gamma * mlp_out_scaled

                x = (res + mlp_contribution).squeeze(0).permute(1, 2, 0)  # [H, W, C]
                block_features[(si, bi)] = x.numpy().copy()
                block_mlp_outputs[(si, bi)] = mlp_contribution.squeeze(0).permute(1, 2, 0).numpy().copy()
                x = x.unsqueeze(0).permute(0, 3, 1, 2)  # back to [1, C, H, W]

    return block_features, block_mlp_outputs


# Collect features for many images with color labels
N_IMAGES = 60
image_data = []
for idx in range(200, 200 + N_IMAGES * 3):
    if len(image_data) >= N_IMAGES: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    cat = classify_patch(hsv)
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    image_data.append({'tensor': t, 'category': cat, 'idx': idx})

print(f'Loaded {len(image_data)} images')
cats = defaultdict(int)
for d in image_data: cats[d['category']] += 1
print(f'Categories: {dict(cats)}')

# Extract features for all images at all blocks
print('Extracting features...')
all_block_features = defaultdict(list)
all_block_mlp = defaultdict(list)
all_categories = []

for i, d in enumerate(image_data):
    bf, bm = extract_block_features(v16, d['tensor'])
    for key in bf:
        all_block_features[key].append(bf[key])
        all_block_mlp[key].append(bm[key])
    all_categories.append(d['category'])
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(image_data)}')

print(f'Done. Analyzing {len(all_block_features)} blocks.')


# ================================================================
# STEP 2: φ-ANGULAR LATTICE PER BLOCK
# ================================================================
print()
print('=' * 70)
print('STEP 2: φ-Angular Lattice per Block')
print('=' * 70)
print()

phi_angles = {
    'π/φ³': np.pi / PHI**3,
    'π/φ²': np.pi / PHI**2,
    'π/2': np.pi / 2,
    'π/φ': np.pi / PHI,
    '2π/φ²': 2 * np.pi / PHI**2,
}

unique_cats = sorted(set(all_categories))
if len(unique_cats) < 3:
    print(f'  Only {len(unique_cats)} categories — need more diversity. Using all blocks anyway.')

def compute_residual_centroids(block_key, features_list, categories):
    """Compute per-category centroids in residual space (feature - image mean)."""
    cat_features = defaultdict(list)
    for feat, cat in zip(features_list, categories):
        # feat: [H, W, C] — compute image mean and subtract
        img_mean = feat.mean(axis=(0, 1), keepdims=True)
        residual = feat - img_mean
        # Average residual per image
        cat_features[cat].append(residual.mean(axis=(0, 1)))

    centroids = {}
    for cat, feats in cat_features.items():
        if len(feats) >= 3:
            centroids[cat] = np.mean(feats, axis=0)
    return centroids

def compute_pairwise_angles(centroids):
    """Compute all pairwise angles between category centroids."""
    cats = sorted(centroids.keys())
    angles = {}
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            v1, v2 = centroids[c1], centroids[c2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles[(c1, c2)] = np.arccos(np.clip(cos, -1, 1))
    return angles

def score_phi_lattice(angles):
    """Score how well pairwise angles match φ-reference angles."""
    if not angles: return 0.0, {}
    ref_vals = list(phi_angles.values())
    best_matches = {}
    total_error = 0
    for pair, angle in angles.items():
        errors = [abs(angle - r) for r in ref_vals]
        best_idx = np.argmin(errors)
        best_ref = list(phi_angles.keys())[best_idx]
        best_error = errors[best_idx]
        best_matches[pair] = (best_ref, np.degrees(angle), np.degrees(best_error))
        total_error += best_error
    mean_error_deg = np.degrees(total_error / len(angles))
    return mean_error_deg, best_matches

print(f'{"Block":<8} {"Importance%":<12} {"N cats":<7} {"N pairs":<8} {"Mean φ-err°":<12} {"Best ref":<12}')
print('-' * 59)

block_phi_scores = {}
for si in range(4):
    for bi in range(depths[si]):
        key = (si, bi)
        centroids = compute_residual_centroids(key, all_block_features[key], all_categories)
        angles = compute_pairwise_angles(centroids)
        mean_err, matches = score_phi_lattice(angles)
        imp = BLOCK_IMPORTANCE.get(key, 0)
        # Count most common reference
        ref_counts = defaultdict(int)
        for _, (ref, _, _) in matches.items():
            ref_counts[ref] += 1
        top_ref = max(ref_counts, key=ref_counts.get) if ref_counts else '—'
        print(f'  {si}.{bi:<5} {imp:+10.1f}%  {len(centroids):<6} {len(angles):<7} {mean_err:<11.1f}  {top_ref}')
        block_phi_scores[key] = mean_err


# ================================================================
# STEP 3: GEOMETRIC SIGNATURES — What distinguishes critical blocks?
# ================================================================
print()
print('=' * 70)
print('STEP 3: Geometric Signatures of Critical vs Harmful Blocks')
print('=' * 70)
print()

def block_geometric_stats(block_key, features_list, mlp_list):
    """Compute geometric statistics for a block."""
    stats = {}

    # Feature magnitude
    norms = [np.linalg.norm(f.mean(axis=(0, 1))) for f in features_list]
    stats['feat_norm_mean'] = np.mean(norms)

    # MLP contribution magnitude
    mlp_norms = [np.linalg.norm(m.mean(axis=(0, 1))) for m in mlp_list]
    stats['mlp_norm_mean'] = np.mean(mlp_norms)

    # MLP/residual ratio
    stats['mlp_ratio'] = stats['mlp_norm_mean'] / (stats['feat_norm_mean'] + 1e-10)

    # Feature variance (how much features vary across images)
    means = np.array([f.mean(axis=(0, 1)) for f in features_list])
    stats['cross_image_var'] = np.var(means, axis=0).mean()

    # Residual angular spread (how spread out are residual directions?)
    residuals = []
    for f in features_list:
        r = f - f.mean(axis=(0, 1), keepdims=True)
        residuals.append(r.mean(axis=(0, 1)))
    residuals = np.array(residuals)
    # Pairwise cosine similarities
    norms_r = np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-10
    normed = residuals / norms_r
    cos_mat = normed @ normed.T
    triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
    stats['residual_mean_cos'] = triu.mean()
    stats['residual_angular_spread'] = np.degrees(np.arccos(np.clip(triu.mean(), -1, 1)))

    # Gamma statistics
    si, bi = block_key
    gamma = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.gamma').numpy()
    stats['gamma_mean'] = np.mean(np.abs(gamma))
    stats['gamma_max'] = np.max(np.abs(gamma))
    stats['gamma_std'] = np.std(gamma)

    # W1 condition number (how well-conditioned is the MLP?)
    w1 = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.pwconv1.weight').numpy()
    try:
        svs = np.linalg.svd(w1, compute_uv=False)
        stats['w1_condition'] = svs[0] / (svs[-1] + 1e-10)
        stats['w1_rank_ratio'] = np.sum(svs > svs[0] * 0.01) / len(svs)
    except:
        stats['w1_condition'] = 0
        stats['w1_rank_ratio'] = 0

    return stats

print(f'{"Block":<8} {"Imp%":<8} {"γ mean":<8} {"γ max":<8} {"MLP/res":<8} '
      f'{"Ang spread":<11} {"W1 cond":<10} {"Cross var":<10}')
print('-' * 81)

all_stats = {}
for si in range(4):
    for bi in range(depths[si]):
        key = (si, bi)
        st = block_geometric_stats(key, all_block_features[key], all_block_mlp[key])
        all_stats[key] = st
        imp = BLOCK_IMPORTANCE.get(key, 0)
        print(f'  {si}.{bi:<5} {imp:+6.1f}  {st["gamma_mean"]:<7.2f} {st["gamma_max"]:<7.2f} '
              f'{st["mlp_ratio"]:<7.3f}  {st["residual_angular_spread"]:<10.1f} '
              f'{st["w1_condition"]:<9.1f} {st["cross_image_var"]:<9.4f}')


# ================================================================
# STEP 4: CORRELATION — Do geometric features predict importance?
# ================================================================
print()
print('=' * 70)
print('STEP 4: Correlation Between Geometric Features and Block Importance')
print('=' * 70)
print()

importances = []
geo_features = defaultdict(list)
for key in sorted(all_stats.keys()):
    importances.append(BLOCK_IMPORTANCE.get(key, 0))
    for feat_name, val in all_stats[key].items():
        geo_features[feat_name].append(val)

importances = np.array(importances)
print(f'{"Geometric Feature":<25} {"Correlation":<12} {"p-val proxy":<12}')
print('-' * 49)

correlations = {}
for feat_name, vals in sorted(geo_features.items()):
    vals = np.array(vals)
    if vals.std() < 1e-10: continue
    corr = np.corrcoef(importances, vals)[0, 1]
    # Simple significance proxy: |corr| * sqrt(N)
    sig = abs(corr) * np.sqrt(len(importances))
    correlations[feat_name] = corr
    marker = ' ***' if sig > 2.0 else ' *' if sig > 1.5 else ''
    print(f'  {feat_name:<23} {corr:+10.3f}   {sig:8.2f}{marker}')


# ================================================================
# STEP 5: NEURON-LEVEL IMPORTANCE IN CRITICAL BLOCKS
# ================================================================
print()
print('=' * 70)
print('STEP 5: Neuron Importance in Critical Blocks (using full pipeline)')
print('=' * 70)
print()

# For the top 3 critical blocks, find which neurons matter most
# by ablating one at a time (or small groups)

def eval_single_image_ablation(v16, img_tensor, gt_ab, ablate_neurons=None):
    """Quick evaluation with ablation."""
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
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{p}.norm.weight'),
                                 v16._get_weight(f'{p}.norm.bias'))
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))
                key = (si, bi)
                if ablate_neurons and key in ablate_neurons:
                    mask = torch.ones(post_gelu.shape[-1])
                    for n in ablate_neurons[key]:
                        mask[n] = 0
                    post_gelu = post_gelu * mask
                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0, 3, 1, 2)
                x = res + v16._get_weight(f'{p}.gamma').view(1, -1, 1, 1) * x

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

    pred_ab = final[0, :2].permute(1, 2, 0).numpy()
    pred_ab_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    return np.sqrt(np.mean((pred_ab_r - gt_ab)**2))


# Use 3 test images for speed
test_imgs = []
for idx in [300, 310, 320]:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_imgs.append((t, gt_ab))

# Baseline
base_rmse = np.mean([eval_single_image_ablation(v16, t, gt) for t, gt in test_imgs])
print(f'Baseline RMSE: {base_rmse:.3f}')

# For top critical blocks, ablate groups of neurons
for target_block, n_neurons_total in [((2, 8), 1536), ((1, 0), 768), ((3, 2), 3072)]:
    si, bi = target_block
    imp = BLOCK_IMPORTANCE.get(target_block, 0)
    print(f'\nBlock {si}.{bi} (importance: {imp:+.1f}%, {n_neurons_total} neurons):')

    # Systematic: ablate groups of 10% and find most impactful group
    group_size = max(1, n_neurons_total // 10)
    group_impacts = []
    for g in range(10):
        start = g * group_size
        end = min(start + group_size, n_neurons_total)
        neurons = set(range(start, end))
        rmse = np.mean([eval_single_image_ablation(v16, t, gt, {target_block: neurons})
                        for t, gt in test_imgs])
        delta_pct = (rmse - base_rmse) / base_rmse * 100
        group_impacts.append((g, delta_pct, start, end))
        print(f'  Neurons {start:4d}-{end:4d}: RMSE={rmse:.3f} ({delta_pct:+.1f}%)')

    # Find most and least impactful groups
    most = max(group_impacts, key=lambda x: x[1])
    least = min(group_impacts, key=lambda x: x[1])
    print(f'  Most impactful:  neurons {most[2]}-{most[3]} ({most[1]:+.1f}%)')
    print(f'  Least impactful: neurons {least[2]}-{least[3]} ({least[1]:+.1f}%)')


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()

# Top correlations
sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print('Top geometric predictors of block importance:')
for name, corr in sorted_corrs[:5]:
    print(f'  {name:<25} r = {corr:+.3f}')

# φ-lattice vs importance
phi_errs = [block_phi_scores.get(k, 99) for k in sorted(BLOCK_IMPORTANCE.keys())]
imps = [BLOCK_IMPORTANCE[k] for k in sorted(BLOCK_IMPORTANCE.keys())]
corr_phi = np.corrcoef(imps, phi_errs)[0, 1]
print(f'\nφ-lattice error vs importance: r = {corr_phi:+.3f}')
print(f'  (negative = tighter φ-lattice → more important)')
