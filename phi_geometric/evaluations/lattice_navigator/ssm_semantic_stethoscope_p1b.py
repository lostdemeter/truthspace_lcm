"""
Phase 1B: The Stethoscope — Natural Semantic Probing

Phase 1A showed: synthetic stimuli (solid colors, geometric patterns) barely
explore the encoder's feature space. The encoder predicts (0,0) for everything
synthetic because it needs NATURAL spatial structure to recognize concepts.

The safe is locked by OBJECT concepts (sky, grass, skin, wood), not geometry.
We need to probe with categorized natural patches where we KNOW the semantics.

Strategy:
  1. Take real images from the validation set
  2. Segment into regions by color (high a*, low a*, high b*, low b*, neutral)
  3. These color regions correspond to semantic categories:
     - High a* (red): skin, brick, clay, autumn leaves
     - Low a* (green): grass, trees, foliage
     - High b* (yellow): sunlight, sand, warm scenes
     - Low b* (blue): sky, water, shadows
     - Neutral: roads, concrete, metal, clouds
  4. Collect features from these categorized positions
  5. Map which neurons fire for which semantic categories
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict
from numpy.linalg import lstsq

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_full(v16, img_tensor, stage_idx=1):
    """Run encoder and record activations at the specified stage."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    activations = {}; features = []

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
                pre_gelu = F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                    v16._get_weight(f'{p}.pwconv1.bias'))

                if si == stage_idx and bi == depths[si] - 1:
                    activations['pre_gelu'] = pre_gelu.squeeze(0).detach().numpy()
                    post_gelu = gate(pre_gelu)
                    activations['post_gelu'] = post_gelu.squeeze(0).detach().numpy()
                else:
                    post_gelu = gate(pre_gelu)

                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))
            if si == stage_idx:
                activations['features'] = xn.squeeze(0).detach().numpy()

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
        activations['color_pred'] = out3.squeeze(0).detach().numpy()

    return activations


# ================================================================
# STEP 1: COLLECT SEMANTICALLY-LABELED NATURAL FEATURES
# ================================================================
print('=' * 70)
print('STEP 1: NATURAL SEMANTIC PROBING — Turning the Right Dial')
print('=' * 70)
print()

# Color-based semantic categories (what color region tells us about content)
# Using Lab color space: a* axis = green(-) to red(+), b* axis = blue(-) to yellow(+)
SEMANTIC_CATEGORIES = {
    'warm_red':    {'a_min': 15, 'a_max': 999, 'b_min': -999, 'b_max': 999, 'sat_min': 15},
    'cool_green':  {'a_min': -999, 'a_max': -15, 'b_min': -999, 'b_max': 999, 'sat_min': 15},
    'warm_yellow': {'a_min': -999, 'a_max': 999, 'b_min': 15, 'b_max': 999, 'sat_min': 15},
    'cool_blue':   {'a_min': -999, 'a_max': 999, 'b_min': -999, 'b_max': -15, 'sat_min': 15},
    'neutral':     {'a_min': -8, 'a_max': 8, 'b_min': -8, 'b_max': 8, 'sat_min': 0},
}

# Collect features per category across many images
cat_features = defaultdict(list)    # category → [feature_vectors]
cat_activations = defaultdict(list) # category → [post_gelu_vectors]
cat_colors = defaultdict(list)      # category → [ab_values]
cat_positions = defaultdict(list)   # category → [(img_idx, y, x)]

n_images = 0
for img_idx in range(50, 200):
    if n_images >= 40: break
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)

    if sat.mean() < 3: continue

    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    acts = run_encoder_full(v16, t, stage_idx=1)
    feats = acts['features']  # [H, W, 192]
    h, w = feats.shape[:2]
    post_gelu = acts['post_gelu']  # [H, W, 768]

    # Resize color to feature map size
    ab_small = cv2.resize(ab, (w, h))
    sat_small = np.sqrt(ab_small[:,:,0]**2 + ab_small[:,:,1]**2)

    # Classify each position
    for y in range(h):
        for x in range(w):
            a_val = ab_small[y, x, 0]
            b_val = ab_small[y, x, 1]
            s_val = sat_small[y, x]

            for cat_name, bounds in SEMANTIC_CATEGORIES.items():
                if (bounds['a_min'] <= a_val <= bounds['a_max'] and
                    bounds['b_min'] <= b_val <= bounds['b_max'] and
                    s_val >= bounds['sat_min']):
                    cat_features[cat_name].append(feats[y, x])
                    cat_activations[cat_name].append(post_gelu[y, x])
                    cat_colors[cat_name].append([a_val, b_val])
                    cat_positions[cat_name].append((img_idx, y, x))
                    break  # first match wins (categories overlap — neutral is last resort)

    n_images += 1

print(f'Processed {n_images} natural images')
print(f'\nSemantic category counts:')
for cat in SEMANTIC_CATEGORIES:
    n = len(cat_features[cat])
    if n > 0:
        cols = np.array(cat_colors[cat])
        print(f'  {cat:<15} {n:5d} positions  '
              f'mean_a={cols[:,0].mean():+5.1f}  mean_b={cols[:,1].mean():+5.1f}')


# ================================================================
# STEP 2: FEATURE DIRECTION ANALYSIS BY SEMANTIC CATEGORY
# ================================================================
print()
print('=' * 70)
print('STEP 2: FEATURE DIRECTIONS BY SEMANTIC CATEGORY')
print('=' * 70)
print()

# Mean feature direction per category
cat_mean_dirs = {}
cat_mean_feats = {}
for cat in SEMANTIC_CATEGORIES:
    if len(cat_features[cat]) < 10: continue
    feats_arr = np.array(cat_features[cat])
    mean_feat = feats_arr.mean(0)
    norm = np.linalg.norm(mean_feat)
    cat_mean_dirs[cat] = mean_feat / (norm + 1e-8)
    cat_mean_feats[cat] = mean_feat

# Angular separation between categories
cats = sorted(cat_mean_dirs.keys())
print('Angular separation between semantic categories (degrees):')
print(f'  {"":>15}', end='')
for c in cats: print(f'{c:>15}', end='')
print()

for ci in cats:
    print(f'  {ci:>15}', end='')
    for cj in cats:
        if ci == cj:
            print(f'{"---":>15}', end='')
        else:
            cos = np.dot(cat_mean_dirs[ci], cat_mean_dirs[cj])
            cos = np.clip(cos, -1, 1)
            angle = np.degrees(np.arccos(cos))
            print(f'{angle:>14.1f}°', end='')
    print()

print(f'\n  π/φ² = {np.degrees(np.pi / PHI**2):.1f}°')

# Within-category angular spread
print(f'\nWithin-category angular spread:')
for cat in cats:
    if len(cat_features[cat]) < 20: continue
    feats_arr = np.array(cat_features[cat])
    # Sample 200 random pairs
    n = min(len(feats_arr), 500)
    sample = feats_arr[np.random.choice(len(feats_arr), n, replace=False)]
    norms = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8
    unit = sample / norms
    # Angle to category mean
    cos_to_mean = unit @ cat_mean_dirs[cat]
    cos_to_mean = np.clip(cos_to_mean, -1, 1)
    angles_to_mean = np.degrees(np.arccos(cos_to_mean))
    print(f'  {cat:<15} mean_angle_to_centroid={np.mean(angles_to_mean):.1f}°  '
          f'std={np.std(angles_to_mean):.1f}°')


# ================================================================
# STEP 3: NEURON FIRING BY SEMANTIC CATEGORY
# ================================================================
print()
print('=' * 70)
print('STEP 3: NEURON FIRING BY SEMANTIC CATEGORY')
print('=' * 70)
print()

# Mean fire rate per category
n_neurons = 768
cat_fire_rates = {}
for cat in cats:
    if len(cat_activations[cat]) < 10: continue
    acts_arr = np.array(cat_activations[cat])  # [N_positions, 768]
    fire_rates = (acts_arr > 0).mean(0)  # [768]
    cat_fire_rates[cat] = fire_rates

# For each neuron, compute category selectivity
print('Neurons most selective for each category:')
print('  (Neuron, fire_rate_in_category, fire_rate_overall, selectivity_ratio)')
print()

# Overall fire rate (across all categories)
all_acts = []
for cat in cats:
    if cat in cat_fire_rates:
        all_acts.append(np.array(cat_activations[cat]))
all_acts = np.vstack(all_acts)
overall_fire_rate = (all_acts > 0).mean(0)

for cat in cats:
    if cat not in cat_fire_rates: continue
    cat_rate = cat_fire_rates[cat]
    # Selectivity: ratio of category rate to overall rate
    selectivity = cat_rate / (overall_fire_rate + 1e-6)
    # Also compute: neurons that fire MORE for this category
    diff = cat_rate - overall_fire_rate

    # Top neurons by differential fire rate
    top_idx = np.argsort(diff)[::-1][:10]
    print(f'  {cat}:')
    for ni in top_idx[:5]:
        print(f'    neuron {ni:3d}: cat_rate={cat_rate[ni]:.2f}  overall={overall_fire_rate[ni]:.2f}  '
              f'selectivity={selectivity[ni]:.1f}x  diff={diff[ni]:+.2f}')
    print()

# Count neurons selective for each category (>2x selectivity, >20% fire rate in category)
print('Summary — neurons selective per category (>2x rate, >20% fire rate):')
total_selective = set()
for cat in cats:
    if cat not in cat_fire_rates: continue
    cat_rate = cat_fire_rates[cat]
    selectivity = cat_rate / (overall_fire_rate + 1e-6)
    selective = set(np.where((selectivity > 2.0) & (cat_rate > 0.2))[0])
    total_selective |= selective
    print(f'  {cat:<15} {len(selective):3d} selective neurons')

print(f'  {"TOTAL (unique)":<15} {len(total_selective):3d}/{n_neurons}')


# ================================================================
# STEP 4: THE SEMANTIC DIRECTIONS — PCA per Category
# ================================================================
print()
print('=' * 70)
print('STEP 4: SEMANTIC DIRECTIONS — What directions encode each concept?')
print('=' * 70)
print()

# For each category, find the principal feature directions
# Then check: are category-specific directions DIFFERENT from overall PCA?

# Overall PCA
all_feats = np.vstack([np.array(cat_features[c])[:500] for c in cats if len(cat_features[c]) >= 10])
overall_mean = all_feats.mean(0)
_, S_overall, Vt_overall = np.linalg.svd(all_feats - overall_mean, full_matrices=False)
print(f'Overall PCA: top 5 SVs = {S_overall[:5]}')

for cat in cats:
    if len(cat_features[cat]) < 50: continue
    feats_arr = np.array(cat_features[cat])[:1000]
    _, S_cat, Vt_cat = np.linalg.svd(feats_arr - feats_arr.mean(0), full_matrices=False)

    # How well do overall PCA directions explain this category?
    cat_centered = feats_arr - feats_arr.mean(0)
    overall_proj = cat_centered @ Vt_overall[:10].T
    overall_var = np.sum(overall_proj**2) / np.sum(cat_centered**2)

    # How well do category-specific directions explain?
    cat_proj = cat_centered @ Vt_cat[:10].T
    cat_var = np.sum(cat_proj**2) / np.sum(cat_centered**2)

    # Alignment: do category directions align with overall directions?
    alignment = np.abs(Vt_cat[:3] @ Vt_overall[:3].T)  # [3, 3]

    print(f'  {cat}:')
    print(f'    Overall PCA (10D): {overall_var:.1%} var | Category PCA (10D): {cat_var:.1%} var')
    print(f'    Top direction alignment with overall: {alignment[0].max():.3f}, {alignment[1].max():.3f}, {alignment[2].max():.3f}')


# ================================================================
# STEP 5: COLOR PREDICTION FROM SEMANTIC FEATURES
# ================================================================
print()
print('=' * 70)
print('STEP 5: COLOR DECODING — Can we predict color from category membership?')
print('=' * 70)
print()

# Build a simple predictor: category label → predicted color
# vs: feature vector → predicted color (linear regression)
# The gap between these shows how much is in the DIRECTION beyond the category

# Prepare data
all_feat_list = []
all_color_list = []
all_cat_labels = []
cat_to_idx = {c: i for i, c in enumerate(cats)}

for cat in cats:
    n = min(len(cat_features[cat]), 500)
    if n < 10: continue
    indices = np.random.choice(len(cat_features[cat]), n, replace=False)
    for idx in indices:
        all_feat_list.append(cat_features[cat][idx])
        all_color_list.append(cat_colors[cat][idx])
        all_cat_labels.append(cat_to_idx[cat])

all_feat_arr = np.array(all_feat_list)
all_color_arr = np.array(all_color_list)
all_cat_arr = np.array(all_cat_labels)

# Method 1: Category mean prediction (just knowing WHICH category)
cat_mean_colors = {}
for cat in cats:
    if len(cat_colors[cat]) >= 10:
        cat_mean_colors[cat] = np.mean(cat_colors[cat], axis=0)

cat_pred = np.array([cat_mean_colors[cats[l]] for l in all_cat_arr])
cat_rmse = np.sqrt(np.mean((cat_pred - all_color_arr)**2))

# Method 2: Linear regression on features
X = np.column_stack([all_feat_arr, np.ones(len(all_feat_arr))])
Wa, *_ = lstsq(X, all_color_arr[:, 0], rcond=None)
Wb, *_ = lstsq(X, all_color_arr[:, 1], rcond=None)
feat_pred_a = X @ Wa
feat_pred_b = X @ Wb
feat_rmse = np.sqrt(np.mean((feat_pred_a - all_color_arr[:, 0])**2 +
                            (feat_pred_b - all_color_arr[:, 1])**2))

# Method 3: Linear regression on post-GELU activations
all_act_list = []
for cat in cats:
    n = min(len(cat_activations[cat]), 500)
    if n < 10: continue
    indices = np.random.choice(len(cat_activations[cat]), n, replace=False)
    for idx in indices:
        all_act_list.append(cat_activations[cat][idx])
all_act_arr = np.array(all_act_list)[:len(all_color_arr)]

X_act = np.column_stack([all_act_arr, np.ones(len(all_act_arr))])
Wa_act, *_ = lstsq(X_act, all_color_arr[:, 0], rcond=None)
Wb_act, *_ = lstsq(X_act, all_color_arr[:, 1], rcond=None)
act_pred_a = X_act @ Wa_act
act_pred_b = X_act @ Wb_act
act_rmse = np.sqrt(np.mean((act_pred_a - all_color_arr[:, 0])**2 +
                            (act_pred_b - all_color_arr[:, 1])**2))

# Method 4: One-hot category + features
cat_onehot = np.zeros((len(all_cat_arr), len(cats)))
for i, l in enumerate(all_cat_arr): cat_onehot[i, l] = 1
X_combo = np.column_stack([all_feat_arr, cat_onehot, np.ones(len(all_feat_arr))])
Wa_c, *_ = lstsq(X_combo, all_color_arr[:, 0], rcond=None)
Wb_c, *_ = lstsq(X_combo, all_color_arr[:, 1], rcond=None)
combo_pred_a = X_combo @ Wa_c
combo_pred_b = X_combo @ Wb_c
combo_rmse = np.sqrt(np.mean((combo_pred_a - all_color_arr[:, 0])**2 +
                              (combo_pred_b - all_color_arr[:, 1])**2))

print('Color prediction RMSE from different representations:')
print(f'  Category label only:           RMSE = {cat_rmse:.2f}')
print(f'  Stage 1 features (192D):       RMSE = {feat_rmse:.2f}')
print(f'  Post-GELU activations (768D):  RMSE = {act_rmse:.2f}')
print(f'  Features + category label:     RMSE = {combo_rmse:.2f}')
print()
print(f'  Feature improvement over category: {(1 - feat_rmse/cat_rmse)*100:.1f}%')
print(f'  → Features contain {cat_rmse - feat_rmse:.1f} more info than just the category label')


# ================================================================
# STEP 6: THE SEMANTIC MAP — Activation Fingerprints
# ================================================================
print()
print('=' * 70)
print('STEP 6: ACTIVATION FINGERPRINTS — The Lock\'s Tumbler Pattern')
print('=' * 70)
print()

# For each category, what is the characteristic activation fingerprint?
# This is the "combination" that opens the lock for that concept

print('Top 10 most distinguishing neurons per category:')
print('(neurons that fire significantly more for this category than others)')
print()

fingerprints = {}
for cat in cats:
    if cat not in cat_fire_rates: continue
    cat_rate = cat_fire_rates[cat]
    diff = cat_rate - overall_fire_rate

    # Top neurons by differential
    top_neurons = np.argsort(diff)[::-1][:10]
    fingerprints[cat] = top_neurons

    print(f'  {cat} fingerprint:')
    print(f'    UP:   ', end='')
    for ni in top_neurons[:5]:
        print(f'n{ni}({diff[ni]:+.2f}) ', end='')
    print()
    # Also show neurons that fire LESS for this category
    bot_neurons = np.argsort(diff)[:10]
    print(f'    DOWN: ', end='')
    for ni in bot_neurons[:5]:
        print(f'n{ni}({diff[ni]:+.2f}) ', end='')
    print()

# Do fingerprints overlap? (shared neurons between categories)
print(f'\nFingerprint overlap (shared neurons in top-10):')
for ci in cats:
    for cj in cats:
        if cj <= ci: continue
        if ci not in fingerprints or cj not in fingerprints: continue
        shared = len(set(fingerprints[ci]) & set(fingerprints[cj]))
        if shared > 0:
            print(f'  {ci} ∩ {cj}: {shared} shared neurons')

# Unique neurons per category (not in any other category's top-10)
all_fp_neurons = set()
for cat in cats:
    if cat in fingerprints:
        all_fp_neurons |= set(fingerprints[cat])
print(f'\n  Total unique fingerprint neurons: {len(all_fp_neurons)}/{n_neurons}')


# ================================================================
# STEP 7: DIRECTION ANGLES — Is π/φ² the concept separator?
# ================================================================
print()
print('=' * 70)
print('STEP 7: ANGULAR STRUCTURE OF SEMANTIC SPACE')
print('=' * 70)
print()

# Sample positions from each category, compute pairwise angles
phi2_angle = np.pi / PHI**2

within_angles = []
between_angles = []
n_pairs_target = 2000

for trial in range(n_pairs_target):
    if trial < n_pairs_target // 2:
        # Within-category pair
        cat = cats[np.random.randint(len(cats))]
        if len(cat_features[cat]) < 2: continue
        i, j = np.random.choice(len(cat_features[cat]), 2, replace=False)
        fi = cat_features[cat][i]
        fj = cat_features[cat][j]
    else:
        # Between-category pair
        ci, cj_idx = np.random.choice(len(cats), 2, replace=False)
        cat_i, cat_j = cats[ci], cats[cj_idx]
        if len(cat_features[cat_i]) < 1 or len(cat_features[cat_j]) < 1: continue
        fi = cat_features[cat_i][np.random.randint(len(cat_features[cat_i]))]
        fj = cat_features[cat_j][np.random.randint(len(cat_features[cat_j]))]

    ni = np.linalg.norm(fi)
    nj = np.linalg.norm(fj)
    if ni < 1e-8 or nj < 1e-8: continue
    cos = np.dot(fi, fj) / (ni * nj)
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)

    if trial < n_pairs_target // 2:
        within_angles.append(angle)
    else:
        between_angles.append(angle)

within_angles = np.array(within_angles)
between_angles = np.array(between_angles)

print(f'Pairwise angles between feature vectors:')
print(f'  Within-category:  mean={np.degrees(np.mean(within_angles)):.1f}°  '
      f'std={np.degrees(np.std(within_angles)):.1f}°')
print(f'  Between-category: mean={np.degrees(np.mean(between_angles)):.1f}°  '
      f'std={np.degrees(np.std(between_angles)):.1f}°')
print(f'  π/φ² = {np.degrees(phi2_angle):.1f}°')
print(f'  Separation ratio: {np.mean(between_angles)/np.mean(within_angles):.2f}x')

# Fraction above/below π/φ²
within_above = (within_angles > phi2_angle).mean()
between_above = (between_angles > phi2_angle).mean()
print(f'\n  Above π/φ² ({np.degrees(phi2_angle):.1f}°):')
print(f'    Within-category:  {within_above:.1%}')
print(f'    Between-category: {between_above:.1%}')

# Angle histogram
print(f'\n  Angle distribution (10° bins):')
bins = np.arange(0, 100, 5)
for lo in bins:
    hi = lo + 5
    lo_r, hi_r = np.radians(lo), np.radians(hi)
    w_count = ((within_angles >= lo_r) & (within_angles < hi_r)).sum()
    b_count = ((between_angles >= lo_r) & (between_angles < hi_r)).sum()
    w_bar = '#' * (w_count * 30 // max(1, len(within_angles) // 10))
    b_bar = '=' * (b_count * 30 // max(1, len(between_angles) // 10))
    marker = ' ← π/φ²' if lo <= np.degrees(phi2_angle) < hi else ''
    print(f'    {lo:3.0f}°-{hi:3.0f}°: W={w_count:4d} {w_bar}')
    print(f'    {"":>9} B={b_count:4d} {b_bar}{marker}')


print()
print('=' * 70)
print('SUMMARY — Phase 1B')
print('=' * 70)
print()
print('The safe-cracking stethoscope is now calibrated to natural semantic space.')
print(f'  {len(cats)} semantic categories identified')
print(f'  {sum(len(v) for v in cat_features.values())} labeled feature positions collected')
print(f'  Neuron selectivity mapped per category')
print(f'  Feature directions mapped per category')
print(f'  Angular structure analyzed')
print()
print('Ready for Phase 2 (direction catalog) and Phase 3 (W₁ row decoding)')
