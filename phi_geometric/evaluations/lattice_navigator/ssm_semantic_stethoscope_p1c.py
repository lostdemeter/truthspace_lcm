"""
Phase 1C: Hearing the Semantic Whisper — Context-Controlled Probing

Phase 1B showed: semantic content is a ~5% perturbation on a 95% structural signal.
Cross-image comparison drowns the whisper in structural noise.

The fix: compare WITHIN images, between positions with SAME structure but
DIFFERENT semantic content. This is like Doc 189's safe dial: you need the
right structural context (plates) to hear the semantic click.

Strategy:
  1. For each image, identify the structural background (mean feature)
  2. Compute residuals: feature - structural_background
  3. These residuals ARE the semantic whisper
  4. Correlate residuals with color across positions WITHIN each image
  5. Then: correlate residuals with color ACROSS images (after removing structure)
  6. Build the semantic vocabulary from these cleaned residuals
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
    """Run encoder, return features and post-GELU at target stage."""
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
    return activations


# ================================================================
# STEP 1: EXTRACT STRUCTURAL AND SEMANTIC COMPONENTS
# ================================================================
print('=' * 70)
print('STEP 1: SEPARATING STRUCTURE FROM SEMANTICS')
print('=' * 70)
print()

# For each image:
#   structural = image-wide mean feature (same for all positions)
#   residual = feature - structural = the semantic whisper
#   Test: does residual predict color better than raw feature ACROSS images?

per_image_data = []
all_residuals = []
all_raw_feats = []
all_colors = []
all_gelu_residuals = []
all_gelu_raw = []
image_ids = []

n_images = 0
for img_idx in range(50, 250):
    if n_images >= 40: break
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue

    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    acts = run_encoder_full(v16, t, stage_idx=1)
    feats = acts['features']  # [H, W, 192]
    post_gelu = acts['post_gelu']  # [H, W, 768]
    h, w = feats.shape[:2]

    flat_feats = feats.reshape(-1, 192)
    flat_gelu = post_gelu.reshape(-1, 768)
    ab_small = cv2.resize(ab, (w, h)).reshape(-1, 2)

    # Structural component = mean across all positions in this image
    struct_feat = flat_feats.mean(0, keepdims=True)
    struct_gelu = flat_gelu.mean(0, keepdims=True)

    # Semantic residual = feature - structural background
    residual = flat_feats - struct_feat
    gelu_residual = flat_gelu - struct_gelu

    per_image_data.append({
        'features': flat_feats,
        'residuals': residual,
        'gelu': flat_gelu,
        'gelu_residuals': gelu_residual,
        'colors': ab_small,
        'struct': struct_feat[0],
    })

    # Subsample for cross-image analysis
    n_sub = min(200, len(flat_feats))
    idx = np.random.choice(len(flat_feats), n_sub, replace=False)
    all_residuals.append(residual[idx])
    all_raw_feats.append(flat_feats[idx])
    all_colors.append(ab_small[idx])
    all_gelu_residuals.append(gelu_residual[idx])
    all_gelu_raw.append(flat_gelu[idx])
    image_ids.extend([n_images] * n_sub)

    n_images += 1

all_residuals = np.vstack(all_residuals)
all_raw_feats = np.vstack(all_raw_feats)
all_colors = np.vstack(all_colors)
all_gelu_residuals = np.vstack(all_gelu_residuals)
all_gelu_raw = np.vstack(all_gelu_raw)
image_ids = np.array(image_ids)

print(f'Collected {n_images} images, {len(all_raw_feats)} total positions')
print(f'Feature dim: {all_raw_feats.shape[1]}, GELU dim: {all_gelu_raw.shape[1]}')


# ================================================================
# STEP 2: WITHIN-IMAGE COLOR PREDICTION — Structure is constant
# ================================================================
print()
print('=' * 70)
print('STEP 2: WITHIN-IMAGE COLOR PREDICTION (Structure Controlled)')
print('=' * 70)
print()

within_raw_rmses = []
within_res_rmses = []
within_gelu_rmses = []
within_gelu_res_rmses = []

for i, data in enumerate(per_image_data):
    f = data['features']; r = data['residuals']
    g = data['gelu']; gr = data['gelu_residuals']
    c = data['colors']

    # Raw features → color
    X = np.column_stack([f, np.ones(len(f))])
    Wa, *_ = lstsq(X, c[:, 0], rcond=None); Wb, *_ = lstsq(X, c[:, 1], rcond=None)
    raw_rmse = np.sqrt(np.mean((X@Wa - c[:,0])**2 + (X@Wb - c[:,1])**2))

    # Residual → color
    X = np.column_stack([r, np.ones(len(r))])
    Wa, *_ = lstsq(X, c[:, 0], rcond=None); Wb, *_ = lstsq(X, c[:, 1], rcond=None)
    res_rmse = np.sqrt(np.mean((X@Wa - c[:,0])**2 + (X@Wb - c[:,1])**2))

    # GELU → color
    X = np.column_stack([g, np.ones(len(g))])
    Wa, *_ = lstsq(X, c[:, 0], rcond=None); Wb, *_ = lstsq(X, c[:, 1], rcond=None)
    gelu_rmse = np.sqrt(np.mean((X@Wa - c[:,0])**2 + (X@Wb - c[:,1])**2))

    # GELU residual → color
    X = np.column_stack([gr, np.ones(len(gr))])
    Wa, *_ = lstsq(X, c[:, 0], rcond=None); Wb, *_ = lstsq(X, c[:, 1], rcond=None)
    gelu_res_rmse = np.sqrt(np.mean((X@Wa - c[:,0])**2 + (X@Wb - c[:,1])**2))

    within_raw_rmses.append(raw_rmse)
    within_res_rmses.append(res_rmse)
    within_gelu_rmses.append(gelu_rmse)
    within_gelu_res_rmses.append(gelu_res_rmse)

print(f'Within-image color prediction (mean over {n_images} images):')
print(f'  Raw features (192D):       RMSE = {np.mean(within_raw_rmses):.2f} ± {np.std(within_raw_rmses):.2f}')
print(f'  Residual features (192D):  RMSE = {np.mean(within_res_rmses):.2f} ± {np.std(within_res_rmses):.2f}')
print(f'  Raw GELU (768D):           RMSE = {np.mean(within_gelu_rmses):.2f} ± {np.std(within_gelu_rmses):.2f}')
print(f'  Residual GELU (768D):      RMSE = {np.mean(within_gelu_res_rmses):.2f} ± {np.std(within_gelu_res_rmses):.2f}')
print()
res_improvement = (1 - np.mean(within_res_rmses)/np.mean(within_raw_rmses)) * 100
print(f'  Residual vs raw improvement: {res_improvement:+.1f}%')
print(f'  → Removing structure {"helps" if res_improvement > 0 else "hurts"} within-image prediction')


# ================================================================
# STEP 3: CROSS-IMAGE COLOR PREDICTION — The Key Test
# ================================================================
print()
print('=' * 70)
print('STEP 3: CROSS-IMAGE COLOR PREDICTION — Does Residual Transfer?')
print('=' * 70)
print()

# Split by image: train on first 30, test on last 10
train_mask = image_ids < 30
test_mask = image_ids >= 30

print(f'Train: {train_mask.sum()} positions, Test: {test_mask.sum()} positions')

def cross_image_test(features, colors, train_mask, test_mask, name):
    """Train linear regression on train images, evaluate on test images."""
    X_train = np.column_stack([features[train_mask], np.ones(train_mask.sum())])
    X_test = np.column_stack([features[test_mask], np.ones(test_mask.sum())])

    Wa, *_ = lstsq(X_train, colors[train_mask, 0], rcond=None)
    Wb, *_ = lstsq(X_train, colors[train_mask, 1], rcond=None)

    pred_a = X_test @ Wa
    pred_b = X_test @ Wb
    rmse = np.sqrt(np.mean((pred_a - colors[test_mask, 0])**2 +
                           (pred_b - colors[test_mask, 1])**2))

    r_a = np.corrcoef(pred_a, colors[test_mask, 0])[0, 1]
    r_b = np.corrcoef(pred_b, colors[test_mask, 1])[0, 1]

    print(f'  {name:<35} RMSE={rmse:.2f}  r_a={r_a:.3f}  r_b={r_b:.3f}')
    return rmse

print('Cross-image color prediction (train→test):')
raw_rmse = cross_image_test(all_raw_feats, all_colors, train_mask, test_mask, 'Raw features (192D)')
res_rmse = cross_image_test(all_residuals, all_colors, train_mask, test_mask, 'Residual features (192D)')
gelu_rmse = cross_image_test(all_gelu_raw, all_colors, train_mask, test_mask, 'Raw GELU (768D)')
gelu_res_rmse = cross_image_test(all_gelu_residuals, all_colors, train_mask, test_mask, 'Residual GELU (768D)')

# Also test: mean-only prediction (just predict dataset mean color)
mean_a = all_colors[train_mask, 0].mean()
mean_b = all_colors[train_mask, 1].mean()
mean_rmse = np.sqrt(np.mean((mean_a - all_colors[test_mask, 0])**2 +
                            (mean_b - all_colors[test_mask, 1])**2))
print(f'  {"Predict dataset mean":35} RMSE={mean_rmse:.2f}')

print()
print(f'  Raw vs residual improvement: {(1 - res_rmse/raw_rmse)*100:+.1f}%')
print(f'  GELU raw vs GELU residual:   {(1 - gelu_res_rmse/gelu_rmse)*100:+.1f}%')


# ================================================================
# STEP 4: THE SEMANTIC WHISPER — PCA of Residuals
# ================================================================
print()
print('=' * 70)
print('STEP 4: PCA OF RESIDUALS — What Does the Whisper Look Like?')
print('=' * 70)
print()

# PCA of raw features vs residuals
_, S_raw, Vt_raw = np.linalg.svd(all_raw_feats[:3000] - all_raw_feats[:3000].mean(0), full_matrices=False)
_, S_res, Vt_res = np.linalg.svd(all_residuals[:3000] - all_residuals[:3000].mean(0), full_matrices=False)

# How many dimensions needed for 90% variance?
raw_cumvar = np.cumsum(S_raw**2) / np.sum(S_raw**2)
res_cumvar = np.cumsum(S_res**2) / np.sum(S_res**2)

raw_90 = np.searchsorted(raw_cumvar, 0.9) + 1
res_90 = np.searchsorted(res_cumvar, 0.9) + 1

print(f'Dimensionality:')
print(f'  Raw features: {raw_90} dims for 90% var (effective rank)')
print(f'  Residuals:    {res_90} dims for 90% var (effective rank)')

# SV spectrum comparison
print(f'\n  Top 10 SVs:')
print(f'    {"Raw":<12} {"Residual":<12} {"Ratio":<8}')
for i in range(10):
    print(f'    {S_raw[i]:<12.1f} {S_res[i]:<12.1f} {S_raw[i]/S_res[i]:.2f}')

# Do residual PCA directions correlate with color?
res_proj = (all_residuals[:3000] - all_residuals[:3000].mean(0)) @ Vt_res[:20].T
print(f'\n  Residual PCA direction ↔ color correlation:')
print(f'    {"Dir":<5} {"r(a*)":<8} {"r(b*)":<8} {"r(sat)":<8}')
sat = np.sqrt(all_colors[:3000, 0]**2 + all_colors[:3000, 1]**2)
for d in range(10):
    ra = np.corrcoef(res_proj[:, d], all_colors[:3000, 0])[0, 1]
    rb = np.corrcoef(res_proj[:, d], all_colors[:3000, 1])[0, 1]
    rs = np.corrcoef(res_proj[:, d], sat)[0, 1]
    marker = ' ← semantic' if abs(ra) > 0.1 or abs(rb) > 0.1 else ''
    print(f'    PC{d:<3} {ra:+.3f}   {rb:+.3f}   {rs:+.3f}{marker}')

# Compare with raw PCA
raw_proj = (all_raw_feats[:3000] - all_raw_feats[:3000].mean(0)) @ Vt_raw[:20].T
print(f'\n  Raw PCA direction ↔ color correlation:')
print(f'    {"Dir":<5} {"r(a*)":<8} {"r(b*)":<8} {"r(sat)":<8}')
for d in range(10):
    ra = np.corrcoef(raw_proj[:, d], all_colors[:3000, 0])[0, 1]
    rb = np.corrcoef(raw_proj[:, d], all_colors[:3000, 1])[0, 1]
    rs = np.corrcoef(raw_proj[:, d], sat)[0, 1]
    marker = ' ← semantic' if abs(ra) > 0.1 or abs(rb) > 0.1 else ''
    print(f'    PC{d:<3} {ra:+.3f}   {rb:+.3f}   {rs:+.3f}{marker}')


# ================================================================
# STEP 5: ANGULAR STRUCTURE OF RESIDUALS
# ================================================================
print()
print('=' * 70)
print('STEP 5: ANGULAR STRUCTURE OF SEMANTIC RESIDUALS')
print('=' * 70)
print()

# Categorize residual positions by color
phi2_angle = np.pi / PHI**2
color_cats = {}
for i in range(len(all_colors)):
    a, b = all_colors[i]
    s = np.sqrt(a**2 + b**2)
    if s < 8:
        cat = 'neutral'
    elif a > 10 and abs(b) < abs(a):
        cat = 'red'
    elif a < -10 and abs(b) < abs(a):
        cat = 'green'
    elif b > 10 and abs(a) < abs(b):
        cat = 'yellow'
    elif b < -10 and abs(a) < abs(b):
        cat = 'blue'
    else:
        cat = 'mixed'
    if cat not in color_cats: color_cats[cat] = []
    color_cats[cat].append(i)

# Mean residual direction per color category
cat_res_dirs = {}
for cat, indices in color_cats.items():
    if len(indices) < 20: continue
    res = all_residuals[indices]
    mean_res = res.mean(0)
    norm = np.linalg.norm(mean_res)
    if norm > 1e-8:
        cat_res_dirs[cat] = mean_res / norm

# Angular separation between color categories IN RESIDUAL SPACE
cats_sorted = sorted(cat_res_dirs.keys())
print('Angular separation of RESIDUAL directions by color category:')
print(f'  {"":>10}', end='')
for c in cats_sorted: print(f'{c:>10}', end='')
print()
for ci in cats_sorted:
    print(f'  {ci:>10}', end='')
    for cj in cats_sorted:
        if ci == cj:
            print(f'{"---":>10}', end='')
        else:
            cos = np.dot(cat_res_dirs[ci], cat_res_dirs[cj])
            cos = np.clip(cos, -1, 1)
            angle = np.degrees(np.arccos(cos))
            print(f'{angle:>9.1f}°', end='')
    print()

print(f'\n  π/φ² = {np.degrees(phi2_angle):.1f}°')

# Compare with raw feature direction separation
cat_raw_dirs = {}
for cat, indices in color_cats.items():
    if len(indices) < 20: continue
    raw = all_raw_feats[indices]
    mean_raw = raw.mean(0)
    norm = np.linalg.norm(mean_raw)
    if norm > 1e-8:
        cat_raw_dirs[cat] = mean_raw / norm

print(f'\nAngular separation of RAW feature directions (for comparison):')
print(f'  {"":>10}', end='')
for c in cats_sorted: print(f'{c:>10}', end='')
print()
for ci in cats_sorted:
    print(f'  {ci:>10}', end='')
    for cj in cats_sorted:
        if ci == cj:
            print(f'{"---":>10}', end='')
        elif ci in cat_raw_dirs and cj in cat_raw_dirs:
            cos = np.dot(cat_raw_dirs[ci], cat_raw_dirs[cj])
            cos = np.clip(cos, -1, 1)
            angle = np.degrees(np.arccos(cos))
            print(f'{angle:>9.1f}°', end='')
        else:
            print(f'{"N/A":>10}', end='')
    print()


# ================================================================
# STEP 6: THE COMBINATION — Residual Activation Fingerprints
# ================================================================
print()
print('=' * 70)
print('STEP 6: RESIDUAL ACTIVATION FINGERPRINTS')
print('=' * 70)
print()

# Which GELU neurons have the most color-correlated residual activations?
# For each neuron: correlate its RESIDUAL activation with a* and b*
n_gelu = all_gelu_residuals.shape[1]  # 768

neuron_color_corr = np.zeros((n_gelu, 2))
for ni in range(n_gelu):
    neuron_res = all_gelu_residuals[:, ni]
    if neuron_res.std() < 1e-8: continue
    neuron_color_corr[ni, 0] = np.corrcoef(neuron_res, all_colors[:, 0])[0, 1]
    neuron_color_corr[ni, 1] = np.corrcoef(neuron_res, all_colors[:, 1])[0, 1]

# Top neurons correlated with a* (red-green)
top_a = np.argsort(np.abs(neuron_color_corr[:, 0]))[::-1]
# Top neurons correlated with b* (blue-yellow)
top_b = np.argsort(np.abs(neuron_color_corr[:, 1]))[::-1]

print('Top 15 neurons by RESIDUAL correlation with color:')
print(f'  {"Neuron":>8} {"r(a*)":>8} {"r(b*)":>8} {"r(max)":>8}  axis')
print(f'  {"-"*45}')

# Combined ranking
max_corr = np.max(np.abs(neuron_color_corr), axis=1)
top_combined = np.argsort(max_corr)[::-1]

semantic_neurons = []
for rank, ni in enumerate(top_combined[:30]):
    ra = neuron_color_corr[ni, 0]
    rb = neuron_color_corr[ni, 1]
    rmax = max(abs(ra), abs(rb))
    if abs(ra) > abs(rb):
        axis = 'a* (red-green)'
    else:
        axis = 'b* (blue-yellow)'
    marker = ' ← SEMANTIC' if rmax > 0.1 else ''
    print(f'  n{ni:>5} {ra:+.3f}   {rb:+.3f}   {rmax:.3f}  {axis}{marker}')
    if rmax > 0.05:
        semantic_neurons.append(ni)

print(f'\n  Neurons with |r| > 0.1: {(max_corr > 0.1).sum()}/768')
print(f'  Neurons with |r| > 0.05: {(max_corr > 0.05).sum()}/768')
print(f'  Neurons with |r| > 0.02: {(max_corr > 0.02).sum()}/768')


# ================================================================
# STEP 7: RESIDUAL-BASED COLOR PREDICTION
# ================================================================
print()
print('=' * 70)
print('STEP 7: RESIDUAL-BASED COLOR PREDICTION — Closing the Gap?')
print('=' * 70)
print()

# Use ONLY the semantic neurons (those with residual-color correlation)
# Compare: all features vs semantic-only features vs residual-only

# Top-k semantic neurons by residual correlation
for k in [10, 30, 50, 100, 200]:
    top_k = top_combined[:k]
    sel_gelu_res = all_gelu_residuals[:, top_k]

    rmse = cross_image_test(sel_gelu_res, all_colors, train_mask, test_mask,
                           f'Top-{k} semantic GELU residuals')

print()
print('Comparison with full representations:')
cross_image_test(all_raw_feats, all_colors, train_mask, test_mask, 'Full raw features (192D)')
cross_image_test(all_residuals, all_colors, train_mask, test_mask, 'Full residuals (192D)')
cross_image_test(all_gelu_raw, all_colors, train_mask, test_mask, 'Full raw GELU (768D)')
cross_image_test(all_gelu_residuals, all_colors, train_mask, test_mask, 'Full GELU residuals (768D)')


print()
print('=' * 70)
print('SUMMARY — Phase 1C: The Semantic Whisper')
print('=' * 70)
print()
print('Structure = image-wide mean feature (constant per image)')
print('Semantics = residual from structure (varies by position)')
print()
print('The safe-cracking insight: the "plates" are the structural context.')
print('To hear the "click", subtract the structure and listen to the residual.')
print('The residual encodes WHICH CONCEPT each position represents.')
