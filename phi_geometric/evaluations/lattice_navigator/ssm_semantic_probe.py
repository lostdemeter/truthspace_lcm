"""
SSM Semantic Space Probe: What Does the Encoder Actually "See"?

The encoder must know WHAT is in the image to colorize it correctly.
This means the feature space encodes semantic content — grass, sky, person, etc.

Questions:
1. What semantic concepts do different feature channels respond to?
2. Can we decode "tokens" from the feature space? (grass → green, sky → blue)
3. What's DIFFERENT between real and first-principles features?
4. Where does the 28.2% dir-W₂ gap live in semantic space?

Approach:
- Run diverse images through encoder
- Extract per-pixel features at each stage
- Cluster features and see what they correspond to spatially
- Compare: real encoder vs fitted-power-law-SVs encoder
- Use the COLOR output as a semantic label (grass→green IS the token)
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895; SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_with_stage_features(v16, img_tensor, muts=None):
    """Run encoder, return BOTH final output AND per-stage features."""
    if muts is None: muts = {}
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
    stage_feats = []
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
                x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'), v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                x = F.linear(x, muts.get(f'{p}.pwconv1.weight', v16._get_weight(f'{p}.pwconv1.weight')),
                             muts.get(f'{p}.pwconv1.bias', v16._get_weight(f'{p}.pwconv1.bias')))
                x = gate(x)
                x = F.linear(x, muts.get(f'{p}.pwconv2.weight', v16._get_weight(f'{p}.pwconv2.weight')),
                             muts.get(f'{p}.pwconv2.bias', v16._get_weight(f'{p}.pwconv2.bias')))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))
            stage_feats.append(xn.squeeze(0).detach().numpy())  # [H, W, C]
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy(), stage_feats


# ================================================================
# PART 1: SEMANTIC CONTENT ANALYSIS — What do the features encode?
# ================================================================
print('=' * 70)
print('PART 1: SEMANTIC CONTENT — Color as Token')
print('=' * 70)
print()
print('If grass → green, the encoder has produced a "grass" token.')
print('The COLOR OUTPUT is the semantic label. We can read it.')
print()

# Collect diverse images and their feature/color mappings
semantic_data = []

for idx in range(80, 130):
    if len(semantic_data) >= 20: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    # Skip low-saturation images
    sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2)
    if sat.mean() < 3: continue
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gbgr.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    
    enc_out, stage_feats = run_encoder_with_stage_features(v16, t)
    
    semantic_data.append({
        'idx': idx,
        'ab_gt': ab_gt,
        'enc_out': enc_out,  # [256, H, W]
        'stage_feats': stage_feats,
        'gray': gray,
        'sat': sat,
    })
    print(f'  Image {idx}: mean_sat={sat.mean():.1f}')

print(f'\nCollected {len(semantic_data)} images')


# ================================================================
# PART 2: FEATURE → COLOR MAPPING — The "Token" Decoder
# ================================================================
print()
print('=' * 70)
print('PART 2: FEATURE → COLOR — Can we decode semantic tokens?')
print('=' * 70)
print()

# For the final encoder output (256-dim), what do the top PCA components encode?
# Collect features + ground truth colors
all_feats = []
all_colors = []
all_sats = []

for d in semantic_data:
    enc = d['enc_out']  # [256, SZ, SZ]
    flat = enc.reshape(256, -1).T  # [SZ*SZ, 256]
    ab = d['ab_gt']  # [SZ, SZ, 2]
    ab_flat = ab.reshape(-1, 2)
    sat = d['sat'].flatten()
    
    # Sample pixels (weighted toward high saturation — those are the semantic tokens)
    weights = sat / (sat.sum() + 1e-6)
    sample = np.random.choice(len(flat), 500, replace=False, p=weights)
    all_feats.append(flat[sample])
    all_colors.append(ab_flat[sample])
    all_sats.append(sat[sample])

all_feats = np.vstack(all_feats)
all_colors = np.vstack(all_colors)
all_sats = np.concatenate(all_sats)

# PCA of features
feat_mean = all_feats.mean(0)
feat_centered = all_feats - feat_mean
U_f, S_f, Vt_f = np.linalg.svd(feat_centered, full_matrices=False)
print(f'Feature space: {all_feats.shape[1]}D')
print(f'Top 5 SV: {S_f[:5].round(1)}')
cumvar = np.cumsum(S_f**2) / (S_f**2).sum()
for k in [1, 2, 3, 5, 10, 20, 50]:
    print(f'  Top {k} PCA dims: {cumvar[k-1]*100:.1f}% variance')

# What color does each PCA direction encode?
print('\nPCA direction → color correlation:')
for pc_idx in range(10):
    proj = feat_centered @ Vt_f[pc_idx]
    corr_a = np.corrcoef(proj, all_colors[:, 0])[0, 1]
    corr_b = np.corrcoef(proj, all_colors[:, 1])[0, 1]
    
    # Interpret: a>0 = red, a<0 = green, b>0 = yellow, b<0 = blue
    color_name = ''
    if abs(corr_a) > 0.15:
        color_name += 'RED' if corr_a > 0 else 'GREEN'
    if abs(corr_b) > 0.15:
        if color_name: color_name += '+'
        color_name += 'YELLOW' if corr_b > 0 else 'BLUE'
    if not color_name: color_name = '(structural)'
    
    print(f'  PC{pc_idx}: corr_a={corr_a:+.3f} corr_b={corr_b:+.3f}  → {color_name}')


# ================================================================
# PART 3: SPATIAL SEMANTIC MAP — Where are the "tokens"?
# ================================================================
print()
print('=' * 70)
print('PART 3: SPATIAL SEMANTIC MAP — Sub-positions in the model')
print('=' * 70)
print()

# For one image, show what the feature space "sees" at each spatial position
d = semantic_data[0]
enc = d['enc_out']  # [256, SZ, SZ]
ab_gt = d['ab_gt']

# Project onto the top color directions
feat_flat = enc.reshape(256, -1).T - feat_mean  # [SZ*SZ, 256]
color_proj_a = feat_flat @ Vt_f[0]  # strongest color direction
color_proj_b = feat_flat @ Vt_f[1]

# Reshape to spatial maps
map_pc0 = color_proj_a.reshape(SZ, SZ)
map_pc1 = color_proj_b.reshape(SZ, SZ)

# Check: do these maps correlate with the actual color?
spatial_corr_a = np.corrcoef(map_pc0.flatten(), ab_gt[:,:,0].flatten())[0,1]
spatial_corr_b = np.corrcoef(map_pc1.flatten(), ab_gt[:,:,1].flatten())[0,1]
print(f'Image 0 spatial correlation:')
print(f'  PC0 map vs a* channel: r={spatial_corr_a:.3f}')
print(f'  PC1 map vs b* channel: r={spatial_corr_b:.3f}')

# Use linear regression to decode color from features
print('\nLinear decoding of color from features (per image):')
for i, d in enumerate(semantic_data[:5]):
    enc = d['enc_out'].reshape(256, -1).T
    ab = d['ab_gt'].reshape(-1, 2)
    
    # Full linear decode: features → color
    X = np.column_stack([enc, np.ones(len(enc))])
    Wa_full, _, _, _ = lstsq(X, ab[:, 0], rcond=None)
    Wb_full, _, _, _ = lstsq(X, ab[:, 1], rcond=None)
    pred_a = X @ Wa_full
    pred_b = X @ Wb_full
    
    r_a = np.corrcoef(pred_a, ab[:, 0])[0, 1]
    r_b = np.corrcoef(pred_b, ab[:, 1])[0, 1]
    rmse = np.sqrt(np.mean((pred_a - ab[:, 0])**2 + (pred_b - ab[:, 1])**2))
    
    print(f'  Image {i}: r_a={r_a:.3f}, r_b={r_b:.3f}, RMSE={rmse:.1f}')


# ================================================================
# PART 4: STAGE-BY-STAGE SEMANTICS — Where do tokens emerge?
# ================================================================
print()
print('=' * 70)
print('PART 4: STAGE-BY-STAGE — Where do semantic tokens emerge?')
print('=' * 70)
print()

d = semantic_data[0]
ab = d['ab_gt']

for si, sf in enumerate(d['stage_feats']):
    h, w, c = sf.shape
    # Resize ab to match feature map size
    ab_resized = cv2.resize(ab, (w, h))
    
    sf_flat = sf.reshape(-1, c)
    ab_flat = ab_resized.reshape(-1, 2)
    
    # Linear decode at this stage
    X = np.column_stack([sf_flat, np.ones(len(sf_flat))])
    Wa_s, _, _, _ = lstsq(X, ab_flat[:, 0], rcond=None)
    Wb_s, _, _, _ = lstsq(X, ab_flat[:, 1], rcond=None)
    pred_a = X @ Wa_s
    pred_b = X @ Wb_s
    
    r_a = np.corrcoef(pred_a, ab_flat[:, 0])[0, 1]
    r_b = np.corrcoef(pred_b, ab_flat[:, 1])[0, 1]
    rmse = np.sqrt(np.mean((pred_a - ab_flat[:, 0])**2 + (pred_b - ab_flat[:, 1])**2))
    
    # Feature statistics
    act_frac = (sf_flat > 0).mean()
    
    print(f'  Stage {si} ({c}D, {h}×{w}): r_a={r_a:.3f}, r_b={r_b:.3f}, '
          f'RMSE={rmse:.1f}, active={act_frac:.1%}')

# Average across all images
print('\n  Average across all images:')
for si in range(4):
    r_as, r_bs, rmses = [], [], []
    for d in semantic_data[:10]:
        sf = d['stage_feats'][si]
        h, w, c = sf.shape
        ab_r = cv2.resize(d['ab_gt'], (w, h))
        sf_flat = sf.reshape(-1, c)
        ab_flat = ab_r.reshape(-1, 2)
        X = np.column_stack([sf_flat, np.ones(len(sf_flat))])
        Wa_s, _, _, _ = lstsq(X, ab_flat[:, 0], rcond=None)
        Wb_s, _, _, _ = lstsq(X, ab_flat[:, 1], rcond=None)
        pa, pb = X @ Wa_s, X @ Wb_s
        r_as.append(np.corrcoef(pa, ab_flat[:, 0])[0, 1])
        r_bs.append(np.corrcoef(pb, ab_flat[:, 1])[0, 1])
        rmses.append(np.sqrt(np.mean((pa - ab_flat[:, 0])**2 + (pb - ab_flat[:, 1])**2)))
    print(f'  Stage {si}: r_a={np.mean(r_as):.3f}, r_b={np.mean(r_bs):.3f}, RMSE={np.mean(rmses):.1f}')


# ================================================================
# PART 5: THE GAP — What do first-principles features MISS?
# ================================================================
print()
print('=' * 70)
print('PART 5: THE 28.2% GAP — What does dir-W₂ coupling encode?')
print('=' * 70)
print()

# Build fitted-power-law mutant
print('Building fitted-power-law mutant (learned dirs + fitted SVs + real W₂)...')
sp = {}
for si in range(4):
    pw1 = v16._get_weight(f'encoder.arch.stages.{si}.0.pwconv1.weight').numpy()
    _, S, _ = np.linalg.svd(pw1, full_matrices=False)
    popt, _ = curve_fit(lambda i, A, a: A*(i+1.)**(-a), np.arange(len(S), dtype=float), S,
                        p0=[S[0], .5], maxfev=50000)
    sp[si] = {'A': popt[0], 'alpha': popt[1]}

def make_fitted_power_muts():
    muts = {}
    for si in range(4):
        d = dims[si]
        for bi in range(depths[si]):
            pf = f'encoder.arch.stages.{si}.{bi}'
            pw1r = v16._get_weight(f'{pf}.pwconv1.weight').numpy()
            U1, _, Vt1 = np.linalg.svd(pw1r, full_matrices=False)
            S_new = sp[si]['A'] * (np.arange(d, dtype=float) + 1.) ** (-sp[si]['alpha'])
            muts[f'{pf}.pwconv1.weight'] = torch.from_numpy((U1 * S_new) @ Vt1).float()
    return muts

fitted_muts = make_fitted_power_muts()

# Now build random-dirs + pinv mutant
print('Building random-dirs + fitted SVs + pinv W₂ mutant...')
def make_random_dir_muts(seed=42):
    np.random.seed(seed); muts = {}
    bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
    bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}
    for si in range(4):
        d, de = dims[si], dims[si]*4
        for bi in range(depths[si]):
            pf = f'encoder.arch.stages.{si}.{bi}'
            pw2r = v16._get_weight(f'{pf}.pwconv2.weight').numpy()
            U1 = np.linalg.qr(np.random.randn(de, d))[0]
            Vt1 = np.linalg.qr(np.random.randn(d, d))[0]
            S1 = sp[si]['A'] * (np.arange(d, dtype=float) + 1.) ** (-sp[si]['alpha'])
            W1 = (U1 * S1) @ Vt1
            W2 = (Vt1.T * (1./(S1 + 1e-6))) @ U1.T
            W2 *= np.linalg.norm(pw2r) / np.linalg.norm(W2)
            b1 = np.random.randn(de) * bias_stds[si] + bias_means[si]
            muts[f'{pf}.pwconv1.weight'] = torch.from_numpy(W1).float()
            muts[f'{pf}.pwconv2.weight'] = torch.from_numpy(W2).float()
            muts[f'{pf}.pwconv1.bias'] = torch.from_numpy(b1).float()
    return muts

random_muts = make_random_dir_muts()

# Compare features for one image
print('\nComparing features for Image 0:')
d0 = semantic_data[0]
t0 = torch.from_numpy(cv2.cvtColor(d0['gray'], cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.

enc_real, sf_real = run_encoder_with_stage_features(v16, t0)
enc_fitted, sf_fitted = run_encoder_with_stage_features(v16, t0, fitted_muts)
enc_random, sf_random = run_encoder_with_stage_features(v16, t0, random_muts)

# Compare final outputs
print('\n  Final encoder output (256D):')
for name, enc in [('Real', enc_real), ('Fitted SVs', enc_fitted), ('Random dirs', enc_random)]:
    flat = enc.reshape(256, -1).T
    # Project onto color directions 
    proj = (flat - feat_mean) @ Vt_f[:5].T  # top 5 PCA
    print(f'    {name:<15}: mean_norm={np.linalg.norm(flat, axis=1).mean():.1f}, '
          f'PC0_range=[{proj[:,0].min():.1f}, {proj[:,0].max():.1f}]')

# Per-pixel feature difference
print('\n  Feature difference (pixel-level):')
diff_fitted = enc_real.reshape(256, -1).T - enc_fitted.reshape(256, -1).T
diff_random = enc_real.reshape(256, -1).T - enc_random.reshape(256, -1).T
print(f'    Real vs Fitted SVs: mean_diff={np.linalg.norm(diff_fitted, axis=1).mean():.2f}, '
      f'cos_sim={np.mean([np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8) for a,b in zip(enc_real.reshape(256,-1).T[:100], enc_fitted.reshape(256,-1).T[:100])]):.3f}')
print(f'    Real vs Random:     mean_diff={np.linalg.norm(diff_random, axis=1).mean():.2f}, '
      f'cos_sim={np.mean([np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8) for a,b in zip(enc_real.reshape(256,-1).T[:100], enc_random.reshape(256,-1).T[:100])]):.3f}')

# Where is the difference largest? Does it correlate with semantic content?
diff_norm_fitted = np.linalg.norm(diff_fitted, axis=1).reshape(SZ, SZ)
diff_norm_random = np.linalg.norm(diff_random, axis=1).reshape(SZ, SZ)
sat_map = d0['sat']

print(f'\n  Does the gap correlate with color saturation?')
corr_sat_fitted = np.corrcoef(diff_norm_fitted.flatten(), sat_map.flatten())[0,1]
corr_sat_random = np.corrcoef(diff_norm_random.flatten(), sat_map.flatten())[0,1]
print(f'    Fitted diff vs saturation: r={corr_sat_fitted:.3f}')
print(f'    Random diff vs saturation: r={corr_sat_random:.3f}')

# Check: does the gap preferentially affect high-color or low-color regions?
high_sat = sat_map.flatten() > np.percentile(sat_map, 75)
low_sat = sat_map.flatten() < np.percentile(sat_map, 25)
print(f'    Random diff in high-sat regions: {diff_norm_random.flatten()[high_sat].mean():.2f}')
print(f'    Random diff in low-sat regions:  {diff_norm_random.flatten()[low_sat].mean():.2f}')
print(f'    Ratio (high/low): {diff_norm_random.flatten()[high_sat].mean() / (diff_norm_random.flatten()[low_sat].mean()+1e-6):.2f}x')


# ================================================================
# PART 6: COLOR DECODING — What "tokens" does each version produce?
# ================================================================
print()
print('=' * 70)
print('PART 6: COLOR DECODING — What tokens does each version produce?')
print('=' * 70)
print()

# Build a global color decoder from the real encoder
print('Building global color decoder...')
all_enc_train, all_ab_train = [], []
for d in semantic_data[:10]:
    t_img = torch.from_numpy(cv2.cvtColor(d['gray'], cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    enc, _ = run_encoder_with_stage_features(v16, t_img)
    flat = enc.reshape(256, -1).T
    ab_flat = d['ab_gt'].reshape(-1, 2)
    s = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    all_enc_train.append(flat[s])
    all_ab_train.append(ab_flat[s])
all_enc_train = np.vstack(all_enc_train)
all_ab_train = np.vstack(all_ab_train)

# SVD-based color directions
enc_m = all_enc_train.mean(0)
C_mat = (all_enc_train - enc_m).T @ all_ab_train / len(all_enc_train)
Uc, Sc, _ = np.linalg.svd(C_mat, full_matrices=False)
cd1, cd2 = Uc[:, 0], Uc[:, 1]
p1 = (all_enc_train - enc_m) @ cd1
p2 = (all_enc_train - enc_m) @ cd2
X2 = np.column_stack([p1, p2, np.ones(len(p1))])
Wa_g, *_ = lstsq(X2, all_ab_train[:, 0], rcond=None)
Wb_g, *_ = lstsq(X2, all_ab_train[:, 1], rcond=None)

# Decode color for each version on test images
print('\n  Color decoding comparison:')
print(f'  {"Image":<8} {"Real":<20} {"Fitted SVs":<20} {"Random dirs":<20}')
print('-' * 70)

for i, d in enumerate(semantic_data[10:15]):
    t_img = torch.from_numpy(cv2.cvtColor(d['gray'], cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    ab = d['ab_gt']
    err_z = np.sqrt(np.mean(ab**2))
    
    results = {}
    for name, muts_dict in [('Real', None), ('Fitted SVs', fitted_muts), ('Random dirs', random_muts)]:
        enc, _ = run_encoder_with_stage_features(v16, t_img, muts_dict)
        flat = (enc.reshape(256, -1).T - enc_m)
        f2 = np.column_stack([flat @ cd1, flat @ cd2, np.ones(SZ*SZ)])
        pred = np.stack([np.clip(f2 @ Wa_g, -50, 50).reshape(SZ,SZ),
                         np.clip(f2 @ Wb_g, -50, 50).reshape(SZ,SZ)], axis=2)
        err = np.sqrt(np.mean((pred - ab)**2))
        gap = (1 - err/err_z) * 100
        
        # What "colors" (semantic tokens) does it predict?
        mean_a = pred[:,:,0].mean()
        mean_b = pred[:,:,1].mean()
        results[name] = f'{gap:+5.1f}% (a={mean_a:+.1f} b={mean_b:+.1f})'
    
    print(f'  {i:<8} {results["Real"]:<20} {results["Fitted SVs"]:<20} {results["Random dirs"]:<20}')


# ================================================================
# PART 7: FEATURE CLUSTERING — Semantic sub-positions
# ================================================================
print()
print('=' * 70)
print('PART 7: FEATURE CLUSTERING — Semantic sub-positions')
print('=' * 70)
print()

# For one image, cluster the pixels by their feature vectors
# Then see what color each cluster corresponds to → these are the "semantic tokens"
d = semantic_data[0]
t_img = torch.from_numpy(cv2.cvtColor(d['gray'], cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
enc, _ = run_encoder_with_stage_features(v16, t_img)
flat = enc.reshape(256, -1).T  # [SZ*SZ, 256]
ab = d['ab_gt'].reshape(-1, 2)

# Simple k-means (manual, no sklearn dependency)
def simple_kmeans(X, k, n_iter=30):
    n, d = X.shape
    # Init: random subset
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assign
        dists = np.zeros((n, k))
        for c in range(k):
            dists[:, c] = np.sum((X - centers[c])**2, axis=1)
        labels = dists.argmin(axis=1)
        # Update
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)
    return labels, centers

# PCA first (for efficiency)
feat_c = flat - flat.mean(0)
_, _, Vt = np.linalg.svd(feat_c[:1000], full_matrices=False)
flat_pca = feat_c @ Vt[:20].T  # top 20 PCA dims

np.random.seed(42)
labels, centers = simple_kmeans(flat_pca, k=8, n_iter=50)

print(f'Found 8 semantic clusters:')
print(f'  {"Cluster":<10} {"Size":<8} {"Mean a*":<10} {"Mean b*":<10} {"Color":<15} {"Spatial"}')
print('-' * 65)

for c in range(8):
    mask = labels == c
    size = mask.sum()
    mean_a = ab[mask, 0].mean()
    mean_b = ab[mask, 1].mean()
    
    # Interpret color
    color = ''
    if abs(mean_a) > 3 or abs(mean_b) > 3:
        if mean_a > 3: color += 'red'
        elif mean_a < -3: color += 'green'
        if mean_b > 3: color += '+yellow' if color else 'yellow'
        elif mean_b < -3: color += '+blue' if color else 'blue'
    else:
        color = 'neutral'
    
    # Spatial position (where in the image?)
    ys = np.where(mask)[0] // SZ
    xs = np.where(mask)[0] % SZ
    spatial = f'y={ys.mean():.0f}±{ys.std():.0f}, x={xs.mean():.0f}±{xs.std():.0f}'
    
    print(f'  {c:<10} {size:<8} {mean_a:<+10.1f} {mean_b:<+10.1f} {color:<15} {spatial}')

# Compare: how does the clustering change with fitted SVs vs random dirs?
print('\nCluster stability across encoder versions:')
for name, muts_dict in [('Fitted SVs', fitted_muts), ('Random dirs', random_muts)]:
    enc_m2, _ = run_encoder_with_stage_features(v16, t_img, muts_dict)
    flat_m = enc_m2.reshape(256, -1).T
    feat_c_m = flat_m - flat_m.mean(0)
    _, _, Vt_m = np.linalg.svd(feat_c_m[:1000], full_matrices=False)
    flat_pca_m = feat_c_m @ Vt_m[:20].T
    labels_m, _ = simple_kmeans(flat_pca_m, k=8, n_iter=50)
    
    # Measure agreement with real clustering (normalized mutual information proxy)
    # Simple: for each real cluster, find the best matching mutant cluster
    agreements = []
    for c in range(8):
        real_mask = labels == c
        if real_mask.sum() == 0: continue
        best_overlap = 0
        for cm in range(8):
            mut_mask = labels_m == cm
            overlap = (real_mask & mut_mask).sum() / (real_mask | mut_mask).sum()
            best_overlap = max(best_overlap, overlap)
        agreements.append(best_overlap)
    
    mean_agree = np.mean(agreements)
    print(f'  {name}: mean cluster overlap = {mean_agree:.3f}')


print()
print('=' * 70)
print('DONE')
print('=' * 70)
