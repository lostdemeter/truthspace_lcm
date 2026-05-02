"""
Encoder Anatomy: What ARE the 2 principal color dimensions?

The encoder produces 256 features per pixel. For color, only 2 matter.
PC1 ↔ b-channel (blue-yellow), PC2 ↔ a-channel (green-red).

Questions:
1. What do PC1 and PC2 look like spatially? Are they smooth? Edge-aligned?
2. What image operations correlate with them? (multi-scale blur, edges, Gabor, etc.)
3. Can we approximate them from grayscale alone using geometric operations?
4. What is the MINIMUM geometric encoder that produces these 2 features?

Strategy: compute the encoder PCs for many images, then systematically test
which handcrafted features best predict each PC value per pixel.
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895


def get_encoder_features(v16, img_tensor):
    """Get final encoder features [256, H, W]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()  # [256, H, W]


def build_geometric_features(gray):
    """
    Build a rich set of handcrafted geometric features from grayscale.
    Each feature is a [H, W] map. We test which ones predict the encoder PCs.
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32) / 255.0
    features = {}

    # 1. Raw brightness at multiple scales
    features['brightness'] = gray_f
    for sigma in [2, 4, 8, 16, 32]:
        features[f'blur_{sigma}'] = cv2.GaussianBlur(gray_f, (0, 0), sigma)

    # 2. Local contrast at multiple scales (pixel - local mean)
    for sigma in [2, 4, 8, 16, 32]:
        local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigma)
        features[f'contrast_{sigma}'] = gray_f - local_mean

    # 3. Local variance at multiple scales
    for sigma in [4, 8, 16, 32]:
        local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigma)
        local_sq = cv2.GaussianBlur(gray_f**2, (0, 0), sigma)
        features[f'variance_{sigma}'] = np.sqrt(np.maximum(local_sq - local_mean**2, 0))

    # 4. Edge magnitude at multiple scales
    for ksize in [3, 5, 7]:
        sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)
        features[f'edge_{ksize}'] = np.sqrt(sx**2 + sy**2)

    # 5. Laplacian at multiple scales (blob detection)
    for ksize in [3, 5, 9]:
        features[f'laplacian_{ksize}'] = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F, ksize=ksize))

    # 6. Position maps
    ys = np.linspace(0, 1, h).reshape(-1, 1) * np.ones((1, w))
    xs = np.linspace(0, 1, w).reshape(1, -1) * np.ones((h, 1))
    features['y_pos'] = ys.astype(np.float32)
    features['x_pos'] = xs.astype(np.float32)

    # 7. Interaction features
    features['bright_x_ypos'] = gray_f * ys.astype(np.float32)
    features['bright_x_top'] = gray_f * (1 - ys).astype(np.float32)
    features['edge3_x_bright'] = features['edge_3'] * gray_f
    features['contrast4_x_bright'] = features['contrast_4'] * gray_f

    # 8. Multi-scale difference of gaussians (DoG) — texture/material indicator
    for s1, s2 in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]:
        b1 = cv2.GaussianBlur(gray_f, (0, 0), s1)
        b2 = cv2.GaussianBlur(gray_f, (0, 0), s2)
        features[f'dog_{s1}_{s2}'] = b1 - b2

    # 9. Local binary pattern approximation (texture)
    # Simplified: compare pixel to its neighborhood
    for r in [1, 3, 5]:
        kernel = np.ones((2*r+1, 2*r+1), np.float32) / (2*r+1)**2
        local_mean = cv2.filter2D(gray_f, -1, kernel)
        features[f'lbp_approx_{r}'] = (gray_f > local_mean).astype(np.float32)

    # 10. Gabor filters at various orientations and scales
    for theta_deg in [0, 45, 90, 135]:
        for lam in [4, 8, 16]:
            theta = np.radians(theta_deg)
            kern = cv2.getGaborKernel((21, 21), sigma=lam/2, theta=theta,
                                       lambd=lam, gamma=0.5, psi=0)
            features[f'gabor_{theta_deg}_{lam}'] = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kern))

    # 11. Histogram equalized (nonlinear brightness transform)
    eq = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    features['hist_eq'] = eq

    # 12. CLAHE (local histogram equalization — reveals local structure)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray).astype(np.float32) / 255.0
    features['clahe'] = cl
    features['clahe_minus_bright'] = cl - gray_f

    return features


print('=== ENCODER ANATOMY ===\n')

v16 = V16GeometricColorizer()

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/encoder_anatomy'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Compute PCA basis from multiple images
# ============================================================
print('=== PART 1: Computing PCA Basis ===\n')

PIXELS_PER_IMAGE = 3000
train_indices = list(range(50, 70))

all_enc = []

for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    enc = get_encoder_features(v16, img_tensor)  # [256, H, W]
    flat = enc.reshape(256, -1).T  # [H*W, 256]

    sample = np.random.choice(len(flat), min(PIXELS_PER_IMAGE, len(flat)), replace=False)
    all_enc.append(flat[sample])

all_enc = np.vstack(all_enc)
print(f'Collected {all_enc.shape[0]} pixel features from {len(train_indices)} images')

# Compute PCA
enc_mean = all_enc.mean(axis=0)
centered = all_enc - enc_mean
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
cumvar = np.cumsum(S**2) / (S**2).sum()

print(f'PCA computed. Top singular value ratios:')
for i in range(5):
    ratio = S[i] / S[i+1]
    phi_err = abs(ratio - PHI) / PHI * 100
    marker = ' ← φ!' if phi_err < 15 else ''
    print(f'  S[{i}]/S[{i+1}] = {ratio:.4f} ({phi_err:.1f}% from φ){marker}')


# ============================================================
# PART 2: Visualize PCs spatially on specific images
# ============================================================
print('\n=== PART 2: Spatial Visualization ===\n')

vis_indices = [54, 56, 58, 60]  # diverse images

for idx in vis_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    enc = get_encoder_features(v16, img_tensor)  # [256, H, W]
    flat = enc.reshape(256, -1).T  # [H*W, 256]

    # Project to PCs
    pcs = (flat - enc_mean) @ Vt[:6].T  # [H*W, 6]
    pc_maps = pcs.T.reshape(6, SZ, SZ)

    # Visualize: normalize each PC map to [0, 255] for display
    vis_strips = []

    # Row 1: Gray, GT color, GT-a, GT-b
    gt_a_vis = np.clip((ab_gt[:,:,0] + 40) / 80 * 255, 0, 255).astype(np.uint8)
    gt_b_vis = np.clip((ab_gt[:,:,1] + 40) / 80 * 255, 0, 255).astype(np.uint8)
    row1 = [gray, cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)]  # placeholder
    # Use colormaps for better visualization
    gt_a_cm = cv2.applyColorMap(gt_a_vis, cv2.COLORMAP_COOL)
    gt_b_cm = cv2.applyColorMap(gt_b_vis, cv2.COLORMAP_AUTUMN)

    # Row 2: PC1 through PC4
    pc_vis = []
    for pc_i in range(4):
        pc_map = pc_maps[pc_i]
        # Normalize to [0, 255]
        vmin, vmax = np.percentile(pc_map, [2, 98])
        if vmax - vmin < 1e-8: vmax = vmin + 1
        normalized = np.clip((pc_map - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        pc_vis.append(cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS))

    # Build strip
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strip_top = np.hstack([gray_bgr, r, gt_a_cm, gt_b_cm])
    strip_bot = np.hstack(pc_vis)

    # Labels
    labels_top = ['Gray', 'GT Color', 'GT a (green-red)', 'GT b (blue-yellow)']
    labels_bot = ['PC1 (→b)', 'PC2 (→a)', 'PC3', 'PC4']

    for i, label in enumerate(labels_top):
        x = i * SZ + 3
        cv2.putText(strip_top, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(strip_top, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)

    for i, label in enumerate(labels_bot):
        x = i * SZ + 3
        cv2.putText(strip_bot, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(strip_bot, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)

    combined = np.vstack([strip_top, strip_bot])
    cv2.imwrite(os.path.join(out_dir, f'anatomy_{name}.jpg'), combined)

    # Per-pixel correlation: PC vs GT
    pc1_flat = pc_maps[0].flatten()
    pc2_flat = pc_maps[1].flatten()
    gt_a_flat = ab_gt[:,:,0].flatten()
    gt_b_flat = ab_gt[:,:,1].flatten()

    c_pc1_a = np.corrcoef(pc1_flat, gt_a_flat)[0,1]
    c_pc1_b = np.corrcoef(pc1_flat, gt_b_flat)[0,1]
    c_pc2_a = np.corrcoef(pc2_flat, gt_a_flat)[0,1]
    c_pc2_b = np.corrcoef(pc2_flat, gt_b_flat)[0,1]

    print(f'  {name}: PC1↔a={c_pc1_a:.3f} PC1↔b={c_pc1_b:.3f} | PC2↔a={c_pc2_a:.3f} PC2↔b={c_pc2_b:.3f}')


# ============================================================
# PART 3: Which geometric features predict the encoder PCs?
# ============================================================
print('\n=== PART 3: Geometric Feature → PC Correlation ===\n')

# Collect per-pixel: geometric features + PC values
all_geo_feats = {}
all_pc1 = []
all_pc2 = []
all_gt_a_vals = []
all_gt_b_vals = []

PIXELS_PER_IMAGE_GEO = 2000

for idx in train_indices[:10]:  # subset for speed
    im = cv2.imread(all_imgs[idx])
    if im is None: continue

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    enc = get_encoder_features(v16, img_tensor)
    flat = enc.reshape(256, -1).T
    pcs = (flat - enc_mean) @ Vt[:2].T  # [H*W, 2]
    pc1_map = pcs[:, 0].reshape(SZ, SZ)
    pc2_map = pcs[:, 1].reshape(SZ, SZ)

    geo = build_geometric_features(gray)

    sample = np.random.choice(SZ * SZ, min(PIXELS_PER_IMAGE_GEO, SZ * SZ), replace=False)
    ys = sample // SZ
    xs = sample % SZ

    all_pc1.append(pc1_map[ys, xs])
    all_pc2.append(pc2_map[ys, xs])
    all_gt_a_vals.append(ab_gt[ys, xs, 0])
    all_gt_b_vals.append(ab_gt[ys, xs, 1])

    for fname, fmap in geo.items():
        if fname not in all_geo_feats:
            all_geo_feats[fname] = []
        all_geo_feats[fname].append(fmap[ys, xs])

pc1_all = np.concatenate(all_pc1)
pc2_all = np.concatenate(all_pc2)
gt_a_all = np.concatenate(all_gt_a_vals)
gt_b_all = np.concatenate(all_gt_b_vals)

print(f'Collected {len(pc1_all)} pixels with {len(all_geo_feats)} geometric features')
print(f'\nCorrelations with encoder PC1 (→ b-channel):')

pc1_corrs = {}
pc2_corrs = {}
gt_a_corrs = {}
gt_b_corrs = {}

for fname in sorted(all_geo_feats.keys()):
    vals = np.concatenate(all_geo_feats[fname])
    if np.std(vals) < 1e-8: continue

    c1 = np.corrcoef(vals, pc1_all)[0, 1]
    c2 = np.corrcoef(vals, pc2_all)[0, 1]
    ca = np.corrcoef(vals, gt_a_all)[0, 1]
    cb = np.corrcoef(vals, gt_b_all)[0, 1]

    pc1_corrs[fname] = c1
    pc2_corrs[fname] = c2
    gt_a_corrs[fname] = ca
    gt_b_corrs[fname] = cb

# Show top correlates for PC1
sorted_pc1 = sorted(pc1_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
print(f'\n  Top PC1 correlates (PC1 ↔ b-channel, r={np.corrcoef(pc1_all, gt_b_all)[0,1]:.3f}):')
for fname, corr in sorted_pc1[:15]:
    gt_b_c = gt_b_corrs[fname]
    print(f'    {fname:<25} r_PC1={corr:+.3f}  r_GT_b={gt_b_c:+.3f}')

sorted_pc2 = sorted(pc2_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
print(f'\n  Top PC2 correlates (PC2 ↔ a-channel, r={np.corrcoef(pc2_all, gt_a_all)[0,1]:.3f}):')
for fname, corr in sorted_pc2[:15]:
    gt_a_c = gt_a_corrs[fname]
    print(f'    {fname:<25} r_PC2={corr:+.3f}  r_GT_a={gt_a_c:+.3f}')

# Direct GT correlations
sorted_gt_b = sorted(gt_b_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
print(f'\n  Top DIRECT b-channel correlates:')
for fname, corr in sorted_gt_b[:10]:
    print(f'    {fname:<25} r_GT_b={corr:+.3f}')

sorted_gt_a = sorted(gt_a_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
print(f'\n  Top DIRECT a-channel correlates:')
for fname, corr in sorted_gt_a[:10]:
    print(f'    {fname:<25} r_GT_a={corr:+.3f}')


# ============================================================
# PART 4: Linear model: geometric features → encoder PCs
# Can we reconstruct the encoder's output without the encoder?
# ============================================================
print('\n=== PART 4: Geometric → Encoder PCs (Linear) ===\n')

# Build feature matrix
feat_names = sorted(all_geo_feats.keys())
X_geo = np.column_stack([np.concatenate(all_geo_feats[f]) for f in feat_names])
X_geo_b = np.column_stack([X_geo, np.ones(len(X_geo))])

# Split
split = int(len(X_geo) * 0.7)
X_tr, X_te = X_geo_b[:split], X_geo_b[split:]
pc1_tr, pc1_te = pc1_all[:split], pc1_all[split:]
pc2_tr, pc2_te = pc2_all[:split], pc2_all[split:]
gt_a_tr, gt_a_te = gt_a_all[:split], gt_a_all[split:]
gt_b_tr, gt_b_te = gt_b_all[:split], gt_b_all[split:]

# Fit: geo → PC1, PC2 (ridge to handle collinearity)
lam = 0.1
XtX = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
W_pc1 = np.linalg.solve(XtX, X_tr.T @ pc1_tr)
W_pc2 = np.linalg.solve(XtX, X_tr.T @ pc2_tr)

pred_pc1 = X_te @ W_pc1
pred_pc2 = X_te @ W_pc2

r2_pc1 = 1 - np.mean((pc1_te - pred_pc1)**2) / (np.var(pc1_te) + 1e-8)
r2_pc2 = 1 - np.mean((pc2_te - pred_pc2)**2) / (np.var(pc2_te) + 1e-8)
corr_pc1 = np.corrcoef(pred_pc1, pc1_te)[0, 1]
corr_pc2 = np.corrcoef(pred_pc2, pc2_te)[0, 1]

print(f'Predicting encoder PCs from geometric features:')
print(f'  PC1: R²={r2_pc1:.3f}, r={corr_pc1:.3f}')
print(f'  PC2: R²={r2_pc2:.3f}, r={corr_pc2:.3f}')

# And the end-to-end test: geo features → PCs → color
# Use the PCA-16 linear model from encoder_linear_v2
# But here: can geo features predict GT color DIRECTLY?
W_a = np.linalg.solve(XtX, X_tr.T @ gt_a_tr)
W_b = np.linalg.solve(XtX, X_tr.T @ gt_b_tr)

pred_a = X_te @ W_a
pred_b = X_te @ W_b
err_geo_direct = np.sqrt(np.mean((gt_a_te - pred_a)**2 + (gt_b_te - pred_b)**2))
err_zero = np.sqrt(np.mean(gt_a_te**2 + gt_b_te**2))

r2_a = 1 - np.mean((gt_a_te - pred_a)**2) / (np.var(gt_a_te) + 1e-8)
r2_b = 1 - np.mean((gt_b_te - pred_b)**2) / (np.var(gt_b_te) + 1e-8)

print(f'\nGeometric features → GT color DIRECTLY:')
print(f'  R²: a={r2_a:.3f}, b={r2_b:.3f}')
print(f'  Error: {err_geo_direct:.2f} (predict zero: {err_zero:.2f})')
print(f'  Gap closed: {(1-err_geo_direct/err_zero)*100:.1f}%')


# ============================================================
# PART 5: Feature importance for the geometric → color model
# ============================================================
print('\n=== PART 5: Feature Importance (Geometric → Color) ===\n')

# Standardized importance
feat_stds = X_tr[:, :-1].std(axis=0)  # exclude bias
W_combined = np.sqrt(W_a[:-1]**2 + W_b[:-1]**2)
importance = W_combined * feat_stds

sorted_imp = np.argsort(importance)[::-1]
print(f'  {"Feature":<25} {"Std×|W|":>10} {"W_a":>10} {"W_b":>10}')
print(f'  {"-"*57}')
for i in sorted_imp[:20]:
    print(f'  {feat_names[i]:<25} {importance[i]:10.3f} {W_a[i]:10.4f} {W_b[i]:10.4f}')


# ============================================================
# PART 6: Progressive geometric encoder
# Which minimal feature set gets closest to encoder quality?
# ============================================================
print('\n=== PART 6: Progressive Geometric Encoder ===\n')

# Greedy feature selection for predicting GT color
remaining = list(range(X_geo.shape[1]))
selected = []
X_geo_test_full = X_geo_b  # for reuse

for step in range(20):
    best_err = float('inf')
    best_feat = None

    for f in remaining:
        trial = selected + [f]
        X_sel_tr = X_tr[:, trial + [-1]]  # selected + bias
        X_sel_te = X_te[:, trial + [-1]]

        try:
            XtX_s = X_sel_tr.T @ X_sel_tr + 0.1 * np.eye(X_sel_tr.shape[1])
            Wa = np.linalg.solve(XtX_s, X_sel_tr.T @ gt_a_tr)
            Wb = np.linalg.solve(XtX_s, X_sel_tr.T @ gt_b_tr)
            pa = X_sel_te @ Wa
            pb = X_sel_te @ Wb
            err = np.sqrt(np.mean((gt_a_te - pa)**2 + (gt_b_te - pb)**2))
            if err < best_err:
                best_err = err
                best_feat = f
        except:
            continue

    if best_feat is None: break
    selected.append(best_feat)
    remaining.remove(best_feat)

    gap = (1 - best_err / err_zero) * 100
    fname = feat_names[best_feat] if best_feat < len(feat_names) else 'bias'
    print(f'  +{fname:<25} → err={best_err:.2f} (gap closed: {gap:.1f}%)')

print(f'\n  Encoder linear model comparison: gap closed ~24.5%')
print(f'  DDColor comparison: gap closed ~45.6%')
print(f'  Best geometric combo ({len(selected)} features): gap closed {gap:.1f}%')


print(f'\nOutput saved to: {out_dir}/')
print('Done!')
