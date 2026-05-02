"""
Reverse Navigation: Find the geometric encoder by working backwards.

FORWARD (failed at 6%):
  "What handcrafted features predict color?"
  → Requires world knowledge, ambiguous

REVERSE (Doc 204):
  "We HAVE the encoder's 2D output. What produces it?"
  → Deterministic target, constrained, solvable

The encoder maps grayscale → 256-dim features.
For color, only 2 dimensions matter.
These 2 dimensions are deterministic functions of the grayscale input.
What are these functions? Can we describe them geometrically?

Strategy:
1. Compute encoder's 2D color projection for many images
2. Analyze spatial structure of these 2 fields
3. Find geometric operations that replicate them
4. Test: geometric approximation → linear → color
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895


def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def get_encoder_features(v16, img_tensor):
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


print('=== REVERSE NAVIGATION: Finding the Geometric Encoder ===\n')

v16 = V16GeometricColorizer()
SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/reverse_encoder'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Build the encoder's 2D color basis from training data
# ============================================================
print('=== PART 1: Encoder Color Basis ===\n')

train_indices = list(range(50, 75))
PIXELS_PER_IMAGE = 3000

all_enc = []
all_gt_ab = []

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
    flat = enc.reshape(256, -1).T

    sample = np.random.choice(len(flat), min(PIXELS_PER_IMAGE, len(flat)), replace=False)
    all_enc.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt_ab.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc = np.vstack(all_enc)
all_gt_ab = np.vstack(all_gt_ab)
print(f'Collected {all_enc.shape[0]} pixels')

# Find the 2 supervised directions: the feature-space directions most predictive of color
enc_mean = all_enc.mean(axis=0)
enc_centered = all_enc - enc_mean

# Cross-covariance SVD: find directions in feature space that predict ab
C = enc_centered.T @ all_gt_ab / len(enc_centered)  # [256, 2]
U_color, S_color, Vt_color = np.linalg.svd(C, full_matrices=False)

print(f'Color directions SVD: S = [{S_color[0]:.4f}, {S_color[1]:.4f}]')
print(f'  S[0]/S[1] = {S_color[0]/S_color[1]:.4f} (φ err = {abs(S_color[0]/S_color[1]-PHI)/PHI*100:.1f}%)')

# The 2 color directions in 256-dim space
color_dir_1 = U_color[:, 0]  # [256]
color_dir_2 = U_color[:, 1]  # [256]

# Project encoder features onto these 2 directions → the 2D "color code"
proj_1 = enc_centered @ color_dir_1  # [N]
proj_2 = enc_centered @ color_dir_2  # [N]

# Verify: these 2 projections predict color well
from numpy.linalg import lstsq
X_2d = np.column_stack([proj_1, proj_2, np.ones(len(proj_1))])
W_a, _, _, _ = lstsq(X_2d, all_gt_ab[:, 0], rcond=None)
W_b, _, _, _ = lstsq(X_2d, all_gt_ab[:, 1], rcond=None)
pred_a = X_2d @ W_a
pred_b = X_2d @ W_b
err_2d = np.sqrt(np.mean((all_gt_ab[:,0] - pred_a)**2 + (all_gt_ab[:,1] - pred_b)**2))
err_zero = np.sqrt(np.mean(all_gt_ab**2))
print(f'2D color code → GT: err={err_2d:.2f} (zero={err_zero:.2f}, gap={((1-err_2d/err_zero)*100):.1f}%)')


# ============================================================
# PART 2: Visualize and analyze the 2D color code maps
# ============================================================
print('\n=== PART 2: What Do The 2 Color Fields Look Like? ===\n')

vis_indices = [51, 54, 56, 58, 60, 62]

for idx in vis_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_encoder_features(v16, img_tensor)
    flat = (enc.reshape(256, -1).T - enc_mean)

    # The 2 color code maps
    field_1 = (flat @ color_dir_1).reshape(SZ, SZ)
    field_2 = (flat @ color_dir_2).reshape(SZ, SZ)

    # Analyze spatial properties
    # Frequency content: what fraction of energy is low-frequency?
    for fi, field in enumerate([field_1, field_2]):
        fft = np.fft.fft2(field)
        power = np.abs(fft)**2
        # Low-freq = center 25% of spectrum
        h, w = field.shape
        mask_low = np.zeros_like(power, dtype=bool)
        ch, cw = h//4, w//4
        mask_low[:ch, :cw] = True
        mask_low[:ch, -cw:] = True
        mask_low[-ch:, :cw] = True
        mask_low[-ch:, -cw:] = True
        low_frac = power[mask_low].sum() / power.sum()

        # Correlation with grayscale
        corr_gray = np.corrcoef(field.flatten(), gray.flatten().astype(float))[0, 1]

        # Smoothness: mean absolute gradient
        gx = np.diff(field, axis=1)
        gy = np.diff(field, axis=0)
        smoothness = np.sqrt(np.mean(gx**2) + np.mean(gy**2))

        # Edge alignment: are field gradients aligned with image edges?
        img_gx = cv2.Sobel(gray.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        img_gy = cv2.Sobel(gray.astype(float), cv2.CV_64F, 0, 1, ksize=3)
        fld_gx = cv2.Sobel(field, cv2.CV_64F, 1, 0, ksize=3)
        fld_gy = cv2.Sobel(field, cv2.CV_64F, 0, 1, ksize=3)
        # Cosine similarity of gradient vectors
        dot = img_gx * fld_gx + img_gy * fld_gy
        mag_img = np.sqrt(img_gx**2 + img_gy**2) + 1e-8
        mag_fld = np.sqrt(fld_gx**2 + fld_gy**2) + 1e-8
        cos_sim = dot / (mag_img * mag_fld)
        # Only at significant edges
        sig_edges = mag_img > np.percentile(mag_img, 75)
        edge_align = np.abs(cos_sim[sig_edges]).mean()

        print(f'  {name} Field {fi+1}: low_freq={low_frac:.3f}, r_gray={corr_gray:+.3f}, '
              f'smoothness={smoothness:.2f}, edge_align={edge_align:.3f}')

    # Visualization
    def normalize_field(f):
        vmin, vmax = np.percentile(f, [2, 98])
        if vmax - vmin < 1e-8: vmax = vmin + 1
        return np.clip((f - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    f1_vis = cv2.applyColorMap(normalize_field(field_1), cv2.COLORMAP_VIRIDIS)
    f2_vis = cv2.applyColorMap(normalize_field(field_2), cv2.COLORMAP_VIRIDIS)
    gt_a_vis = cv2.applyColorMap(normalize_field(ab_gt[:,:,0]), cv2.COLORMAP_COOL)
    gt_b_vis = cv2.applyColorMap(normalize_field(ab_gt[:,:,1]), cv2.COLORMAP_AUTUMN)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strip = np.hstack([gray_bgr, f1_vis, f2_vis, gt_a_vis, gt_b_vis, r])
    labels = ['Gray', 'Color Field 1', 'Color Field 2', 'GT a', 'GT b', 'GT Color']
    for i, label in enumerate(labels):
        x = i * SZ + 3
        cv2.putText(strip, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(strip, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    cv2.imwrite(os.path.join(out_dir, f'fields_{name}.jpg'), strip)


# ============================================================
# PART 3: REVERSE — Can geometric features predict the FIELDS?
# This is different from predicting color! The fields are
# deterministic functions of the grayscale input.
# ============================================================
print('\n=== PART 3: Reverse — Geometric Features → Encoder Fields ===\n')

from geometric_encoder import phi_pyramid_features

all_geo = []
all_f1 = []
all_f2 = []
all_gt_a_rev = []
all_gt_b_rev = []
PIX_PER = 2000

for idx in train_indices[:15]:
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
    flat = (enc.reshape(256, -1).T - enc_mean)

    field_1 = (flat @ color_dir_1).reshape(SZ, SZ)
    field_2 = (flat @ color_dir_2).reshape(SZ, SZ)

    geo_feats, feat_names = phi_pyramid_features(gray)

    sample = np.random.choice(SZ*SZ, min(PIX_PER, SZ*SZ), replace=False)
    ys, xs = sample // SZ, sample % SZ

    geo_pix = geo_feats[:, ys, xs].T
    all_geo.append(geo_pix)
    all_f1.append(field_1[ys, xs])
    all_f2.append(field_2[ys, xs])
    all_gt_a_rev.append(ab_gt[ys, xs, 0])
    all_gt_b_rev.append(ab_gt[ys, xs, 1])

X_geo = np.vstack(all_geo)
f1_target = np.concatenate(all_f1)
f2_target = np.concatenate(all_f2)
gt_a_rev = np.concatenate(all_gt_a_rev)
gt_b_rev = np.concatenate(all_gt_b_rev)

print(f'Collected {len(X_geo)} pixels, {X_geo.shape[1]} features')

# Standardize
geo_mean = X_geo.mean(axis=0)
geo_std = X_geo.std(axis=0)
geo_std[geo_std < 1e-8] = 1.0
X_normed = (X_geo - geo_mean) / geo_std

# Split
split = int(len(X_normed) * 0.7)
X_tr = X_normed[:split]
X_te = X_normed[split:]
f1_tr, f1_te = f1_target[:split], f1_target[split:]
f2_tr, f2_te = f2_target[:split], f2_target[split:]
gt_a_tr, gt_a_te = gt_a_rev[:split], gt_a_rev[split:]
gt_b_tr, gt_b_te = gt_b_rev[:split], gt_b_rev[split:]

# Fit: geo → field_1, field_2 (REVERSE target = encoder output)
for lam in [0.1, 1.0, 10.0]:
    X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_te_b = np.column_stack([X_te, np.ones(len(X_te))])
    XtX = X_tr_b.T @ X_tr_b + lam * np.eye(X_tr_b.shape[1])

    W_f1 = np.linalg.solve(XtX, X_tr_b.T @ f1_tr)
    W_f2 = np.linalg.solve(XtX, X_tr_b.T @ f2_tr)

    pred_f1 = X_te_b @ W_f1
    pred_f2 = X_te_b @ W_f2

    r2_f1 = 1 - np.mean((f1_te - pred_f1)**2) / (np.var(f1_te) + 1e-8)
    r2_f2 = 1 - np.mean((f2_te - pred_f2)**2) / (np.var(f2_te) + 1e-8)
    corr_f1 = np.corrcoef(pred_f1, f1_te)[0, 1]
    corr_f2 = np.corrcoef(pred_f2, f2_te)[0, 1]

    # End-to-end: geo → predicted fields → color
    pred_fields = np.column_stack([pred_f1, pred_f2, np.ones(len(pred_f1))])
    pred_a = pred_fields @ W_a
    pred_b = pred_fields @ W_b
    err_e2e = np.sqrt(np.mean((gt_a_te - pred_a)**2 + (gt_b_te - pred_b)**2))
    err_z = np.sqrt(np.mean(gt_a_te**2 + gt_b_te**2))

    print(f'  λ={lam:4.1f}: Field1 R²={r2_f1:.3f}(r={corr_f1:.3f}) '
          f'Field2 R²={r2_f2:.3f}(r={corr_f2:.3f}) '
          f'E2E err={err_e2e:.2f} gap={((1-err_e2e/err_z)*100):.1f}%')

# Compare: geo → color DIRECTLY (forward approach)
print(f'\nComparison:')
X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
X_te_b = np.column_stack([X_te, np.ones(len(X_te))])
XtX = X_tr_b.T @ X_tr_b + 10.0 * np.eye(X_tr_b.shape[1])
W_a_direct = np.linalg.solve(XtX, X_tr_b.T @ gt_a_tr)
W_b_direct = np.linalg.solve(XtX, X_tr_b.T @ gt_b_tr)
pred_a_dir = X_te_b @ W_a_direct
pred_b_dir = X_te_b @ W_b_direct
err_direct = np.sqrt(np.mean((gt_a_te - pred_a_dir)**2 + (gt_b_te - pred_b_dir)**2))
print(f'  Forward (geo → color directly): err={err_direct:.2f}, gap={(1-err_direct/err_z)*100:.1f}%')
print(f'  Reverse (geo → fields → color): err={err_e2e:.2f}, gap={(1-err_e2e/err_z)*100:.1f}%')
print(f'  Encoder (enc fields → color):    err={err_2d:.2f}, gap={(1-err_2d/err_zero)*100:.1f}%')


# ============================================================
# PART 4: What makes the encoder fields DIFFERENT from our prediction?
# The residual = what the CNN knows that geometry doesn't
# ============================================================
print('\n=== PART 4: The Residual — What CNN Knows That Geometry Doesn\'t ===\n')

# For each test image, compute: encoder field, predicted field, residual
print('Per-image analysis of the residual:')

for idx in train_indices[:6]:
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
    enc = get_encoder_features(v16, img_tensor)
    flat = (enc.reshape(256, -1).T - enc_mean)

    # True encoder fields
    true_f1 = (flat @ color_dir_1).reshape(SZ, SZ)
    true_f2 = (flat @ color_dir_2).reshape(SZ, SZ)

    # Predicted fields from geometry
    geo_feats, _ = phi_pyramid_features(gray)
    feat_flat = geo_feats.reshape(len(geo_feats), -1).T
    feat_normed = (feat_flat - geo_mean) / geo_std
    feat_b = np.column_stack([feat_normed, np.ones(len(feat_normed))])

    # Use best λ weights
    XtX = X_tr_b.T @ X_tr_b + 1.0 * np.eye(X_tr_b.shape[1])
    W_f1 = np.linalg.solve(XtX, X_tr_b.T @ f1_tr)
    W_f2 = np.linalg.solve(XtX, X_tr_b.T @ f2_tr)

    pred_f1 = (feat_b @ W_f1).reshape(SZ, SZ)
    pred_f2 = (feat_b @ W_f2).reshape(SZ, SZ)

    # Residuals
    res_f1 = true_f1 - pred_f1
    res_f2 = true_f2 - pred_f2

    # Analyze residuals
    r2_f1_img = 1 - np.var(res_f1) / (np.var(true_f1) + 1e-8)
    r2_f2_img = 1 - np.var(res_f2) / (np.var(true_f2) + 1e-8)

    # Is the residual correlated with GT color? (i.e., does the residual carry color info?)
    res_f1_flat = res_f1.flatten()
    res_f2_flat = res_f2.flatten()
    gt_a_flat = ab_gt[:,:,0].flatten()
    gt_b_flat = ab_gt[:,:,1].flatten()
    corr_res_a = np.corrcoef(res_f1_flat, gt_a_flat)[0, 1]
    corr_res_b = np.corrcoef(res_f1_flat, gt_b_flat)[0, 1]

    # Is residual smooth or noisy?
    res_gx = np.diff(res_f1, axis=1)
    res_gy = np.diff(res_f1, axis=0)
    res_smooth = np.sqrt(np.mean(res_gx**2) + np.mean(res_gy**2))
    true_gx = np.diff(true_f1, axis=1)
    true_gy = np.diff(true_f1, axis=0)
    true_smooth = np.sqrt(np.mean(true_gx**2) + np.mean(true_gy**2))

    print(f'  {name}: R²_f1={r2_f1_img:.3f} R²_f2={r2_f2_img:.3f} | '
          f'res↔a={corr_res_a:+.3f} res↔b={corr_res_b:+.3f} | '
          f'smooth: true={true_smooth:.2f} res={res_smooth:.2f}')

    # Visualize: true field, predicted field, residual, GT
    def norm(f):
        vmin, vmax = np.percentile(f, [2, 98])
        if vmax - vmin < 1e-8: vmax = vmin + 1
        return np.clip((f - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    true_vis = cv2.applyColorMap(norm(true_f1), cv2.COLORMAP_VIRIDIS)
    pred_vis = cv2.applyColorMap(norm(pred_f1), cv2.COLORMAP_VIRIDIS)
    res_vis = cv2.applyColorMap(norm(res_f1), cv2.COLORMAP_HOT)

    # End-to-end color from geometric approximation
    pred_fields_img = np.column_stack([pred_f1.flatten(), pred_f2.flatten(),
                                       np.ones(SZ*SZ)])
    pred_a_img = np.clip(pred_fields_img @ W_a, -50, 50).reshape(SZ, SZ)
    pred_b_img = np.clip(pred_fields_img @ W_b, -50, 50).reshape(SZ, SZ)
    ab_geo = np.stack([pred_a_img, pred_b_img], axis=2)

    # Color from true encoder fields (for comparison)
    true_fields_img = np.column_stack([true_f1.flatten(), true_f2.flatten(),
                                       np.ones(SZ*SZ)])
    true_a_img = np.clip(true_fields_img @ W_a, -50, 50).reshape(SZ, SZ)
    true_b_img = np.clip(true_fields_img @ W_b, -50, 50).reshape(SZ, SZ)
    ab_enc = np.stack([true_a_img, true_b_img], axis=2)

    L = lab_gt[:,:,0]
    bgr_geo = ab_to_bgr(ab_geo, L)
    bgr_enc = ab_to_bgr(ab_enc, L)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strip = np.hstack([gray_bgr, true_vis, pred_vis, res_vis, bgr_enc, bgr_geo, r])
    labels = ['Gray', 'True F1', 'Predicted F1', 'Residual', 'Enc→Color', 'Geo→Color', 'GT']
    for i, label in enumerate(labels):
        x = i * SZ + 3
        cv2.putText(strip, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(strip, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    cv2.imwrite(os.path.join(out_dir, f'reverse_{name}.jpg'), strip)


# ============================================================
# PART 5: Held-out test — full pipeline comparison
# ============================================================
print('\n=== PART 5: Held-Out Test — Full Pipeline ===\n')

test_indices = list(range(80, 90))
results = []

# Refit on all training data
X_all_b = np.column_stack([X_normed, np.ones(len(X_normed))])
XtX = X_all_b.T @ X_all_b + 1.0 * np.eye(X_all_b.shape[1])
W_f1_final = np.linalg.solve(XtX, X_all_b.T @ f1_target)
W_f2_final = np.linalg.solve(XtX, X_all_b.T @ f2_target)

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat_img = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat_img < 2: continue

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    # True encoder fields
    enc = get_encoder_features(v16, img_tensor)
    flat = (enc.reshape(256, -1).T - enc_mean)
    true_f1 = (flat @ color_dir_1).reshape(SZ, SZ)
    true_f2 = (flat @ color_dir_2).reshape(SZ, SZ)

    # Geometric approximation of fields
    geo_feats, _ = phi_pyramid_features(gray)
    feat_flat = geo_feats.reshape(len(geo_feats), -1).T
    feat_normed_img = (feat_flat - geo_mean) / geo_std
    feat_b = np.column_stack([feat_normed_img, np.ones(len(feat_normed_img))])
    pred_f1 = (feat_b @ W_f1_final).reshape(SZ, SZ)
    pred_f2 = (feat_b @ W_f2_final).reshape(SZ, SZ)

    # Color from encoder fields (the geometric AI from doc 237)
    enc_fields = np.column_stack([true_f1.flatten(), true_f2.flatten(), np.ones(SZ*SZ)])
    enc_a = np.clip(enc_fields @ W_a, -50, 50).reshape(SZ, SZ)
    enc_b = np.clip(enc_fields @ W_b, -50, 50).reshape(SZ, SZ)
    ab_enc = np.stack([enc_a, enc_b], axis=2)

    # Color from geometric fields (the reverse-navigated encoder)
    geo_fields = np.column_stack([pred_f1.flatten(), pred_f2.flatten(), np.ones(SZ*SZ)])
    geo_a = np.clip(geo_fields @ W_a, -50, 50).reshape(SZ, SZ)
    geo_b = np.clip(geo_fields @ W_b, -50, 50).reshape(SZ, SZ)
    ab_geo = np.stack([geo_a, geo_b], axis=2)

    # DDColor
    ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    err_enc = np.sqrt(np.mean((ab_enc - ab_gt)**2))
    err_geo = np.sqrt(np.mean((ab_geo - ab_gt)**2))
    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_z = np.sqrt(np.mean(ab_gt**2))

    # How well did we replicate the fields?
    r2_f1 = 1 - np.var(true_f1 - pred_f1) / (np.var(true_f1) + 1e-8)
    r2_f2 = 1 - np.var(true_f2 - pred_f2) / (np.var(true_f2) + 1e-8)

    print(f'  {name}: Enc2D={err_enc:.1f} GeoReverse={err_geo:.1f} DDColor={err_dd:.1f} Zero={err_z:.1f} | '
          f'Field R²=[{r2_f1:.3f}, {r2_f2:.3f}]')

    results.append({
        'name': name, 'err_enc': err_enc, 'err_geo': err_geo,
        'err_dd': err_dd, 'err_z': err_z, 'r2_f1': r2_f1, 'r2_f2': r2_f2
    })

    bgr_enc = ab_to_bgr(ab_enc, L)
    bgr_geo = ab_to_bgr(ab_geo, L)
    bgr_dd = ab_to_bgr(ab_dd, L)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    imgs = [
        (gray_bgr, 'Gray'),
        (bgr_geo, f'GeoReverse e={err_geo:.1f}'),
        (bgr_enc, f'Enc2D e={err_enc:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'held_{name}.jpg'), strip)


print('\n=== SUMMARY ===\n')
if results:
    mean_enc = np.mean([r['err_enc'] for r in results])
    mean_geo = np.mean([r['err_geo'] for r in results])
    mean_dd = np.mean([r['err_dd'] for r in results])
    mean_z = np.mean([r['err_z'] for r in results])
    mean_r2_f1 = np.mean([r['r2_f1'] for r in results])
    mean_r2_f2 = np.mean([r['r2_f2'] for r in results])

    print(f'Mean errors (held-out):')
    print(f'  Zero (gray):      {mean_z:.2f}')
    print(f'  GeoReverse:       {mean_geo:.2f} (gap={(1-mean_geo/mean_z)*100:.1f}%)')
    print(f'  Enc 2D:           {mean_enc:.2f} (gap={(1-mean_enc/mean_z)*100:.1f}%)')
    print(f'  DDColor:          {mean_dd:.2f} (gap={(1-mean_dd/mean_z)*100:.1f}%)')
    print(f'  Field replication: R²=[{mean_r2_f1:.3f}, {mean_r2_f2:.3f}]')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
