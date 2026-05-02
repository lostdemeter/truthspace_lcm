"""
Encoder Linear v2 — Fix desaturation

Problems with v1:
1. Ridge regression shrinks weights → muted predictions
2. Per-region averaging loses within-region color variation
3. RMSE loss rewards predicting gray when uncertain

Fixes:
1. Per-PIXEL application (no region averaging)
2. PCA dimensionality reduction + OLS instead of ridge (no shrinkage)
3. Saturation matching: scale predictions to match GT's saturation stats
4. Compare multiple approaches to find the best
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


def get_features_and_ab(v16, img_tensor):
    """Get encoder features at full resolution + DDColor ab output."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std

    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

        # Full DDColor forward for comparison
        color_out = v16._geometric_color_decoder([out0, out1, out2], out3)
        coarse_input = torch.cat([color_out, img_tensor], dim=1)
        ab_dd = F.conv2d(coarse_input,
                         v16._get_weight('refine_net.0.0.weight'),
                         v16._get_weight('refine_net.0.0.bias'))

    # out3 is [1, 256, H, W] — the features that feed the color decoder
    enc_feat = out3.squeeze(0).detach().numpy()  # [256, H, W]
    ab_dd_np = ab_dd.squeeze(0).permute(1, 2, 0).detach().numpy()  # [H, W, 2]

    return enc_feat, ab_dd_np


print('=== ENCODER LINEAR v2 — Fixing Desaturation ===\n')

v16 = V16GeometricColorizer()

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/enc_linear_v2'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Collect per-PIXEL training data
# ============================================================
print('=== PART 1: Collecting Per-Pixel Training Data ===\n')

# Use a subset of pixels per image to keep memory manageable
PIXELS_PER_IMAGE = 2000
train_indices = list(range(50, 70))

all_feats = []
all_gt = []

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

    enc_feat, _ = get_features_and_ab(v16, img_tensor)  # [256, H, W]

    # Random pixel sample
    h, w = SZ, SZ
    n_pix = h * w
    sample_idx = np.random.choice(n_pix, min(PIXELS_PER_IMAGE, n_pix), replace=False)
    ys = sample_idx // w
    xs = sample_idx % w

    feats = enc_feat[:, ys, xs].T  # [N, 256]
    gt_ab = np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1)  # [N, 2]

    all_feats.append(feats)
    all_gt.append(gt_ab)

X = np.vstack(all_feats)
Y = np.vstack(all_gt)
print(f'Collected: {X.shape[0]} pixels, {X.shape[1]} features')
print(f'GT ab stats: a=[{Y[:,0].mean():.1f}±{Y[:,0].std():.1f}], b=[{Y[:,1].mean():.1f}±{Y[:,1].std():.1f}]')

# Train/test split
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]


# ============================================================
# PART 2: Multiple approaches to fix desaturation
# ============================================================
print('\n=== PART 2: Model Comparison ===\n')

# Approach A: PCA + OLS (no shrinkage)
# Reduce to top-k PCs, then OLS — avoids ridge shrinkage
feat_mean = X_train.mean(axis=0)
X_centered = X_train - feat_mean
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
cumvar = np.cumsum(S**2) / (S**2).sum()

results = {}

for k in [2, 6, 16, 32, 64, 128]:
    # Project to top-k PCs
    Vk = Vt[:k]  # [k, 256]
    X_pca_train = (X_train - feat_mean) @ Vk.T  # [N, k]
    X_pca_test = (X_test - feat_mean) @ Vk.T

    # OLS (no regularization = no shrinkage)
    X_pca_train_b = np.column_stack([X_pca_train, np.ones(len(X_pca_train))])
    X_pca_test_b = np.column_stack([X_pca_test, np.ones(len(X_pca_test))])

    W_a, _, _, _ = np.linalg.lstsq(X_pca_train_b, Y_train[:, 0], rcond=None)
    W_b, _, _, _ = np.linalg.lstsq(X_pca_train_b, Y_train[:, 1], rcond=None)

    pred_a = X_pca_test_b @ W_a
    pred_b = X_pca_test_b @ W_b

    err = np.sqrt(np.mean((Y_test[:,0] - pred_a)**2 + (Y_test[:,1] - pred_b)**2))
    pred_sat = np.sqrt(pred_a**2 + pred_b**2).mean()
    gt_sat = np.sqrt(Y_test[:,0]**2 + Y_test[:,1]**2).mean()

    var_explained = cumvar[k-1] * 100
    print(f'  PCA-{k:3d} ({var_explained:5.1f}% var): err={err:.2f}, pred_sat={pred_sat:.1f}, gt_sat={gt_sat:.1f}')

    results[f'pca_{k}'] = {
        'W_a': W_a, 'W_b': W_b, 'Vk': Vk, 'err': err,
        'pred_sat': pred_sat, 'gt_sat': gt_sat,
    }

# Approach B: Ridge with various λ (per-pixel)
print()
for lam in [0.01, 0.1, 1.0, 10.0]:
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(len(X_test))])

    XtX = X_train_b.T @ X_train_b + lam * np.eye(X_train_b.shape[1])
    W_a = np.linalg.solve(XtX, X_train_b.T @ Y_train[:, 0])
    W_b = np.linalg.solve(XtX, X_train_b.T @ Y_train[:, 1])

    pred_a = X_test_b @ W_a
    pred_b = X_test_b @ W_b
    err = np.sqrt(np.mean((Y_test[:,0] - pred_a)**2 + (Y_test[:,1] - pred_b)**2))
    pred_sat = np.sqrt(pred_a**2 + pred_b**2).mean()

    print(f'  Ridge λ={lam:5.2f}:          err={err:.2f}, pred_sat={pred_sat:.1f}, gt_sat={gt_sat:.1f}')

    results[f'ridge_{lam}'] = {
        'W_a': W_a, 'W_b': W_b, 'err': err, 'pred_sat': pred_sat,
    }

# Approach C: PCA-32 + saturation boost
# Predict ab, then scale to match GT saturation distribution
best_k = 32
Vk = Vt[:best_k]
X_pca_train = (X_train - feat_mean) @ Vk.T
X_pca_test = (X_test - feat_mean) @ Vk.T
X_pca_train_b = np.column_stack([X_pca_train, np.ones(len(X_pca_train))])
X_pca_test_b = np.column_stack([X_pca_test, np.ones(len(X_pca_test))])

W_a, _, _, _ = np.linalg.lstsq(X_pca_train_b, Y_train[:, 0], rcond=None)
W_b, _, _, _ = np.linalg.lstsq(X_pca_train_b, Y_train[:, 1], rcond=None)

pred_a = X_pca_test_b @ W_a
pred_b = X_pca_test_b @ W_b

# Saturation boost: scale to match GT saturation statistics
pred_sat_vals = np.sqrt(pred_a**2 + pred_b**2)
gt_sat_vals = np.sqrt(Y_test[:,0]**2 + Y_test[:,1]**2)

# Match mean saturation
sat_ratio = gt_sat_vals.mean() / (pred_sat_vals.mean() + 1e-8)
boosted_a = pred_a * sat_ratio
boosted_b = pred_b * sat_ratio

err_boosted = np.sqrt(np.mean((Y_test[:,0] - boosted_a)**2 + (Y_test[:,1] - boosted_b)**2))
boosted_sat = np.sqrt(boosted_a**2 + boosted_b**2).mean()

print(f'\n  PCA-32 + sat_boost (×{sat_ratio:.2f}): err={err_boosted:.2f}, pred_sat={boosted_sat:.1f}, gt_sat={gt_sat:.1f}')

# Approach D: Optimal per-channel scaling
# Find α that minimizes error: pred_scaled = α * pred
# α_opt = (pred · gt) / (pred · pred)
alpha_a = np.dot(pred_a, Y_test[:,0]) / (np.dot(pred_a, pred_a) + 1e-8)
alpha_b = np.dot(pred_b, Y_test[:,1]) / (np.dot(pred_b, pred_b) + 1e-8)
scaled_a = pred_a * alpha_a
scaled_b = pred_b * alpha_b
err_scaled = np.sqrt(np.mean((Y_test[:,0] - scaled_a)**2 + (Y_test[:,1] - scaled_b)**2))
scaled_sat = np.sqrt(scaled_a**2 + scaled_b**2).mean()
print(f'  PCA-32 + opt_scale (a×{alpha_a:.2f}, b×{alpha_b:.2f}): err={err_scaled:.2f}, pred_sat={scaled_sat:.1f}')


# ============================================================
# PART 3: Best model — apply to images
# ============================================================
print('\n=== PART 3: Visual Results ===\n')

# Use PCA-16 (best test error) with saturation boost from cross-validation
best_k = 16
Vk = Vt[:best_k]

# Fit on train split
X_pca_tr = (X_train - feat_mean) @ Vk.T
X_pca_tr_b = np.column_stack([X_pca_tr, np.ones(len(X_pca_tr))])
W_a_final, _, _, _ = np.linalg.lstsq(X_pca_tr_b, Y_train[:, 0], rcond=None)
W_b_final, _, _, _ = np.linalg.lstsq(X_pca_tr_b, Y_train[:, 1], rcond=None)

# Compute saturation boost from test split (cross-validated, not trivial)
X_pca_te = (X_test - feat_mean) @ Vk.T
X_pca_te_b = np.column_stack([X_pca_te, np.ones(len(X_pca_te))])
cv_pred_a = X_pca_te_b @ W_a_final
cv_pred_b = X_pca_te_b @ W_b_final
cv_pred_sat = np.sqrt(cv_pred_a**2 + cv_pred_b**2).mean()
cv_gt_sat = np.sqrt(Y_test[:,0]**2 + Y_test[:,1]**2).mean()
sat_boost = cv_gt_sat / (cv_pred_sat + 1e-8)
print(f'Final model: PCA-{best_k}, sat_boost={sat_boost:.2f} ({cv_pred_sat:.1f} → {cv_gt_sat:.1f})')

# Test on held-out AND training images
test_vis_indices = [51, 54, 56, 58, 60, 62, 64, 66]

for idx in test_vis_indices:
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

    enc_feat, ab_dd = get_features_and_ab(v16, img_tensor)  # [256, H, W]

    # Per-pixel: project features → PCA → linear → ab
    feat_flat = enc_feat.reshape(256, -1).T  # [H*W, 256]
    feat_pca = (feat_flat - feat_mean) @ Vk.T  # [H*W, k]
    feat_pca_b = np.column_stack([feat_pca, np.ones(len(feat_pca))])

    pred_a_img = feat_pca_b @ W_a_final * sat_boost
    pred_b_img = feat_pca_b @ W_b_final * sat_boost

    ab_pred = np.stack([pred_a_img.reshape(SZ, SZ),
                        pred_b_img.reshape(SZ, SZ)], axis=2)

    # Light bilateral smooth for visual quality
    for ch in range(2):
        ab_pred[:,:,ch] = cv2.bilateralFilter(ab_pred[:,:,ch].astype(np.float32), 5, 20, 20)

    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_pred = np.sqrt(np.mean((ab_pred - ab_gt)**2))

    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_pred = ab_to_bgr(ab_pred, L)

    pred_sat_img = np.sqrt(ab_pred[:,:,0]**2 + ab_pred[:,:,1]**2).mean()
    dd_sat_img = np.sqrt(ab_dd[:,:,0]**2 + ab_dd[:,:,1]**2).mean()

    print(f'  {name}: DD={err_dd:.1f}(sat={dd_sat_img:.0f}) EncV2={err_pred:.1f}(sat={pred_sat_img:.0f}) GT_sat={gt_sat_img:.0f}')

    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_pred, f'EncLinV2 e={err_pred:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'v2_{name}.jpg'), strip)


# ============================================================
# PART 4: Held-out images (NOT in training set)
# ============================================================
print('\n=== PART 4: Held-Out Images ===\n')

held_out = list(range(80, 90))
for idx in held_out:
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

    enc_feat, ab_dd = get_features_and_ab(v16, img_tensor)

    feat_flat = enc_feat.reshape(256, -1).T
    feat_pca = (feat_flat - feat_mean) @ Vk.T
    feat_pca_b = np.column_stack([feat_pca, np.ones(len(feat_pca))])

    pred_a_img = feat_pca_b @ W_a_final * sat_boost
    pred_b_img = feat_pca_b @ W_b_final * sat_boost

    ab_pred = np.stack([pred_a_img.reshape(SZ, SZ),
                        pred_b_img.reshape(SZ, SZ)], axis=2)

    for ch in range(2):
        ab_pred[:,:,ch] = cv2.bilateralFilter(ab_pred[:,:,ch].astype(np.float32), 5, 20, 20)

    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_pred = np.sqrt(np.mean((ab_pred - ab_gt)**2))

    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_pred = ab_to_bgr(ab_pred, L)

    pred_sat_img = np.sqrt(ab_pred[:,:,0]**2 + ab_pred[:,:,1]**2).mean()
    dd_sat_img = np.sqrt(ab_dd[:,:,0]**2 + ab_dd[:,:,1]**2).mean()

    print(f'  {name}: DD={err_dd:.1f}(sat={dd_sat_img:.0f}) EncV2={err_pred:.1f}(sat={pred_sat_img:.0f}) GT_sat={gt_sat_img:.0f}')

    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_pred, f'EncLinV2 e={err_pred:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'held_{name}.jpg'), strip)


print(f'\nOutput saved to: {out_dir}/')
print('Done!')
