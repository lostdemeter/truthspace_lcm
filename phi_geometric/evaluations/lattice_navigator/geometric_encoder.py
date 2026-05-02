"""
Geometric Encoder: Replace ConvNeXt with geometric operations

The anatomy study found: handcrafted features get 14.7% gap closure
vs encoder's 24.5%. The missing 10% likely comes from:
1. Global context (CNN receptive field = whole image)
2. Nonlinear feature interactions
3. Multi-scale context POOLING (not just local statistics)

Strategy:
- Build φ-scaled multi-scale pyramid (scales at φ^n)
- At each scale, compute: mean, variance, gradient magnitude, Gabor energy
- Pool context from increasingly large neighborhoods
- Add nonlinear interactions between scales
- Test end-to-end: geometric features → linear → color
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch

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
    return out3.squeeze(0).detach().numpy()


def phi_pyramid_features(gray, n_scales=8):
    """
    Build a φ-scaled feature pyramid.

    At each scale σ = φ^k, compute local statistics.
    This mimics the CNN's multi-scale receptive field with explicit geometry.
    The key insight: CNNs build features hierarchically by pooling local
    features into progressively larger receptive fields. We do the same
    with explicit Gaussian pooling at φ-spaced scales.
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32) / 255.0
    features = []
    feat_names = []

    # φ-scaled sigmas: 1, φ, φ², φ³, ...
    sigmas = [PHI**k for k in range(n_scales)]

    # === Layer 1: Per-scale local statistics ===
    blurred = {}
    for i, sigma in enumerate(sigmas):
        # Clamp kernel size to reasonable range
        ksize = int(sigma * 6) | 1  # ensure odd
        if ksize > 255: ksize = 255
        if ksize < 3: ksize = 3

        blur = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
        blurred[i] = blur

        # Mean at this scale
        features.append(blur)
        feat_names.append(f'mean_φ{i}')

        # Local contrast (pixel minus local mean)
        contrast = gray_f - blur
        features.append(contrast)
        feat_names.append(f'contrast_φ{i}')

        # Local variance at this scale
        blur_sq = cv2.GaussianBlur(gray_f**2, (ksize, ksize), sigma)
        var = np.sqrt(np.maximum(blur_sq - blur**2, 0))
        features.append(var)
        feat_names.append(f'var_φ{i}')

        # Edge energy at this scale
        sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx**2 + sy**2)
        features.append(edge)
        feat_names.append(f'edge_φ{i}')

    # === Layer 2: Cross-scale interactions (the CNN's depth) ===
    # Difference of Gaussians between consecutive φ-scales
    for i in range(len(sigmas) - 1):
        dog = blurred[i] - blurred[i+1]
        features.append(dog)
        feat_names.append(f'dog_φ{i}_{i+1}')

    # Ratio of variances between scales (texture scale signature)
    for i in range(len(sigmas) - 1):
        v1 = features[feat_names.index(f'var_φ{i}')]
        v2 = features[feat_names.index(f'var_φ{i+1}')]
        ratio = v1 / (v2 + 1e-8)
        features.append(ratio)
        feat_names.append(f'varratio_φ{i}_{i+1}')

    # === Layer 3: Gabor energy at φ-scales (material detection) ===
    for i, sigma in enumerate(sigmas[:5]):  # first 5 scales
        for theta_deg in [0, 45, 90, 135]:
            theta = np.radians(theta_deg)
            lam = sigma * 2
            gksize = int(sigma * 8) | 1
            if gksize < 3: gksize = 3
            if gksize > 31: gksize = 31
            kern = cv2.getGaborKernel((gksize, gksize), sigma=sigma,
                                       theta=theta, lambd=lam, gamma=0.5, psi=0)
            resp = cv2.filter2D(gray_f, cv2.CV_32F, kern)
            features.append(np.abs(resp))
            feat_names.append(f'gabor_{theta_deg}_φ{i}')

    # Total Gabor energy per scale (orientation-invariant texture)
    for i, sigma in enumerate(sigmas[:5]):
        energies = []
        for theta_deg in [0, 45, 90, 135]:
            idx = feat_names.index(f'gabor_{theta_deg}_φ{i}')
            energies.append(features[idx])
        total_energy = np.sqrt(sum(e**2 for e in energies))
        features.append(total_energy)
        feat_names.append(f'gabor_energy_φ{i}')

    # === Layer 4: Global context (what CNN's deep layers provide) ===
    # Image-level statistics broadcast to every pixel
    global_mean = gray_f.mean()
    global_std = gray_f.std()
    features.append(np.full_like(gray_f, global_mean))
    feat_names.append('global_mean')
    features.append(np.full_like(gray_f, global_std))
    feat_names.append('global_std')

    # Relative brightness (how this pixel compares to the whole image)
    features.append(gray_f - global_mean)
    feat_names.append('relative_bright')

    # Position encoding
    ys = np.linspace(0, 1, h).reshape(-1, 1) * np.ones((1, w))
    xs = np.linspace(0, 1, w).reshape(1, -1) * np.ones((h, 1))
    features.append(ys.astype(np.float32))
    feat_names.append('y_pos')
    features.append(xs.astype(np.float32))
    feat_names.append('x_pos')

    # === Layer 5: CLAHE (local adaptive equalization — powerful texture signal) ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray).astype(np.float32) / 255.0
    features.append(cl)
    feat_names.append('clahe')
    features.append(cl - gray_f)
    feat_names.append('clahe_diff')

    # === Layer 6: Nonlinear interactions (what CNN ReLUs provide) ===
    # These are the KEY features that linear models miss

    # Brightness × position (scene semantics)
    features.append(gray_f * ys.astype(np.float32))
    feat_names.append('bright_x_ypos')
    features.append(gray_f * (1 - ys).astype(np.float32))
    feat_names.append('bright_x_top')

    # Texture × brightness (material in context)
    var4_idx = feat_names.index('var_φ2')  # φ² ≈ 2.6
    features.append(features[var4_idx] * gray_f)
    feat_names.append('texture_x_bright')

    # Edge × contrast (boundary salience)
    edge0_idx = feat_names.index('edge_φ0')
    contrast2_idx = feat_names.index('contrast_φ2')
    features.append(features[edge0_idx] * np.abs(features[contrast2_idx]))
    feat_names.append('edge_x_contrast')

    # Large-scale context × local detail (semantic binding)
    mean5_idx = feat_names.index('mean_φ5')
    contrast1_idx = feat_names.index('contrast_φ1')
    features.append(features[mean5_idx] * features[contrast1_idx])
    feat_names.append('context_x_detail')

    # Gabor energy × position (oriented texture in scene context)
    ge2_idx = feat_names.index('gabor_energy_φ2')
    features.append(features[ge2_idx] * ys.astype(np.float32))
    feat_names.append('gabor_x_ypos')

    # === Layer 7: Quadrant context pooling ===
    # Approximate CNN's large receptive field by pooling to quadrants
    # then broadcasting back — each pixel knows its quadrant's character
    q_h, q_w = h // 4, w // 4
    for fname_src in ['mean_φ0', 'var_φ2', 'edge_φ1', 'clahe']:
        src_idx = feat_names.index(fname_src)
        src = features[src_idx]
        # Pool to 4x4 grid then upsample
        pooled = cv2.resize(src, (4, 4), interpolation=cv2.INTER_AREA)
        broadcast = cv2.resize(pooled, (w, h), interpolation=cv2.INTER_LINEAR)
        features.append(broadcast)
        feat_names.append(f'quad_{fname_src}')
        # Also: pixel minus its quadrant context
        features.append(src - broadcast)
        feat_names.append(f'quad_diff_{fname_src}')

    return np.array(features), feat_names


print('=== GEOMETRIC ENCODER ===\n')

v16 = V16GeometricColorizer()
SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
train_indices = list(range(50, 75))
test_indices = list(range(75, 90))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/geometric_encoder'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Collect training data (per-pixel)
# ============================================================
print('=== PART 1: Collecting Training Data ===\n')

PIXELS_PER_IMAGE = 2000

train_X = []
train_Y = []

for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    geo_feats, feat_names = phi_pyramid_features(gray)
    n_feat = len(geo_feats)

    sample = np.random.choice(SZ*SZ, min(PIXELS_PER_IMAGE, SZ*SZ), replace=False)
    ys = sample // SZ
    xs = sample % SZ

    X = geo_feats[:, ys, xs].T  # [N, n_feat]
    Y = np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1)

    train_X.append(X)
    train_Y.append(Y)

train_X = np.vstack(train_X)
train_Y = np.vstack(train_Y)
print(f'Training: {train_X.shape[0]} pixels, {train_X.shape[1]} features ({len(feat_names)} named)')


# ============================================================
# PART 2: Fit the geometric encoder (ridge regression)
# ============================================================
print('\n=== PART 2: Fitting Geometric Encoder ===\n')

# Add bias
X_b = np.column_stack([train_X, np.ones(len(train_X))])

# Step 1: STANDARDIZE features (z-score) so variance ratios don't dominate
feat_mean = train_X.mean(axis=0)
feat_std = train_X.std(axis=0)
feat_std[feat_std < 1e-8] = 1.0  # avoid division by zero
X_normed = (train_X - feat_mean) / feat_std

print(f'After standardization: all features have unit variance')

# Use 70/30 internal split for hyperparameter selection
split = int(len(X_normed) * 0.7)
X_tr, X_val = X_normed[:split], X_normed[split:]
Y_tr, Y_val = train_Y[:split], train_Y[split:]
err_zero = np.sqrt(np.mean(Y_val[:,0]**2 + Y_val[:,1]**2))

# Step 2: Supervised feature directions via cross-covariance SVD
# This finds the directions in feature space most predictive of color
# (Partial Least Squares / Canonical Correlation Analysis idea)
C = X_tr.T @ np.column_stack([Y_tr[:, 0], Y_tr[:, 1]]) / len(X_tr)  # [92, 2]
U_sup, S_sup, Vt_sup = np.linalg.svd(C, full_matrices=False)
print(f'\nSupervised directions SVD: S = [{S_sup[0]:.4f}, {S_sup[1]:.4f}]')
print(f'  S[0]/S[1] = {S_sup[0]/S_sup[1]:.4f} (φ={PHI:.4f}, err={abs(S_sup[0]/S_sup[1]-PHI)/PHI*100:.1f}%)')

# Test: use supervised directions vs PCA vs ridge
print(f'\nApproach comparison:')

# A: Supervised directions (top-k from cross-covariance SVD)
for k in [2, 4, 8, 16, 32]:
    Vk = U_sup[:, :min(k, 2)]  # Only 2 supervised directions possible (2 targets)
    # Augment with PCA of residuals for k > 2
    if k > 2:
        U_pca, S_pca, Vt_pca = np.linalg.svd(X_tr, full_matrices=False)
        Vk_pca = Vt_pca[:k-2].T
        Vk = np.column_stack([Vk, Vk_pca])
    
    X_proj_tr = X_tr @ Vk
    X_proj_val = X_val @ Vk
    X_proj_tr_b = np.column_stack([X_proj_tr, np.ones(len(X_proj_tr))])
    X_proj_val_b = np.column_stack([X_proj_val, np.ones(len(X_proj_val))])
    
    W_a, _, _, _ = np.linalg.lstsq(X_proj_tr_b, Y_tr[:, 0], rcond=None)
    W_b, _, _, _ = np.linalg.lstsq(X_proj_tr_b, Y_tr[:, 1], rcond=None)
    pred_a = np.clip(X_proj_val_b @ W_a, -50, 50)
    pred_b = np.clip(X_proj_val_b @ W_b, -50, 50)
    err = np.sqrt(np.mean((Y_val[:,0] - pred_a)**2 + (Y_val[:,1] - pred_b)**2))
    gap = (1 - err / err_zero) * 100
    pred_sat = np.sqrt(pred_a**2 + pred_b**2).mean()
    gt_sat_v = np.sqrt(Y_val[:,0]**2 + Y_val[:,1]**2).mean()
    print(f'  Supervised-{k:2d}: err={err:.2f}, gap={gap:.1f}%, sat={pred_sat:.1f}/{gt_sat_v:.1f}')

# B: Ridge on standardized features (various λ)
for lam in [0.1, 1.0, 10.0, 100.0]:
    X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_val_b = np.column_stack([X_val, np.ones(len(X_val))])
    XtX = X_tr_b.T @ X_tr_b + lam * np.eye(X_tr_b.shape[1])
    W_a = np.linalg.solve(XtX, X_tr_b.T @ Y_tr[:, 0])
    W_b = np.linalg.solve(XtX, X_tr_b.T @ Y_tr[:, 1])
    pred_a = np.clip(X_val_b @ W_a, -50, 50)
    pred_b = np.clip(X_val_b @ W_b, -50, 50)
    err = np.sqrt(np.mean((Y_val[:,0] - pred_a)**2 + (Y_val[:,1] - pred_b)**2))
    gap = (1 - err / err_zero) * 100
    pred_sat = np.sqrt(pred_a**2 + pred_b**2).mean()
    print(f'  Ridge λ={lam:5.1f}:   err={err:.2f}, gap={gap:.1f}%, sat={pred_sat:.1f}/{gt_sat_v:.1f}')

# Pick best approach — use standardized ridge with moderate λ
best_lam = 10.0
X_full_normed = X_normed
X_full_b = np.column_stack([X_full_normed, np.ones(len(X_full_normed))])
XtX = X_full_b.T @ X_full_b + best_lam * np.eye(X_full_b.shape[1])
W_a_final = np.linalg.solve(XtX, X_full_b.T @ train_Y[:, 0])
W_b_final = np.linalg.solve(XtX, X_full_b.T @ train_Y[:, 1])

# Compute saturation boost from validation
X_val_b = np.column_stack([X_val, np.ones(len(X_val))])
pred_a_val = np.clip(X_val_b @ W_a_final, -50, 50)
pred_b_val = np.clip(X_val_b @ W_b_final, -50, 50)
pred_sat_val = np.sqrt(pred_a_val**2 + pred_b_val**2).mean()
gt_sat_val = np.sqrt(Y_val[:,0]**2 + Y_val[:,1]**2).mean()
sat_boost = gt_sat_val / (pred_sat_val + 1e-8)
print(f'\nFinal model: Ridge λ={best_lam}, sat_boost={sat_boost:.2f}')

# Store for use in projection
Vk_final = None  # signal to use raw standardized features


# ============================================================
# PART 3: Feature importance
# ============================================================
print('\n=== PART 3: Feature Importance ===\n')

# Feature importance from ridge weights (already standardized so directly comparable)
W_combined = np.sqrt(W_a_final[:-1]**2 + W_b_final[:-1]**2)
sorted_imp = np.argsort(W_combined)[::-1]
print(f'  {"Feature":<30} {"Importance":>10} {"W_a":>10} {"W_b":>10}')
print(f'  {"-"*62}')
for i in sorted_imp[:20]:
    print(f'  {feat_names[i]:<30} {W_combined[i]:10.4f} {W_a_final[i]:10.4f} {W_b_final[i]:10.4f}')

print(f'\nSupervised direction analysis:')
print(f'  Direction 1 (S={S_sup[0]:.4f}) top features:')
for i in np.argsort(np.abs(U_sup[:, 0]))[::-1][:8]:
    print(f'    {feat_names[i]:<25} loading={U_sup[i, 0]:+.3f}')
print(f'  Direction 2 (S={S_sup[1]:.4f}) top features:')
for i in np.argsort(np.abs(U_sup[:, 1]))[::-1][:8]:
    print(f'    {feat_names[i]:<25} loading={U_sup[i, 1]:+.3f}')


# ============================================================
# PART 4: Test on held-out images
# ============================================================
print('\n=== PART 4: Held-Out Image Test ===\n')

results = []

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

    # Geometric encoder
    geo_feats, _ = phi_pyramid_features(gray)
    feat_flat = geo_feats.reshape(len(geo_feats), -1).T  # [H*W, n_feat]
    feat_normed = (feat_flat - feat_mean) / feat_std
    feat_b = np.column_stack([feat_normed, np.ones(len(feat_normed))])

    pred_a = np.clip(feat_b @ W_a_final * sat_boost, -50, 50)
    pred_b_ch = np.clip(feat_b @ W_b_final * sat_boost, -50, 50)
    ab_geo = np.stack([pred_a.reshape(SZ, SZ), pred_b_ch.reshape(SZ, SZ)], axis=2)

    for ch in range(2):
        ab_geo[:,:,ch] = cv2.bilateralFilter(ab_geo[:,:,ch].astype(np.float32), 5, 20, 20)

    # DDColor for comparison
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    # Encoder linear for comparison
    enc_feat = get_encoder_features(v16, img_tensor)
    # (We'd need the PCA basis from encoder_linear_v2 — skip for now, compare geo vs DD)

    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_geo = np.sqrt(np.mean((ab_geo - ab_gt)**2))
    err_z = np.sqrt(np.mean(ab_gt**2))

    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_geo = ab_to_bgr(ab_geo, L)

    geo_sat = np.sqrt(ab_geo[:,:,0]**2 + ab_geo[:,:,1]**2).mean()
    dd_sat = np.sqrt(ab_dd[:,:,0]**2 + ab_dd[:,:,1]**2).mean()

    gap_geo = (1 - err_geo / err_z) * 100
    gap_dd = (1 - err_dd / err_z) * 100

    winner = 'GEO' if err_geo < err_dd else 'DD'
    print(f'  {name}: Geo={err_geo:.1f}({gap_geo:.0f}%) DD={err_dd:.1f}({gap_dd:.0f}%) [{winner}] sat: GT={gt_sat_img:.0f} Geo={geo_sat:.0f} DD={dd_sat:.0f}')

    results.append({'name': name, 'err_geo': err_geo, 'err_dd': err_dd, 'err_z': err_z})

    # Save comparison
    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_geo, f'GeoEnc e={err_geo:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'geo_{name}.jpg'), strip)


# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY ===\n')

mean_geo = np.mean([r['err_geo'] for r in results])
mean_dd = np.mean([r['err_dd'] for r in results])
mean_z = np.mean([r['err_z'] for r in results])
geo_wins = sum(1 for r in results if r['err_geo'] < r['err_dd'])

print(f'Held-out images: {len(results)}')
print(f'Mean error: Geometric={mean_geo:.2f}, DDColor={mean_dd:.2f}, Zero={mean_z:.2f}')
print(f'Gap closed: Geometric={(1-mean_geo/mean_z)*100:.1f}%, DDColor={(1-mean_dd/mean_z)*100:.1f}%')
print(f'Geometric wins: {geo_wins}/{len(results)}')
print(f'\nThe geometric encoder uses {len(feat_names)} handcrafted features')
print(f'with φ-scaled multi-scale pyramid + nonlinear interactions.')
print(f'No learned parameters except the linear projection.')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
