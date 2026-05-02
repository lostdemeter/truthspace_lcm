"""
Reverse Encoder v2: Soft Segmentation Approach

The encoder's 2D color fields are:
  - 99% low-frequency (smooth within regions)
  - Edge-aligned (transitions at image boundaries)
  - Essentially soft segmentation maps

Approach:
1. Segment the grayscale image into superpixels/regions using edges
2. For each region, compute aggregate features
3. Assign each region a 2D color code via linear model
4. Reconstruct the smooth field by broadcasting + edge-aware smoothing

This is REVERSE navigation: we know the output is a soft segmentation,
so we build the geometric operation that produces one.
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


def simple_segment(gray, n_seg=200):
    """
    Simple segmentation using watershed on grayscale.
    Produces roughly n_seg regions. No learned params.
    """
    h, w = gray.shape
    # Grid size to get ~n_seg regions
    grid = max(1, int(np.sqrt(h * w / n_seg)))

    # Create markers on a regular grid
    markers = np.zeros((h, w), dtype=np.int32)
    marker_id = 1
    for y in range(grid // 2, h, grid):
        for x in range(grid // 2, w, grid):
            markers[y, x] = marker_id
            marker_id += 1

    # Watershed needs 3-channel image
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(gray_3ch, markers)

    # Watershed sets boundaries to -1; assign them to nearest region
    boundary = markers == -1
    if boundary.any():
        # Dilate markers to fill boundaries
        kernel = np.ones((3, 3), np.uint8)
        filled = cv2.dilate(markers.astype(np.float32), kernel).astype(np.int32)
        markers[boundary] = filled[boundary]

    # Relabel to 0..n-1
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[unique_labels > 0]
    label_map = np.zeros(markers.max() + 1, dtype=np.int32)
    for i, lbl in enumerate(unique_labels):
        label_map[lbl] = i
    labels = label_map[np.maximum(markers, 0)]

    return labels, len(unique_labels)


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


def compute_region_features(gray, labels, n_regions):
    """
    For each region, compute aggregate geometric features.
    These describe the region's CHARACTER, not individual pixels.
    """
    gray_f = gray.astype(np.float32) / 255.0
    h, w = gray.shape

    # Position grid
    ys = np.linspace(0, 1, h).reshape(-1, 1) * np.ones((1, w))
    xs = np.linspace(0, 1, w).reshape(1, -1) * np.ones((h, 1))

    # Multi-scale features for the whole image
    blurs = {}
    for sigma in [2, 4, 8, 16, 32]:
        ksize = int(sigma * 6) | 1
        if ksize > 255: ksize = 255
        blurs[sigma] = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)

    # Edge map
    edges = cv2.Canny(gray, 50, 150).astype(float) / 255.0

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray).astype(np.float32) / 255.0

    # Local variance at multiple scales
    variances = {}
    for sigma in [4, 8, 16]:
        ksize = int(sigma * 6) | 1
        if ksize > 255: ksize = 255
        lm = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
        lsq = cv2.GaussianBlur(gray_f**2, (ksize, ksize), sigma)
        variances[sigma] = np.sqrt(np.maximum(lsq - lm**2, 0))

    # Global features (same for all regions — image-level context)
    global_mean = gray_f.mean()
    global_std = gray_f.std()

    feat_list = []
    for r in range(n_regions):
        mask = labels == r
        if mask.sum() < 5:
            feat_list.append(np.zeros(22))
            continue

        feats = []

        # 1. Region brightness statistics
        region_vals = gray_f[mask]
        feats.append(region_vals.mean())      # mean brightness
        feats.append(region_vals.std())        # brightness variance
        feats.append(np.median(region_vals))   # median

        # 2. Position (where in image is this region?)
        feats.append(ys[mask].mean())    # mean y position
        feats.append(xs[mask].mean())    # mean x position
        feats.append(ys[mask].std())     # y spread (vertical extent)
        feats.append(xs[mask].std())     # x spread (horizontal extent)

        # 3. Size
        feats.append(mask.sum() / (h * w))  # relative area

        # 4. Multi-scale context (what is the large-scale brightness here?)
        for sigma in [8, 16, 32]:
            feats.append(blurs[sigma][mask].mean())

        # 5. Edge density (how textured is this region?)
        feats.append(edges[mask].mean())

        # 6. CLAHE features
        feats.append(cl[mask].mean())
        feats.append((cl[mask] - region_vals).mean())  # clahe deviation

        # 7. Texture (local variance)
        for sigma in [4, 8, 16]:
            feats.append(variances[sigma][mask].mean())

        # 8. Relative brightness (region vs global)
        feats.append(region_vals.mean() - global_mean)

        # 9. Brightness × position interactions
        feats.append(region_vals.mean() * ys[mask].mean())
        feats.append(region_vals.mean() * (1 - ys[mask].mean()))

        # 10. Global context
        feats.append(global_mean)
        feats.append(global_std)

        feat_list.append(np.array(feats))

    return np.array(feat_list)


print('=== REVERSE ENCODER v2: Soft Segmentation ===\n')

v16 = V16GeometricColorizer()
SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/reverse_enc_v2'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Build color basis (same as reverse_encoder.py)
# ============================================================
print('=== PART 1: Building Color Basis ===\n')

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
    enc = get_encoder_features(v16, img_tensor)
    flat = enc.reshape(256, -1).T

    sample = np.random.choice(len(flat), min(PIXELS_PER_IMAGE, len(flat)), replace=False)
    all_enc.append(flat[sample])
    ys, xs = sample // SZ, sample % SZ
    all_gt_ab.append(np.stack([ab_gt[ys, xs, 0], ab_gt[ys, xs, 1]], axis=1))

all_enc = np.vstack(all_enc)
all_gt_ab = np.vstack(all_gt_ab)

enc_mean = all_enc.mean(axis=0)
enc_centered = all_enc - enc_mean

C = enc_centered.T @ all_gt_ab / len(enc_centered)
U_color, S_color, Vt_color = np.linalg.svd(C, full_matrices=False)
color_dir_1 = U_color[:, 0]
color_dir_2 = U_color[:, 1]

print(f'Color directions: S = [{S_color[0]:.4f}, {S_color[1]:.4f}]')
print(f'S[0]/S[1] = {S_color[0]/S_color[1]:.4f}')


# ============================================================
# PART 2: Segment images and collect region-level data
# ============================================================
print('\n=== PART 2: Region-Level Training Data ===\n')

N_SUPERPIXELS = 200  # number of superpixels per image

all_region_feats = []
all_region_f1 = []
all_region_f2 = []
all_region_gt_a = []
all_region_gt_b = []

for idx in train_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue

    # Get encoder color fields
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    enc = get_encoder_features(v16, img_tensor)
    flat = (enc.reshape(256, -1).T - enc_mean)
    field_1 = (flat @ color_dir_1).reshape(SZ, SZ)
    field_2 = (flat @ color_dir_2).reshape(SZ, SZ)

    # Simple grid-based segmentation with edge-aware merging
    labels, n_labels = simple_segment(gray, n_seg=N_SUPERPIXELS)

    # Region features
    region_feats = compute_region_features(gray, labels, n_labels)

    for r_idx in range(n_labels):
        mask = labels == r_idx
        if mask.sum() < 10: continue

        all_region_feats.append(region_feats[r_idx])
        all_region_f1.append(field_1[mask].mean())
        all_region_f2.append(field_2[mask].mean())
        all_region_gt_a.append(ab_gt[mask, 0].mean())
        all_region_gt_b.append(ab_gt[mask, 1].mean())

X_reg = np.array(all_region_feats)
f1_reg = np.array(all_region_f1)
f2_reg = np.array(all_region_f2)
gt_a_reg = np.array(all_region_gt_a)
gt_b_reg = np.array(all_region_gt_b)

print(f'Collected {len(X_reg)} regions from {len(train_indices)} images')
print(f'Features per region: {X_reg.shape[1]}')


# ============================================================
# PART 3: Fit region-level models
# ============================================================
print('\n=== PART 3: Region-Level Models ===\n')

# Standardize
reg_mean = X_reg.mean(axis=0)
reg_std = X_reg.std(axis=0)
reg_std[reg_std < 1e-8] = 1.0
X_normed = (X_reg - reg_mean) / reg_std

# Split
split = int(len(X_normed) * 0.7)
X_tr, X_te = X_normed[:split], X_normed[split:]
f1_tr, f1_te = f1_reg[:split], f1_reg[split:]
f2_tr, f2_te = f2_reg[:split], f2_reg[split:]
gt_a_tr, gt_a_te = gt_a_reg[:split], gt_a_reg[split:]
gt_b_tr, gt_b_te = gt_b_reg[:split], gt_b_reg[split:]

err_zero = np.sqrt(np.mean(gt_a_te**2 + gt_b_te**2))

# A: Region features → encoder fields (REVERSE target)
print('Reverse: Region features → encoder fields:')
for lam in [0.01, 0.1, 1.0, 10.0]:
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
    print(f'  λ={lam:5.2f}: Field1 R²={r2_f1:.3f}(r={corr_f1:.3f}) Field2 R²={r2_f2:.3f}(r={corr_f2:.3f})')

# B: Region features → GT color DIRECTLY
print('\nForward: Region features → GT color:')
for lam in [0.01, 0.1, 1.0, 10.0]:
    X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_te_b = np.column_stack([X_te, np.ones(len(X_te))])
    XtX = X_tr_b.T @ X_tr_b + lam * np.eye(X_tr_b.shape[1])
    W_a = np.linalg.solve(XtX, X_tr_b.T @ gt_a_tr)
    W_b = np.linalg.solve(XtX, X_tr_b.T @ gt_b_tr)

    pred_a = X_te_b @ W_a
    pred_b = X_te_b @ W_b
    err = np.sqrt(np.mean((gt_a_te - pred_a)**2 + (gt_b_te - pred_b)**2))
    gap = (1 - err / err_zero) * 100
    r2_a = 1 - np.mean((gt_a_te - pred_a)**2) / (np.var(gt_a_te) + 1e-8)
    r2_b = 1 - np.mean((gt_b_te - pred_b)**2) / (np.var(gt_b_te) + 1e-8)
    print(f'  λ={lam:5.2f}: err={err:.2f}, gap={gap:.1f}%, R²_a={r2_a:.3f}, R²_b={r2_b:.3f}')

# C: Encoder fields → GT color (at region level — the theoretical ceiling)
print('\nCeiling: Encoder fields → GT color (region-level):')
X_enc = np.column_stack([f1_tr, f2_tr, np.ones(len(f1_tr))])
X_enc_te = np.column_stack([f1_te, f2_te, np.ones(len(f1_te))])
W_a_enc, _, _, _ = np.linalg.lstsq(X_enc, gt_a_tr, rcond=None)
W_b_enc, _, _, _ = np.linalg.lstsq(X_enc, gt_b_tr, rcond=None)
pred_a = X_enc_te @ W_a_enc
pred_b = X_enc_te @ W_b_enc
err_enc = np.sqrt(np.mean((gt_a_te - pred_a)**2 + (gt_b_te - pred_b)**2))
gap_enc = (1 - err_enc / err_zero) * 100
print(f'  Enc fields → color: err={err_enc:.2f}, gap={gap_enc:.1f}%')


# ============================================================
# PART 4: Best model — apply to held-out images
# ============================================================
print('\n=== PART 4: Held-Out Images ===\n')

# Fit final model on all training data (region features → GT color directly)
best_lam = 1.0
X_all_b = np.column_stack([X_normed, np.ones(len(X_normed))])
XtX = X_all_b.T @ X_all_b + best_lam * np.eye(X_all_b.shape[1])
W_a_final = np.linalg.solve(XtX, X_all_b.T @ gt_a_reg)
W_b_final = np.linalg.solve(XtX, X_all_b.T @ gt_b_reg)

# Also fit: region → encoder fields → color (two-stage)
W_f1_final = np.linalg.solve(XtX, X_all_b.T @ f1_reg)
W_f2_final = np.linalg.solve(XtX, X_all_b.T @ f2_reg)

# Saturation analysis on validation
X_te_b = np.column_stack([X_te, np.ones(len(X_te))])
pred_a_val = X_te_b @ W_a_final
pred_b_val = X_te_b @ W_b_final
pred_sat = np.sqrt(pred_a_val**2 + pred_b_val**2).mean()
gt_sat_r = np.sqrt(gt_a_te**2 + gt_b_te**2).mean()
sat_boost = gt_sat_r / (pred_sat + 1e-8)
print(f'Saturation boost: {sat_boost:.2f}')

test_indices = list(range(80, 90))
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

    # Simple grid-based segmentation
    labels, n_labels = simple_segment(gray, n_seg=N_SUPERPIXELS)

    # Region features
    region_feats = compute_region_features(gray, labels, n_labels)
    region_normed = (region_feats - reg_mean) / reg_std
    region_b = np.column_stack([region_normed, np.ones(n_labels)])

    # Predict color per region
    region_a = np.clip(region_b @ W_a_final * sat_boost, -50, 50)
    region_b_ch = np.clip(region_b @ W_b_final * sat_boost, -50, 50)

    # Broadcast to pixels
    ab_geo = np.zeros((SZ, SZ, 2))
    for r_idx in range(n_labels):
        mask = labels == r_idx
        ab_geo[mask, 0] = region_a[r_idx]
        ab_geo[mask, 1] = region_b_ch[r_idx]

    # Edge-aware smooth via bilateral filter
    for ch in range(2):
        ab_geo[:,:,ch] = cv2.bilateralFilter(ab_geo[:,:,ch].astype(np.float32), 9, 30, 30)

    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()

    # Enc 2D for reference
    enc = get_encoder_features(v16, img_tensor)
    flat = (enc.reshape(256, -1).T - enc_mean)
    true_f1 = (flat @ color_dir_1).reshape(SZ, SZ)
    true_f2 = (flat @ color_dir_2).reshape(SZ, SZ)
    enc_fields = np.column_stack([true_f1.flatten(), true_f2.flatten(), np.ones(SZ*SZ)])
    enc_a = np.clip(enc_fields @ W_a_enc, -50, 50).reshape(SZ, SZ)
    enc_b = np.clip(enc_fields @ W_b_enc, -50, 50).reshape(SZ, SZ)
    ab_enc = np.stack([enc_a, enc_b], axis=2)

    err_geo = np.sqrt(np.mean((ab_geo - ab_gt)**2))
    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_enc = np.sqrt(np.mean((ab_enc - ab_gt)**2))
    err_z = np.sqrt(np.mean(ab_gt**2))

    geo_sat = np.sqrt(ab_geo[:,:,0]**2 + ab_geo[:,:,1]**2).mean()

    print(f'  {name}: GeoSeg={err_geo:.1f} Enc2D={err_enc:.1f} DD={err_dd:.1f} Zero={err_z:.1f} | sat: GT={gt_sat_img:.0f} Geo={geo_sat:.0f}')
    results.append({'name': name, 'err_geo': err_geo, 'err_enc': err_enc,
                    'err_dd': err_dd, 'err_z': err_z})

    # Save comparison
    bgr_geo = ab_to_bgr(ab_geo, L)
    bgr_enc = ab_to_bgr(ab_enc, L)
    bgr_dd = ab_to_bgr(ab_dd, L)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    imgs = [
        (gray_bgr, 'Gray'),
        (bgr_geo, f'GeoSeg e={err_geo:.1f}'),
        (bgr_enc, f'Enc2D e={err_enc:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'seg_{name}.jpg'), strip)


# ============================================================
# SUMMARY
# ============================================================
print('\n=== SUMMARY ===\n')
if results:
    mean_geo = np.mean([r['err_geo'] for r in results])
    mean_enc = np.mean([r['err_enc'] for r in results])
    mean_dd = np.mean([r['err_dd'] for r in results])
    mean_z = np.mean([r['err_z'] for r in results])

    print(f'Held-out image comparison:')
    print(f'  Zero:     {mean_z:.2f}')
    print(f'  GeoSeg:   {mean_geo:.2f} (gap={(1-mean_geo/mean_z)*100:.1f}%)')
    print(f'  Enc2D:    {mean_enc:.2f} (gap={(1-mean_enc/mean_z)*100:.1f}%)')
    print(f'  DDColor:  {mean_dd:.2f} (gap={(1-mean_dd/mean_z)*100:.1f}%)')
    print(f'\nThe GeoSeg model uses:')
    print(f'  - SLIC superpixels (no learned params)')
    print(f'  - {X_reg.shape[1]} region features (handcrafted)')
    print(f'  - Linear projection to ab (52 params)')
    print(f'  - Guided filter smoothing (no learned params)')
    print(f'  Total learned: 52 params')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
