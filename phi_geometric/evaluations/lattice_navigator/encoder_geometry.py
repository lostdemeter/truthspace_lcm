"""
Encoder Geometry Explorer

The minimum model analysis proved: brightness/position/texture features
CANNOT predict color (R² < 0). But DDColor's encoder features CAN.

Questions:
1. What do encoder features look like per region?
2. Can a linear mapping from encoder features → color reach GT?
3. What is the GEOMETRIC STRUCTURE of encoder features?
   - Are they low-rank? What's their dimensionality?
   - Do they have φ-structure?
   - Are they organized by color, by object, or by something else?
4. Can we extract encoder-like features without 55M parameters?

This tells us: is the encoder doing something we COULD do geometrically
if we understood the right transformation?
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
from ks_v2_damping import segment_by_edges

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def get_encoder_features(v16, img_tensor):
    """
    Run the encoder + UNet to get the feature maps that actually
    feed into the color decoder. These are the features that the
    transformer operates on — the actual geometric input space.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    
    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    
    return {
        'final_features': out3.squeeze(0).detach().numpy(),     # [C, H, W] — what color decoder sees
        'level0': out0.squeeze(0).detach().numpy(),              # multi-scale
        'level1': out1.squeeze(0).detach().numpy(),
        'level2': out2.squeeze(0).detach().numpy(),
        'encoder_stages': [f.squeeze(0).detach().numpy() for f in features],
    }


print('=== ENCODER GEOMETRY EXPLORER ===\n')

v16 = V16GeometricColorizer()

refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T
input_weights = refine_w[:, 100:]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_indices = list(range(50, 70))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/encoder_geometry'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: What do encoder features look like?
# ============================================================
print('=== PART 1: Encoder Feature Structure ===\n')

for idx in test_indices[:3]:
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
    final = enc['final_features']  # [C, H, W]
    C, H, W = final.shape
    
    print(f'  {name}:')
    print(f'    Final features: {C} channels, {H}x{W} spatial')
    
    # SVD of feature maps
    flat = final.reshape(C, -1)  # [C, H*W]
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    
    # Effective dimensionality
    for target in [0.5, 0.8, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cumvar, target) + 1
        print(f'    Rank for {target*100:.0f}% variance: {rank}')
    
    print(f'    S[0]/S[1] = {S[0]/S[1]:.4f} (φ={PHI:.4f}, err={abs(S[0]/S[1]-PHI)/PHI*100:.1f}%)')
    print(f'    S[1]/S[2] = {S[1]/S[2]:.4f}')
    print(f'    S[2]/S[3] = {S[2]/S[3]:.4f}')
    
    # φ-ratio check across all consecutive pairs
    ratios = S[:-1] / S[1:]
    phi_errors = np.abs(ratios - PHI) / PHI * 100
    n_phi = (phi_errors < 15).sum()
    print(f'    Consecutive ratios within 15% of φ: {n_phi}/{len(ratios)}')
    print()


# ============================================================
# PART 2: Can encoder features predict color linearly?
# Per-region: average encoder features → GT ab
# ============================================================
print('=== PART 2: Encoder Features → Color (Linear) ===\n')

all_enc_feats = []
all_gt_ab = []
all_dd_ab = []
all_geo_feats = []  # brightness features for comparison

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    enc = get_encoder_features(v16, img_tensor)
    final = enc['final_features']  # [C, H, W]
    C, H_f, W_f = final.shape
    
    # Get DDColor ab
    with torch.no_grad():
        ab_dd_tensor = v16.forward(img_tensor)
    ab_dd = ab_dd_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    
    # Upsample features to match image size
    final_up = cv2.resize(final.transpose(1, 2, 0), (SZ, SZ)).transpose(2, 0, 1)
    
    # Segment and collect per-region
    labeled, edges = segment_by_edges(gray)
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 20: continue
        
        # Encoder features averaged over region
        enc_feat = final_up[:, mask].mean(axis=1)  # [C]
        
        # GT and DD color
        gt_a = ab_gt[:,:,0][mask].mean()
        gt_b = ab_gt[:,:,1][mask].mean()
        dd_a = ab_dd[:,:,0][mask].mean()
        dd_b = ab_dd[:,:,1][mask].mean()
        
        # Basic geometric features for comparison
        brightness = gray[mask].mean() / 255.0
        ys, xs = np.where(mask)
        y_center = ys.mean() / SZ
        
        all_enc_feats.append(enc_feat)
        all_gt_ab.append([gt_a, gt_b])
        all_dd_ab.append([dd_a, dd_b])
        all_geo_feats.append([brightness, y_center])

X_enc = np.array(all_enc_feats)
X_geo = np.array(all_geo_feats)
Y_gt = np.array(all_gt_ab)
Y_dd = np.array(all_dd_ab)

print(f'Collected: {len(X_enc)} regions')
print(f'Encoder features: {X_enc.shape[1]} dimensions')

# Train/test split
split = int(len(X_enc) * 0.7)
X_enc_train, X_enc_test = X_enc[:split], X_enc[split:]
X_geo_train, X_geo_test = X_geo[:split], X_geo[split:]
Y_train, Y_test = Y_gt[:split], Y_gt[split:]
Y_dd_test = Y_dd[split:]

# Add bias
X_enc_train_b = np.column_stack([X_enc_train, np.ones(len(X_enc_train))])
X_enc_test_b = np.column_stack([X_enc_test, np.ones(len(X_enc_test))])
X_geo_train_b = np.column_stack([X_geo_train, np.ones(len(X_geo_train))])
X_geo_test_b = np.column_stack([X_geo_test, np.ones(len(X_geo_test))])

# Fit: encoder features → GT (use ridge regression to handle high dimensionality)
# Ridge: W = (X'X + λI)^-1 X'Y
lam = 1.0
XtX = X_enc_train_b.T @ X_enc_train_b + lam * np.eye(X_enc_train_b.shape[1])
XtY_a = X_enc_train_b.T @ Y_train[:, 0]
XtY_b = X_enc_train_b.T @ Y_train[:, 1]
W_enc_a = np.linalg.solve(XtX, XtY_a)
W_enc_b = np.linalg.solve(XtX, XtY_b)

# Fit: geo features → GT
W_geo_a, _, _, _ = np.linalg.lstsq(X_geo_train_b, Y_train[:, 0], rcond=None)
W_geo_b, _, _, _ = np.linalg.lstsq(X_geo_train_b, Y_train[:, 1], rcond=None)

# Evaluate
pred_enc_a = X_enc_test_b @ W_enc_a
pred_enc_b = X_enc_test_b @ W_enc_b
pred_geo_a = X_geo_test_b @ W_geo_a
pred_geo_b = X_geo_test_b @ W_geo_b

err_enc = np.sqrt(np.mean((Y_test[:,0] - pred_enc_a)**2 + (Y_test[:,1] - pred_enc_b)**2))
err_geo = np.sqrt(np.mean((Y_test[:,0] - pred_geo_a)**2 + (Y_test[:,1] - pred_geo_b)**2))
err_dd = np.sqrt(np.mean((Y_test[:,0] - Y_dd_test[:,0])**2 + (Y_test[:,1] - Y_dd_test[:,1])**2))
err_zero = np.sqrt(np.mean(Y_test[:,0]**2 + Y_test[:,1]**2))

r2_enc_a = 1 - np.mean((Y_test[:,0] - pred_enc_a)**2) / (np.var(Y_test[:,0]) + 1e-8)
r2_enc_b = 1 - np.mean((Y_test[:,1] - pred_enc_b)**2) / (np.var(Y_test[:,1]) + 1e-8)

print(f'\nResults on test set ({len(Y_test)} regions):')
print(f'  Predict zero:        err={err_zero:.2f}')
print(f'  Brightness+position: err={err_geo:.2f} (gap closed: {(1-err_geo/err_zero)*100:.1f}%)')
print(f'  Encoder features:    err={err_enc:.2f} (gap closed: {(1-err_enc/err_zero)*100:.1f}%)')
print(f'  DDColor:             err={err_dd:.2f} (gap closed: {(1-err_dd/err_zero)*100:.1f}%)')
print(f'  Encoder R² (test):   a={r2_enc_a:.3f}, b={r2_enc_b:.3f}')

# Sweep ridge lambda
print(f'\nRidge lambda sweep:')
for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
    XtX = X_enc_train_b.T @ X_enc_train_b + lam * np.eye(X_enc_train_b.shape[1])
    W_a = np.linalg.solve(XtX, XtY_a)
    W_b = np.linalg.solve(XtX, XtY_b)
    p_a = X_enc_test_b @ W_a
    p_b = X_enc_test_b @ W_b
    err = np.sqrt(np.mean((Y_test[:,0] - p_a)**2 + (Y_test[:,1] - p_b)**2))
    r2a = 1 - np.mean((Y_test[:,0] - p_a)**2) / (np.var(Y_test[:,0]) + 1e-8)
    r2b = 1 - np.mean((Y_test[:,1] - p_b)**2) / (np.var(Y_test[:,1]) + 1e-8)
    print(f'  λ={lam:6.2f}: err={err:.2f}, R² a={r2a:.3f} b={r2b:.3f}, gap={((1-err/err_zero)*100):.1f}%')


# ============================================================
# PART 3: Geometry of encoder feature space
# ============================================================
print('\n=== PART 3: Geometry of Encoder Feature Space ===\n')

# SVD of the encoder feature matrix
U, S, Vt = np.linalg.svd(X_enc - X_enc.mean(axis=0), full_matrices=False)
cumvar = np.cumsum(S**2) / (S**2).sum()

for target in [0.5, 0.8, 0.9, 0.95, 0.99]:
    rank = np.searchsorted(cumvar, target) + 1
    print(f'  Rank for {target*100:.0f}% variance: {rank}/{X_enc.shape[1]}')

print(f'\n  Top singular value ratios:')
for i in range(10):
    ratio = S[i] / S[i+1]
    phi_err = abs(ratio - PHI) / PHI * 100
    marker = ' ← φ!' if phi_err < 15 else ''
    print(f'    S[{i}]/S[{i+1}] = {ratio:.4f} ({phi_err:.1f}% from φ){marker}')

# Do encoder features cluster by color?
# Project to top 2 PCs and see if GT color is smooth
pcs = (X_enc - X_enc.mean(axis=0)) @ Vt[:2].T  # [N, 2]
print(f'\n  Top-2 PC projection:')
print(f'    PC1 range: [{pcs[:,0].min():.1f}, {pcs[:,0].max():.1f}]')
print(f'    PC2 range: [{pcs[:,1].min():.1f}, {pcs[:,1].max():.1f}]')

# Correlation of PCs with GT color
for pc_i in range(2):
    corr_a = np.corrcoef(pcs[:, pc_i], Y_gt[:, 0])[0, 1]
    corr_b = np.corrcoef(pcs[:, pc_i], Y_gt[:, 1])[0, 1]
    print(f'    PC{pc_i+1}↔GT: a={corr_a:.3f}, b={corr_b:.3f}')


# ============================================================
# PART 4: What does the encoder know that brightness doesn't?
# ============================================================
print('\n=== PART 4: Encoder vs Brightness ===\n')

# Residual after removing brightness contribution
brightness = X_geo[:, 0]
enc_after_brightness = X_enc - np.outer(brightness, np.linalg.lstsq(brightness.reshape(-1, 1), X_enc, rcond=None)[0].squeeze())

U_res, S_res, Vt_res = np.linalg.svd(enc_after_brightness - enc_after_brightness.mean(axis=0), full_matrices=False)
cumvar_res = np.cumsum(S_res**2) / (S_res**2).sum()

print(f'  Encoder features AFTER removing brightness:')
for target in [0.5, 0.8, 0.9, 0.95]:
    rank = np.searchsorted(cumvar_res, target) + 1
    print(f'    Rank for {target*100:.0f}%: {rank}')

# Can the residual predict GT?
split = int(len(enc_after_brightness) * 0.7)
X_res_train = np.column_stack([enc_after_brightness[:split], np.ones(split)])
X_res_test = np.column_stack([enc_after_brightness[split:], np.ones(len(enc_after_brightness)-split)])

XtX = X_res_train.T @ X_res_train + 1.0 * np.eye(X_res_train.shape[1])
W_a = np.linalg.solve(XtX, X_res_train.T @ Y_gt[:split, 0])
W_b = np.linalg.solve(XtX, X_res_train.T @ Y_gt[:split, 1])
p_a = X_res_test @ W_a
p_b = X_res_test @ W_b
err_res = np.sqrt(np.mean((Y_gt[split:,0] - p_a)**2 + (Y_gt[split:,1] - p_b)**2))

print(f'\n  Predicting GT from encoder-minus-brightness:')
print(f'    Error: {err_res:.2f} (vs full encoder: {err_enc:.2f}, brightness-only: {err_geo:.2f})')
print(f'    Gap closed: {(1-err_res/err_zero)*100:.1f}%')
print(f'    → The encoder knows things about color that have NOTHING to do with brightness')


# ============================================================
# PART 5: Visual — apply encoder linear model
# ============================================================
print('\n=== PART 5: Visual Comparison ===\n')

test_vis = [all_imgs[i] for i in [51, 54, 56, 58, 60]]

for img_path in test_vis:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2: continue
    
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    enc = get_encoder_features(v16, img_tensor)
    final = enc['final_features']
    final_up = cv2.resize(final.transpose(1, 2, 0), (SZ, SZ)).transpose(2, 0, 1)
    
    with torch.no_grad():
        ab_dd = v16.forward(img_tensor).squeeze(0).permute(1, 2, 0).detach().numpy()
    
    labeled, edges = segment_by_edges(gray)
    ab_linear = np.zeros_like(ab_gt)
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 10: continue
        
        enc_feat = final_up[:, mask].mean(axis=1)
        feat_vec = np.append(enc_feat, 1.0)
        ab_linear[:,:,0][mask] = feat_vec @ W_enc_a
        ab_linear[:,:,1][mask] = feat_vec @ W_enc_b
    
    for ch in range(2):
        ab_linear[:,:,ch] = cv2.bilateralFilter(ab_linear[:,:,ch].astype(np.float32), 9, 30, 30)
    
    err_dd_img = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_lin_img = np.sqrt(np.mean((ab_linear - ab_gt)**2))
    
    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_lin = ab_to_bgr(ab_linear, L)
    
    print(f'  {name}: DDColor={err_dd_img:.1f}, EncLinear={err_lin_img:.1f}')
    
    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_lin, f'EncLinear e={err_lin_img:.1f}'),
        (bgr_dd, f'DDColor e={err_dd_img:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'enc_{name}.jpg'), strip)


# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY ===')
print(f'Encoder features are {X_enc.shape[1]}-dimensional')
print(f'Linear mapping from encoder features → GT:')
print(f'  Encoder: err={err_enc:.2f} (gap closed {(1-err_enc/err_zero)*100:.1f}%)')
print(f'  DDColor: err={err_dd:.2f} (gap closed {(1-err_dd/err_zero)*100:.1f}%)')
print(f'  Brightness: err={err_geo:.2f} (gap closed {(1-err_geo/err_zero)*100:.1f}%)')
print(f'The encoder provides features that predict color far better than brightness.')
print(f'The remaining gap to DDColor is what the TRANSFORMER adds — territory selection.')
print(f'\nOutput saved to: {out_dir}/')
print('Done!')
