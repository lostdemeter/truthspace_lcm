"""
Minimum Model Explorer

The question: what is the SIMPLEST geometric mapping from image features
to color that can reach ground truth?

We answer this by:
1. Computing per-region GT colors across many images
2. Extracting geometric features per region
3. Finding the OPTIMAL linear mapping (theoretical ceiling for linear)
4. Testing: what features matter? Is the mapping a rotation? A projection?
5. Building per-region feedback: for each region, what's wrong and why?

The result tells us the minimum model architecture:
- If linear suffices → the mapping IS a projection (geometric!)
- If not → we learn what nonlinear structure is needed
- The optimal weights reveal how shape encodes color
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
from territory_mapper import get_ddcolor_territories
from ks_v2_damping import segment_by_edges

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def extract_region_features(gray, labeled, rid, edges):
    """Extract a rich geometric feature vector for a region."""
    mask = labeled == rid
    if mask.sum() < 10:
        return None, mask
    
    h, w = gray.shape
    ys, xs = np.where(mask)
    
    brightness = gray[mask].mean() / 255.0
    brightness_std = gray[mask].std() / 255.0
    brightness_median = np.median(gray[mask]) / 255.0
    
    y_center = ys.mean() / h
    x_center = xs.mean() / w
    y_spread = ys.std() / h if len(ys) > 1 else 0
    x_spread = xs.std() / w if len(xs) > 1 else 0
    
    size = mask.sum() / (h * w)
    
    # Edge/texture features
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx**2 + sy**2)
    texture = edge_mag[mask].mean() / (edge_mag.max() + 1e-8)
    texture_std = edge_mag[mask].std() / (edge_mag.max() + 1e-8)
    
    # Gradient direction (dominant orientation)
    angles = np.arctan2(sy[mask], sx[mask])
    grad_dir_sin = np.sin(angles).mean()
    grad_dir_cos = np.cos(angles).mean()
    
    # Neighbor contrast
    dilated = ndimage.binary_dilation(mask, iterations=3)
    border = dilated & ~mask
    if border.sum() > 5:
        neighbor_brightness = gray[border].mean() / 255.0
        contrast = brightness - neighbor_brightness
    else:
        contrast = 0.0
    
    # Local context: mean brightness in surrounding 64x64 patch
    cy, cx = int(ys.mean()), int(xs.mean())
    r = 32
    patch = gray[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    local_brightness = patch.mean() / 255.0 if patch.size > 0 else brightness
    local_contrast = brightness - local_brightness
    
    # Global position features (sky/ground/center)
    is_top = max(0, 1 - y_center * 3)  # 1 at top, 0 at y=0.33+
    is_bottom = max(0, y_center * 3 - 2)  # 1 at bottom, 0 at y=0.67-
    is_center_y = max(0, 1 - abs(y_center - 0.5) * 4)
    is_center_x = max(0, 1 - abs(x_center - 0.5) * 4)
    
    # Brightness histogram features (shape of distribution within region)
    hist_vals = gray[mask]
    q25 = np.percentile(hist_vals, 25) / 255.0
    q75 = np.percentile(hist_vals, 75) / 255.0
    skewness = (brightness - brightness_median) / (brightness_std + 1e-8)
    
    features = np.array([
        # Basic (5)
        brightness, brightness_std, brightness_median, size, contrast,
        # Position (6)
        y_center, x_center, is_top, is_bottom, is_center_y, is_center_x,
        # Texture (4)
        texture, texture_std, grad_dir_sin, grad_dir_cos,
        # Context (2)
        local_brightness, local_contrast,
        # Shape (2)
        y_spread, x_spread,
        # Distribution (3)
        q25, q75, skewness,
        # Nonlinear interactions (6)
        brightness * is_top,         # bright sky
        brightness * is_bottom,      # bright ground
        (1-brightness) * is_top,     # dark sky (unusual)
        texture * brightness,        # textured bright
        contrast * brightness,       # contrast in bright
        size * brightness,           # large bright region
    ])
    
    return features, mask


print('=== MINIMUM MODEL EXPLORER ===\n')

v16 = V16GeometricColorizer()
refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T
input_weights = refine_w[:, 100:]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

# Use more images for better statistics
test_indices = list(range(50, 80))

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/minimum_model'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Collect per-region data across many images
# For each region: features, GT color, DDColor color
# ============================================================
print('=== PART 1: Collecting Per-Region Data ===\n')

all_feats = []
all_gt_ab = []
all_dd_ab = []
all_meta = []  # for feedback later

n_grayscale = 0
n_color = 0

for idx in test_indices:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    name = os.path.basename(all_imgs[idx]).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
    
    # Check if grayscale
    gt_sat = np.sqrt(ab_gt[:,:,0]**2 + ab_gt[:,:,1]**2).mean()
    if gt_sat < 2:
        n_grayscale += 1
        continue
    n_color += 1
    
    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    _, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    # Segment and extract
    labeled, edges = segment_by_edges(gray)
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        
        feats, mask = extract_region_features(gray, labeled, rid, edges)
        if feats is None: continue
        
        # GT color for this region
        gt_a = ab_gt[:,:,0][mask].mean()
        gt_b = ab_gt[:,:,1][mask].mean()
        
        # DDColor color for this region
        dd_a = ab_dd[:,:,0][mask].mean()
        dd_b = ab_dd[:,:,1][mask].mean()
        
        all_feats.append(feats)
        all_gt_ab.append([gt_a, gt_b])
        all_dd_ab.append([dd_a, dd_b])
        all_meta.append({'name': name, 'rid': rid, 'mask_size': mask.sum()})

X = np.array(all_feats)
Y_gt = np.array(all_gt_ab)
Y_dd = np.array(all_dd_ab)

print(f'Collected: {len(X)} regions from {n_color} color images ({n_grayscale} grayscale skipped)')
print(f'Features: {X.shape[1]}')
print(f'GT ab range: a=[{Y_gt[:,0].min():.1f}, {Y_gt[:,0].max():.1f}], b=[{Y_gt[:,1].min():.1f}, {Y_gt[:,1].max():.1f}]')
print(f'DD ab range: a=[{Y_dd[:,0].min():.1f}, {Y_dd[:,0].max():.1f}], b=[{Y_dd[:,1].min():.1f}, {Y_dd[:,1].max():.1f}]')


# ============================================================
# PART 2: Optimal linear mapping — the ceiling
# ============================================================
print('\n=== PART 2: Optimal Linear Mapping ===\n')

# Add bias term
X_bias = np.column_stack([X, np.ones(len(X))])

# Train/test split (first 70% train, last 30% test)
split = int(len(X) * 0.7)
X_train, X_test = X_bias[:split], X_bias[split:]
Y_train_gt, Y_test_gt = Y_gt[:split], Y_gt[split:]
Y_train_dd, Y_test_dd = Y_dd[:split], Y_dd[split:]

# Fit: features → GT
W_gt_a, _, _, _ = np.linalg.lstsq(X_train, Y_train_gt[:, 0], rcond=None)
W_gt_b, _, _, _ = np.linalg.lstsq(X_train, Y_train_gt[:, 1], rcond=None)

# Fit: features → DDColor
W_dd_a, _, _, _ = np.linalg.lstsq(X_train, Y_train_dd[:, 0], rcond=None)
W_dd_b, _, _, _ = np.linalg.lstsq(X_train, Y_train_dd[:, 1], rcond=None)

# Evaluate on test set
pred_gt_a = X_test @ W_gt_a
pred_gt_b = X_test @ W_gt_b
pred_dd_a = X_test @ W_dd_a
pred_dd_b = X_test @ W_dd_b

r2_gt_a = 1 - np.mean((Y_test_gt[:,0] - pred_gt_a)**2) / (np.var(Y_test_gt[:,0]) + 1e-8)
r2_gt_b = 1 - np.mean((Y_test_gt[:,1] - pred_gt_b)**2) / (np.var(Y_test_gt[:,1]) + 1e-8)
r2_dd_a = 1 - np.mean((Y_test_dd[:,0] - pred_dd_a)**2) / (np.var(Y_test_dd[:,0]) + 1e-8)
r2_dd_b = 1 - np.mean((Y_test_dd[:,1] - pred_dd_b)**2) / (np.var(Y_test_dd[:,1]) + 1e-8)

err_gt_linear = np.sqrt(np.mean((Y_test_gt[:,0] - pred_gt_a)**2 + (Y_test_gt[:,1] - pred_gt_b)**2))
err_dd_linear = np.sqrt(np.mean((Y_test_dd[:,0] - pred_dd_a)**2 + (Y_test_dd[:,1] - pred_dd_b)**2))

# Baseline: predict mean
err_gt_mean = np.sqrt(np.mean(Y_test_gt[:,0]**2 + Y_test_gt[:,1]**2))  # predict 0
err_gt_trainmean = np.sqrt(np.mean((Y_test_gt[:,0] - Y_train_gt[:,0].mean())**2 + 
                                    (Y_test_gt[:,1] - Y_train_gt[:,1].mean())**2))

print(f'Predicting GT color from geometric features:')
print(f'  R² (test): a={r2_gt_a:.3f}, b={r2_gt_b:.3f}')
print(f'  RMSE: linear={err_gt_linear:.2f}, predict_zero={err_gt_mean:.2f}, predict_mean={err_gt_trainmean:.2f}')
print(f'  Improvement over zero: {(1 - err_gt_linear/err_gt_mean)*100:.1f}%')
print()
print(f'Predicting DDColor from geometric features:')
print(f'  R² (test): a={r2_dd_a:.3f}, b={r2_dd_b:.3f}')
print(f'  RMSE: linear={err_dd_linear:.2f}')


# ============================================================
# PART 3: Feature importance — what features predict color?
# ============================================================
print('\n=== PART 3: Feature Importance ===\n')

feat_names = [
    'brightness', 'bright_std', 'bright_med', 'size', 'contrast',
    'y_pos', 'x_pos', 'is_top', 'is_bottom', 'center_y', 'center_x',
    'texture', 'texture_std', 'grad_sin', 'grad_cos',
    'local_bright', 'local_contrast',
    'y_spread', 'x_spread',
    'q25', 'q75', 'skewness',
    'bright×top', 'bright×bot', 'dark×top', 'tex×bright', 'con×bright', 'size×bright',
    'bias',
]

# Combined importance for predicting GT
importance = np.sqrt(W_gt_a**2 + W_gt_b**2)
# Normalize by feature std to get standardized importance
feat_std = X_train.std(axis=0)
feat_std[-1] = 1  # bias
standardized = importance * feat_std

sorted_idx = np.argsort(standardized)[::-1]
print(f'  {"Feature":<15} {"Std×|W|":>10} {"W_gt_a":>10} {"W_gt_b":>10}')
print(f'  {"-"*47}')
for i in sorted_idx[:15]:
    print(f'  {feat_names[i]:<15} {standardized[i]:10.3f} {W_gt_a[i]:10.3f} {W_gt_b[i]:10.3f}')


# ============================================================
# PART 4: Structure of the optimal mapping
# Is it a rotation? A φ-projection? Something recognizable?
# ============================================================
print('\n=== PART 4: Structure of the Optimal Mapping ===\n')

W_gt = np.column_stack([W_gt_a, W_gt_b])  # [n_features+1, 2]

# SVD of the mapping (excluding bias)
W_nobias = W_gt[:-1]
U, S, Vt = np.linalg.svd(W_nobias, full_matrices=False)

print(f'Mapping SVD: S = [{S[0]:.4f}, {S[1]:.4f}]')
print(f'S[0]/S[1] = {S[0]/S[1]:.4f} (φ = {PHI:.4f}, error = {abs(S[0]/S[1] - PHI)/PHI*100:.1f}%)')
print(f'Rank-1 variance: {S[0]**2 / (S**2).sum() * 100:.1f}%')

# Is the output space a rotation of the input projection?
# V tells us how the a,b channels relate
print(f'\nOutput mixing (V):')
print(f'  V[0] = [{Vt[0,0]:.3f}, {Vt[0,1]:.3f}]  (angle = {np.degrees(np.arctan2(Vt[0,1], Vt[0,0])):.1f}°)')
print(f'  V[1] = [{Vt[1,0]:.3f}, {Vt[1,1]:.3f}]  (angle = {np.degrees(np.arctan2(Vt[1,1], Vt[1,0])):.1f}°)')

# Top input features for mode 1 vs mode 2
print(f'\nTop features for Mode 1 (S={S[0]:.3f}):')
mode1_importance = np.abs(U[:, 0]) * feat_std[:-1]
for i in np.argsort(mode1_importance)[::-1][:5]:
    print(f'  {feat_names[i]:<15} weight={U[i,0]:+.3f} (std×|w|={mode1_importance[i]:.3f})')

print(f'\nTop features for Mode 2 (S={S[1]:.3f}):')
mode2_importance = np.abs(U[:, 1]) * feat_std[:-1]
for i in np.argsort(mode2_importance)[::-1][:5]:
    print(f'  {feat_names[i]:<15} weight={U[i,1]:+.3f} (std×|w|={mode2_importance[i]:.3f})')


# ============================================================
# PART 5: The gap — what linear mapping CAN'T capture
# ============================================================
print('\n=== PART 5: What Linear Mapping Cannot Capture ===\n')

# Residuals
residual_a = Y_test_gt[:, 0] - pred_gt_a
residual_b = Y_test_gt[:, 1] - pred_gt_b

print(f'Residual stats:')
print(f'  a: mean={residual_a.mean():.2f}, std={residual_a.std():.2f}')
print(f'  b: mean={residual_b.mean():.2f}, std={residual_b.std():.2f}')

# Are residuals correlated with any feature? (nonlinear signal)
print(f'\nResidual-feature correlations (nonlinear signal):')
for i in range(min(22, X.shape[1])):  # original features only
    corr_a = np.corrcoef(X_test[:, i], residual_a)[0, 1] if np.std(X_test[:, i]) > 0 else 0
    corr_b = np.corrcoef(X_test[:, i], residual_b)[0, 1] if np.std(X_test[:, i]) > 0 else 0
    if abs(corr_a) > 0.1 or abs(corr_b) > 0.1:
        print(f'  {feat_names[i]:<15} r_a={corr_a:+.3f}  r_b={corr_b:+.3f}  ** nonlinear signal')

# How does DDColor do vs our linear model?
err_dd_test = np.sqrt(np.mean((Y_test_gt[:,0] - Y_test_dd[:,0])**2 + 
                                (Y_test_gt[:,1] - Y_test_dd[:,1])**2))
print(f'\nComparison on test set:')
print(f'  DDColor error to GT:      {err_dd_test:.2f}')
print(f'  Linear model error to GT: {err_gt_linear:.2f}')
print(f'  Predict zero error:       {err_gt_mean:.2f}')
print(f'  Linear model closes {(1 - err_gt_linear/err_gt_mean)*100:.1f}% of zero→GT gap')
print(f'  DDColor closes      {(1 - err_dd_test/err_gt_mean)*100:.1f}% of zero→GT gap')


# ============================================================
# PART 6: Apply optimal linear model — visual comparison
# ============================================================
print('\n=== PART 6: Visual Comparison ===\n')

test_vis = [all_imgs[i] for i in [51, 54, 56, 58, 60, 62]]

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
    
    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    _, ab_dd = get_ddcolor_territories(v16, img_tensor)
    
    # Optimal linear model
    labeled, edges = segment_by_edges(gray)
    ab_linear = np.zeros_like(ab_gt)
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        feats, mask = extract_region_features(gray, labeled, rid, edges)
        if feats is None: continue
        
        feat_vec = np.append(feats, 1.0)  # add bias
        pred_a = feat_vec @ W_gt_a
        pred_b = feat_vec @ W_gt_b
        
        ab_linear[:,:,0][mask] = pred_a
        ab_linear[:,:,1][mask] = pred_b
    
    # Smooth
    for ch in range(2):
        ab_linear[:,:,ch] = cv2.bilateralFilter(ab_linear[:,:,ch].astype(np.float32), 9, 30, 30)
    
    err_dd = np.sqrt(np.mean((ab_dd - ab_gt)**2))
    err_lin = np.sqrt(np.mean((ab_linear - ab_gt)**2))
    
    bgr_dd = ab_to_bgr(ab_dd, L)
    bgr_lin = ab_to_bgr(ab_linear, L)
    
    print(f'  {name}: DDColor={err_dd:.1f}, Linear={err_lin:.1f}')
    
    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_lin, f'Linear e={err_lin:.1f}'),
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'minmodel_{name}.jpg'), strip)


# ============================================================
# PART 7: The minimum model specification
# ============================================================
print('\n=== PART 7: Minimum Model Specification ===\n')

# How many features are actually needed?
# Progressive feature addition — which features reduce error most?
print('Progressive feature analysis (greedy):')
remaining = list(range(X_bias.shape[1]))
selected = []
prev_err = err_gt_mean

for step in range(min(10, len(remaining))):
    best_err = float('inf')
    best_feat = None
    
    for f in remaining:
        trial = selected + [f]
        X_sel_train = X_train[:, trial]
        X_sel_test = X_test[:, trial]
        
        try:
            W_a, _, _, _ = np.linalg.lstsq(X_sel_train, Y_train_gt[:, 0], rcond=None)
            W_b, _, _, _ = np.linalg.lstsq(X_sel_train, Y_train_gt[:, 1], rcond=None)
            pred_a = X_sel_test @ W_a
            pred_b = X_sel_test @ W_b
            err = np.sqrt(np.mean((Y_test_gt[:,0] - pred_a)**2 + (Y_test_gt[:,1] - pred_b)**2))
            if err < best_err:
                best_err = err
                best_feat = f
        except:
            continue
    
    if best_feat is None: break
    selected.append(best_feat)
    remaining.remove(best_feat)
    improvement = (1 - best_err / prev_err) * 100
    gap_closed = (1 - best_err / err_gt_mean) * 100
    
    fname = feat_names[best_feat] if best_feat < len(feat_names) else f'feat_{best_feat}'
    print(f'  +{fname:<15} → err={best_err:.2f} (gap closed: {gap_closed:.1f}%, step improvement: {improvement:.1f}%)')
    prev_err = best_err

print(f'\n  DDColor comparison:  err={err_dd_test:.2f} (gap closed: {(1-err_dd_test/err_gt_mean)*100:.1f}%)')

print(f'\n=== CONCLUSION ===')
print(f'The minimum model needs {len(selected)} features to reach its ceiling.')
print(f'Linear ceiling: {prev_err:.2f} error to GT')
print(f'DDColor:        {err_dd_test:.2f} error to GT')
if prev_err > err_dd_test:
    gap = (prev_err - err_dd_test) / err_dd_test * 100
    print(f'Linear geometric features are {gap:.0f}% worse than DDColor.')
    print(f'The gap represents what the ENCODER provides — spatial feature extraction')
    print(f'that pure brightness/position/texture cannot replicate.')
else:
    print(f'Linear geometric features MATCH OR BEAT DDColor!')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
