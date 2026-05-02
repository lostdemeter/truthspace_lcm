"""
Derive the 5 Spatial Correction Modes Geometrically

Strategy:
1. EXAMINE: What do the actual correction modes look like? (using DDColor as ground truth)
2. CORRELATE: What grayscale image features predict each mode?
3. DERIVE: Build pure geometric mode predictor from image features alone
4. TEST: Apply derived modes to lattice output, compare to DDColor

The goal: replace DDColor's 55M parameters with ~5 geometric rules.
"""
import numpy as np
import cv2
import sys
import glob
import os

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from color_lattice import LatticeNavigator
import torch
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()

def extract_features(gray, sz):
    """Extract geometric features from grayscale image."""
    h, w = gray.shape
    
    # 1. Brightness (normalized)
    brightness = gray.astype(float) / 255.0
    
    # 2. Vertical position
    yy = np.tile(np.arange(h).reshape(-1, 1) / h, (1, w))
    
    # 3. Horizontal position
    xx = np.tile(np.arange(w).reshape(1, -1) / w, (h, 1))
    
    # 4. Local texture (Gabor energy)
    gabor_e = np.zeros((h, w))
    for theta_idx in range(4):
        theta = theta_idx * np.pi / 4
        kernel = cv2.getGaborKernel((11, 11), 3.0, theta, 0.1, 0.5, 0)
        resp = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gabor_e += resp**2
    gabor_e = np.sqrt(gabor_e)
    gabor_e = gabor_e / (gabor_e.max() + 1e-8)
    
    # 5. Edge density
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sx**2 + sy**2)
    edges_smooth = cv2.GaussianBlur(edges, (15, 15), 0)
    edges_smooth = edges_smooth / (edges_smooth.max() + 1e-8)
    
    # 6. Local contrast
    blur = cv2.GaussianBlur(gray.astype(float), (15, 15), 0)
    local_var = cv2.GaussianBlur(gray.astype(float)**2, (15, 15), 0) - blur**2
    local_contrast = np.sqrt(np.maximum(local_var, 0)) / 128.0
    
    # 7. Smoothness (inverse of edges)
    smoothness = 1.0 - edges_smooth
    
    # 8. Brightness gradient (vertical derivative of brightness)
    bright_grad = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=5)
    bright_grad = bright_grad / (np.abs(bright_grad).max() + 1e-8)
    
    return {
        'brightness': brightness,
        'y_pos': yy,
        'x_pos': xx,
        'texture': gabor_e,
        'edges': edges_smooth,
        'contrast': local_contrast,
        'smoothness': smoothness,
        'bright_grad': bright_grad,
    }

print('=== DERIVING 5 SPATIAL MODES GEOMETRICALLY ===')
print()

# Initialize
image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
nav = LatticeNavigator()
nav.initialize(image_paths)
nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=False)
v16 = V16GeometricColorizer()

SZ = 128
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
# Use 16 images for analysis, hold out 4 for testing
analyze_indices = list(range(50, 66))
test_indices = list(range(66, 74))
analyze_paths = [all_imgs[i] for i in analyze_indices if i < len(all_imgs)]
test_paths = [all_imgs[i] for i in test_indices if i < len(all_imgs)]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/derived_modes'
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# STEP 1: Collect correction fields and features
# ================================================================
print('Step 1: Collecting correction fields and image features...')

all_features = []  # List of feature dicts
all_corrections_a = []  # Correction channel a
all_corrections_b = []  # Correction channel b
all_L = []
all_lattice = []
all_ddcolor = []
all_gray = []
all_names = []

for img_path in analyze_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t_in = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        ab_dd = v16.forward(t_in)
    ab_ddcolor = cv2.resize(ab_dd[0].permute(1,2,0).numpy(), (SZ, SZ))
    
    # Lattice
    ab_lattice = nav.colorize(gray)
    
    # Correction
    correction = ab_ddcolor - ab_lattice
    
    # Features
    feats = extract_features(gray, SZ)
    
    all_features.append(feats)
    all_corrections_a.append(correction[:,:,0])
    all_corrections_b.append(correction[:,:,1])
    all_L.append(L)
    all_lattice.append(ab_lattice)
    all_ddcolor.append(ab_ddcolor)
    all_gray.append(gray)
    all_names.append(name)
    
    print(f'  {name}: correction |a|={np.abs(correction[:,:,0]).mean():.1f}, |b|={np.abs(correction[:,:,1]).mean():.1f}')

N = len(all_corrections_a)
print(f'\nCollected {N} images')

# ================================================================
# STEP 2: Correlate each correction with image features
# ================================================================
print('\nStep 2: Correlating corrections with image features...')

feat_names = list(all_features[0].keys())

# For each feature and each channel, compute correlation across all images
print(f'\n{"Feature":<15} {"corr_a":>8} {"corr_b":>8} {"mean(|r|)":>10}')
print('-' * 45)

feature_correlations = {}
for fn in feat_names:
    corrs_a = []
    corrs_b = []
    for i in range(N):
        feat_flat = all_features[i][fn].flatten()
        ca = np.corrcoef(feat_flat, all_corrections_a[i].flatten())[0, 1]
        cb = np.corrcoef(feat_flat, all_corrections_b[i].flatten())[0, 1]
        corrs_a.append(ca)
        corrs_b.append(cb)
    
    mean_a = np.mean(corrs_a)
    mean_b = np.mean(corrs_b)
    mean_abs = (abs(mean_a) + abs(mean_b)) / 2
    feature_correlations[fn] = (mean_a, mean_b)
    print(f'{fn:<15} {mean_a:>8.3f} {mean_b:>8.3f} {mean_abs:>10.3f}')

# ================================================================
# STEP 3: Build geometric mode predictor
# ================================================================
print('\n\nStep 3: Building geometric mode predictor...')
print('Using top correlated features to predict correction fields.')

# Stack all features and corrections for regression
# For each pixel: [brightness, y_pos, x_pos, texture, edges, contrast, smoothness, bright_grad]
# Predict: [correction_a, correction_b]

X_all = []  # [N*SZ*SZ, n_features]
Y_a_all = []  # [N*SZ*SZ]
Y_b_all = []

for i in range(N):
    feat_stack = np.stack([all_features[i][fn].flatten() for fn in feat_names], axis=1)
    X_all.append(feat_stack)
    Y_a_all.append(all_corrections_a[i].flatten())
    Y_b_all.append(all_corrections_b[i].flatten())

X = np.vstack(X_all)  # [total_pixels, 8]
Y_a = np.concatenate(Y_a_all)
Y_b = np.concatenate(Y_b_all)

print(f'Regression data: X={X.shape}, Y_a={Y_a.shape}')

# Add interaction terms (feature products) for richer representation
# Key interactions: brightness*y_pos, texture*y_pos, brightness*texture
X_interact = np.column_stack([
    X,
    X[:, 0] * X[:, 1],  # brightness * y_pos
    X[:, 0] * X[:, 3],  # brightness * texture
    X[:, 1] * X[:, 3],  # y_pos * texture
    X[:, 0]**2,          # brightness^2
    X[:, 1]**2,          # y_pos^2
])

print(f'With interactions: X={X_interact.shape}')

# Solve via least squares: Y = X @ W + b
# Add bias column
X_bias = np.column_stack([X_interact, np.ones(X_interact.shape[0])])

# Use SVD-based pseudo-inverse for numerical stability
print('Solving least squares...')
W_a, res_a, rank_a, sv_a = np.linalg.lstsq(X_bias, Y_a, rcond=None)
W_b, res_b, rank_b, sv_b = np.linalg.lstsq(X_bias, Y_b, rcond=None)

# Predict
Y_a_pred = X_bias @ W_a
Y_b_pred = X_bias @ W_b

# Reconstruction error
err_a = np.sqrt(np.mean((Y_a - Y_a_pred)**2))
err_b = np.sqrt(np.mean((Y_b - Y_b_pred)**2))
total_var_a = np.std(Y_a)
total_var_b = np.std(Y_b)

r2_a = 1 - np.mean((Y_a - Y_a_pred)**2) / np.var(Y_a)
r2_b = 1 - np.mean((Y_b - Y_b_pred)**2) / np.var(Y_b)

print(f'\nLinear regression results:')
print(f'  Channel a: RMSE={err_a:.2f}, R²={r2_a:.3f} (std={total_var_a:.2f})')
print(f'  Channel b: RMSE={err_b:.2f}, R²={r2_b:.3f} (std={total_var_b:.2f})')

# Print weight magnitudes for each feature
print(f'\nFeature importance (|weight|):')
feat_labels = feat_names + ['bright*ypos', 'bright*tex', 'ypos*tex', 'bright²', 'ypos²', 'bias']
for j, fl in enumerate(feat_labels):
    print(f'  {fl:<18} w_a={W_a[j]:>7.2f}  w_b={W_b[j]:>7.2f}  |w|={(abs(W_a[j])+abs(W_b[j]))/2:>6.2f}')

# ================================================================
# STEP 4: Apply geometric modes to lattice output (on analysis images)
# ================================================================
print('\n\nStep 4: Applying geometric correction to analysis images...')

for i in range(min(4, N)):
    name = all_names[i]
    L = all_L[i]
    ab_lat = all_lattice[i]
    ab_dd = all_ddcolor[i]
    
    # Compute features
    feat_stack = np.stack([all_features[i][fn] for fn in feat_names], axis=-1)  # [SZ,SZ,8]
    flat = feat_stack.reshape(-1, len(feat_names))
    
    # Add interactions
    flat_interact = np.column_stack([
        flat,
        flat[:, 0] * flat[:, 1],
        flat[:, 0] * flat[:, 3],
        flat[:, 1] * flat[:, 3],
        flat[:, 0]**2,
        flat[:, 1]**2,
    ])
    flat_bias = np.column_stack([flat_interact, np.ones(flat_interact.shape[0])])
    
    # Predict corrections
    pred_a = (flat_bias @ W_a).reshape(SZ, SZ)
    pred_b = (flat_bias @ W_b).reshape(SZ, SZ)
    
    # Apply to lattice
    ab_corrected = ab_lat.copy()
    ab_corrected[:,:,0] += pred_a
    ab_corrected[:,:,1] += pred_b
    
    # Smooth
    for ch in range(2):
        ab_corrected[:,:,ch] = cv2.bilateralFilter(ab_corrected[:,:,ch].astype(np.float32), 7, 40, 40)
    
    bgr_lat = ab_to_bgr(ab_lat, L)
    bgr_corr = ab_to_bgr(ab_corrected, L)
    bgr_dd = ab_to_bgr(ab_dd, L)
    gt = cv2.resize(cv2.imread(analyze_paths[i]), (SZ, SZ))
    
    # Labels
    for img, label in [(bgr_lat, f'Lattice s={get_sat(bgr_lat):.0f}'),
                        (bgr_corr, f'Geometric s={get_sat(bgr_corr):.0f}'),
                        (bgr_dd, f'DDColor s={get_sat(bgr_dd):.0f}'),
                        (gt, f'GT s={get_sat(gt):.0f}')]:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    strip = np.hstack([bgr_lat, bgr_corr, bgr_dd, gt])
    cv2.imwrite(os.path.join(out_dir, f'train_{name}.jpg'), strip)
    
    err_to_dd = np.sqrt(np.mean((ab_corrected - ab_dd)**2))
    err_lat_to_dd = np.sqrt(np.mean((ab_lat - ab_dd)**2))
    print(f'  {name}: lattice→DDColor err={err_lat_to_dd:.1f}, '
          f'geometric→DDColor err={err_to_dd:.1f}, '
          f'reduction={1-err_to_dd/err_lat_to_dd:.0%}')

# ================================================================
# STEP 5: Test on HELD-OUT images (never seen during regression)
# ================================================================
print('\n\nStep 5: Testing on HELD-OUT images...')

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t_in = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        ab_dd = v16.forward(t_in)
    ab_ddcolor = cv2.resize(ab_dd[0].permute(1,2,0).numpy(), (SZ, SZ))
    
    # Lattice
    ab_lattice = nav.colorize(gray)
    
    # Geometric correction
    feats = extract_features(gray, SZ)
    feat_stack = np.stack([feats[fn] for fn in feat_names], axis=-1)
    flat = feat_stack.reshape(-1, len(feat_names))
    flat_interact = np.column_stack([
        flat,
        flat[:, 0] * flat[:, 1],
        flat[:, 0] * flat[:, 3],
        flat[:, 1] * flat[:, 3],
        flat[:, 0]**2,
        flat[:, 1]**2,
    ])
    flat_bias = np.column_stack([flat_interact, np.ones(flat_interact.shape[0])])
    
    pred_a = (flat_bias @ W_a).reshape(SZ, SZ)
    pred_b = (flat_bias @ W_b).reshape(SZ, SZ)
    
    ab_corrected = ab_lattice.copy()
    ab_corrected[:,:,0] += pred_a
    ab_corrected[:,:,1] += pred_b
    
    for ch in range(2):
        ab_corrected[:,:,ch] = cv2.bilateralFilter(ab_corrected[:,:,ch].astype(np.float32), 7, 40, 40)
    
    bgr_lat = ab_to_bgr(ab_lattice, L)
    bgr_corr = ab_to_bgr(ab_corrected, L)
    bgr_dd = ab_to_bgr(ab_ddcolor, L)
    
    for img, label in [(bgr_lat, f'Lattice s={get_sat(bgr_lat):.0f}'),
                        (bgr_corr, f'Geometric s={get_sat(bgr_corr):.0f}'),
                        (bgr_dd, f'DDColor s={get_sat(bgr_dd):.0f}'),
                        (r, f'GT s={get_sat(r):.0f}')]:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    strip = np.hstack([bgr_lat, bgr_corr, bgr_dd, r])
    cv2.imwrite(os.path.join(out_dir, f'test_{name}.jpg'), strip)
    
    err_to_dd = np.sqrt(np.mean((ab_corrected - ab_ddcolor)**2))
    err_lat_to_dd = np.sqrt(np.mean((ab_lattice - ab_ddcolor)**2))
    print(f'  {name}: lattice→DD err={err_lat_to_dd:.1f}, '
          f'geometric→DD err={err_to_dd:.1f}, '
          f'reduction={1-err_to_dd/err_lat_to_dd:.0%}')

# ================================================================
# STEP 6: Save the geometric weights (the "5 modes")
# ================================================================
print('\n\nStep 6: The geometric correction formula')
print('='*60)
print('correction_a(pixel) = Σ w_a[i] * feature[i](pixel)')
print('correction_b(pixel) = Σ w_b[i] * feature[i](pixel)')
print()
print('These weights ARE the 5 spatial modes, expressed as a')
print('linear combination of image features. No neural network needed.')
print()
print(f'Total parameters: {len(W_a) + len(W_b)} ({len(W_a)} per channel)')
print(f'vs DDColor: 55,000,000 parameters')
print(f'Compression: {55_000_000 / (len(W_a) + len(W_b)):.0f}x')

# Save weights
np.savez(os.path.join(out_dir, 'geometric_correction_weights.npz'),
         W_a=W_a, W_b=W_b,
         feat_names=feat_names,
         r2_a=r2_a, r2_b=r2_b)
print(f'\nWeights saved to {out_dir}/geometric_correction_weights.npz')

print('\nDone!')
