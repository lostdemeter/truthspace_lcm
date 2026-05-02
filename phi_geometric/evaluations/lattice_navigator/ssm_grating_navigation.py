"""
SSM + Diffraction Grating Navigation: Direction Matching via Interference

From Part 16: φ encodes MAGNITUDE (importance), not DIRECTION (identity).
From Doc 058: A diffraction grating creates interference from TWO ORTHOGONAL VIEWS.
  - Constructive interference = both views align = same semantic content
  - Destructive interference = views disagree = different content

The Architecture:
  Feature = φ-magnitude × direction
            ↑ scaffolding    ↑ content
            ↑ φ-lattice      ↑ grating

  1. Decompose each feature into magnitude (φ-level) + direction (unit vector)
  2. Project direction onto orthogonal "views" (like horizontal + vertical slits)
  3. Compute interference between positions: both views agree → constructive
  4. Use interference as attention weights
  5. Aggregate from positions with constructive interference

The key: φ-angles. If directions can be quantized as θ = π/φ^k,
we get a complete φ-native system — magnitude AND direction on φ-lattices.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from numpy.linalg import lstsq

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
LOG_PHI = np.log(PHI)
SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_stage_features(v16, img_tensor):
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []; stage_feats = []
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
                x = F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'), v16._get_weight(f'{p}.pwconv1.bias'))
                x = gate(x)
                x = F.linear(x, v16._get_weight(f'{p}.pwconv2.weight'), v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))
            stage_feats.append(xn.squeeze(0).detach().numpy())
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy(), stage_feats


# ================================================================
# PART 1: DIRECTION DECOMPOSITION — Separate magnitude from direction
# ================================================================
print('=' * 70)
print('PART 1: DIRECTION DECOMPOSITION')
print('=' * 70)
print()

# Build semantic basis from training images
train_feats, train_colors = [], []
for idx in range(50, 75):
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 3: continue
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    _, sfeats = run_encoder_stage_features(v16, t)
    flat = sfeats[1].reshape(-1, 192)
    ab_r = cv2.resize(ab, (32, 32)).reshape(-1, 2)
    s = np.random.choice(len(flat), min(500, len(flat)), replace=False)
    train_feats.append(flat[s]); train_colors.append(ab_r[s])
train_feats = np.vstack(train_feats); train_colors = np.vstack(train_colors)
feat_mean = train_feats.mean(0)

# PCA basis for creating orthogonal "views"
feat_centered = train_feats - feat_mean
_, _, Vt = np.linalg.svd(feat_centered[:2000], full_matrices=False)

# The "views" for the diffraction grating:
# View 1: Top PCA directions (dominant structure)
# View 2: Color-correlated directions (semantic content)
C = feat_centered.T @ train_colors / len(train_feats)
U_color, S_color, _ = np.linalg.svd(C, full_matrices=False)
color_dirs = U_color[:, :2]  # [192, 2] — the "color slit"

# Structural view: top PCA orthogonal to color
struct_dirs = Vt[:5].T  # [192, 5] — the "structure slit"
# Orthogonalize to color directions
for i in range(5):
    for j in range(2):
        struct_dirs[:, i] -= np.dot(struct_dirs[:, i], color_dirs[:, j]) * color_dirs[:, j]
    norm = np.linalg.norm(struct_dirs[:, i])
    if norm > 1e-6: struct_dirs[:, i] /= norm

print(f'View 1 (Color slit): {color_dirs.shape} — encodes WHAT (semantic content)')
print(f'View 2 (Structure slit): {struct_dirs.shape} — encodes HOW (structural pattern)')
print(f'Orthogonality check: {np.abs(color_dirs.T @ struct_dirs).max():.6f} (should be ~0)')


# ================================================================
# PART 2: φ-ANGLE QUANTIZATION — Can directions be φ-structured?
# ================================================================
print()
print('=' * 70)
print('PART 2: φ-ANGLE QUANTIZATION')
print('=' * 70)
print()

# Test image
im = cv2.imread(all_imgs[85])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
ab = lab[:,:,1:].astype(float) - 128.0
t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
_, sfeats = run_encoder_stage_features(v16, t)
sf1 = sfeats[1]  # [32, 32, 192]
flat = sf1.reshape(-1, 192)
ab_r = cv2.resize(ab, (32, 32)).reshape(-1, 2)

# Project to color view
color_proj = (flat - feat_mean) @ color_dirs  # [1024, 2]

# Compute angle in color space
angles = np.arctan2(color_proj[:, 1], color_proj[:, 0])  # [-π, π]

# Are angles φ-structured? Test if angles cluster at π/φ^k
phi_angles = []
for k in range(-5, 6):
    phi_angles.append(np.pi / PHI**k)
    phi_angles.append(-np.pi / PHI**k)
phi_angles = np.array(sorted(set(phi_angles)))

# For each angle, find nearest φ-angle
nearest_phi_angle = np.array([phi_angles[np.argmin(np.abs(a - phi_angles))] for a in angles])
angle_errors = np.abs(angles - nearest_phi_angle)

print(f'Angle distribution in color view:')
print(f'  Mean angle: {np.mean(angles):.3f} rad ({np.degrees(np.mean(angles)):.1f}°)')
print(f'  Std angle: {np.std(angles):.3f} rad ({np.degrees(np.std(angles)):.1f}°)')
print(f'  φ-angle quantization error: {np.mean(angle_errors):.3f} rad ({np.degrees(np.mean(angle_errors)):.1f}°)')

# How well do φ-angles predict color?
print(f'\n  Angle ↔ color correlation:')
corr_a = np.corrcoef(angles, ab_r[:, 0])[0, 1]
corr_b = np.corrcoef(angles, ab_r[:, 1])[0, 1]
print(f'    angle ↔ a*: r = {corr_a:.3f}')
print(f'    angle ↔ b*: r = {corr_b:.3f}')

# How about magnitude in color view?
color_mag = np.linalg.norm(color_proj, axis=1)
corr_mag_sat = np.corrcoef(color_mag, np.sqrt(ab_r[:, 0]**2 + ab_r[:, 1]**2))[0, 1]
print(f'    color_magnitude ↔ saturation: r = {corr_mag_sat:.3f}')
print(f'\n  → Angle = WHAT color, Magnitude = HOW MUCH color')


# ================================================================
# PART 3: DIFFRACTION GRATING INTERFERENCE
# ================================================================
print()
print('=' * 70)
print('PART 3: DIFFRACTION GRATING INTERFERENCE')
print('=' * 70)
print()

def grating_interference(features, color_dirs, struct_dirs, feat_mean):
    """
    Compute interference between all pairs using diffraction grating.
    
    View 1 (Color): alignment in color direction space
    View 2 (Structure): alignment in structural direction space
    
    Constructive = both views agree
    Destructive = views disagree
    """
    n = len(features)
    centered = features - feat_mean
    
    # Project to each view
    color_proj = centered @ color_dirs   # [N, 2]
    struct_proj = centered @ struct_dirs  # [N, 5]
    
    # Normalize to unit directions (separate magnitude from direction)
    color_norms = np.linalg.norm(color_proj, axis=1, keepdims=True) + 1e-8
    struct_norms = np.linalg.norm(struct_proj, axis=1, keepdims=True) + 1e-8
    color_unit = color_proj / color_norms
    struct_unit = struct_proj / struct_norms
    
    # View 1: Color direction alignment (cosine similarity in 2D color space)
    view1 = color_unit @ color_unit.T  # [N, N] — ranges [-1, 1]
    
    # View 2: Structural alignment (cosine similarity in 5D structure space)
    view2 = struct_unit @ struct_unit.T  # [N, N]
    
    # Interference: both must agree (from Doc 058)
    # Constructive: both positive → geometric mean
    # Destructive: mixed signs → zero
    interference = np.zeros((n, n))
    both_pos = (view1 > 0) & (view2 > 0)
    both_neg = (view1 < 0) & (view2 < 0)
    interference[both_pos] = np.sqrt(view1[both_pos] * view2[both_pos])
    interference[both_neg] = -np.sqrt(np.abs(view1[both_neg] * view2[both_neg]))
    # Mixed = destructive (stays 0)
    
    np.fill_diagonal(interference, 0)
    
    return interference, view1, view2


# Compute grating interference for test image
interference, v1, v2 = grating_interference(flat, color_dirs, struct_dirs, feat_mean)

print(f'Interference matrix stats:')
print(f'  Constructive (>0): {(interference > 0).sum() / interference.size:.1%}')
print(f'  Destructive (<0): {(interference < 0).sum() / interference.size:.1%}')
print(f'  Neutral (=0): {(interference == 0).sum() / interference.size:.1%}')
print(f'  Mean |interference|: {np.abs(interference).mean():.4f}')

# Does interference predict color similarity?
np.random.seed(42)
n_pairs = 3000
pi = np.random.choice(len(flat), n_pairs)
pj = np.random.choice(len(flat), n_pairs)
mask = pi != pj; pi, pj = pi[mask], pj[mask]

interf_vals = interference[pi, pj]
color_dists = np.sqrt(np.sum((ab_r[pi] - ab_r[pj])**2, axis=1))
v1_vals = v1[pi, pj]
v2_vals = v2[pi, pj]

corr_interf_color = np.corrcoef(interf_vals, color_dists)[0, 1]
corr_v1_color = np.corrcoef(v1_vals, color_dists)[0, 1]
corr_v2_color = np.corrcoef(v2_vals, color_dists)[0, 1]

print(f'\nDirection matching ↔ color distance:')
print(f'  View 1 (color direction):     r = {corr_v1_color:.3f}')
print(f'  View 2 (structure direction):  r = {corr_v2_color:.3f}')
print(f'  Combined interference:         r = {corr_interf_color:.3f}')

# Compare with Part 16's raw φ-distance (r=0.074)
# and semantic φ-distance (r=-0.018)
print(f'\n  vs Part 16 baselines:')
print(f'    Raw φ-distance:     r = 0.074')
print(f'    Semantic φ-distance: r = -0.018')
print(f'    Grating interference: r = {corr_interf_color:.3f}')


# ================================================================
# PART 4: GRATING-BASED AGGREGATION — The Attention Replacement
# ================================================================
print()
print('=' * 70)
print('PART 4: GRATING AGGREGATION — Interference as Attention')
print('=' * 70)
print()

def grating_aggregate(features, color_dirs, struct_dirs, feat_mean, k=8):
    """
    Aggregate features using diffraction grating interference.
    
    For each position:
    1. Compute interference with all other positions
    2. Select top-k constructive interference neighbors
    3. Weight by interference strength
    4. Aggregate full features
    """
    n = len(features)
    interference, _, _ = grating_interference(features, color_dirs, struct_dirs, feat_mean)
    
    aggregated = np.zeros_like(features)
    for i in range(n):
        # Top-k by constructive interference (highest positive values)
        scores = interference[i]
        nn_idx = np.argsort(scores)[::-1][:k]
        weights = np.maximum(scores[nn_idx], 0)
        
        if weights.sum() > 0:
            weights /= weights.sum()
            aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
        else:
            aggregated[i] = features[i]
    
    return aggregated


def cosine_aggregate(features, k=8):
    """Baseline: cosine similarity aggregation."""
    n = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    sims = (features / norms) @ (features / norms).T
    np.fill_diagonal(sims, -np.inf)
    aggregated = np.zeros_like(features)
    for i in range(n):
        nn_idx = np.argsort(sims[i])[::-1][:k]
        weights = np.maximum(sims[i, nn_idx], 0)
        if weights.sum() > 0: weights /= weights.sum()
        else: weights = np.ones(k) / k
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


def spatial_aggregate(features, h, w, k=8):
    """Baseline: spatial neighbor aggregation."""
    n = len(features)
    aggregated = np.zeros_like(features)
    for i in range(n):
        yi, xi = i // w, i % w
        dists = np.array([np.sqrt((yi - j//w)**2 + (xi - j%w)**2) for j in range(n)])
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        weights = 1.0 / (dists[nn_idx] + 1e-6)
        weights /= weights.sum()
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


def eval_decode(features, ab_map, name):
    X = np.column_stack([features, np.ones(len(features))])
    Wa, *_ = lstsq(X, ab_map[:, 0], rcond=None)
    Wb, *_ = lstsq(X, ab_map[:, 1], rcond=None)
    pa, pb = X @ Wa, X @ Wb
    rmse = np.sqrt(np.mean((pa - ab_map[:, 0])**2 + (pb - ab_map[:, 1])**2))
    ra = np.corrcoef(pa, ab_map[:, 0])[0, 1]
    rb = np.corrcoef(pb, ab_map[:, 1])[0, 1]
    print(f'  {name:<40} RMSE={rmse:.2f}  r_a={ra:.3f}  r_b={rb:.3f}')
    return rmse

print('Single-image color decoding:')
eval_decode(flat, ab_r, 'Raw features')
eval_decode(grating_aggregate(flat, color_dirs, struct_dirs, feat_mean, k=8), ab_r, 'Grating interference (k=8)')
eval_decode(grating_aggregate(flat, color_dirs, struct_dirs, feat_mean, k=3), ab_r, 'Grating interference (k=3)')
eval_decode(cosine_aggregate(flat, k=8), ab_r, 'Cosine attention (k=8)')
eval_decode(spatial_aggregate(flat, 32, 32, k=8), ab_r, 'Spatial (k=8)')


# ================================================================
# PART 5: MULTI-IMAGE VALIDATION
# ================================================================
print()
print('=' * 70)
print('PART 5: MULTI-IMAGE VALIDATION')
print('=' * 70)
print()

methods = {
    'Raw': lambda f, h, w: f,
    'Grating k=5': lambda f, h, w: grating_aggregate(f, color_dirs, struct_dirs, feat_mean, k=5),
    'Grating k=3': lambda f, h, w: grating_aggregate(f, color_dirs, struct_dirs, feat_mean, k=3),
    'Cosine k=5': lambda f, h, w: cosine_aggregate(f, k=5),
    'Spatial k=5': lambda f, h, w: spatial_aggregate(f, h, w, k=5),
}

results = {k: [] for k in methods}
test_count = 0

for idx in range(80, 130):
    if test_count >= 10: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 3: continue
    
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    _, sfeats = run_encoder_stage_features(v16, t)
    sf1 = sfeats[1]
    h1, w1 = sf1.shape[:2]
    flat1 = sf1.reshape(-1, 192)
    ab_r1 = cv2.resize(ab, (w1, h1)).reshape(-1, 2)
    
    for name, fn in methods.items():
        agg = fn(flat1, h1, w1)
        X = np.column_stack([agg, np.ones(len(agg))])
        Wa, *_ = lstsq(X, ab_r1[:, 0], rcond=None)
        Wb, *_ = lstsq(X, ab_r1[:, 1], rcond=None)
        pa, pb = X @ Wa, X @ Wb
        rmse = np.sqrt(np.mean((pa - ab_r1[:, 0])**2 + (pb - ab_r1[:, 1])**2))
        results[name].append(rmse)
    
    test_count += 1

raw_mean = np.mean(results['Raw'])
print(f'Results across {test_count} images:')
print(f'  {"Method":<25} {"Mean RMSE":<12} {"Improvement":<12} {"Std"}')
print('-' * 60)
for name in methods:
    m = np.mean(results[name])
    s = np.std(results[name])
    imp = (1 - m/raw_mean) * 100
    print(f'  {name:<25} {m:<12.2f} {imp:+6.1f}%      {s:.2f}')


# ================================================================
# PART 6: WHAT THE GRATING FINDS — Neighbor Analysis
# ================================================================
print()
print('=' * 70)
print('PART 6: WHAT THE GRATING FINDS')
print('=' * 70)
print()

# For each position, what do its grating-matched neighbors look like?
interference, v1, v2 = grating_interference(flat, color_dirs, struct_dirs, feat_mean)
rgb_small = cv2.resize(r, (32, 32)).reshape(-1, 3)

np.random.seed(42)
sample_pos = [100, 200, 400, 600, 800, 900]

print(f'Grating neighbors vs cosine neighbors vs spatial neighbors:')
for pi in sample_pos:
    yi, xi = pi // 32, pi % 32
    ab_p = ab_r[pi]
    
    # Grating neighbors
    g_nn = np.argsort(interference[pi])[::-1][:5]
    g_color_err = np.mean([np.sqrt(np.sum((ab_r[n] - ab_p)**2)) for n in g_nn])
    g_sp_dist = np.mean([np.sqrt((yi - n//32)**2 + (xi - n%32)**2) for n in g_nn])
    
    # Cosine neighbors
    norms = np.linalg.norm(flat, axis=1) + 1e-8
    sims = (flat / norms[:, None]) @ (flat[pi] / norms[pi])
    sims[pi] = -np.inf
    c_nn = np.argsort(sims)[::-1][:5]
    c_color_err = np.mean([np.sqrt(np.sum((ab_r[n] - ab_p)**2)) for n in c_nn])
    c_sp_dist = np.mean([np.sqrt((yi - n//32)**2 + (xi - n%32)**2) for n in c_nn])
    
    # Spatial neighbors
    sp_dists = np.array([np.sqrt((yi - j//32)**2 + (xi - j%32)**2) for j in range(len(flat))])
    sp_dists[pi] = np.inf
    s_nn = np.argsort(sp_dists)[:5]
    s_color_err = np.mean([np.sqrt(np.sum((ab_r[n] - ab_p)**2)) for n in s_nn])
    
    print(f'  pos ({yi:2d},{xi:2d}) a={ab_p[0]:+5.1f} b={ab_p[1]:+5.1f}:')
    print(f'    Grating:  color_err={g_color_err:5.1f}, sp_dist={g_sp_dist:4.1f}')
    print(f'    Cosine:   color_err={c_color_err:5.1f}, sp_dist={c_sp_dist:4.1f}')
    print(f'    Spatial:  color_err={s_color_err:5.1f}')


# ================================================================
# PART 7: φ-ANGLE STRUCTURE IN GRATING
# ================================================================
print()
print('=' * 70)
print('PART 7: φ-ANGLE STRUCTURE')
print('=' * 70)
print()

# Are the angles between grating-matched positions φ-structured?
# If the grating finds positions with similar direction, the ANGLE
# between matched pairs should cluster at specific φ-related values

matched_angles = []
for i in range(min(200, len(flat))):
    top_match = np.argmax(interference[i])
    if interference[i, top_match] > 0:
        # Angle between the two feature directions
        ci = (flat[i] - feat_mean)
        cj = (flat[top_match] - feat_mean)
        cos_angle = np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        matched_angles.append(angle)

matched_angles = np.array(matched_angles)

print(f'Angles between grating-matched positions:')
print(f'  Mean angle: {np.mean(matched_angles):.3f} rad ({np.degrees(np.mean(matched_angles)):.1f}°)')
print(f'  Std angle: {np.std(matched_angles):.3f} rad ({np.degrees(np.std(matched_angles)):.1f}°)')
print(f'  Median angle: {np.median(matched_angles):.3f} rad ({np.degrees(np.median(matched_angles)):.1f}°)')

# Check if angles cluster at φ-related values
phi_ref_angles = [np.pi/PHI**k for k in range(-3, 6)]
print(f'\n  Reference φ-angles (π/φ^k):')
for k in range(-3, 6):
    a = np.pi / PHI**k
    # How many matched angles are near this?
    near = np.sum(np.abs(matched_angles - a) < 0.1)
    pct = near / len(matched_angles) * 100
    marker = ' ← cluster' if pct > 10 else ''
    print(f'    k={k:+d}: {a:.3f} rad ({np.degrees(a):5.1f}°) — {pct:.0f}% nearby{marker}')

# Distribution of matched angles in bins
print(f'\n  Angle distribution (10° bins):')
bins = np.arange(0, np.pi + 0.01, np.radians(10))
counts, _ = np.histogram(matched_angles, bins)
for i in range(len(counts)):
    bar = '#' * (counts[i] * 40 // max(counts.max(), 1))
    deg_lo = np.degrees(bins[i])
    deg_hi = np.degrees(bins[i+1])
    print(f'    {deg_lo:5.0f}°-{deg_hi:5.0f}°: {counts[i]:3d} {bar}')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('Feature = φ-magnitude × direction')
print('  φ-magnitude: scaffolding (Part 16 confirmed)')
print('  Direction: semantic content (this experiment)')
print()
print('Diffraction grating decomposes direction into two views:')
print('  View 1 (Color slit): WHAT semantic content')
print('  View 2 (Structure slit): HOW it\'s structured')
print('  Interference: both must agree → constructive')
print()
print('The grating IS the attention mechanism:')
print('  Traditional attention: Q·K^T (dot product in full space)')
print('  Grating attention: interference(View1, View2) (geometric agreement)')
