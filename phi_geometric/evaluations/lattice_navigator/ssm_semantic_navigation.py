"""
SSM + Semantic φ-Navigation: Navigate the φ-lattice IN semantic space

The problem with Part 3's navigation: we navigated in full 192D feature space
where φ-distance is dominated by structural noise (r=0.074 with color).

The fix: project to semantic subspace FIRST (3 PCA color directions from Part 14),
THEN navigate the φ-lattice. From PHI_UNIVERSAL_COORDINATE_SYSTEM.md:
"φ can represent ANY coordinate system" — so we use φ as coordinates in
the semantic basis where distance = meaning.

Architecture:
  1. SSM: within-position computation → 192D features
  2. Project to semantic subspace (3D color PCA)
  3. Convert to φ-coordinates: level = log(|x|)/log(φ)
  4. Navigate: φ-nearest neighbors IN SEMANTIC SPACE
  5. Aggregate neighbor features (full 192D, weighted by semantic φ-distance)
  6. Decode color from aggregated features
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
# STEP 1: Build semantic basis from training images
# ================================================================
print('=' * 70)
print('STEP 1: BUILD SEMANTIC BASIS')
print('=' * 70)
print()

train_feats = []
train_colors = []
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
    sf1 = sfeats[1]  # Stage 1: [32, 32, 192]
    flat = sf1.reshape(-1, 192)
    ab_r = cv2.resize(ab, (32, 32)).reshape(-1, 2)
    s = np.random.choice(len(flat), min(500, len(flat)), replace=False)
    train_feats.append(flat[s])
    train_colors.append(ab_r[s])

train_feats = np.vstack(train_feats)
train_colors = np.vstack(train_colors)
feat_mean = train_feats.mean(0)

# Find semantic directions: features → color correlation
C = (train_feats - feat_mean).T @ train_colors / len(train_feats)
U_sem, S_sem, Vt_sem = np.linalg.svd(C, full_matrices=False)
sem_dirs = U_sem[:, :3]  # Top 3 semantic directions [192, 3]

print(f'Semantic basis: {sem_dirs.shape}')
print(f'Semantic SVs: {S_sem[:5].round(3)}')
print(f'Top 3 directions capture {(S_sem[:3]**2).sum()/(S_sem**2).sum()*100:.1f}% of feat→color covariance')

# Also get the structural subspace (remaining PCA directions)
feat_centered = train_feats - feat_mean
_, _, Vt_full = np.linalg.svd(feat_centered[:2000], full_matrices=False)
struct_dirs = Vt_full[:10].T  # Top 10 PCA directions [192, 10]


# ================================================================
# STEP 2: φ-NAVIGATION IN SEMANTIC SUBSPACE
# ================================================================
print()
print('=' * 70)
print('STEP 2: φ-NAVIGATION IN SEMANTIC SUBSPACE')
print('=' * 70)
print()

def to_phi_levels(x):
    """Convert values to φ-levels: level = log(|x|) / log(φ)"""
    signs = np.sign(x)
    levels = np.log(np.abs(x) + 1e-10) / LOG_PHI
    return signs, levels

def semantic_phi_navigate(features, sem_basis, feat_mean, k=8):
    """
    Navigate in semantic φ-space:
    1. Project features to semantic subspace
    2. Compute φ-levels in semantic space
    3. Find k nearest neighbors by semantic φ-distance
    4. Aggregate FULL features weighted by semantic φ-distance
    """
    n, c = features.shape
    
    # Project to semantic subspace [N, 3]
    sem_proj = (features - feat_mean) @ sem_basis
    
    # Convert to φ-levels in semantic space [N, 3]
    sem_signs, sem_levels = to_phi_levels(sem_proj)
    
    aggregated = np.zeros_like(features)
    
    for i in range(n):
        # φ-distance in semantic space (3D — meaningful!)
        phi_dists = np.sum(np.abs(sem_levels[i:i+1] - sem_levels), axis=1)
        phi_dists[i] = np.inf
        
        # Top-k nearest in semantic φ-space
        nn_idx = np.argsort(phi_dists)[:k]
        nn_dists = phi_dists[nn_idx]
        
        # Weights: inverse φ-distance
        weights = 1.0 / (nn_dists + 0.1)
        weights /= weights.sum()
        
        # Aggregate: self + weighted sum of neighbors (FULL features)
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    
    return aggregated


def raw_phi_navigate(features, k=8):
    """Old method: φ-distance in full feature space (192D)."""
    n, c = features.shape
    _, levels = to_phi_levels(features)
    aggregated = np.zeros_like(features)
    for i in range(n):
        phi_dists = np.sum(np.abs(levels[i:i+1] - levels), axis=1)
        phi_dists[i] = np.inf
        nn_idx = np.argsort(phi_dists)[:k]
        nn_dists = phi_dists[nn_idx]
        weights = 1.0 / (nn_dists + 1e-6)
        weights /= weights.sum()
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


def cosine_navigate(features, k=8):
    """Cosine similarity navigation (like dot-product attention)."""
    n, c = features.shape
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms
    sims = normed @ normed.T
    np.fill_diagonal(sims, -np.inf)
    aggregated = np.zeros_like(features)
    for i in range(n):
        nn_idx = np.argsort(sims[i])[::-1][:k]
        weights = np.maximum(sims[i, nn_idx], 0)
        if weights.sum() > 0: weights /= weights.sum()
        else: weights = np.ones(k) / k
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


def spatial_navigate(features, h, w, k=8):
    """Spatial neighbor aggregation (like convolution)."""
    n, c = features.shape
    aggregated = np.zeros_like(features)
    for i in range(n):
        yi, xi = i // w, i % w
        sp_dists = np.array([np.sqrt((yi - j//w)**2 + (xi - j%w)**2) for j in range(n)])
        sp_dists[i] = np.inf
        nn_idx = np.argsort(sp_dists)[:k]
        nn_dists = sp_dists[nn_idx]
        weights = 1.0 / (nn_dists + 1e-6)
        weights /= weights.sum()
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


def semantic_cosine_navigate(features, sem_basis, feat_mean, k=8):
    """Cosine similarity in semantic subspace (3D)."""
    sem_proj = (features - feat_mean) @ sem_basis
    norms = np.linalg.norm(sem_proj, axis=1, keepdims=True) + 1e-8
    normed = sem_proj / norms
    sims = normed @ normed.T
    np.fill_diagonal(sims, -np.inf)
    n = len(features)
    aggregated = np.zeros_like(features)
    for i in range(n):
        nn_idx = np.argsort(sims[i])[::-1][:k]
        weights = np.maximum(sims[i, nn_idx], 0)
        if weights.sum() > 0: weights /= weights.sum()
        else: weights = np.ones(k) / k
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    return aggregated


# ================================================================
# STEP 3: SINGLE-IMAGE TEST
# ================================================================
print('Single-image test (image 85):')
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

# Verify: φ-distance in semantic space correlates with color distance
sem_proj = (flat - feat_mean) @ sem_dirs
_, sem_levels = to_phi_levels(sem_proj)

np.random.seed(42)
pairs_i = np.random.choice(len(flat), 2000)
pairs_j = np.random.choice(len(flat), 2000)
mask = pairs_i != pairs_j; pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]

phi_dists_sem = np.array([np.sum(np.abs(sem_levels[i] - sem_levels[j])) for i, j in zip(pairs_i, pairs_j)])
phi_dists_full = np.array([np.sum(np.abs(to_phi_levels(flat[i])[1] - to_phi_levels(flat[j])[1])) for i, j in zip(pairs_i, pairs_j)])
color_dists = np.array([np.sqrt(np.sum((ab_r[i] - ab_r[j])**2)) for i, j in zip(pairs_i, pairs_j)])

r_sem = np.corrcoef(phi_dists_sem, color_dists)[0, 1]
r_full = np.corrcoef(phi_dists_full, color_dists)[0, 1]
print(f'  φ-distance ↔ color distance:')
print(f'    In semantic space (3D): r = {r_sem:.3f}')
print(f'    In full space (192D):   r = {r_full:.3f}')
print(f'    Improvement: {abs(r_sem)/max(abs(r_full), 0.001):.1f}x')


# Color decoding test
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

print(f'\nColor decoding comparison:')
eval_decode(flat, ab_r, 'Raw features (no navigation)')
eval_decode(semantic_phi_navigate(flat, sem_dirs, feat_mean, k=8), ab_r, 'Semantic φ-navigation (k=8)')
eval_decode(semantic_cosine_navigate(flat, sem_dirs, feat_mean, k=8), ab_r, 'Semantic cosine navigation (k=8)')
eval_decode(raw_phi_navigate(flat, k=8), ab_r, 'Full-space φ-navigation (k=8)')
eval_decode(cosine_navigate(flat, k=8), ab_r, 'Full-space cosine (k=8)')
eval_decode(spatial_navigate(flat, 32, 32, k=8), ab_r, 'Spatial (k=8)')


# ================================================================
# STEP 4: MULTI-IMAGE VALIDATION
# ================================================================
print()
print('=' * 70)
print('STEP 4: MULTI-IMAGE VALIDATION')
print('=' * 70)
print()

methods = {
    'Raw': lambda f, h, w: f,
    'Sem φ-nav k=5': lambda f, h, w: semantic_phi_navigate(f, sem_dirs, feat_mean, k=5),
    'Sem φ-nav k=3': lambda f, h, w: semantic_phi_navigate(f, sem_dirs, feat_mean, k=3),
    'Sem cos-nav k=5': lambda f, h, w: semantic_cosine_navigate(f, sem_dirs, feat_mean, k=5),
    'Full φ-nav k=5': lambda f, h, w: raw_phi_navigate(f, k=5),
    'Full cos k=5': lambda f, h, w: cosine_navigate(f, k=5),
    'Spatial k=5': lambda f, h, w: spatial_navigate(f, h, w, k=5),
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
print(f'  {"Method":<25} {"Mean RMSE":<12} {"Improvement":<12} {"p-value"}')
print('-' * 65)
for name in methods:
    m = np.mean(results[name])
    imp = (1 - m/raw_mean) * 100
    # Simple t-test
    diffs = np.array(results[name]) - np.array(results['Raw'])
    se = np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 1
    t_stat = np.mean(diffs) / (se + 1e-8)
    # Approximate p-value from t-distribution (two-tailed)
    from scipy import stats
    p_val = 2 * stats.t.sf(abs(t_stat), df=len(diffs)-1) if len(diffs) > 1 else 1.0
    sig = '*' if p_val < 0.05 else ''
    print(f'  {name:<25} {m:<12.2f} {imp:+6.1f}%      {p_val:.3f} {sig}')


# ================================================================
# STEP 5: ANALYZE WHAT SEMANTIC NAVIGATION FINDS
# ================================================================
print()
print('=' * 70)
print('STEP 5: WHAT DOES SEMANTIC φ-NAVIGATION FIND?')
print('=' * 70)
print()

# For one image, show what each position's semantic φ-neighbors look like
im = cv2.imread(all_imgs[85])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
ab = lab[:,:,1:].astype(float) - 128.0
t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
_, sfeats = run_encoder_stage_features(v16, t)
sf1 = sfeats[1]
flat = sf1.reshape(-1, 192)
ab_r = cv2.resize(ab, (32, 32)).reshape(-1, 2)
rgb_small = cv2.resize(r, (32, 32)).reshape(-1, 3)

sem_proj = (flat - feat_mean) @ sem_dirs
_, sem_levels = to_phi_levels(sem_proj)

# Sample positions and show their semantic φ-neighbors
np.random.seed(42)
sample_pos = [100, 200, 400, 600, 800, 900]  # Various spatial positions

print(f'Semantic φ-neighbors for sample positions:')
for pi in sample_pos:
    yi, xi = pi // 32, pi % 32
    phi_dists = np.sum(np.abs(sem_levels[pi:pi+1] - sem_levels), axis=1)
    phi_dists[pi] = np.inf
    nn = np.argsort(phi_dists)[:5]
    
    # This position
    rgb_p = rgb_small[pi]
    ab_p = ab_r[pi]
    
    # Neighbors
    nn_ys = nn // 32
    nn_xs = nn % 32
    nn_ab = ab_r[nn]
    nn_rgb = rgb_small[nn]
    
    # Mean color error of neighbors
    color_err = np.mean([np.sqrt(np.sum((ab_r[n] - ab_p)**2)) for n in nn])
    
    # Spatial distance of neighbors
    sp_dist = np.mean([np.sqrt((yi - n//32)**2 + (xi - n%32)**2) for n in nn])
    
    # Name the color
    def color_name(a, b):
        n = ''
        if abs(a) > 3: n += 'RED' if a > 0 else 'GREEN'
        if abs(b) > 3:
            if n: n += '+'
            n += 'YEL' if b > 0 else 'BLU'
        return n if n else 'neutral'
    
    pos_color = color_name(ab_p[0], ab_p[1])
    nn_colors = [color_name(ab_r[n][0], ab_r[n][1]) for n in nn]
    same_type = sum(1 for c in nn_colors if c == pos_color)
    
    print(f'  pos ({yi:2d},{xi:2d}) [{pos_color:>12}] a={ab_p[0]:+5.1f} b={ab_p[1]:+5.1f} → '
          f'neighbors: color_err={color_err:.1f}, sp_dist={sp_dist:.1f}, same_type={same_type}/5')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('The key question: does navigating the φ-lattice in semantic subspace')
print('find semantically related positions better than raw φ-navigation?')
print()
r_sem_str = f'{r_sem:.3f}' if 'r_sem' in dir() else '?'
r_full_str = f'{r_full:.3f}' if 'r_full' in dir() else '?'
print(f'φ-distance ↔ color correlation:')
print(f'  Semantic space (3D): r = {r_sem_str}')
print(f'  Full space (192D):   r = {r_full_str}')
