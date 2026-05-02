"""
SSM + φ-Navigation: Replacing Attention with Geometric Navigation

The Hypothesis:
  Attention = O(N²) pairwise comparison (Q·K^T)
  Navigation = O(N) φ-coordinate reading (already encoded in structure)

From Doc 210: "Navigation IS Reading — the optimal path is already encoded"
From Doc 204: "Reverse: Goal → {x : f(x) = y} (only specific x work)"

Architecture:
  1. SSM: within-position computation (what IS this pixel/token?)
     expand → GELU gate → compress
  2. Navigation: cross-position computation (how do positions relate?)
     φ-coordinates → Fibonacci neighbors → aggregate

If features are φ-structured (proven in Doc 210), then cross-position
relationships are encoded as φ-level differences, and Fibonacci-sized
moves navigate between related positions. No O(N²) needed.

Test: Can φ-navigation between spatial positions in the ConvNeXt encoder
capture cross-position information that improves colorization?
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
SZ = 256
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_stage_features(v16, img_tensor):
    """Run encoder, return per-stage features."""
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
            stage_feats.append(xn.squeeze(0).detach().numpy())  # [H, W, C]
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy(), stage_feats


# ================================================================
# PART 1: ARE THE FEATURES φ-STRUCTURED?
# ================================================================
print('=' * 70)
print('PART 1: φ-STRUCTURE IN ENCODER FEATURES')
print('=' * 70)
print()

# Load test image
im = cv2.imread(all_imgs[85])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
ab_gt = lab_gt[:,:,1:].astype(float) - 128.0
t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.

enc_out, stage_feats = run_encoder_stage_features(v16, t)

for si, sf in enumerate(stage_feats):
    h, w, c = sf.shape
    flat = sf.reshape(-1, c)  # [H*W, C]
    
    # Convert to φ-levels: level = round(log(|x|) / log(φ))
    abs_vals = np.abs(flat[flat != 0])
    if len(abs_vals) == 0: continue
    
    phi_levels = np.round(np.log(abs_vals + 1e-10) / np.log(PHI))
    unique_levels = len(np.unique(phi_levels))
    
    # Level differences between adjacent features
    level_diffs = []
    for pos in range(min(500, len(flat))):
        row = flat[pos]
        nz = row[row != 0]
        if len(nz) < 2: continue
        levels = np.round(np.log(np.abs(nz) + 1e-10) / np.log(PHI))
        diffs = np.abs(np.diff(levels))
        level_diffs.extend(diffs.tolist())
    
    level_diffs = np.array(level_diffs)
    fibs = [0, 1, 2, 3, 5, 8, 13, 21]
    near_fib = sum(min(abs(d - f) for f in fibs) <= 1 for d in level_diffs)
    fib_pct = near_fib / len(level_diffs) * 100 if len(level_diffs) > 0 else 0
    
    print(f'  Stage {si} ({c}D, {h}×{w}):')
    print(f'    Unique φ-levels: {unique_levels} (from {len(abs_vals)} values)')
    print(f'    Level diffs near Fibonacci: {fib_pct:.1f}%')
    print(f'    Mean level: {phi_levels.mean():.1f}, Std: {phi_levels.std():.1f}')


# ================================================================
# PART 2: φ-NAVIGATION — Cross-position relationships
# ================================================================
print()
print('=' * 70)
print('PART 2: φ-NAVIGATION — Finding Related Positions')
print('=' * 70)
print()

# For Stage 1 (32×32, 192D) — good balance of resolution and richness
sf = stage_feats[1]  # [32, 32, 192]
h, w, c = sf.shape
flat = sf.reshape(-1, c)  # [1024, 192]

# Compute φ-coordinates for each position
def to_phi_coords(features):
    """Convert feature vectors to φ-level coordinates."""
    signs = np.sign(features)
    abs_f = np.abs(features) + 1e-10
    levels = np.log(abs_f) / np.log(PHI)
    return signs, levels

signs, levels = to_phi_coords(flat)

# φ-distance between positions: sum of |level_i - level_j| (Manhattan on φ-lattice)
def phi_distance(levels_a, levels_b):
    return np.sum(np.abs(levels_a - levels_b))

# For each position, find its φ-nearest neighbors (navigation targets)
print('Computing φ-neighbors for Stage 1 (32×32)...')
n_pos = h * w

# Compare: φ-distance neighbors vs Euclidean neighbors vs same-color neighbors
ab_s1 = cv2.resize(ab_gt, (w, h)).reshape(-1, 2)

# Sample positions
np.random.seed(42)
sample_pos = np.random.choice(n_pos, 50, replace=False)

phi_same_color = []  # φ-neighbors have same color?
euc_same_color = []  # Euclidean neighbors have same color?
spatial_same_color = []  # Spatial neighbors have same color?

k_neighbors = 5

for pi in sample_pos:
    # φ-distances to all other positions
    phi_dists = np.array([np.sum(np.abs(levels[pi] - levels[j])) for j in range(n_pos)])
    phi_dists[pi] = np.inf
    phi_nn = np.argsort(phi_dists)[:k_neighbors]
    
    # Euclidean distances in feature space
    euc_dists = np.array([np.sqrt(np.sum((flat[pi] - flat[j])**2)) for j in range(n_pos)])
    euc_dists[pi] = np.inf
    euc_nn = np.argsort(euc_dists)[:k_neighbors]
    
    # Spatial neighbors (nearby pixels)
    yi, xi = pi // w, pi % w
    sp_dists = np.array([np.sqrt((yi - j//w)**2 + (xi - j%w)**2) for j in range(n_pos)])
    sp_dists[pi] = np.inf
    sp_nn = np.argsort(sp_dists)[:k_neighbors]
    
    # Color similarity: how similar are neighbors' colors to this position?
    color_pi = ab_s1[pi]
    phi_color_sim = np.mean([np.sqrt(np.sum((ab_s1[j] - color_pi)**2)) for j in phi_nn])
    euc_color_sim = np.mean([np.sqrt(np.sum((ab_s1[j] - color_pi)**2)) for j in euc_nn])
    sp_color_sim = np.mean([np.sqrt(np.sum((ab_s1[j] - color_pi)**2)) for j in sp_nn])
    
    phi_same_color.append(phi_color_sim)
    euc_same_color.append(euc_color_sim)
    spatial_same_color.append(sp_color_sim)

print(f'Color error of k={k_neighbors} nearest neighbors:')
print(f'  φ-navigation neighbors:    {np.mean(phi_same_color):.2f}')
print(f'  Euclidean feat neighbors:   {np.mean(euc_same_color):.2f}')
print(f'  Spatial neighbors:          {np.mean(spatial_same_color):.2f}')
print(f'  Random baseline:            {np.std(ab_s1, axis=0).mean() * 2:.2f}')

# Are φ-neighbors semantically related (same concept)?
print(f'\nDo φ-neighbors share semantic content?')
# Check: are φ-neighbors in the same quadrant of color space?
for name, nn_colors in [('φ-nav', phi_same_color), ('Euclidean', euc_same_color), ('Spatial', spatial_same_color)]:
    good = sum(1 for c in nn_colors if c < 5.0)
    print(f'  {name}: {good}/{len(nn_colors)} neighbors have color error < 5.0 ({100*good/len(nn_colors):.0f}%)')


# ================================================================
# PART 3: NAVIGATION-BASED AGGREGATION — The Attention Replacement
# ================================================================
print()
print('=' * 70)
print('PART 3: NAVIGATION AGGREGATION — Replacing Attention')
print('=' * 70)
print()

# The idea: for each position, aggregate information from φ-nearest neighbors
# weighted by inverse φ-distance. This replaces Q·K^T attention.

def phi_navigate_aggregate(features, k=8):
    """
    For each position, find k φ-nearest neighbors and aggregate.
    Returns: aggregated features [N, C] where each position
    contains information from its φ-neighborhood.
    
    This is the navigation equivalent of attention.
    """
    n, c = features.shape
    signs, levels = to_phi_coords(features)
    
    aggregated = np.zeros_like(features)
    
    for i in range(n):
        # φ-distances to all positions
        phi_dists = np.sum(np.abs(levels[i:i+1] - levels), axis=1)
        phi_dists[i] = np.inf
        
        # Top-k nearest on φ-lattice
        nn_idx = np.argsort(phi_dists)[:k]
        nn_dists = phi_dists[nn_idx]
        
        # Weights: inverse φ-distance (like attention weights)
        weights = 1.0 / (nn_dists + 1e-6)
        weights /= weights.sum()
        
        # Aggregate: weighted sum of neighbor features
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    
    return aggregated


def spatial_aggregate(features, h, w, k=8):
    """Baseline: aggregate from spatial neighbors (like a conv)."""
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


def cosine_aggregate(features, k=8):
    """Baseline: aggregate using cosine similarity (like dot-product attention)."""
    n, c = features.shape
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms
    sims = normed @ normed.T
    np.fill_diagonal(sims, -np.inf)
    
    aggregated = np.zeros_like(features)
    for i in range(n):
        nn_idx = np.argsort(sims[i])[::-1][:k]
        weights = sims[i, nn_idx]
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(k) / k
        aggregated[i] = features[i] + np.sum(features[nn_idx] * weights[:, None], axis=0)
    
    return aggregated


# Test: does navigation aggregation improve color prediction?
print('Testing aggregation methods on Stage 1 features...')
print('(Stage 1: 32×32, 192D — 1024 positions)\n')

# Build color decoder from raw features
def test_color_decode(features, ab_map, name):
    """Test how well features predict color."""
    n = len(features)
    X = np.column_stack([features, np.ones(n)])
    Wa, _, _, _ = lstsq(X, ab_map[:, 0], rcond=None)
    Wb, _, _, _ = lstsq(X, ab_map[:, 1], rcond=None)
    pred_a = X @ Wa
    pred_b = X @ Wb
    rmse = np.sqrt(np.mean((pred_a - ab_map[:, 0])**2 + (pred_b - ab_map[:, 1])**2))
    r_a = np.corrcoef(pred_a, ab_map[:, 0])[0, 1]
    r_b = np.corrcoef(pred_b, ab_map[:, 1])[0, 1]
    print(f'  {name:<35} RMSE={rmse:.2f}  r_a={r_a:.3f}  r_b={r_b:.3f}')
    return rmse

# Raw features
rmse_raw = test_color_decode(flat, ab_s1, 'Raw features (no aggregation)')

# φ-navigation aggregation
print('  Computing φ-navigation aggregation...')
phi_agg = phi_navigate_aggregate(flat, k=8)
rmse_phi = test_color_decode(phi_agg, ab_s1, 'φ-navigation (k=8)')

# Cosine aggregation (like attention)
print('  Computing cosine aggregation...')
cos_agg = cosine_aggregate(flat, k=8)
rmse_cos = test_color_decode(cos_agg, ab_s1, 'Cosine attention (k=8)')

# Spatial aggregation (like convolution)
print('  Computing spatial aggregation...')
sp_agg = spatial_aggregate(flat, h, w, k=8)
rmse_sp = test_color_decode(sp_agg, ab_s1, 'Spatial (k=8)')


# ================================================================
# PART 4: MULTI-IMAGE VALIDATION
# ================================================================
print()
print('=' * 70)
print('PART 4: MULTI-IMAGE VALIDATION')
print('=' * 70)
print()

results = {'raw': [], 'phi_nav': [], 'cosine': [], 'spatial': []}

for idx in range(80, 100):
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
    h1, w1, c1 = sf1.shape
    flat1 = sf1.reshape(-1, c1)
    ab_r = cv2.resize(ab, (w1, h1)).reshape(-1, 2)
    
    def eval_rmse(features, ab_map):
        X = np.column_stack([features, np.ones(len(features))])
        Wa, *_ = lstsq(X, ab_map[:, 0], rcond=None)
        Wb, *_ = lstsq(X, ab_map[:, 1], rcond=None)
        pa, pb = X @ Wa, X @ Wb
        return np.sqrt(np.mean((pa - ab_map[:, 0])**2 + (pb - ab_map[:, 1])**2))
    
    results['raw'].append(eval_rmse(flat1, ab_r))
    results['phi_nav'].append(eval_rmse(phi_navigate_aggregate(flat1, k=5), ab_r))
    results['cosine'].append(eval_rmse(cosine_aggregate(flat1, k=5), ab_r))
    results['spatial'].append(eval_rmse(spatial_aggregate(flat1, h1, w1, k=5), ab_r))
    
    if len(results['raw']) >= 8:
        break

print(f'Results across {len(results["raw"])} images:')
print(f'  {"Method":<35} {"Mean RMSE":<12} {"Improvement"}')
print('-' * 60)
raw_mean = np.mean(results['raw'])
for name, key in [('Raw features', 'raw'), ('φ-navigation (k=5)', 'phi_nav'), 
                   ('Cosine attention (k=5)', 'cosine'), ('Spatial (k=5)', 'spatial')]:
    m = np.mean(results[key])
    imp = (1 - m/raw_mean) * 100
    print(f'  {name:<35} {m:<12.2f} {imp:+.1f}%')


# ================================================================
# PART 5: φ-LEVEL DIFFERENCES → SEMANTIC RELATIONSHIPS
# ================================================================
print()
print('=' * 70)
print('PART 5: φ-LEVEL DIFFERENCES AS SEMANTIC RELATIONSHIPS')
print('=' * 70)
print()

# Key test: do positions with similar φ-levels have similar SEMANTIC content?
# If yes, navigation IS semantic attention.

sf1 = stage_feats[1]
h1, w1, c1 = sf1.shape
flat1 = sf1.reshape(-1, c1)
ab_r1 = cv2.resize(ab_gt, (w1, h1)).reshape(-1, 2)

signs1, levels1 = to_phi_coords(flat1)

# For each pair of positions, measure:
# 1. φ-distance (sum of |level differences|)
# 2. Color similarity (Lab distance)
# 3. Feature cosine similarity

n_pairs = 2000
np.random.seed(42)
pairs_i = np.random.choice(len(flat1), n_pairs)
pairs_j = np.random.choice(len(flat1), n_pairs)
# Avoid self-pairs
mask = pairs_i != pairs_j
pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]

phi_dists = np.array([np.sum(np.abs(levels1[i] - levels1[j])) for i, j in zip(pairs_i, pairs_j)])
color_dists = np.array([np.sqrt(np.sum((ab_r1[i] - ab_r1[j])**2)) for i, j in zip(pairs_i, pairs_j)])
cos_sims = np.array([np.dot(flat1[i], flat1[j]) / (np.linalg.norm(flat1[i]) * np.linalg.norm(flat1[j]) + 1e-8) 
                      for i, j in zip(pairs_i, pairs_j)])

# Correlation: φ-distance → color distance
corr_phi_color = np.corrcoef(phi_dists, color_dists)[0, 1]
corr_cos_color = np.corrcoef(cos_sims, color_dists)[0, 1]

print(f'Pairwise relationship analysis ({len(pairs_i)} pairs):')
print(f'  φ-distance ↔ color distance: r = {corr_phi_color:.3f}')
print(f'  Cosine sim ↔ color distance: r = {corr_cos_color:.3f}')

# Binned analysis: what's the color distance at different φ-distance ranges?
bins = np.percentile(phi_dists, [0, 20, 40, 60, 80, 100])
print(f'\nColor distance by φ-distance bin:')
for i in range(len(bins)-1):
    mask = (phi_dists >= bins[i]) & (phi_dists < bins[i+1])
    if mask.sum() > 0:
        mean_color = np.mean(color_dists[mask])
        print(f'  φ-dist [{bins[i]:.0f}-{bins[i+1]:.0f}]: color_dist = {mean_color:.2f} (n={mask.sum()})')


# ================================================================
# PART 6: THE FIBONACCI MOVE STRUCTURE
# ================================================================
print()
print('=' * 70)
print('PART 6: FIBONACCI MOVES — Do φ-neighbors jump by Fibonacci?')
print('=' * 70)
print()

# For φ-nearest neighbors, what are the level differences per dimension?
np.random.seed(42)
sample = np.random.choice(len(flat1), 100, replace=False)
all_level_jumps = []

for pi in sample:
    phi_dists_p = np.sum(np.abs(levels1[pi:pi+1] - levels1), axis=1)
    phi_dists_p[pi] = np.inf
    nn = np.argsort(phi_dists_p)[:5]
    
    for ni in nn:
        jumps = np.abs(levels1[pi] - levels1[ni])
        all_level_jumps.extend(jumps[jumps > 0.5].tolist())

all_level_jumps = np.round(np.array(all_level_jumps)).astype(int)
jump_counts = {}
for j in all_level_jumps:
    jump_counts[j] = jump_counts.get(j, 0) + 1

# Sort by frequency
sorted_jumps = sorted(jump_counts.items(), key=lambda x: -x[1])
total = len(all_level_jumps)
fibs_set = {0, 1, 2, 3, 5, 8, 13, 21, 34, 55}

print(f'Level jump distribution (φ-nearest neighbors):')
print(f'  {"Jump":<8} {"Count":<10} {"Fraction":<12} {"Fibonacci?"}')
print('-' * 45)
fib_total = 0
for jump, count in sorted_jumps[:15]:
    frac = count / total
    is_fib = jump in fibs_set
    if is_fib: fib_total += count
    marker = '✓' if is_fib else ''
    near_fib = any(abs(jump - f) <= 1 for f in fibs_set)
    if near_fib and not is_fib:
        marker = '~'
        fib_total += count
    print(f'  {jump:<8} {count:<10} {frac:<12.1%} {marker}')

print(f'\n  Fibonacci or near-Fibonacci: {fib_total}/{total} = {fib_total/total:.1%}')


# ================================================================
# PART 7: THE COMPLETE ARCHITECTURE — SSM + Navigation
# ================================================================
print()
print('=' * 70)
print('PART 7: SSM + NAVIGATION — The Complete Architecture')
print('=' * 70)
print()

print('Architecture:')
print('  1. Input → DW Conv (spatial) → Layer Norm')
print('  2. SSM: expand → GELU gate → compress (within-position)')
print('  3. Navigation: φ-coordinate neighbors → aggregate (cross-position)')  
print('  4. SSM: expand → GELU gate → compress (process aggregated)')
print('  5. Output')
print()
print('SSM handles: WHAT is at each position')
print('Navigation handles: HOW positions relate')
print('Together: full sequence/spatial modeling without attention')
print()

# Test: SSM (Stage 1 features) + Navigation vs SSM alone
# Stage 1 features ARE the output of SSM blocks
# Adding navigation aggregation = adding cross-position information

# Use the multi-image results
if results['raw'] and results['phi_nav']:
    raw_m = np.mean(results['raw'])
    phi_m = np.mean(results['phi_nav'])
    cos_m = np.mean(results['cosine'])
    sp_m = np.mean(results['spatial'])
    
    print(f'Empirical comparison (Stage 1 color decoding):')
    print(f'  SSM only (raw features):        RMSE = {raw_m:.2f}')
    print(f'  SSM + φ-navigation:             RMSE = {phi_m:.2f} ({(1-phi_m/raw_m)*100:+.1f}%)')
    print(f'  SSM + cosine attention:         RMSE = {cos_m:.2f} ({(1-cos_m/raw_m)*100:+.1f}%)')
    print(f'  SSM + spatial aggregation:      RMSE = {sp_m:.2f} ({(1-sp_m/raw_m)*100:+.1f}%)')

print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('1. Are features φ-structured? → Check Part 1')
print('2. Do φ-neighbors share semantic content? → Check Parts 2, 5')
print('3. Does navigation improve predictions? → Check Parts 3, 4')
print('4. Do jumps follow Fibonacci? → Check Part 6')
print('5. Can SSM + Navigation replace SSM + Attention? → Check Part 7')
