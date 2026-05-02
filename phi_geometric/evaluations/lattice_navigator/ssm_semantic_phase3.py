"""
Phase 3: W₁ Row Decoding — Reading the Holographic Code

Phase 2 established: semantic residuals sit on a φ-angular lattice with
5 reference angles (π/φ³, π/φ², π/2, π/φ, 2π/φ²).

Phase 3 asks: how do the W₁ rows (768 directions in 192D) relate to this lattice?

Each W₁ row is a 192D direction that "reads" from the feature space.
The GELU gate then selects which rows activate.

Questions:
  1. What semantic category does each W₁ row prefer?
  2. Do W₁ rows sit on the same φ-angular lattice as the residuals?
  3. How does the W₁ row direction relate to W₂ readout?
  4. Can we read: W₁ row → concept, and verify it?
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict
from numpy.linalg import lstsq

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_stage1(v16, img_tensor):
    """Run encoder, return stage 1 features [H, W, 192]."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(2):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                x = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                  v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(x, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            if si == 1:
                xn = x.permute(0,2,3,1)
                xn = F.layer_norm(xn, (d,),
                                  v16._get_weight(f'encoder.arch.norm{si}.weight'),
                                  v16._get_weight(f'encoder.arch.norm{si}.bias'))
                return xn.squeeze(0).detach().numpy()


# ================================================================
# STEP 0: EXTRACT W₁ AND W₂ FROM STAGE 1 LAST BLOCK
# ================================================================
print('=' * 70)
print('STEP 0: EXTRACT W₁ AND W₂')
print('=' * 70)
print()

# Stage 1 (si=1), last block (bi=2)
p = 'encoder.arch.stages.1.2'
W1 = v16._get_weight(f'{p}.pwconv1.weight').detach().numpy()  # [768, 192]
b1 = v16._get_weight(f'{p}.pwconv1.bias').detach().numpy()    # [768]
W2 = v16._get_weight(f'{p}.pwconv2.weight').detach().numpy()  # [192, 768]
b2 = v16._get_weight(f'{p}.pwconv2.bias').detach().numpy()    # [192]
gamma = v16._get_weight(f'{p}.gamma').detach().numpy()        # [192]

print(f'W₁: {W1.shape} — 768 rows reading from 192D feature space')
print(f'W₂: {W2.shape} — projects 768D expanded back to 192D')
print(f'b₁: {b1.shape}, b₂: {b2.shape}, γ: {gamma.shape}')

# Row norms of W₁
w1_norms = np.linalg.norm(W1, axis=1)
# Unit directions
W1_unit = W1 / (w1_norms[:, None] + 1e-8)

print(f'\nW₁ row norms: mean={w1_norms.mean():.3f}  std={w1_norms.std():.3f}  '
      f'range=[{w1_norms.min():.3f}, {w1_norms.max():.3f}]')


# ================================================================
# STEP 1: BUILD SEMANTIC CATEGORY RESIDUALS (from Phase 2 data)
# ================================================================
print()
print('=' * 70)
print('STEP 1: BUILD SEMANTIC RESIDUAL CENTROIDS')
print('=' * 70)
print()

def classify_color(a, b):
    sat = np.sqrt(a**2 + b**2)
    if sat < 5: return 'neutral'
    hue = np.degrees(np.arctan2(b, a))
    if hue < 0: hue += 360
    sector = int(hue / 30) % 12
    names = ['red', 'orange', 'yellow', 'chartreuse', 'green', 'spring',
             'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose']
    return names[sector]

cat_residuals = defaultdict(list)
cat_colors = defaultdict(list)

n_images = 0
for img_idx in range(20, 400):
    if n_images >= 80: break
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue

    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    feats = run_encoder_stage1(v16, t)
    h, w = feats.shape[:2]
    flat = feats.reshape(-1, 192)
    struct_mean = flat.mean(0, keepdims=True)
    residuals = flat - struct_mean
    ab_small = cv2.resize(ab, (w, h)).reshape(-1, 2)

    for j in range(len(flat)):
        cat = classify_color(ab_small[j, 0], ab_small[j, 1])
        cat_residuals[cat].append(residuals[j])
        cat_colors[cat].append(ab_small[j])

    n_images += 1

# Compute centroids
cat_dirs = {}
cats_with_data = []
for cat in sorted(cat_residuals.keys()):
    if len(cat_residuals[cat]) < 100:
        continue
    arr = np.array(cat_residuals[cat])
    mean = arr.mean(0)
    norm = np.linalg.norm(mean)
    if norm > 1e-8:
        cat_dirs[cat] = mean / norm
        cats_with_data.append(cat)

print(f'Built {len(cats_with_data)} semantic centroids from {n_images} images')
for cat in cats_with_data:
    cols = np.array(cat_colors[cat])
    print(f'  {cat:<12} n={len(cat_residuals[cat]):5d}  a*={cols[:,0].mean():+5.1f}  b*={cols[:,1].mean():+5.1f}')


# ================================================================
# STEP 2: W₁ ROW → SEMANTIC ALIGNMENT
# ================================================================
print()
print('=' * 70)
print('STEP 2: W₁ ROW → SEMANTIC ALIGNMENT')
print('=' * 70)
print()

# For each W₁ row, compute cosine similarity with each semantic centroid
semantic_matrix = np.zeros((768, len(cats_with_data)))  # [768, N_cats]
for ci, cat in enumerate(cats_with_data):
    semantic_matrix[:, ci] = W1_unit @ cat_dirs[cat]

# Each row's preferred category
preferred_cat = np.argmax(np.abs(semantic_matrix), axis=1)
preferred_sign = np.array([semantic_matrix[i, preferred_cat[i]] for i in range(768)])

# Distribution of preferences
pref_counts = defaultdict(int)
for i in range(768):
    cat = cats_with_data[preferred_cat[i]]
    pref_counts[cat] += 1

print('W₁ row preferences (by max |cosine| with category centroid):')
for cat in cats_with_data:
    n = pref_counts.get(cat, 0)
    print(f'  {cat:<12} {n:3d} rows ({n/768:.1%})')

# Strength of alignment
max_align = np.max(np.abs(semantic_matrix), axis=1)
print(f'\nAlignment strength:')
print(f'  Mean max |cos|: {max_align.mean():.3f}')
print(f'  Max: {max_align.max():.3f}')
print(f'  Rows with |cos| > 0.3: {(max_align > 0.3).sum()}/768')
print(f'  Rows with |cos| > 0.2: {(max_align > 0.2).sum()}/768')
print(f'  Rows with |cos| > 0.1: {(max_align > 0.1).sum()}/768')


# ================================================================
# STEP 3: ANGULAR STRUCTURE OF W₁ ROWS
# ================================================================
print()
print('=' * 70)
print('STEP 3: ANGULAR STRUCTURE OF W₁ ROWS')
print('=' * 70)
print()

# Pairwise angles between W₁ rows
n_sample_pairs = 5000
np.random.seed(42)
pi = np.random.choice(768, n_sample_pairs)
pj = np.random.choice(768, n_sample_pairs)
mask = pi != pj; pi, pj = pi[mask], pj[mask]

w1_angles = []
for i, j in zip(pi, pj):
    cos = np.dot(W1_unit[i], W1_unit[j])
    cos = np.clip(cos, -1, 1)
    w1_angles.append(np.degrees(np.arccos(cos)))
w1_angles = np.array(w1_angles)

print(f'Pairwise angles between W₁ row directions:')
print(f'  Mean: {w1_angles.mean():.1f}°  Std: {w1_angles.std():.1f}°')
print(f'  Range: [{w1_angles.min():.1f}°, {w1_angles.max():.1f}°]')

# φ-reference histogram
phi_refs = {
    'π/φ³': np.degrees(np.pi / PHI**3),
    'π/φ²': np.degrees(np.pi / PHI**2),
    'π/2': 90.0,
    'π/φ': np.degrees(np.pi / PHI),
    '2π/φ²': np.degrees(2*np.pi / PHI**2),
}

print(f'\n  Angle distribution (10° bins):')
bins = np.arange(0, 185, 10)
counts, _ = np.histogram(w1_angles, bins)
for i in range(len(counts)):
    lo, hi = bins[i], bins[i+1]
    bar = '#' * (counts[i] * 40 // max(counts.max(), 1))
    # Mark φ-references
    markers = [name for name, val in phi_refs.items() if lo <= val < hi]
    marker_str = f' ← {",".join(markers)}' if markers else ''
    print(f'    {lo:5.0f}°-{hi:5.0f}°: {counts[i]:4d} {bar}{marker_str}')


# ================================================================
# STEP 4: W₁ ROWS vs SEMANTIC CENTROIDS — Angular Alignment
# ================================================================
print()
print('=' * 70)
print('STEP 4: W₁ ROWS vs SEMANTIC CENTROIDS')
print('=' * 70)
print()

# For each W₁ row, what angle is it from the nearest semantic centroid?
centroid_matrix = np.array([cat_dirs[c] for c in cats_with_data])  # [N_cats, 192]
cos_to_centroids = W1_unit @ centroid_matrix.T  # [768, N_cats]
min_angle_to_centroid = np.degrees(np.arccos(np.clip(np.max(np.abs(cos_to_centroids), axis=1), 0, 1)))

print(f'Angle from each W₁ row to nearest semantic centroid:')
print(f'  Mean: {min_angle_to_centroid.mean():.1f}°  Std: {min_angle_to_centroid.std():.1f}°')
print(f'  Min: {min_angle_to_centroid.min():.1f}°  Max: {min_angle_to_centroid.max():.1f}°')
print(f'  Rows within 45° of a centroid: {(min_angle_to_centroid < 45).sum()}/768')
print(f'  Rows within 60° of a centroid: {(min_angle_to_centroid < 60).sum()}/768')
print(f'  Rows within π/φ² (68.8°): {(min_angle_to_centroid < 68.8).sum()}/768')

# Are the angles to centroids at φ-references?
print(f'\n  Distance to nearest centroid — distribution:')
bins = np.arange(0, 95, 5)
counts, _ = np.histogram(min_angle_to_centroid, bins)
for i in range(len(counts)):
    lo, hi = bins[i], bins[i+1]
    bar = '#' * (counts[i] * 40 // max(counts.max(), 1))
    print(f'    {lo:5.0f}°-{hi:5.0f}°: {counts[i]:4d} {bar}')


# ================================================================
# STEP 5: THE HOLOGRAPHIC CODE — W₁·W₂ Coupling by Category
# ================================================================
print()
print('=' * 70)
print('STEP 5: W₁·W₂ COUPLING — The Lock Mechanism')
print('=' * 70)
print()

# W₂ has shape [192, 768]. Column j of W₂ is the "readout" for neuron j.
# W₂[:, j] determines HOW neuron j's activation contributes to the output.
# The full path: input → W₁ row j (read direction) → GELU → W₂ col j (write direction)

W2_cols = W2.T  # [768, 192] — each row is a neuron's write direction
W2_norms = np.linalg.norm(W2_cols, axis=1)
W2_unit = W2_cols / (W2_norms[:, None] + 1e-8)

# Read-write alignment: how aligned is each neuron's read and write direction?
read_write_cos = np.array([np.dot(W1_unit[i], W2_unit[i]) for i in range(768)])

print(f'Read-Write alignment (W₁ row · W₂ col per neuron):')
print(f'  Mean cos: {read_write_cos.mean():.3f}')
print(f'  Std: {read_write_cos.std():.3f}')
print(f'  Aligned (cos > 0.3): {(read_write_cos > 0.3).sum()}/768')
print(f'  Anti-aligned (cos < -0.3): {(read_write_cos < -0.3).sum()}/768')
print(f'  Orthogonal (|cos| < 0.1): {(np.abs(read_write_cos) < 0.1).sum()}/768')

# For each semantic category: what is the NET read-write direction?
# When neurons aligned with category X fire, where does the output go?
print(f'\nPer-category net effect (sum of W₂ cols weighted by W₁-category alignment):')
for ci, cat in enumerate(cats_with_data):
    alignment = semantic_matrix[:, ci]  # [768] — cos(W₁_row, cat_centroid)
    # Weight W₂ columns by alignment
    weighted_write = (alignment[:, None] * W2_cols).sum(0)  # [192]
    write_norm = np.linalg.norm(weighted_write)
    if write_norm > 1e-8:
        write_dir = weighted_write / write_norm
    else:
        write_dir = weighted_write

    # How aligned is the net write direction with the SAME category centroid?
    self_align = np.dot(write_dir, cat_dirs[cat])

    # How aligned with OTHER category centroids?
    other_aligns = []
    for cj, other_cat in enumerate(cats_with_data):
        if cj == ci: continue
        other_aligns.append(np.dot(write_dir, cat_dirs[other_cat]))
    max_other = max(other_aligns) if other_aligns else 0

    print(f'  {cat:<12} self_align={self_align:+.3f}  max_other={max_other:+.3f}  '
          f'write_mag={write_norm:.3f}')


# ================================================================
# STEP 6: NEURON GROUPS — Functional Clustering
# ================================================================
print()
print('=' * 70)
print('STEP 6: FUNCTIONAL NEURON GROUPS')
print('=' * 70)
print()

# Group neurons by their semantic preference profile
# Use the full semantic_matrix [768, N_cats] as a feature vector per neuron
from sklearn.cluster import KMeans

# Normalize each neuron's semantic profile
sem_norms = np.linalg.norm(semantic_matrix, axis=1, keepdims=True) + 1e-8
sem_unit = semantic_matrix / sem_norms

for n_clusters in [6, 8, 12]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(sem_unit)

    print(f'  {n_clusters} functional groups:')
    for gi in range(n_clusters):
        members = np.where(labels == gi)[0]
        # Group's mean semantic profile
        group_profile = semantic_matrix[members].mean(0)
        top_cat_idx = np.argmax(np.abs(group_profile))
        top_cat = cats_with_data[top_cat_idx]
        top_val = group_profile[top_cat_idx]

        # Group's mean read-write alignment
        rw = read_write_cos[members].mean()

        # Group's mean W₁ norm
        mn = w1_norms[members].mean()

        print(f'    Group {gi}: {len(members):3d} neurons  '
              f'top_cat={top_cat:<12} ({top_val:+.3f})  '
              f'rw_align={rw:+.3f}  W₁_norm={mn:.3f}')
    print()


# ================================================================
# STEP 7: THE LOCK COMBINATION — Per-Category Neuron Sets
# ================================================================
print()
print('=' * 70)
print('STEP 7: THE LOCK COMBINATION — Neurons per Category')
print('=' * 70)
print()

# For each category, identify the neurons that:
# 1. Read from that category's direction (W₁ alignment)
# 2. Write back toward that category (W₂ alignment)
# 3. Both read AND write = "resonant" for that category

for cat in ['red', 'blue', 'green', 'yellow', 'neutral', 'orange', 'cyan', 'magenta']:
    if cat not in cats_with_data: continue
    ci = cats_with_data.index(cat)

    # Read alignment: W₁ row · category centroid
    read_align = semantic_matrix[:, ci]

    # Write alignment: W₂ col · category centroid
    write_align = W2_unit @ cat_dirs[cat]

    # Resonant neurons: both read AND write toward this category
    resonant = (read_align > 0.1) & (write_align > 0.1)
    anti_resonant = (read_align < -0.1) & (write_align < -0.1)

    # Strong readers
    strong_readers = np.where(np.abs(read_align) > 0.15)[0]
    strong_writers = np.where(np.abs(write_align) > 0.15)[0]

    print(f'  {cat}:')
    print(f'    Strong readers (|cos|>0.15): {len(strong_readers)}/768')
    print(f'    Strong writers (|cos|>0.15): {len(strong_writers)}/768')
    print(f'    Resonant (both >0.1):  {resonant.sum()}/768')
    print(f'    Anti-resonant (<-0.1): {anti_resonant.sum()}/768')

    # Top 5 resonant neurons
    resonant_score = read_align * write_align
    top_res = np.argsort(resonant_score)[::-1][:5]
    print(f'    Top resonant: ', end='')
    for ni in top_res:
        print(f'n{ni}(r={read_align[ni]:+.2f},w={write_align[ni]:+.2f}) ', end='')
    print()


# ================================================================
# STEP 8: W₁ ANGLE TO SEMANTIC CENTROIDS — φ-Structure?
# ================================================================
print()
print('=' * 70)
print('STEP 8: W₁ ROW ANGLES TO SEMANTIC CENTROIDS — φ-STRUCTURE?')
print('=' * 70)
print()

# For each W₁ row, compute angle to ALL semantic centroids
# Then check: do these angles cluster at φ-references?
all_row_centroid_angles = []
for i in range(768):
    for ci, cat in enumerate(cats_with_data):
        cos = np.dot(W1_unit[i], cat_dirs[cat])
        cos = np.clip(cos, -1, 1)
        angle = np.degrees(np.arccos(abs(cos)))  # Use |cos| → angles in [0, 90]
        all_row_centroid_angles.append(angle)

all_row_centroid_angles = np.array(all_row_centroid_angles)

print(f'W₁ row → semantic centroid angles ({len(all_row_centroid_angles)} pairs):')
print(f'  Mean: {all_row_centroid_angles.mean():.1f}°  Std: {all_row_centroid_angles.std():.1f}°')

# Distribution
print(f'\n  Distribution (5° bins):')
bins = np.arange(0, 95, 5)
counts, _ = np.histogram(all_row_centroid_angles, bins)
for i in range(len(counts)):
    lo, hi = bins[i], bins[i+1]
    bar = '#' * (counts[i] * 40 // max(counts.max(), 1))
    markers = [name for name, val in phi_refs.items() if lo <= val < hi]
    marker_str = f' ← {",".join(markers)}' if markers else ''
    print(f'    {lo:5.0f}°-{hi:5.0f}°: {counts[i]:5d} {bar}{marker_str}')


print()
print('=' * 70)
print('SUMMARY — Phase 3')
print('=' * 70)
print()
print('W₁ rows (768 × 192D directions) encode the "read" side of the lock.')
print('W₂ cols (768 × 192D directions) encode the "write" side.')
print()
print('Key findings:')
print(f'  • {len(cats_with_data)} semantic categories mapped to W₁ directions')
print(f'  • Max alignment strength: {max_align.max():.3f}')
print(f'  • Read-write alignment: mean cos = {read_write_cos.mean():.3f}')
print(f'  • Resonant neurons per category reveal the lock combination')
print(f'  • W₁ rows span the full angular range of the semantic lattice')
