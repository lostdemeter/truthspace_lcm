"""
SSM Semantic Vocabulary: What "tokens" does the encoder have?

From Part 14 we know:
- Only 3/10 PCA directions encode color (semantic content)
- 7/10 are structural scaffolding
- Random dirs → constant output (one "average" token)
- The 28.2% gap IS the semantic vocabulary

Questions:
1. How many distinct "semantic tokens" does the encoder have?
2. Which W₁ directions carry which tokens?
3. What do the GELU activation patterns look like for different concepts?
4. Can we read object identity (grass, sky, person) from activations?
5. Where exactly in W₁'s SVD do the semantic sub-positions live?
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

PHI = 1.618033988749895; SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_encoder_with_activations(v16, img_tensor, target_stage=0, target_block=0):
    """Run encoder, capture pre-GELU and post-GELU activations at one block."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []; pre_gelu = None; post_gelu = None
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
                h = F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'), v16._get_weight(f'{p}.pwconv1.bias'))
                if si == target_stage and bi == target_block:
                    pre_gelu = h.squeeze(0).detach().numpy()  # [H, W, 4d]
                g = gate(h)
                if si == target_stage and bi == target_block:
                    post_gelu = g.squeeze(0).detach().numpy()
                x = F.linear(g, v16._get_weight(f'{p}.pwconv2.weight'), v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy(), pre_gelu, post_gelu


# ================================================================
# PART 1: ACTIVATION FINGERPRINTS — What fires for what?
# ================================================================
print('=' * 70)
print('PART 1: ACTIVATION FINGERPRINTS — The Semantic Vocabulary')
print('=' * 70)
print()

# Collect images with diverse content
images = []
for idx in range(80, 200):
    if len(images) >= 20: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 3: continue
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    images.append({'idx': idx, 'gray': gray, 'ab': ab, 'sat': sat, 'rgb': r})

print(f'Collected {len(images)} diverse images')

# For Stage 2 Block 0 (384→1536, most sensitive stage from attractor experiment)
print('\nAnalyzing Stage 2, Block 0 (384→1536 expansion):')

all_act_patterns = []  # per-image: [H*W, 1536] binary activation patterns
all_ab_maps = []       # per-image: [H*W, 2] ground truth colors
all_pre_gelu = []

for d in images[:10]:
    t = torch.from_numpy(cv2.cvtColor(d['gray'], cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    _, pre_g, post_g = run_encoder_with_activations(v16, t, target_stage=2, target_block=0)
    
    h, w, c = pre_g.shape  # [16, 16, 1536]
    active = (pre_g > 0).astype(float).reshape(-1, c)  # binary: which neurons fire?
    ab_r = cv2.resize(d['ab'], (w, h)).reshape(-1, 2)
    
    all_act_patterns.append(active)
    all_ab_maps.append(ab_r)
    all_pre_gelu.append(pre_g.reshape(-1, c))

act_all = np.vstack(all_act_patterns)
ab_all = np.vstack(all_ab_maps)
pre_all = np.vstack(all_pre_gelu)

n_pixels = len(act_all)
mean_active = act_all.mean(axis=0)  # per-neuron activation frequency

print(f'  Pixels analyzed: {n_pixels}')
print(f'  Mean activation rate: {act_all.mean():.3f} ({act_all.mean()*1536:.0f}/1536 neurons per pixel)')

# Which neurons are content-selective? (high variance in activation)
act_var = act_all.var(axis=0)
n_always_on = (mean_active > 0.95).sum()
n_always_off = (mean_active < 0.05).sum()
n_selective = ((mean_active > 0.05) & (mean_active < 0.95)).sum()
print(f'  Always ON: {n_always_on}, Always OFF: {n_always_off}, SELECTIVE: {n_selective}')


# ================================================================
# PART 2: NEURON → CONCEPT MAPPING
# ================================================================
print()
print('=' * 70)
print('PART 2: NEURON → CONCEPT — Which neurons encode which concepts?')
print('=' * 70)
print()

# For each selective neuron, correlate its activation with color
# High correlation → this neuron "knows" about a specific concept
selective_mask = (mean_active > 0.05) & (mean_active < 0.95)
selective_idx = np.where(selective_mask)[0]

neuron_corr_a = np.zeros(1536)
neuron_corr_b = np.zeros(1536)

for ni in selective_idx:
    neuron_corr_a[ni] = np.corrcoef(act_all[:, ni], ab_all[:, 0])[0, 1]
    neuron_corr_b[ni] = np.corrcoef(act_all[:, ni], ab_all[:, 1])[0, 1]

# Find the most semantically meaningful neurons
semantic_strength = np.sqrt(neuron_corr_a**2 + neuron_corr_b**2)
top_semantic = np.argsort(semantic_strength)[::-1][:30]

print(f'Top 30 most semantic neurons (out of {n_selective} selective):')
print(f'  {"Neuron":<8} {"Freq":<8} {"corr_a":<10} {"corr_b":<10} {"Concept"}')
print('-' * 55)
for ni in top_semantic:
    freq = mean_active[ni]
    ca = neuron_corr_a[ni]
    cb = neuron_corr_b[ni]
    concept = ''
    if abs(ca) > 0.1:
        concept += 'RED' if ca > 0 else 'GREEN'
    if abs(cb) > 0.1:
        if concept: concept += '+'
        concept += 'YELLOW' if cb > 0 else 'BLUE'
    if not concept: concept = 'subtle'
    print(f'  {ni:<8} {freq:<8.2f} {ca:<+10.3f} {cb:<+10.3f} {concept}')

# Count: how many neurons have significant semantic content?
threshold = 0.15
n_semantic = (semantic_strength > threshold).sum()
n_structural = selective_mask.sum() - n_semantic
print(f'\nSemantic neurons (|r| > {threshold}): {n_semantic} / {selective_mask.sum()} selective')
print(f'Structural neurons: {n_structural}')
print(f'Semantic fraction: {n_semantic/selective_mask.sum():.1%}')


# ================================================================
# PART 3: HOW MANY DISTINCT TOKENS?
# ================================================================
print()
print('=' * 70)
print('PART 3: HOW MANY DISTINCT TOKENS? — The Semantic Vocabulary Size')
print('=' * 70)
print()

# Cluster the activation patterns to find distinct "tokens"
# Each token = a unique combination of active neurons

# Use binary activation patterns as features
# Simple: cluster by PCA of activation patterns

act_centered = act_all - act_all.mean(0)
# Subsample for SVD
sub = np.random.choice(n_pixels, min(2000, n_pixels), replace=False)
_, S_act, Vt_act = np.linalg.svd(act_centered[sub], full_matrices=False)

print(f'Activation pattern SVD:')
cumvar = np.cumsum(S_act**2) / (S_act**2).sum()
for k in [1, 2, 3, 5, 10, 20, 50]:
    if k < len(cumvar):
        print(f'  Top {k} dims: {cumvar[k-1]*100:.1f}% variance')

# Project all patterns to low-dim space
act_pca = act_centered @ Vt_act[:20].T

# k-means in activation PCA space
def simple_kmeans(X, k, n_iter=30):
    n = X.shape[0]
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx].copy()
    for _ in range(n_iter):
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centers]).T
        labels = dists.argmin(axis=1)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0: centers[c] = X[mask].mean(0)
    return labels, centers

# Try different k values — find the "natural" vocabulary size
print('\nVocabulary size scan:')
print(f'  {"k":<6} {"Inertia":<15} {"Δ Inertia":<15} {"Semantic purity"}')
prev_inertia = None
np.random.seed(42)

for k in [2, 4, 8, 12, 16, 24, 32]:
    labels, centers = simple_kmeans(act_pca, k, n_iter=40)
    
    # Inertia
    inertia = 0
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            inertia += np.sum((act_pca[mask] - centers[c])**2)
    
    # Semantic purity: how consistent is the color within each cluster?
    purities = []
    for c in range(k):
        mask = labels == c
        if mask.sum() > 5:
            ab_c = ab_all[mask]
            # Purity = 1 - (within-cluster variance / total variance)
            within_var = np.var(ab_c, axis=0).sum()
            total_var = np.var(ab_all, axis=0).sum()
            purities.append(1 - within_var / total_var)
    
    mean_purity = np.mean(purities) if purities else 0
    delta = f'{prev_inertia - inertia:.0f}' if prev_inertia else '-'
    print(f'  {k:<6} {inertia:<15.0f} {delta:<15} {mean_purity:.3f}')
    prev_inertia = inertia


# ================================================================
# PART 4: THE SEMANTIC MAP — Tokens at k=8
# ================================================================
print()
print('=' * 70)
print('PART 4: SEMANTIC MAP — 8-token vocabulary')
print('=' * 70)
print()

np.random.seed(42)
labels_8, centers_8 = simple_kmeans(act_pca, 8, n_iter=50)

# Analyze each token
print(f'{"Token":<8} {"Size":<8} {"a*":<8} {"b*":<8} {"Act%":<8} {"Color":<15} {"Concept"}')
print('-' * 75)

token_colors = {}
for c in range(8):
    mask = labels_8 == c
    size = mask.sum()
    mean_a = ab_all[mask, 0].mean()
    mean_b = ab_all[mask, 1].mean()
    mean_act = act_all[mask].mean()
    
    # Which neurons are MOST active for this token vs others?
    token_act = act_all[mask].mean(0)  # [1536] average activation for this token
    other_act = act_all[~mask].mean(0)
    selectivity = token_act - other_act  # positive = more active for this token
    
    top_neurons = np.argsort(selectivity)[::-1][:5]
    top_sel = selectivity[top_neurons]
    
    # Color interpretation
    color = ''
    if abs(mean_a) > 2:
        color += 'RED' if mean_a > 0 else 'GREEN'
    if abs(mean_b) > 2:
        if color: color += '+'
        color += 'YELLOW' if mean_b > 0 else 'BLUE'
    if not color: color = 'NEUTRAL'
    
    # Concept interpretation from color
    concept = ''
    if 'GREEN' in color: concept = 'vegetation/grass'
    elif 'BLUE' in color: concept = 'sky/water'
    elif 'RED' in color and 'YELLOW' in color: concept = 'warm/skin/food'
    elif 'RED' in color: concept = 'warm objects'
    elif 'YELLOW' in color: concept = 'sunlight/sand'
    elif color == 'NEUTRAL': concept = 'structure/gray'
    
    token_colors[c] = color
    print(f'{c:<8} {size:<8} {mean_a:<+8.1f} {mean_b:<+8.1f} {mean_act:<8.1%} {color:<15} {concept}')

# Cross-image consistency: same token → same concept across images?
print('\nCross-image token consistency:')
for img_i in range(min(5, len(images))):
    n_pix = all_act_patterns[img_i].shape[0]
    start = sum(a.shape[0] for a in all_act_patterns[:img_i])
    img_labels = labels_8[start:start+n_pix]
    img_ab = all_ab_maps[img_i]
    
    tokens_present = []
    for c in range(8):
        mask = img_labels == c
        if mask.sum() > 0:
            ma = img_ab[mask, 0].mean()
            mb = img_ab[mask, 1].mean()
            tokens_present.append(f'T{c}({ma:+.0f},{mb:+.0f})')
    
    print(f'  Image {img_i}: {", ".join(tokens_present)}')


# ================================================================
# PART 5: WHERE IN W₁ DO SEMANTIC TOKENS LIVE?
# ================================================================
print()
print('=' * 70)
print('PART 5: WHERE IN W₁ — Semantic sub-positions in weight space')
print('=' * 70)
print()

# The semantic tokens are created by W₁ (expand) → GELU gate
# Which ROWS of W₁ are the semantic neurons?
pw1 = v16._get_weight('encoder.arch.stages.2.0.pwconv1.weight').numpy()  # [1536, 384]
U1, S1, Vt1 = np.linalg.svd(pw1, full_matrices=False)

# U1 columns are the "directions" in expanded space
# Each neuron in expanded space = a row of W₁ = a specific direction in input space
# Semantic neurons = specific rows of W₁

# Which SVD components carry the semantic information?
# Project each SVD component's row norms to see if they correlate with semantic neurons
print('SVD component → semantic neuron correlation:')
print(f'  {"SVD mode":<12} {"S value":<10} {"corr w/ semantic":<20}')
print('-' * 45)

for mode in range(min(20, len(S1))):
    # This SVD mode's contribution to each row of W₁
    mode_contrib = np.abs(U1[:, mode]) * S1[mode]  # [1536]
    # Correlation with semantic strength
    corr = np.corrcoef(mode_contrib, semantic_strength)[0, 1]
    marker = ' ← SEMANTIC' if abs(corr) > 0.1 else ''
    print(f'  mode {mode:<6} {S1[mode]:<10.3f} {corr:<+20.3f}{marker}')

# Which input dimensions (columns of Vt1) matter for semantic neurons?
semantic_neuron_idx = np.where(semantic_strength > threshold)[0]
structural_neuron_idx = np.where((selective_mask) & (semantic_strength <= threshold))[0]

# Average W₁ row for semantic vs structural neurons
W1_semantic = pw1[semantic_neuron_idx].mean(0)
W1_structural = pw1[structural_neuron_idx].mean(0)

print(f'\nW₁ row analysis:')
print(f'  Semantic neurons ({len(semantic_neuron_idx)}): mean_norm={np.linalg.norm(pw1[semantic_neuron_idx], axis=1).mean():.4f}')
print(f'  Structural neurons ({len(structural_neuron_idx)}): mean_norm={np.linalg.norm(pw1[structural_neuron_idx], axis=1).mean():.4f}')

# Do semantic neurons live in specific SVD subspaces?
# Project semantic neuron rows onto SVD modes
sem_proj = np.abs(pw1[semantic_neuron_idx] @ Vt1.T)  # [n_sem, 384] → contribution per SVD mode
str_proj = np.abs(pw1[structural_neuron_idx] @ Vt1.T)

sem_energy = (sem_proj**2).mean(0)
str_energy = (str_proj**2).mean(0)

# Where does semantic energy concentrate?
sem_cumvar = np.cumsum(sem_energy) / sem_energy.sum()
str_cumvar = np.cumsum(str_energy) / str_energy.sum()

print(f'\nSVD mode energy distribution:')
for k in [5, 10, 20, 50, 100]:
    if k < len(sem_cumvar):
        print(f'  Top {k} modes: semantic={sem_cumvar[k-1]:.1%}, structural={str_cumvar[k-1]:.1%}')


# ================================================================
# PART 6: THE NULL-SPACE CONNECTION
# ================================================================
print()
print('=' * 70)
print('PART 6: NULL-SPACE — Where do the tokens hide?')
print('=' * 70)
print()

# W₂ has 65% null-space content. Does this null-space encode the semantic tokens?
pw2 = v16._get_weight('encoder.arch.stages.2.0.pwconv2.weight').numpy()  # [384, 1536]

# Null space: W₂ columns can be decomposed into range(W₁) + null(W₁ᵀ)
# proj onto range(W₁) in 1536-dim space: W₁ @ pinv(W₁) is [1536, 1536]
proj_W1 = pw1 @ np.linalg.pinv(pw1)  # [1536, 1536] projection onto range(W₁)
W2_range = pw2 @ proj_W1  # W₂ projected onto range(W₁)
W2_null = pw2 - W2_range  # W₂ component in null(W₁ᵀ)

range_frac = np.linalg.norm(W2_range) / np.linalg.norm(pw2)
null_frac = np.linalg.norm(W2_null) / np.linalg.norm(pw2)
print(f'W₂ decomposition: range={range_frac:.1%}, null={null_frac:.1%}')

# Do semantic neurons have more null-space content in W₂?
W2_range_per_col = np.linalg.norm(W2_range, axis=0)  # [1536]
W2_null_per_col = np.linalg.norm(W2_null, axis=0)    # [1536]
null_ratio = W2_null_per_col / (W2_range_per_col + W2_null_per_col + 1e-10)

sem_null_ratio = null_ratio[semantic_neuron_idx].mean()
str_null_ratio = null_ratio[structural_neuron_idx].mean()
always_off_null = null_ratio[mean_active < 0.05].mean()

print(f'\nNull-space ratio by neuron type:')
print(f'  Semantic neurons:   {sem_null_ratio:.3f}')
print(f'  Structural neurons: {str_null_ratio:.3f}')
print(f'  Always-OFF neurons: {always_off_null:.3f}')
print(f'  Difference: {sem_null_ratio - str_null_ratio:+.3f}')

# Correlation between null-space ratio and semantic strength
corr_null_sem = np.corrcoef(null_ratio[selective_mask], semantic_strength[selective_mask])[0, 1]
print(f'  Correlation (null ratio ↔ semantic strength): r={corr_null_sem:.3f}')


# ================================================================
# PART 7: CAN WE NAME THE TOKENS?
# ================================================================
print()
print('=' * 70)
print('PART 7: NAMING TOKENS — What the encoder "says"')
print('=' * 70)
print()

# For each of the 8 tokens, find which image REGIONS they correspond to
# Use the spatial positions to understand what each token "sees"

for img_i in range(min(3, len(images))):
    d = images[img_i]
    n_pix = all_act_patterns[img_i].shape[0]
    start = sum(a.shape[0] for a in all_act_patterns[:img_i])
    img_labels = labels_8[start:start+n_pix]
    img_ab = all_ab_maps[img_i]
    
    # Get original RGB for the token regions
    rgb_small = cv2.resize(d['rgb'], (16, 16))
    rgb_flat = rgb_small.reshape(-1, 3)
    
    print(f'Image {img_i} (#{d["idx"]}):')
    for c in range(8):
        mask = img_labels == c
        if mask.sum() == 0: continue
        
        # Average RGB in this token's region
        mean_rgb = rgb_flat[mask].mean(0).astype(int)
        r, g, b = mean_rgb[2], mean_rgb[1], mean_rgb[0]  # BGR→RGB
        
        # Average position
        ys = np.where(mask)[0] // 16
        xs = np.where(mask)[0] % 16
        pos = f'({ys.mean():.0f},{xs.mean():.0f})'
        
        # Name based on RGB + position
        name = ''
        if g > r + 30 and g > b + 30: name = 'GRASS/TREE'
        elif b > r + 30 and b > g + 30: name = 'SKY/WATER'
        elif r > g + 30 and r > b + 30: name = 'WARM/SKIN'
        elif r > 150 and g > 150 and b > 150: name = 'BRIGHT'
        elif r < 80 and g < 80 and b < 80: name = 'DARK/SHADOW'
        elif abs(int(r)-int(g)) < 30 and abs(int(g)-int(b)) < 30: name = 'GRAY/NEUTRAL'
        else: name = f'MIXED'
        
        mean_a = img_ab[mask, 0].mean()
        mean_b = img_ab[mask, 1].mean()
        
        print(f'  Token {c}: {mask.sum():3d}px at {pos:>7s} RGB=({r:3d},{g:3d},{b:3d}) '
              f'Lab=({mean_a:+5.1f},{mean_b:+5.1f}) → {name}')
    print()


print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('The encoder has a SEMANTIC VOCABULARY encoded in W₁ directions.')
print('Each "token" = a specific activation pattern in the 1536-dim expanded space.')
print('The GELU gate selects which tokens fire for each spatial position.')
print('W₂ reads the tokens and injects null-space content = the semantic output.')
print('The 28.2% gap = the vocabulary itself (which neurons → which concepts).')
