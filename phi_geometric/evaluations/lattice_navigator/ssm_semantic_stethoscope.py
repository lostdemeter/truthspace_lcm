"""
Phase 1: The Stethoscope — Controlled Stimulus Probing of Semantic Space

Safe-cracking analogy (Doc 189):
  - Dial = our controlled inputs (we turn it systematically)
  - Tumblers = W₁ row directions (1536 expanded neurons)
  - Click = neuron activation (GELU gate opens)
  - Contents = the full semantic vocabulary

We generate ~50 synthetic stimuli with KNOWN semantic content,
run each through the encoder, and record everything:
  - Feature vectors at every position (the "direction" in 192D)
  - Which expanded neurons fire and how strongly (the "clicks")
  - Color output predictions (what "opens")

This builds the activation database for Phases 2-5.
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895
SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


# ================================================================
# ENCODER WITH FULL ACTIVATION RECORDING
# ================================================================

def run_encoder_full(v16, img_tensor, stage_idx=1):
    """
    Run encoder and record EVERYTHING at the specified stage:
    - Pre-GELU expanded features (W₁x + b₁)
    - Post-GELU activations
    - Which neurons fire (binary)
    - Final output features
    - Color prediction
    """
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s

    activations = {}
    features = []

    with torch.no_grad():
        # Stem
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

                # THE KEY: record pre-GELU and post-GELU at target stage
                pre_gelu = F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                    v16._get_weight(f'{p}.pwconv1.bias'))

                if si == stage_idx and bi == depths[si] - 1:
                    activations['pre_gelu'] = pre_gelu.squeeze(0).detach().numpy()  # [H, W, 4*d]
                    post_gelu = gate(pre_gelu)
                    activations['post_gelu'] = post_gelu.squeeze(0).detach().numpy()
                    activations['firing'] = (activations['post_gelu'] > 0).astype(np.float32)
                    activations['fire_rate'] = activations['firing'].reshape(-1, activations['firing'].shape[-1]).mean(0)
                else:
                    post_gelu = gate(pre_gelu)

                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x

            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

            if si == stage_idx:
                activations['features'] = xn.squeeze(0).detach().numpy()  # [H, W, d]

        # Run through UNet to get color prediction
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
        activations['color_pred'] = out3.squeeze(0).detach().numpy()  # [2, H, W]

    return activations


# ================================================================
# STIMULUS GENERATION
# ================================================================

def make_stimuli():
    """Generate controlled stimuli with known semantic content."""
    stimuli = {}

    # --- SOLID COLORS (18 stimuli) ---
    colors_bgr = {
        'solid_red': (0, 0, 255), 'solid_green': (0, 255, 0), 'solid_blue': (255, 0, 0),
        'solid_cyan': (255, 255, 0), 'solid_magenta': (255, 0, 255), 'solid_yellow': (0, 255, 255),
        'solid_white': (255, 255, 255), 'solid_black': (0, 0, 0),
        'solid_orange': (0, 165, 255), 'solid_pink': (203, 192, 255),
        'solid_brown': (42, 42, 165), 'solid_purple': (128, 0, 128),
    }
    for name, bgr in colors_bgr.items():
        img = np.full((SZ, SZ, 3), bgr, dtype=np.uint8)
        stimuli[name] = {'img': img, 'category': 'solid_color', 'subcategory': name.split('_')[1]}

    # Gray levels
    for level in [32, 64, 96, 128, 160, 192, 224]:
        name = f'solid_gray_{level}'
        img = np.full((SZ, SZ, 3), (level, level, level), dtype=np.uint8)
        stimuli[name] = {'img': img, 'category': 'solid_gray', 'subcategory': f'gray_{level}'}

    # --- GRADIENTS (6 stimuli) ---
    for direction, name in [('lr', 'gradient_lr'), ('tb', 'gradient_tb')]:
        for color_axis, cname in [(0, 'blue'), (1, 'green'), (2, 'red')]:
            img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
            if direction == 'lr':
                grad = np.linspace(0, 255, SZ).astype(np.uint8)
                img[:, :, color_axis] = grad[None, :]
            else:
                grad = np.linspace(0, 255, SZ).astype(np.uint8)
                img[:, :, color_axis] = grad[:, None]
            sname = f'{name}_{cname}'
            stimuli[sname] = {'img': img, 'category': 'gradient', 'subcategory': f'{direction}_{cname}'}

    # --- TEXTURES (6 stimuli) ---
    # Horizontal stripes
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for y in range(SZ):
        if (y // 16) % 2 == 0: img[y, :] = 255
    stimuli['texture_h_stripes'] = {'img': img, 'category': 'texture', 'subcategory': 'h_stripes'}

    # Vertical stripes
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for x in range(SZ):
        if (x // 16) % 2 == 0: img[:, x] = 255
    stimuli['texture_v_stripes'] = {'img': img, 'category': 'texture', 'subcategory': 'v_stripes'}

    # Checkerboard
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for y in range(SZ):
        for x in range(SZ):
            if ((y // 16) + (x // 16)) % 2 == 0: img[y, x] = 255
    stimuli['texture_checker'] = {'img': img, 'category': 'texture', 'subcategory': 'checker'}

    # Random noise
    np.random.seed(42)
    img = np.random.randint(0, 256, (SZ, SZ, 3), dtype=np.uint8)
    stimuli['texture_noise'] = {'img': img, 'category': 'texture', 'subcategory': 'noise'}

    # Fine texture (high freq)
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for y in range(SZ):
        for x in range(SZ):
            if ((y // 4) + (x // 4)) % 2 == 0: img[y, x] = 255
    stimuli['texture_fine_checker'] = {'img': img, 'category': 'texture', 'subcategory': 'fine_checker'}

    # Smooth gradient texture
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for y in range(SZ):
        for x in range(SZ):
            val = int(128 + 127 * np.sin(2 * np.pi * x / 32) * np.sin(2 * np.pi * y / 32))
            img[y, x] = val
    stimuli['texture_sine'] = {'img': img, 'category': 'texture', 'subcategory': 'sine'}

    # --- EDGES (6 stimuli) ---
    # Sharp horizontal edge
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    img[:SZ//2, :] = 255
    stimuli['edge_horizontal'] = {'img': img, 'category': 'edge', 'subcategory': 'horizontal'}

    # Sharp vertical edge
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    img[:, :SZ//2] = 255
    stimuli['edge_vertical'] = {'img': img, 'category': 'edge', 'subcategory': 'vertical'}

    # Diagonal edge
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for y in range(SZ):
        for x in range(SZ):
            if x > y: img[y, x] = 255
    stimuli['edge_diagonal'] = {'img': img, 'category': 'edge', 'subcategory': 'diagonal'}

    # Color edge (red|blue)
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    img[:, :SZ//2] = (255, 0, 0)  # blue (BGR)
    img[:, SZ//2:] = (0, 0, 255)  # red (BGR)
    stimuli['edge_color_rb'] = {'img': img, 'category': 'edge', 'subcategory': 'color_rb'}

    # Color edge (green|yellow)
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    img[:, :SZ//2] = (0, 255, 0)  # green
    img[:, SZ//2:] = (0, 255, 255)  # yellow
    stimuli['edge_color_gy'] = {'img': img, 'category': 'edge', 'subcategory': 'color_gy'}

    # Soft edge (gradient boundary)
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for x in range(SZ):
        val = int(255 / (1 + np.exp(-(x - SZ//2) / 10)))
        img[:, x] = val
    stimuli['edge_soft'] = {'img': img, 'category': 'edge', 'subcategory': 'soft'}

    # --- SHAPES (6 stimuli) ---
    # White circle on black
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    cv2.circle(img, (SZ//2, SZ//2), SZ//4, (255, 255, 255), -1)
    stimuli['shape_circle_wb'] = {'img': img, 'category': 'shape', 'subcategory': 'circle_wb'}

    # Black circle on white
    img = np.full((SZ, SZ, 3), 255, dtype=np.uint8)
    cv2.circle(img, (SZ//2, SZ//2), SZ//4, (0, 0, 0), -1)
    stimuli['shape_circle_bw'] = {'img': img, 'category': 'shape', 'subcategory': 'circle_bw'}

    # Red circle on green
    img = np.full((SZ, SZ, 3), (0, 255, 0), dtype=np.uint8)
    cv2.circle(img, (SZ//2, SZ//2), SZ//4, (0, 0, 255), -1)
    stimuli['shape_circle_rg'] = {'img': img, 'category': 'shape', 'subcategory': 'circle_rg'}

    # White rectangle on black
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    cv2.rectangle(img, (SZ//4, SZ//4), (3*SZ//4, 3*SZ//4), (255, 255, 255), -1)
    stimuli['shape_rect_wb'] = {'img': img, 'category': 'shape', 'subcategory': 'rect_wb'}

    # White triangle on black
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    pts = np.array([[SZ//2, SZ//4], [SZ//4, 3*SZ//4], [3*SZ//4, 3*SZ//4]])
    cv2.fillPoly(img, [pts], (255, 255, 255))
    stimuli['shape_triangle_wb'] = {'img': img, 'category': 'shape', 'subcategory': 'triangle_wb'}

    # Multiple small circles
    img = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    for cy in range(SZ//8, SZ, SZ//4):
        for cx in range(SZ//8, SZ, SZ//4):
            cv2.circle(img, (cx, cy), SZ//10, (255, 255, 255), -1)
    stimuli['shape_multi_circles'] = {'img': img, 'category': 'shape', 'subcategory': 'multi_circles'}

    # --- NATURAL PATCHES (from real images) ---
    natural_indices = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i, idx in enumerate(natural_indices):
        if idx < len(all_imgs):
            im = cv2.imread(all_imgs[idx])
            if im is not None:
                img = cv2.resize(im, (SZ, SZ))
                stimuli[f'natural_{i}'] = {'img': img, 'category': 'natural', 'subcategory': f'image_{idx}'}

    return stimuli


# ================================================================
# PHASE 1: RUN ALL STIMULI
# ================================================================

print('=' * 70)
print('PHASE 1: THE STETHOSCOPE — Controlled Stimulus Probing')
print('=' * 70)
print()

stimuli = make_stimuli()
print(f'Generated {len(stimuli)} stimuli:')
categories = defaultdict(list)
for name, s in stimuli.items():
    categories[s['category']].append(name)
for cat, names in sorted(categories.items()):
    print(f'  {cat}: {len(names)} stimuli')
print()

# Run each stimulus through the encoder
database = {}  # name → activations dict
for i, (name, stim) in enumerate(stimuli.items()):
    img = stim['img']

    # Convert to grayscale input (as the colorizer expects)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0) / 255.

    # Also get ground-truth Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    ab_gt = lab[:,:,1:].astype(float) - 128.0

    acts = run_encoder_full(v16, t, stage_idx=1)
    acts['ab_gt'] = ab_gt
    acts['category'] = stim['category']
    acts['subcategory'] = stim['subcategory']

    database[name] = acts

    if (i + 1) % 10 == 0 or i == 0:
        h, w = acts['features'].shape[:2]
        n_firing = (acts['fire_rate'] > 0.5).sum()
        n_silent = (acts['fire_rate'] < 0.01).sum()
        print(f'  [{i+1:2d}/{len(stimuli)}] {name:<30} feat={h}x{w}x192  '
              f'firing={n_firing}/768  silent={n_silent}/768')

print(f'\nAll {len(database)} stimuli processed.')


# ================================================================
# ANALYSIS 1: FIRING PATTERNS — What clicks for what?
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 1: FIRING PATTERNS — Which neurons click for which stimuli?')
print('=' * 70)
print()

# Collect fire rates per stimulus
n_neurons = 768  # stage 1: 4 * 192 = 768
all_fire_rates = {}
for name, acts in database.items():
    all_fire_rates[name] = acts['fire_rate']  # [768]

fire_rate_matrix = np.array([all_fire_rates[name] for name in database.keys()])  # [N_stimuli, 768]
stim_names = list(database.keys())

print(f'Fire rate matrix: {fire_rate_matrix.shape}')
print(f'  Mean fire rate: {fire_rate_matrix.mean():.3f}')
print(f'  Neurons always firing (>90% for all stimuli): {(fire_rate_matrix.min(0) > 0.9).sum()}')
print(f'  Neurons always silent (<1% for all stimuli): {(fire_rate_matrix.max(0) < 0.01).sum()}')
print(f'  Selective neurons (>50% some, <10% others): ', end='')
selective = ((fire_rate_matrix.max(0) > 0.5) & (fire_rate_matrix.min(0) < 0.1)).sum()
print(f'{selective}')

# Most selective neurons: highest variance in fire rate across stimuli
fire_var = fire_rate_matrix.var(0)
top_selective = np.argsort(fire_var)[::-1][:20]

print(f'\n  Top 20 most selective neurons (highest fire rate variance):')
for rank, ni in enumerate(top_selective[:20]):
    rates = fire_rate_matrix[:, ni]
    top_stim = stim_names[np.argmax(rates)]
    bot_stim = stim_names[np.argmin(rates)]
    print(f'    neuron {ni:3d}: var={fire_var[ni]:.3f}  '
          f'max={rates.max():.2f} ({top_stim})  '
          f'min={rates.min():.2f} ({bot_stim})')


# ================================================================
# ANALYSIS 2: CATEGORY SIGNATURES — Does each category have a unique firing pattern?
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 2: CATEGORY SIGNATURES')
print('=' * 70)
print()

cat_profiles = {}
for cat in categories:
    cat_names = [n for n in stim_names if database[n]['category'] == cat]
    cat_rates = np.array([all_fire_rates[n] for n in cat_names])
    cat_profiles[cat] = {
        'mean': cat_rates.mean(0),
        'std': cat_rates.std(0),
        'n': len(cat_names),
    }

# Which neurons discriminate between categories?
print('Category-average fire rates:')
for cat, prof in sorted(cat_profiles.items()):
    active = (prof['mean'] > 0.5).sum()
    silent = (prof['mean'] < 0.01).sum()
    print(f'  {cat:<15} n={prof["n"]:2d}  active(>50%)={active:3d}  silent(<1%)={silent:3d}')

# Cross-category discrimination
print(f'\nCategory separation (cosine distance of mean fire-rate profiles):')
cat_names_sorted = sorted(cat_profiles.keys())
for i, c1 in enumerate(cat_names_sorted):
    for j, c2 in enumerate(cat_names_sorted):
        if j <= i: continue
        p1, p2 = cat_profiles[c1]['mean'], cat_profiles[c2]['mean']
        cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
        print(f'  {c1:<15} vs {c2:<15} cos={cos:.3f}')


# ================================================================
# ANALYSIS 3: FEATURE DIRECTIONS — What directions do different stimuli produce?
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 3: FEATURE DIRECTIONS')
print('=' * 70)
print()

# Collect mean feature direction per stimulus (average over positions, then normalize)
mean_dirs = {}
for name, acts in database.items():
    feat = acts['features'].reshape(-1, 192)  # [H*W, 192]
    mean_feat = feat.mean(0)
    norm = np.linalg.norm(mean_feat)
    if norm > 1e-8:
        mean_dirs[name] = mean_feat / norm
    else:
        mean_dirs[name] = mean_feat

dir_matrix = np.array([mean_dirs[name] for name in stim_names])  # [N, 192]

# Angular separation between stimuli
print('Angular separation between stimuli (degrees):')
print()

# Within-category vs between-category angles
within_angles = []
between_angles = []
for i in range(len(stim_names)):
    for j in range(i+1, len(stim_names)):
        cos = np.dot(dir_matrix[i], dir_matrix[j])
        cos = np.clip(cos, -1, 1)
        angle = np.degrees(np.arccos(cos))
        cat_i = database[stim_names[i]]['category']
        cat_j = database[stim_names[j]]['category']
        if cat_i == cat_j:
            within_angles.append(angle)
        else:
            between_angles.append(angle)

within_angles = np.array(within_angles) if within_angles else np.array([0])
between_angles = np.array(between_angles) if between_angles else np.array([0])

print(f'  Within-category:  mean={np.mean(within_angles):.1f}°  std={np.std(within_angles):.1f}°  '
      f'min={np.min(within_angles):.1f}°  max={np.max(within_angles):.1f}°')
print(f'  Between-category: mean={np.mean(between_angles):.1f}°  std={np.std(between_angles):.1f}°  '
      f'min={np.min(between_angles):.1f}°  max={np.max(between_angles):.1f}°')
print(f'  π/φ² = {np.degrees(np.pi / PHI**2):.1f}°')
print(f'  Separation ratio: {np.mean(between_angles) / np.mean(within_angles):.2f}x')

# Check if π/φ² is a natural separator
phi2_angle = np.degrees(np.pi / PHI**2)
within_above = (within_angles > phi2_angle).mean()
between_above = (between_angles > phi2_angle).mean()
print(f'\n  Angles above π/φ² ({phi2_angle:.1f}°):')
print(f'    Within-category:  {within_above:.1%}')
print(f'    Between-category: {between_above:.1%}')


# ================================================================
# ANALYSIS 4: COLOR SPACE MAPPING — What colors does each stimulus predict?
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 4: COLOR OUTPUT MAPPING')
print('=' * 70)
print()

print(f'{"Stimulus":<30} {"Pred a*":>8} {"Pred b*":>8} {"GT a*":>8} {"GT b*":>8} {"RMSE":>6}  Category')
print('-' * 95)

for name in stim_names:
    acts = database[name]
    pred = acts['color_pred']  # [2, H, W]
    gt_ab = acts['ab_gt']
    gt_small = cv2.resize(gt_ab, (pred.shape[2], pred.shape[1]))

    pred_a_mean = pred[0].mean()
    pred_b_mean = pred[1].mean()
    gt_a_mean = gt_small[:,:,0].mean()
    gt_b_mean = gt_small[:,:,1].mean()
    rmse = np.sqrt(np.mean((pred[0] - gt_small[:,:,0])**2 + (pred[1] - gt_small[:,:,1])**2))

    print(f'  {name:<30} {pred_a_mean:+7.1f} {pred_b_mean:+7.1f} '
          f'{gt_a_mean:+7.1f} {gt_b_mean:+7.1f} {rmse:5.1f}  {acts["category"]}')


# ================================================================
# ANALYSIS 5: NEURON→CONCEPT MAPPING (Expanded)
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 5: NEURON → CONCEPT MAPPING')
print('=' * 70)
print()

# For each neuron, compute correlation with stimulus categories
# Use one-hot encoding of categories
unique_cats = sorted(set(database[n]['category'] for n in stim_names))
cat_labels = np.array([unique_cats.index(database[n]['category']) for n in stim_names])

# For each neuron, which category does it most prefer?
neuron_preferences = []
for ni in range(n_neurons):
    rates = fire_rate_matrix[:, ni]
    if rates.std() < 0.01:
        neuron_preferences.append(('none', 0.0))
        continue
    # Correlation with each category
    best_cat = 'none'
    best_corr = 0
    for ci, cat in enumerate(unique_cats):
        indicator = (cat_labels == ci).astype(float)
        if indicator.std() < 0.01: continue
        r = np.corrcoef(rates, indicator)[0, 1]
        if abs(r) > abs(best_corr):
            best_corr = r
            best_cat = cat
    neuron_preferences.append((best_cat, best_corr))

# Count neurons per category preference
pref_counts = defaultdict(int)
strong_prefs = defaultdict(list)
for ni, (cat, corr) in enumerate(neuron_preferences):
    if abs(corr) > 0.3:
        pref_counts[cat] += 1
        strong_prefs[cat].append((ni, corr))

print('Neurons with strong category preference (|r| > 0.3):')
for cat in unique_cats + ['none']:
    n = pref_counts.get(cat, 0)
    print(f'  {cat:<15} {n:3d} neurons')
    if cat in strong_prefs:
        top = sorted(strong_prefs[cat], key=lambda x: -abs(x[1]))[:5]
        for ni, corr in top:
            print(f'    neuron {ni:3d}: r={corr:+.3f}')

# Total classified vs unclassified
classified = sum(1 for _, (_, c) in enumerate(neuron_preferences) if abs(c) > 0.3)
print(f'\n  Total classified (|r|>0.3): {classified}/{n_neurons} ({classified/n_neurons:.1%})')
print(f'  Unclassified: {n_neurons - classified}/{n_neurons}')


# ================================================================
# ANALYSIS 6: THE COMBINATION — Co-activation Patterns
# ================================================================
print()
print('=' * 70)
print('ANALYSIS 6: CO-ACTIVATION PATTERNS')
print('=' * 70)
print()

# Which neurons tend to fire together?
# Compute co-activation correlation matrix (using fire rates across stimuli)
# Focus on the selective neurons
selective_mask = fire_var > np.percentile(fire_var, 75)  # top 25% most variable
selective_idx = np.where(selective_mask)[0]
selective_rates = fire_rate_matrix[:, selective_idx]  # [N_stim, N_selective]

print(f'Analyzing {len(selective_idx)} most selective neurons (top 25% by variance)')

# Correlation between selective neurons
if len(selective_idx) > 1:
    corr_mat = np.corrcoef(selective_rates.T)
    np.fill_diagonal(corr_mat, 0)

    # Find strongly co-activated pairs
    strong_pos = np.sum(corr_mat > 0.7) // 2
    strong_neg = np.sum(corr_mat < -0.7) // 2
    print(f'  Strong co-activation (r>0.7): {strong_pos} pairs')
    print(f'  Strong anti-correlation (r<-0.7): {strong_neg} pairs')

    # Cluster selective neurons
    from sklearn.cluster import KMeans
    n_clusters = min(8, len(selective_idx))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(selective_rates.T)

    print(f'\n  {n_clusters} neuron clusters:')
    for ci in range(n_clusters):
        members = selective_idx[labels == ci]
        # What stimuli activate this cluster most?
        cluster_rates = fire_rate_matrix[:, members].mean(1)  # mean rate per stimulus
        top_stim_idx = np.argsort(cluster_rates)[::-1][:3]
        top_stims = [(stim_names[i], cluster_rates[i]) for i in top_stim_idx]

        print(f'    Cluster {ci}: {len(members)} neurons')
        for sname, rate in top_stims:
            print(f'      {sname:<30} rate={rate:.2f}')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print(f'Stimuli: {len(stimuli)} controlled inputs across {len(categories)} categories')
print(f'Feature directions: within-cat={np.mean(within_angles):.1f}° vs between-cat={np.mean(between_angles):.1f}°')
print(f'π/φ² = {phi2_angle:.1f}° — natural separator? within above: {within_above:.1%}, between above: {between_above:.1%}')
print(f'Selective neurons: {selective} with strong category preference')
print(f'The stethoscope database is ready for Phase 2 (direction catalog) and Phase 3 (W₁ decoding)')
