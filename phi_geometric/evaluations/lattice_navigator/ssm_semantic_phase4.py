"""
Phase 4: The Combination — Ablation and Sparsity of the Holographic Code

Phase 3 found: 768 tiny rotations, each orthogonal to the semantic answer.
Only the collective sum produces semantic content. 1-8 resonant neurons per category.

Phase 4 asks:
  1. If we ablate resonant neurons, does that category suffer?
  2. If we keep ONLY resonant neurons, can we still predict?
  3. How many neurons does each concept NEED? (sparsity)
  4. How much overlap between category neuron sets?
  5. Is the code truly holographic (all neurons needed) or sparse?
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


def run_encoder_with_ablation(v16, img_tensor, ablated_neurons=None, kept_neurons=None):
    """
    Run encoder with neuron ablation at stage 1, last block.
    ablated_neurons: set of neuron indices to zero out (post-GELU)
    kept_neurons: set of neuron indices to KEEP (zero everything else)
    Returns color prediction [2, H, W].
    """
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
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
                post_gelu = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                          v16._get_weight(f'{p}.pwconv1.bias')))

                # ABLATION: at stage 1, last block
                if si == 1 and bi == depths[si] - 1:
                    if ablated_neurons is not None:
                        mask = torch.ones(post_gelu.shape[-1])
                        for n in ablated_neurons:
                            mask[n] = 0
                        post_gelu = post_gelu * mask
                    elif kept_neurons is not None:
                        mask = torch.zeros(post_gelu.shape[-1])
                        for n in kept_neurons:
                            mask[n] = 1
                        post_gelu = post_gelu * mask

                x = F.linear(post_gelu, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()  # [2, H, W]


# ================================================================
# STEP 0: EXTRACT WEIGHTS AND BUILD SEMANTIC PROFILES
# ================================================================
print('=' * 70)
print('STEP 0: BUILD SEMANTIC PROFILES')
print('=' * 70)
print()

p = 'encoder.arch.stages.1.2'
W1 = v16._get_weight(f'{p}.pwconv1.weight').detach().numpy()  # [768, 192]
W2 = v16._get_weight(f'{p}.pwconv2.weight').detach().numpy()  # [192, 768]

W1_norms = np.linalg.norm(W1, axis=1)
W1_unit = W1 / (W1_norms[:, None] + 1e-8)
W2_cols = W2.T  # [768, 192]
W2_norms = np.linalg.norm(W2_cols, axis=1)
W2_unit = W2_cols / (W2_norms[:, None] + 1e-8)

def classify_color(a, b):
    sat = np.sqrt(a**2 + b**2)
    if sat < 5: return 'neutral'
    hue = np.degrees(np.arctan2(b, a))
    if hue < 0: hue += 360
    sector = int(hue / 30) % 12
    names = ['red', 'orange', 'yellow', 'chartreuse', 'green', 'spring',
             'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose']
    return names[sector]

# Build semantic centroids from 60 images
cat_residuals = defaultdict(list)
for img_idx in range(20, 300):
    if len(cat_residuals.get('neutral', [])) > 5000: break
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue

    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    with torch.no_grad():
        gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
        m_t = torch.tensor([.485,.456,.406]).view(1,3,1,1)
        s_t = torch.tensor([.229,.224,.225]).view(1,3,1,1)
        x = (t - m_t) / s_t
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(2):
            d = dims[si]
            if si > 0:
                pp = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{pp}.0.weight'), v16._get_weight(f'{pp}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{pp}.1.weight'), v16._get_weight(f'{pp}.1.bias'), stride=2)
            for bi in range(depths[si]):
                pp = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{pp}.dwconv.weight'),
                             v16._get_weight(f'{pp}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{pp}.norm.weight'), v16._get_weight(f'{pp}.norm.bias'))
                x = gate(F.linear(x, v16._get_weight(f'{pp}.pwconv1.weight'),
                                  v16._get_weight(f'{pp}.pwconv1.bias')))
                x = F.linear(x, v16._get_weight(f'{pp}.pwconv2.weight'),
                             v16._get_weight(f'{pp}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{pp}.gamma').view(1,-1,1,1) * x
            if si == 1:
                xn = x.permute(0,2,3,1)
                xn = F.layer_norm(xn, (d,),
                                  v16._get_weight(f'encoder.arch.norm{si}.weight'),
                                  v16._get_weight(f'encoder.arch.norm{si}.bias'))
                feats = xn.squeeze(0).detach().numpy()

    h, w = feats.shape[:2]
    flat = feats.reshape(-1, 192)
    struct = flat.mean(0, keepdims=True)
    residuals = flat - struct
    ab_small = cv2.resize(ab, (w, h)).reshape(-1, 2)
    for j in range(len(flat)):
        cat = classify_color(ab_small[j, 0], ab_small[j, 1])
        cat_residuals[cat].append(residuals[j])

cat_dirs = {}
cats = []
for cat in sorted(cat_residuals.keys()):
    if len(cat_residuals[cat]) < 100:
        continue
    arr = np.array(cat_residuals[cat])
    mean = arr.mean(0)
    norm = np.linalg.norm(mean)
    if norm > 1e-8:
        cat_dirs[cat] = mean / norm
        cats.append(cat)

print(f'{len(cats)} semantic categories')

# Build semantic alignment matrix
centroid_matrix = np.array([cat_dirs[c] for c in cats])
read_align = W1_unit @ centroid_matrix.T   # [768, N_cats]
write_align = W2_unit @ centroid_matrix.T  # [768, N_cats]

# For each category, rank neurons by "resonance score" = read_align * write_align
resonance = read_align * write_align  # [768, N_cats]

print('Top resonant neurons per category:')
for ci, cat in enumerate(cats):
    top = np.argsort(np.abs(resonance[:, ci]))[::-1][:5]
    scores = [f'n{n}({resonance[n,ci]:+.3f})' for n in top]
    print(f'  {cat:<12} {" ".join(scores)}')


# ================================================================
# STEP 1: ABLATION — Remove neurons and measure color prediction
# ================================================================
print()
print('=' * 70)
print('STEP 1: ABLATION EXPERIMENTS')
print('=' * 70)
print()

# Test images
test_images = []
for img_idx in range(300, 400):
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    test_images.append({'tensor': t, 'ab': ab, 'idx': img_idx})
    if len(test_images) >= 10: break

print(f'Testing on {len(test_images)} images')

def eval_ablation(test_images, ablated=None, kept=None, name=''):
    """Evaluate color prediction with ablation."""
    rmses = []
    for ti in test_images:
        pred = run_encoder_with_ablation(v16, ti['tensor'], ablated_neurons=ablated, kept_neurons=kept)
        ab_pred_size = cv2.resize(ti['ab'], (pred.shape[2], pred.shape[1]))
        rmse = np.sqrt(np.mean((pred[0] - ab_pred_size[:,:,0])**2 +
                               (pred[1] - ab_pred_size[:,:,1])**2))
        rmses.append(rmse)
    mean_rmse = np.mean(rmses)
    return mean_rmse

# Baseline: no ablation
baseline_rmse = eval_ablation(test_images, name='baseline')
print(f'\n  Baseline (no ablation):    RMSE = {baseline_rmse:.2f}')

# Ablate ALL neurons (zero the entire block)
all_neurons = set(range(768))
all_ablated_rmse = eval_ablation(test_images, ablated=all_neurons, name='all ablated')
print(f'  All neurons ablated:       RMSE = {all_ablated_rmse:.2f}')

# Ablate random subsets of neurons
for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
    n_ablate = int(768 * frac)
    np.random.seed(42)
    ablated = set(np.random.choice(768, n_ablate, replace=False))
    rmse = eval_ablation(test_images, ablated=ablated)
    delta = (rmse - baseline_rmse) / baseline_rmse * 100
    print(f'  Random {frac:.0%} ablated ({n_ablate}): RMSE = {rmse:.2f} ({delta:+.1f}%)')

# Keep only random subsets
print()
for frac in [0.1, 0.25, 0.5]:
    n_keep = int(768 * frac)
    np.random.seed(42)
    kept = set(np.random.choice(768, n_keep, replace=False))
    rmse = eval_ablation(test_images, kept=kept)
    delta = (rmse - baseline_rmse) / baseline_rmse * 100
    print(f'  Keep random {frac:.0%} ({n_keep}):     RMSE = {rmse:.2f} ({delta:+.1f}%)')


# ================================================================
# STEP 2: TARGETED ABLATION — Remove category-specific neurons
# ================================================================
print()
print('=' * 70)
print('STEP 2: TARGETED ABLATION — Category-Specific')
print('=' * 70)
print()

# For each category, find top-N resonant neurons and ablate them
focus_cats = ['red', 'blue', 'green', 'yellow', 'neutral', 'orange']

for cat in focus_cats:
    if cat not in cats: continue
    ci = cats.index(cat)

    # Top resonant neurons for this category
    res_scores = np.abs(resonance[:, ci])
    top_neurons = np.argsort(res_scores)[::-1]

    print(f'  {cat}:')
    for n_ablate in [5, 10, 20, 50]:
        ablated = set(top_neurons[:n_ablate])
        rmse = eval_ablation(test_images, ablated=ablated)
        delta = (rmse - baseline_rmse) / baseline_rmse * 100
        print(f'    Ablate top-{n_ablate:2d} resonant: RMSE = {rmse:.2f} ({delta:+.1f}%)')

    # Keep ONLY top-N resonant
    for n_keep in [10, 50, 100, 200]:
        kept = set(top_neurons[:n_keep])
        rmse = eval_ablation(test_images, kept=kept)
        delta = (rmse - baseline_rmse) / baseline_rmse * 100
        print(f'    Keep  top-{n_keep:3d} resonant: RMSE = {rmse:.2f} ({delta:+.1f}%)')
    print()


# ================================================================
# STEP 3: SMART ABLATION — Neurons ranked by overall importance
# ================================================================
print()
print('=' * 70)
print('STEP 3: NEURON IMPORTANCE RANKING')
print('=' * 70)
print()

# Rank neurons by: norm(W₁_row) * norm(W₂_col) (throughput)
throughput = W1_norms * W2_norms  # [768]
throughput_rank = np.argsort(throughput)[::-1]

# Rank by max resonance score across ALL categories
max_resonance = np.max(np.abs(resonance), axis=1)  # [768]
resonance_rank = np.argsort(max_resonance)[::-1]

# Rank by mean activation level (from data — which neurons fire most?)
# Use the first few test images
mean_fire_rates = np.zeros(768)
n_counted = 0
for ti in test_images[:5]:
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m_t = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s_t = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (ti['tensor'] - m_t) / s_t
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
                pp = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{pp}.0.weight'), v16._get_weight(f'{pp}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{pp}.1.weight'), v16._get_weight(f'{pp}.1.bias'), stride=2)
            for bi in range(depths[si]):
                pp = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{pp}.dwconv.weight'),
                             v16._get_weight(f'{pp}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{pp}.norm.weight'), v16._get_weight(f'{pp}.norm.bias'))
                pre = F.linear(x, v16._get_weight(f'{pp}.pwconv1.weight'),
                               v16._get_weight(f'{pp}.pwconv1.bias'))
                post = gate(pre)
                if si == 1 and bi == depths[si] - 1:
                    fr = (post.squeeze(0).detach().numpy() > 0).reshape(-1, 768).mean(0)
                    mean_fire_rates += fr
                    n_counted += 1
                x = F.linear(post, v16._get_weight(f'{pp}.pwconv2.weight'),
                             v16._get_weight(f'{pp}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{pp}.gamma').view(1,-1,1,1) * x

mean_fire_rates /= max(n_counted, 1)
activity_rank = np.argsort(mean_fire_rates)[::-1]

# Combined importance: fire_rate × throughput × max_resonance
combined_importance = mean_fire_rates * throughput * (max_resonance + 0.01)
combined_rank = np.argsort(combined_importance)[::-1]

print('Ablation by different importance rankings:')
print(f'  {"Ranking":<25} {"Keep 100":<12} {"Keep 200":<12} {"Keep 384":<12} {"Keep 500":<12}')
print(f'  {"-"*65}')

for rank_name, rank in [('Throughput', throughput_rank),
                         ('Resonance', resonance_rank),
                         ('Activity', activity_rank),
                         ('Combined', combined_rank),
                         ('Random', np.random.permutation(768))]:
    results = []
    for n_keep in [100, 200, 384, 500]:
        kept = set(rank[:n_keep])
        rmse = eval_ablation(test_images, kept=kept)
        results.append(f'{rmse:.2f}')
    print(f'  {rank_name:<25} {"  ".join(results)}')

print(f'\n  Baseline: {baseline_rmse:.2f}')


# ================================================================
# STEP 4: OVERLAP ANALYSIS — Shared neurons between categories
# ================================================================
print()
print('=' * 70)
print('STEP 4: CATEGORY OVERLAP — How shared is the code?')
print('=' * 70)
print()

# Top-50 resonant neurons per category
top_n = 50
cat_neuron_sets = {}
for ci, cat in enumerate(cats):
    top = set(np.argsort(np.abs(resonance[:, ci]))[::-1][:top_n])
    cat_neuron_sets[cat] = top

# Overlap matrix
print(f'Overlap in top-{top_n} resonant neurons between categories:')
focus = [c for c in focus_cats if c in cats]
header = f'{"":>10}'
for c in focus: header += f'{c:>10}'
print(header)

for ci in focus:
    row = f'{ci:>10}'
    for cj in focus:
        overlap = len(cat_neuron_sets[ci] & cat_neuron_sets[cj])
        if ci == cj:
            row += f'{"---":>10}'
        else:
            row += f'{overlap:>10}'
    print(row)

# Union size
all_used = set()
for cat in focus:
    all_used |= cat_neuron_sets[cat]
print(f'\n  Union of all top-{top_n} sets: {len(all_used)}/768 neurons')
print(f'  Coverage: {len(all_used)/768:.1%}')

# How many categories does each neuron serve?
neuron_cat_count = np.zeros(768)
for cat in cats:
    for n in cat_neuron_sets[cat]:
        neuron_cat_count[n] += 1

print(f'\n  Neurons serving N categories (out of top-{top_n} per cat):')
for k in range(int(neuron_cat_count.max()) + 1):
    count = (neuron_cat_count == k).sum()
    if count > 0:
        print(f'    {k} categories: {count} neurons')


# ================================================================
# STEP 5: THE SPARSITY CURVE — How many neurons for X% performance?
# ================================================================
print()
print('=' * 70)
print('STEP 5: SPARSITY CURVE — How Many Neurons Are Needed?')
print('=' * 70)
print()

# Use combined importance ranking
print(f'Keep top-K by combined importance:')
print(f'  {"K":<8} {"RMSE":<10} {"Delta":<10} {"% of 768"}')
print(f'  {"-"*35}')

for k in [10, 25, 50, 100, 150, 200, 300, 384, 500, 600, 700, 768]:
    if k >= 768:
        rmse = baseline_rmse
    else:
        kept = set(combined_rank[:k])
        rmse = eval_ablation(test_images, kept=kept)
    delta = (rmse - baseline_rmse) / baseline_rmse * 100
    pct = k / 768 * 100
    print(f'  {k:<8} {rmse:<10.2f} {delta:+8.1f}%  {pct:5.1f}%')


print()
print('=' * 70)
print('SUMMARY — Phase 4')
print('=' * 70)
print()
print('The lock combination:')
print(f'  • Baseline RMSE: {baseline_rmse:.2f}')
print(f'  • All neurons ablated: {all_ablated_rmse:.2f}')
print(f'  • The code is {"holographic (all needed)" if all_ablated_rmse > baseline_rmse * 1.5 else "sparse (subset sufficient)"}')
print()
print('Each category needs its resonant neurons, but there is massive overlap.')
print('The holographic code is distributed but not uniformly — some neurons')
print('contribute more than others, and importance can be ranked.')
