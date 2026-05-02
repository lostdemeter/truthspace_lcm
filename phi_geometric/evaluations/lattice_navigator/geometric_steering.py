"""
Geometric Steering Explorer

We've proven: every colorization = a point in activation-map space.
GT is reachable via rank-2 correction. Now we explore:

1. WHERE in the pipeline does the DDColor→GT error enter?
2. Can we steer at intermediate transformer layers?
3. Is the correction spatially structured (edge-aligned)?
4. Can we predict correction coefficients from geometric features?
5. What does the steering geometry teach us about how shape stores info?
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


def full_forward_with_layer_snapshots(v16, img_tensor):
    """
    Run DDColor forward pass, capturing the query state after each
    transformer layer. Returns snapshots at all 9 layers + final output.
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

        x_list = [out0, out1, out2]
        src, pos = [], []
        for i, xl in enumerate(x_list):
            proj = F.conv2d(xl,
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.weight'),
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.bias'))
            src.append(proj.flatten(2).permute(2, 0, 1))
            pe = v16.pe_layer(proj)
            pos.append(pe.flatten(2).permute(2, 0, 1))

        for i in range(3):
            src[i] = src[i] + v16._get_weight('decoder.color_decoder.level_embed.weight')[i]

        bs = src[0].shape[1]
        query_embed = v16._get_weight('decoder.color_decoder.query_embed.weight').unsqueeze(1).repeat(1, bs, 1)
        output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, bs, 1)

        snapshots = []
        snapshots.append(('init', output.clone()))

        for layer_i in range(9):
            level_index = layer_i % 3
            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer_i}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, src[level_index] + pos[level_index], src[level_index],
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer_i}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, output + query_embed, output,
                v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            prefix = f'decoder.color_decoder.transformer_ffn_layers.{layer_i}'
            ffn_out = F.relu(F.linear(output,
                                      v16._get_weight(f'{prefix}.linear1.weight'),
                                      v16._get_weight(f'{prefix}.linear1.bias')))
            ffn_out = F.linear(ffn_out,
                               v16._get_weight(f'{prefix}.linear2.weight'),
                               v16._get_weight(f'{prefix}.linear2.bias'))
            output = F.layer_norm(output + ffn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            snapshots.append((f'layer_{layer_i}', output.clone()))

        # Convert each snapshot to color maps + ab output
        decoder_norm_w = v16._get_weight('decoder.color_decoder.decoder_norm.weight')
        decoder_norm_b = v16._get_weight('decoder.color_decoder.decoder_norm.bias')

        results = []
        for name, snap in snapshots:
            normed = F.layer_norm(snap, (256,), decoder_norm_w, decoder_norm_b).transpose(0, 1)
            x_c = normed
            for i in range(3):
                x_c = F.linear(x_c,
                             v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                             v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
                if i < 2:
                    x_c = F.relu(x_c)

            color_maps = torch.einsum('bqc,bchw->bqhw', x_c, out3)
            coarse_input = torch.cat([color_maps, img_tensor], dim=1)
            ab_out = F.conv2d(coarse_input,
                              v16._get_weight('refine_net.0.0.weight'),
                              v16._get_weight('refine_net.0.0.bias'))

            results.append({
                'name': name,
                'query_state': snap.squeeze(1).permute(1, 0).detach().numpy(),  # [100, 256]
                'color_maps': color_maps.squeeze(0).detach().numpy(),  # [100, H, W]
                'ab': ab_out.squeeze(0).permute(1, 2, 0).detach().numpy(),  # [H, W, 2]
            })

        return results, out3.detach()


print('=== GEOMETRIC STEERING EXPLORER ===\n')

v16 = V16GeometricColorizer()

refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()
color_wheel = refine_w[:, :100].T  # [100, 2]

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56]]

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/steering'
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PART 1: Where does the error enter?
# Track query states and ab error at every layer
# ============================================================
print('=== PART 1: Error Propagation Through Layers ===\n')

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    results, out3 = full_forward_with_layer_snapshots(v16, img_tensor)

    print(f'  {name}:')
    print(f'    {"Layer":<12} {"ab_err":>8} {"Δ_err":>8} {"query_norm":>12} {"query_Δ":>10}')
    print(f'    {"-"*52}')

    prev_err = None
    prev_state = None
    for res in results:
        err = np.sqrt(np.mean((res['ab'] - ab_gt)**2))
        q_norm = np.linalg.norm(res['query_state'])

        delta_err = err - prev_err if prev_err is not None else 0
        delta_q = np.linalg.norm(res['query_state'] - prev_state) if prev_state is not None else 0

        print(f'    {res["name"]:<12} {err:8.2f} {delta_err:+8.2f} {q_norm:12.1f} {delta_q:10.1f}')

        prev_err = err
        prev_state = res['query_state']

    # Save layer-by-layer colorization strip
    frames = []
    for res in results:
        bgr = ab_to_bgr(res['ab'], L)
        err = np.sqrt(np.mean((res['ab'] - ab_gt)**2))
        label = f'{res["name"]} e={err:.1f}'
        cv2.putText(bgr, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 2)
        cv2.putText(bgr, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 1)
        frames.append(bgr)
    frames.append(r.copy())
    cv2.putText(frames[-1], 'GT', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 2)
    cv2.putText(frames[-1], 'GT', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 1)

    strip = np.hstack(frames)
    cv2.imwrite(os.path.join(out_dir, f'layers_{name}.jpg'), strip)
    print()


# ============================================================
# PART 2: Query vector steering — can we nudge individual
# query vectors at a specific layer to reach GT?
# ============================================================
print('=== PART 2: Single-Layer Query Steering ===\n')

for img_path in test_paths[:2]:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    results, out3 = full_forward_with_layer_snapshots(v16, img_tensor)
    final_ab = results[-1]['ab']
    err_baseline = np.sqrt(np.mean((final_ab - ab_gt)**2))

    # For each layer, compute: what query state CHANGE at that layer
    # would be needed to produce GT from the final output?
    # This is the final activation map correction, projected back
    # through the color_embed MLP to query space.
    
    # We can only easily steer at the FINAL layer (post-transformer),
    # because the MLP is nonlinear. So let's measure: how much does
    # each layer's query state change contribute to the final error?

    print(f'  {name} (baseline err={err_baseline:.2f}):')

    # Measure: query state DIFFERENCE between consecutive layers
    # and its correlation with the final error
    final_maps = results[-1]['color_maps']  # [100, H, W]

    # The correction needed in color-map space
    Wt = color_wheel.T  # [2, 100]
    WWT_inv = np.linalg.inv(Wt @ Wt.T)
    pinv = Wt.T @ WWT_inv  # [100, 2]
    delta_ab = (ab_gt - final_ab).reshape(-1, 2)
    delta_maps_needed = (delta_ab @ pinv.T).T.reshape(100, SZ, SZ)

    # For each pair of adjacent layers, measure the query state change
    # and see how it correlates with the needed correction
    print(f'    {"Transition":<20} {"query_Δ":>10} {"map_Δ_corr":>12} {"error_Δ":>10}')
    print(f'    {"-"*55}')

    for i in range(len(results) - 1):
        q_delta = results[i+1]['query_state'] - results[i]['query_state']  # [100, 256]
        map_delta = results[i+1]['color_maps'] - results[i]['color_maps']  # [100, H, W]

        # Correlation between this layer's map change and the needed correction
        flat_delta = map_delta.flatten()
        flat_needed = delta_maps_needed.flatten()
        if np.std(flat_delta) > 0 and np.std(flat_needed) > 0:
            corr = np.corrcoef(flat_delta, flat_needed)[0, 1]
        else:
            corr = 0.0

        err_at_i = np.sqrt(np.mean((results[i]['ab'] - ab_gt)**2))
        err_at_i1 = np.sqrt(np.mean((results[i+1]['ab'] - ab_gt)**2))

        transition = f'{results[i]["name"]}→{results[i+1]["name"]}'
        print(f'    {transition:<20} {np.linalg.norm(q_delta):10.1f} {corr:12.3f} {err_at_i1-err_at_i:+10.2f}')

    print()


# ============================================================
# PART 3: Spatial structure — is the correction edge-aligned?
# ============================================================
print('=== PART 3: Is Correction Spatially Structured? ===\n')

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    results, _ = full_forward_with_layer_snapshots(v16, img_tensor)
    final_ab = results[-1]['ab']

    # Correction in ab space
    delta_ab = ab_gt - final_ab  # [H, W, 2]

    # Edge regions
    labeled, edges = segment_by_edges(gray)
    n_regions = len(np.unique(labeled)) - 1

    # For each region, compute mean correction
    region_means_a = ndimage.mean(delta_ab[:,:,0], labeled, range(1, n_regions + 1))
    region_means_b = ndimage.mean(delta_ab[:,:,1], labeled, range(1, n_regions + 1))

    # Reconstruct piecewise-constant correction from region means
    ab_piecewise = np.zeros_like(delta_ab)
    for idx, rid in enumerate(range(1, n_regions + 1)):
        mask = labeled == rid
        ab_piecewise[:,:,0][mask] = region_means_a[idx]
        ab_piecewise[:,:,1][mask] = region_means_b[idx]

    # How much of the correction is captured by piecewise-constant (edge-aligned)?
    total_var = np.var(delta_ab)
    residual_var = np.var(delta_ab - ab_piecewise)
    piecewise_explained = (1 - residual_var / (total_var + 1e-10)) * 100

    # Apply piecewise correction to DDColor
    ab_steered = final_ab + ab_piecewise
    err_steered = np.sqrt(np.mean((ab_steered - ab_gt)**2))
    err_baseline = np.sqrt(np.mean((final_ab - ab_gt)**2))
    gap_closed = (1 - err_steered / err_baseline) * 100

    print(f'  {name}: {n_regions} regions')
    print(f'    Baseline err: {err_baseline:.2f}')
    print(f'    Piecewise correction explains: {piecewise_explained:.1f}% of spatial correction')
    print(f'    After piecewise steering: err={err_steered:.2f}, gap_closed={gap_closed:.1f}%')

    # Per-region correction magnitude distribution
    region_mags = np.sqrt(np.array(region_means_a)**2 + np.array(region_means_b)**2)
    print(f'    Region correction magnitudes: mean={region_mags.mean():.2f}, '
          f'max={region_mags.max():.2f}, std={region_mags.std():.2f}')

    # Are the corrections organized? Check: do bright/dark regions get different corrections?
    region_brightness = ndimage.mean(gray, labeled, range(1, n_regions + 1)) / 255.0
    corr_bright_a = np.corrcoef(region_brightness, region_means_a)[0, 1] if len(region_brightness) > 2 else 0
    corr_bright_b = np.corrcoef(region_brightness, region_means_b)[0, 1] if len(region_brightness) > 2 else 0
    print(f'    Brightness↔correction: r_a={corr_bright_a:.3f}, r_b={corr_bright_b:.3f}')

    # Save steering comparison
    bgr_dd = ab_to_bgr(final_ab, L)
    bgr_steered = ab_to_bgr(ab_steered, L)

    # Correction heatmap
    corr_mag = np.sqrt(delta_ab[:,:,0]**2 + delta_ab[:,:,1]**2)
    corr_vis = (corr_mag / (corr_mag.max() + 1e-8) * 255).astype(np.uint8)
    corr_color = cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET)

    imgs = [
        (bgr_dd, f'DDColor e={err_baseline:.1f}'),
        (corr_color, f'Correction'),
        (bgr_steered, f'Steered e={err_steered:.1f}'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'spatial_{name}.jpg'), strip)
    print()


# ============================================================
# PART 4: Feature-based correction prediction
# Can geometric features predict the per-region correction?
# ============================================================
print('=== PART 4: Predicting Correction from Geometric Features ===\n')

# Collect training data: per-region features → per-region correction
train_feats = []
train_corr = []

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    results, _ = full_forward_with_layer_snapshots(v16, img_tensor)
    final_ab = results[-1]['ab']
    delta_ab = ab_gt - final_ab

    labeled, edges = segment_by_edges(gray)

    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 20: continue

        ys, xs = np.where(mask)
        h, w = gray.shape

        brightness = gray[mask].mean() / 255.0
        brightness_std = gray[mask].std() / 255.0
        y_center = ys.mean() / h
        x_center = xs.mean() / w
        size = mask.sum() / (h * w)

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sx**2 + sy**2)
        edge_density = edge_mag[mask].mean() / (edge_mag.max() + 1e-8)

        # DDColor's current color for this region
        dd_a_mean = final_ab[:,:,0][mask].mean()
        dd_b_mean = final_ab[:,:,1][mask].mean()
        dd_sat = np.sqrt(dd_a_mean**2 + dd_b_mean**2)
        dd_angle = np.arctan2(dd_b_mean, dd_a_mean)

        feats = np.array([
            brightness, brightness_std, y_center, x_center, size,
            edge_density, dd_a_mean, dd_b_mean, dd_sat, dd_angle,
            brightness * dd_sat,  # interaction
            brightness**2,
        ])

        corr_a = delta_ab[:,:,0][mask].mean()
        corr_b = delta_ab[:,:,1][mask].mean()

        train_feats.append(feats)
        train_corr.append([corr_a, corr_b])

X = np.array(train_feats)
Y = np.array(train_corr)
X_bias = np.column_stack([X, np.ones(X.shape[0])])

print(f'Training: {X.shape[0]} regions, {X.shape[1]} features')

# Fit linear model
W_a, _, _, _ = np.linalg.lstsq(X_bias, Y[:, 0], rcond=None)
W_b, _, _, _ = np.linalg.lstsq(X_bias, Y[:, 1], rcond=None)

pred_a = X_bias @ W_a
pred_b = X_bias @ W_b
r2_a = 1 - np.mean((Y[:, 0] - pred_a)**2) / (np.var(Y[:, 0]) + 1e-8)
r2_b = 1 - np.mean((Y[:, 1] - pred_b)**2) / (np.var(Y[:, 1]) + 1e-8)
print(f'Linear model R²: a={r2_a:.3f}, b={r2_b:.3f}')

# Feature importance
feat_names = ['brightness', 'bright_std', 'y_pos', 'x_pos', 'size',
              'edge_dens', 'dd_a', 'dd_b', 'dd_sat', 'dd_angle',
              'bright×sat', 'bright²', 'bias']
print(f'\nFeature importance for correction prediction:')
for j, fn in enumerate(feat_names):
    importance = (abs(W_a[j]) + abs(W_b[j])) / 2
    print(f'  {fn:<12} |w|={importance:.4f}')

# Key question: does knowing DDColor's CURRENT color help predict the correction?
# (i.e., is the error systematic by color region?)
r2_nodd_a = 1 - np.mean((Y[:, 0] - X_bias[:, :6] @ W_a[:6])**2) / (np.var(Y[:, 0]) + 1e-8)
r2_nodd_b = 1 - np.mean((Y[:, 1] - X_bias[:, :6] @ W_b[:6])**2) / (np.var(Y[:, 1]) + 1e-8)
print(f'\nWithout DDColor features: R² a={r2_nodd_a:.3f}, b={r2_nodd_b:.3f}')
print(f'With DDColor features:    R² a={r2_a:.3f}, b={r2_b:.3f}')
print(f'DDColor features add:     Δa={r2_a-r2_nodd_a:.3f}, Δb={r2_b-r2_nodd_b:.3f}')


# ============================================================
# PART 5: Apply learned correction to get steered colorization
# ============================================================
print('\n=== PART 5: Steered Colorization Results ===\n')

# Test on 4 more images
extra_paths = [all_imgs[i] for i in [58, 60, 62, 64]]

for img_path in extra_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')

    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab_gt = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab_gt[:,:,0]
    ab_gt = lab_gt[:,:,1:].astype(float) - 128.0

    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    results, _ = full_forward_with_layer_snapshots(v16, img_tensor)
    final_ab = results[-1]['ab']

    # Apply learned correction per region
    labeled, edges = segment_by_edges(gray)
    ab_steered = final_ab.copy()

    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 20: continue

        ys, xs = np.where(mask)
        h, w = gray.shape

        brightness = gray[mask].mean() / 255.0
        brightness_std = gray[mask].std() / 255.0
        y_center = ys.mean() / h
        x_center = xs.mean() / w
        size = mask.sum() / (h * w)

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sx**2 + sy**2)
        edge_density = edge_mag[mask].mean() / (edge_mag.max() + 1e-8)

        dd_a_mean = final_ab[:,:,0][mask].mean()
        dd_b_mean = final_ab[:,:,1][mask].mean()
        dd_sat = np.sqrt(dd_a_mean**2 + dd_b_mean**2)
        dd_angle = np.arctan2(dd_b_mean, dd_a_mean)

        feats = np.array([
            brightness, brightness_std, y_center, x_center, size,
            edge_density, dd_a_mean, dd_b_mean, dd_sat, dd_angle,
            brightness * dd_sat, brightness**2, 1.0
        ])

        pred_corr_a = feats @ W_a
        pred_corr_b = feats @ W_b

        ab_steered[:,:,0][mask] += pred_corr_a
        ab_steered[:,:,1][mask] += pred_corr_b

    err_dd = np.sqrt(np.mean((final_ab - ab_gt)**2))
    err_steered = np.sqrt(np.mean((ab_steered - ab_gt)**2))
    gap_closed = (1 - err_steered / err_dd) * 100

    bgr_dd = ab_to_bgr(final_ab, L)
    bgr_steered = ab_to_bgr(ab_steered, L)

    print(f'  {name}: DDColor={err_dd:.2f}, steered={err_steered:.2f}, gap_closed={gap_closed:.1f}%')

    imgs = [
        (bgr_dd, f'DDColor e={err_dd:.1f}'),
        (bgr_steered, f'Steered e={err_steered:.1f} ({gap_closed:.0f}%)'),
        (r, 'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'steered_{name}.jpg'), strip)


# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY ===\n')
print('1. Error propagation: tracked through 9 transformer layers')
print('2. Layer transitions: measured correlation with needed correction')
print('3. Spatial structure: tested piecewise-constant edge alignment')
print('4. Feature prediction: linear model using geometric + DDColor features')
print('5. Steered colorization: applied predicted corrections to held-out images')
print(f'\nOutput saved to: {out_dir}/')
print('Done!')
