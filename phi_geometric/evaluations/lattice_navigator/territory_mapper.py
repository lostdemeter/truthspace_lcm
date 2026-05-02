"""
Territory Mapper: Edge Regions → Color Wheel

We know from the rank-2 analysis that DDColor's entire output is:
    ab(pixel) = Σ_q  activation(q, pixel) × [w_a(q), w_b(q)]

Where the 200 numbers [w_a, w_b] are FIXED (the universal color wheel).
The 55M parameters only compute: which query claims which pixel.

Our approach:
1. Segment image into edge-bounded regions (≈ DDColor's query territories)
2. Extract geometric features per region (brightness, texture, position, edges)
3. Map each region to the BEST color wheel position
4. Apply: region × color_direction → ab output

The mapping is the key question: given a region's geometric features,
which of the 100 color wheel directions should it get?

We test two approaches:
A. GEOMETRIC: use feature→color_direction mapping derived from image statistics
B. ORACLE: use DDColor's actual territory assignment as upper bound
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

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

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()


def extract_region_features(gray, labeled, region_id):
    """Extract geometric features for a single region."""
    mask = labeled == region_id
    h, w = gray.shape
    
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    
    brightness = gray[mask].mean() / 255.0
    brightness_std = gray[mask].std() / 255.0
    
    # Position
    y_center = ys.mean() / h
    x_center = xs.mean() / w
    
    # Size (fraction of image)
    size = mask.sum() / (h * w)
    
    # Texture (local variance)
    local_var = gray[mask].astype(float).var() / (128**2)
    
    # Edge density within region
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sx**2 + sy**2)
    edge_density = edges[mask].mean() / (edges.max() + 1e-8)
    
    # Brightness relative to neighbors
    dilated = ndimage.binary_dilation(mask, iterations=3)
    border = dilated & ~mask
    if border.sum() > 0:
        neighbor_brightness = gray[border].mean() / 255.0
        brightness_contrast = brightness - neighbor_brightness
    else:
        brightness_contrast = 0.0
    
    # Compactness (perimeter² / area) — blob vs. elongated
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        area = mask.sum()
        compactness = (perimeter**2) / (4 * np.pi * area + 1e-8)
    else:
        compactness = 1.0
    
    return np.array([
        brightness,           # 0: how bright
        brightness_std,       # 1: brightness variation within region
        y_center,             # 2: vertical position
        x_center,             # 3: horizontal position
        size,                 # 4: region size
        local_var,            # 5: texture
        edge_density,         # 6: internal edge density
        brightness_contrast,  # 7: contrast with neighbors
        compactness,          # 8: shape compactness
    ])


def get_ddcolor_territories(v16, img_tensor):
    """Run DDColor and return per-pixel query assignments + color maps."""
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
        
        decoder_output = F.layer_norm(
            output, (256,),
            v16._get_weight('decoder.color_decoder.decoder_norm.weight'),
            v16._get_weight('decoder.color_decoder.decoder_norm.bias')).transpose(0, 1)
        
        x_color = decoder_output
        for i in range(3):
            x_color = F.linear(x_color,
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
            if i < 2:
                x_color = F.relu(x_color)
        
        color_maps = torch.einsum('bqc,bchw->bqhw', x_color, out3)
        
        # Full ab output
        coarse_input = torch.cat([color_maps, img_tensor], dim=1)
        ab_out = F.conv2d(coarse_input,
                          v16._get_weight('refine_net.0.0.weight'),
                          v16._get_weight('refine_net.0.0.bias'))
    
    return color_maps.squeeze(0).numpy(), ab_out.squeeze(0).permute(1, 2, 0).numpy()


print('=== TERRITORY MAPPER: Edge Regions → Color Wheel ===')
print()

v16 = V16GeometricColorizer()

# Load the 200 color wheel numbers
refine_w = v16._get_weight('refine_net.0.0.weight').numpy().reshape(2, 103)
color_wheel = refine_w[:, :100].T  # [100, 2] — each query's (w_a, w_b)

print(f'Color wheel: {color_wheel.shape} (100 queries × 2 ab channels)')
print(f'  Magnitude range: [{np.linalg.norm(color_wheel, axis=1).min():.4f}, '
      f'{np.linalg.norm(color_wheel, axis=1).max():.4f}]')

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/territory_mapping'
os.makedirs(out_dir, exist_ok=True)

# Phase 1: Collect training data — for each edge region, what query does DDColor assign?
print('\n=== Phase 1: Learning region→query mapping from DDColor ===')

train_paths = [all_imgs[i] for i in range(50, 66)]
test_paths = [all_imgs[i] for i in range(66, 74)]

train_features = []  # [N_regions, 9] geometric features per region
train_query_ids = [] # [N_regions] which DDColor query dominates this region
train_activations = []  # [N_regions] how strongly the dominant query activates

for img_path in train_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    
    # Edge segmentation (our geometric territories)
    labeled, edges = segment_by_edges(gray)
    
    # DDColor's query assignments
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps, ab_ddcolor = get_ddcolor_territories(v16, img_tensor)
    # color_maps: [100, H, W] — activation of each query at each pixel
    
    # For each edge region, find which DDColor query dominates
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 15: continue
        
        # Extract geometric features
        feats = extract_region_features(gray, labeled, rid)
        if feats is None: continue
        
        # Find DDColor's dominant query for this region
        # Average the absolute activation of each query within this region
        region_activations = np.abs(color_maps[:, mask]).mean(axis=1)  # [100]
        dominant_q = np.argmax(region_activations)
        dominant_activation = region_activations[dominant_q]
        
        train_features.append(feats)
        train_query_ids.append(dominant_q)
        train_activations.append(dominant_activation)
    
    print(f'  {name}: {len(np.unique(labeled))-1} regions')

X_train = np.array(train_features)
Y_train = np.array(train_query_ids)
A_train = np.array(train_activations)

print(f'\nTraining data: {X_train.shape[0]} regions, {len(np.unique(Y_train))} unique queries used')

# Phase 2: Build the mapping — what geometric features predict which query?
print('\n=== Phase 2: Building feature → query mapping ===')

# Instead of predicting the query ID (100-way classification),
# predict the query's COLOR DIRECTION (2D regression)
# This is more geometric — we predict WHERE on the color wheel, not WHICH index

# For each training region, the target is the dominant query's ab direction
Y_ab = color_wheel[Y_train]  # [N, 2] — the color direction of each region's query

# Scale by activation strength
Y_ab_scaled = Y_ab * A_train.reshape(-1, 1)

# Add interaction features
X_interact = np.column_stack([
    X_train,
    X_train[:, 0] * X_train[:, 2],  # brightness × y_pos
    X_train[:, 0] * X_train[:, 5],  # brightness × texture
    X_train[:, 2] * X_train[:, 5],  # y_pos × texture
    X_train[:, 0]**2,               # brightness²
    X_train[:, 2]**2,               # y_pos²
    X_train[:, 7] * X_train[:, 0],  # contrast × brightness
])

# Add bias
X_bias = np.column_stack([X_interact, np.ones(X_interact.shape[0])])

print(f'Feature matrix: {X_bias.shape}')

# Solve: X @ W = Y_ab_scaled
W_a, _, _, _ = np.linalg.lstsq(X_bias, Y_ab_scaled[:, 0], rcond=None)
W_b, _, _, _ = np.linalg.lstsq(X_bias, Y_ab_scaled[:, 1], rcond=None)

# Training accuracy
pred_a = X_bias @ W_a
pred_b = X_bias @ W_b
r2_a = 1 - np.mean((Y_ab_scaled[:, 0] - pred_a)**2) / (np.var(Y_ab_scaled[:, 0]) + 1e-8)
r2_b = 1 - np.mean((Y_ab_scaled[:, 1] - pred_b)**2) / (np.var(Y_ab_scaled[:, 1]) + 1e-8)
print(f'Training R²: a={r2_a:.3f}, b={r2_b:.3f}')

# Feature importance
feat_names = ['brightness', 'bright_std', 'y_pos', 'x_pos', 'size', 'texture',
              'edge_density', 'bright_contrast', 'compactness',
              'bright×ypos', 'bright×tex', 'ypos×tex', 'bright²', 'ypos²', 'contrast×bright', 'bias']
print(f'\nFeature importance:')
for j, fn in enumerate(feat_names):
    importance = (abs(W_a[j]) + abs(W_b[j])) / 2
    print(f'  {fn:<18} |w|={importance:.5f}')

# Phase 3: Apply to images and generate colorization
print('\n=== Phase 3: Territory-Mapped Colorization ===')

def territory_colorize(gray, W_a, W_b, color_wheel):
    """Colorize using edge regions mapped to color wheel positions."""
    h, w = gray.shape
    labeled, edges = segment_by_edges(gray)
    
    ab_out = np.zeros((h, w, 2))
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 5: continue
        
        feats = extract_region_features(gray, labeled, rid)
        if feats is None: continue
        
        # Build feature vector with interactions
        f_interact = np.concatenate([
            feats,
            [feats[0] * feats[2], feats[0] * feats[5], feats[2] * feats[5],
             feats[0]**2, feats[2]**2, feats[7] * feats[0]],
        ])
        f_bias = np.append(f_interact, 1.0)
        
        # Predict color direction
        pred_wa = f_bias @ W_a
        pred_wb = f_bias @ W_b
        
        # Apply to region
        ab_out[:,:,0][mask] = pred_wa * 50  # Scale up from weight-space to ab-space
        ab_out[:,:,1][mask] = pred_wb * 50
    
    # Smooth boundaries
    for ch in range(2):
        ab_out[:,:,ch] = cv2.bilateralFilter(ab_out[:,:,ch].astype(np.float32), 9, 30, 30)
    
    return ab_out


# Also build oracle version: use DDColor's ACTUAL per-region assignment
def oracle_colorize(gray, color_maps, color_wheel):
    """Oracle: use DDColor's actual query assignments per edge region."""
    h, w = gray.shape
    labeled, edges = segment_by_edges(gray)
    
    ab_out = np.zeros((h, w, 2))
    
    for rid in np.unique(labeled):
        if rid == 0: continue
        mask = labeled == rid
        if mask.sum() < 5: continue
        
        # Get DDColor's average activation per query for this region
        region_acts = color_maps[:, mask].mean(axis=1)  # [100]
        
        # Weighted sum: Σ activation × color_direction
        ab_out[:,:,0][mask] = (region_acts * color_wheel[:, 0]).sum()
        ab_out[:,:,1][mask] = (region_acts * color_wheel[:, 1]).sum()
    
    for ch in range(2):
        ab_out[:,:,ch] = cv2.bilateralFilter(ab_out[:,:,ch].astype(np.float32), 9, 30, 30)
    
    return ab_out


# Test on held-out images
print('\nTesting on held-out images...\n')

results = []
for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0]
    
    # DDColor reference
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    color_maps, ab_ddcolor = get_ddcolor_territories(v16, img_tensor)
    
    # Geometric territory mapping
    ab_territory = territory_colorize(gray, W_a, W_b, color_wheel)
    
    # Oracle territory mapping (upper bound)
    ab_oracle = oracle_colorize(gray, color_maps, color_wheel)
    
    # Errors
    err_territory = np.sqrt(np.mean((ab_territory - ab_ddcolor)**2))
    err_oracle = np.sqrt(np.mean((ab_oracle - ab_ddcolor)**2))
    
    bgr_territory = ab_to_bgr(ab_territory, L)
    bgr_oracle = ab_to_bgr(ab_oracle, L)
    bgr_dd = ab_to_bgr(ab_ddcolor, L)
    
    print(f'  {name}: territory_err={err_territory:.2f}, oracle_err={err_oracle:.2f}, '
          f'sat_t={get_sat(bgr_territory):.0f}, sat_o={get_sat(bgr_oracle):.0f}, '
          f'sat_dd={get_sat(bgr_dd):.0f}')
    
    results.append({
        'name': name, 'err_territory': err_territory, 'err_oracle': err_oracle,
    })
    
    # Save comparison strip
    imgs = [
        (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray'),
        (bgr_territory, f'Territory e={err_territory:.1f}'),
        (bgr_oracle, f'Oracle e={err_oracle:.1f}'),
        (bgr_dd, f'DDColor'),
        (r, f'GT'),
    ]
    for img, label in imgs:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
    
    strip = np.hstack([img for img, _ in imgs])
    cv2.imwrite(os.path.join(out_dir, f'mapped_{name}.jpg'), strip)

# Summary
print('\n=== SUMMARY ===')
mean_territory = np.mean([r['err_territory'] for r in results])
mean_oracle = np.mean([r['err_oracle'] for r in results])
print(f'Mean territory error: {mean_territory:.2f}')
print(f'Mean oracle error: {mean_oracle:.2f}')
print(f'Oracle shows the BEST possible with edge regions + color wheel: {mean_oracle:.2f}')
print(f'Territory mapping captures {1 - mean_territory/mean_oracle if mean_oracle > 0 else 0:.0%} beyond oracle baseline')

# Save weights
np.savez(os.path.join(out_dir, 'territory_weights.npz'),
         W_a=W_a, W_b=W_b, color_wheel=color_wheel)

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
