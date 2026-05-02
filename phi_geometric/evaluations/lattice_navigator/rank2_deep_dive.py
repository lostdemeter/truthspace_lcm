"""
Rank-2 Projection Deep Dive

The color vectors from DDColor's transformer live on a 2D plane (99.3% variance).
This script analyzes:

1. What ARE the 2 basis vectors of this plane? (SVD → V[0], V[1])
2. How do 100 queries distribute on this plane? (scatter plot in rank-2 space)
3. Does this plane CHANGE per image, or is it fixed?
4. What is the relationship between the rank-2 coordinates and actual ab output?
5. Does the plane orientation have geometric structure (phi patterns)?
6. Can we express the entire color decoder as: features → 2 scalars → ab?

If the plane is FIXED across images, then the entire color decoder is:
    color(pixel) = Σ_q  dot(query_q, pixel_feature) * [alpha_q, beta_q]
    
Where [alpha_q, beta_q] is each query's position on the fixed 2D plane.
This would mean: 100 queries × 2 coords = 200 numbers IS the entire color knowledge.
"""
import numpy as np
import cv2
import sys
import glob
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

print('=== RANK-2 PROJECTION DEEP DIVE ===')
print()

v16 = V16GeometricColorizer()

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/rank2_analysis'
os.makedirs(out_dir, exist_ok=True)

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56, 58, 60, 62, 64]]

# ================================================================
# Helper: run forward and capture color vectors
# ================================================================
def get_color_vectors_and_maps(v16, img_tensor):
    """Run DDColor and return intermediate color vectors + final output."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    
    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
        
        # Color decoder internals
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
        
        # Color embed MLP
        x_color = decoder_output
        for i in range(3):
            x_color = F.linear(x_color,
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
            if i < 2:
                x_color = F.relu(x_color)
        
        # x_color: [1, 100, 256] — the color vectors
        color_maps = torch.einsum('bqc,bchw->bqhw', x_color, out3)
        
        # Refine net
        coarse_input = torch.cat([color_maps, img_tensor], dim=1)
        ab_out = F.conv2d(coarse_input,
                          v16._get_weight('refine_net.0.0.weight'),
                          v16._get_weight('refine_net.0.0.bias'))
        
    return x_color.squeeze(0).detach(), color_maps.detach(), out3.detach(), ab_out.detach()


# ================================================================
# PART 1: Collect color vectors across multiple images
# ================================================================
print('=== PART 1: Collecting color vectors across images ===')

all_color_vecs = []  # [n_images, 100, 256]
all_names = []

for img_path in test_paths:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    gray = cv2.cvtColor(cv2.resize(im, (SZ, SZ)), cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    color_vecs, _, _, _ = get_color_vectors_and_maps(v16, img_tensor)
    all_color_vecs.append(color_vecs.numpy())
    all_names.append(name)
    print(f'  {name}: color_vecs shape={color_vecs.shape}, '
          f'norm range=[{color_vecs.norm(dim=1).min().item():.1f}, {color_vecs.norm(dim=1).max().item():.1f}]')

# ================================================================
# PART 2: SVD each image's color vectors — is the plane FIXED?
# ================================================================
print('\n=== PART 2: Is the rank-2 plane FIXED across images? ===')

all_V = []  # The 2D basis for each image
all_S = []

for i, cv_np in enumerate(all_color_vecs):
    U, S, Vt = np.linalg.svd(cv_np, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    
    all_V.append(Vt[:2])  # Top 2 right singular vectors: the plane basis
    all_S.append(S[:5])
    
    print(f'  {all_names[i]}: rank-1={cumvar[0]*100:.1f}%, rank-2={cumvar[1]*100:.1f}%, '
          f'S[0:3]=[{S[0]:.1f}, {S[1]:.1f}, {S[2]:.1f}]')

# Compare planes across images: compute alignment (inner products of basis vectors)
print(f'\n  Plane alignment (|dot product| of basis vectors):')
n_imgs = len(all_V)
alignment_matrix = np.zeros((n_imgs, n_imgs))

for i in range(n_imgs):
    for j in range(n_imgs):
        # Alignment = Frobenius inner product of the two 2×256 basis matrices
        # Using principal angles between subspaces
        M = all_V[i] @ all_V[j].T  # [2, 2]
        _, sigmas, _ = np.linalg.svd(M)
        # Principal angles: cos(theta) = sigma
        alignment = np.prod(sigmas)  # Product of cosines = volume alignment
        alignment_matrix[i, j] = alignment

print(f'  {"":>12}', end='')
for n in all_names:
    print(f'{n[-4:]:>8}', end='')
print()

for i in range(n_imgs):
    print(f'  {all_names[i][-8:]:>12}', end='')
    for j in range(n_imgs):
        print(f'{alignment_matrix[i,j]:>8.3f}', end='')
    print()

# Average off-diagonal alignment
off_diag = alignment_matrix[np.triu_indices(n_imgs, k=1)]
print(f'\n  Mean plane alignment: {off_diag.mean():.3f} (1.0=identical, 0.0=orthogonal)')

# ================================================================
# PART 3: Project queries onto the rank-2 plane
# ================================================================
print('\n=== PART 3: Query positions on the rank-2 plane ===')

# Use the first image's plane as reference
ref_V = all_V[0]  # [2, 256]

for i, cv_np in enumerate(all_color_vecs):
    # Project 100 queries onto the 2D plane
    coords_2d = cv_np @ ref_V.T  # [100, 2]
    
    # How much of each query's energy is captured by the 2D projection?
    proj_norms = np.linalg.norm(coords_2d, axis=1)
    full_norms = np.linalg.norm(cv_np, axis=1)
    frac_captured = proj_norms / (full_norms + 1e-8)
    
    print(f'  {all_names[i]}: 2D captures {frac_captured.mean()*100:.1f}% of query norms')
    print(f'    Coord range: x=[{coords_2d[:,0].min():.1f}, {coords_2d[:,0].max():.1f}], '
          f'y=[{coords_2d[:,1].min():.1f}, {coords_2d[:,1].max():.1f}]')
    
    # Are queries uniformly distributed or clustered?
    from scipy.spatial.distance import pdist
    dists = pdist(coords_2d)
    print(f'    Inter-query distance: mean={dists.mean():.1f}, std={dists.std():.1f}, '
          f'min={dists.min():.1f}, max={dists.max():.1f}')

# ================================================================
# PART 4: What IS the rank-2 plane in terms of actual ab color?
# ================================================================
print('\n=== PART 4: Mapping rank-2 plane to ab color space ===')

# The full pipeline: color_vectors @ img_features → color_maps → refine → ab
# The refine net is: [2, 103, 1, 1] conv
# Input to refine: [100 color_maps, 3 input_channels] → [2 ab channels]

refine_w = v16._get_weight('refine_net.0.0.weight').numpy()  # [2, 103, 1, 1]
refine_b = v16._get_weight('refine_net.0.0.bias').numpy()     # [2]
refine_w = refine_w.reshape(2, 103)

# The first 100 columns correspond to the 100 color queries
# The last 3 columns correspond to the grayscale input channels
color_weights = refine_w[:, :100]  # [2, 100] — how each query maps to a, b
input_weights = refine_w[:, 100:]  # [2, 3] — how input channels contribute

print(f'Refine net structure:')
print(f'  Color query weights: {color_weights.shape} = [a_row, b_row] × 100 queries')
print(f'  Input channel weights: {input_weights.shape}')
print(f'  Bias: {refine_b}')

# Each query has a (weight_a, weight_b) pair in the refine net
# This IS the final color assignment per query
query_ab = color_weights.T  # [100, 2] — each query's contribution to (a, b)

print(f'\n  Query → ab mapping (refine weights):')
print(f'    a-range: [{query_ab[:,0].min():.4f}, {query_ab[:,0].max():.4f}]')
print(f'    b-range: [{query_ab[:,1].min():.4f}, {query_ab[:,1].max():.4f}]')

# SVD of the query→ab mapping
U_ab, S_ab, Vt_ab = np.linalg.svd(query_ab, full_matrices=False)
print(f'    SVD of query→ab: S = [{S_ab[0]:.4f}, {S_ab[1]:.4f}]')
print(f'    Ratio S[0]/S[1] = {S_ab[0]/S_ab[1]:.4f} (phi={PHI:.4f}, err={abs(S_ab[0]/S_ab[1]-PHI)/PHI*100:.1f}%)')

# The right singular vectors tell us the two "directions" in query space
# that map to a and b
print(f'    V[0] (a-direction): [{Vt_ab[0,0]:.4f}, {Vt_ab[0,1]:.4f}]')
print(f'    V[1] (b-direction): [{Vt_ab[1,0]:.4f}, {Vt_ab[1,1]:.4f}]')

# ================================================================
# PART 5: For each image, what are the ACTUAL query activations?
# ================================================================
print('\n=== PART 5: Query activations per image ===')

for img_path in test_paths[:4]:
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    gray = cv2.cvtColor(cv2.resize(im, (SZ, SZ)), cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(gbgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    color_vecs, color_maps, img_feats, ab_out = get_color_vectors_and_maps(v16, img_tensor)
    
    # color_maps: [1, 100, 256, 256] — each query's spatial activation
    cm = color_maps.squeeze(0).numpy()  # [100, H, W]
    
    # For each pixel, which query dominates?
    dominant_query = np.argmax(np.abs(cm), axis=0)  # [H, W]
    n_unique = len(np.unique(dominant_query))
    
    # Weighted activation: how much does each query contribute overall?
    query_total = np.abs(cm).sum(axis=(1, 2))  # [100]
    query_total_sorted = np.sort(query_total)[::-1]
    cumulative = np.cumsum(query_total_sorted) / query_total_sorted.sum()
    
    n_90 = (cumulative < 0.9).sum() + 1
    n_95 = (cumulative < 0.95).sum() + 1
    
    # Each query's contribution to final ab via refine weights
    # Final a_pixel = Σ_q color_map[q, h, w] * refine_w_a[q]
    # Final b_pixel = Σ_q color_map[q, h, w] * refine_w_b[q]
    
    # So the EFFECTIVE ab at each pixel is:
    effective_a = (cm * query_ab[:, 0].reshape(100, 1, 1)).sum(axis=0)  # [H, W]
    effective_b = (cm * query_ab[:, 1].reshape(100, 1, 1)).sum(axis=0)  # [H, W]
    
    # Compare to actual ab output (which also includes input channel contribution)
    ab_actual = ab_out.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 2]
    
    query_only_err = np.sqrt(np.mean((effective_a - ab_actual[:,:,0])**2 + 
                                      (effective_b - ab_actual[:,:,1])**2))
    
    print(f'\n  {name}:')
    print(f'    Unique dominant queries: {n_unique}/100')
    print(f'    Queries for 90% activation: {n_90}')
    print(f'    Queries for 95% activation: {n_95}')
    print(f'    Query-only vs full ab error: {query_only_err:.2f} '
          f'(residual = input channel contribution)')
    
    # Visualize: paint each pixel with its dominant query's ab color
    dom_a = query_ab[dominant_query, 0]  # [H, W] 
    dom_b = query_ab[dominant_query, 1]  # [H, W]
    
    # Scale for visibility (these weights are small)
    scale = cm.max()
    
    # The territory map: which query owns which pixel
    territory = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    query_colors = rng.randint(30, 255, (100, 3))
    for h in range(SZ):
        for w in range(SZ):
            territory[h, w] = query_colors[dominant_query[h, w]]
    
    r = cv2.resize(im, (SZ, SZ))
    L = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)[:,:,0]
    
    bgr_out = ab_to_bgr(ab_actual, L)
    
    for img, label in [(r, 'GT'), (bgr_out, 'DDColor'), (territory, f'Territory ({n_unique}q)')]:
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 2)
        cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    strip = np.hstack([r, bgr_out, territory])
    cv2.imwrite(os.path.join(out_dir, f'territory_{name}.jpg'), strip)

# ================================================================
# PART 6: The complete rank-2 decomposition
# ================================================================
print('\n\n=== PART 6: The Complete Rank-2 Decomposition ===')
print()
print('The entire DDColor color decoder reduces to:')
print()
print('  ab(pixel) = Σ_q  activation(q, pixel) × [w_a(q), w_b(q)]  +  bias')
print()
print('Where:')
print('  - activation(q, pixel) = dot(color_vector_q, img_feature(pixel))')
print('  - [w_a(q), w_b(q)] = refine net weights for query q')
print('  - These are 100 × 2 = 200 final color parameters')
print()
print('The 55M parameters exist to compute:')
print('  1. img_feature(pixel) — WHERE in feature space (encoder: 27.8M)')
print('  2. color_vector_q — WHAT each query represents (decoder: ~25M)')  
print('  3. [w_a(q), w_b(q)] — HOW queries map to ab (refine: 206 params)')
print()

# Print the actual 200 numbers
print('THE 200 NUMBERS (query → ab mapping):')
print(f'{"Query":>6} {"w_a":>10} {"w_b":>10} {"magnitude":>10} {"angle(°)":>10}')
print('-' * 48)
for q in range(100):
    wa = query_ab[q, 0]
    wb = query_ab[q, 1]
    mag = np.sqrt(wa**2 + wb**2)
    angle = np.degrees(np.arctan2(wb, wa))
    print(f'{q:>6} {wa:>10.5f} {wb:>10.5f} {mag:>10.5f} {angle:>10.1f}°')

# Distribution of magnitudes and angles
mags = np.sqrt(query_ab[:,0]**2 + query_ab[:,1]**2)
angles = np.degrees(np.arctan2(query_ab[:,1], query_ab[:,0]))

print(f'\nMagnitude stats: mean={mags.mean():.5f}, std={mags.std():.5f}, '
      f'min={mags.min():.5f}, max={mags.max():.5f}')
print(f'Angle stats: mean={angles.mean():.1f}°, std={angles.std():.1f}°')

# Are magnitudes phi-distributed?
sorted_mags = np.sort(mags)[::-1]
print(f'\nMagnitude ratios (sorted):')
for i in range(min(10, len(sorted_mags)-1)):
    if sorted_mags[i+1] > 0:
        ratio = sorted_mags[i] / sorted_mags[i+1]
        phi_err = abs(ratio - PHI) / PHI
        marker = ' ** PHI' if phi_err < 0.1 else ''
        print(f'  M[{i}]/M[{i+1}] = {ratio:.4f} (phi err={phi_err*100:.1f}%){marker}')

# Check if angles are uniformly distributed or clustered
print(f'\nAngle histogram (12 bins of 30°):')
hist, edges = np.histogram(angles, bins=12, range=(-180, 180))
for i in range(12):
    bar = '█' * hist[i]
    print(f'  [{edges[i]:>5.0f}°, {edges[i+1]:>5.0f}°): {hist[i]:>3} {bar}')

print(f'\nOutput saved to: {out_dir}/')
print('Done!')
