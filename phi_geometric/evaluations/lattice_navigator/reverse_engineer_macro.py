"""
Reverse Engineering DDColor's Macro Operations

DDColor's core mechanism (line 304 of V16):
    torch.einsum('bqc,bchw->bqhw', x, img_features)

This is: 100 color queries × per-pixel features → per-pixel colors

The mechanism:
1. 100 "color queries" live in 256-dim space (query_embed)
2. Each pixel has a 256-dim feature vector (from UNet decoder)
3. Dot product determines which queries claim which pixels
4. The claimed query's color gets applied

Questions to answer:
A. What do the 100 color queries look like? (SVD, clustering, phi patterns?)
B. What does the attention pattern look like? (is it sparse? how many queries dominate?)
C. What does the color_embed projection do? (maps 256-dim → ab space)
D. Is this a rotation? projection? selection? sorting? filtering?
E. Can we replicate this operation without 55M params?
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

print('=== REVERSE ENGINEERING DDColor MACRO OPERATIONS ===')
print()

# Load V16 (has all weights)
v16 = V16GeometricColorizer()

out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/macro_analysis'
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# PART A: Analyze the Color Queries
# ================================================================
print('=== PART A: The 100 Color Queries ===')

query_embed = v16._get_weight('decoder.color_decoder.query_embed.weight')
query_feat = v16._get_weight('decoder.color_decoder.query_feat.weight')

print(f'query_embed shape: {query_embed.shape}')  # [100, 256]
print(f'query_feat shape: {query_feat.shape}')    # [100, 256]

# SVD of query embeddings
U_q, S_q, Vt_q = torch.linalg.svd(query_embed)
cumvar_q = torch.cumsum(S_q**2, 0) / torch.sum(S_q**2)

print(f'\nQuery embed SVD:')
print(f'  Top 10 singular values: {S_q[:10].numpy()}')
for k in [5, 10, 20, 50]:
    print(f'  Rank {k}: {cumvar_q[k-1].item()*100:.1f}% variance')

# Check phi patterns in singular values
print(f'\n  Phi pattern check (adjacent ratios):')
for i in range(min(10, len(S_q)-1)):
    if S_q[i+1] > 0:
        ratio = (S_q[i] / S_q[i+1]).item()
        phi_err = abs(ratio - PHI) / PHI
        print(f'    S[{i}]/S[{i+1}] = {ratio:.4f} (phi err={phi_err*100:.1f}%)')

# Cluster the 100 queries
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

q_np = query_embed.numpy()
Z = linkage(q_np, method='ward')

for n_clusters in [3, 5, 8, 13]:
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    print(f'\n  {n_clusters} query clusters:')
    for c in range(1, n_clusters + 1):
        mask = labels == c
        print(f'    Cluster {c}: {mask.sum()} queries, '
              f'mean_norm={np.linalg.norm(q_np[mask], axis=1).mean():.2f}')

# ================================================================
# PART B: Analyze the Color Embedding (query → ab color)
# ================================================================
print('\n\n=== PART B: Color Embedding (256-dim → colors) ===')

# The color_embed is a 3-layer MLP: 256 → 256 → 256 → Q_out
# Then einsum with image features
for i in range(3):
    w = v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight')
    b = v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias')
    print(f'  color_embed layer {i}: weight={w.shape}, bias={b.shape}')
    
    # SVD of each layer
    U_w, S_w, Vt_w = torch.linalg.svd(w)
    cumvar = torch.cumsum(S_w**2, 0) / torch.sum(S_w**2)
    for k in [5, 10, 20]:
        if k <= len(cumvar):
            print(f'    Rank {k}: {cumvar[k-1].item()*100:.1f}% variance')
    
    # Phi patterns
    for j in range(min(5, len(S_w)-1)):
        if S_w[j+1] > 0:
            ratio = (S_w[j] / S_w[j+1]).item()
            phi_err = abs(ratio - PHI) / PHI
            if phi_err < 0.15:
                print(f'    ** S[{j}]/S[{j+1}] = {ratio:.4f} (phi match, err={phi_err*100:.1f}%)')

# ================================================================
# PART C: Run images through and capture intermediate states
# ================================================================
print('\n\n=== PART C: Tracing the Macro Operations on Real Images ===')

SZ = 256
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
test_paths = [all_imgs[i] for i in [50, 52, 54, 56]]

for img_path in test_paths[:2]:  # Just 2 for speed
    im = cv2.imread(img_path)
    if im is None: continue
    name = os.path.basename(img_path).replace('.jpg', '')
    
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_tensor = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    
    print(f'\n  Image: {name}')
    
    # Run encoder
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean) / std
    
    with torch.no_grad():
        features = v16._geometric_encoder(x)
    
    print(f'  Encoder features: {[f.shape for f in features]}')
    
    # Run UNet decoder
    with torch.no_grad():
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)  # img_features for einsum
    
    print(f'  UNet output (img_features): {out3.shape}')
    
    # Run color decoder step by step, capturing attention
    # We need to manually trace the color decoder
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
    query_embed_r = v16._get_weight('decoder.color_decoder.query_embed.weight').unsqueeze(1).repeat(1, bs, 1)
    output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, bs, 1)
    
    # Track how queries evolve through 9 transformer layers
    query_norms = []
    query_similarities = []
    
    with torch.no_grad():
        for layer_i in range(9):
            level_index = layer_i % 3
            
            # Cross-attention (queries attend to image features)
            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer_i}'
            
            # Capture attention weights manually
            q_in = output + query_embed_r
            k_in = src[level_index] + pos[level_index]
            v_in = src[level_index]
            
            in_proj_w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight')
            in_proj_b = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias')
            
            embed_dim = 256
            num_heads = 8
            head_dim = embed_dim // num_heads
            
            q = F.linear(q_in, in_proj_w[:embed_dim], in_proj_b[:embed_dim])
            k = F.linear(k_in, in_proj_w[embed_dim:2*embed_dim], in_proj_b[embed_dim:2*embed_dim])
            v = F.linear(v_in, in_proj_w[2*embed_dim:], in_proj_b[2*embed_dim:])
            
            seq_len = q.shape[0]
            src_len = k.shape[0]
            
            q_r = q.view(seq_len, bs * num_heads, head_dim).transpose(0, 1)
            k_r = k.view(src_len, bs * num_heads, head_dim).transpose(0, 1)
            v_r = v.view(src_len, bs * num_heads, head_dim).transpose(0, 1)
            
            attn_weights = F.softmax(torch.bmm(q_r, k_r.transpose(1, 2)) * (head_dim ** -0.5), dim=-1)
            # attn_weights: [heads, 100_queries, spatial_positions]
            
            attn_out_r = torch.bmm(attn_weights, v_r)
            attn_out = attn_out_r.transpose(0, 1).contiguous().view(seq_len, bs, embed_dim)
            out_proj_w = v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight')
            out_proj_b = v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias')
            attn_out = F.linear(attn_out, out_proj_w, out_proj_b)
            
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))
            
            # Self-attention
            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer_i}'
            self_attn_out = v16._geometric_multihead_attention(
                output + query_embed_r, output + query_embed_r, output,
                v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
            output = F.layer_norm(output + self_attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))
            
            # FFN
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
            
            # Analyze attention pattern
            # Average across heads: [100, spatial]
            attn_avg = attn_weights.view(num_heads, bs, seq_len, src_len).mean(dim=0).squeeze(0)
            # attn_avg: [100, spatial]
            
            # How many queries are "active" (have attention > 1/N)?
            max_attn_per_query = attn_avg.max(dim=1)[0]  # [100]
            active_queries = (max_attn_per_query > 0.05).sum().item()
            
            # How sparse is each query's attention?
            entropy = -(attn_avg * torch.log(attn_avg + 1e-10)).sum(dim=1)
            mean_entropy = entropy.mean().item()
            max_possible_entropy = np.log(src_len)
            
            # Track query evolution
            query_norms.append(output.squeeze(1).norm(dim=1).numpy())
            
            if layer_i % 3 == 0 or layer_i == 8:
                print(f'    Layer {layer_i} (level {level_index}): '
                      f'active_queries={active_queries}/100, '
                      f'attn_entropy={mean_entropy:.1f}/{max_possible_entropy:.1f} '
                      f'({"sparse" if mean_entropy < max_possible_entropy * 0.5 else "diffuse"})')
    
    # After all transformer layers, trace through color_embed
    decoder_output = F.layer_norm(
        output, (256,),
        v16._get_weight('decoder.color_decoder.decoder_norm.weight'),
        v16._get_weight('decoder.color_decoder.decoder_norm.bias')).transpose(0, 1)
    
    # Color embed MLP: what does it DO to the queries?
    x_color = decoder_output
    for i in range(3):
        x_color = F.linear(x_color,
                     v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                     v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
        if i < 2:
            x_color = F.relu(x_color)
    
    # x_color: [1, 100, C] where C = output channels
    print(f'  Color embed output: {x_color.shape}')
    print(f'  Color embed value range: [{x_color.min().item():.3f}, {x_color.max().item():.3f}]')
    
    # The FINAL operation: einsum('bqc,bchw->bqhw')
    # This is: for each pixel (h,w), dot product of color_query[q] with img_feature[c,h,w]
    # Result: Q color maps, one per query
    
    color_maps = torch.einsum('bqc,bchw->bqhw', x_color, out3)
    print(f'  Color maps (pre-refine): {color_maps.shape}')
    
    # How many queries actually contribute?
    query_energy = (color_maps**2).sum(dim=(2,3)).squeeze(0)  # [Q]
    total_energy = query_energy.sum().item()
    sorted_energy, sorted_idx = query_energy.sort(descending=True)
    cumulative = torch.cumsum(sorted_energy, 0) / total_energy
    
    n_for_90 = (cumulative < 0.9).sum().item() + 1
    n_for_95 = (cumulative < 0.95).sum().item() + 1
    n_for_99 = (cumulative < 0.99).sum().item() + 1
    
    print(f'\n  Query contribution analysis:')
    print(f'    Queries for 90% energy: {n_for_90}/100')
    print(f'    Queries for 95% energy: {n_for_95}/100')
    print(f'    Queries for 99% energy: {n_for_99}/100')
    print(f'    Top 5 query indices: {sorted_idx[:5].numpy()}')
    print(f'    Top 5 energy %: {(sorted_energy[:5]/total_energy*100).numpy()}')
    
    # Visualize top query attention maps
    # The color maps essentially show "which pixels does each query color?"
    for qi in range(min(5, n_for_90)):
        q_idx = sorted_idx[qi].item()
        q_map = color_maps[0, q_idx].numpy()  # [H, W]
        
        # Normalize for visualization
        q_vis = (q_map - q_map.min()) / (q_map.max() - q_map.min() + 1e-8) * 255
        q_vis = q_vis.astype(np.uint8)
        q_vis = cv2.resize(q_vis, (SZ, SZ))
        q_vis_color = cv2.applyColorMap(q_vis, cv2.COLORMAP_JET)
        
        cv2.imwrite(os.path.join(out_dir, f'{name}_query_{qi}_idx{q_idx}.jpg'), q_vis_color)
    
    # Save query contribution histogram
    print(f'    Saved top {min(5, n_for_90)} query maps')

# ================================================================
# PART D: What IS the geometric operation?
# ================================================================
print('\n\n=== PART D: Identifying the Geometric Operation ===')

# The pipeline is:
# 1. ConvNeXt: image → 768-dim features (rotation + projection at each layer)
# 2. UNet: multi-scale features → per-pixel 256-dim features 
# 3. 100 queries attend to features (SELECTION: which features matter?)
# 4. Queries refined through self-attention (SORTING: queries negotiate)
# 5. color_embed MLP maps queries to color channels (PROJECTION: 256 → C)
# 6. einsum: dot product of query colors × pixel features (ASSIGNMENT: who gets what)

# Test: is the query→color mapping LOW RANK?
# If so, the 100 queries don't all do different things

# Already computed x_color for last image: [1, 100, C]
color_vectors = x_color.squeeze(0).detach().numpy()  # [100, C]
print(f'Color vectors shape: {color_vectors.shape}')

U_c, S_c, Vt_c = np.linalg.svd(color_vectors, full_matrices=False)
cumvar_c = np.cumsum(S_c**2) / np.sum(S_c**2)

print(f'Color vector SVD:')
for k in [1, 2, 3, 5, 8, 13]:
    if k <= len(cumvar_c):
        print(f'  Rank {k}: {cumvar_c[k-1]*100:.1f}% variance')

# Phi patterns in color vector singular values?
print(f'\nColor vector S-value ratios:')
for i in range(min(8, len(S_c)-1)):
    if S_c[i+1] > 0:
        ratio = S_c[i] / S_c[i+1]
        phi_err = abs(ratio - PHI) / PHI
        marker = ' ** PHI' if phi_err < 0.1 else ''
        print(f'  S[{i}]/S[{i+1}] = {ratio:.4f} (phi err={phi_err*100:.1f}%){marker}')

# Test: is the img_features → color mapping essentially a projection?
# For each pixel, color = dot(color_queries, pixel_feature)
# If color_queries are low-rank, this is a projection onto a low-dim subspace

# Also: what does the refine_net do?
refine_w = v16._get_weight('refine_net.0.0.weight')
refine_b = v16._get_weight('refine_net.0.0.bias')
print(f'\nRefine net: weight={refine_w.shape}, bias={refine_b.shape}')

# SVD of refine weights
refine_2d = refine_w.reshape(refine_w.shape[0], -1)
U_r, S_r, Vt_r = torch.linalg.svd(refine_2d)
cumvar_r = torch.cumsum(S_r**2, 0) / torch.sum(S_r**2)
print(f'Refine net SVD:')
for k in [1, 2, 5]:
    if k <= len(cumvar_r):
        print(f'  Rank {k}: {cumvar_r[k-1].item()*100:.1f}% variance')

# ================================================================
# PART E: Summary — What IS the mechanism?
# ================================================================
print('\n\n' + '='*60)
print('MACRO OPERATION IDENTIFICATION')
print('='*60)
print()
print('DDColor\'s colorization = 4 geometric operations:')
print()
print('1. ENCODING (ConvNeXt): image → feature vectors')
print('   - 18 blocks of: depthwise conv → layernorm → pointwise → GELU → residual')
print('   - Each block = rotation + projection + nonlinear scaling')
print('   - Output: per-pixel 768-dim feature vectors')
print()
print('2. DECODING (UNet): multi-scale features → unified per-pixel features')
print('   - Skip connections + pixel shuffle upsampling')
print('   - Output: per-pixel vectors in color-decoder space')
print()
print('3. QUERY REFINEMENT (Transformer): 100 queries negotiate color assignments')
print('   - Cross-attention: queries SELECT which image features matter')
print('   - Self-attention: queries SORT themselves (negotiate territories)')
print('   - FFN: queries SHARPEN their color assignments')
print('   - 9 layers cycling through 3 scales')
print()
print('4. ASSIGNMENT (einsum): queries claim pixels via dot product')
print('   - Each pixel gets colored by whichever query it projects onto most')
print('   - This is GEOMETRIC SELECTION: proximity in feature space = color assignment')
print()
print('The mechanism is: ENCODE → SELECT → SORT → ASSIGN')
print('No step requires "knowing" what grass is.')
print('It requires knowing WHERE in feature space grass-like features cluster.')

print('\nDone!')
