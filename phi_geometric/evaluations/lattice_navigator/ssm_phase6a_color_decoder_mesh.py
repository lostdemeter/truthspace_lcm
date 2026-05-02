"""
Phase 6A: Unwind the Color Decoder's Attention

The color decoder is a 9-layer transformer with:
  - 256 learned color queries (query_feat + query_embed)
  - Cross-attention: queries attend to encoder features (3 resolution levels, cycling)
  - Self-attention: queries attend to each other
  - FFN: ReLU MLP
  - 8 heads, head_dim=32, embed_dim=256

MESH = W_q.T @ W_k captures the Q-K geometric relationship.
We extract MESH for both cross-attention and self-attention at each layer.

From doc 190 (Qwen2 unwinding): biases matter!
From doc 129: MESH singular values follow φ-Zipf with α ≈ 1/φ

Questions:
  1. What is the MESH structure of the cross-attention?
  2. Do MESH singular values follow φ-Zipf?
  3. What do the color queries attend to in the encoder features?
  4. How do attention patterns evolve across 9 layers?
  5. Do color queries have fixed-point behavior (doc 176)?
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256
v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

NUM_HEADS = 8
EMBED_DIM = 256
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 32
NUM_LAYERS = 9
NUM_QUERIES = 100

# ================================================================
# STEP 1: EXTRACT MESH FROM ALL LAYERS
# ================================================================
print('=' * 70)
print('STEP 1: Extract MESH (W_q.T @ W_k) per head per layer')
print('=' * 70)
print()

cross_meshes = {}    # {layer: {head: MESH}}
self_meshes = {}
cross_wq = {}        # raw weight matrices
cross_wk = {}
cross_wv = {}
cross_bq = {}
cross_bk = {}

for layer in range(NUM_LAYERS):
    level = layer % 3

    # Cross-attention weights
    prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer}'
    w_full = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight').numpy()  # [768, 256]
    b_full = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias').numpy()    # [768]

    Wq = w_full[:EMBED_DIM]          # [256, 256]
    Wk = w_full[EMBED_DIM:2*EMBED_DIM]
    Wv = w_full[2*EMBED_DIM:]
    Bq = b_full[:EMBED_DIM]
    Bk = b_full[EMBED_DIM:2*EMBED_DIM]

    cross_wq[layer] = Wq
    cross_wk[layer] = Wk
    cross_wv[layer] = Wv
    cross_bq[layer] = Bq
    cross_bk[layer] = Bk

    # Per-head MESH
    cross_meshes[layer] = {}
    for h in range(NUM_HEADS):
        wq_h = Wq[h*HEAD_DIM:(h+1)*HEAD_DIM]  # [32, 256]
        wk_h = Wk[h*HEAD_DIM:(h+1)*HEAD_DIM]  # [32, 256]
        # MESH = W_q.T @ W_k → [256, 256] — how Q (from queries) relates to K (from encoder)
        mesh = wq_h.T @ wk_h
        cross_meshes[layer][h] = mesh

    # Self-attention weights
    prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer}'
    w_full = v16._get_weight(f'{prefix}.self_attn.in_proj_weight').numpy()
    b_full = v16._get_weight(f'{prefix}.self_attn.in_proj_bias').numpy()

    Wq_self = w_full[:EMBED_DIM]
    Wk_self = w_full[EMBED_DIM:2*EMBED_DIM]

    self_meshes[layer] = {}
    for h in range(NUM_HEADS):
        wq_h = Wq_self[h*HEAD_DIM:(h+1)*HEAD_DIM]
        wk_h = Wk_self[h*HEAD_DIM:(h+1)*HEAD_DIM]
        mesh = wq_h.T @ wk_h
        self_meshes[layer][h] = mesh

print(f'Extracted MESH for {NUM_LAYERS} layers × {NUM_HEADS} heads × 2 (cross + self)')
print(f'MESH shape: [{EMBED_DIM}, {EMBED_DIM}] per head')


# ================================================================
# STEP 2: SVD ANALYSIS OF MESH — φ-Zipf structure?
# ================================================================
print()
print('=' * 70)
print('STEP 2: SVD Analysis of MESH — φ-Zipf structure?')
print('=' * 70)
print()

from scipy.optimize import curve_fit

def zipf_law(ranks, s0, alpha):
    return s0 / ranks**alpha

print('CROSS-ATTENTION MESH:')
print(f'{"Layer":<6} {"Level":<6} {"Head":<5} {"S[0]":<8} {"S[1]":<8} {"S0/S1":<8} '
      f'{"Zipf α":<8} {"Rank90%":<8} {"EffRank":<8}')
print('-' * 65)

cross_alphas = []
cross_s0s1 = []
for layer in range(NUM_LAYERS):
    for h in range(NUM_HEADS):
        mesh = cross_meshes[layer][h]
        svs = np.linalg.svd(mesh, compute_uv=False)
        s0, s1 = svs[0], svs[1]
        ratio = s0 / (s1 + 1e-10)

        # Fit Zipf
        ranks = np.arange(1, len(svs) + 1).astype(float)
        try:
            popt, _ = curve_fit(zipf_law, ranks, svs, p0=[svs[0], 0.5], maxfev=5000)
            alpha = popt[1]
        except:
            alpha = 0.0

        # Effective rank (90% variance)
        cumvar = np.cumsum(svs**2)
        rank90 = np.searchsorted(cumvar, 0.9 * cumvar[-1]) + 1
        eff_rank = np.exp(-np.sum((svs/svs.sum()) * np.log(svs/svs.sum() + 1e-10)))

        if h == 0 or h == 7:  # Print first and last head per layer
            level = layer % 3
            print(f'  {layer:<5} {level:<5} {h:<4} {s0:<7.2f} {s1:<7.2f} {ratio:<7.2f} '
                  f'{alpha:<7.3f} {rank90:<7d} {eff_rank:<7.1f}')

        cross_alphas.append(alpha)
        cross_s0s1.append(ratio)

print(f'\n  Mean Zipf α: {np.mean(cross_alphas):.4f} (target 1/φ = {1/PHI:.4f})')
print(f'  Mean S[0]/S[1]: {np.mean(cross_s0s1):.4f} (target φ = {PHI:.4f})')

print('\nSELF-ATTENTION MESH:')
self_alphas = []
self_s0s1 = []
for layer in range(NUM_LAYERS):
    for h in range(NUM_HEADS):
        mesh = self_meshes[layer][h]
        svs = np.linalg.svd(mesh, compute_uv=False)
        s0, s1 = svs[0], svs[1]
        ratio = s0 / (s1 + 1e-10)
        ranks = np.arange(1, len(svs) + 1).astype(float)
        try:
            popt, _ = curve_fit(zipf_law, ranks, svs, p0=[svs[0], 0.5], maxfev=5000)
            alpha = popt[1]
        except:
            alpha = 0.0
        if h == 0:
            level = layer % 3
            cumvar = np.cumsum(svs**2)
            rank90 = np.searchsorted(cumvar, 0.9 * cumvar[-1]) + 1
            print(f'  Layer {layer} (lvl {level}), head 0: S0/S1={ratio:.2f}, α={alpha:.3f}, rank90={rank90}')
        self_alphas.append(alpha)
        self_s0s1.append(ratio)

print(f'\n  Mean Zipf α: {np.mean(self_alphas):.4f}')
print(f'  Mean S[0]/S[1]: {np.mean(self_s0s1):.4f}')


# ================================================================
# STEP 3: QUERY EMBED AND QUERY FEAT STRUCTURE
# ================================================================
print()
print('=' * 70)
print('STEP 3: Color Query Structure (query_embed + query_feat)')
print('=' * 70)
print()

query_embed = v16._get_weight('decoder.color_decoder.query_embed.weight').numpy()  # [256, 256]
query_feat = v16._get_weight('decoder.color_decoder.query_feat.weight').numpy()    # [256, 256]

print(f'query_embed shape: {query_embed.shape}')
print(f'query_feat shape:  {query_feat.shape}')
print(f'query_embed norms: mean={np.linalg.norm(query_embed, axis=1).mean():.3f}, '
      f'std={np.linalg.norm(query_embed, axis=1).std():.3f}')
print(f'query_feat norms:  mean={np.linalg.norm(query_feat, axis=1).mean():.3f}, '
      f'std={np.linalg.norm(query_feat, axis=1).std():.3f}')

# SVD of query_embed
svs_qe = np.linalg.svd(query_embed, compute_uv=False)
svs_qf = np.linalg.svd(query_feat, compute_uv=False)
print(f'\nquery_embed SVD: S[0]={svs_qe[0]:.2f}, S[1]={svs_qe[1]:.2f}, S0/S1={svs_qe[0]/svs_qe[1]:.3f}')
print(f'query_feat SVD:  S[0]={svs_qf[0]:.2f}, S[1]={svs_qf[1]:.2f}, S0/S1={svs_qf[0]/svs_qf[1]:.3f}')

# Pairwise similarity of color queries
cos_sim = query_embed @ query_embed.T
norms = np.linalg.norm(query_embed, axis=1, keepdims=True) + 1e-10
cos_sim = (query_embed / norms) @ (query_embed / norms).T
triu = cos_sim[np.triu_indices_from(cos_sim, k=1)]
print(f'\nQuery pairwise cosine: mean={triu.mean():.4f}, std={triu.std():.4f}')
print(f'  min={triu.min():.4f}, max={triu.max():.4f}')

# Cluster queries
from scipy.cluster.hierarchy import fcluster, linkage
# Use query_embed for clustering
dist = 1 - cos_sim
np.fill_diagonal(dist, 0)
Z = linkage(dist[np.triu_indices_from(dist, k=1)], method='ward')
for n_clust in [4, 8, 16]:
    labels = fcluster(Z, n_clust, criterion='maxclust')
    sizes = [np.sum(labels == c) for c in range(1, n_clust+1)]
    print(f'  {n_clust} clusters: sizes = {sorted(sizes, reverse=True)[:8]}...')


# ================================================================
# STEP 4: RUN ON REAL IMAGES — TRACE ATTENTION PATTERNS
# ================================================================
print()
print('=' * 70)
print('STEP 4: Trace attention through color decoder on real images')
print('=' * 70)
print()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

def run_color_decoder_traced(v16, img_tensor):
    """Run full pipeline and capture attention maps + query states at each layer."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_input = (img_tensor - mean_t) / std_t
    x = x_input.clone()

    with torch.no_grad():
        # Encoder
        features = v16._geometric_encoder(x)
        # UNet
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

        # Color decoder — traced
        x_list = [out0, out1, out2]
        img_features = out3

        src, pos = [], []
        for i, xx in enumerate(x_list):
            proj = F.conv2d(xx,
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.weight'),
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.bias'))
            src.append(proj.flatten(2).permute(2, 0, 1))
            pe = v16.pe_layer(proj)
            pos.append(pe.flatten(2).permute(2, 0, 1))

        for i in range(3):
            src[i] = src[i] + v16._get_weight('decoder.color_decoder.level_embed.weight')[i]

        bs = src[0].shape[1]
        query_embed_t = v16._get_weight('decoder.color_decoder.query_embed.weight').unsqueeze(1).repeat(1, bs, 1)
        output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, bs, 1)

        layer_states = []  # query states at each layer
        cross_attn_maps = []  # attention maps at each layer

        for i in range(9):
            level_index = i % 3

            # ---- Cross-attention (traced) ----
            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{i}'
            query_in = output + query_embed_t
            key_in = src[level_index] + pos[level_index]
            value_in = src[level_index]

            w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight')
            b = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias')
            wout = v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight')
            bout = v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias')

            seq_len = query_in.shape[0]
            src_len = key_in.shape[0]

            q = F.linear(query_in, w[:EMBED_DIM], b[:EMBED_DIM])
            k = F.linear(key_in, w[EMBED_DIM:2*EMBED_DIM], b[EMBED_DIM:2*EMBED_DIM])
            v_proj = F.linear(value_in, w[2*EMBED_DIM:], b[2*EMBED_DIM:])

            q = q.view(seq_len, bs * NUM_HEADS, HEAD_DIM).transpose(0, 1)
            k = k.view(src_len, bs * NUM_HEADS, HEAD_DIM).transpose(0, 1)
            v_proj = v_proj.view(src_len, bs * NUM_HEADS, HEAD_DIM).transpose(0, 1)

            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * (HEAD_DIM ** -0.5), dim=-1)
            cross_attn_maps.append(attn.squeeze(0).numpy())  # [H, Q, S]

            attn_out = torch.bmm(attn, v_proj)
            attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, bs, EMBED_DIM)
            attn_out = F.linear(attn_out, wout, bout)

            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            # ---- Self-attention ----
            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{i}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed_t, output + query_embed_t, output,
                v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            # ---- FFN ----
            prefix = f'decoder.color_decoder.transformer_ffn_layers.{i}'
            ffn_out = F.relu(F.linear(output,
                                      v16._get_weight(f'{prefix}.linear1.weight'),
                                      v16._get_weight(f'{prefix}.linear1.bias')))
            ffn_out = F.linear(ffn_out,
                               v16._get_weight(f'{prefix}.linear2.weight'),
                               v16._get_weight(f'{prefix}.linear2.bias'))
            output = F.layer_norm(output + ffn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            layer_states.append(output.squeeze(1).numpy())  # [Q, D]

    return {
        'layer_states': layer_states,           # 9 × [256, 256]
        'cross_attn_maps': cross_attn_maps,     # 9 × [H, Q, S]
        'src_shapes': [s.shape[0] for s in src],  # source lengths per level
    }


# Run on 5 test images
test_imgs = []
for idx in [250, 260, 270, 280, 290]:
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    test_imgs.append({'tensor': t, 'hsv': hsv, 'idx': idx})

print(f'Running traced decoder on {len(test_imgs)} images...')
traces = []
for img_d in test_imgs:
    tr = run_color_decoder_traced(v16, img_d['tensor'])
    traces.append(tr)
    print(f'  Image {img_d["idx"]}: src_shapes={tr["src_shapes"]}, '
          f'attn_maps[0]={tr["cross_attn_maps"][0].shape}')


# ================================================================
# STEP 5: ANALYZE ATTENTION PATTERNS
# ================================================================
print()
print('=' * 70)
print('STEP 5: Attention Pattern Analysis')
print('=' * 70)
print()

# For each layer, what does the attention look like?
for layer in range(NUM_LAYERS):
    level = layer % 3
    print(f'Layer {layer} (level {level}):')

    # Average attention across images
    all_attn = []
    for tr in traces:
        attn = tr['cross_attn_maps'][layer]  # [H, Q, S]
        all_attn.append(attn)

    mean_attn = np.mean(all_attn, axis=0)  # [H, Q, S]

    # Attention entropy per head
    for h in range(NUM_HEADS):
        attn_h = mean_attn[h]  # [Q, S]
        entropy = -np.sum(attn_h * np.log(attn_h + 1e-10), axis=1).mean()
        max_attn = attn_h.max(axis=1).mean()
        # Sparsity: what fraction of source positions get >1% attention?
        sparsity = (attn_h > 0.01).mean()
        if h == 0 or h == 7:
            print(f'  Head {h}: entropy={entropy:.2f}, max_attn={max_attn:.3f}, sparsity={sparsity:.3f}')

    # Is attention consistent across images?
    if len(all_attn) >= 2:
        # Compare attention maps between image pairs
        cos_sims = []
        for i in range(len(all_attn)):
            for j in range(i+1, len(all_attn)):
                a1 = all_attn[i].reshape(-1)
                a2 = all_attn[j].reshape(-1)
                cs = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-10)
                cos_sims.append(cs)
        print(f'  Cross-image attn consistency: {np.mean(cos_sims):.4f}')
    print()


# ================================================================
# STEP 6: QUERY STATE TRAJECTORY — Geodesic + Bulge?
# ================================================================
print()
print('=' * 70)
print('STEP 6: Query State Trajectory Through 9 Layers')
print('=' * 70)
print()

for img_idx, tr in enumerate(traces):
    states = tr['layer_states']  # 9 × [Q, D]
    init_state = v16._get_weight('decoder.color_decoder.query_feat.weight').numpy()  # [Q, D]

    # Track a few representative queries
    for q_idx in [0, 25, 50, 75]:
        traj = [init_state[q_idx]] + [s[q_idx] for s in states]
        traj = np.array(traj)  # [10, 256]

        # Geodesic: linear interpolation from start to end
        start = traj[0]
        end = traj[-1]
        geodesic = np.array([start + t * (end - start) for t in np.linspace(0, 1, len(traj))])

        # Bulge: deviation from geodesic
        bulge = traj - geodesic
        bulge_norms = np.linalg.norm(bulge, axis=1)

        # Trajectory angle from start
        cos_from_start = [np.dot(traj[i], start) / (np.linalg.norm(traj[i]) * np.linalg.norm(start) + 1e-10)
                          for i in range(len(traj))]
        angle_from_start = [np.degrees(np.arccos(np.clip(c, -1, 1))) for c in cos_from_start]

        if img_idx == 0:
            print(f'  Query {q_idx}: start→end angle={angle_from_start[-1]:.1f}°')
            print(f'    Bulge norms: {" ".join(f"{b:.1f}" for b in bulge_norms)}')

    # Overall trajectory statistics
    init_norms = np.linalg.norm(init_state, axis=1)
    final_norms = np.linalg.norm(states[-1], axis=1)

    # How much do queries change?
    deltas = states[-1] - init_state
    delta_norms = np.linalg.norm(deltas, axis=1)

    if img_idx == 0:
        print(f'\n  All queries (image 0):')
        print(f'    Init norms: mean={init_norms.mean():.2f}, std={init_norms.std():.2f}')
        print(f'    Final norms: mean={final_norms.mean():.2f}, std={final_norms.std():.2f}')
        print(f'    Delta norms: mean={delta_norms.mean():.2f}, std={delta_norms.std():.2f}')
        print(f'    Delta/init ratio: {(delta_norms/init_norms).mean():.3f}')

# Compare query states across images (are they image-dependent?)
if len(traces) >= 2:
    print(f'\n  Query state cross-image similarity:')
    for layer in [0, 4, 8]:
        sims = []
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                s1 = traces[i]['layer_states'][layer].reshape(-1)
                s2 = traces[j]['layer_states'][layer].reshape(-1)
                cs = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-10)
                sims.append(cs)
        print(f'    Layer {layer}: mean cosine = {np.mean(sims):.4f}')


# ================================================================
# STEP 7: FIXED POINTS — Do queries converge to fixed points?
# ================================================================
print()
print('=' * 70)
print('STEP 7: Fixed Point Analysis — Do queries have attractors?')
print('=' * 70)
print()

# From doc 176: delta ≈ target - h_before
# Test if each layer moves queries toward a fixed target
for layer in range(NUM_LAYERS):
    all_h_before = []
    all_h_after = []
    for tr in traces:
        if layer == 0:
            h_before = v16._get_weight('decoder.color_decoder.query_feat.weight').numpy()
        else:
            h_before = tr['layer_states'][layer - 1]
        h_after = tr['layer_states'][layer]
        all_h_before.append(h_before)
        all_h_after.append(h_after)

    # Compute average target per query
    all_h_after = np.array(all_h_after)  # [N_imgs, Q, D]
    all_h_before = np.array(all_h_before)

    # For each query, is h_after consistent across images?
    after_sims = []
    for q in range(NUM_QUERIES):
        vecs = all_h_after[:, q, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        normed = vecs / norms
        cos_mat = normed @ normed.T
        tri = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        after_sims.extend(tri)

    before_sims = []
    for q in range(NUM_QUERIES):
        vecs = all_h_before[:, q, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        normed = vecs / norms
        cos_mat = normed @ normed.T
        tri = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        before_sims.extend(tri)

    # Test: delta ≈ target - h_before
    mean_target = all_h_after.mean(axis=0)  # [Q, D]
    delta_pred_corrs = []
    for img_idx in range(len(traces)):
        delta_actual = all_h_after[img_idx] - all_h_before[img_idx]  # [Q, D]
        delta_pred = mean_target - all_h_before[img_idx]
        # Per-query correlation
        for q in range(0, NUM_QUERIES, 10):
            da = delta_actual[q]
            dp = delta_pred[q]
            cos = np.dot(da, dp) / (np.linalg.norm(da) * np.linalg.norm(dp) + 1e-10)
            delta_pred_corrs.append(cos)

    if layer in [0, 4, 8]:
        print(f'Layer {layer}:')
        print(f'  h_before cross-image sim: {np.mean(before_sims):.4f}')
        print(f'  h_after cross-image sim:  {np.mean(after_sims):.4f}')
        print(f'  delta ≈ target - h_before: cos = {np.mean(delta_pred_corrs):.4f}')
        print(f'  {"CONVERGING" if np.mean(after_sims) > np.mean(before_sims) else "DIVERGING"}')
        print()


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print(f'Cross-attention MESH:')
print(f'  Mean Zipf α = {np.mean(cross_alphas):.4f} (1/φ = {1/PHI:.4f})')
print(f'  Mean S[0]/S[1] = {np.mean(cross_s0s1):.4f} (φ = {PHI:.4f})')
print()
print(f'Self-attention MESH:')
print(f'  Mean Zipf α = {np.mean(self_alphas):.4f}')
print(f'  Mean S[0]/S[1] = {np.mean(self_s0s1):.4f}')
print()
print('Query structure:')
print(f'  256 learned color queries in 256-dim space')
print(f'  Query embed: S0/S1 = {svs_qe[0]/svs_qe[1]:.3f}')
print(f'  Query feat:  S0/S1 = {svs_qf[0]/svs_qf[1]:.3f}')
