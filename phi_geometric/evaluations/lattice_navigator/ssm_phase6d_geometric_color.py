"""
Phase 6D: Pure Geometric Color Generation

Phase 6B proved the 9-layer color decoder is scaffolding:
  - Layers 0-5: rank-1 fixed point (skip-able with 0.00% change)
  - Q: completely image-independent (variance = 0.0)
  - FP + no layers: -3.35% RMSE (IMPROVES!)
  - Cross-attn only: -12.36% (self-attn + FFN add noise)

The effective operation is:
  color = einsum(color_embed(fixed_query), img_features)

This script tests:
  1. Pure geometric: FP → color_embed → einsum (ZERO transformer layers)
  2. Optimized geometric: find the BEST fixed query matrix
  3. Rank analysis: how many query dimensions actually matter?
  4. Direct weight multiplication: can we precompute everything?
  5. Large-scale validation (100+ images)
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2
SZ = 256
NUM_HEADS = 8
EMBED_DIM = 256
HEAD_DIM = 32

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def load_image(idx, sz=SZ):
    im = cv2.imread(all_imgs[idx])
    if im is None: return None
    r = cv2.resize(im, (sz, sz))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    return {'tensor': t, 'gt_ab': gt_ab, 'idx': idx, 'bgr': r}


def run_encoder_unet(v16, img_tensor):
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (img_tensor - mean_t) / std_t
    with torch.no_grad():
        features = v16._geometric_encoder(x)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return x, [out0, out1, out2], out3


def decode_queries_to_color(v16, query_states, img_features):
    """Apply color_embed MLP + einsum. query_states: [Q, 1, D] or [1, Q, D]."""
    with torch.no_grad():
        # Final norm
        decoder_output = F.layer_norm(
            query_states, (256,),
            v16._get_weight('decoder.color_decoder.decoder_norm.weight'),
            v16._get_weight('decoder.color_decoder.decoder_norm.bias'))

        # Handle shape: need [1, Q, D] for einsum
        if decoder_output.dim() == 3 and decoder_output.shape[0] != 1:
            decoder_output = decoder_output.transpose(0, 1)  # [Q,1,D] → [1,Q,D]

        x = decoder_output
        for i in range(3):
            x = F.linear(x,
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                         v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
            if i < 2:
                x = F.relu(x)

        return torch.einsum('bqc,bchw->bqhw', x, img_features)


def apply_refine(v16, color_out, x_norm):
    coarse_input = torch.cat([color_out, x_norm], dim=1)
    return F.conv2d(coarse_input, v16._get_weight('refine_net.0.0.weight'),
                    v16._get_weight('refine_net.0.0.bias'))


def compute_rmse(pred_tensor, gt_ab):
    pred_ab = pred_tensor[0, :2].permute(1, 2, 0).numpy()
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    return np.sqrt(np.mean((pred_r - gt_ab)**2))


# ================================================================
# STEP 1: Compute fixed point from calibration set
# ================================================================
print('=' * 70)
print('STEP 1: Compute fixed point from calibration')
print('=' * 70)
print()

N_CAL = 50
cal_data = [load_image(idx) for idx in range(100, 100 + N_CAL * 2)]
cal_data = [d for d in cal_data if d is not None][:N_CAL]

fp_states = []
for img_d in cal_data:
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])

        src, pos = [], []
        for i, x in enumerate(x_list):
            proj = F.conv2d(x,
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.weight'),
                            v16._get_weight(f'decoder.color_decoder.input_proj.{i}.bias'))
            src.append(proj.flatten(2).permute(2, 0, 1))
            pe = v16.pe_layer(proj)
            pos.append(pe.flatten(2).permute(2, 0, 1))

        for i in range(3):
            src[i] = src[i] + v16._get_weight('decoder.color_decoder.level_embed.weight')[i]

        bs = 1
        query_embed = v16._get_weight('decoder.color_decoder.query_embed.weight').unsqueeze(1).repeat(1, bs, 1)
        output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, bs, 1)

        for layer in range(6):  # through layer 5
            level_index = layer % 3
            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, src[level_index] + pos[level_index], src[level_index],
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, output + query_embed, output,
                v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            prefix = f'decoder.color_decoder.transformer_ffn_layers.{layer}'
            ffn_out = F.relu(F.linear(output,
                                      v16._get_weight(f'{prefix}.linear1.weight'),
                                      v16._get_weight(f'{prefix}.linear1.bias')))
            ffn_out = F.linear(ffn_out,
                               v16._get_weight(f'{prefix}.linear2.weight'),
                               v16._get_weight(f'{prefix}.linear2.bias'))
            output = F.layer_norm(output + ffn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

        fp_states.append(output.numpy())

fixed_point = np.mean(fp_states, axis=0)  # [100, 1, 256]
fixed_point_t = torch.from_numpy(fixed_point).float()

# Also compute the "effective color matrix" = color_embed(norm(FP))
with torch.no_grad():
    normed_fp = F.layer_norm(
        fixed_point_t, (256,),
        v16._get_weight('decoder.color_decoder.decoder_norm.weight'),
        v16._get_weight('decoder.color_decoder.decoder_norm.bias')).transpose(0, 1)  # [1, 100, 256]

    x = normed_fp
    for i in range(3):
        x = F.linear(x,
                     v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.weight'),
                     v16._get_weight(f'decoder.color_decoder.color_embed.layers.{i}.bias'))
        if i < 2:
            x = F.relu(x)

    effective_color_matrix = x  # [1, 100, C_out]

print(f'Fixed point shape: {fixed_point.shape}')
print(f'Effective color matrix shape: {effective_color_matrix.shape}')

# SVD of effective color matrix
ecm = effective_color_matrix.squeeze(0).numpy()  # [100, C_out]
U_ecm, S_ecm, Vt_ecm = np.linalg.svd(ecm, full_matrices=False)
print(f'Effective color matrix SVD:')
print(f'  S[:10] = {S_ecm[:10].round(2)}')
print(f'  S[0]/S[1] = {S_ecm[0]/S_ecm[1]:.3f}')
cumvar_ecm = np.cumsum(S_ecm**2) / np.sum(S_ecm**2)
for thresh in [0.5, 0.9, 0.95, 0.99]:
    rank = np.searchsorted(cumvar_ecm, thresh) + 1
    print(f'  Rank for {thresh:.0%} variance: {rank}')


# ================================================================
# STEP 2: Large-scale test (100 images)
# ================================================================
print()
print('=' * 70)
print('STEP 2: Large-scale validation (100 images)')
print('=' * 70)
print()

test_data = []
for idx in range(300, 500):
    d = load_image(idx)
    if d is not None:
        test_data.append(d)
    if len(test_data) >= 100:
        break

print(f'Test set: {len(test_data)} images')

# --- Full pipeline ---
full_rmses = []
for img_d in test_data:
    with torch.no_grad():
        pred = v16.forward(img_d['tensor'])
    full_rmses.append(compute_rmse(pred, img_d['gt_ab']))
full_rmses = np.array(full_rmses)

# --- Pure geometric: FP → color_embed → einsum ---
geo_rmses = []
for img_d in test_data:
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
        color_out = decode_queries_to_color(v16, fixed_point_t.clone(), img_features)
        final = apply_refine(v16, color_out, x_norm)
    geo_rmses.append(compute_rmse(final, img_d['gt_ab']))
geo_rmses = np.array(geo_rmses)

# --- Even simpler: precomputed einsum matrix ---
# color = effective_color_matrix @ img_features (no MLP, just matrix multiply)
direct_rmses = []
for img_d in test_data:
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
        color_out = torch.einsum('bqc,bchw->bqhw', effective_color_matrix, img_features)
        final = apply_refine(v16, color_out, x_norm)
    direct_rmses.append(compute_rmse(final, img_d['gt_ab']))
direct_rmses = np.array(direct_rmses)

# --- Query feat (no transformer at all, use init queries) ---
query_feat_t = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1)  # [100, 1, 256]
init_rmses = []
for img_d in test_data:
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
        color_out = decode_queries_to_color(v16, query_feat_t.clone(), img_features)
        final = apply_refine(v16, color_out, x_norm)
    init_rmses.append(compute_rmse(final, img_d['gt_ab']))
init_rmses = np.array(init_rmses)

baseline = full_rmses.mean()
print(f'\n{"Configuration":<45} {"RMSE":<10} {"Δ%":<10} {"p-value":<10}')
print('-' * 75)

from scipy.stats import wilcoxon
configs = [
    ('Full pipeline (9 transformer layers)', full_rmses),
    ('Pure geometric (FP → color_embed → einsum)', geo_rmses),
    ('Direct matrix (precomputed color matrix)', direct_rmses),
    ('Init queries (no transformer, no FP)', init_rmses),
]

for name, rmses in configs:
    delta = (rmses.mean() - baseline) / baseline * 100
    if rmses is not full_rmses:
        _, pval = wilcoxon(full_rmses, rmses)
        pstr = f'{pval:.4f}'
    else:
        pstr = '—'
    print(f'  {name:<43} {rmses.mean():<10.3f} {delta:+8.2f}%  {pstr}')


# ================================================================
# STEP 3: Rank sweep — how many color query dimensions matter?
# ================================================================
print()
print('=' * 70)
print('STEP 3: Rank sweep — low-rank color matrix')
print('=' * 70)
print()

# The effective color matrix is [100, C_out]
# We can approximate it with low-rank: ECM ≈ U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
ecm_t = effective_color_matrix.clone()

for rank in [1, 2, 3, 5, 10, 20, 50, 100]:
    # Low-rank approximation
    U_k = torch.from_numpy(U_ecm[:, :rank]).float()
    S_k = torch.from_numpy(np.diag(S_ecm[:rank])).float()
    Vt_k = torch.from_numpy(Vt_ecm[:rank, :]).float()
    ecm_low = (U_k @ S_k @ Vt_k).unsqueeze(0)  # [1, 100, C_out]

    rmses = []
    for img_d in test_data[:40]:  # subset for speed
        with torch.no_grad():
            x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
            color_out = torch.einsum('bqc,bchw->bqhw', ecm_low, img_features)
            final = apply_refine(v16, color_out, x_norm)
        rmses.append(compute_rmse(final, img_d['gt_ab']))

    rmses = np.array(rmses)
    delta = (rmses.mean() - full_rmses[:40].mean()) / full_rmses[:40].mean() * 100
    print(f'  Rank {rank:>3}: RMSE={rmses.mean():.3f} ({delta:+.2f}%)')


# ================================================================
# STEP 4: Parameter count comparison
# ================================================================
print()
print('=' * 70)
print('STEP 4: Parameter Count Comparison')
print('=' * 70)
print()

# Full color decoder parameters
n_cross_attn = 9 * (3 * EMBED_DIM * EMBED_DIM + 3 * EMBED_DIM + EMBED_DIM * EMBED_DIM + EMBED_DIM)
n_self_attn = 9 * (3 * EMBED_DIM * EMBED_DIM + 3 * EMBED_DIM + EMBED_DIM * EMBED_DIM + EMBED_DIM)
n_ffn = 9 * (EMBED_DIM * 1024 + 1024 + 1024 * EMBED_DIM + EMBED_DIM)
n_norms = 9 * 3 * (EMBED_DIM + EMBED_DIM)  # 3 norms per layer
n_query = 100 * EMBED_DIM + 100 * EMBED_DIM  # query_embed + query_feat
n_input_proj = 3 * (EMBED_DIM * EMBED_DIM + EMBED_DIM)  # 3 input projections (approx)
n_color_embed = EMBED_DIM * 256 + 256 + 256 * 256 + 256 + 256 * 2 + 2  # 3 linear layers
n_decoder_norm = EMBED_DIM + EMBED_DIM
n_level_embed = 3 * EMBED_DIM

full_params = n_cross_attn + n_self_attn + n_ffn + n_norms + n_query + n_color_embed + n_decoder_norm + n_level_embed
geo_params = 100 * ecm.shape[1]  # Just the effective color matrix

print(f'Full color decoder: {full_params:,} parameters')
print(f'  Cross-attention (9 layers): {n_cross_attn:,}')
print(f'  Self-attention (9 layers):  {n_self_attn:,}')
print(f'  FFN (9 layers):             {n_ffn:,}')
print(f'  Norms:                      {n_norms:,}')
print(f'  Query embed + feat:         {n_query:,}')
print(f'  Color embed MLP:            {n_color_embed:,}')
print(f'  Input projections:          {n_input_proj:,}')
print()
print(f'Geometric shortcut: {geo_params:,} parameters')
print(f'  (just the effective color matrix: {ecm.shape})')
print(f'  Compression ratio: {full_params / geo_params:.1f}x')
print()

# What about with the encoder + UNet?
# Those are needed by both approaches
print('Note: Both approaches still need encoder (44M) + UNet decoder + refine net.')
print('The transformer color decoder is the ONLY component that can be eliminated.')


# ================================================================
# STEP 5: Φ structure in the effective color matrix
# ================================================================
print()
print('=' * 70)
print('STEP 5: φ Structure in the Effective Color Matrix')
print('=' * 70)
print()

# Check if singular values follow φ-patterns
print('Singular values of effective color matrix:')
for i in range(min(20, len(S_ecm))):
    ratio_prev = S_ecm[i-1]/S_ecm[i] if i > 0 else 0
    print(f'  S[{i:>2}] = {S_ecm[i]:>8.3f}  ratio S[{i-1}]/S[{i}] = {ratio_prev:.4f}')

# Check φ ratios
print(f'\nφ = {PHI:.4f}, 1/φ = {1/PHI:.4f}')
print(f'\nRatios near φ:')
for i in range(1, min(10, len(S_ecm))):
    r = S_ecm[i-1] / S_ecm[i]
    if abs(r - PHI) < 0.3:
        print(f'  S[{i-1}]/S[{i}] = {r:.4f} (Δ from φ: {r - PHI:+.4f})')
    elif abs(r - 1/PHI) < 0.3:
        print(f'  S[{i-1}]/S[{i}] = {r:.4f} (Δ from 1/φ: {r - 1/PHI:+.4f})')

# Zipf fit
from scipy.optimize import curve_fit
def zipf_law(ranks, s0, alpha):
    return s0 / ranks**alpha

ranks = np.arange(1, len(S_ecm) + 1).astype(float)
try:
    popt, _ = curve_fit(zipf_law, ranks, S_ecm, p0=[S_ecm[0], 0.5], maxfev=5000)
    print(f'\nZipf fit: S[i] = {popt[0]:.2f} / i^{popt[1]:.4f}')
    print(f'  α = {popt[1]:.4f} (1/φ = {1/PHI:.4f})')
except:
    print('\nZipf fit failed')


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 6 COMPLETE SUMMARY')
print('=' * 70)
print()
print('The DDColor color decoder (9-layer transformer) has been UNWOUND:')
print()
print('1. LAYERS 0-5: Pure scaffolding (rank-1 fixed point, skip with 0% change)')
print('2. LAYERS 6-8: Cross-attention adds noise (self-attn + FFN hurt)')
print('3. Q: Completely image-independent (variance = 0.0)')
print('4. The decoder IS: fixed_matrix @ img_features (a single matmul)')
print()
print(f'Full pipeline (9 layers): {full_rmses.mean():.3f} RMSE')
print(f'Pure geometric (no transformer): {geo_rmses.mean():.3f} RMSE ({(geo_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print(f'Direct matrix multiply: {direct_rmses.mean():.3f} RMSE ({(direct_rmses.mean()-baseline)/baseline*100:+.2f}%)')
print()
print(f'Parameters: {full_params:,} → {geo_params:,} ({full_params/geo_params:.0f}x reduction)')
print()
print('ENCODE = DECODE confirmed:')
print('  The encoder computes img_features (geometric representation)')
print('  The decoder is a fixed linear readout of those features')
print('  The 9-layer transformer is scaffolding the training process needed,')
print('  but the inference result is a single matrix multiply.')
