"""
Phase 6B Part 2: What Do Content Layers 6-8 Actually Do?

Phase 6B Part 1 proved:
  - Layers 0-5 are deterministic scaffolding (can be replaced with rank-1 fixed point)
  - ALL image information enters through cross-attention K/V in layers 6-8
  - The fixed point is rank-1 (683:1 ratio)

Now we probe:
  1. Which of layers 6, 7, 8 is most critical?
  2. What does the cross-attention in layer 6 (decision layer) actually compute?
  3. Can we reduce 9 layers → 3 layers (6, 7, 8) with fixed point initialization?
  4. Can we reduce further to just layer 6?
  5. What is the MESH of layer 6 doing geometrically?
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
NUM_HEADS = 8
EMBED_DIM = 256
HEAD_DIM = 32

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


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


def prepare_decoder_inputs(v16, x_list, img_features):
    """Prepare src, pos, query_embed, and initial output for color decoder."""
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

    bs = src[0].shape[1]
    query_embed = v16._get_weight('decoder.color_decoder.query_embed.weight').unsqueeze(1).repeat(1, bs, 1)

    return src, pos, query_embed


def run_single_layer(v16, output, query_embed, src, pos, layer_idx):
    """Run a single transformer layer."""
    level_index = layer_idx % 3

    # Cross-attention
    prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer_idx}'
    attn_out = v16._geometric_multihead_attention(
        output + query_embed, src[level_index] + pos[level_index], src[level_index],
        v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight'),
        v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias'),
        v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight'),
        v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias'))
    output = F.layer_norm(output + attn_out, (256,),
                          v16._get_weight(f'{prefix}.norm.weight'),
                          v16._get_weight(f'{prefix}.norm.bias'))

    # Self-attention
    prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer_idx}'
    attn_out = v16._geometric_multihead_attention(
        output + query_embed, output + query_embed, output,
        v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
        v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
        v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
        v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
    output = F.layer_norm(output + attn_out, (256,),
                          v16._get_weight(f'{prefix}.norm.weight'),
                          v16._get_weight(f'{prefix}.norm.bias'))

    # FFN
    prefix = f'decoder.color_decoder.transformer_ffn_layers.{layer_idx}'
    ffn_out = F.relu(F.linear(output,
                              v16._get_weight(f'{prefix}.linear1.weight'),
                              v16._get_weight(f'{prefix}.linear1.bias')))
    ffn_out = F.linear(ffn_out,
                       v16._get_weight(f'{prefix}.linear2.weight'),
                       v16._get_weight(f'{prefix}.linear2.bias'))
    output = F.layer_norm(output + ffn_out, (256,),
                          v16._get_weight(f'{prefix}.norm.weight'),
                          v16._get_weight(f'{prefix}.norm.bias'))

    return output


def run_cross_attn_only(v16, output, query_embed, src, pos, layer_idx):
    """Run ONLY cross-attention of a single layer (skip self-attn and FFN)."""
    level_index = layer_idx % 3

    prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer_idx}'
    attn_out = v16._geometric_multihead_attention(
        output + query_embed, src[level_index] + pos[level_index], src[level_index],
        v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight'),
        v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias'),
        v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight'),
        v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias'))
    output = F.layer_norm(output + attn_out, (256,),
                          v16._get_weight(f'{prefix}.norm.weight'),
                          v16._get_weight(f'{prefix}.norm.bias'))

    return output


def decode_to_color(v16, output, img_features):
    """Decode query states to color output via final norm + color embed + einsum."""
    decoder_output = F.layer_norm(
        output, (256,),
        v16._get_weight('decoder.color_decoder.decoder_norm.weight'),
        v16._get_weight('decoder.color_decoder.decoder_norm.bias')).transpose(0, 1)

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


def compute_rmse(pred_ab, gt_ab):
    pred_r = cv2.resize(pred_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    return np.sqrt(np.mean((pred_r - gt_ab)**2))


# Load test data
test_data = []
for idx in range(300, 400):
    if len(test_data) >= 40: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    test_data.append({'tensor': t, 'gt_ab': gt_ab, 'idx': idx})

print(f'Test set: {len(test_data)} images')


# ================================================================
# STEP 1: Compute the universal fixed point from calibration
# ================================================================
print()
print('=' * 70)
print('STEP 1: Compute universal fixed point')
print('=' * 70)

# Use first 20 images as calibration for fixed point
fp_states = []
for idx in range(100, 130):
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, t)
        src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)
        output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, 1, 1)
        for layer in range(6):  # Run through layer 5
            output = run_single_layer(v16, output, query_embed, src, pos, layer)
        fp_states.append(output.numpy())

fixed_point = np.mean(fp_states, axis=0)  # [100, 1, 256]
fixed_point_t = torch.from_numpy(fixed_point).float()
print(f'  Fixed point computed from {len(fp_states)} images')
print(f'  Shape: {fixed_point.shape}')


# ================================================================
# STEP 2: Layer ablation — which of 6, 7, 8 is most critical?
# ================================================================
print()
print('=' * 70)
print('STEP 2: Layer ablation (from fixed point)')
print('=' * 70)
print()

configs = {
    'Full (9 layers)': list(range(9)),
    'FP + L6,7,8': [6, 7, 8],
    'FP + L6,7': [6, 7],
    'FP + L7,8': [7, 8],
    'FP + L6,8': [6, 8],
    'FP + L6 only': [6],
    'FP + L7 only': [7],
    'FP + L8 only': [8],
    'FP + no layers': [],
}

results = {}
for config_name, layers in configs.items():
    rmses = []
    for img_d in test_data:
        with torch.no_grad():
            x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
            src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)

            if config_name == 'Full (9 layers)':
                output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, 1, 1)
                for layer in range(9):
                    output = run_single_layer(v16, output, query_embed, src, pos, layer)
            else:
                output = fixed_point_t.clone()
                for layer in layers:
                    output = run_single_layer(v16, output, query_embed, src, pos, layer)

            color_out = decode_to_color(v16, output, img_features)
            final = apply_refine(v16, color_out, x_norm)

        pred_ab = final[0, :2].permute(1, 2, 0).numpy()
        rmses.append(compute_rmse(pred_ab, img_d['gt_ab']))

    rmses = np.array(rmses)
    results[config_name] = rmses

baseline = results['Full (9 layers)'].mean()
print(f'{"Configuration":<25} {"RMSE":<10} {"Δ%":<10}')
print('-' * 45)
for config_name, rmses in results.items():
    delta = (rmses.mean() - baseline) / baseline * 100
    print(f'  {config_name:<23} {rmses.mean():<10.3f} {delta:+8.2f}%')


# ================================================================
# STEP 3: Cross-attention only (skip self-attention + FFN)
# ================================================================
print()
print('=' * 70)
print('STEP 3: Cross-attention only (skip self-attn + FFN)')
print('=' * 70)
print()

ca_configs = {
    'FP + CA6,7,8': [6, 7, 8],
    'FP + CA6 only': [6],
    'FP + CA7 only': [7],
    'FP + CA8 only': [8],
}

for config_name, layers in ca_configs.items():
    rmses = []
    for img_d in test_data:
        with torch.no_grad():
            x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
            src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)

            output = fixed_point_t.clone()
            for layer in layers:
                output = run_cross_attn_only(v16, output, query_embed, src, pos, layer)

            color_out = decode_to_color(v16, output, img_features)
            final = apply_refine(v16, color_out, x_norm)

        pred_ab = final[0, :2].permute(1, 2, 0).numpy()
        rmses.append(compute_rmse(pred_ab, img_d['gt_ab']))

    rmses = np.array(rmses)
    delta = (rmses.mean() - baseline) / baseline * 100
    print(f'  {config_name:<23} {rmses.mean():<10.3f} {delta:+8.2f}%')


# ================================================================
# STEP 4: Direct MESH computation — can we replace attention?
# ================================================================
print()
print('=' * 70)
print('STEP 4: Direct MESH computation — replace softmax attention')
print('=' * 70)
print()

# For layer 6: Q = (output + query_embed) @ W_q, K = (src + pos) @ W_k
# Attention = softmax(Q @ K.T / sqrt(d))
# The MESH = W_q.T @ W_k captures the geometric relationship
# Can we use the fixed-point Q directly?

# Extract attention maps from layer 6 for all test images
layer6_attn_maps = []  # per-image attention from layer 6
layer6_values = []     # per-image value matrices

for img_d in test_data[:10]:  # First 10 for analysis
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
        src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)

        output = fixed_point_t.clone()
        q_input = output + query_embed

        level_index = 6 % 3  # level 0
        k_input = src[level_index] + pos[level_index]
        v_input = src[level_index]

        prefix = 'decoder.color_decoder.transformer_cross_attention_layers.6'
        w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight')
        b = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias')

        q = F.linear(q_input, w[:EMBED_DIM], b[:EMBED_DIM])
        k = F.linear(k_input, w[EMBED_DIM:2*EMBED_DIM], b[EMBED_DIM:2*EMBED_DIM])
        v_proj = F.linear(v_input, w[2*EMBED_DIM:], b[2*EMBED_DIM:])

        seq_len = q.shape[0]
        src_len = k.shape[0]
        bs = 1

        q = q.view(seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1)  # [H, Q, d]
        k = k.view(src_len, NUM_HEADS, HEAD_DIM).transpose(0, 1)  # [H, S, d]
        v_proj = v_proj.view(src_len, NUM_HEADS, HEAD_DIM).transpose(0, 1)

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * (HEAD_DIM ** -0.5), dim=-1)
        layer6_attn_maps.append(attn.numpy())

        # Pre-softmax scores
        scores = torch.bmm(q, k.transpose(1, 2)) * (HEAD_DIM ** -0.5)

print(f'Layer 6 attention map shape: {layer6_attn_maps[0].shape}')  # [H, 100, 256]

# Since Q comes from a fixed point, the query side is IMAGE-INDEPENDENT
# Only K varies per image. This means:
# attn[h, q, s] = softmax( q_fixed[h,q,:] @ K_image[h,s,:].T / sqrt(d) )

# Test: how much does Q vary across images?
all_q = []
for img_d in test_data[:10]:
    with torch.no_grad():
        x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
        src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)

        output = fixed_point_t.clone()
        q_input = output + query_embed
        prefix = 'decoder.color_decoder.transformer_cross_attention_layers.6'
        w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight')
        b = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias')
        q = F.linear(q_input, w[:EMBED_DIM], b[:EMBED_DIM])
        all_q.append(q.squeeze(1).numpy())  # [100, 256]

all_q = np.array(all_q)  # [10, 100, 256]
q_var = all_q.std(axis=0).mean()
q_mean = np.abs(all_q.mean(axis=0)).mean()
print(f'Q variance across images: {q_var:.6f} (mean magnitude: {q_mean:.4f})')
print(f'Q is {"IMAGE-INDEPENDENT" if q_var < 0.001 else "image-dependent"}')

# Since Q is image-independent, we can PRECOMPUTE Q!
# Then attention = softmax(Q_fixed @ K_image.T / sqrt(d))
# This is a single matrix multiply per image, not a full transformer layer.

# Analyze attention sparsity at layer 6
all_attn = np.array(layer6_attn_maps)  # [10, H, Q, S]
mean_attn = all_attn.mean(axis=0)

print(f'\nLayer 6 attention statistics:')
for h in range(NUM_HEADS):
    entropy = -np.sum(mean_attn[h] * np.log(mean_attn[h] + 1e-10), axis=1).mean()
    max_val = mean_attn[h].max(axis=1).mean()
    top5 = np.sort(mean_attn[h], axis=1)[:, -5:].sum(axis=1).mean()
    top10 = np.sort(mean_attn[h], axis=1)[:, -10:].sum(axis=1).mean()
    print(f'  Head {h}: entropy={entropy:.2f}, max={max_val:.3f}, '
          f'top5={top5:.3f}, top10={top10:.3f}')

# What percentage of attention is in top-K positions?
for topk in [1, 3, 5, 10, 20, 50]:
    fracs = []
    for h in range(NUM_HEADS):
        sorted_attn = np.sort(mean_attn[h], axis=1)
        frac = sorted_attn[:, -topk:].sum(axis=1).mean()
        fracs.append(frac)
    print(f'  Top-{topk:>2} positions: {np.mean(fracs):.3f} of attention mass '
          f'(min head: {np.min(fracs):.3f}, max head: {np.max(fracs):.3f})')


# ================================================================
# STEP 5: The complete geometric picture
# ================================================================
print()
print('=' * 70)
print('STEP 5: The Complete Geometric Picture')
print('=' * 70)
print()

# The color decoder can be described as:
# 1. Fixed-point initialization (precomputed, rank-1)
# 2. Three cross-attention lookups into encoder features at levels 0, 1, 2
# 3. Final projection to color space
# 4. Einsum with image features to produce spatial color map

# Can we compute the "effective" operation?
# output_final = FP + Σ_layer CA_layer(FP, encoder_features)

# Test: linear approximation
# If attention is soft but concentrated, we can approximate:
# CA output ≈ weighted sum of a few encoder positions

# For each image, compute the effective operation
print('Testing compression: replacing attention with top-K spatial lookups')
print()

for topk in [1, 3, 5, 10, 20, 50, 100]:
    rmses = []
    for img_d in test_data:
        with torch.no_grad():
            x_norm, x_list, img_features = run_encoder_unet(v16, img_d['tensor'])
            src, pos, query_embed = prepare_decoder_inputs(v16, x_list, img_features)

            output = fixed_point_t.clone()

            # For each of layers 6, 7, 8: run cross-attention but zero out
            # all but top-K attention weights
            for layer_idx in [6, 7, 8]:
                level_index = layer_idx % 3
                q_input = output + query_embed
                k_input = src[level_index] + pos[level_index]
                v_input = src[level_index]

                prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{layer_idx}'
                w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight')
                b_w = v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias')
                wout = v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight')
                bout = v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias')

                q = F.linear(q_input, w[:EMBED_DIM], b_w[:EMBED_DIM])
                k = F.linear(k_input, w[EMBED_DIM:2*EMBED_DIM], b_w[EMBED_DIM:2*EMBED_DIM])
                vp = F.linear(v_input, w[2*EMBED_DIM:], b_w[2*EMBED_DIM:])

                seq_len = q.shape[0]
                src_len_l = k.shape[0]

                q = q.view(seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1)
                k = k.view(src_len_l, NUM_HEADS, HEAD_DIM).transpose(0, 1)
                vp = vp.view(src_len_l, NUM_HEADS, HEAD_DIM).transpose(0, 1)

                scores = torch.bmm(q, k.transpose(1, 2)) * (HEAD_DIM ** -0.5)

                # Top-K masking
                if topk < src_len_l:
                    topk_vals, topk_idxs = scores.topk(topk, dim=-1)
                    mask = torch.full_like(scores, float('-inf'))
                    mask.scatter_(-1, topk_idxs, 0.0)
                    scores = scores + mask

                attn = F.softmax(scores, dim=-1)
                attn_out = torch.bmm(attn, vp)
                attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, 1, EMBED_DIM)
                attn_out = F.linear(attn_out, wout, bout)

                output = F.layer_norm(output + attn_out, (256,),
                                      v16._get_weight(f'{prefix}.norm.weight'),
                                      v16._get_weight(f'{prefix}.norm.bias'))

                # Self-attention (full)
                prefix = f'decoder.color_decoder.transformer_self_attention_layers.{layer_idx}'
                attn_out = v16._geometric_multihead_attention(
                    output + query_embed, output + query_embed, output,
                    v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                    v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                    v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                    v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
                output = F.layer_norm(output + attn_out, (256,),
                                      v16._get_weight(f'{prefix}.norm.weight'),
                                      v16._get_weight(f'{prefix}.norm.bias'))

                # FFN
                prefix = f'decoder.color_decoder.transformer_ffn_layers.{layer_idx}'
                ffn_out = F.relu(F.linear(output,
                                          v16._get_weight(f'{prefix}.linear1.weight'),
                                          v16._get_weight(f'{prefix}.linear1.bias')))
                ffn_out = F.linear(ffn_out,
                                   v16._get_weight(f'{prefix}.linear2.weight'),
                                   v16._get_weight(f'{prefix}.linear2.bias'))
                output = F.layer_norm(output + ffn_out, (256,),
                                      v16._get_weight(f'{prefix}.norm.weight'),
                                      v16._get_weight(f'{prefix}.norm.bias'))

            color_out = decode_to_color(v16, output, img_features)
            final = apply_refine(v16, color_out, x_norm)

        pred_ab = final[0, :2].permute(1, 2, 0).numpy()
        rmses.append(compute_rmse(pred_ab, img_d['gt_ab']))

    rmses = np.array(rmses)
    delta = (rmses.mean() - baseline) / baseline * 100
    print(f'  Top-{topk:>3} cross-attn (L6-8): RMSE={rmses.mean():.3f} ({delta:+.2f}%)')


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY: The Color Decoder Unwound')
print('=' * 70)
print()
print('The 9-layer color decoder can be described as:')
print('  1. RANK-1 FIXED POINT (layers 0-5 are scaffolding)')
print('  2. THREE CONTENT LAYERS (6, 7, 8) with cross-attention to encoder')
print('  3. Q is IMAGE-INDEPENDENT (precomputable)')
print('  4. Only K and V vary per image (from encoder features)')
print()
baseline_rmse = results['Full (9 layers)'].mean() if 'results' in dir() else baseline
print(f'Baseline: {baseline_rmse:.3f}')
