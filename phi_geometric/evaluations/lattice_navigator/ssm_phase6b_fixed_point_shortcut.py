"""
Phase 6B: The Layer-4 Fixed Point — Can We Skip Structural Layers?

Phase 6A found:
  - All images converge to nearly identical query states at layer 4 (cosine 1.000)
  - Layers 5-8 diverge based on image-specific encoder features
  - Layer 6 is the "decision layer" (entropy 1.11, max_attn 0.51)

This script tests:
  1. Extract the layer-4 fixed point (average across images)
  2. Skip layers 0-4, inject fixed point directly into layer 5
  3. Compare: full pipeline vs shortcut pipeline
  4. Measure RMSE impact of skipping structural layers

If this works: layers 0-4 are the "geodesic" (structural), layers 5-8 are the "bulge" (content).
We can precompute the geodesic and only run the content phase.
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


def run_full_pipeline(v16, img_tensor):
    """Run full pipeline, return final color output."""
    with torch.no_grad():
        return v16.forward(img_tensor)


def run_encoder_and_unet(v16, img_tensor):
    """Run encoder + UNet, return intermediate features for color decoder."""
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_input = (img_tensor - mean_t) / std_t

    with torch.no_grad():
        features = v16._geometric_encoder(x_input)
        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)

    return x_input, [out0, out1, out2], out3


def run_color_decoder_full(v16, x_list, img_features):
    """Run full 9-layer color decoder."""
    with torch.no_grad():
        return v16._geometric_color_decoder(x_list, img_features)


def run_color_decoder_from_layer(v16, x_list, img_features, start_layer, init_output):
    """Run color decoder starting from a specific layer with given initial output state."""
    with torch.no_grad():
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
        output = init_output  # Start from provided state

        # Run only layers start_layer through 8
        for i in range(start_layer, 9):
            level_index = i % 3

            # Cross-attention
            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{i}'
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
            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{i}'
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

        # Final norm and color embed
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


def run_color_decoder_capture_state(v16, x_list, img_features, capture_after_layer):
    """Run color decoder and capture query state after a specific layer."""
    with torch.no_grad():
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
        output = v16._get_weight('decoder.color_decoder.query_feat.weight').unsqueeze(1).repeat(1, bs, 1)

        for i in range(capture_after_layer + 1):
            level_index = i % 3

            prefix = f'decoder.color_decoder.transformer_cross_attention_layers.{i}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, src[level_index] + pos[level_index], src[level_index],
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.multihead_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.multihead_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

            prefix = f'decoder.color_decoder.transformer_self_attention_layers.{i}'
            attn_out = v16._geometric_multihead_attention(
                output + query_embed, output + query_embed, output,
                v16._get_weight(f'{prefix}.self_attn.in_proj_weight'),
                v16._get_weight(f'{prefix}.self_attn.in_proj_bias'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.weight'),
                v16._get_weight(f'{prefix}.self_attn.out_proj.bias'))
            output = F.layer_norm(output + attn_out, (256,),
                                  v16._get_weight(f'{prefix}.norm.weight'),
                                  v16._get_weight(f'{prefix}.norm.bias'))

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

        return output


# ================================================================
# STEP 1: Collect fixed point states at each layer
# ================================================================
print('=' * 70)
print('STEP 1: Collect layer states across many images')
print('=' * 70)
print()

N_CALIBRATION = 30
calibration_data = []

for idx in range(100, 100 + N_CALIBRATION * 2):
    if len(calibration_data) >= N_CALIBRATION: break
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0
    calibration_data.append({'tensor': t, 'gt_ab': gt_ab, 'idx': idx})

print(f'Loaded {len(calibration_data)} calibration images')

# Capture states at layers 3, 4, 5 for all images
layer_states = {3: [], 4: [], 5: []}
for img_d in calibration_data:
    x_input, x_list, img_features = run_encoder_and_unet(v16, img_d['tensor'])
    for capture_layer in layer_states.keys():
        state = run_color_decoder_capture_state(v16, x_list, img_features, capture_layer)
        layer_states[capture_layer].append(state.numpy())

# Compute average fixed point at each layer
fixed_points = {}
for layer, states in layer_states.items():
    states_arr = np.array(states)  # [N, Q, 1, D]
    mean_state = states_arr.mean(axis=0)  # [Q, 1, D]
    fixed_points[layer] = mean_state

    # Cross-image similarity
    flat = states_arr.reshape(len(states), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-10
    normed = flat / norms
    cos_mat = normed @ normed.T
    tri = cos_mat[np.triu_indices_from(cos_mat, k=1)]
    print(f'  Layer {layer}: cross-image cosine = {tri.mean():.6f} ± {tri.std():.6f}')


# ================================================================
# STEP 2: Test shortcut — skip layers 0-N, use fixed point
# ================================================================
print()
print('=' * 70)
print('STEP 2: Shortcut Test — Skip structural layers')
print('=' * 70)
print()

# Test images (different from calibration)
test_data = []
for idx in range(300, 400):
    if len(test_data) >= 30: break
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


def eval_pipeline(v16, test_data, mode='full', skip_to_layer=None, fixed_point=None):
    """Evaluate different pipeline configurations."""
    rmses = []
    for img_d in test_data:
        x_input, x_list, img_features = run_encoder_and_unet(v16, img_d['tensor'])

        if mode == 'full':
            color_out = run_color_decoder_full(v16, x_list, img_features)
        elif mode == 'shortcut':
            fp = torch.from_numpy(fixed_point).float()
            color_out = run_color_decoder_from_layer(v16, x_list, img_features, skip_to_layer, fp)

        # Refine net
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_orig = (img_d['tensor'] - mean_t) / std_t
        coarse_input = torch.cat([color_out, x_orig], dim=1)
        final = F.conv2d(coarse_input, v16._get_weight('refine_net.0.0.weight'),
                         v16._get_weight('refine_net.0.0.bias'))

        pred_ab = final[0, :2].permute(1, 2, 0).numpy()
        pred_ab_r = cv2.resize(pred_ab, (img_d['gt_ab'].shape[1], img_d['gt_ab'].shape[0]))
        rmses.append(np.sqrt(np.mean((pred_ab_r - img_d['gt_ab'])**2)))

    return np.array(rmses)


# Baseline
print('Running baseline (full pipeline)...')
baseline_rmses = eval_pipeline(v16, test_data, mode='full')
print(f'  Full pipeline RMSE: {baseline_rmses.mean():.3f} ± {baseline_rmses.std():.3f}')

# Test shortcuts at different skip points
print()
print(f'{"Configuration":<40} {"Mean RMSE":<10} {"Δ%":<10} {"Cosine":<10}')
print('-' * 70)
print(f'  {"Full pipeline (baseline)":<38} {baseline_rmses.mean():<10.3f}')

for skip_to in [1, 2, 3, 4, 5, 6]:
    # Use fixed point closest to skip_to
    fp_layer = min(layer_states.keys(), key=lambda l: abs(l - (skip_to - 1)))
    fp = fixed_points[fp_layer]

    rmses = eval_pipeline(v16, test_data, mode='shortcut',
                          skip_to_layer=skip_to, fixed_point=fp)
    delta_pct = (rmses.mean() - baseline_rmses.mean()) / baseline_rmses.mean() * 100

    # Also compute per-image similarity to baseline
    cos_sim = np.corrcoef(baseline_rmses, rmses)[0, 1]
    print(f'  Skip to layer {skip_to} (fp from L{fp_layer}){"":<17} {rmses.mean():<10.3f} '
          f'{delta_pct:+8.2f}%  {cos_sim:.4f}')


# ================================================================
# STEP 3: Per-image fixed point from that image's own early layers
# ================================================================
print()
print('=' * 70)
print('STEP 3: Per-image fixed point (run early layers per-image)')
print('=' * 70)
print()

# Instead of a global fixed point, capture each image's own L4 state
# and skip from there. This tests if the divergence is useful in L0-4.

print(f'{"Configuration":<45} {"Mean RMSE":<10} {"Δ%":<10}')
print('-' * 65)
print(f'  {"Full pipeline (baseline)":<43} {baseline_rmses.mean():<10.3f}')

for skip_after in [2, 3, 4, 5, 6]:
    rmses = []
    for img_d in test_data:
        x_input, x_list, img_features = run_encoder_and_unet(v16, img_d['tensor'])

        # Run through layer skip_after, capture state
        own_state = run_color_decoder_capture_state(v16, x_list, img_features, skip_after)

        # Continue from skip_after+1
        color_out = run_color_decoder_from_layer(v16, x_list, img_features,
                                                  skip_after + 1, own_state)

        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_orig = (img_d['tensor'] - mean_t) / std_t
        coarse_input = torch.cat([color_out, x_orig], dim=1)
        final = F.conv2d(coarse_input, v16._get_weight('refine_net.0.0.weight'),
                         v16._get_weight('refine_net.0.0.bias'))

        pred_ab = final[0, :2].permute(1, 2, 0).numpy()
        pred_ab_r = cv2.resize(pred_ab, (img_d['gt_ab'].shape[1], img_d['gt_ab'].shape[0]))
        rmses.append(np.sqrt(np.mean((pred_ab_r - img_d['gt_ab'])**2)))

    rmses = np.array(rmses)
    delta_pct = (rmses.mean() - baseline_rmses.mean()) / baseline_rmses.mean() * 100
    print(f'  Run L0-{skip_after}, skip L{skip_after+1}+ (sanity check) {"":<5} {rmses.mean():<10.3f} '
          f'{delta_pct:+8.2f}%')


# ================================================================
# STEP 4: What does the fixed point encode?
# ================================================================
print()
print('=' * 70)
print('STEP 4: What does the layer-4 fixed point encode?')
print('=' * 70)
print()

# SVD of the fixed point matrix [100 queries × 256 dims]
fp4 = fixed_points[4].squeeze(1)  # [100, 256]
U, S, Vt = np.linalg.svd(fp4, full_matrices=False)

print(f'Fixed point at layer 4:')
print(f'  Shape: {fp4.shape}')
print(f'  Norm: mean={np.linalg.norm(fp4, axis=1).mean():.2f}')
print(f'  SVD: S[0]={S[0]:.2f}, S[1]={S[1]:.2f}, S0/S1={S[0]/S[1]:.3f}')

# Effective dimensionality
cumvar = np.cumsum(S**2) / np.sum(S**2)
for thresh in [0.5, 0.9, 0.95, 0.99]:
    rank = np.searchsorted(cumvar, thresh) + 1
    print(f'  Rank for {thresh:.0%} variance: {rank}')

# Compare fixed point to query_feat
query_feat = v16._get_weight('decoder.color_decoder.query_feat.weight').numpy()
cos_feat = np.sum(fp4 * query_feat, axis=1) / (
    np.linalg.norm(fp4, axis=1) * np.linalg.norm(query_feat, axis=1) + 1e-10)
print(f'\n  Cosine(fixed_point, query_feat): mean={cos_feat.mean():.4f}, std={cos_feat.std():.4f}')
angle = np.degrees(np.arccos(np.clip(cos_feat.mean(), -1, 1)))
print(f'  Average angle from init: {angle:.1f}°')


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('Can we skip structural layers (0-4) with a precomputed fixed point?')
print('If RMSE barely changes, layers 0-4 are pure "scaffolding".')
print('If RMSE degrades, those layers encode image-specific information')
print('despite having similar query states across images.')
