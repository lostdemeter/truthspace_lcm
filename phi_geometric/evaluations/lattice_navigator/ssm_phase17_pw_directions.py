"""
Phase 17: Cracking the PW Direction Problem

25.9M params (47.1% of DDColor) live in PW1/PW2 directions.
Phase 12 showed: sparse (88%), heavy-tailed, block-specific.

Four attack vectors tested here:
  A. Sparse Dictionary — shared basis across blocks
  B. Effective Gated Transform — what GELU+composition actually computes
  C. Active Channel Patterns — which expanded dims survive GELU?
  D. Shared Subspace Extraction — the 33% overlap between blocks

Goal: find ANY structure that allows analytic construction or massive compression.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
import glob
from collections import defaultdict

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/models')
from geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = (1 + np.sqrt(5)) / 2

print("Loading V16...")
v16 = V16GeometricColorizer()

dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

# ================================================================
# Collect all PW1/PW2 weights and their SVDs
# ================================================================
all_pw = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        for pw in ['pwconv1', 'pwconv2']:
            key = (stage_idx, block_idx, pw)
            w = v16._get_weight(f'{prefix}.{pw}.weight')
            b = v16._get_weight(f'{prefix}.{pw}.bias')
            if w is None:
                continue
            w_np = w.numpy()
            U, S, Vt = np.linalg.svd(w_np, full_matrices=False)
            all_pw[key] = {
                'w': w_np, 'b': b.numpy() if b is not None else None,
                'U': U, 'S': S, 'Vt': Vt, 'shape': w_np.shape
            }

# Also collect norm weights (needed for computing pre-GELU distribution)
all_norms = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'
        nw = v16._get_weight(f'{prefix}.norm.weight')
        nb = v16._get_weight(f'{prefix}.norm.bias')
        if nw is not None:
            all_norms[(stage_idx, block_idx)] = {
                'weight': nw.numpy(), 'bias': nb.numpy()
            }


# ================================================================
# ATTACK A: Sparse Dictionary
# ================================================================
print()
print('=' * 70)
print('ATTACK A: Sparse Dictionary Across Blocks')
print('=' * 70)
print()

# For each stage, stack the top-K singular vectors from ALL blocks
# Then find a shared dictionary via SVD of the stacked vectors

for stage_idx in range(4):
    C = dims[stage_idx]
    expand = 4 * C

    # Stack U vectors (left singular vectors of PW1, shape [4C, rank])
    # These are the "output directions" of PW1 = "expanded space directions"
    u_vecs_pw1 = []
    v_vecs_pw1 = []  # Right singular vectors = "input directions"
    u_vecs_pw2 = []
    v_vecs_pw2 = []

    K_top = min(50, C)  # Top singular vectors per block

    for block_idx in range(depths[stage_idx]):
        key1 = (stage_idx, block_idx, 'pwconv1')
        key2 = (stage_idx, block_idx, 'pwconv2')
        if key1 in all_pw:
            u_vecs_pw1.append(all_pw[key1]['U'][:, :K_top])  # [4C, K]
            v_vecs_pw1.append(all_pw[key1]['Vt'][:K_top, :])  # [K, C]
        if key2 in all_pw:
            u_vecs_pw2.append(all_pw[key2]['U'][:, :K_top])  # [C, K]
            v_vecs_pw2.append(all_pw[key2]['Vt'][:K_top, :])  # [K, 4C]

    if not u_vecs_pw1:
        continue

    n_blocks = depths[stage_idx]

    # Stack: [4C, K*n_blocks] — all U vectors from all blocks
    U_stacked = np.hstack(u_vecs_pw1)  # [4C, K*n_blocks]
    V_stacked = np.vstack(v_vecs_pw1)  # [K*n_blocks, C]

    # SVD of stacked vectors to find shared dictionary
    _, S_dict_u, _ = np.linalg.svd(U_stacked, full_matrices=False)
    _, S_dict_v, _ = np.linalg.svd(V_stacked, full_matrices=False)

    cumvar_u = np.cumsum(S_dict_u**2) / np.sum(S_dict_u**2)
    cumvar_v = np.cumsum(S_dict_v**2) / np.sum(S_dict_v**2)

    rank90_u = np.searchsorted(cumvar_u, 0.90) + 1
    rank95_u = np.searchsorted(cumvar_u, 0.95) + 1
    rank99_u = np.searchsorted(cumvar_u, 0.99) + 1

    rank90_v = np.searchsorted(cumvar_v, 0.90) + 1
    rank95_v = np.searchsorted(cumvar_v, 0.95) + 1
    rank99_v = np.searchsorted(cumvar_v, 0.99) + 1

    total_vecs = K_top * n_blocks
    print(f"Stage {stage_idx} (C={C}, 4C={expand}, {n_blocks} blocks, {total_vecs} vectors):")
    print(f"  U directions (expanded space, dim={expand}):")
    print(f"    90% variance: rank {rank90_u}/{total_vecs} ({rank90_u/total_vecs*100:.0f}%)")
    print(f"    95% variance: rank {rank95_u}/{total_vecs}")
    print(f"    99% variance: rank {rank99_u}/{total_vecs}")
    print(f"  V directions (input space, dim={C}):")
    print(f"    90% variance: rank {rank90_v}/{total_vecs} ({rank90_v/total_vecs*100:.0f}%)")
    print(f"    95% variance: rank {rank95_v}/{total_vecs}")
    print(f"    99% variance: rank {rank99_v}/{total_vecs}")

    # Key question: can we reconstruct individual block's U from the shared dictionary?
    # Take the shared dictionary (top-K_dict singular vectors of stacked U)
    U_full, S_full, Vt_full = np.linalg.svd(U_stacked, full_matrices=False)

    for K_dict in [C // 4, C // 2, C, min(2*C, U_stacked.shape[1])]:
        if K_dict > U_stacked.shape[1]:
            continue
        D = U_full[:, :K_dict]  # Shared dictionary [4C, K_dict]

        # For each block, project its U vectors onto D and measure reconstruction
        recon_errors = []
        for bi, u_block in enumerate(u_vecs_pw1):
            # u_block: [4C, K_top]
            # Project: coeffs = D.T @ u_block, recon = D @ coeffs
            coeffs = D.T @ u_block  # [K_dict, K_top]
            recon = D @ coeffs      # [4C, K_top]
            err = np.linalg.norm(u_block - recon) / np.linalg.norm(u_block)
            recon_errors.append(err)

        mean_err = np.mean(recon_errors)
        print(f"  Dictionary K={K_dict}: mean recon error = {mean_err:.4f} "
              f"({(1-mean_err)*100:.1f}% captured)")

    print()


# ================================================================
# ATTACK B: Effective Gated Transform
# ================================================================
print('=' * 70)
print('ATTACK B: Effective Gated Transform (W2 @ diag(gate) @ W1)')
print('=' * 70)
print()

# Run real images through the encoder, capture pre-GELU activations
# Then compute what the EFFECTIVE transform is

images = sorted(glob.glob(
    '/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

# Process 5 images and capture gate patterns
N_IMGS = 5
gate_stats = defaultdict(list)

for img_idx in range(300, 300 + N_IMGS * 2):
    if len(gate_stats.get((0, 0), [])) >= N_IMGS:
        break
    im = cv2.imread(images[img_idx])
    if im is None:
        continue

    r = cv2.resize(im, (256, 256))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(g3.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (t - mean_t) / std_t

    with torch.no_grad():
        # Run through encoder, capturing gate activations
        # Stem
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (96,),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0, 3, 1, 2)

        for stage_idx in range(4):
            dim = dims[stage_idx]

            if stage_idx > 0:
                prefix = f'encoder.arch.downsample_layers.{stage_idx}'
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, (dims[stage_idx-1],),
                                 v16._get_weight(f'{prefix}.0.weight'),
                                 v16._get_weight(f'{prefix}.0.bias'))
                x = x.permute(0, 3, 1, 2)
                x = F.conv2d(x, v16._get_weight(f'{prefix}.1.weight'),
                             v16._get_weight(f'{prefix}.1.bias'), stride=2)

            for block_idx in range(depths[stage_idx]):
                residual = x
                prefix = f'encoder.arch.stages.{stage_idx}.{block_idx}'

                # DW conv
                xb = F.conv2d(x, v16._get_weight(f'{prefix}.dwconv.weight'),
                             v16._get_weight(f'{prefix}.dwconv.bias'),
                             padding=3, groups=dim)

                # LayerNorm
                xb = xb.permute(0, 2, 3, 1)
                xb = F.layer_norm(xb, (dim,),
                                 v16._get_weight(f'{prefix}.norm.weight'),
                                 v16._get_weight(f'{prefix}.norm.bias'))

                # PW1
                pre_gelu = F.linear(xb,
                                    v16._get_weight(f'{prefix}.pwconv1.weight'),
                                    v16._get_weight(f'{prefix}.pwconv1.bias'))

                # Capture gate pattern
                gate_mask = (pre_gelu > 0).float()
                survival = gate_mask.mean().item()
                gate_stats[(stage_idx, block_idx)].append({
                    'survival': survival,
                    'pre_gelu_mean': pre_gelu.mean().item(),
                    'pre_gelu_std': pre_gelu.std().item(),
                    # Per-channel survival rate (average over spatial dims)
                    'channel_survival': gate_mask.mean(dim=(0, 1, 2)).numpy(),
                })

                # Complete the block
                post_gelu = pre_gelu * 0.5 * (1.0 + torch.erf(pre_gelu / np.sqrt(2.0)))
                xb = F.linear(post_gelu,
                             v16._get_weight(f'{prefix}.pwconv2.weight'),
                             v16._get_weight(f'{prefix}.pwconv2.bias'))
                xb = xb.permute(0, 3, 1, 2)

                gamma = v16._get_weight(f'{prefix}.gamma')
                x = residual + gamma.view(1, -1, 1, 1) * xb

# Analyze gate patterns
print(f"{'Block':<10} {'Survival':<10} {'Dead (<5%)':<12} {'Always (>95%)':<14} "
      f"{'Variable':<10} {'Eff rank':<10}")
print("-" * 66)

effective_ranks = {}
for stage_idx in range(4):
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        stats = gate_stats[key]

        # Average channel survival across images
        avg_ch_survival = np.mean([s['channel_survival'] for s in stats], axis=0)
        mean_surv = np.mean([s['survival'] for s in stats])

        dead = np.sum(avg_ch_survival < 0.05)
        always_on = np.sum(avg_ch_survival > 0.95)
        variable = len(avg_ch_survival) - dead - always_on

        # Effective rank: number of channels that sometimes activate
        eff_rank = np.sum((avg_ch_survival > 0.05) & (avg_ch_survival < 0.95))
        effective_ranks[key] = {
            'dead': dead, 'always_on': always_on,
            'variable': variable, 'eff_rank': eff_rank,
            'total': len(avg_ch_survival),
            'channel_survival': avg_ch_survival
        }

        print(f"  {stage_idx}.{block_idx:<7} {mean_surv:<10.1%} {dead:<12} "
              f"{always_on:<14} {variable:<10} {eff_rank:<10}")

# Summary
total_dead = sum(v['dead'] for v in effective_ranks.values())
total_always = sum(v['always_on'] for v in effective_ranks.values())
total_variable = sum(v['variable'] for v in effective_ranks.values())
total_channels = sum(v['total'] for v in effective_ranks.values())

print(f"\n  Total expanded channels: {total_channels}")
print(f"  Always dead (<5%):    {total_dead} ({total_dead/total_channels*100:.1f}%)")
print(f"  Always alive (>95%):  {total_always} ({total_always/total_channels*100:.1f}%)")
print(f"  Variable:             {total_variable} ({total_variable/total_channels*100:.1f}%)")
print(f"\n  If we pre-prune dead channels: {total_dead/total_channels*100:.0f}% param reduction in PW")


# ================================================================
# ATTACK C: Effective Transform Rank
# ================================================================
print()
print('=' * 70)
print('ATTACK C: Effective Transform Rank (W2 @ alive_mask @ W1)')
print('=' * 70)
print()

# For each block, compute the effective transform using only "alive" channels
# Compare rank to full transform

print(f"{'Block':<10} {'Full rank':<12} {'Alive rank':<12} {'Dead rank':<12} "
      f"{'Alive 90%':<12} {'Compression':<12}")
print("-" * 70)

for stage_idx in range(4):
    C = dims[stage_idx]
    for block_idx in range(depths[stage_idx]):
        key = (stage_idx, block_idx)
        key1 = (stage_idx, block_idx, 'pwconv1')
        key2 = (stage_idx, block_idx, 'pwconv2')

        if key1 not in all_pw or key2 not in all_pw:
            continue

        W1 = all_pw[key1]['w']  # [4C, C]
        W2 = all_pw[key2]['w']  # [C, 4C]

        ch_surv = effective_ranks[key]['channel_survival']
        alive = ch_surv > 0.05
        dead = ~alive
        n_alive = alive.sum()

        # Full effective transform: W2 @ W1 (ignoring nonlinearity)
        W_full = W2 @ W1  # [C, C]
        S_full = np.linalg.svdvals(W_full)
        cumvar_full = np.cumsum(S_full**2) / np.sum(S_full**2)
        rank90_full = np.searchsorted(cumvar_full, 0.90) + 1

        # Alive-only: W2[:, alive] @ W1[alive, :]
        W_alive = W2[:, alive] @ W1[alive, :]  # [C, C]
        S_alive = np.linalg.svdvals(W_alive)
        cumvar_alive = np.cumsum(S_alive**2) / np.sum(S_alive**2)
        rank90_alive = np.searchsorted(cumvar_alive, 0.90) + 1

        # Dead-only: what we lose by pruning
        W_dead = W2[:, dead] @ W1[dead, :]
        energy_dead = np.sum(W_dead**2) / np.sum(W_full**2)

        compression = 1.0 - n_alive / (4 * C)

        print(f"  {stage_idx}.{block_idx:<7} {rank90_full:<12} {rank90_alive:<12} "
              f"{energy_dead:<12.4f} {rank90_alive:<12} {compression:<12.1%}")


# ================================================================
# ATTACK D: Shared Subspace Extraction
# ================================================================
print()
print('=' * 70)
print('ATTACK D: Shared vs Block-Specific Subspace')
print('=' * 70)
print()

# For stage 2 (9 blocks, C=384), find the shared subspace
# Method: SVD of concatenated V vectors, keep shared part, measure residual

for stage_idx in [2]:  # Focus on the big stage
    C = dims[stage_idx]
    n_blocks = depths[stage_idx]

    # Collect the FULL weight matrices (not just top SVs)
    W1_list = []
    W2_list = []
    for block_idx in range(n_blocks):
        key1 = (stage_idx, block_idx, 'pwconv1')
        key2 = (stage_idx, block_idx, 'pwconv2')
        W1_list.append(all_pw[key1]['w'])
        W2_list.append(all_pw[key2]['w'])

    # Method: for PW1, compute V1t (right singular vectors = input directions)
    # Stack all of them: [9*rank, C]
    # SVD to find shared input subspace

    K = min(C, 50)  # Top K SVs per block
    V_input_all = np.vstack([all_pw[(stage_idx, b, 'pwconv1')]['Vt'][:K] for b in range(n_blocks)])
    # [9*K, C]

    U_shared, S_shared, Vt_shared = np.linalg.svd(V_input_all, full_matrices=False)
    cumvar_shared = np.cumsum(S_shared**2) / np.sum(S_shared**2)

    rank50 = np.searchsorted(cumvar_shared, 0.50) + 1
    rank75 = np.searchsorted(cumvar_shared, 0.75) + 1
    rank90 = np.searchsorted(cumvar_shared, 0.90) + 1
    rank95 = np.searchsorted(cumvar_shared, 0.95) + 1

    print(f"Stage {stage_idx}: Input directions (V of PW1), {n_blocks} blocks × {K} SVs")
    print(f"  Shared subspace dimensionality:")
    print(f"    50% variance: {rank50}/{K*n_blocks}")
    print(f"    75% variance: {rank75}")
    print(f"    90% variance: {rank90}")
    print(f"    95% variance: {rank95}")
    print(f"    Full rank available: {C}")
    print()

    # Now the key test: reconstruct each block's W1 using ONLY the shared subspace
    # W1 ≈ W1 @ P_shared, where P_shared = V_shared @ V_shared.T is the projector
    print(f"  Reconstruction from shared subspace (project W1 onto shared V):")
    print(f"  {'K_shared':<12} {'Mean recon':<15} {'RMSE impact':<15}")
    print(f"  " + "-" * 42)

    for K_shared in [C // 8, C // 4, C // 2, C * 3 // 4, C]:
        if K_shared > len(S_shared):
            continue
        V_proj = Vt_shared[:K_shared]  # [K_shared, C]
        P = V_proj.T @ V_proj  # [C, C] projector

        recon_quality = []
        for b in range(n_blocks):
            W1_orig = W1_list[b]  # [4C, C]
            W1_proj = W1_orig @ P  # [4C, C]
            err = np.linalg.norm(W1_orig - W1_proj) / np.linalg.norm(W1_orig)
            recon_quality.append(err)

        print(f"  {K_shared:<12} {np.mean(recon_quality):<15.4f} "
              f"(~{np.mean(recon_quality)*100:.1f}% energy lost)")

    # Now test: what if we decompose W1 = W1_shared + W1_specific?
    # W1_shared lives in the shared subspace
    # W1_specific is the block-specific residual
    # How complex is W1_specific?
    print(f"\n  Block-specific residual complexity (after removing shared subspace):")
    K_shared = C // 2  # Use 50% shared subspace
    V_proj = Vt_shared[:K_shared]
    P = V_proj.T @ V_proj

    print(f"  Using K_shared={K_shared} (50% of C={C}):")
    print(f"  {'Block':<10} {'Residual norm':<15} {'Residual rank@90%':<20} {'Original rank@90%':<20}")
    print(f"  " + "-" * 65)

    for b in range(n_blocks):
        W1_orig = W1_list[b]
        W1_shared_part = W1_orig @ P
        W1_residual = W1_orig - W1_shared_part

        S_orig = np.linalg.svdvals(W1_orig)
        S_resid = np.linalg.svdvals(W1_residual)

        cumvar_orig = np.cumsum(S_orig**2) / np.sum(S_orig**2)
        cumvar_resid = np.cumsum(S_resid**2) / (np.sum(S_resid**2) + 1e-10)

        rank90_orig = np.searchsorted(cumvar_orig, 0.90) + 1
        rank90_resid = np.searchsorted(cumvar_resid, 0.90) + 1

        resid_energy = np.sum(W1_residual**2) / np.sum(W1_orig**2)

        print(f"  {b:<10} {resid_energy:<15.4f} {rank90_resid:<20} {rank90_orig:<20}")


# ================================================================
# SUMMARY
# ================================================================
print()
print('=' * 70)
print('PHASE 17 SUMMARY')
print('=' * 70)
print()
print(f"A. Sparse Dictionary: see per-stage results above")
print(f"B. Gate pattern: {total_dead/total_channels*100:.0f}% of expanded channels are always dead")
print(f"C. Effective transform: alive-only preserves rank structure")
print(f"D. Shared subspace: see stage 2 analysis above")
print()
print("MOST PROMISING ATTACK VECTOR: [determined by results]")
