#!/usr/bin/env python3
"""
Experiment 5: Layer 1 MESH Anomaly Investigation
=================================================

Phase 1.5 found that Layer 1 is a structural outlier:
  - Zipf α = 1.28  (all other layers ≈ 0.55-0.72, mean ≈ 0.65 ≈ 1/φ)
  - Condition number κ = 718  (all other layers ≈ 17-120)
  - Top-1 variance = 18.1%  (all other layers ≈ 3-8%)

This means Layer 1's MESH has ONE dominant singular value that captures
18% of all attention variance — a single "axis of attention" that dwarfs
everything else.

Questions:
1. Is this per-head or across all heads?
2. What does the dominant singular vector represent?
3. How does Layer 1 compare to its neighbors (Layer 0, Layer 2)?
4. Is this related to the five-zone architecture (Layer 1 = DRUM zone)?
5. Does the anomaly appear in MLP weights too, or only attention?
6. What is the φ-level distribution — does it differ from other layers?
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS


def efficient_mesh_svd(W_q_head, W_k_head):
    """Compute MESH singular values efficiently via QR + small SVD."""
    A = W_q_head.T.astype(np.float64)  # (3584, 128)
    B = W_k_head.astype(np.float64)     # (128, 3584)
    Q, R = np.linalg.qr(A)
    C = R @ B  # (128, 3584)
    U_c, S, Vt_c = np.linalg.svd(C, full_matrices=False)
    # Full SVD factors: U_mesh = Q @ U_c, Vt_mesh = Vt_c
    U_mesh = Q @ U_c  # (3584, 128)
    return U_mesh, S, Vt_c


def load_layer_weights(layer_idx):
    """Load all weight matrices for a layer."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    weights = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj']:
        phi = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        weights[name] = phi.decode()
    return weights


def analyze_mesh_per_head(W_q, W_k, layer_idx):
    """Detailed per-head MESH analysis."""
    W_q_heads = W_q.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_k_heads = W_k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

    results = []
    for head_idx in range(NUM_HEADS):
        kv_idx = head_idx // HEADS_PER_KV
        U, S, Vt = efficient_mesh_svd(W_q_heads[head_idx], W_k_heads[kv_idx])

        condition = S[0] / S[-1] if S[-1] > 0 else float('inf')
        top1_frac = (S[0] ** 2) / (S ** 2).sum()
        top5_frac = (S[:5] ** 2).sum() / (S ** 2).sum()

        # Zipf α
        ranks = np.arange(1, len(S) + 1)
        alpha = -np.polyfit(np.log(ranks), np.log(S + 1e-20), 1)[0]

        results.append({
            'head': head_idx,
            'kv_group': kv_idx,
            'condition': condition,
            'alpha': alpha,
            'top1_frac': top1_frac,
            'top5_frac': top5_frac,
            'S_top5': S[:5].tolist(),
            'S_ratio_1_2': S[0] / S[1] if S[1] > 0 else float('inf'),
            'U_top1': U[:, 0],  # dominant left singular vector
            'Vt_top1': Vt[0],   # dominant right singular vector
        })

    return results


def analyze_weight_statistics(weights, layer_idx):
    """Compare weight matrix statistics."""
    stats = {}
    for name, W in weights.items():
        flat = W.flatten()
        stats[name] = {
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'abs_mean': float(np.mean(np.abs(flat))),
            'max': float(np.max(np.abs(flat))),
            'sparsity': float((np.abs(flat) < 1e-6).mean()),
            'pos_frac': float((flat > 0).mean()),
        }

        # φ-level entropy
        phi = PhiEncoded.encode(W)
        unique_levels = len(np.unique(phi.exponents))
        stats[name]['unique_levels'] = unique_levels

    return stats


def main():
    print()
    print("=" * 80)
    print("  Experiment 5: Layer 1 MESH Anomaly Investigation")
    print("  α=1.28 (vs mean 0.65≈1/φ), κ=718 (vs typical 20-60)")
    print("=" * 80)
    print()

    # ================================================================
    # Part 1: Per-Head Breakdown — Is it all heads or specific ones?
    # ================================================================
    print("─" * 80)
    print("  Part 1: Per-Head MESH Analysis — Layers 0, 1, 2")
    print("─" * 80)
    print()

    for layer_idx in [0, 1, 2]:
        weights = load_layer_weights(layer_idx)
        heads = analyze_mesh_per_head(weights['q_proj'], weights['k_proj'], layer_idx)

        print(f"  Layer {layer_idx}:")
        print(f"  {'Head':>4s}  {'KV':>2s}  {'κ':>8s}  {'α':>7s}  "
              f"{'top1%':>6s}  {'top5%':>6s}  {'S[0]/S[1]':>9s}  {'S[0]':>10s}")
        print("  " + "-" * 70)

        for h in heads:
            phi_mark = " ←1/φ" if abs(h['alpha'] - 0.618) < 0.1 else ""
            anom_mark = " ★" if h['condition'] > 200 else ""
            print(f"  {h['head']:4d}  {h['kv_group']:2d}  "
                  f"{h['condition']:8.1f}  {h['alpha']:7.4f}  "
                  f"{h['top1_frac']*100:5.1f}%  {h['top5_frac']*100:5.1f}%  "
                  f"{h['S_ratio_1_2']:9.2f}  {h['S_top5'][0]:10.2f}"
                  f"{phi_mark}{anom_mark}")
        print()

        # Summary
        alphas = [h['alpha'] for h in heads]
        conditions = [h['condition'] for h in heads]
        top1s = [h['top1_frac'] for h in heads]
        print(f"  Layer {layer_idx} summary: "
              f"α={np.mean(alphas):.4f} ± {np.std(alphas):.4f}  "
              f"κ={np.mean(conditions):.1f} ± {np.std(conditions):.1f}  "
              f"top1={np.mean(top1s)*100:.1f}%")

        n_anomalous = sum(1 for c in conditions if c > 200)
        if n_anomalous > 0:
            print(f"  ★ {n_anomalous}/{NUM_HEADS} heads have κ > 200")
        print()

        # Save layer 1 head data for deeper analysis
        if layer_idx == 1:
            layer1_heads = heads

    # ================================================================
    # Part 2: The Dominant Singular Vector — What direction is it?
    # ================================================================
    print("─" * 80)
    print("  Part 2: Dominant Singular Vectors of Layer 1")
    print("  What direction captures 18% of attention variance?")
    print("─" * 80)
    print()

    # Find the most anomalous head in layer 1
    worst_head = max(layer1_heads, key=lambda h: h['condition'])
    best_head = min(layer1_heads, key=lambda h: h['condition'])

    print(f"  Most anomalous head: {worst_head['head']} "
          f"(KV group {worst_head['kv_group']}, κ={worst_head['condition']:.1f})")
    print(f"  Least anomalous head: {best_head['head']} "
          f"(KV group {best_head['kv_group']}, κ={best_head['condition']:.1f})")
    print()

    # Analyze the dominant singular vector
    u_top = worst_head['U_top1']  # (3584,) — what input dimensions it attends to
    v_top = worst_head['Vt_top1']  # (3584,) — what output dimensions it produces

    # How concentrated is it?
    u_abs = np.abs(u_top)
    v_abs = np.abs(v_top)
    u_sorted = np.sort(u_abs)[::-1]
    v_sorted = np.sort(v_abs)[::-1]
    u_cumvar = np.cumsum(u_sorted**2) / (u_sorted**2).sum()
    v_cumvar = np.cumsum(v_sorted**2) / (v_sorted**2).sum()

    u_90 = np.searchsorted(u_cumvar, 0.90) + 1
    u_99 = np.searchsorted(u_cumvar, 0.99) + 1
    v_90 = np.searchsorted(v_cumvar, 0.90) + 1
    v_99 = np.searchsorted(v_cumvar, 0.99) + 1

    print(f"  Dominant U vector (input attention pattern):")
    print(f"    90% energy in {u_90}/{HIDDEN_DIM} dims ({u_90/HIDDEN_DIM*100:.1f}%)")
    print(f"    99% energy in {u_99}/{HIDDEN_DIM} dims ({u_99/HIDDEN_DIM*100:.1f}%)")
    print(f"    Effective rank: {1/np.sum((u_abs/u_abs.sum())**2):.0f} dims")
    print()

    print(f"  Dominant V vector (output attention pattern):")
    print(f"    90% energy in {v_90}/{HIDDEN_DIM} dims ({v_90/HIDDEN_DIM*100:.1f}%)")
    print(f"    99% energy in {v_99}/{HIDDEN_DIM} dims ({v_99/HIDDEN_DIM*100:.1f}%)")
    print(f"    Effective rank: {1/np.sum((v_abs/v_abs.sum())**2):.0f} dims")
    print()

    # Cross-head comparison: do all heads share the same dominant direction?
    print("  Cross-head dominant vector similarity (cosine):")
    u_vectors = np.array([h['U_top1'] for h in layer1_heads])
    v_vectors = np.array([h['Vt_top1'] for h in layer1_heads])

    # Normalize
    u_norms = np.linalg.norm(u_vectors, axis=1, keepdims=True)
    u_normed = u_vectors / (u_norms + 1e-20)
    u_sim = np.abs(u_normed @ u_normed.T)

    v_norms = np.linalg.norm(v_vectors, axis=1, keepdims=True)
    v_normed = v_vectors / (v_norms + 1e-20)
    v_sim = np.abs(v_normed @ v_normed.T)

    triu = np.triu_indices(NUM_HEADS, k=1)
    u_avg_sim = float(u_sim[triu].mean())
    v_avg_sim = float(v_sim[triu].mean())

    print(f"    U vectors: avg |cos| = {u_avg_sim:.4f}")
    print(f"    V vectors: avg |cos| = {v_avg_sim:.4f}")

    if u_avg_sim > 0.5:
        print("    → Heads share a COMMON dominant input direction")
    else:
        print("    → Heads have DIVERSE dominant input directions")

    # Check within KV groups (7 Q heads share 1 K head)
    print()
    print("  Within-KV-group similarity (U vectors):")
    for kv_idx in range(NUM_KV_HEADS):
        group_heads = [h for h in layer1_heads if h['kv_group'] == kv_idx]
        group_u = np.array([h['U_top1'] for h in group_heads])
        group_u_norm = group_u / (np.linalg.norm(group_u, axis=1, keepdims=True) + 1e-20)
        group_sim = np.abs(group_u_norm @ group_u_norm.T)
        group_triu = np.triu_indices(len(group_heads), k=1)
        avg = float(group_sim[group_triu].mean()) if len(group_triu[0]) > 0 else 0
        conditions = [h['condition'] for h in group_heads]
        print(f"    KV group {kv_idx}: avg |cos|={avg:.4f}  "
              f"κ range=[{min(conditions):.0f}, {max(conditions):.0f}]")

    print()

    # ================================================================
    # Part 3: Weight Statistics — Is Layer 1 different overall?
    # ================================================================
    print("─" * 80)
    print("  Part 3: Weight Statistics Comparison — Layers 0, 1, 2")
    print("─" * 80)
    print()

    for layer_idx in [0, 1, 2]:
        weights = load_layer_weights(layer_idx)
        stats = analyze_weight_statistics(weights, layer_idx)

        print(f"  Layer {layer_idx}:")
        print(f"  {'Matrix':>12s}  {'std':>8s}  {'|mean|':>8s}  "
              f"{'max':>8s}  {'pos%':>6s}  {'levels':>6s}")
        print("  " + "-" * 56)
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj']:
            s = stats[name]
            print(f"  {name:>12s}  {s['std']:8.5f}  {s['abs_mean']:8.5f}  "
                  f"{s['max']:8.4f}  {s['pos_frac']*100:5.1f}%  "
                  f"{s['unique_levels']:6d}")
        print()

    # ================================================================
    # Part 4: Singular Value Spectrum Deep Dive
    # ================================================================
    print("─" * 80)
    print("  Part 4: Singular Value Spectrum — Layer 1 vs Others")
    print("─" * 80)
    print()

    # Compare full SV spectra for layers 0, 1, 2, 5 (peak), 14 (mid), 27 (last)
    for layer_idx in [0, 1, 2, 5, 14, 27]:
        weights = load_layer_weights(layer_idx)
        W_q = weights['q_proj']
        W_k = weights['k_proj']
        W_q_heads = W_q.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
        W_k_heads = W_k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

        # Average SV spectrum across heads
        all_S = []
        for head_idx in range(NUM_HEADS):
            kv_idx = head_idx // HEADS_PER_KV
            _, S, _ = efficient_mesh_svd(W_q_heads[head_idx], W_k_heads[kv_idx])
            all_S.append(S / S[0])  # Normalize by top SV

        avg_S = np.mean(all_S, axis=0)

        # Show the first 10 normalized SVs
        sv_str = "  ".join(f"{s:.4f}" for s in avg_S[:10])
        print(f"  Layer {layer_idx:2d}: [{sv_str}  ...]")

        # Check for "gap" in spectrum
        ratios = avg_S[:-1] / avg_S[1:]
        max_gap_idx = np.argmax(ratios)
        max_gap_ratio = ratios[max_gap_idx]
        print(f"           max gap at rank {max_gap_idx+1}→{max_gap_idx+2}: "
              f"{max_gap_ratio:.2f}× drop")
        print()

    # ================================================================
    # Part 5: Layer 1 in the Five-Zone Architecture
    # ================================================================
    print("─" * 80)
    print("  Part 5: Layer 1 in the Five-Zone Architecture")
    print("─" * 80)
    print()

    print("  Zone mapping (from Experiment 1c):")
    print("    DRUM (layers 0-2):        Low structure, initial processing")
    print("    TRANSITION (layer 3):     Mode boundary")
    print("    COMB-early (layers 4-6):  Peak structure (88%)")
    print("    COMB-late (layers 7-25):  Stable high structure (70%)")
    print("    MUSIC (layers 26-27):     Output preparation")
    print()

    # Compare all DRUM layers
    print("  DRUM zone MESH comparison:")
    for layer_idx in [0, 1, 2]:
        weights = load_layer_weights(layer_idx)
        heads = analyze_mesh_per_head(weights['q_proj'], weights['k_proj'], layer_idx)

        alphas = [h['alpha'] for h in heads]
        conditions = [h['condition'] for h in heads]
        top1s = [h['top1_frac'] for h in heads]

        print(f"    Layer {layer_idx}: "
              f"α={np.mean(alphas):.4f}  "
              f"κ={np.mean(conditions):.0f}  "
              f"top1={np.mean(top1s)*100:.1f}%  "
              f"{'★ ANOMALY' if np.mean(conditions) > 200 else '✓ normal'}")

    print()

    # ================================================================
    # Part 6: φ-Structure of the Anomaly
    # ================================================================
    print("─" * 80)
    print("  Part 6: φ-Level Distribution — Layer 1 vs Layer 5 (peak)")
    print("─" * 80)
    print()

    for layer_idx in [1, 5]:
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

        for wname in ['q_proj', 'k_proj']:
            path = os.path.join(layer_dir, f'{wname}.npz')
            data = np.load(path)
            exps = data['exponents'].flatten()
            signs = data['signs'].flatten()

            # Histogram of most common levels
            counter = Counter(exps.tolist())
            top_levels = counter.most_common(10)

            pos_frac = (signs > 0).sum() / len(signs)

            print(f"  Layer {layer_idx}/{wname}:")
            print(f"    Unique levels: {len(counter)}")
            print(f"    Sign balance: {pos_frac*100:.1f}% positive")
            print(f"    Top 10 levels (exponent, count, fraction):")
            total = len(exps)
            for exp_val, count in top_levels:
                phi_val = PHI ** (exp_val / PHI_GRID)
                print(f"      exp={exp_val:5d}  "
                      f"φ^({exp_val/PHI_GRID:+.3f}) = {phi_val:.6f}  "
                      f"count={count:8d} ({count/total*100:.2f}%)")
            print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 80)
    print("  SUMMARY: Layer 1 Anomaly")
    print("=" * 80)
    print()

    # Collect the key finding
    worst = max(layer1_heads, key=lambda h: h['condition'])
    best = min(layer1_heads, key=lambda h: h['condition'])
    avg_cond = np.mean([h['condition'] for h in layer1_heads])
    avg_alpha = np.mean([h['alpha'] for h in layer1_heads])
    n_high_cond = sum(1 for h in layer1_heads if h['condition'] > 200)

    print(f"  Layer 1 has {n_high_cond}/{NUM_HEADS} heads with κ > 200")
    print(f"  Average α = {avg_alpha:.4f} (vs 0.618 = 1/φ)")
    print(f"  Average κ = {avg_cond:.0f} (vs typical 20-60)")
    print(f"  Worst head: {worst['head']} (κ={worst['condition']:.0f}, "
          f"S[0]/S[1]={worst['S_ratio_1_2']:.1f})")
    print(f"  Best head:  {best['head']} (κ={best['condition']:.0f}, "
          f"S[0]/S[1]={best['S_ratio_1_2']:.1f})")
    print()
    print(f"  Cross-head dominant direction similarity: |cos|={u_avg_sim:.4f}")
    print()


if __name__ == '__main__':
    main()
