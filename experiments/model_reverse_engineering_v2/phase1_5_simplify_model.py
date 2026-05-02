#!/usr/bin/env python3
"""
Phase 1.5: Mesh Simplification — Analyze & Simplify φ-Encoded Qwen2-7B
========================================================================

Applies the same AIG/mesh simplification pipeline used for IPA model:

Part 1: φ-Level Histogram & Entropy
    → Like IPA's information content analysis
    → How many bits does each weight ACTUALLY carry?

Part 2: MESH Low-Rank Analysis
    → Like AIG's "shared sub-expressions"
    → MESH = W_q.T @ W_k is rank-128 → 14× attention speedup

Part 3: Cross-Layer Structural Hashing
    → Like AIG's structural hashing
    → Which layers share the same φ-level patterns?

Part 4: Level-Grouped Format
    → Like IPA's SimplifiedExecutor
    → Reorganize weights by φ-level for grouped integer matmul

IPA result:  159 gate_steps → 283 bytes
Qwen2 goal:  7.6B params → ? GB of actual information
"""

import sys
import os
import json
import time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "phi_model_simplified")

# Qwen2-7B config
NUM_LAYERS = 28
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 7


def load_exponents(path):
    """Load just the exponent array from a φ-encoded .npz file."""
    data = np.load(path)
    return data['exponents']


def load_signs(path):
    """Load just the sign array from a φ-encoded .npz file."""
    data = np.load(path)
    return data['signs']


def entropy_bits(exponents):
    """Shannon entropy of exponent distribution in bits."""
    flat = exponents.flatten()
    counts = np.bincount(flat.astype(np.int32) - flat.min())
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def sign_entropy_bits(signs):
    """Shannon entropy of sign distribution in bits."""
    flat = signs.flatten()
    n_pos = (flat > 0).sum()
    n_neg = (flat < 0).sum()
    total = n_pos + n_neg
    if total == 0 or n_pos == 0 or n_neg == 0:
        return 0.0
    p_pos = n_pos / total
    p_neg = n_neg / total
    return float(-p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg))


# ============================================================================
# Part 1: φ-Level Histogram & Entropy
# ============================================================================

def analyze_information_content():
    """Compute per-component entropy and true information content."""
    print("=" * 80)
    print("  Part 1: φ-LEVEL HISTOGRAM & ENTROPY")
    print("  How many bits does each weight actually carry?")
    print("=" * 80)
    print()

    results = []
    total_params = 0
    total_info_bits = 0

    # Process global components
    for name in ['embed_tokens', 'lm_head']:
        path = os.path.join(MODEL_DIR, f'{name}.npz')
        exps = load_exponents(path)
        sgns = load_signs(path)

        n_unique = len(np.unique(exps))
        exp_H = entropy_bits(exps)
        sgn_H = sign_entropy_bits(sgns)
        total_H = exp_H + sgn_H
        n_params = exps.size
        info_bits = total_H * n_params

        total_params += n_params
        total_info_bits += info_bits

        results.append({
            'name': name,
            'params': int(n_params),
            'unique_levels': int(n_unique),
            'exp_entropy_bits': round(exp_H, 3),
            'sign_entropy_bits': round(sgn_H, 3),
            'total_entropy_bits': round(total_H, 3),
            'info_bytes': int(info_bits / 8),
        })

        print(f"  {name:40s}  levels={n_unique:5d}  "
              f"H_exp={exp_H:.3f}  H_sgn={sgn_H:.3f}  "
              f"H_total={total_H:.3f} bits/weight  "
              f"info={info_bits/8/1e6:.1f} MB")

    # Process per-layer components
    weight_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']

    # Collect per-layer histograms for cross-layer analysis
    layer_histograms = {}

    for layer_idx in range(NUM_LAYERS):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        layer_total_H = 0
        layer_params = 0

        for wname in weight_names:
            path = os.path.join(layer_dir, f'{wname}.npz')
            exps = load_exponents(path)
            sgns = load_signs(path)

            n_unique = len(np.unique(exps))
            exp_H = entropy_bits(exps)
            sgn_H = sign_entropy_bits(sgns)
            total_H = exp_H + sgn_H
            n_params = exps.size
            info_bits = total_H * n_params

            total_params += n_params
            total_info_bits += info_bits
            layer_total_H += info_bits
            layer_params += n_params

            # Store histogram for cross-layer analysis
            key = f'layer_{layer_idx:02d}/{wname}'
            hist = Counter(exps.flatten().tolist())
            layer_histograms[key] = hist

            results.append({
                'name': f'layer_{layer_idx:02d}/{wname}',
                'params': int(n_params),
                'unique_levels': int(n_unique),
                'exp_entropy_bits': round(exp_H, 3),
                'sign_entropy_bits': round(sgn_H, 3),
                'total_entropy_bits': round(total_H, 3),
                'info_bytes': int(info_bits / 8),
            })

        layer_H_avg = layer_total_H / layer_params if layer_params > 0 else 0
        print(f"  layer_{layer_idx:02d}  "
              f"avg H={layer_H_avg:.3f} bits/weight  "
              f"info={layer_total_H/8/1e6:.1f} MB  "
              f"({layer_params:,} params)")

    # Summary
    avg_H = total_info_bits / total_params if total_params > 0 else 0
    raw_bytes = total_params * 3  # sign(1) + exp(2) per value
    float32_bytes = total_params * 4
    info_bytes = total_info_bits / 8

    print()
    print(f"  TOTAL: {total_params:,} parameters")
    print(f"    Float32 encoding:   {float32_bytes/1e9:.2f} GB  (32.0 bits/weight)")
    print(f"    φ-encoded (raw):    {raw_bytes/1e9:.2f} GB  (24.0 bits/weight)")
    print(f"    Information content: {info_bytes/1e9:.2f} GB  ({avg_H:.3f} bits/weight)")
    print(f"    Redundancy ratio:   {raw_bytes/info_bytes:.2f}×")
    print()

    return results, layer_histograms, {
        'total_params': int(total_params),
        'avg_entropy_bits': round(avg_H, 4),
        'info_bytes': int(info_bytes),
        'info_gb': round(info_bytes / 1e9, 3),
        'raw_bytes': int(raw_bytes),
        'float32_bytes': int(float32_bytes),
        'redundancy_ratio': round(raw_bytes / info_bytes, 2),
    }


# ============================================================================
# Part 2: MESH Low-Rank Analysis
# ============================================================================

def analyze_mesh():
    """Analyze MESH = W_q_head.T @ W_k_head structure using efficient factored SVD.

    Since MESH is rank-128 by construction (product of 128-wide matrices),
    we use QR + small SVD instead of full (3584×3584) SVD:

        A = W_q_head.T  (3584, 128)
        B = W_k_head    (128, 3584)
        QR: A = Q @ R   (Q: 3584×128, R: 128×128)
        C = R @ B       (128, 3584)
        SVD(C) → U_c, S, Vt  (128×128 problem — instant)

    This is 780× faster than full SVD of (3584, 3584).
    """
    print("=" * 80)
    print("  Part 2: MESH LOW-RANK ANALYSIS")
    print("  MESH = W_q_head.T @ W_k_head — rank 128 by construction")
    print("=" * 80)
    print()

    mesh_results = []
    all_sv_spectra = []  # Collect for φ-Zipf analysis

    for layer_idx in range(NUM_LAYERS):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

        # Load Q and K weights, decode to float
        q_phi = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
        k_phi = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
        W_q = q_phi.decode()  # (3584, 3584)
        W_k = k_phi.decode()  # (512, 3584)

        # Reshape for multi-head
        W_q_heads = W_q.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
        W_k_heads = W_k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

        layer_condition = []
        layer_sv_decay = []
        layer_top1_frac = []

        for head_idx in range(NUM_HEADS):
            kv_idx = head_idx // HEADS_PER_KV
            # A = W_q_head.T (3584, 128), B = W_k_head (128, 3584)
            A = W_q_heads[head_idx].T.astype(np.float64)
            B = W_k_heads[kv_idx].astype(np.float64)

            # Efficient factored SVD: QR(A) then SVD(R @ B)
            Q, R = np.linalg.qr(A)           # Q: (3584, 128), R: (128, 128)
            C = R @ B                          # (128, 3584) — small!
            _, S, _ = np.linalg.svd(C, full_matrices=False)
            # S: (128,) — the singular values of MESH

            condition = S[0] / S[-1] if S[-1] > 0 else float('inf')
            top1_frac = (S[0] ** 2) / (S ** 2).sum()
            # Zipf-like decay: fit log(S) vs log(rank)
            ranks = np.arange(1, len(S) + 1)
            log_ranks = np.log(ranks)
            log_S = np.log(S + 1e-20)
            # Linear fit: log(S) = -α log(rank) + b
            if len(log_ranks) > 1:
                alpha = -np.polyfit(log_ranks, log_S, 1)[0]
            else:
                alpha = 0.0

            layer_condition.append(condition)
            layer_sv_decay.append(alpha)
            layer_top1_frac.append(top1_frac)

        all_sv_spectra.append(np.mean(layer_sv_decay))

        avg_condition = np.mean(layer_condition)
        avg_alpha = np.mean(layer_sv_decay)
        avg_top1 = np.mean(layer_top1_frac)

        mesh_results.append({
            'layer': layer_idx,
            'avg_condition_number': round(float(avg_condition), 1),
            'avg_zipf_alpha': round(float(avg_alpha), 4),
            'avg_top1_variance_frac': round(float(avg_top1), 4),
        })

        phi_indicator = " ← 1/φ!" if abs(avg_alpha - 0.618) < 0.1 else ""
        print(f"  Layer {layer_idx:2d}: "
              f"κ={avg_condition:8.1f}  "
              f"Zipf α={avg_alpha:.4f}{phi_indicator}  "
              f"top-1 var={avg_top1*100:.1f}%")

    # Compute savings
    total_mesh_original = NUM_HEADS * NUM_LAYERS * HIDDEN_DIM * HIDDEN_DIM
    total_mesh_factored = NUM_HEADS * NUM_LAYERS * (
        HIDDEN_DIM * HEAD_DIM + HEAD_DIM + HEAD_DIM * HIDDEN_DIM)
    compression = total_mesh_original / total_mesh_factored
    original_bytes = total_mesh_original * 3
    factored_bytes = total_mesh_factored * 3

    avg_alpha = np.mean(all_sv_spectra)
    print()
    print(f"  MESH Summary:")
    print(f"    Matrices:          {NUM_HEADS * NUM_LAYERS} MESH (28 heads × 28 layers)")
    print(f"    Each:              (3584, 3584) rank-128")
    print(f"    Avg Zipf decay α:  {avg_alpha:.4f}  "
          f"({'near 1/φ!' if abs(avg_alpha - 0.618) < 0.1 else 'not 1/φ'})")
    print(f"    Original MESH:     {original_bytes/1e9:.2f} GB")
    print(f"    Factored (Q×K):    {factored_bytes/1e9:.2f} GB")
    print(f"    Compression:       {compression:.1f}×")
    print(f"    Compute savings:   12.8M → 918K ops per head = 14×")
    print()

    return mesh_results, {
        'n_mesh_matrices': NUM_HEADS * NUM_LAYERS,
        'original_bytes': int(original_bytes),
        'factored_bytes': int(factored_bytes),
        'compression': round(compression, 1),
        'compute_speedup': '14x',
        'avg_zipf_alpha': round(float(avg_alpha), 4),
    }


# ============================================================================
# Part 3: Cross-Layer Structural Hashing
# ============================================================================

def analyze_cross_layer(layer_histograms):
    """Find shared φ-level patterns across layers (AIG structural hashing)."""
    print("=" * 80)
    print("  Part 3: CROSS-LAYER STRUCTURAL HASHING")
    print("  Which layers share the same φ-level patterns?")
    print("=" * 80)
    print()

    weight_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']

    cross_layer_results = {}

    for wname in weight_names:
        # Build a matrix of exponent distributions
        all_hists = []
        for layer_idx in range(NUM_LAYERS):
            key = f'layer_{layer_idx:02d}/{wname}'
            hist = layer_histograms.get(key, {})
            all_hists.append(hist)

        # Compute pairwise cosine similarity of histogram vectors
        # First, find the union of all exponent levels
        all_levels = set()
        for h in all_hists:
            all_levels.update(h.keys())
        all_levels = sorted(all_levels)
        level_idx = {lv: i for i, lv in enumerate(all_levels)}

        # Build histogram matrix (layers × levels)
        hist_matrix = np.zeros((NUM_LAYERS, len(all_levels)), dtype=np.float64)
        for layer_idx, h in enumerate(all_hists):
            for lv, count in h.items():
                hist_matrix[layer_idx, level_idx[lv]] = count

        # Normalize each row
        norms = np.linalg.norm(hist_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        hist_norm = hist_matrix / norms

        # Cosine similarity matrix
        sim_matrix = hist_norm @ hist_norm.T

        # Find clusters of similar layers (cosine > 0.99)
        threshold = 0.99
        n_above = (sim_matrix > threshold).sum() - NUM_LAYERS  # exclude diagonal
        n_pairs = NUM_LAYERS * (NUM_LAYERS - 1)
        similarity_fraction = n_above / n_pairs if n_pairs > 0 else 0

        # Average similarity
        triu_idx = np.triu_indices(NUM_LAYERS, k=1)
        avg_sim = float(sim_matrix[triu_idx].mean())
        min_sim = float(sim_matrix[triu_idx].min())
        max_sim = float(sim_matrix[triu_idx].max())

        cross_layer_results[wname] = {
            'avg_similarity': round(avg_sim, 4),
            'min_similarity': round(min_sim, 4),
            'max_similarity': round(max_sim, 4),
            'pairs_above_99': int(n_above // 2),
            'unique_levels': len(all_levels),
        }

        print(f"  {wname:12s}  "
              f"avg_sim={avg_sim:.4f}  "
              f"min={min_sim:.4f}  max={max_sim:.4f}  "
              f"pairs>0.99: {n_above//2}/{NUM_LAYERS*(NUM_LAYERS-1)//2}  "
              f"levels={len(all_levels)}")

    print()

    # Overall summary
    all_avg_sims = [v['avg_similarity'] for v in cross_layer_results.values()]
    overall_avg = np.mean(all_avg_sims)
    print(f"  Overall cross-layer similarity: {overall_avg:.4f}")
    if overall_avg > 0.95:
        print("  → HIGH redundancy: layers share very similar φ-level distributions")
        print("  → Potential for shared basis or delta encoding between layers")
    elif overall_avg > 0.85:
        print("  → MODERATE redundancy: some shared structure, some unique")
    else:
        print("  → LOW redundancy: each layer has distinct φ-level patterns")
    print()

    return cross_layer_results


# ============================================================================
# Part 4: Level-Grouped Format Analysis
# ============================================================================

def analyze_level_grouping():
    """Analyze how much compute savings level-grouping provides."""
    print("=" * 80)
    print("  Part 4: φ-LEVEL GROUPING ANALYSIS")
    print("  How many unique levels per row? (determines grouped matmul speedup)")
    print("=" * 80)
    print()

    weight_names = ['gate_proj', 'up_proj', 'down_proj']  # MLP only (biggest)
    grouping_results = {}

    for layer_idx in [0, 5, 14, 27]:  # Sample layers
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')

        for wname in weight_names:
            exps = load_exponents(os.path.join(layer_dir, f'{wname}.npz'))
            n_rows, n_cols = exps.shape

            # Count unique levels per row
            levels_per_row = np.array([len(np.unique(exps[r])) for r in range(n_rows)])
            avg_levels = float(levels_per_row.mean())
            min_levels = int(levels_per_row.min())
            max_levels = int(levels_per_row.max())

            # Global unique levels
            global_unique = len(np.unique(exps))

            # Speedup: n_cols / avg_levels_per_row
            speedup = n_cols / avg_levels

            key = f'layer_{layer_idx:02d}/{wname}'
            grouping_results[key] = {
                'shape': list(exps.shape),
                'global_unique_levels': int(global_unique),
                'avg_levels_per_row': round(avg_levels, 1),
                'min_levels_per_row': min_levels,
                'max_levels_per_row': max_levels,
                'speedup': round(speedup, 1),
            }

            print(f"  {key:30s}  "
                  f"global={global_unique:4d}  "
                  f"per_row={avg_levels:.0f} (min={min_levels}, max={max_levels})  "
                  f"speedup={speedup:.1f}×")

    print()
    avg_speedup = np.mean([v['speedup'] for v in grouping_results.values()])
    print(f"  Average MLP speedup from φ-level grouping: {avg_speedup:.1f}×")
    print(f"  (3584 float muls → ~{3584/avg_speedup:.0f} float muls per output dim)")
    print()

    return grouping_results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.perf_counter()

    print()
    print("╔" + "═" * 76 + "╗")
    print("║" + " Phase 1.5: MESH SIMPLIFICATION — AIG/IPA Pipeline for Qwen2-7B ".center(76) + "║")
    print("╚" + "═" * 76 + "╝")
    print()
    print(f"  Input:  {MODEL_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Part 1: Information Content
    info_results, layer_histograms, info_summary = analyze_information_content()

    # Part 2: MESH Analysis
    mesh_results, mesh_summary = analyze_mesh()

    # Part 3: Cross-Layer Structural Hashing
    cross_layer_results = analyze_cross_layer(layer_histograms)

    # Part 4: Level-Grouped Format
    grouping_results = analyze_level_grouping()

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.perf_counter() - t_start

    print("=" * 80)
    print("  SUMMARY: Phase 1.5 Mesh Simplification")
    print("=" * 80)
    print()

    # The IPA comparison
    print("  ┌─────────────────────┬──────────────────────────────────────┐")
    print("  │ IPA Model           │ Qwen2-7B                             │")
    print("  ├─────────────────────┼──────────────────────────────────────┤")
    tp = f"{info_summary['total_params']:,}"
    print(f"  │ 159 gate_steps      │ {tp} weights" + " " * (23 - len(tp)) + "│")
    enc = f"{info_summary['raw_bytes']/1e9:.2f} GB"
    print(f"  │ Encoding: 1,908 B   │ Encoding: {enc}" + " " * (27 - len(enc)) + "│")
    inf = f"{info_summary['info_gb']:.2f} GB"
    print(f"  │ Information: 283 B  │ Information: {inf}" + " " * (24 - len(inf)) + "│")
    red = f"{info_summary['redundancy_ratio']:.2f}×"
    print(f"  │ Redundancy: 6.7×    │ Redundancy: {red}" + " " * (25 - len(red)) + "│")
    bpw = f"{info_summary['avg_entropy_bits']:.3f}"
    print(f"  │ Bits/param: 1.5     │ Bits/param: {bpw}" + " " * (25 - len(bpw)) + "│")
    print("  └─────────────────────┴──────────────────────────────────────┘")
    print()

    print(f"  Information content: {info_summary['info_gb']:.2f} GB")
    print(f"    vs φ-encoded:     {info_summary['raw_bytes']/1e9:.2f} GB  "
          f"({info_summary['redundancy_ratio']:.2f}× redundant)")
    print(f"    vs float32:       {info_summary['float32_bytes']/1e9:.2f} GB  "
          f"({info_summary['float32_bytes']/info_summary['info_bytes']:.2f}× redundant)")
    print()

    print(f"  MESH low-rank: {mesh_summary['compression']}× compression, "
          f"{mesh_summary['compute_speedup']} compute speedup, "
          f"Zipf α={mesh_summary.get('avg_zipf_alpha', 'N/A')}")
    print()

    avg_speedup = np.mean([v['speedup'] for v in grouping_results.values()])
    print(f"  MLP level-grouping: {avg_speedup:.1f}× fewer float multiplies")
    print()

    cross_avg = np.mean([v['avg_similarity'] for v in cross_layer_results.values()])
    print(f"  Cross-layer similarity: {cross_avg:.4f} average")
    print()

    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print()

    # Save full report
    report = {
        'information_content': {
            'summary': info_summary,
            'per_component': info_results,
        },
        'mesh_analysis': {
            'summary': mesh_summary,
            'per_layer': mesh_results,
        },
        'cross_layer': cross_layer_results,
        'level_grouping': grouping_results,
    }

    report_path = os.path.join(OUTPUT_DIR, 'simplification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")
    print()


if __name__ == '__main__':
    main()
