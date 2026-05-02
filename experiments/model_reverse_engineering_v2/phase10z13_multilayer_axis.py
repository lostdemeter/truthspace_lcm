"""
Phase 10z13: Multi-Layer Axis Analysis
=======================================

F114 showed ALL routing heads in Layer 23 share ONE d_k direction.
Is this universal across all 28 layers, or specific to Layer 23?

For each layer:
  1. Classify heads (fixed vs routing) using cached hidden states
  2. Extract MESH SVD for all heads → rank-1 dominance
  3. Compute d_k directions for routing heads
  4. SVD of the d_k matrix → effective dimensionality
  5. Angle structure between d_k vectors

If the one-axis pattern is universal:
  → The Euler product structure holds everywhere
  → Each layer is one "frequency band" of the ζ sum
  → The transformer IS a discretized Riemann-Siegel sum

If it varies by layer:
  → Early layers may be multi-axis (broader selection)
  → Later layers may narrow to one axis (specific retrieval)
  → The ζ structure emerges through depth
"""

import sys
import os
import numpy as np
import time
import gc
import json

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# Calibration prompts for head classification
CAL_PROMPTS = [
    'The capital of France is',
    'The capital of Japan is',
    'The capital of Germany is',
    'The largest ocean is the',
    'Water freezes at zero degrees',
    'The speed of light is approximately',
    'The chemical symbol for gold is',
    'Barack Obama was the',
    'Albert Einstein developed the theory of',
    'The color of grass is',
    'To be or not to',
    'Roses are red, violets are',
]


def cache_all_hidden_states(engine, tokenizer, prompts):
    """Run each prompt through ALL layers, caching hidden states at each layer boundary.

    Returns: dict[prompt] → list of hidden states (one per layer boundary).
    h_cache[prompt][i] = hidden state BEFORE layer i's attention.
    """
    n_layers = len(engine.layers)
    h_cache = {}

    for prompt in prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]

        states = [h.copy()]  # state before layer 0
        for layer in engine.layers:
            h = layer(h)
            states.append(h.copy())  # state before layer i+1

        h_cache[prompt] = states

    return h_cache


def classify_heads_cached(engine, layer_idx, h_cache, prompts):
    """Classify heads as fixed/routing using cached hidden states."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads

    pos0_counts = np.zeros(num_heads)
    total = 0

    for prompt in prompts:
        h = h_cache[prompt][layer_idx]  # hidden state before this layer

        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)

        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_exp = np.repeat(K, heads_per_kv, axis=1)

        for hi in range(num_heads):
            scores = Q[0, hi, -1, :] @ K_exp[0, hi, :, :].T
            if np.argmax(scores) == 0:
                pos0_counts[hi] += 1
        total += 1

    fixed = [h for h in range(num_heads) if pos0_counts[h] == total]
    routing = [h for h in range(num_heads) if pos0_counts[h] < total]
    return fixed, routing


def extract_mesh_svd(engine, layer_idx):
    """Extract MESH SVD for all heads in a layer. Returns rotation params."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    hidden_dim = engine.hidden_dim

    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512

    W_q_all = np.zeros((num_heads, head_dim, hidden_dim), dtype=np.float32)
    W_k_all = np.zeros((num_kv_heads, head_dim, hidden_dim), dtype=np.float32)

    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]

        q_out = phi_linear(attn.W_q, chunk, attn.b_q)
        k_out = phi_linear(attn.W_k, chunk, attn.b_k)

        q_reshaped = q_out[0].reshape(-1, num_heads, head_dim)
        k_reshaped = k_out[0].reshape(-1, num_kv_heads, head_dim)

        for h in range(num_heads):
            W_q_all[h, :, start:end] = q_reshaped[:, h, :].T
        for g in range(num_kv_heads):
            W_k_all[g, :, start:end] = k_reshaped[:, g, :].T

    # Compute MESH SVD for each head
    rotations = []
    for h in range(num_heads):
        kv_group = h // heads_per_kv
        MESH = W_q_all[h] @ W_k_all[kv_group].T
        U, S, Vt = np.linalg.svd(MESH)

        u1 = U[:, 0]
        v1 = Vt[0, :]

        d_q = W_q_all[h].T @ u1
        d_k = W_k_all[kv_group].T @ v1

        cos_dq_dk = float(np.dot(d_q, d_k) / (
            np.linalg.norm(d_q) * np.linalg.norm(d_k) + 1e-20))

        rotations.append({
            'head': h,
            'kv_group': kv_group,
            'S0': float(S[0]),
            'S1': float(S[1]),
            'sv_ratio': float(S[0] / S[1]) if S[1] > 1e-10 else float('inf'),
            'rank1_var': float(S[0]**2 / np.sum(S**2) * 100),
            'd_k': d_k,
            'cos_dq_dk': cos_dq_dk,
        })

    return rotations


def analyze_dk_structure(rotations, routing_heads):
    """Analyze the d_k vector structure for routing heads.

    Returns:
      - dk_rank: effective rank of the d_k matrix
      - dk_svd: singular values
      - angle_stats: mean, median, min, max of pairwise angles
      - n_clusters: number of angular clusters (0°, 180° = 2 clusters = 1 axis)
    """
    if len(routing_heads) < 2:
        return {
            'dk_rank_90': 0, 'dk_rank_95': 0, 'dk_rank_99': 0,
            'dk_svd': [], 'angle_mean': 0, 'angle_median': 0,
            'n_clusters': 0,
        }

    routing_rots = [rotations[h] for h in routing_heads]
    dk_vecs = np.array([r['d_k'] for r in routing_rots])

    # Normalize
    dk_norms = np.linalg.norm(dk_vecs, axis=1, keepdims=True)
    dk_unit = dk_vecs / (dk_norms + 1e-20)

    # SVD of d_k matrix
    U_dk, S_dk, Vt_dk = np.linalg.svd(dk_vecs, full_matrices=False)
    total_var = np.sum(S_dk**2)

    ranks = {}
    for threshold in [90, 95, 99]:
        cumvar = np.cumsum(S_dk**2) / total_var * 100
        ranks[threshold] = int(np.searchsorted(cumvar, threshold)) + 1

    # Pairwise angles
    cos_matrix = dk_unit @ dk_unit.T
    angles = []
    n = len(routing_rots)
    for i in range(n):
        for j in range(i+1, n):
            angle = np.arccos(np.clip(cos_matrix[i, j], -1, 1)) * 180 / np.pi
            angles.append(angle)
    angles = np.array(angles)

    # Cluster detection: how many distinct angle values?
    # If all angles are near 0° or 180°, it's 1 axis (2 poles)
    # If angles are near 90°, independent axes
    near_0 = np.sum(angles < 10)
    near_180 = np.sum(angles > 170)
    near_90 = np.sum((angles > 80) & (angles < 100))
    other = len(angles) - near_0 - near_180 - near_90

    # Determine pattern
    if near_0 + near_180 > 0.8 * len(angles):
        pattern = "ONE_AXIS"
        n_axes = 1
    elif near_90 > 0.5 * len(angles):
        pattern = "ORTHOGONAL"
        n_axes = ranks[90]
    else:
        pattern = "MIXED"
        n_axes = ranks[90]

    return {
        'dk_rank_90': ranks[90],
        'dk_rank_95': ranks[95],
        'dk_rank_99': ranks[99],
        'dk_svd': S_dk.tolist(),
        'angle_mean': float(np.mean(angles)) if len(angles) > 0 else 0,
        'angle_median': float(np.median(angles)) if len(angles) > 0 else 0,
        'angle_min': float(np.min(angles)) if len(angles) > 0 else 0,
        'angle_max': float(np.max(angles)) if len(angles) > 0 else 0,
        'near_0': int(near_0),
        'near_180': int(near_180),
        'near_90': int(near_90),
        'other': int(other),
        'pattern': pattern,
        'n_axes': n_axes,
    }


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z13: MULTI-LAYER AXIS ANALYSIS")
    print("=" * 72)
    print(f"  Layers: 0-{n_layers-1}")
    print(f"  Question: Is the one-axis d_k pattern universal?")
    print(f"  Calibration prompts: {len(CAL_PROMPTS)}")

    # ── Step 1: Cache hidden states ──
    print("\n" + "─" * 72)
    print("  Step 1: Cache hidden states for all prompts × all layers")
    print("─" * 72)
    t1 = time.time()
    h_cache = cache_all_hidden_states(engine, tokenizer, CAL_PROMPTS)
    print(f"    Cached {len(CAL_PROMPTS)} prompts × {n_layers+1} states in {time.time()-t1:.1f}s")

    # ── Step 2: Analyze each layer ──
    print("\n" + "─" * 72)
    print("  Step 2: Per-layer analysis")
    print("─" * 72)

    layer_results = []

    for layer_idx in range(n_layers):
        t_layer = time.time()

        # Classify heads
        fixed, routing = classify_heads_cached(engine, layer_idx, h_cache, CAL_PROMPTS)

        # Extract MESH SVD
        rotations = extract_mesh_svd(engine, layer_idx)

        # Analyze d_k structure
        dk_analysis = analyze_dk_structure(rotations, routing)

        # Aggregate MESH rank-1 stats for routing heads
        routing_rots = [rotations[h] for h in routing]
        if routing_rots:
            min_sv_ratio = min(r['sv_ratio'] for r in routing_rots)
            mean_sv_ratio = np.mean([r['sv_ratio'] for r in routing_rots])
            mean_cos = np.mean([r['cos_dq_dk'] for r in routing_rots])
            all_rank1 = all(r['rank1_var'] > 99.0 for r in routing_rots)
        else:
            min_sv_ratio = 0
            mean_sv_ratio = 0
            mean_cos = 0
            all_rank1 = True

        dt = time.time() - t_layer

        result = {
            'layer': layer_idx,
            'n_fixed': len(fixed),
            'n_routing': len(routing),
            'fixed_heads': fixed,
            'routing_heads': routing,
            'min_sv_ratio': float(min_sv_ratio),
            'mean_sv_ratio': float(mean_sv_ratio),
            'mean_cos_dq_dk': float(mean_cos),
            'all_rank1': all_rank1,
            'dk_analysis': dk_analysis,
            'time': dt,
        }
        layer_results.append(result)

        # Print summary line
        pattern = dk_analysis['pattern']
        r90 = dk_analysis['dk_rank_90']
        ang_mean = dk_analysis['angle_mean']
        svd_top = dk_analysis['dk_svd'][:3] if dk_analysis['dk_svd'] else []
        svd_str = ", ".join(f"{s:.1f}" for s in svd_top)

        marker = "★" if pattern == "ONE_AXIS" else "◆" if pattern == "ORTHOGONAL" else "○"

        print(f"    L{layer_idx:2d}: {marker} {len(fixed):2d}F {len(routing):2d}R  "
              f"rank90={r90:2d}  ∠mean={ang_mean:5.1f}°  "
              f"pattern={pattern:<11s}  "
              f"cos(dq,dk)={mean_cos:+.4f}  "
              f"σ=[{svd_str}]  "
              f"({dt:.1f}s)")

    # Free cached states
    del h_cache
    gc.collect()

    # ── Step 3: Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY: Axis Pattern Across All Layers")
    print("=" * 72)

    one_axis_layers = [r for r in layer_results if r['dk_analysis']['pattern'] == 'ONE_AXIS']
    ortho_layers = [r for r in layer_results if r['dk_analysis']['pattern'] == 'ORTHOGONAL']
    mixed_layers = [r for r in layer_results if r['dk_analysis']['pattern'] == 'MIXED']

    print(f"\n  ONE_AXIS (all d_k parallel/antiparallel): {len(one_axis_layers)} layers")
    if one_axis_layers:
        print(f"    Layers: {[r['layer'] for r in one_axis_layers]}")

    print(f"  ORTHOGONAL (d_k near 90°): {len(ortho_layers)} layers")
    if ortho_layers:
        print(f"    Layers: {[r['layer'] for r in ortho_layers]}")

    print(f"  MIXED: {len(mixed_layers)} layers")
    if mixed_layers:
        print(f"    Layers: {[r['layer'] for r in mixed_layers]}")

    # Rank evolution
    print(f"\n  d_k rank (90% variance) by layer:")
    ranks = [r['dk_analysis']['dk_rank_90'] for r in layer_results]
    for i, r in enumerate(layer_results):
        bar = "█" * r['dk_analysis']['dk_rank_90']
        print(f"    L{i:2d}: {r['dk_analysis']['dk_rank_90']:2d} {bar}")

    # cos(d_q, d_k) evolution
    print(f"\n  cos(d_q, d_k) by layer:")
    for r in layer_results:
        cos = r['mean_cos_dq_dk']
        if cos > 0.99:
            bar = "████████████████████ +1.0"
        elif cos > 0.5:
            n = int(cos * 20)
            bar = "█" * n + f" {cos:+.3f}"
        elif cos > -0.5:
            bar = f"  ~0 ({cos:+.3f})"
        else:
            n = int(-cos * 20)
            bar = "▓" * n + f" {cos:+.3f}"
        print(f"    L{r['layer']:2d}: {bar}")

    # Routing head count evolution
    print(f"\n  Fixed vs Routing heads by layer:")
    for r in layer_results:
        nf = r['n_fixed']
        nr = r['n_routing']
        bar_f = "F" * nf
        bar_r = "R" * nr
        print(f"    L{r['layer']:2d}: {bar_f}{bar_r}  ({nf}F/{nr}R)")

    # MESH rank-1 universality
    all_rank1_layers = [r['layer'] for r in layer_results if r['all_rank1']]
    print(f"\n  ALL routing heads rank-1 (>99% variance): "
          f"{len(all_rank1_layers)}/{n_layers} layers")
    if len(all_rank1_layers) < n_layers:
        not_rank1 = [r['layer'] for r in layer_results if not r['all_rank1']]
        print(f"    Not all rank-1: layers {not_rank1}")

    # ── Save ──
    print("\n" + "=" * 72)
    print("  SAVING RESULTS")
    print("=" * 72)

    save_data = {
        'n_layers': n_layers,
        'cal_prompts': CAL_PROMPTS,
        'layers': [{
            'layer': r['layer'],
            'n_fixed': r['n_fixed'],
            'n_routing': r['n_routing'],
            'fixed_heads': r['fixed_heads'],
            'routing_heads': r['routing_heads'],
            'min_sv_ratio': r['min_sv_ratio'],
            'mean_sv_ratio': r['mean_sv_ratio'],
            'mean_cos_dq_dk': r['mean_cos_dq_dk'],
            'all_rank1': r['all_rank1'],
            'dk_rank_90': r['dk_analysis']['dk_rank_90'],
            'dk_rank_95': r['dk_analysis']['dk_rank_95'],
            'dk_rank_99': r['dk_analysis']['dk_rank_99'],
            'dk_svd_top5': r['dk_analysis']['dk_svd'][:5],
            'angle_mean': r['dk_analysis']['angle_mean'],
            'angle_median': r['dk_analysis']['angle_median'],
            'pattern': r['dk_analysis']['pattern'],
            'n_axes': r['dk_analysis']['n_axes'],
            'near_0': r['dk_analysis']['near_0'],
            'near_180': r['dk_analysis']['near_180'],
            'near_90': r['dk_analysis']['near_90'],
        } for r in layer_results],
    }

    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z13_multilayer_axis.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Conclusion ──
    print("\n" + "=" * 72)
    print("  CONCLUSION")
    print("=" * 72)

    if len(one_axis_layers) == n_layers:
        print("  ★ ONE-AXIS PATTERN IS UNIVERSAL")
        print("  Every layer has all routing d_k vectors on ONE axis.")
        print("  The Euler product structure holds EVERYWHERE.")
    elif len(one_axis_layers) > n_layers * 0.7:
        print(f"  ◆ ONE-AXIS PATTERN IS DOMINANT ({len(one_axis_layers)}/{n_layers})")
        print("  Most layers show one-axis structure.")
        print("  Exceptions may be transition/mixing layers.")
    else:
        print(f"  ○ MIXED PATTERNS ({len(one_axis_layers)} one-axis, "
              f"{len(ortho_layers)} orthogonal, {len(mixed_layers)} mixed)")
        print("  The one-axis pattern is NOT universal.")
        print("  Different layers use different geometric strategies.")


if __name__ == '__main__':
    main()
