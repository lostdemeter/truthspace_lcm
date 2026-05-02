"""
Phase 10z13b: Cross-Layer Axis Alignment
=========================================

F116 showed every layer has ONE d_k axis. But are the 28 axes:
  (a) The SAME direction? → one global axis, like θ(t) in ζ
  (b) DIFFERENT directions? → each layer has its own axis, like
      each prime in the Euler product

Extract the dominant d_k direction for each layer, compute the
28×28 angle matrix, and determine the cross-layer structure.
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
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def extract_dominant_dk(engine, layer_idx):
    """Extract the dominant d_k axis for a layer.

    Returns the first right singular vector of the routing heads' d_k matrix,
    which captures the shared direction.
    """
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

    # Compute d_k for all heads
    dk_vecs = []
    for h in range(num_heads):
        kv_group = h // heads_per_kv
        MESH = W_q_all[h] @ W_k_all[kv_group].T
        _, S, Vt = np.linalg.svd(MESH)
        v1 = Vt[0, :]
        d_k = W_k_all[kv_group].T @ v1
        dk_vecs.append(d_k)

    dk_matrix = np.array(dk_vecs)

    # SVD of d_k matrix to get the dominant axis
    U, S, Vt = np.linalg.svd(dk_matrix, full_matrices=False)
    dominant_axis = Vt[0, :]  # first right singular vector (hidden_dim,)
    dominant_axis = dominant_axis / np.linalg.norm(dominant_axis)

    return dominant_axis, S


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    n_layers = len(engine.layers)
    hidden_dim = engine.hidden_dim
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z13b: CROSS-LAYER AXIS ALIGNMENT")
    print("=" * 72)
    print(f"  Question: Are the 28 per-layer d_k axes the SAME or DIFFERENT?")

    # Extract dominant d_k axis for each layer
    print("\n" + "─" * 72)
    print("  Extracting dominant d_k axis per layer...")
    print("─" * 72)

    axes = []
    svd_data = []
    for i in range(n_layers):
        t1 = time.time()
        axis, S = extract_dominant_dk(engine, i)
        axes.append(axis)
        svd_data.append(S)
        ratio = S[0] / S[1] if S[1] > 1e-10 else float('inf')
        print(f"    L{i:2d}: σ[0]={S[0]:10.1f}  σ[1]={S[1]:6.2f}  "
              f"ratio={ratio:10.0f}  ({time.time()-t1:.1f}s)")

    axes = np.array(axes)  # (28, 3584)

    # ── Cross-layer angle matrix ──
    print("\n" + "─" * 72)
    print("  Cross-layer angle matrix (degrees)")
    print("─" * 72)

    cos_matrix = axes @ axes.T
    angle_matrix = np.arccos(np.clip(np.abs(cos_matrix), 0, 1)) * 180 / np.pi
    # Use abs(cos) because parallel and antiparallel are both "same axis"

    # Print compact angle matrix
    header = "      " + "".join(f"  L{i:2d}" for i in range(n_layers))
    print(header)
    for i in range(n_layers):
        row = f"  L{i:2d}:"
        for j in range(n_layers):
            if j <= i:
                row += f" {angle_matrix[i,j]:4.1f}"
            else:
                row += "     "
        print(row)

    # ── Statistics ──
    print("\n" + "─" * 72)
    print("  Cross-layer angle statistics")
    print("─" * 72)

    # Collect unique pairs (upper triangle)
    unique_angles = []
    for i in range(n_layers):
        for j in range(i+1, n_layers):
            unique_angles.append(angle_matrix[i, j])
    unique_angles = np.array(unique_angles)

    print(f"    Pairs: {len(unique_angles)}")
    print(f"    Mean:   {np.mean(unique_angles):.2f}°")
    print(f"    Median: {np.median(unique_angles):.2f}°")
    print(f"    Min:    {np.min(unique_angles):.2f}°")
    print(f"    Max:    {np.max(unique_angles):.2f}°")
    print(f"    Std:    {np.std(unique_angles):.2f}°")

    near_0 = np.sum(unique_angles < 10)
    near_45 = np.sum((unique_angles > 35) & (unique_angles < 55))
    near_90 = np.sum(unique_angles > 80)
    print(f"    Near 0° (same axis):    {near_0}/{len(unique_angles)}")
    print(f"    Near 45° (intermediate): {near_45}/{len(unique_angles)}")
    print(f"    Near 90° (orthogonal):  {near_90}/{len(unique_angles)}")

    # ── SVD of the 28 layer axes ──
    print("\n" + "─" * 72)
    print(f"  SVD of layer axes (28 vectors × {hidden_dim} dims)")
    print("─" * 72)

    U_all, S_all, Vt_all = np.linalg.svd(axes, full_matrices=False)
    total_var = np.sum(S_all**2)
    print(f"    Top singular values:")
    for i in range(min(28, 15)):
        cumvar = np.sum(S_all[:i+1]**2) / total_var * 100
        phi_level = np.log(S_all[i]) / LOG_PHI if S_all[i] > 1e-10 else float('-inf')
        print(f"      σ[{i:2d}] = {S_all[i]:10.4f}  cumvar={cumvar:5.1f}%  "
              f"φ-level={phi_level:+.2f}")

    for threshold in [50, 80, 90, 95, 99]:
        cumvar = np.cumsum(S_all**2) / total_var * 100
        rank = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"    Rank for {threshold}% variance: {rank}")

    # ── Layer grouping by axis similarity ──
    print("\n" + "─" * 72)
    print("  Layer grouping by axis similarity (hierarchical)")
    print("─" * 72)

    # Find pairs of layers with most similar axes
    closest = []
    for i in range(n_layers):
        for j in range(i+1, n_layers):
            closest.append((angle_matrix[i,j], i, j))
    closest.sort()

    print(f"    Most aligned layer pairs:")
    for angle, i, j in closest[:10]:
        print(f"      L{i:2d} ↔ L{j:2d}: {angle:.2f}°")

    print(f"\n    Most orthogonal layer pairs:")
    for angle, i, j in closest[-5:]:
        print(f"      L{i:2d} ↔ L{j:2d}: {angle:.2f}°")

    # ── Determine overall pattern ──
    print("\n" + "=" * 72)
    print("  CONCLUSION")
    print("=" * 72)

    if np.mean(unique_angles) < 20:
        print("  ★ ALL LAYERS SHARE ONE GLOBAL AXIS")
        print("  The entire transformer has a single d_k direction.")
        print("  This is θ(t) — the same base phase for all terms.")
    elif np.mean(unique_angles) < 45:
        print("  ◆ LAYERS CLUSTER IN A FEW GROUPS")
        print(f"  Mean angle = {np.mean(unique_angles):.1f}°")
        print("  Layers form distinct but related axis groups.")
    elif np.mean(unique_angles) > 70:
        print("  ○ LAYERS HAVE INDEPENDENT AXES")
        print(f"  Mean angle = {np.mean(unique_angles):.1f}°")
        print("  Each layer selects along its OWN direction.")
        print("  Like the Euler product: each prime has its own factor.")
    else:
        svd_rank_90 = int(np.searchsorted(np.cumsum(S_all**2)/total_var*100, 90)) + 1
        print(f"  ◇ INTERMEDIATE: {svd_rank_90} effective axes across 28 layers")
        print(f"  Mean angle = {np.mean(unique_angles):.1f}°")
        print("  The axes span a subspace, neither fully shared nor independent.")

    # ── Save ──
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z13b_crosslayer_axes.json')
    save_data = {
        'n_layers': n_layers,
        'angle_matrix': angle_matrix.tolist(),
        'angle_stats': {
            'mean': float(np.mean(unique_angles)),
            'median': float(np.median(unique_angles)),
            'min': float(np.min(unique_angles)),
            'max': float(np.max(unique_angles)),
            'std': float(np.std(unique_angles)),
        },
        'axes_svd': S_all.tolist(),
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
