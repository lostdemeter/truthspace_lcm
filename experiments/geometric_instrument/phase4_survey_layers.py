"""
Phase 4, Step 19: Survey All 28 Layers for Geometric Routing Properties
========================================================================

For each layer, extract the MESH SVD for every KV group and measure:
  - MESH rank-1 ratio: S[0]/S[1] (how close to rank-1?)
  - d_k frac_neg: fraction of d_k components that are negative
  - d_k norm: magnitude of the selection direction
  - Polarity: all-negative, all-positive, or mixed?

If MESH is rank-1 at every layer, geometric selectors can replace
softmax attention across the entire model.

Depends on: F128/F129 (Phase 3 extraction layer results)
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.phi_integer import phi_to_float

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def extract_mesh_info(attn, kv_group):
    """Extract MESH SVD info for one KV group of one layer.
    
    Returns dict with rank1_ratio, d_k_norm, frac_neg, d_k vector.
    """
    hd = attn.head_dim
    nh = attn.num_heads
    nkv = attn.num_kv_heads
    hpk = nh // nkv
    
    # Use first head in this KV group for Q (they share K)
    head_idx = kv_group * hpk
    
    W_q = phi_to_float(attn.W_q.signs, attn.W_q.exponents)
    W_q_h = W_q[head_idx * hd:(head_idx + 1) * hd, :]
    b_q_h = attn.b_q[head_idx * hd:(head_idx + 1) * hd]
    
    W_k = phi_to_float(attn.W_k.signs, attn.W_k.exponents)
    W_k_h = W_k[kv_group * hd:(kv_group + 1) * hd, :]
    b_k_h = attn.b_k[kv_group * hd:(kv_group + 1) * hd]
    
    # Bias-inclusive matrices
    W_q_b = W_q_h + b_q_h[:, None]
    W_k_b = W_k_h + b_k_h[:, None]
    
    # MESH = W_q_b @ W_k_b.T
    MESH = W_q_b @ W_k_b.T
    S = np.linalg.svd(MESH, compute_uv=False)
    
    rank1_ratio = float(S[0] / S[1]) if S[1] > 0 else float('inf')
    
    # Extract d_k via full SVD
    _, _, Vt = np.linalg.svd(MESH)
    v1 = Vt[0]
    d_k = W_k_b.T @ v1
    
    d_k_norm = float(np.linalg.norm(d_k))
    frac_neg = float(np.mean(d_k < 0))
    
    # Bias magnitude ratio
    b_q_norm = float(np.linalg.norm(b_q_h))
    b_k_norm = float(np.linalg.norm(b_k_h))
    wq_norm = float(np.linalg.norm(W_q_h))
    wk_norm = float(np.linalg.norm(W_k_h))
    bias_ratio_q = b_q_norm / wq_norm if wq_norm > 0 else 0
    bias_ratio_k = b_k_norm / wk_norm if wk_norm > 0 else 0
    
    return {
        'rank1_ratio': rank1_ratio,
        'd_k_norm': d_k_norm,
        'frac_neg': frac_neg,
        'bias_ratio_q': bias_ratio_q,
        'bias_ratio_k': bias_ratio_k,
        'S0': float(S[0]),
        'S1': float(S[1]),
        'd_k': d_k,
    }


def polarity_label(frac_neg):
    """Classify the polarity of d_k."""
    if frac_neg > 0.99:
        return "ALL-NEG"
    elif frac_neg < 0.01:
        return "ALL-POS"
    elif frac_neg > 0.9:
        return "mostly-neg"
    elif frac_neg < 0.1:
        return "mostly-pos"
    else:
        return "MIXED"


def main():
    print("=" * 80)
    print("  PHASE 4, STEP 19: Survey All 28 Layers for Geometric Routing")
    print("=" * 80)
    
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    nkv = engine.layers[0].attention.num_kv_heads  # 4
    
    print(f"  Layers: {n_layers}, KV groups per layer: {nkv}")
    print()
    
    # ── Survey header ──
    print("─" * 100)
    print(f"  {'Layer':>5}  {'KVG':>3}  {'S[0]/S[1]':>12}  {'||d_k||':>10}  "
          f"{'frac_neg':>8}  {'polarity':>10}  {'b/w Q':>6}  {'b/w K':>6}  {'S[0]':>12}")
    print("─" * 100)
    
    # Collect per-layer summaries
    all_results = []
    rank1_layers = 0     # layers where ALL KV groups are rank-1 (ratio > 100)
    pure_polar_layers = 0  # layers where ALL groups are all-neg or all-pos
    
    for li in range(n_layers):
        attn = engine.layers[li].attention
        layer_results = []
        layer_is_rank1 = True
        layer_is_polar = True
        
        for kv in range(nkv):
            info = extract_mesh_info(attn, kv)
            layer_results.append(info)
            
            pol = polarity_label(info['frac_neg'])
            
            if info['rank1_ratio'] < 100:
                layer_is_rank1 = False
            if pol not in ("ALL-NEG", "ALL-POS"):
                layer_is_polar = False
            
            r1_str = f"{info['rank1_ratio']:.1f}" if info['rank1_ratio'] < 1e12 else "∞"
            
            print(f"  {li:>5}  {kv:>3}  {r1_str:>12}  {info['d_k_norm']:>10.2f}  "
                  f"{info['frac_neg']:>8.3f}  {pol:>10}  "
                  f"{info['bias_ratio_q']:>6.2f}  {info['bias_ratio_k']:>6.2f}  "
                  f"{info['S0']:>12.1f}")
        
        all_results.append(layer_results)
        if layer_is_rank1:
            rank1_layers += 1
        if layer_is_polar:
            pure_polar_layers += 1
        
        # Print layer separator every 4 layers
        if (li + 1) % 4 == 0:
            print("  " + "· " * 50)
    
    print("─" * 100)
    
    # ── Summary statistics ──
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    # Collect all rank-1 ratios
    all_ratios = []
    all_frac_negs = []
    for li, lr in enumerate(all_results):
        for kv, info in enumerate(lr):
            all_ratios.append(info['rank1_ratio'])
            all_frac_negs.append(info['frac_neg'])
    
    all_ratios = np.array(all_ratios)
    all_frac_negs = np.array(all_frac_negs)
    
    print(f"\n  Total KV groups surveyed: {len(all_ratios)}")
    print(f"  Rank-1 ratio > 100:    {np.sum(all_ratios > 100)}/{len(all_ratios)} "
          f"({np.sum(all_ratios > 100)/len(all_ratios)*100:.1f}%)")
    print(f"  Rank-1 ratio > 1000:   {np.sum(all_ratios > 1000)}/{len(all_ratios)} "
          f"({np.sum(all_ratios > 1000)/len(all_ratios)*100:.1f}%)")
    print(f"  Rank-1 ratio > 10000:  {np.sum(all_ratios > 10000)}/{len(all_ratios)} "
          f"({np.sum(all_ratios > 10000)/len(all_ratios)*100:.1f}%)")
    
    print(f"\n  Pure polarity (ALL-NEG or ALL-POS): {np.sum((all_frac_negs > 0.99) | (all_frac_negs < 0.01))}/{len(all_frac_negs)}")
    print(f"  Mixed polarity (0.1 < frac < 0.9):  {np.sum((all_frac_negs > 0.1) & (all_frac_negs < 0.9))}/{len(all_frac_negs)}")
    
    print(f"\n  Layers where ALL 4 KV groups are rank-1 (>100): {rank1_layers}/{n_layers}")
    print(f"  Layers where ALL 4 KV groups are pure polar:    {pure_polar_layers}/{n_layers}")
    
    # ── Per-layer summary ──
    print("\n  Per-layer classification:")
    print(f"  {'Layer':>5}  {'min(S[0]/S[1])':>14}  {'Polarities':>30}  {'Classification':>15}")
    print("  " + "─" * 70)
    
    for li, lr in enumerate(all_results):
        min_ratio = min(info['rank1_ratio'] for info in lr)
        pols = [polarity_label(info['frac_neg']) for info in lr]
        pol_str = ", ".join(pols)
        
        if min_ratio > 1000 and all(p in ("ALL-NEG", "ALL-POS") for p in pols):
            classification = "✓ GEOMETRIC"
        elif min_ratio > 100:
            classification = "~ near-geo"
        elif min_ratio > 10:
            classification = "? weak rank-1"
        else:
            classification = "✗ NOT rank-1"
        
        r1_str = f"{min_ratio:.1f}" if min_ratio < 1e12 else "∞"
        print(f"  {li:>5}  {r1_str:>14}  {pol_str:>30}  {classification:>15}")
    
    print()
    
    # ── Histogram of rank-1 ratios ──
    print("  Rank-1 ratio distribution:")
    bins = [0, 1, 10, 100, 1000, 10000, 100000, float('inf')]
    labels = ["<1", "1-10", "10-100", "100-1K", "1K-10K", "10K-100K", ">100K"]
    for i in range(len(bins)-1):
        count = np.sum((all_ratios >= bins[i]) & (all_ratios < bins[i+1]))
        bar = "█" * count
        print(f"    {labels[i]:>10}: {count:>3}  {bar}")
    
    print()


if __name__ == '__main__':
    main()
