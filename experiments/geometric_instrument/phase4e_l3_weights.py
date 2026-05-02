"""
Phase 4e (part 2): L3 Weight Geometry & Explosion Direction
============================================================
Investigations 5-7 from the L3 deep-dive (weights are PhiEncoded).
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

VARIED_PROMPTS = [
    'The capital of France is',
    'I know the capital of France is',
    'Can you tell me the capital of France is',
    'Hello world',
    'The quick brown fox jumps over',
]


def decode_weight(w):
    """Get numpy array from weight (handles PhiEncoded or ndarray)."""
    if isinstance(w, PhiEncoded):
        return w.decode()
    return w


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decompose_layer(engine, h, layer_idx):
    """Run a layer and return intermediate states."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    
    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q)
    K = phi_linear(attn.W_k, normed, attn.b_k)
    V = phi_linear(attn.W_v, normed, attn.b_v)
    Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    Ve = np.repeat(V, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if seq_len > 1:
        scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    weights = phi_softmax(scores, axis=-1)
    
    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    attn_out = phi_linear(attn.W_o, ao)
    h_post_attn = h + attn_out
    
    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    h_post_mlp = h_post_attn + mlp_out
    
    return {
        'h_in': h,
        'attn_out': attn_out,
        'h_post_attn': h_post_attn,
        'mlp_out': mlp_out,
        'h_post_mlp': h_post_mlp,
        'normed_mlp': nm,
        'gate_pre': g,
        'up_pre': u,
    }


def main():
    print("=" * 80)
    print("  Phase 4e (part 2): L3 Weight Geometry & Explosion Direction")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Weight norms across layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 5: Weight Frobenius Norms")
    print("=" * 80)
    
    print(f"\n  MLP weight norms:")
    print(f"  {'Layer':>5}  {'||W_gate||':>12}  {'||W_up||':>12}  {'||W_down||':>12}  "
          f"{'||norm_w||':>10}")
    print("  " + "─" * 60)
    
    for li in [0, 1, 2, 3, 4, 5, 10, 20, 26, 27]:
        mlp = engine.layers[li].mlp
        norms = []
        for w in [mlp.W_gate, mlp.W_up, mlp.W_down]:
            arr = decode_weight(w)
            norms.append(float(np.linalg.norm(arr)))
        nw = float(np.linalg.norm(decode_weight(mlp.norm_weight)))
        marker = " ←" if li == 3 else " ←←" if li == 26 else ""
        print(f"  L{li:>3}  " + "  ".join(f"{n:>12.2f}" for n in norms) +
              f"  {nw:>10.4f}{marker}")
    
    print(f"\n  Attention W_o norms:")
    print(f"  {'Layer':>5}  {'||W_o||':>12}")
    print("  " + "─" * 20)
    
    for li in [0, 1, 2, 3, 4, 5, 10, 20, 27]:
        arr = decode_weight(engine.layers[li].attention.W_o)
        n = float(np.linalg.norm(arr))
        marker = " ←" if li == 3 else ""
        print(f"  L{li:>3}  {n:>12.2f}{marker}")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 5b: Singular values of L3 MLP weights
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 5b: Singular Values of MLP Weights")
    print("=" * 80)
    
    for li in [2, 3, 4, 26]:
        mlp = engine.layers[li].mlp
        label = f"L{li}"
        marker = " ← EXPLOSIVE" if li == 3 else " ← COLLAPSE" if li == 26 else ""
        
        W_down = decode_weight(mlp.W_down)
        S_down = np.linalg.svd(W_down, compute_uv=False)
        
        W_gate = decode_weight(mlp.W_gate)
        S_gate = np.linalg.svd(W_gate, compute_uv=False)
        
        print(f"\n  {label} W_down: S[0]={S_down[0]:.4f}, S[0]/S[1]={S_down[0]/S_down[1]:.2f}, "
              f"S[:5]=[{', '.join(f'{s:.3f}' for s in S_down[:5])}]{marker}")
        print(f"  {label} W_gate: S[0]={S_gate[0]:.4f}, S[0]/S[1]={S_gate[0]/S_gate[1]:.2f}, "
              f"S[:5]=[{', '.join(f'{s:.3f}' for s in S_gate[:5])}]")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 6: L3 MLP gating analysis at BOS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 6: L3 MLP Gating at BOS")
    print("=" * 80)
    
    france_tids = tokenizer.encode('The capital of France is')
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    for li in range(3):
        h = engine.layers[li](h)
    
    dd = decompose_layer(engine, h, 3)
    
    # Gate and up values at BOS
    gate_bos = dd['gate_pre'][0, 0, :]  # [d_ff]
    up_bos = dd['up_pre'][0, 0, :]      # [d_ff]
    silu_gate = phi_silu(gate_bos.reshape(1, 1, -1))[0, 0, :]
    product = silu_gate * up_bos
    
    print(f"\n  L3 MLP at BOS (position 0):")
    print(f"    ||gate||      = {float(np.linalg.norm(gate_bos)):.2f}")
    print(f"    ||up||        = {float(np.linalg.norm(up_bos)):.2f}")
    print(f"    ||silu(gate)||= {float(np.linalg.norm(silu_gate)):.2f}")
    print(f"    ||silu*up||   = {float(np.linalg.norm(product)):.2f}")
    print(f"    ||mlp_out||   = {float(np.linalg.norm(dd['mlp_out'][0, 0, :])):.2f}")
    
    # How many neurons are active?
    gate_activated = np.abs(silu_gate) > 0.01
    n_active = int(gate_activated.sum())
    n_total = len(gate_bos)
    print(f"\n    Active neurons: {n_active}/{n_total} ({100*n_active/n_total:.1f}%)")
    
    # Top activated neurons
    abs_product = np.abs(product)
    top_idx = np.argsort(-abs_product)[:20]
    print(f"    Top 20 |silu(gate)*up| values:")
    for i, idx in enumerate(top_idx):
        print(f"      [{idx:>5}]: gate={gate_bos[idx]:>8.3f}, up={up_bos[idx]:>8.3f}, "
              f"silu*up={product[idx]:>8.3f}")
    
    # Compare with non-BOS position
    gate_last = dd['gate_pre'][0, -1, :]
    up_last = dd['up_pre'][0, -1, :]
    silu_last = phi_silu(gate_last.reshape(1, 1, -1))[0, 0, :]
    product_last = silu_last * up_last
    
    print(f"\n  L3 MLP at LAST position (p4) for comparison:")
    print(f"    ||gate||      = {float(np.linalg.norm(gate_last)):.2f}")
    print(f"    ||up||        = {float(np.linalg.norm(up_last)):.2f}")
    print(f"    ||silu*up||   = {float(np.linalg.norm(product_last)):.2f}")
    print(f"    ||mlp_out||   = {float(np.linalg.norm(dd['mlp_out'][0, -1, :])):.2f}")
    
    # Ratio
    bos_product_norm = float(np.linalg.norm(product))
    last_product_norm = float(np.linalg.norm(product_last))
    print(f"\n    BOS/last ratio: {bos_product_norm/last_product_norm:.1f}x")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 7: Explosion direction analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 7: Explosion Direction")
    print("=" * 80)
    
    h_before = dd['h_in'][0, 0, :]
    h_after = dd['h_post_mlp'][0, 0, :]
    mlp_out_bos = dd['mlp_out'][0, 0, :]
    delta = h_after - h_before
    
    print(f"\n  BOS state change at L3:")
    print(f"    ||h_before||    = {float(np.linalg.norm(h_before)):.1f}")
    print(f"    ||h_after||     = {float(np.linalg.norm(h_after)):.1f}")
    print(f"    ||delta||       = {float(np.linalg.norm(delta)):.1f}")
    print(f"    ||mlp_out||     = {float(np.linalg.norm(mlp_out_bos)):.1f}")
    print(f"    cos(before, after) = {cos_sim(h_before, h_after):.4f}")
    print(f"    cos(before, mlp)   = {cos_sim(h_before, mlp_out_bos):.4f}")
    print(f"    cos(delta, mlp)    = {cos_sim(delta, mlp_out_bos):.4f}")
    
    # Check alignment with W_down's principal direction
    W_down = decode_weight(engine.layers[3].mlp.W_down)
    U_d, S_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
    
    delta_n = delta / np.linalg.norm(delta)
    mlp_n = mlp_out_bos / np.linalg.norm(mlp_out_bos)
    
    print(f"\n  Alignment with W_down singular vectors:")
    for i in range(10):
        cos_delta = abs(cos_sim(delta_n, U_d[:, i]))
        cos_mlp = abs(cos_sim(mlp_n, U_d[:, i]))
        print(f"    SV{i}: |cos(delta, U[:,{i}])|={cos_delta:.4f}, "
              f"|cos(mlp_out, U[:,{i}])|={cos_mlp:.4f}, S[{i}]={S_d[i]:.4f}")
    
    # Is the explosion direction universal across prompts?
    print(f"\n  Explosion direction stability across prompts:")
    explosion_dirs = []
    mlp_dirs = []
    
    for prompt in VARIED_PROMPTS:
        tids = tokenizer.encode(prompt)
        h_p = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(3):
            h_p = engine.layers[li](h_p)
        dd_p = decompose_layer(engine, h_p, 3)
        
        mlp_p = dd_p['mlp_out'][0, 0, :]
        mlp_dirs.append(mlp_p / np.linalg.norm(mlp_p))
        
        delta_p = dd_p['h_post_mlp'][0, 0, :] - dd_p['h_in'][0, 0, :]
        explosion_dirs.append(delta_p / np.linalg.norm(delta_p))
    
    print(f"\n  Pairwise cos(mlp_out direction) across prompts:")
    for i in range(len(VARIED_PROMPTS)):
        sims = [cos_sim(mlp_dirs[i], mlp_dirs[j]) for j in range(len(VARIED_PROMPTS))]
        short = VARIED_PROMPTS[i][:30]
        print(f"    {short:>30}  " + "  ".join(f"{s:.3f}" for s in sims))
    
    # Also check: L26 collapse direction
    print(f"\n  L26 collapse direction comparison:")
    collapse_dirs = []
    for prompt in VARIED_PROMPTS[:3]:
        tids = tokenizer.encode(prompt)
        h_p = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(26):
            h_p = engine.layers[li](h_p)
        dd_p = decompose_layer(engine, h_p, 26)
        mlp_p = dd_p['mlp_out'][0, 0, :]
        collapse_dirs.append(mlp_p / np.linalg.norm(mlp_p))
    
    for i in range(len(collapse_dirs)):
        for j in range(i+1, len(collapse_dirs)):
            cos = cos_sim(collapse_dirs[i], collapse_dirs[j])
            print(f"    cos(P{i}, P{j}) = {cos:.4f}")
    
    # L3 explosion vs L26 collapse: are they opposite?
    print(f"\n  L3 explosion vs L26 collapse direction:")
    cos_exp_col = cos_sim(mlp_dirs[0], collapse_dirs[0])
    print(f"    cos(L3_mlp, L26_mlp) = {cos_exp_col:.4f}")
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    mean_cos = np.mean([cos_sim(mlp_dirs[0], mlp_dirs[i]) 
                        for i in range(1, len(mlp_dirs))])
    print(f"\n  L3 MLP explosion direction: cross-prompt mean cos = {mean_cos:.4f}")
    
    if mean_cos > 0.9:
        print("  → Explosion direction is UNIVERSAL (content-independent)")
    elif mean_cos > 0.5:
        print("  → Explosion direction is PARTIALLY content-dependent")
    else:
        print("  → Explosion direction varies across prompts")
    
    print(f"\n  L3 vs L26: cos = {cos_exp_col:.4f}")
    if abs(cos_exp_col) > 0.5:
        if cos_exp_col > 0:
            print("  → L26 collapse is ALIGNED with L3 explosion (same direction)")
        else:
            print("  → L26 collapse REVERSES L3 explosion (opposite direction)")
    else:
        print("  → L3 and L26 are ORTHOGONAL (different operations)")
    
    print()


if __name__ == '__main__':
    main()
