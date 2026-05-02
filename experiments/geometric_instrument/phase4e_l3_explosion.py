"""
Phase 4e: L3 Explosion Deep-Dive
==================================

F132-F133 revealed that BOS norm jumps 108x at layer 3 (66.6 → 7185.8).
This single layer creates the information reservoir that all other layers
depend on for BOS-sink attention.

Investigations:
  1. Decompose L3: attention vs MLP contribution to the explosion
  2. Per-position analysis: does only BOS explode, or all positions?
  3. Universality: is L3 always the explosive layer across prompts/lengths?
  4. Weight geometry: what's special about L3's weights vs other layers?
  5. Direction analysis: what direction does the explosion point?
  6. Comparison with L0-L2 (mild growth) and L4+ (plateau)
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}

VARIED_PROMPTS = [
    'The capital of France is',
    'I know the capital of France is',
    'Can you tell me the capital of France is',
    'Hello world',
    'The quick brown fox jumps over',
]


def decompose_layer(engine, h, layer_idx):
    """Run a layer and return intermediate states for BOS analysis."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    
    # Attention
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
    
    # MLP
    mlp = layer.mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    h_post_mlp = h_post_attn + mlp_out
    
    return {
        'h_in': h,
        'normed_attn': normed,
        'attn_weights': weights[0],  # [nh, seq, seq]
        'attn_out': attn_out,        # [1, seq, d]
        'h_post_attn': h_post_attn,
        'normed_mlp': nm,
        'mlp_out': mlp_out,
        'h_post_mlp': h_post_mlp,
        'V': V,                       # [1, nkv, seq, hd]
    }


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 80)
    print("  Phase 4e: L3 Explosion Deep-Dive")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 1: Decompose L3 — attention vs MLP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 1: Decompose L3 (Attention vs MLP)")
    print("=" * 80)
    
    france_tids = tokenizer.encode('The capital of France is')
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    
    # Run L0-L2 normally to get to L3 input
    for li in range(3):
        h = engine.layers[li](h)
    
    # Now decompose L3
    d = decompose_layer(engine, h, 3)
    
    print(f"\n  France prompt, Layer 3 decomposition (position 0 = 'The'):")
    print(f"  {'Stage':>20}  {'||p0||':>10}  {'||p1||':>10}  {'||p2||':>10}  "
          f"{'||p3||':>10}  {'||p4||':>10}")
    print("  " + "─" * 75)
    
    for name, tensor in [
        ('h_in', d['h_in']),
        ('attn_out', d['attn_out']),
        ('h_post_attn', d['h_post_attn']),
        ('mlp_out', d['mlp_out']),
        ('h_post_mlp', d['h_post_mlp']),
    ]:
        norms = [float(np.linalg.norm(tensor[0, i, :])) for i in range(5)]
        print(f"  {name:>20}  " + "  ".join(f"{n:>10.1f}" for n in norms))
    
    # Ratio analysis
    print(f"\n  Contribution to BOS (p0) norm:")
    h_in_norm = float(np.linalg.norm(d['h_in'][0, 0, :]))
    attn_out_norm = float(np.linalg.norm(d['attn_out'][0, 0, :]))
    mlp_out_norm = float(np.linalg.norm(d['mlp_out'][0, 0, :]))
    h_post_attn_norm = float(np.linalg.norm(d['h_post_attn'][0, 0, :]))
    h_post_mlp_norm = float(np.linalg.norm(d['h_post_mlp'][0, 0, :]))
    
    print(f"    ||h_in[0]||       = {h_in_norm:.1f}")
    print(f"    ||attn_out[0]||   = {attn_out_norm:.1f}")
    print(f"    ||h_post_attn[0]||= {h_post_attn_norm:.1f}")
    print(f"    ||mlp_out[0]||    = {mlp_out_norm:.1f}")
    print(f"    ||h_post_mlp[0]|| = {h_post_mlp_norm:.1f}")
    
    cos_in_attn = cos_sim(d['h_in'][0, 0, :], d['attn_out'][0, 0, :])
    cos_in_mlp = cos_sim(d['h_in'][0, 0, :], d['mlp_out'][0, 0, :])
    cos_attn_mlp = cos_sim(d['attn_out'][0, 0, :], d['mlp_out'][0, 0, :])
    
    print(f"\n  Direction analysis (BOS position):")
    print(f"    cos(h_in, attn_out)  = {cos_in_attn:.4f}")
    print(f"    cos(h_in, mlp_out)   = {cos_in_mlp:.4f}")
    print(f"    cos(attn_out, mlp_out) = {cos_attn_mlp:.4f}")
    
    # What fraction of final norm comes from each?
    print(f"\n  Is the explosion in attention or MLP?")
    if attn_out_norm > mlp_out_norm:
        print(f"    ATTENTION dominates: {attn_out_norm:.1f} vs {mlp_out_norm:.1f}")
    else:
        print(f"    MLP dominates: {mlp_out_norm:.1f} vs {attn_out_norm:.1f}")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 2: Attention at L3 — what does BOS attend to?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 2: L3 Attention Pattern")
    print("=" * 80)
    
    attn_w = d['attn_weights']  # [nh, seq, seq]
    # BOS attends to... (row 0 of each head's attention matrix)
    print(f"\n  L3 attention weights FROM position 0 (BOS) TO all positions:")
    print(f"  {'Head':>4}  {'→p0':>6}  {'→p1':>6}  {'→p2':>6}  {'→p3':>6}  {'→p4':>6}")
    print("  " + "─" * 36)
    for hi in range(28):
        vals = attn_w[hi, 0, :]  # BOS can only attend to itself
        print(f"  H{hi:>2}  " + "  ".join(f"{v:.4f}" for v in vals))
    
    # BOS at position 0 can only attend to position 0 (causal mask)
    print(f"\n  NOTE: BOS (pos 0) can ONLY attend to itself (causal mask).")
    print(f"  So attn_out[0] = W_o @ V[0], where V[0] = W_v @ rms_norm(h[0])")
    
    # What about the last token attending to BOS?
    print(f"\n  L3 attention TO position 0 (BOS) FROM last token (p4):")
    for hi in range(28):
        w_bos = float(attn_w[hi, -1, 0])
        if hi % 7 == 0:
            print(f"    H{hi:>2}: {w_bos:.4f}", end="")
        elif hi % 7 == 6:
            print(f"  H{hi:>2}: {w_bos:.4f}")
        else:
            print(f"  H{hi:>2}: {w_bos:.4f}", end="")
    print()
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 3: Compare ALL layers' decomposition at BOS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 3: All Layers — Attention vs MLP at BOS")
    print("=" * 80)
    
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    
    print(f"\n  {'Layer':>5}  {'||h_in||':>10}  {'||attn_out||':>12}  {'||h_post_a||':>10}  "
          f"{'||mlp_out||':>11}  {'||h_final||':>10}  {'attn/mlp':>8}  "
          f"{'cos(a,m)':>8}")
    print("  " + "─" * 95)
    
    for li in range(n_layers):
        dd = decompose_layer(engine, h, li)
        
        h_in_n = float(np.linalg.norm(dd['h_in'][0, 0, :]))
        a_out_n = float(np.linalg.norm(dd['attn_out'][0, 0, :]))
        h_pa_n = float(np.linalg.norm(dd['h_post_attn'][0, 0, :]))
        m_out_n = float(np.linalg.norm(dd['mlp_out'][0, 0, :]))
        h_fin_n = float(np.linalg.norm(dd['h_post_mlp'][0, 0, :]))
        ratio = a_out_n / (m_out_n + 1e-12)
        cos_am = cos_sim(dd['attn_out'][0, 0, :], dd['mlp_out'][0, 0, :])
        
        marker = " ← EXPLOSIVE" if li == 3 else ""
        print(f"  L{li:>3}  {h_in_n:>10.1f}  {a_out_n:>12.1f}  {h_pa_n:>10.1f}  "
              f"{m_out_n:>11.1f}  {h_fin_n:>10.1f}  {ratio:>8.2f}  "
              f"{cos_am:>8.4f}{marker}")
        
        h = dd['h_post_mlp']
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 4: Universality across prompts
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 4: Universality Across Prompts")
    print("=" * 80)
    
    print(f"\n  BOS norm before/after each of L0-L5:")
    print(f"  {'Prompt':>40}  {'emb':>6}  {'L0':>7}  {'L1':>7}  {'L2':>7}  "
          f"{'L3':>7}  {'L4':>7}  {'L5':>7}")
    print("  " + "─" * 95)
    
    for prompt in VARIED_PROMPTS:
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        
        norms = [float(np.linalg.norm(h[0, 0, :]))]
        for li in range(6):
            h = engine.layers[li](h)
            norms.append(float(np.linalg.norm(h[0, 0, :])))
        
        norm_strs = [f"{n:>7.1f}" for n in norms]
        print(f"  {prompt:>40}  " + "  ".join(norm_strs))
    
    # Also check: does every prompt explode at L3?
    print(f"\n  Per-layer BOS norm growth ratio (||h_out[0]|| / ||h_in[0]||):")
    print(f"  {'Prompt':>40}  {'L0':>6}  {'L1':>6}  {'L2':>6}  {'L3':>6}  {'L4':>6}  {'L5':>6}")
    print("  " + "─" * 80)
    
    for prompt in VARIED_PROMPTS:
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        
        prev_norm = float(np.linalg.norm(h[0, 0, :]))
        ratios = []
        for li in range(6):
            h = engine.layers[li](h)
            cur_norm = float(np.linalg.norm(h[0, 0, :]))
            ratios.append(cur_norm / prev_norm if prev_norm > 0 else 0)
            prev_norm = cur_norm
        
        print(f"  {prompt:>40}  " + "  ".join(f"{r:>6.1f}" for r in ratios))
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 5: What's special about L3's weights?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 5: L3 Weight Geometry")
    print("=" * 80)
    
    # Compare weight norms across layers
    print(f"\n  Weight Frobenius norms (attention components):")
    print(f"  {'Layer':>5}  {'||W_q||':>10}  {'||b_q||':>10}  {'||W_k||':>10}  "
          f"{'||b_k||':>10}  {'||W_v||':>10}  {'||b_v||':>10}  {'||W_o||':>10}")
    print("  " + "─" * 80)
    
    for li in [0, 1, 2, 3, 4, 5, 10, 20, 27]:
        attn = engine.layers[li].attention
        norms = [
            float(np.linalg.norm(attn.W_q)),
            float(np.linalg.norm(attn.b_q)),
            float(np.linalg.norm(attn.W_k)),
            float(np.linalg.norm(attn.b_k)),
            float(np.linalg.norm(attn.W_v)),
            float(np.linalg.norm(attn.b_v)),
            float(np.linalg.norm(attn.W_o)),
        ]
        marker = " ←" if li == 3 else ""
        print(f"  L{li:>3}  " + "  ".join(f"{n:>10.2f}" for n in norms) + marker)
    
    print(f"\n  Weight Frobenius norms (MLP components):")
    print(f"  {'Layer':>5}  {'||W_gate||':>12}  {'||W_up||':>10}  {'||W_down||':>12}")
    print("  " + "─" * 42)
    
    for li in [0, 1, 2, 3, 4, 5, 10, 20, 27]:
        mlp = engine.layers[li].mlp
        norms = [
            float(np.linalg.norm(mlp.W_gate)),
            float(np.linalg.norm(mlp.W_up)),
            float(np.linalg.norm(mlp.W_down)),
        ]
        marker = " ←" if li == 3 else ""
        print(f"  L{li:>3}  " + "  ".join(f"{n:>12.2f}" for n in norms) + marker)
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 6: L3's V and W_o at BOS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 6: L3's V·W_o Product at BOS")
    print("=" * 80)
    
    # BOS can only self-attend, so attn_out[0] = W_o @ V[0]
    # What does V[0] look like? What does W_o amplify?
    
    h = engine.embedding(france_tids)[np.newaxis, :, :]
    for li in range(3):
        h = engine.layers[li](h)
    
    dd = decompose_layer(engine, h, 3)
    
    # V values at BOS
    V = dd['V']  # [1, nkv, seq, hd]
    print(f"\n  V values at position 0 (BOS), per KV group:")
    for kv in range(V.shape[1]):
        v_bos = V[0, kv, 0, :]  # [hd]
        v_norm = float(np.linalg.norm(v_bos))
        print(f"    KV{kv}: ||V[0]|| = {v_norm:.4f}, "
              f"max={float(v_bos.max()):.4f}, min={float(v_bos.min()):.4f}")
    
    # The rms_norm of h_in at BOS
    normed_bos = dd['normed_attn'][0, 0, :]
    print(f"\n  RMS-normed BOS hidden state: ||normed|| = {float(np.linalg.norm(normed_bos)):.4f}")
    
    # Singular values of W_o
    attn = engine.layers[3].attention
    U, S, Vt = np.linalg.svd(attn.W_o, full_matrices=False)
    print(f"\n  W_o singular values (L3): top 10 = {S[:10].tolist()}")
    print(f"  W_o singular values (L3): S[0]/S[1] = {S[0]/S[1]:.2f}")
    print(f"  W_o singular values (L3): sum = {float(S.sum()):.2f}")
    
    # Compare with L2 and L4
    for cmp_li in [2, 4]:
        U_c, S_c, Vt_c = np.linalg.svd(engine.layers[cmp_li].attention.W_o, full_matrices=False)
        print(f"  W_o singular values (L{cmp_li}): S[0]={S_c[0]:.4f}, S[0]/S[1]={S_c[0]/S_c[1]:.2f}")
    
    # MLP singular values
    print(f"\n  MLP weight singular values at L3:")
    for name, W in [('W_gate', engine.layers[3].mlp.W_gate),
                     ('W_up', engine.layers[3].mlp.W_up),
                     ('W_down', engine.layers[3].mlp.W_down)]:
        U_m, S_m, Vt_m = np.linalg.svd(W, full_matrices=False)
        print(f"    {name}: S[0]={S_m[0]:.4f}, S[0]/S[1]={S_m[0]/S_m[1]:.2f}, sum={float(S_m.sum()):.2f}")
    
    # ═══════════════════════════════════════════════════════════
    # Investigation 7: The explosion direction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  INVESTIGATION 7: Explosion Direction Analysis")
    print("=" * 80)
    
    # What direction does BOS explode into?
    h_before = dd['h_in'][0, 0, :]
    h_after = dd['h_post_mlp'][0, 0, :]
    delta = h_after - h_before
    
    print(f"\n  BOS state change at L3:")
    print(f"    ||h_before|| = {float(np.linalg.norm(h_before)):.1f}")
    print(f"    ||h_after||  = {float(np.linalg.norm(h_after)):.1f}")
    print(f"    ||delta||    = {float(np.linalg.norm(delta)):.1f}")
    print(f"    cos(before, after) = {cos_sim(h_before, h_after):.4f}")
    print(f"    cos(before, delta) = {cos_sim(h_before, delta):.4f}")
    
    # SVD of the delta direction
    # Is the delta aligned with any principal component of the layer's weights?
    delta_normalized = delta / np.linalg.norm(delta)
    
    # Check alignment with W_down's principal direction
    U_d, S_d, Vt_d = np.linalg.svd(engine.layers[3].mlp.W_down, full_matrices=False)
    for i in range(5):
        cos_with_sv = abs(cos_sim(delta_normalized, U_d[:, i]))
        print(f"    cos(delta, W_down_U[:,{i}]) = {cos_with_sv:.4f}")
    
    # Is the explosion direction the same across prompts?
    print(f"\n  Explosion direction stability across prompts:")
    explosion_dirs = []
    
    for prompt in VARIED_PROMPTS:
        tids = tokenizer.encode(prompt)
        h_p = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(3):
            h_p = engine.layers[li](h_p)
        h_before_p = h_p[0, 0, :].copy()
        h_p = engine.layers[3](h_p)
        h_after_p = h_p[0, 0, :]
        delta_p = h_after_p - h_before_p
        delta_p_norm = delta_p / np.linalg.norm(delta_p)
        explosion_dirs.append(delta_p_norm)
    
    # Pairwise cosine similarities
    print(f"  {'':>5}  " + "  ".join(f"P{i}" for i in range(len(VARIED_PROMPTS))))
    for i in range(len(VARIED_PROMPTS)):
        sims = [f"{cos_sim(explosion_dirs[i], explosion_dirs[j]):.3f}" 
                for j in range(len(VARIED_PROMPTS))]
        print(f"  P{i:>3}  " + "  ".join(f"{s:>5}" for s in sims))
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    print(f"\n  L3 BOS explosion: {h_in_norm:.1f} → {h_post_mlp_norm:.1f} "
          f"({h_post_mlp_norm/h_in_norm:.1f}x)")
    
    if attn_out_norm > mlp_out_norm:
        print(f"  Driver: ATTENTION ({attn_out_norm:.1f} vs MLP {mlp_out_norm:.1f})")
    else:
        print(f"  Driver: MLP ({mlp_out_norm:.1f} vs attention {attn_out_norm:.1f})")
    
    # Cross-prompt similarity of explosion direction
    mean_cos = np.mean([cos_sim(explosion_dirs[0], explosion_dirs[i]) 
                        for i in range(1, len(explosion_dirs))])
    print(f"  Explosion direction cross-prompt cos: mean={mean_cos:.4f}")
    if mean_cos > 0.9:
        print(f"  → Explosion direction is UNIVERSAL (content-independent)")
    elif mean_cos > 0.5:
        print(f"  → Explosion direction is PARTIALLY content-dependent")
    else:
        print(f"  → Explosion direction is CONTENT-DEPENDENT")
    
    print()


if __name__ == '__main__':
    main()
