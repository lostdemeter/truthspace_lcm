"""
Phase 10z25: Four Open Questions from DC 276
=============================================

Q1: What geometric structure lives in the MLP amplification layers (L24-L31)?
Q2: Do the geometric structures have cross-layer versions?
Q3: Can the five structures form a composition algebra?
Q4: Do other L22-L23 heads implement the Selector-Resonator-Lens triad?

Plan:
  Part A (Q1): MLP Amplification Anatomy
    - Track answer rank through each sub-operation (attn, MLP) at L22-L31
    - Measure answer signal strength at each stage
    - Identify which MLP layers amplify vs suppress

  Part B (Q4): Other Heads — Triad Census
    - For all 28 heads at L23: compute binding, measure answer rank
    - Check MESH rank-1 property for all heads
    - Classify heads by structure type

  Part C (Q2): Cross-Layer Structure
    - Compare Selector directions (d_k) across L22/L23
    - Compare Lens SVD bases across layers
    - Measure structural continuity

  Part D (Q3): Composition Algebra
    - Measure head output orthogonality
    - Check if structures compose linearly
    - Look for algebraic closure
"""

import sys, os, numpy as np, time, gc, json
sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_integer import phi_to_float

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

def get_logits(engine, hidden_3d):
    normed = rms_norm(hidden_3d, engine.final_norm_weight)
    return engine.lm_head(normed)[0, -1, :]

def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids: return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])

def decode_lm_row(engine, tid):
    s = engine.lm_head.weight.signs[tid:tid+1, :]
    e = engine.lm_head.weight.exponents[tid:tid+1, :]
    return phi_to_float(s, e)[0]

def predecode_layer_weights(engine, layer_idx):
    attn = engine.layers[layer_idx].attention
    W_v = phi_to_float(attn.W_v.signs, attn.W_v.exponents)
    W_o = phi_to_float(attn.W_o.signs, attn.W_o.exponents)
    return W_v, attn.b_v.copy(), W_o

def get_head_matrices(W_v, b_v, W_o, head_idx, hd=128, nh=28, nkv=4):
    kv = head_idx // (nh // nkv)
    W_v_h = W_v[kv*hd:(kv+1)*hd, :]
    b_v_h = b_v[kv*hd:(kv+1)*hd]
    W_o_h = W_o[:, head_idx*hd:(head_idx+1)*hd]
    return W_v_h, b_v_h, W_o_h

def full_forward_capture_detailed(engine, prompt_ids):
    """
    Forward pass capturing post-attention AND post-MLP states at each layer.
    Returns per-layer dict with:
      normed: pre-attention normed state
      h_post_attn: state after attention + residual (before MLP)
      h_post_mlp: state after MLP + residual
      attn_weights: attention weights for last token
    """
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    seq_len = h.shape[1]
    layer_data = []
    
    for li, layer in enumerate(engine.layers):
        attn = layer.attention
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        
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
            scores += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        weights = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ao = phi_linear(attn.W_o, ao)
        
        h_post_attn = h + ao
        
        mlp = layer.mlp
        nm = rms_norm(h_post_attn, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        h_post_mlp = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)
        
        # Save last-token states only to save memory
        last = seq_len - 1
        ld = {
            'normed': normed[0, last].copy(),
            'h_pre': h[0, last].copy(),
            'h_post_attn': h_post_attn[0, last].copy(),
            'h_post_mlp': h_post_mlp[0, last].copy(),
            'attn_weights_last': weights[0, :, last, :].copy(),  # (nh, seq_len)
        }
        layer_data.append(ld)
        h = h_post_mlp
    
    return layer_data, get_logits(engine, h)


def find_country_pos(tokens, country):
    for i, t in enumerate(tokens):
        if country.lower() in t.lower():
            return i
    return None


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("  PHASE 10z25: FOUR OPEN QUESTIONS FROM DC 276", flush=True)
    print("=" * 72, flush=True)

    facts = {
        'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
        'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
        'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
        'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
    }

    # Get answer token directions from LM head
    answer_dirs = {}
    for country, info in facts.items():
        tids = tokenizer.encode(info['answer'])
        if tids:
            answer_dirs[country] = decode_lm_row(engine, tids[0])
    
    # ══════════════════════════════════════════════════════════════════
    # PART A: MLP Amplification Anatomy (Q1)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72, flush=True)
    print("  Part A (Q1): MLP Amplification — What Structure Lives in the MLP?", flush=True)
    print("─" * 72, flush=True)

    # Run detailed forward for France 
    p_ids = tokenizer.encode(facts['France']['prompt'])
    print(f"\n  Running detailed forward capture for France...", flush=True)
    layer_data, final_logits = full_forward_capture_detailed(engine, p_ids)
    fr_rank_final, _ = get_rank(final_logits, facts['France']['answer'], tokenizer)
    print(f"  Final answer rank: {fr_rank_final}", flush=True)

    # Track answer rank and signal through each sub-operation
    print(f"\n  Answer rank trajectory (France → Paris):", flush=True)
    print(f"  {'Layer':>6s}  {'PostAttn':>8s}  {'PostMLP':>8s}  {'AttnΔ':>8s}  {'MLPΔ':>8s}  "
          f"{'AnsProj_pre':>11s}  {'AnsProj_attn':>12s}  {'AnsProj_mlp':>11s}", flush=True)
    print("  " + "─" * 95, flush=True)

    ans_dir_fr = answer_dirs['France']
    ans_dir_fr_unit = ans_dir_fr / np.linalg.norm(ans_dir_fr)

    results_a = {}
    for li in range(28):
        ld = layer_data[li]
        
        # Get answer ranks at post-attn and post-mlp states
        h_attn_3d = ld['h_post_attn'][np.newaxis, np.newaxis, :].astype(np.float32)
        h_mlp_3d = ld['h_post_mlp'][np.newaxis, np.newaxis, :].astype(np.float32)
        
        logits_attn = get_logits(engine, h_attn_3d)
        logits_mlp = get_logits(engine, h_mlp_3d)
        
        rank_attn, _ = get_rank(logits_attn, ' Paris', tokenizer)
        rank_mlp, _ = get_rank(logits_mlp, ' Paris', tokenizer)
        
        # Answer signal projection
        proj_pre = float(np.dot(ld['h_pre'], ans_dir_fr_unit))
        proj_attn = float(np.dot(ld['h_post_attn'], ans_dir_fr_unit))
        proj_mlp = float(np.dot(ld['h_post_mlp'], ans_dir_fr_unit))
        
        attn_delta = rank_attn - (results_a[li-1]['rank_mlp'] if li > 0 else rank_attn)
        mlp_delta = rank_mlp - rank_attn
        
        results_a[li] = {
            'rank_attn': rank_attn, 'rank_mlp': rank_mlp,
            'proj_pre': proj_pre, 'proj_attn': proj_attn, 'proj_mlp': proj_mlp,
        }
        
        # Print L20+ in detail, summary for earlier
        if li >= 20 or li in [0, 5, 10, 15]:
            marker = ""
            if mlp_delta < -10:
                marker = " ▲▲ MLP BOOST"
            elif mlp_delta > 10:
                marker = " ▼▼ MLP HURT"
            print(f"  L{li:4d}  {rank_attn:8d}  {rank_mlp:8d}  {attn_delta:+8d}  {mlp_delta:+8d}  "
                  f"{proj_pre:11.3f}  {proj_attn:12.3f}  {proj_mlp:11.3f}{marker}", flush=True)

    # Now do the same for all 6 countries to confirm pattern
    print(f"\n  MLP amplification per country (L22-L27 post-MLP ranks):", flush=True)
    print(f"  {'Country':>10s}", end="", flush=True)
    for li in range(20, 28):
        print(f"  {'L'+str(li):>6s}", end="")
    print(f"  {'Final':>6s}", flush=True)
    print("  " + "─" * 75, flush=True)

    results_a_multi = {}
    for country, info in facts.items():
        p_ids_c = tokenizer.encode(info['prompt'])
        ld_c, logits_c = full_forward_capture_detailed(engine, p_ids_c)
        
        ranks_c = []
        for li in range(20, 28):
            h3d = ld_c[li]['h_post_mlp'][np.newaxis, np.newaxis, :].astype(np.float32)
            log_c = get_logits(engine, h3d)
            r, _ = get_rank(log_c, info['answer'], tokenizer)
            ranks_c.append(r)
        
        final_r, _ = get_rank(logits_c, info['answer'], tokenizer)
        results_a_multi[country] = {'layer_ranks': ranks_c, 'final': final_r}
        
        print(f"  {country:>10s}", end="")
        for r in ranks_c:
            print(f"  {r:6d}", end="")
        print(f"  {final_r:6d}", flush=True)
    
    del layer_data  # free memory
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PART B: Other Heads — Triad Census (Q4)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72, flush=True)
    print("  Part B (Q4): Other Heads — Do They Implement the Triad?", flush=True)
    print("─" * 72, flush=True)

    # Pre-decode L23 weights
    print("  Pre-decoding L23 weights...", flush=True)
    W_v_23, b_v_23, W_o_23 = predecode_layer_weights(engine, 23)
    
    # Get QK weights for MESH analysis
    attn_23 = engine.layers[23].attention
    W_q_23 = phi_to_float(attn_23.W_q.signs, attn_23.W_q.exponents)
    W_k_23 = phi_to_float(attn_23.W_k.signs, attn_23.W_k.exponents)
    b_q_23 = attn_23.b_q.copy()
    b_k_23 = attn_23.b_k.copy()
    print("  Done", flush=True)

    # Get entity normed states at L23
    entity_normed = {}
    p_ids_fr = tokenizer.encode(facts['France']['prompt'])
    tokens_fr = [tokenizer.decode([tid]) for tid in p_ids_fr]
    cpos_fr = find_country_pos(tokens_fr, 'France')
    
    # Quick forward capture for France only (need normed at L23)
    ld_fr, _ = full_forward_capture_detailed(engine, p_ids_fr)
    normed_fr_23 = ld_fr[23]['normed']  # Already last-token... wait, need country pos
    
    # Actually need country-position normed state. Let me redo capture differently.
    # The full_forward_capture only saves last token. Need country pos too.
    # Let me do a simpler targeted capture.
    h = engine.embedding(p_ids_fr)[np.newaxis, :, :]
    seq_len = h.shape[1]
    for li, layer in enumerate(engine.layers):
        attn = layer.attention
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        normed = rms_norm(h, attn.norm_weight)
        
        if li == 23:
            normed_fr_23_cpos = normed[0, cpos_fr].copy()
            normed_fr_23_last = normed[0, -1].copy()
        
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        V = phi_linear(attn.W_v, normed, attn.b_v)
        Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if seq_len > 1:
            scores += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        weights = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ao = phi_linear(attn.W_o, ao)
        h_post = h + ao
        mlp = layer.mlp
        nm = rms_norm(h_post, mlp.norm_weight)
        g, u = phi_linear(mlp.W_gate, nm), phi_linear(mlp.W_up, nm)
        h = h_post + phi_linear(mlp.W_down, phi_silu(g) * u)

    del ld_fr
    gc.collect()
    
    # Test ALL 28 heads at L23 for binding quality
    print(f"\n  Per-head binding quality (L23, France → Paris):", flush=True)
    print(f"  {'Head':>6s}  {'Rank':>6s}  {'Rank_last':>9s}  {'S0/S1_Wo':>9s}  "
          f"{'Eff_rank90':>10s}  {'||bind||':>8s}", flush=True)
    print("  " + "─" * 60, flush=True)

    results_b = {}
    for hi in range(28):
        W_v_h, b_v_h, W_o_h = get_head_matrices(W_v_23, b_v_23, W_o_23, hi)
        
        # Binding from country position
        v_cpos = normed_fr_23_cpos @ W_v_h.T + b_v_h
        binding_cpos = v_cpos @ W_o_h.T
        
        # Binding from last position
        v_last = normed_fr_23_last @ W_v_h.T + b_v_h
        binding_last = v_last @ W_o_h.T
        
        # Get answer rank from binding
        b_3d_c = binding_cpos[np.newaxis, np.newaxis, :].astype(np.float32)
        b_3d_l = binding_last[np.newaxis, np.newaxis, :].astype(np.float32)
        logits_c = get_logits(engine, b_3d_c)
        logits_l = get_logits(engine, b_3d_l)
        rank_c, _ = get_rank(logits_c, ' Paris', tokenizer)
        rank_l, _ = get_rank(logits_l, ' Paris', tokenizer)
        
        # SVD of W_o_h.T for aperture analysis
        _, S_wo, _ = np.linalg.svd(W_o_h.T, full_matrices=False)
        ratio_wo = float(S_wo[0] / S_wo[1]) if S_wo[1] > 0 else float('inf')
        en = np.cumsum(S_wo**2) / np.sum(S_wo**2)
        eff_rank_90 = int(np.searchsorted(en, 0.9) + 1)
        
        bind_norm = float(np.linalg.norm(binding_cpos))
        
        results_b[hi] = {
            'rank_cpos': rank_c, 'rank_last': rank_l,
            'S0_S1_Wo': ratio_wo, 'eff_rank_90': eff_rank_90,
            'bind_norm': bind_norm,
        }
        
        marker = " <<<" if hi == 6 else ""
        if rank_c < 100 or rank_l < 100:
            marker += " ★"
        print(f"  H{hi:4d}  {rank_c:6d}  {rank_l:9d}  {ratio_wo:9.3f}  "
              f"{eff_rank_90:10d}  {bind_norm:8.3f}{marker}", flush=True)

    # MESH rank-1 analysis for all heads
    print(f"\n  MESH rank-1 analysis (all L23 heads):", flush=True)
    print(f"  {'Head':>6s}  {'MESH_S0/S1':>10s}  {'Pattern':>10s}", flush=True)
    print("  " + "─" * 35, flush=True)

    hd = 128
    nkv = 4
    nh = 28
    hpk = nh // nkv
    
    mesh_results = {}
    for hi in range(28):
        kv = hi // hpk
        # Q head: row block of W_q
        W_q_h = W_q_23[hi*hd:(hi+1)*hd, :]
        b_q_h = b_q_23[hi*hd:(hi+1)*hd]
        # K head: kv group
        W_k_h = W_k_23[kv*hd:(kv+1)*hd, :]
        b_k_h = b_k_23[kv*hd:(kv+1)*hd]
        
        # MESH = W_q_h @ W_k_h.T (in head space, 128x128)
        # But the dominant structure is bias: MESH_bias = b_q_h @ b_k_h.T
        mesh_ww = W_q_h @ W_k_h.T  # weight-weight
        mesh_full = mesh_ww  # The bias contribution goes through differently...
        # Actually MESH in hidden space: for positions i,j:
        #   score = (h_i W_q + b_q) · (h_j W_k + b_k) / sqrt(d)
        # The bias-bias term: b_q · b_k (scalar, same for all positions)
        # So the rank-1 structure is from the bias outer product in hidden space:
        #   d_q = W_q.T @ b_q (project bias into hidden space)
        #   d_k = W_k.T @ b_k
        #   MESH_bias ∝ d_q ⊗ d_k
        
        d_q_h = W_q_h.T @ b_q_h  # 3584-d
        d_k_h = W_k_h.T @ b_k_h  # 3584-d
        cos_qk = float(np.dot(d_q_h, d_k_h) / (np.linalg.norm(d_q_h) * np.linalg.norm(d_k_h) + 1e-10))
        
        # Check bias-bias dominance: ||b_q|| * ||b_k|| vs ||W_q|| * ||W_k||
        bias_mag = np.linalg.norm(b_q_h) * np.linalg.norm(b_k_h)
        weight_mag = np.linalg.norm(mesh_ww, 'fro')
        ratio = bias_mag / (weight_mag + 1e-10)
        
        # SVD of mesh in head space
        S_mesh = np.linalg.svd(mesh_ww, compute_uv=False)
        mesh_ratio = float(S_mesh[0] / S_mesh[1]) if S_mesh[1] > 0 else float('inf')
        
        pattern = "rank-1" if mesh_ratio > 100 else ("structured" if mesh_ratio > 5 else "full-rank")
        mesh_results[hi] = {
            'mesh_ratio': mesh_ratio, 'cos_qk': cos_qk,
            'bias_weight_ratio': float(ratio), 'pattern': pattern,
        }
        
        marker = " <<<" if hi == 6 else ""
        print(f"  H{hi:4d}  {mesh_ratio:10.1f}  {pattern:>10s}{marker}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART C: Cross-Layer Structure (Q2)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72, flush=True)
    print("  Part C (Q2): Cross-Layer Structure — Do Structures Persist?", flush=True)
    print("─" * 72, flush=True)

    # Pre-decode L22 weights
    print("  Pre-decoding L22 weights...", flush=True)
    W_v_22, b_v_22, W_o_22 = predecode_layer_weights(engine, 22)
    attn_22 = engine.layers[22].attention
    W_q_22 = phi_to_float(attn_22.W_q.signs, attn_22.W_q.exponents)
    W_k_22 = phi_to_float(attn_22.W_k.signs, attn_22.W_k.exponents)
    b_q_22 = attn_22.b_q.copy()
    b_k_22 = attn_22.b_k.copy()
    print("  Done", flush=True)

    # Compare d_k directions between L22 and L23
    print(f"\n  Selector direction (d_k) comparison L22 vs L23:", flush=True)
    print(f"  {'Head':>6s}  {'cos(d_k22, d_k23)':>18s}  {'||d_k22||':>10s}  {'||d_k23||':>10s}", flush=True)
    print("  " + "─" * 50, flush=True)

    results_c_dk = {}
    for hi in range(28):
        kv = hi // hpk
        # L22 d_k
        W_k_h22 = W_k_22[kv*hd:(kv+1)*hd, :]
        b_k_h22 = b_k_22[kv*hd:(kv+1)*hd]
        d_k_22 = W_k_h22.T @ b_k_h22
        
        # L23 d_k
        W_k_h23 = W_k_23[kv*hd:(kv+1)*hd, :]
        b_k_h23 = b_k_23[kv*hd:(kv+1)*hd]
        d_k_23 = W_k_h23.T @ b_k_h23
        
        cos_dk = float(np.dot(d_k_22, d_k_23) / (np.linalg.norm(d_k_22) * np.linalg.norm(d_k_23) + 1e-10))
        
        results_c_dk[hi] = {
            'cos_dk_22_23': cos_dk,
            'norm_dk_22': float(np.linalg.norm(d_k_22)),
            'norm_dk_23': float(np.linalg.norm(d_k_23)),
        }
        
        if hi in [6, 15, 19] or abs(cos_dk) > 0.5:
            marker = " <<<" if hi == 6 else ""
            print(f"  H{hi:4d}  {cos_dk:18.4f}  {np.linalg.norm(d_k_22):10.3f}  "
                  f"{np.linalg.norm(d_k_23):10.3f}{marker}", flush=True)

    # Compare Lens SVD bases between L22 and L23
    print(f"\n  Lens (M_h) SVD basis comparison L22 vs L23:", flush=True)
    print(f"  {'Head':>6s}  {'SubspaceAngle':>13s}  {'cos(u1_22,u1_23)':>17s}", flush=True)
    print("  " + "─" * 45, flush=True)

    results_c_lens = {}
    for hi in [6, 15, 19, 0, 1, 2]:
        # L22 M_h SVD
        W_v_h22, _, W_o_h22 = get_head_matrices(W_v_22, b_v_22, W_o_22, hi)
        inner_22 = W_v_h22 @ W_o_h22  # 128x128
        U_22, S_22, Vt_22 = np.linalg.svd(inner_22, full_matrices=False)
        
        # L23 M_h SVD
        W_v_h23, _, W_o_h23 = get_head_matrices(W_v_23, b_v_23, W_o_23, hi)
        inner_23 = W_v_h23 @ W_o_h23
        U_23, S_23, Vt_23 = np.linalg.svd(inner_23, full_matrices=False)
        
        # Subspace angle between top-10 SVD bases
        # Principal angles: cos(θ_i) = singular values of U_22[:,:k].T @ U_23[:,:k]
        k = 10
        cross = U_22[:, :k].T @ U_23[:, :k]  # k x k
        s_cross = np.linalg.svd(cross, compute_uv=False)
        mean_angle = float(np.mean(np.arccos(np.clip(s_cross, -1, 1))) * 180 / np.pi)
        
        # Top-1 alignment
        cos_u1 = float(abs(np.dot(U_22[:, 0], U_23[:, 0])))
        
        results_c_lens[hi] = {
            'subspace_angle_deg': mean_angle,
            'cos_u1': cos_u1,
        }
        
        marker = " <<<" if hi == 6 else ""
        print(f"  H{hi:4d}  {mean_angle:13.1f}°  {cos_u1:17.4f}{marker}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART D: Composition Algebra (Q3)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72, flush=True)
    print("  Part D (Q3): Composition — How Do Structures Combine?", flush=True)
    print("─" * 72, flush=True)

    # Test 1: Are head outputs orthogonal?
    # Compute binding from each L23 head for France, check pairwise cosines
    print(f"\n  Head output orthogonality (L23, France binding):", flush=True)
    
    head_bindings = []
    for hi in range(28):
        W_v_h, b_v_h, W_o_h = get_head_matrices(W_v_23, b_v_23, W_o_23, hi)
        v = normed_fr_23_last @ W_v_h.T + b_v_h
        binding = v @ W_o_h.T
        head_bindings.append(binding)
    
    head_bindings = np.array(head_bindings)  # (28, 3584)
    
    # Pairwise cosine matrix
    norms = np.linalg.norm(head_bindings, axis=1, keepdims=True)
    normalized = head_bindings / (norms + 1e-10)
    cos_matrix = normalized @ normalized.T  # (28, 28)
    
    # Report statistics
    off_diag = cos_matrix[np.triu_indices(28, k=1)]
    print(f"  Pairwise cosine statistics (off-diagonal):", flush=True)
    print(f"    Mean:   {np.mean(off_diag):.4f}", flush=True)
    print(f"    Std:    {np.std(off_diag):.4f}", flush=True)
    print(f"    Max:    {np.max(off_diag):.4f}", flush=True)
    print(f"    Min:    {np.min(off_diag):.4f}", flush=True)
    print(f"    |cos|>0.3: {np.sum(np.abs(off_diag) > 0.3)} / {len(off_diag)}", flush=True)

    # Test 2: Does the SUM of head bindings improve over individual?
    sum_binding = np.sum(head_bindings, axis=0)
    sum_3d = sum_binding[np.newaxis, np.newaxis, :].astype(np.float32)
    logits_sum = get_logits(engine, sum_3d)
    rank_sum, _ = get_rank(logits_sum, ' Paris', tokenizer)
    
    # Top-k heads by individual rank
    individual_ranks = [results_b[hi]['rank_cpos'] for hi in range(28)]
    sorted_heads = sorted(range(28), key=lambda h: individual_ranks[h])
    
    print(f"\n  Sum of all 28 head bindings → Paris rank: {rank_sum}", flush=True)
    print(f"  Best individual head (H{sorted_heads[0]}) → Paris rank: {individual_ranks[sorted_heads[0]]}", flush=True)
    
    # Test 3: Cumulative sum — add heads in order of quality, track rank
    print(f"\n  Cumulative head addition (best → worst):", flush=True)
    print(f"  {'#Heads':>7s}  {'Added':>6s}  {'Rank':>6s}", flush=True)
    print("  " + "─" * 25, flush=True)
    
    cumulative = np.zeros(3584)
    results_d_cumulative = {}
    for i, hi in enumerate(sorted_heads):
        cumulative += head_bindings[hi]
        if (i+1) in [1, 2, 3, 5, 7, 10, 14, 20, 28]:
            cum_3d = cumulative[np.newaxis, np.newaxis, :].astype(np.float32)
            logits_cum = get_logits(engine, cum_3d)
            rank_cum, _ = get_rank(logits_cum, ' Paris', tokenizer)
            results_d_cumulative[i+1] = {'rank': rank_cum, 'head': hi}
            print(f"  {i+1:7d}  H{hi:4d}  {rank_cum:6d}", flush=True)

    # Test 4: Dimensional hierarchy analysis
    # The structures operate at scales 1, 10, 66, 128, 3584
    # Check if they form a geometric series
    scales = [1, 10, 66, 128, 3584]
    ratios = [scales[i+1]/scales[i] for i in range(len(scales)-1)]
    print(f"\n  Dimensional hierarchy:", flush=True)
    print(f"    Scales: {scales}", flush=True)
    print(f"    Ratios: {[f'{r:.1f}' for r in ratios]}", flush=True)
    print(f"    Product: {np.prod(ratios):.0f} (=3584)", flush=True)
    
    # Check if related to phi
    import math
    phi = (1 + math.sqrt(5)) / 2
    print(f"    φ^1 = {phi:.3f}", flush=True)
    print(f"    φ^5 = {phi**5:.1f} vs 10 (ratio {10/phi**5:.2f})", flush=True)
    print(f"    φ^9 = {phi**9:.1f} vs 66 (ratio {66/phi**9:.2f})", flush=True)
    print(f"    φ^12 = {phi**12:.1f} vs 128 (ratio {128/phi**12:.2f})", flush=True)
    print(f"    φ^18 = {phi**18:.1f} vs 3584 (ratio {3584/phi**18:.2f})", flush=True)
    
    # Test 5: Does Attn + MLP compose as a single transformation?
    # Check if the residual stream change from L23 is dominated by attn or MLP
    # Use the detailed France capture we already have
    p_ids_fr2 = tokenizer.encode(facts['France']['prompt'])
    ld_fr2, _ = full_forward_capture_detailed(engine, p_ids_fr2)
    
    for li in [22, 23, 24, 25, 26, 27]:
        ld = ld_fr2[li]
        delta_attn = ld['h_post_attn'] - ld['h_pre']
        delta_mlp = ld['h_post_mlp'] - ld['h_post_attn']
        delta_total = ld['h_post_mlp'] - ld['h_pre']
        
        # How much of total is explained by attn vs mlp?
        cos_attn_total = float(np.dot(delta_attn, delta_total) / 
                              (np.linalg.norm(delta_attn) * np.linalg.norm(delta_total) + 1e-10))
        cos_mlp_total = float(np.dot(delta_mlp, delta_total) / 
                             (np.linalg.norm(delta_mlp) * np.linalg.norm(delta_total) + 1e-10))
        cos_attn_mlp = float(np.dot(delta_attn, delta_mlp) /
                            (np.linalg.norm(delta_attn) * np.linalg.norm(delta_mlp) + 1e-10))
        
        norm_ratio = float(np.linalg.norm(delta_mlp) / (np.linalg.norm(delta_attn) + 1e-10))
        
        print(f"  L{li}: cos(Δattn, Δmlp)={cos_attn_mlp:+.3f}, "
              f"||Δmlp||/||Δattn||={norm_ratio:.2f}, "
              f"cos(Δattn,Δtot)={cos_attn_total:.3f}, "
              f"cos(Δmlp,Δtot)={cos_mlp_total:.3f}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 72, flush=True)

    # Q1 Summary
    print("\n  Q1 (MLP Structure):", flush=True)
    for li in range(20, 28):
        r = results_a[li]
        delta = r['rank_mlp'] - r['rank_attn']
        label = "AMPLIFY" if delta < -5 else ("SUPPRESS" if delta > 5 else "neutral")
        print(f"    L{li}: attn→{r['rank_attn']:5d}, mlp→{r['rank_mlp']:5d} ({label})", flush=True)

    # Q4 Summary  
    print("\n  Q4 (Other Heads):", flush=True)
    good_heads = [hi for hi in range(28) if results_b[hi]['rank_cpos'] < 100]
    print(f"    Heads producing Paris at rank<100 from binding: {good_heads}", flush=True)
    rank1_heads = [hi for hi in range(28) if mesh_results[hi]['mesh_ratio'] > 100]
    print(f"    Heads with rank-1 MESH: {rank1_heads}", flush=True)

    # Q2 Summary
    print("\n  Q2 (Cross-Layer):", flush=True)
    for hi in [6, 15, 19]:
        if hi in results_c_dk:
            print(f"    H{hi}: cos(d_k_L22, d_k_L23) = {results_c_dk[hi]['cos_dk_22_23']:.4f}", flush=True)

    # Q3 Summary
    print("\n  Q3 (Composition):", flush=True)
    print(f"    Head outputs mean |cos| = {np.mean(np.abs(off_diag)):.4f}", flush=True)
    print(f"    Sum of 28 heads → rank {rank_sum}", flush=True)

    # Save results
    out = {
        'part_a_france_trajectory': {str(k): {kk: (vv if not isinstance(vv, np.floating) else float(vv)) 
                                               for kk, vv in v.items()} for k, v in results_a.items()},
        'part_a_multi_country': {c: v for c, v in results_a_multi.items()},
        'part_b_heads': {str(k): v for k, v in results_b.items()},
        'part_b_mesh': {str(k): v for k, v in mesh_results.items()},
        'part_c_dk': {str(k): v for k, v in results_c_dk.items()},
        'part_c_lens': {str(k): v for k, v in results_c_lens.items()},
        'part_d_orthogonality': {
            'mean_cos': float(np.mean(off_diag)),
            'std_cos': float(np.std(off_diag)),
            'max_cos': float(np.max(off_diag)),
        },
        'part_d_cumulative': {str(k): v for k, v in results_d_cumulative.items()},
    }
    out_path = 'experiments/model_reverse_engineering_v2/results/phase10z25_four_questions.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}", flush=True)
    print(f"  Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
