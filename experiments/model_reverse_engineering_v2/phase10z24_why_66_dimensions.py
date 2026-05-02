"""
Phase 10z24: Why 66 Dimensions? — The Lens Aperture
=====================================================

F123 showed M_h has effective rank ~66 (90% energy). But WHY 66?
Is this related to the number of facts, the vocabulary structure,
the architecture, or something deeper?

Plan:
  Part A: Rank truncation — Truncate M_h's SVD to rank k (1..128),
          apply to known entities. Where does answer quality break?
          Is there a sharp phase transition or gradual degradation?

  Part B: Singular dimension analysis — Project entity bindings into
          M_h's SVD basis. Which dimensions carry entity-distinguishing
          vs answer-producing information?

  Part C: Vocabulary alignment — How do M_h's singular vectors align
          with the LM head's answer tokens? Does the 66-d subspace
          correspond to the span of answer vocabulary?

  Part D: Cross-head aperture — Compare effective ranks across all
          heads at L22-L23. Is 66 special to H6 or universal?
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

def top_k_tokens(tokenizer, logits, k=5):
    top_idx = np.argsort(logits)[-k:][::-1]
    return [(tokenizer.decode([int(i)]), float(logits[i])) for i in top_idx]

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

def compute_M_h(W_v_h, W_o_h):
    """M_h = W_v_h.T @ W_o_h.T  (3584x3584, but via 128-d bottleneck)"""
    return W_v_h.T @ W_o_h.T

def compute_binding(W_v_h, b_v_h, W_o_h, normed):
    v = normed @ W_v_h.T + b_v_h
    return v @ W_o_h.T

def full_forward_capture(engine, prompt_ids):
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    seq_len = h.shape[1]
    layer_data = []
    for layer in engine.layers:
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
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if seq_len > 1:
            scores += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        weights = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ao = phi_linear(attn.W_o, ao)
        layer_data.append({'normed': normed[0].copy(), 'h_pre': h[0].copy()})
        h_post = h + ao
        mlp = layer.mlp
        nm = rms_norm(h_post, mlp.norm_weight)
        g, u = phi_linear(mlp.W_gate, nm), phi_linear(mlp.W_up, nm)
        h = h_post + phi_linear(mlp.W_down, phi_silu(g) * u)
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

    print("\n" + "=" * 72)
    print("  PHASE 10z24: WHY 66 DIMENSIONS?")
    print("=" * 72)

    facts = {
        'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
        'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
        'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
        'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
        'Brazil':  {'prompt': 'The capital of Brazil is',  'answer': ' Bras'},
        'Canada':  {'prompt': 'The capital of Canada is',  'answer': ' Ottawa'},
    }

    extended = {
        'China':     {'prompt': 'The capital of China is',     'answer': ' Beijing'},
        'Russia':    {'prompt': 'The capital of Russia is',    'answer': ' Moscow'},
        'Australia': {'prompt': 'The capital of Australia is', 'answer': ' Canberra'},
        'Mexico':    {'prompt': 'The capital of Mexico is',    'answer': ' Mexico'},
        'Thailand':  {'prompt': 'The capital of Thailand is',  'answer': ' Bangkok'},
        'Poland':    {'prompt': 'The capital of Poland is',    'answer': ' Warsaw'},
        'Sweden':    {'prompt': 'The capital of Sweden is',    'answer': ' Stockholm'},
        'Norway':    {'prompt': 'The capital of Norway is',    'answer': ' Oslo'},
    }

    # Pre-decode L23 weights
    print("\n  Pre-decoding weights...", flush=True)
    import sys
    W_v_23, b_v_23, W_o_23 = predecode_layer_weights(engine, 23)
    W_v_h6, b_v_h6, W_o_h6 = get_head_matrices(W_v_23, b_v_23, W_o_23, 6)

    # Compute M_h and its SVD
    # M_h operates as: binding = normed @ W_v_h.T @ W_o_h.T
    # The bottleneck is the 128-d value space.
    # SVD of the 128x128 inner product: W_v_h @ W_o_h
    inner = W_v_h6 @ W_o_h6  # 128 x 128
    U_inner, S_inner, Vt_inner = np.linalg.svd(inner, full_matrices=False)

    print(f"\n  M_h inner matrix (W_v @ W_o): {inner.shape}")
    print(f"  Singular values (first 20): {np.round(S_inner[:20], 3)}")
    print(f"  Singular values (last 20):  {np.round(S_inner[-20:], 3)}")
    energy = np.cumsum(S_inner**2) / np.sum(S_inner**2)
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rank_thr = int(np.searchsorted(energy, thr) + 1)
        print(f"  Rank for {thr*100:.0f}% energy: {rank_thr}")

    print("    Done", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # Collect entity bindings for all countries
    # ══════════════════════════════════════════════════════════════════
    print("\n  Collecting entity hidden states...", flush=True)
    all_facts = {**facts, **extended}
    entity_normed = {}
    entity_bindings = {}

    for country, info in all_facts.items():
        p_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = find_country_pos(tokens, country)
        if cpos is None:
            print(f"    {country}: SKIP")
            continue
        layer_data, _ = full_forward_capture(engine, p_ids)
        normed_23 = layer_data[23]['normed'][cpos]
        entity_normed[country] = normed_23.copy()
        binding = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_23)
        entity_bindings[country] = binding.copy()

    countries = list(entity_normed.keys())
    print(f"    Collected {len(countries)} countries", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART A: Rank Truncation Sweep
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72, flush=True)
    print("  Part A: Rank Truncation — Where Does Answer Quality Break?", flush=True)
    print("─" * 72, flush=True)

    ranks_to_test = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 66, 70, 80, 90, 100, 110, 120, 128]

    # SVD of W_o_h.T (the output projection from value space to hidden space)
    W_o_h6_T = W_o_h6.T  # 128 x 3584
    U_o, S_o, Vt_o = np.linalg.svd(W_o_h6_T, full_matrices=False)  # U: 128x128, S: 128, Vt: 3584x128... no
    # W_o_h6.T is (128, 3584) so SVD gives U (128x128), S (128), Vt (128x3584)
    
    print(f"\n  SVD of W_o_h6.T (128 x 3584):")
    print(f"  S_o (first 20): {np.round(S_o[:20], 3)}")
    print(f"  S_o (last 20):  {np.round(S_o[-20:], 3)}")
    energy_o = np.cumsum(S_o**2) / np.sum(S_o**2)
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rank_thr = int(np.searchsorted(energy_o, thr) + 1)
        print(f"  Rank for {thr*100:.0f}% energy: {rank_thr}")

    # Now do SVD of the full pipeline: M_h = W_v_h.T @ W_o_h.T (3584 x 3584, rank ≤ 128)
    # But this is huge. Instead, work in the 128-d bottleneck.
    # v = normed @ W_v_h.T + b_v_h  (128-d)
    # binding = v @ W_o_h.T (3584-d)
    # 
    # In SVD of W_o_h.T: binding = v @ U_o @ diag(S_o) @ Vt_o
    # Truncate to rank k: binding_k = (v @ U_o[:,:k]) * S_o[:k] @ Vt_o[:k,:]
    
    print(f"\n  Rank truncation sweep:", flush=True)
    print(f"  {'Rank':>5s}  {'Energy%':>8s}  ", end="", flush=True)
    for c in ['France', 'Japan', 'Germany', 'Italy', 'Spain', 'Egypt']:
        print(f"  {c:>8s}", end="")
    print(f"  {'MeanRank':>9s}")
    print("  " + "─" * 85)

    results_a = {}
    for rank_k in ranks_to_test:
        energy_pct = float(energy_o[min(rank_k-1, 127)]) * 100 if rank_k <= 128 else 100.0
        
        country_ranks = {}
        for country in ['France', 'Japan', 'Germany', 'Italy', 'Spain', 'Egypt',
                        'China', 'Russia', 'Australia', 'Thailand', 'Poland', 'Sweden']:
            if country not in entity_normed:
                continue
            normed = entity_normed[country]
            v = normed @ W_v_h6.T + b_v_h6  # 128-d
            
            # Project v into SVD basis of W_o_h.T, truncate, reconstruct
            v_svd = v @ U_o  # 128-d in SVD basis
            v_trunc = np.zeros_like(v_svd)
            v_trunc[:rank_k] = v_svd[:rank_k]
            
            # Reconstruct binding using truncated v
            binding_k = (v_trunc * S_o) @ Vt_o  # 3584-d
            
            b_3d = binding_k[np.newaxis, np.newaxis, :].astype(np.float32)
            logits = get_logits(engine, b_3d)
            ans = all_facts[country]['answer']
            r, _ = get_rank(logits, ans, tokenizer)
            country_ranks[country] = r

        # Print compact summary for original 6
        mean_rank = np.mean([country_ranks.get(c, 999999) for c in 
                           ['France', 'Japan', 'Germany', 'Italy', 'Spain', 'Egypt']
                           if c in country_ranks])
        print(f"  {rank_k:5d}  {energy_pct:7.1f}%  ", end="")
        for c in ['France', 'Japan', 'Germany', 'Italy', 'Spain', 'Egypt']:
            r = country_ranks.get(c, -1)
            print(f"  {r:8d}", end="")
        print(f"  {mean_rank:9.1f}", flush=True)

        results_a[rank_k] = {
            'energy_pct': energy_pct,
            'ranks': country_ranks,
            'mean_rank_6': float(mean_rank),
        }

    # ══════════════════════════════════════════════════════════════════
    # PART B: What's in Each Singular Dimension?
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part B: What Each Singular Dimension Encodes")
    print("─" * 72)

    # For each entity, project v into SVD basis
    v_svd_all = {}
    for country in countries:
        normed = entity_normed[country]
        v = normed @ W_v_h6.T + b_v_h6
        v_svd_all[country] = v @ U_o  # 128-d in SVD basis

    # Check which SVD dimensions discriminate between entities
    # Compute variance of each dimension across entities
    v_matrix = np.array([v_svd_all[c] for c in countries])  # (N, 128)
    dim_variance = np.var(v_matrix, axis=0)  # variance per SVD dimension
    dim_mean = np.mean(np.abs(v_matrix), axis=0)  # mean magnitude
    
    # Signal-to-noise: variance / mean²
    snr = dim_variance / (dim_mean**2 + 1e-10)
    
    print(f"\n  Entity discrimination per SVD dimension:")
    print(f"  {'Dim':>5s}  {'S_value':>8s}  {'MeanAbs':>8s}  {'Variance':>10s}  {'SNR':>8s}")
    print("  " + "─" * 50)
    for d in range(min(30, 128)):
        print(f"  {d:5d}  {S_o[d]:8.3f}  {dim_mean[d]:8.3f}  {dim_variance[d]:10.5f}  {snr[d]:8.4f}")

    # Which dimensions separate specific countries?
    print(f"\n  Per-entity projections (first 10 SVD dims, weighted by S):")
    print(f"  {'Country':>12s}", end="")
    for d in range(10):
        print(f"  {'d'+str(d):>8s}", end="")
    print()
    print("  " + "─" * 100)
    for country in countries[:12]:
        weighted = v_svd_all[country][:10] * S_o[:10]
        print(f"  {country:>12s}", end="")
        for d in range(10):
            print(f"  {weighted[d]:8.2f}", end="")
        print()

    # ══════════════════════════════════════════════════════════════════
    # PART C: Vocabulary Alignment
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part C: Do SVD Dimensions Align with Answer Vocabulary?")
    print("─" * 72)

    # Get LM head rows for answer tokens
    answer_tids = {}
    answer_vecs = {}
    for country, info in all_facts.items():
        tids = tokenizer.encode(info['answer'])
        if tids:
            answer_tids[country] = tids[0]
            answer_vecs[country] = decode_lm_row(engine, tids[0])

    # Project answer vocab vectors into SVD output basis (Vt_o)
    # Vt_o is (128, 3584), so Vt_o @ answer_vec gives 128-d projection
    print(f"\n  Answer token alignment with SVD output basis:")
    print(f"  {'Country':>12s}  {'||proj||':>8s}  {'||full||':>8s}  {'ratio':>8s}  top3_dims")
    print("  " + "─" * 65)
    for country in countries[:12]:
        if country not in answer_vecs:
            continue
        a_vec = answer_vecs[country]
        proj = Vt_o @ a_vec  # 128-d projection
        proj_norm = np.linalg.norm(proj)
        full_norm = np.linalg.norm(a_vec)
        ratio = proj_norm / full_norm
        # Which SVD dims have most answer energy?
        proj_energy = proj**2
        top3 = np.argsort(proj_energy)[-3:][::-1]
        print(f"  {country:>12s}  {proj_norm:8.3f}  {full_norm:8.3f}  {ratio:8.4f}  "
              f"d{top3[0]}({proj_energy[top3[0]]:.2f}), "
              f"d{top3[1]}({proj_energy[top3[1]]:.2f}), "
              f"d{top3[2]}({proj_energy[top3[2]]:.2f})")

    # How much of the answer vocabulary space is captured by top-k SVD dims?
    print(f"\n  Cumulative answer energy captured by SVD dimensions:")
    all_answer_matrix = np.array([answer_vecs[c] for c in countries if c in answer_vecs])
    # Project all answers into SVD basis
    all_proj = all_answer_matrix @ Vt_o.T  # (N, 128)
    # Energy per SVD dim across all answers
    ans_energy_per_dim = np.mean(all_proj**2, axis=0)
    total_ans_energy = np.mean(np.sum(all_proj**2, axis=1))
    total_ans_full = np.mean(np.sum(all_answer_matrix**2, axis=1))

    cum_ans_energy = np.cumsum(np.sort(ans_energy_per_dim)[::-1])
    print(f"  Total answer energy in SVD basis: {total_ans_energy:.3f} "
          f"({total_ans_energy/total_ans_full*100:.1f}% of full)")
    for k in [1, 5, 10, 20, 30, 40, 50, 66, 80, 100, 128]:
        pct = cum_ans_energy[min(k-1, 127)] / total_ans_energy * 100
        print(f"  Top {k:3d} SVD dims: {pct:6.1f}% of answer energy in SVD basis")

    # ══════════════════════════════════════════════════════════════════
    # PART D: Cross-Head Aperture Comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part D: Cross-Head Aperture — Is 66 Universal?")
    print("─" * 72)

    # Decode L22 and L23 weights for all heads
    W_v_22, b_v_22, W_o_22 = predecode_layer_weights(engine, 22)
    
    print(f"\n  {'Layer':>5s}  {'Head':>4s}  {'Rank50':>6s}  {'Rank70':>6s}  {'Rank80':>6s}  "
          f"{'Rank90':>6s}  {'Rank95':>6s}  {'Rank99':>6s}  {'S0/S1':>8s}  {'CV':>6s}")
    print("  " + "─" * 75)

    results_d = {}
    for li, (W_v, b_v, W_o) in [(22, (W_v_22, b_v_22, W_o_22)), 
                                  (23, (W_v_23, b_v_23, W_o_23))]:
        for hi in range(28):
            W_v_h, b_v_h, W_o_h = get_head_matrices(W_v, b_v, W_o, hi)
            W_o_h_T = W_o_h.T  # 128 x 3584
            _, S_h, _ = np.linalg.svd(W_o_h_T, full_matrices=False)
            
            en = np.cumsum(S_h**2) / np.sum(S_h**2)
            ranks = {}
            for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
                ranks[thr] = int(np.searchsorted(en, thr) + 1)
            
            ratio = float(S_h[0] / S_h[1]) if S_h[1] > 0 else float('inf')
            cv = float(np.std(S_h) / np.mean(S_h))
            
            results_d[f"L{li}H{hi}"] = {
                'ranks': ranks, 'ratio': ratio, 'cv': cv
            }
            
            # Only print interesting heads
            if hi in [6, 15, 19] or li == 23:
                marker = " <<<" if (li == 23 and hi == 6) else ""
                print(f"  L{li:2d}  H{hi:2d}  {ranks[0.5]:6d}  {ranks[0.7]:6d}  "
                      f"{ranks[0.8]:6d}  {ranks[0.9]:6d}  {ranks[0.95]:6d}  "
                      f"{ranks[0.99]:6d}  {ratio:8.3f}  {cv:6.3f}{marker}")

    # Summary statistics
    l23_ranks90 = [results_d[f"L23H{h}"]['ranks'][0.9] for h in range(28)]
    l22_ranks90 = [results_d[f"L22H{h}"]['ranks'][0.9] for h in range(28)]
    print(f"\n  L22 rank@90% energy: mean={np.mean(l22_ranks90):.1f}, "
          f"std={np.std(l22_ranks90):.1f}, "
          f"range=[{np.min(l22_ranks90)}, {np.max(l22_ranks90)}]")
    print(f"  L23 rank@90% energy: mean={np.mean(l23_ranks90):.1f}, "
          f"std={np.std(l23_ranks90):.1f}, "
          f"range=[{np.min(l23_ranks90)}, {np.max(l23_ranks90)}]")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Phase transition analysis
    print("\n  Rank truncation phase transition:")
    for k, v in results_a.items():
        mr = v['mean_rank_6']
        status = "✓" if mr < 50 else ("~" if mr < 500 else "✗")
        print(f"    rank={k:3d} ({v['energy_pct']:5.1f}%): mean_rank={mr:8.1f} {status}")

    # Save results
    out = {
        'part_a': {str(k): v for k, v in results_a.items()},
        'part_b': {
            'svd_singular_values': S_o.tolist(),
            'dim_variance': dim_variance.tolist(),
            'dim_snr': snr.tolist(),
        },
        'part_d': results_d,
        'singular_values_inner': S_inner.tolist(),
        'singular_values_output': S_o.tolist(),
    }
    out_path = 'experiments/model_reverse_engineering_v2/results/phase10z24_why_66.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
