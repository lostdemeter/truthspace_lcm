"""
Phase 10z23: Novel Entity Projection — Does M_h Generalize?
=============================================================

F123 showed M_h = W_v_h.T @ W_o_h.T is a near-isometric universal
transformation that maps entity hidden states to answers (rank 4-18).
The bottleneck for novel entities: does their hidden state project
correctly into M_h's 66-d fact subspace?

Plan:
  Part A: Extended countries — 12 countries NOT in original 8-country
          set. Apply M_h, check binding rank vs baseline.

  Part B: Obscure countries — Countries where baseline might be poor.
          Does M_h still extract an answer signal?

  Part C: M_h without attention — Apply M_h to entity hidden state
          directly, bypassing L23's attention routing. Does the
          transformation alone suffice?

  Part D: Cross-fact-type — Apply CAPITAL M_h to language prompts.
          Is M_h fact-type-specific or entity-specific?
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
        layer_data.append({
            'normed': normed[0].copy(),
            'attn_weights': weights[0].copy(),
            'h_pre': h[0].copy(),
        })
        h_post = h + ao
        mlp = layer.mlp
        nm = rms_norm(h_post, mlp.norm_weight)
        g, u = phi_linear(mlp.W_gate, nm), phi_linear(mlp.W_up, nm)
        h = h_post + phi_linear(mlp.W_down, phi_silu(g) * u)
    return layer_data, get_logits(engine, h)


def find_country_pos(tokens, country):
    """Find position of country token in tokenized prompt."""
    for i, t in enumerate(tokens):
        if country.lower() in t.lower():
            return i
    # Try partial match for multi-token countries
    country_lower = country.lower()
    for i, t in enumerate(tokens):
        t_clean = t.strip().lower()
        if len(t_clean) >= 3 and t_clean in country_lower:
            return i
    return None


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\n" + "=" * 72)
    print("  PHASE 10z23: NOVEL ENTITY PROJECTION")
    print("=" * 72)

    # Original 8 countries (reference)
    original = ['France', 'Japan', 'Germany', 'Italy', 'Brazil', 'Egypt', 'Spain', 'Canada']

    # Part A: Extended well-known countries
    extended_facts = {
        'China':     {'prompt': 'The capital of China is',     'answer': ' Beijing'},
        'Russia':    {'prompt': 'The capital of Russia is',    'answer': ' Moscow'},
        'India':     {'prompt': 'The capital of India is',     'answer': ' New'},
        'Australia': {'prompt': 'The capital of Australia is', 'answer': ' Canberra'},
        'Mexico':    {'prompt': 'The capital of Mexico is',    'answer': ' Mexico'},
        'Turkey':    {'prompt': 'The capital of Turkey is',    'answer': ' Ankara'},
        'Thailand':  {'prompt': 'The capital of Thailand is',  'answer': ' Bangkok'},
        'Poland':    {'prompt': 'The capital of Poland is',    'answer': ' Warsaw'},
        'Argentina': {'prompt': 'The capital of Argentina is', 'answer': ' Buenos'},
        'Sweden':    {'prompt': 'The capital of Sweden is',    'answer': ' Stockholm'},
        'Norway':    {'prompt': 'The capital of Norway is',    'answer': ' Oslo'},
        'Kenya':     {'prompt': 'The capital of Kenya is',     'answer': ' Nair'},
    }

    # Part B: Obscure countries
    obscure_facts = {
        'Bhutan':      {'prompt': 'The capital of Bhutan is',      'answer': ' Th'},
        'Latvia':      {'prompt': 'The capital of Latvia is',      'answer': ' Riga'},
        'Paraguay':    {'prompt': 'The capital of Paraguay is',    'answer': ' As'},
        'Madagascar':  {'prompt': 'The capital of Madagascar is',  'answer': ' Ant'},
        'Luxembourg':  {'prompt': 'The capital of Luxembourg is',  'answer': ' Luxembourg'},
    }

    # Part D: Cross-fact-type (language prompts, apply CAPITAL M_h)
    language_facts = {
        'France':  {'prompt': 'The language of France is',  'answer': ' French',  'capital': ' Paris'},
        'Japan':   {'prompt': 'The language of Japan is',   'answer': ' Japanese','capital': ' Tokyo'},
        'Germany': {'prompt': 'The language of Germany is', 'answer': ' German',  'capital': ' Berlin'},
        'Italy':   {'prompt': 'The language of Italy is',   'answer': ' Italian', 'capital': ' Rome'},
        'Spain':   {'prompt': 'The language of Spain is',   'answer': ' Spanish', 'capital': ' Madrid'},
        'Brazil':  {'prompt': 'The language of Brazil is',  'answer': ' Portug',  'capital': ' Bras'},
    }

    # Pre-decode key weights
    print("\n  Pre-decoding weights...", flush=True)
    KEY_LAYERS = [22, 23]
    decoded = {}
    for li in KEY_LAYERS:
        decoded[li] = predecode_layer_weights(engine, li)
    print("    Done", flush=True)

    # Get M_h for L23 H6 (primary) and L22 H15 (supporting)
    W_v_23, b_v_23, W_o_23 = decoded[23]
    W_v_h6, b_v_h6, W_o_h6 = get_head_matrices(W_v_23, b_v_23, W_o_23, 6)
    W_v_22, b_v_22, W_o_22 = decoded[22]
    W_v_h15, b_v_h15, W_o_h15 = get_head_matrices(W_v_22, b_v_22, W_o_22, 15)

    # ══════════════════════════════════════════════════════════════════
    # PART A: Extended Countries
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part A: Extended Countries (12 new)")
    print("─" * 72)

    results_a = {}
    for country, info in extended_facts.items():
        p_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = find_country_pos(tokens, country)

        if cpos is None:
            print(f"\n  {country}: SKIP (country token not found in {tokens})")
            continue

        layer_data, baseline_logits = full_forward_capture(engine, p_ids)
        baseline_rank, _ = get_rank(baseline_logits, info['answer'], tokenizer)

        # M_h binding at L23 H6
        normed_23 = layer_data[23]['normed'][cpos]
        binding_23 = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_23)
        bind_3d = binding_23[np.newaxis, np.newaxis, :].astype(np.float32)
        bind_logits = get_logits(engine, bind_3d)
        bind_rank, _ = get_rank(bind_logits, info['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, bind_logits, 5)

        # Multi-layer: L22 H15 + L23 H6 aggregate
        normed_22 = layer_data[22]['normed'][cpos]
        binding_22 = compute_binding(W_v_h15, b_v_h15, W_o_h15, normed_22)
        # Weight by attention from last token
        attn_23 = float(layer_data[23]['attn_weights'][6, -1, cpos])
        attn_22 = float(layer_data[22]['attn_weights'][15, -1, cpos])
        agg = attn_23 * binding_23 + attn_22 * binding_22
        agg_3d = agg[np.newaxis, np.newaxis, :].astype(np.float32)
        agg_logits = get_logits(engine, agg_3d)
        agg_rank, _ = get_rank(agg_logits, info['answer'], tokenizer)

        print(f"\n  {country:12s}: baseline={baseline_rank:3d}, "
              f"M_h(L23H6)={bind_rank:4d}, "
              f"agg(L22+L23)={agg_rank:4d}")
        print(f"    tokens: {tokens}, cpos={cpos}")
        print(f"    top5: {[t[0] for t in top5]}")

        results_a[country] = {
            'baseline': baseline_rank, 'bind_L23H6': bind_rank,
            'agg': agg_rank, 'top5': [t[0] for t in top5],
        }

    # ══════════════════════════════════════════════════════════════════
    # PART B: Obscure Countries
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part B: Obscure Countries")
    print("─" * 72)

    results_b = {}
    for country, info in obscure_facts.items():
        p_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = find_country_pos(tokens, country)

        if cpos is None:
            print(f"\n  {country}: SKIP (not found in {tokens})")
            continue

        layer_data, baseline_logits = full_forward_capture(engine, p_ids)
        baseline_rank, _ = get_rank(baseline_logits, info['answer'], tokenizer)

        normed_23 = layer_data[23]['normed'][cpos]
        binding_23 = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_23)
        bind_3d = binding_23[np.newaxis, np.newaxis, :].astype(np.float32)
        bind_logits = get_logits(engine, bind_3d)
        bind_rank, _ = get_rank(bind_logits, info['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, bind_logits, 5)

        print(f"\n  {country:12s}: baseline={baseline_rank:3d}, "
              f"M_h(L23H6)={bind_rank:5d}")
        print(f"    tokens: {tokens}, cpos={cpos}")
        print(f"    top5: {[t[0] for t in top5]}")

        results_b[country] = {
            'baseline': baseline_rank, 'bind_L23H6': bind_rank,
            'top5': [t[0] for t in top5],
        }

    # ══════════════════════════════════════════════════════════════════
    # PART C: M_h Without Attention (direct application)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part C: M_h Without Attention Routing")
    print("─" * 72)
    print("  (Apply M_h to entity hidden state WITHOUT attention selection)")

    # For the original 8 countries, compare:
    # 1. M_h applied at the attention-selected position (what we've been doing)
    # 2. M_h applied to the raw hidden state at L23 entry (h_pre, not normed by attn layer)
    original_facts = {
        'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
        'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
        'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
        'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
    }

    results_c = {}
    for country, info in original_facts.items():
        p_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = find_country_pos(tokens, country)

        layer_data, baseline_logits = full_forward_capture(engine, p_ids)
        baseline_rank, _ = get_rank(baseline_logits, info['answer'], tokenizer)

        # Method 1: normed hidden state (standard — what attention sees)
        normed_23 = layer_data[23]['normed'][cpos]
        bind_normed = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_23)
        b1_3d = bind_normed[np.newaxis, np.newaxis, :].astype(np.float32)
        rank_normed, _ = get_rank(get_logits(engine, b1_3d), info['answer'], tokenizer)

        # Method 2: raw pre-attention hidden state (h_pre at L23)
        h_pre_23 = layer_data[23]['h_pre'][cpos]
        # Need to normalize it with the attention norm to be comparable
        attn_norm_w = engine.layers[23].attention.norm_weight
        h_pre_normed = rms_norm(h_pre_23[np.newaxis, np.newaxis, :], attn_norm_w)[0, 0]
        bind_raw = compute_binding(W_v_h6, b_v_h6, W_o_h6, h_pre_normed)
        b2_3d = bind_raw[np.newaxis, np.newaxis, :].astype(np.float32)
        rank_raw, _ = get_rank(get_logits(engine, b2_3d), info['answer'], tokenizer)

        # Method 3: Apply M_h at LAST token position instead of country position
        normed_last = layer_data[23]['normed'][-1]
        bind_last = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_last)
        b3_3d = bind_last[np.newaxis, np.newaxis, :].astype(np.float32)
        rank_last, _ = get_rank(get_logits(engine, b3_3d), info['answer'], tokenizer)

        print(f"\n  {country:12s}: baseline={baseline_rank:3d}, "
              f"normed@cpos={rank_normed:4d}, "
              f"raw@cpos={rank_raw:4d}, "
              f"normed@last={rank_last:4d}")

        results_c[country] = {
            'baseline': baseline_rank, 'normed_cpos': rank_normed,
            'raw_cpos': rank_raw, 'normed_last': rank_last,
        }

    # ══════════════════════════════════════════════════════════════════
    # PART D: Cross-Fact-Type (capital M_h on language prompts)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part D: Cross-Fact-Type (capital M_h on language prompts)")
    print("─" * 72)

    results_d = {}
    for country, info in language_facts.items():
        p_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = find_country_pos(tokens, country)

        if cpos is None:
            print(f"\n  {country}: SKIP")
            continue

        layer_data, baseline_logits = full_forward_capture(engine, p_ids)
        baseline_rank_lang, _ = get_rank(baseline_logits, info['answer'], tokenizer)
        baseline_rank_cap, _ = get_rank(baseline_logits, info['capital'], tokenizer)

        # Apply CAPITAL M_h to language prompt's entity hidden state
        normed_23 = layer_data[23]['normed'][cpos]
        binding = compute_binding(W_v_h6, b_v_h6, W_o_h6, normed_23)
        b_3d = binding[np.newaxis, np.newaxis, :].astype(np.float32)
        bind_logits = get_logits(engine, b_3d)
        bind_rank_lang, _ = get_rank(bind_logits, info['answer'], tokenizer)
        bind_rank_cap, _ = get_rank(bind_logits, info['capital'], tokenizer)
        top5 = top_k_tokens(tokenizer, bind_logits, 5)

        print(f"\n  {country:12s}:")
        print(f"    baseline: language={baseline_rank_lang:3d}, capital={baseline_rank_cap:5d}")
        print(f"    cap_M_h:  language={bind_rank_lang:4d}, capital={bind_rank_cap:5d}")
        print(f"    top5: {[t[0] for t in top5]}")

        results_d[country] = {
            'baseline_lang': baseline_rank_lang,
            'baseline_cap': baseline_rank_cap,
            'mh_lang': bind_rank_lang,
            'mh_cap': bind_rank_cap,
            'top5': [t[0] for t in top5],
        }

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    print("\n  Part A — Extended Countries (M_h L23 H6):")
    for c, v in results_a.items():
        delta = "BETTER" if v['bind_L23H6'] < v['baseline'] * 10 else "WORSE"
        print(f"    {c:12s}: baseline={v['baseline']:3d}, M_h={v['bind_L23H6']:4d}")

    print("\n  Part B — Obscure Countries:")
    for c, v in results_b.items():
        print(f"    {c:12s}: baseline={v['baseline']:3d}, M_h={v['bind_L23H6']:5d}")

    print("\n  Part C — Position Comparison:")
    for c, v in results_c.items():
        print(f"    {c:12s}: normed@cpos={v['normed_cpos']:4d}, "
              f"raw@cpos={v['raw_cpos']:4d}, @last={v['normed_last']:4d}")

    print("\n  Part D — Cross-Fact-Type:")
    for c, v in results_d.items():
        print(f"    {c:12s}: cap_M_h→language={v['mh_lang']:4d}, "
              f"cap_M_h→capital={v['mh_cap']:5d}")

    # Save results
    out = {'part_a': results_a, 'part_b': results_b,
           'part_c': results_c, 'part_d': results_d}
    out_path = 'experiments/model_reverse_engineering_v2/results/phase10z23_novel_entity.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
