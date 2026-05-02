"""
Phase 10z21: V·W_o Binding Extraction
======================================

F121 showed triangulation fails because entity position alone can't
determine the specific answer (the binding problem). But the binding
France→Paris MUST live somewhere — and from F40 we know L23 Head 6
is a geometric selector that picks the country token.

Hypothesis: V·W_o applied to the entity's hidden state IS the binding.
The weights directly encode: "if you select France, output Paris."

Plan:
  Part A: For each country token, compute per-head V·W_o at key layers.
          Does it point toward the answer (W_lm[answer])?

  Part B: Sum the V·W_o contributions across heads, weighted by
          attention. Does the aggregate point toward the answer?

  Part C: The binding test — for each country, read the answer
          DIRECTLY from V·W_o without knowing it. Feed the V·W_o
          output through the LM head. Does the correct answer rank high?

  Part D: Cross-country test — does V·W_o at the France position
          give Paris, and at the Japan position give Tokyo, using
          the SAME weights? This proves the binding is positional.
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
from phi_geometric.inference.phi_integer import phi_to_float

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def get_logits(engine, hidden_3d):
    normed = rms_norm(hidden_3d, engine.final_norm_weight)
    logits = engine.lm_head(normed)
    return logits[0, -1, :]


def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    rank = int(np.sum(logits > logits[tid]))
    return rank, float(logits[tid])


def decode_lm_row(engine, tid):
    s = engine.lm_head.weight.signs[tid:tid+1, :]
    e = engine.lm_head.weight.exponents[tid:tid+1, :]
    return phi_to_float(s, e)[0]


def top_k_tokens(tokenizer, logits, k=10):
    top_idx = np.argsort(logits)[-k:][::-1]
    return [(tokenizer.decode([int(i)]), float(logits[i])) for i in top_idx]


def decode_weight_slice(W, row_start, row_end):
    """Decode a slice of rows from a PhiEncoded weight matrix."""
    s = W.signs[row_start:row_end, :]
    e = W.exponents[row_start:row_end, :]
    return phi_to_float(s, e)


def decode_weight_cols(W, col_start, col_end):
    """Decode specific columns from a PhiEncoded weight matrix."""
    s = W.signs[:, col_start:col_end]
    e = W.exponents[:, col_start:col_end]
    return phi_to_float(s, e)


def full_forward_capture(engine, prompt_ids):
    """Run full forward pass, capturing per-layer hidden states, attention
    weights, and the position of the last content token."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    seq_len = h.shape[1]

    layer_data = []

    for layer in engine.layers:
        attn = layer.attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        heads_per_kv = num_heads // num_kv_heads
        head_dim = attn.head_dim

        normed = rms_norm(h, attn.norm_weight)

        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        V = phi_linear(attn.W_v, normed, attn.b_v)

        Q = Q.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)

        K_exp = np.repeat(K, heads_per_kv, axis=1)
        V_exp = np.repeat(V, heads_per_kv, axis=1)

        scores = np.einsum('bhqd,bhkd->bhqk', Q, K_exp) * attn.scale
        if seq_len > 1:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask
        weights = phi_softmax(scores, axis=-1)

        attn_output = np.einsum('bhqk,bhkd->bhqd', weights, V_exp)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        attn_output = phi_linear(attn.W_o, attn_output)

        layer_data.append({
            'normed': normed[0].copy(),         # (seq, hidden_dim)
            'V': V[0].copy(),                    # (kv_heads, seq, head_dim)
            'V_exp': V_exp[0].copy(),            # (num_heads, seq, head_dim)
            'attn_weights': weights[0].copy(),   # (num_heads, seq, seq)
            'h_pre': h[0].copy(),                # (seq, hidden_dim)
        })

        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    logits = get_logits(engine, h)
    return layer_data, logits


def predecode_layer_weights(engine, layer_idx):
    """Pre-decode V and W_o weight matrices for a layer (do once, reuse)."""
    attn = engine.layers[layer_idx].attention
    # Full W_v: (kv_heads*head_dim, hidden_dim) = (512, 3584)
    W_v_float = phi_to_float(attn.W_v.signs, attn.W_v.exponents)
    b_v = attn.b_v.copy()
    # Full W_o: (hidden_dim, num_heads*head_dim) = (3584, 3584)
    W_o_float = phi_to_float(attn.W_o.signs, attn.W_o.exponents)
    return W_v_float, b_v, W_o_float


def compute_vwo_binding_fast(W_v_float, b_v, W_o_float, normed_at_pos, head_idx,
                              num_heads=28, num_kv_heads=4, head_dim=128):
    """
    Compute V·W_o for a specific position and head using pre-decoded weights.
    Returns: (3584,) vector — the hidden-space binding direction.
    """
    heads_per_kv = num_heads // num_kv_heads
    kv_head = head_idx // heads_per_kv

    kv_start = kv_head * head_dim
    kv_end = (kv_head + 1) * head_dim
    v = normed_at_pos @ W_v_float[kv_start:kv_end, :].T + b_v[kv_start:kv_end]

    h_start = head_idx * head_dim
    h_end = (head_idx + 1) * head_dim
    binding = v @ W_o_float[:, h_start:h_end].T  # (3584,)
    return binding


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    num_heads = engine.num_heads  # 28
    head_dim = engine.head_dim    # 128
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z21: V·W_o BINDING EXTRACTION")
    print("=" * 72)

    facts = {
        'France': {'prompt': 'The capital of France is', 'answer': ' Paris'},
        'Japan': {'prompt': 'The capital of Japan is', 'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy': {'prompt': 'The capital of Italy is', 'answer': ' Rome'},
        'Brazil': {'prompt': 'The capital of Brazil is', 'answer': ' Brasilia'},
        'Egypt': {'prompt': 'The capital of Egypt is', 'answer': ' Cairo'},
        'Spain': {'prompt': 'The capital of Spain is', 'answer': ' Madrid'},
        'Canada': {'prompt': 'The capital of Canada is', 'answer': ' Ottawa'},
    }

    # Get answer directions from LM head
    answer_dirs = {}
    for c in facts:
        tid = tokenizer.encode(facts[c]['answer'])[0]
        answer_dirs[c] = decode_lm_row(engine, tid)
        answer_dirs[c] = answer_dirs[c] / np.linalg.norm(answer_dirs[c])

    KEY_LAYERS = [22, 23, 27]

    # Pre-decode weights for key layers (do this ONCE)
    print("\n  Pre-decoding V and W_o weights for key layers...", flush=True)
    decoded_weights = {}
    for li in KEY_LAYERS:
        t1 = time.time()
        W_v_f, b_v, W_o_f = predecode_layer_weights(engine, li)
        decoded_weights[li] = (W_v_f, b_v, W_o_f)
        print(f"    L{li}: decoded in {time.time()-t1:.1f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART A: Per-head V·W_o cosine with answer direction
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part A: Per-head V·W_o binding → answer cosine")
    print("─" * 72)

    results_a = {}

    for country in ['France', 'Japan', 'Germany', 'Italy']:
        t1 = time.time()
        prompt = facts[country]['prompt']
        p_ids = tokenizer.encode(prompt)

        # Find the country token position
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        country_pos = None
        for i, tok in enumerate(tokens):
            if country.lower() in tok.lower():
                country_pos = i
                break
        if country_pos is None:
            print(f"  WARNING: could not find {country} in tokens: {tokens}")
            continue

        print(f"\n  {country} (pos={country_pos}, tokens={tokens})", flush=True)

        layer_data, logits = full_forward_capture(engine, p_ids)
        rank_base, _ = get_rank(logits, facts[country]['answer'], tokenizer)
        print(f"    Baseline rank: {rank_base}", flush=True)

        country_results = {}
        for li in KEY_LAYERS:
            ld = layer_data[li]
            normed_country = ld['normed'][country_pos]  # (3584,)
            attn_to_country = ld['attn_weights'][:, -1, country_pos]  # (28,)
            W_v_f, b_v, W_o_f = decoded_weights[li]

            head_cosines = []
            head_bindings = []
            for hi in range(num_heads):
                binding = compute_vwo_binding_fast(
                    W_v_f, b_v, W_o_f, normed_country, hi)
                binding_hat = binding / (np.linalg.norm(binding) + 1e-10)
                cos = float(np.dot(binding_hat, answer_dirs[country]))
                attn_w = float(attn_to_country[hi])
                head_cosines.append((hi, cos, attn_w))
                head_bindings.append(binding)

            head_cosines.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"    L{li} top-5 heads by |cos(V·W_o, answer)|:")
            for hi, cos, attn_w in head_cosines[:5]:
                print(f"      H{hi:2d}: cos={cos:+.4f}, attn_weight={attn_w:.4f}")

            aggregate = sum(
                attn_to_country[hi] * head_bindings[hi]
                for hi in range(num_heads)
            )
            agg_hat = aggregate / (np.linalg.norm(aggregate) + 1e-10)
            agg_cos = float(np.dot(agg_hat, answer_dirs[country]))
            print(f"    L{li} aggregate (attn-weighted): cos={agg_cos:+.4f}")

            country_results[f'L{li}'] = {
                'top_heads': [(h, c, a) for h, c, a in head_cosines[:5]],
                'aggregate_cos': agg_cos,
            }

        results_a[country] = country_results
        print(f"    ({time.time()-t1:.1f}s)", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART B: Read the answer directly from V·W_o
    # Feed the aggregate V·W_o output through the LM head
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part B: Read answer directly from V·W_o (all countries)")
    print("─" * 72)

    results_b = {}

    for country in facts:
        t1 = time.time()
        prompt = facts[country]['prompt']
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([tid]) for tid in p_ids]

        country_pos = None
        for i, tok in enumerate(tokens):
            if country.lower() in tok.lower():
                country_pos = i
                break
        if country_pos is None:
            print(f"  WARNING: could not find {country} in tokens: {tokens}")
            continue

        layer_data, logits = full_forward_capture(engine, p_ids)
        rank_base, _ = get_rank(logits, facts[country]['answer'], tokenizer)

        best_rank = 999999
        best_layer = None

        for li in KEY_LAYERS:
            ld = layer_data[li]
            normed_country = ld['normed'][country_pos]
            attn_to_country = ld['attn_weights'][:, -1, country_pos]
            W_v_f, b_v, W_o_f = decoded_weights[li]

            aggregate = np.zeros(3584, dtype=np.float32)
            for hi in range(num_heads):
                binding = compute_vwo_binding_fast(
                    W_v_f, b_v, W_o_f, normed_country, hi)
                aggregate += attn_to_country[hi] * binding

            agg_3d = aggregate[np.newaxis, np.newaxis, :]
            binding_logits = get_logits(engine, agg_3d)

            rank_bind, logit_val = get_rank(binding_logits, facts[country]['answer'],
                                            tokenizer)
            top5 = top_k_tokens(tokenizer, binding_logits, 5)

            if rank_bind is not None and rank_bind < best_rank:
                best_rank = rank_bind
                best_layer = li

            print(f"  {country:12s} L{li}: bind_rank={rank_bind}, "
                  f"top5={[t[0] for t in top5]}")

        results_b[country] = {'best_rank': best_rank, 'best_layer': best_layer,
                               'baseline_rank': rank_base}
        print(f"    baseline={rank_base}, best_bind={best_rank} (L{best_layer}) "
              f" ({time.time()-t1:.1f}s)", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART C: Sum V·W_o across key layers
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part C: Multi-layer V·W_o aggregate")
    print("─" * 72)

    results_c = {}

    for country in facts:
        t1 = time.time()
        prompt = facts[country]['prompt']
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([tid]) for tid in p_ids]

        country_pos = None
        for i, tok in enumerate(tokens):
            if country.lower() in tok.lower():
                country_pos = i
                break
        if country_pos is None:
            continue

        layer_data, logits = full_forward_capture(engine, p_ids)

        multi_layer_agg = np.zeros(3584, dtype=np.float32)
        for li in KEY_LAYERS:
            ld = layer_data[li]
            normed_country = ld['normed'][country_pos]
            attn_to_country = ld['attn_weights'][:, -1, country_pos]
            W_v_f, b_v, W_o_f = decoded_weights[li]

            for hi in range(num_heads):
                binding = compute_vwo_binding_fast(
                    W_v_f, b_v, W_o_f, normed_country, hi)
                multi_layer_agg += attn_to_country[hi] * binding

        agg_3d = multi_layer_agg[np.newaxis, np.newaxis, :]
        ml_logits = get_logits(engine, agg_3d)
        rank_ml, _ = get_rank(ml_logits, facts[country]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, ml_logits, 5)

        print(f"  {country:12s}: rank={rank_ml}, top5={[t[0] for t in top5]}  "
              f"({time.time()-t1:.1f}s)", flush=True)
        results_c[country] = {'rank': rank_ml}

    # ══════════════════════════════════════════════════════════════════
    # PART D: Full attention-weighted V·W_o (all positions, key layers)
    # ══════════════════════════════════════════════════════════════════

    # Part D: Skip all-layer decode (too expensive). Instead, use a
    # vectorized approach: compute V·W_o for ALL positions at once,
    # then let attention weights select.

    print("\n" + "─" * 72)
    print("  Part D: Vectorized V·W_o — full attention-weighted binding")
    print("─" * 72)

    results_d = {}

    for country in facts:
        t1 = time.time()
        prompt = facts[country]['prompt']
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        seq_len = len(p_ids)

        layer_data, logits = full_forward_capture(engine, p_ids)
        rank_base, _ = get_rank(logits, facts[country]['answer'], tokenizer)

        # For each key layer, compute the FULL attention output from V·W_o
        # for ALL positions, weighted by attention from last token.
        # This is: Σ_pos Σ_head attn[last,pos,head] * V(normed[pos]) @ W_o_head
        multi_agg = np.zeros(3584, dtype=np.float32)
        for li in KEY_LAYERS:
            ld = layer_data[li]
            W_v_f, b_v, W_o_f = decoded_weights[li]
            attn_w = ld['attn_weights'][:, -1, :]  # (28, seq)

            for pos in range(seq_len):
                normed_pos = ld['normed'][pos]
                for hi in range(num_heads):
                    w = float(attn_w[hi, pos])
                    if abs(w) < 1e-6:
                        continue
                    binding = compute_vwo_binding_fast(
                        W_v_f, b_v, W_o_f, normed_pos, hi)
                    multi_agg += w * binding

        agg_3d = multi_agg[np.newaxis, np.newaxis, :]
        d_logits = get_logits(engine, agg_3d)
        rank_d, _ = get_rank(d_logits, facts[country]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, d_logits, 5)

        print(f"  {country:12s}: rank={rank_d} (base={rank_base}), "
              f"top5={[t[0] for t in top5]}  ({time.time()-t1:.1f}s)", flush=True)
        results_d[country] = {'rank': rank_d, 'baseline': rank_base}

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    print("\n  Part B — Direct binding read (best single layer):")
    for c in facts:
        if c in results_b:
            rb = results_b[c]
            print(f"    {c:12s}: baseline={rb['baseline_rank']}, "
                  f"bind={rb['best_rank']} (L{rb['best_layer']})")

    print("\n  Part C — Multi-layer key aggregate:")
    for c in facts:
        if c in results_c:
            print(f"    {c:12s}: rank={results_c[c]['rank']}")

    print("\n  Part D — All-layer aggregate:")
    for c in facts:
        if c in results_d:
            print(f"    {c:12s}: rank={results_d[c]['rank']}")

    elapsed = time.time() - t0
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z21_vwo_binding.json')
    results = {
        'part_a': {k: v for k, v in results_a.items()},
        'part_b': results_b,
        'part_c': results_c,
        'part_d': results_d,
        'total_time': elapsed,
    }
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
