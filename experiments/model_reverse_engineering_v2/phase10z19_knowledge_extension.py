"""
Phase 10z19: Knowledge Extension — Navigating TruthSpace
==========================================================

Can the geometric structure of known facts predict facts the model
has never seen? Is knowledge relative or absolute?

Part A: Baseline — what does the model predict for obscure capitals?
Part B: Manifold prediction — use 8 known capitals to predict test capitals
Part C: Navigation test — is country→capital a consistent direction?
Part D: Cross-manifold — do different fact types share structure?
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


def extract_all_layer_attn_and_hidden(engine, prompt_ids):
    """Run forward pass, return per-layer attention outputs and final hidden."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    per_layer_attn = []

    for layer in engine.layers:
        attn = layer.attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        heads_per_kv = num_heads // num_kv_heads
        head_dim = attn.head_dim
        seq_len = h.shape[1]

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

        per_layer_attn.append(attn_output[0, -1, :].copy())

        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    logits = get_logits(engine, h)
    return per_layer_attn, logits, h


def run_with_deltas(engine, prompt_ids, deltas, delta_layers):
    """Forward pass with deltas added to attention at specified layers."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]

    for layer in engine.layers:
        li = layer.layer_idx
        attn = layer.attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        heads_per_kv = num_heads // num_kv_heads
        head_dim = attn.head_dim
        seq_len = h.shape[1]

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

        if li in delta_layers:
            attn_output[0, -1, :] += deltas[li]

        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    return get_logits(engine, h)


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


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    all_layers = set(range(n_layers))
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z19: KNOWLEDGE EXTENSION — NAVIGATING TRUTHSPACE")
    print("=" * 72)

    # ── Training manifold: 8 known capitals ──
    train_facts = {
        'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
        'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
        'Brazil':  {'prompt': 'The capital of Brazil is',  'answer': ' Brasilia'},
        'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
        'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
        'Canada':  {'prompt': 'The capital of Canada is',  'answer': ' Ottawa'},
    }

    # ── Test facts: varying difficulty ──
    test_facts = {
        # Tricky (common misconception)
        'Australia':   {'prompt': 'The capital of Australia is',   'answer': ' Canberra'},
        'Turkey':      {'prompt': 'The capital of Turkey is',      'answer': ' Ankara'},
        'Switzerland': {'prompt': 'The capital of Switzerland is', 'answer': ' Bern'},
        # Medium difficulty
        'Poland':      {'prompt': 'The capital of Poland is',      'answer': ' Warsaw'},
        'Thailand':    {'prompt': 'The capital of Thailand is',    'answer': ' Bangkok'},
        'Nigeria':     {'prompt': 'The capital of Nigeria is',     'answer': ' Abuja'},
        'Vietnam':     {'prompt': 'The capital of Vietnam is',     'answer': ' Hanoi'},
        # Obscure
        'Myanmar':     {'prompt': 'The capital of Myanmar is',     'answer': ' Nay'},  # Naypyidaw
        'Palau':       {'prompt': 'The capital of Palau is',       'answer': ' Ng'},    # Ngerulmud
        'Tuvalu':      {'prompt': 'The capital of Tuvalu is',      'answer': ' Fun'},   # Funafuti
        'Bhutan':      {'prompt': 'The capital of Bhutan is',      'answer': ' Th'},    # Thimphu
    }

    # ══════════════════════════════════════════════════════════════════
    # PART A: BASELINE — what does the model predict?
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part A: Baseline predictions")
    print("─" * 72)

    # Extract training fact data
    train_data = {}
    for name, fact in train_facts.items():
        t1 = time.time()
        p_ids = tokenizer.encode(fact['prompt'])
        attn_list, logits, h = extract_all_layer_attn_and_hidden(engine, p_ids)
        rank, logit = get_rank(logits, fact['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits, 5)
        train_data[name] = {
            'attn_list': attn_list,
            'logits': logits,
            'rank': rank,
        }
        print(f"  Train: {name:15s} → '{fact['answer']}' rank={rank}  "
              f"top3=[{', '.join(t[0] for t,_ in zip(top5[:3], range(3)))}]  "
              f"({time.time()-t1:.1f}s)", flush=True)

    # Baseline for test facts
    test_data = {}
    for name, fact in test_facts.items():
        t1 = time.time()
        p_ids = tokenizer.encode(fact['prompt'])
        attn_list, logits, h = extract_all_layer_attn_and_hidden(engine, p_ids)
        rank, logit = get_rank(logits, fact['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits, 5)
        test_data[name] = {
            'attn_list': attn_list,
            'logits': logits,
            'rank': rank,
            'top5': top5,
        }
        print(f"  Test:  {name:15s} → '{fact['answer']}' rank={rank}  "
              f"top5=[{', '.join(f'{t[0]}' for t in top5)}]  "
              f"({time.time()-t1:.1f}s)", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART B: MANIFOLD PREDICTION
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part B: Manifold prediction of test capitals")
    print("─" * 72)

    # Get answer LM head rows for all train facts
    train_answer_dirs = {}
    for name, fact in train_facts.items():
        tid = tokenizer.encode(fact['answer'])[0]
        train_answer_dirs[name] = decode_lm_row(engine, tid)

    # Compute mean answer direction (generic "capital city" direction)
    all_answer_vecs = np.array([train_answer_dirs[n] for n in train_facts])
    mean_answer_dir = np.mean(all_answer_vecs, axis=0)
    mean_answer_dir_hat = mean_answer_dir / np.linalg.norm(mean_answer_dir)

    train_names = list(train_facts.keys())

    # For each test fact, try manifold projection and structure-only
    print(f"\n  {'Country':15s} {'Baseline':>10} {'Manifold':>10} {'Structure':>10} "
          f"{'Manifold top1':>15} {'Structure top1':>15}")

    results = {}

    for test_name, test_fact in test_facts.items():
        test_attn = test_data[test_name]['attn_list']
        test_answer = test_fact['answer']

        # Get test answer direction (for manifold_proj — uses answer knowledge)
        test_answer_tid = tokenizer.encode(test_answer)
        if test_answer_tid:
            test_answer_dir = decode_lm_row(engine, test_answer_tid[0])
            test_answer_hat = test_answer_dir / np.linalg.norm(test_answer_dir)
        else:
            test_answer_hat = mean_answer_dir_hat

        # Strategy 1: Manifold projection (uses answer direction)
        deltas_manifold = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train_names), test_attn[li].shape[0]))
            for i, tn in enumerate(train_names):
                train_deltas[i] = train_data[tn]['attn_list'][li] - test_attn[li]

            U, S, Vt = np.linalg.svd(train_deltas, full_matrices=False)
            cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
            k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))
            basis = Vt[:k]
            coeffs = basis @ test_answer_hat
            mean_delta_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj_delta = (basis.T @ coeffs)
            proj_delta = proj_delta / (np.linalg.norm(proj_delta) + 1e-10) * mean_delta_norm
            deltas_manifold[li] = proj_delta

        # Strategy 2: Structure-only (NO answer direction — uses generic capital dir)
        deltas_structure = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train_names), test_attn[li].shape[0]))
            for i, tn in enumerate(train_names):
                train_deltas[i] = train_data[tn]['attn_list'][li] - test_attn[li]

            U, S, Vt = np.linalg.svd(train_deltas, full_matrices=False)
            cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
            k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))
            basis = Vt[:k]
            coeffs = basis @ mean_answer_dir_hat
            mean_delta_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj_delta = (basis.T @ coeffs)
            proj_delta = proj_delta / (np.linalg.norm(proj_delta) + 1e-10) * mean_delta_norm
            deltas_structure[li] = proj_delta

        # Run both strategies
        p_ids = tokenizer.encode(test_fact['prompt'])

        logits_manifold = run_with_deltas(engine, p_ids, deltas_manifold, all_layers)
        rank_manifold, _ = get_rank(logits_manifold, test_answer, tokenizer)
        top1_manifold = tokenizer.decode([int(np.argmax(logits_manifold))])

        logits_structure = run_with_deltas(engine, p_ids, deltas_structure, all_layers)
        rank_structure, _ = get_rank(logits_structure, test_answer, tokenizer)
        top1_structure = tokenizer.decode([int(np.argmax(logits_structure))])

        baseline_rank = test_data[test_name]['rank']

        # Get full top-5 for structure prediction
        top5_structure = top_k_tokens(tokenizer, logits_structure, 5)
        top5_manifold = top_k_tokens(tokenizer, logits_manifold, 5)

        results[test_name] = {
            'baseline_rank': baseline_rank,
            'manifold_rank': rank_manifold,
            'structure_rank': rank_structure,
            'top1_manifold': top1_manifold,
            'top1_structure': top1_structure,
            'top5_structure': top5_structure,
            'top5_manifold': top5_manifold,
        }

        print(f"  {test_name:15s} {baseline_rank:>10} {rank_manifold:>10} {rank_structure:>10} "
              f"{top1_manifold:>15} {top1_structure:>15}")

    # Detailed output for each test fact
    print(f"\n  Detailed results:")
    for test_name in test_facts:
        r = results[test_name]
        print(f"\n  {test_name} (answer: '{test_facts[test_name]['answer']}')")
        print(f"    Baseline rank: {r['baseline_rank']}")
        print(f"    Manifold rank: {r['manifold_rank']}  "
              f"top5: {[t[0] for t in r['top5_manifold']]}")
        print(f"    Structure rank: {r['structure_rank']}  "
              f"top5: {[t[0] for t in r['top5_structure']]}")

    # ══════════════════════════════════════════════════════════════════
    # PART C: NAVIGATION TEST — is country→capital a consistent direction?
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part C: Navigation — country→capital displacement vectors")
    print("─" * 72)

    # For each training fact, compute the displacement:
    #   delta_L = attn_L(country) - mean(attn_L(all_others))
    # This gives us the fact-specific direction at each layer.
    # Then check: are these directions consistent across facts?

    # Compute per-fact, per-layer displacement from the grand mean
    grand_mean_attn = {}
    for li in range(n_layers):
        grand_mean_attn[li] = np.mean(
            [train_data[n]['attn_list'][li] for n in train_names], axis=0)

    # For each pair of training facts, compute the "displacement direction"
    # (attn[fact] - grand_mean) and check cosine similarity
    displacements = {}
    for name in train_names:
        disp = []
        for li in range(n_layers):
            d = train_data[name]['attn_list'][li] - grand_mean_attn[li]
            disp.append(d)
        displacements[name] = disp

    # Compute answer-direction displacement: W_lm[answer] for each fact
    # Check if the attention displacement aligns with the answer direction
    print(f"\n  Alignment: attention displacement vs answer direction")
    print(f"  (cosine of concat displacement with W_lm[answer])")

    for name in train_names:
        # Concat displacements at key layers (22, 23, 27)
        key_layers = [22, 23, 27]
        disp_concat = np.concatenate([displacements[name][li] for li in key_layers])
        disp_hat = disp_concat / (np.linalg.norm(disp_concat) + 1e-10)

        # Answer direction (tile to match concat length)
        w = train_answer_dirs[name]
        w_tiled = np.tile(w, len(key_layers))
        w_hat = w_tiled / (np.linalg.norm(w_tiled) + 1e-10)

        cos = float(np.dot(disp_hat, w_hat))
        print(f"    {name:12s}: cos(disp, W_lm[answer]) = {cos:.4f}")

    # Now the key test: for each training fact, compute the
    # "country→capital direction" at each layer, then apply it to
    # test facts and see if it predicts the right capital.
    #
    # Direction: the per-layer difference between what the model
    # actually computes for this fact vs the grand mean.
    # If this direction is consistent, applying it to a new country
    # should point toward that country's capital.

    print(f"\n  Cross-fact direction consistency:")
    print(f"  (cosine between displacement vectors at L23)")

    for i, n1 in enumerate(train_names):
        for j, n2 in enumerate(train_names):
            if j > i:
                d1 = displacements[n1][23]
                d2 = displacements[n2][23]
                cos = float(np.dot(d1, d2) /
                           (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))
                if abs(cos) > 0.2:
                    print(f"    {n1:10s} × {n2:10s}: cos = {cos:+.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PART D: CROSS-MANIFOLD TEST — different fact types
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part D: Cross-manifold — do different fact types share structure?")
    print("─" * 72)

    # Extract attention for a DIFFERENT fact type: languages
    language_facts = {
        'France':  {'prompt': 'The official language of France is',  'answer': ' French'},
        'Japan':   {'prompt': 'The official language of Japan is',   'answer': ' Japanese'},
        'Germany': {'prompt': 'The official language of Germany is', 'answer': ' German'},
        'Italy':   {'prompt': 'The official language of Italy is',   'answer': ' Italian'},
        'Brazil':  {'prompt': 'The official language of Brazil is',  'answer': ' Portuguese'},
        'Egypt':   {'prompt': 'The official language of Egypt is',   'answer': ' Arabic'},
        'Spain':   {'prompt': 'The official language of Spain is',   'answer': ' Spanish'},
    }

    lang_data = {}
    for name, fact in language_facts.items():
        t1 = time.time()
        p_ids = tokenizer.encode(fact['prompt'])
        attn_list, logits, h = extract_all_layer_attn_and_hidden(engine, p_ids)
        rank, logit = get_rank(logits, fact['answer'], tokenizer)
        lang_data[name] = {
            'attn_list': attn_list,
            'rank': rank,
        }
        print(f"  Language: {name:12s} → '{fact['answer']}' rank={rank}  "
              f"({time.time()-t1:.1f}s)", flush=True)

    # Cross-manifold consistency: for each country, compare its
    # displacement in the capital manifold vs the language manifold
    lang_grand_mean = {}
    lang_names = list(language_facts.keys())
    for li in range(n_layers):
        lang_grand_mean[li] = np.mean(
            [lang_data[n]['attn_list'][li] for n in lang_names], axis=0)

    lang_displacements = {}
    for name in lang_names:
        disp = []
        for li in range(n_layers):
            d = lang_data[name]['attn_list'][li] - lang_grand_mean[li]
            disp.append(d)
        lang_displacements[name] = disp

    print(f"\n  Cross-manifold consistency (capital vs language displacement):")
    print(f"  (cosine at L23 — same country, different fact type)")

    shared_countries = [n for n in train_names if n in lang_names]
    cross_cosines = []
    for name in shared_countries:
        d_cap = displacements[name][23]
        d_lang = lang_displacements[name][23]
        cos = float(np.dot(d_cap, d_lang) /
                    (np.linalg.norm(d_cap) * np.linalg.norm(d_lang) + 1e-10))
        cross_cosines.append(cos)
        print(f"    {name:12s}: cos(capital, language) = {cos:+.4f}")

    mean_cross = np.mean(cross_cosines) if cross_cosines else 0
    print(f"\n    Mean cross-manifold cosine: {mean_cross:.4f}")
    if mean_cross > 0.5:
        print(f"    → STRONG alignment: country identity is shared across fact types")
        print(f"    → Evidence for TRUTHSPACE (unified geometric structure)")
    elif mean_cross > 0.2:
        print(f"    → MODERATE alignment: partial structure sharing")
    else:
        print(f"    → WEAK alignment: fact types are largely independent")
        print(f"    → Evidence for RELATIVE knowledge (separate manifolds)")

    # Concat across key layers for a stronger signal
    print(f"\n  Cross-manifold consistency (concat of L22+L23+L27):")
    key_layers = [22, 23, 27]
    cross_cosines_concat = []
    for name in shared_countries:
        d_cap = np.concatenate([displacements[name][li] for li in key_layers])
        d_lang = np.concatenate([lang_displacements[name][li] for li in key_layers])
        cos = float(np.dot(d_cap, d_lang) /
                    (np.linalg.norm(d_cap) * np.linalg.norm(d_lang) + 1e-10))
        cross_cosines_concat.append(cos)
        print(f"    {name:12s}: cos(capital, language) = {cos:+.4f}")

    mean_cross_c = np.mean(cross_cosines_concat) if cross_cosines_concat else 0
    print(f"\n    Mean cross-manifold cosine (concat): {mean_cross_c:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Part B summary table
    print(f"\n  Part B: Manifold prediction results")
    print(f"  {'Country':15s} {'Baseline':>10} {'Manifold':>10} {'Structure':>10} {'Verdict':>10}")
    for test_name in test_facts:
        r = results[test_name]
        bl = r['baseline_rank']
        mp = r['manifold_rank']
        sp = r['structure_rank']
        if sp < bl:
            verdict = "IMPROVED"
        elif sp == bl:
            verdict = "same"
        else:
            verdict = "worse"
        print(f"  {test_name:15s} {bl:>10} {mp:>10} {sp:>10} {verdict:>10}")

    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z19_knowledge_extension.json')
    save_data = {
        'train_facts': {n: f for n, f in train_facts.items()},
        'test_facts': {n: f for n, f in test_facts.items()},
        'results': {n: {k: v for k, v in r.items()
                       if k not in ('top5_structure', 'top5_manifold')}
                   for n, r in results.items()},
        'cross_manifold_cosine_L23': mean_cross,
        'cross_manifold_cosine_concat': mean_cross_c,
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
