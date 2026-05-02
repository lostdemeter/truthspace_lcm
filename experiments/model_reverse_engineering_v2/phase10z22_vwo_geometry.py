"""
Phase 10z22: V·W_o Geometry — Is the Binding a Geometric Transformation?
=========================================================================

F122 showed V·W_o at the entity token directly encodes factual bindings.
M_h = W_v_h.T @ W_o_h.T is the SAME matrix for all entities — entity-
specificity comes from the input hidden state.

Questions:
  1. What kind of geometric transformation is M_h?
  2. Do entity hidden states map to answers in a structured way?
  3. Can we predict a novel entity's binding from known bindings?

Plan:
  Part A: M_h Geometry — SVD, effective rank, transformation type.
  Part B: Entity→Answer Structure — PCA of bindings, linearity test.
  Part C: Binding Transfer — Leave-one-out prediction.
  Part D: Binding Subspace — Project M_h onto LM head answer rows.
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

def top_k_tokens(tokenizer, logits, k=10):
    top_idx = np.argsort(logits)[-k:][::-1]
    return [(tokenizer.decode([int(i)]), float(logits[i])) for i in top_idx]

def predecode_layer_weights(engine, layer_idx):
    attn = engine.layers[layer_idx].attention
    W_v = phi_to_float(attn.W_v.signs, attn.W_v.exponents)
    W_o = phi_to_float(attn.W_o.signs, attn.W_o.exponents)
    return W_v, attn.b_v.copy(), W_o

def get_head_matrices(W_v, b_v, W_o, head_idx, hd=128, nh=28, nkv=4):
    kv = head_idx // (nh // nkv)
    W_v_h = W_v[kv*hd:(kv+1)*hd, :]       # (128, 3584)
    b_v_h = b_v[kv*hd:(kv+1)*hd]           # (128,)
    W_o_h = W_o[:, head_idx*hd:(head_idx+1)*hd]  # (3584, 128)
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
        layer_data.append({'normed': normed[0].copy(), 'attn_weights': weights[0].copy()})
        h_post = h + ao
        mlp = layer.mlp
        nm = rms_norm(h_post, mlp.norm_weight)
        g, u = phi_linear(mlp.W_gate, nm), phi_linear(mlp.W_up, nm)
        h = h_post + phi_linear(mlp.W_down, phi_silu(g) * u)
    return layer_data, get_logits(engine, h)


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\n" + "=" * 72)
    print("  PHASE 10z22: V·W_o GEOMETRY")
    print("=" * 72)

    facts = {
        'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
        'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
        'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
        'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
        'Brazil':  {'prompt': 'The capital of Brazil is',  'answer': ' Bras'},
        'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
        'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
        'Canada':  {'prompt': 'The capital of Canada is',  'answer': ' Ottawa'},
    }
    countries = list(facts.keys())

    answer_dirs = {}
    for c in facts:
        tid = tokenizer.encode(facts[c]['answer'])[0]
        row = decode_lm_row(engine, tid)
        answer_dirs[c] = row / np.linalg.norm(row)

    KEY_HEADS = {22: [15, 19], 23: [6, 4]}

    print("\n  Pre-decoding weights...", flush=True)
    decoded = {}
    for li in KEY_HEADS:
        decoded[li] = predecode_layer_weights(engine, li)
        print(f"    L{li}: done", flush=True)

    # Collect entity hidden states
    print("\n  Collecting entity hidden states...", flush=True)
    entity_normed = {li: {} for li in KEY_HEADS}
    entity_attn = {li: {} for li in KEY_HEADS}
    for country in countries:
        p_ids = tokenizer.encode(facts[country]['prompt'])
        tokens = [tokenizer.decode([tid]) for tid in p_ids]
        cpos = next(i for i, t in enumerate(tokens) if country.lower() in t.lower())
        ld, _ = full_forward_capture(engine, p_ids)
        for li in KEY_HEADS:
            entity_normed[li][country] = ld[li]['normed'][cpos].copy()
            entity_attn[li][country] = ld[li]['attn_weights'][:, -1, cpos].copy()
    print(f"    Done ({time.time()-t0:.1f}s)", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # PART A: M_h Geometry — SVD analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part A: M_h Geometry — SVD of V·W_o")
    print("─" * 72)

    results_a = {}
    for li in KEY_HEADS:
        W_v, b_v, W_o = decoded[li]
        for hi in KEY_HEADS[li]:
            W_v_h, b_v_h, W_o_h = get_head_matrices(W_v, b_v, W_o, hi)
            # SVD of the 128×128 core: W_v_h @ W_o_h = (128,3584) @ (3584,128)
            core = W_v_h @ W_o_h  # (128, 128)
            U, S, Vt = np.linalg.svd(core)
            # Effective rank (90% energy)
            energy = np.cumsum(S**2) / np.sum(S**2)
            rank90 = int(np.searchsorted(energy, 0.90) + 1)
            rank99 = int(np.searchsorted(energy, 0.99) + 1)
            ratio = S[0] / (S[1] + 1e-10)
            # Is it rotation-like? Check if singular values are similar
            sv_std = np.std(S[:rank90]) / (np.mean(S[:rank90]) + 1e-10)
            # Bias contribution magnitude
            bias_binding = b_v_h @ W_o_h.T  # (3584,)
            bias_norm = np.linalg.norm(bias_binding)

            print(f"\n  L{li} H{hi}:")
            print(f"    S[0:5] = {S[:5].round(2)}")
            print(f"    S[0]/S[1] = {ratio:.1f}")
            print(f"    rank(90%) = {rank90}, rank(99%) = {rank99}")
            print(f"    SV coefficient of variation (top-{rank90}): {sv_std:.3f}")
            print(f"    ||bias_binding|| = {bias_norm:.2f}")

            # Test: how much of binding comes from bias vs input?
            input_norms = []
            for c in countries:
                x = entity_normed[li][c]
                b_input = x @ W_v_h.T @ W_o_h.T  # input contribution
                input_norms.append(np.linalg.norm(b_input))
            mean_input = np.mean(input_norms)
            print(f"    mean ||input_binding|| = {mean_input:.2f}")
            print(f"    bias/input ratio = {bias_norm/mean_input:.3f}")

            results_a[f'L{li}_H{hi}'] = {
                'top_sv': S[:10].tolist(), 'ratio': float(ratio),
                'rank90': rank90, 'rank99': rank99,
                'sv_cv': float(sv_std), 'bias_norm': float(bias_norm),
                'mean_input_norm': float(mean_input),
            }

    # ══════════════════════════════════════════════════════════════════
    # PART B: Entity→Answer Structure
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part B: Entity→Answer Structure")
    print("─" * 72)

    results_b = {}
    for li in KEY_HEADS:
        W_v, b_v, W_o = decoded[li]
        for hi in KEY_HEADS[li]:
            W_v_h, b_v_h, W_o_h = get_head_matrices(W_v, b_v, W_o, hi)
            print(f"\n  L{li} H{hi}:", flush=True)

            # Compute binding vectors for all entities
            bindings = np.zeros((len(countries), 3584), dtype=np.float32)
            entities = np.zeros((len(countries), 3584), dtype=np.float32)
            for idx, c in enumerate(countries):
                entities[idx] = entity_normed[li][c]
                bindings[idx] = compute_binding(W_v_h, b_v_h, W_o_h,
                                                entity_normed[li][c])

            # PCA of binding vectors
            bind_mean = bindings.mean(axis=0)
            bind_centered = bindings - bind_mean
            _, S_bind, Vt_bind = np.linalg.svd(bind_centered, full_matrices=False)
            bind_energy = np.cumsum(S_bind**2) / np.sum(S_bind**2)
            print(f"    Binding PCA: S = {S_bind[:5].round(2)}")
            print(f"    Binding dims for 90% = {int(np.searchsorted(bind_energy, 0.90)+1)}")

            # PCA of entity vectors
            ent_mean = entities.mean(axis=0)
            ent_centered = entities - ent_mean
            _, S_ent, Vt_ent = np.linalg.svd(ent_centered, full_matrices=False)
            ent_energy = np.cumsum(S_ent**2) / np.sum(S_ent**2)
            print(f"    Entity PCA: S = {S_ent[:5].round(2)}")
            print(f"    Entity dims for 90% = {int(np.searchsorted(ent_energy, 0.90)+1)}")

            # Cosine between binding direction and answer direction
            print(f"    Per-country binding→answer cosine:")
            cos_list = []
            for idx, c in enumerate(countries):
                b_hat = bindings[idx] / (np.linalg.norm(bindings[idx]) + 1e-10)
                cos = float(np.dot(b_hat, answer_dirs[c]))
                cos_list.append(cos)
                print(f"      {c:12s}: cos = {cos:+.4f}")
            print(f"    Mean |cos| = {np.mean(np.abs(cos_list)):.4f}")

            # Linearity test: does delta_binding ∝ delta_entity?
            # For each pair, compare displacement directions
            pair_cos = []
            for i in range(len(countries)):
                for j in range(i+1, len(countries)):
                    de = entities[j] - entities[i]
                    db = bindings[j] - bindings[i]
                    de_hat = de / (np.linalg.norm(de) + 1e-10)
                    db_hat = db / (np.linalg.norm(db) + 1e-10)
                    pair_cos.append(float(np.dot(de_hat, db_hat)))
            print(f"    Linearity: mean cos(Δentity, Δbinding) = "
                  f"{np.mean(pair_cos):.4f} ± {np.std(pair_cos):.4f}")

            # Does M_h preserve or destroy entity structure?
            # Compare pairwise entity cosines vs pairwise binding cosines
            ent_cos_pairs, bind_cos_pairs = [], []
            for i in range(len(countries)):
                ei = entities[i] / np.linalg.norm(entities[i])
                bi = bindings[i] / np.linalg.norm(bindings[i])
                for j in range(i+1, len(countries)):
                    ej = entities[j] / np.linalg.norm(entities[j])
                    bj = bindings[j] / np.linalg.norm(bindings[j])
                    ent_cos_pairs.append(float(np.dot(ei, ej)))
                    bind_cos_pairs.append(float(np.dot(bi, bj)))
            corr = np.corrcoef(ent_cos_pairs, bind_cos_pairs)[0, 1]
            print(f"    Structure preservation: r(entity_cos, binding_cos) = {corr:.4f}")

            results_b[f'L{li}_H{hi}'] = {
                'bind_sv': S_bind[:5].tolist(),
                'bind_dims90': int(np.searchsorted(bind_energy, 0.90)+1),
                'ent_sv': S_ent[:5].tolist(),
                'mean_abs_cos': float(np.mean(np.abs(cos_list))),
                'linearity': float(np.mean(pair_cos)),
                'structure_corr': float(corr),
            }

    # ══════════════════════════════════════════════════════════════════
    # PART C: Binding Transfer — Leave-one-out
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part C: Binding Transfer (leave-one-out)")
    print("─" * 72)

    results_c = {}
    # Focus on L23 H6 (primary fact head)
    li, hi = 23, 6
    W_v, b_v, W_o = decoded[li]
    W_v_h, b_v_h, W_o_h = get_head_matrices(W_v, b_v, W_o, hi)

    # Collect all bindings and entity states
    all_entities = np.array([entity_normed[li][c] for c in countries])
    all_bindings = np.array([compute_binding(W_v_h, b_v_h, W_o_h,
                                             entity_normed[li][c])
                             for c in countries])

    print(f"\n  L{li} H{hi}: Leave-one-out binding prediction", flush=True)

    for test_idx in range(len(countries)):
        test_country = countries[test_idx]
        train_mask = [i for i in range(len(countries)) if i != test_idx]

        train_ent = all_entities[train_mask]   # (7, 3584)
        train_bind = all_bindings[train_mask]  # (7, 3584)

        # Method 1: Mean binding (baseline)
        pred_mean = train_bind.mean(axis=0)

        # Method 2: Nearest neighbor in entity space
        test_ent = all_entities[test_idx]
        dists = np.array([np.linalg.norm(test_ent - train_ent[i])
                          for i in range(len(train_mask))])
        nn_idx = np.argmin(dists)
        pred_nn = train_bind[nn_idx]

        # Method 3: Linear regression (entity → binding)
        # binding ≈ entity @ A + b
        # Solve via pseudoinverse
        X = np.column_stack([train_ent, np.ones(len(train_mask))])  # (7, 3585)
        # This is underdetermined (3585 params, 7 samples) — use ridge
        lam = 1.0
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        XtY = X.T @ train_bind
        W_reg = np.linalg.solve(XtX, XtY)  # (3585, 3584)
        x_test = np.append(test_ent, 1.0)
        pred_reg = x_test @ W_reg

        # Method 4: Delta transfer — use displacement from nearest neighbor
        # pred = true_nn_binding + M_h @ (test_entity - nn_entity)
        # Since M_h is the same, this is just: M_h @ test_entity + bias
        # which IS the actual binding. So delta transfer = exact.
        # Instead, use displacement in binding space scaled by entity displacement
        nn_country = countries[train_mask[nn_idx]]
        delta_ent = test_ent - train_ent[nn_idx]
        # Scale: how much does binding change per unit entity change?
        # Use average scale from training pairs
        scales = []
        for i in range(len(train_mask)):
            for j in range(i+1, len(train_mask)):
                de = np.linalg.norm(train_ent[j] - train_ent[i])
                db = np.linalg.norm(train_bind[j] - train_bind[i])
                if de > 1e-6:
                    scales.append(db / de)
        mean_scale = np.mean(scales) if scales else 1.0
        # Project delta through average binding direction
        delta_bind_dir = (train_bind - train_bind.mean(axis=0))
        delta_ent_dir = (train_ent - train_ent.mean(axis=0))
        # Simple: just use M_h directly (the actual transformation)
        pred_direct = compute_binding(W_v_h, b_v_h, W_o_h, test_ent)

        # Evaluate each prediction
        methods = {
            'mean': pred_mean,
            'nn': pred_nn,
            'ridge': pred_reg,
            'direct_Mh': pred_direct,
        }

        print(f"\n  {test_country} (held out):")
        for name, pred in methods.items():
            pred_3d = pred[np.newaxis, np.newaxis, :].astype(np.float32)
            pred_logits = get_logits(engine, pred_3d)
            rank, _ = get_rank(pred_logits, facts[test_country]['answer'], tokenizer)
            top3 = top_k_tokens(tokenizer, pred_logits, 3)
            # Also check cosine with actual binding
            actual = all_bindings[test_idx]
            cos_actual = float(np.dot(
                pred / (np.linalg.norm(pred) + 1e-10),
                actual / (np.linalg.norm(actual) + 1e-10)))
            print(f"    {name:12s}: rank={rank:5d}, cos_actual={cos_actual:+.4f}, "
                  f"top3={[t[0] for t in top3]}")

        results_c[test_country] = {}

    # ══════════════════════════════════════════════════════════════════
    # PART D: Binding Subspace — does M_h have a "fact subspace"?
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Part D: Fact Subspace in M_h")
    print("─" * 72)

    # For L23 H6: project answer directions through M_h^{-1}
    # (pseudo-inverse) to find what input directions produce answers
    li, hi = 23, 6
    W_v, b_v, W_o = decoded[li]
    W_v_h, b_v_h, W_o_h = get_head_matrices(W_v, b_v, W_o, hi)

    # The "answer-producing input" for each country: what input x
    # would make M_h @ x point toward the answer?
    # M_h = W_v_h.T @ W_o_h.T, so we need x such that
    # x @ W_v_h.T @ W_o_h.T ≈ answer_dir
    # i.e., x @ A ≈ y where A = W_v_h.T @ W_o_h.T

    # Get answer-producing inputs via pseudoinverse
    # x = answer_dir @ pinv(A) where A = W_v_h.T @ W_o_h.T
    # But A is 3584×3584 rank-128. Use factored form:
    # A = W_v_h.T @ W_o_h.T
    # x @ A = y => x @ W_v_h.T @ W_o_h.T = y
    # Let z = x @ W_v_h.T (128-d), then z @ W_o_h.T = y
    # z = y @ pinv(W_o_h.T) = y @ W_o_h @ pinv(W_o_h.T @ W_o_h) ... complex
    # Simpler: z = y @ W_o_h @ inv(W_o_h.T @ W_o_h)
    WoTWo = W_o_h.T @ W_o_h  # (128, 128)
    WoTWo_inv = np.linalg.inv(WoTWo + 1e-6 * np.eye(128))

    print(f"\n  L{li} H{hi}: Answer-producing inputs", flush=True)

    answer_inputs = {}
    for c in countries:
        y = answer_dirs[c]  # (3584,)
        z = y @ W_o_h @ WoTWo_inv  # (128,)
        # Now x @ W_v_h.T = z => x = z @ W_v_h
        # But this gives the minimum-norm x in the row space of W_v_h
        x_answer = z @ W_v_h  # (3584,)
        answer_inputs[c] = x_answer

    # Compare answer-producing inputs to actual entity hidden states
    print(f"    cos(answer_input, entity_state):")
    for c in countries:
        ai_hat = answer_inputs[c] / (np.linalg.norm(answer_inputs[c]) + 1e-10)
        ent_hat = entity_normed[li][c] / (np.linalg.norm(entity_normed[li][c]) + 1e-10)
        cos = float(np.dot(ai_hat, ent_hat))
        print(f"      {c:12s}: cos = {cos:+.4f}")

    # PCA of answer-producing inputs — do they span a small subspace?
    ai_matrix = np.array([answer_inputs[c] for c in countries])
    ai_mean = ai_matrix.mean(axis=0)
    ai_centered = ai_matrix - ai_mean
    _, S_ai, _ = np.linalg.svd(ai_centered, full_matrices=False)
    ai_energy = np.cumsum(S_ai**2) / np.sum(S_ai**2)
    print(f"\n    Answer-input PCA: S = {S_ai[:5].round(4)}")
    print(f"    Dims for 90% = {int(np.searchsorted(ai_energy, 0.90)+1)}")

    # Cross-entity structure: are answer-producing inputs similar?
    print(f"\n    Pairwise cos(answer_input_i, answer_input_j):")
    ai_cos_matrix = np.zeros((len(countries), len(countries)))
    for i in range(len(countries)):
        ai = answer_inputs[countries[i]]
        ai_hat = ai / (np.linalg.norm(ai) + 1e-10)
        for j in range(len(countries)):
            aj = answer_inputs[countries[j]]
            aj_hat = aj / (np.linalg.norm(aj) + 1e-10)
            ai_cos_matrix[i, j] = np.dot(ai_hat, aj_hat)
    # Print mean off-diagonal
    mask = ~np.eye(len(countries), dtype=bool)
    print(f"    Mean off-diagonal cos = {ai_cos_matrix[mask].mean():.4f}")
    print(f"    Min = {ai_cos_matrix[mask].min():.4f}, "
          f"Max = {ai_cos_matrix[mask].max():.4f}")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    print("\n  Part A — M_h geometry:")
    for key, v in results_a.items():
        print(f"    {key}: S[0]/S[1]={v['ratio']:.1f}, rank90={v['rank90']}, "
              f"bias/input={v['bias_norm']/v['mean_input_norm']:.3f}")

    print("\n  Part B — Entity→Answer:")
    for key, v in results_b.items():
        print(f"    {key}: |cos|={v['mean_abs_cos']:.4f}, "
              f"linearity={v['linearity']:.4f}, "
              f"struct_corr={v['structure_corr']:.4f}")

    # Save results
    out = {'part_a': results_a, 'part_b': results_b}
    out_path = 'experiments/model_reverse_engineering_v2/results/phase10z22_vwo_geometry.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
