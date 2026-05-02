"""
Phase 10z20: Triangulation — Predicting Answers from Multiple Manifolds
=========================================================================

F120 showed:
  - Entity positions are ABSOLUTE (cos=0.94 across fact types)
  - Relationships are RELATIVE (no universal "capital of" direction)
  - Structure-only fails for unseen facts (need answer direction)

Can we triangulate the answer using MULTIPLE relationship manifolds?

Part A: Entity-Answer Mapping
  - Learn a mapping from entity displacement → answer direction in
    the manifold's low-dimensional subspace.
  - Leave-one-out test: predict held-out capital from entity position.

Part B: Cross-Relationship Transfer
  - Use the language manifold to predict capitals (and vice versa).
  - The entity position from one query type should transfer to another.

Part C: Multi-Manifold Triangulation
  - Combine capital + language + continent constraints.
  - The intersection should narrow to the correct answer.
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
    return per_layer_attn, logits


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


def extract_facts(engine, tokenizer, facts_dict):
    """Extract attention data for a set of facts."""
    data = {}
    for name, fact in facts_dict.items():
        t1 = time.time()
        p_ids = tokenizer.encode(fact['prompt'])
        attn_list, logits = extract_all_layer_attn_and_hidden(engine, p_ids)
        rank, _ = get_rank(logits, fact['answer'], tokenizer)
        data[name] = {'attn_list': attn_list, 'logits': logits, 'rank': rank}
        print(f"    {name:12s} → '{fact['answer']}' rank={rank}  ({time.time()-t1:.1f}s)",
              flush=True)
    return data


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    all_layers = set(range(n_layers))
    KEY_LAYERS = [22, 23, 27]
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z20: TRIANGULATION — MULTIPLE MANIFOLDS")
    print("=" * 72)

    # ── Fact sets ──
    countries = ['France', 'Japan', 'Germany', 'Italy', 'Brazil', 'Egypt', 'Spain', 'Canada']

    capital_facts = {c: {'prompt': f'The capital of {c} is', 'answer': a} for c, a in [
        ('France', ' Paris'), ('Japan', ' Tokyo'), ('Germany', ' Berlin'),
        ('Italy', ' Rome'), ('Brazil', ' Brasilia'), ('Egypt', ' Cairo'),
        ('Spain', ' Madrid'), ('Canada', ' Ottawa'),
    ]}

    language_facts = {c: {'prompt': f'The official language of {c} is', 'answer': a} for c, a in [
        ('France', ' French'), ('Japan', ' Japanese'), ('Germany', ' German'),
        ('Italy', ' Italian'), ('Brazil', ' Portuguese'), ('Egypt', ' Arabic'),
        ('Spain', ' Spanish'), ('Canada', ' English'),
    ]}

    continent_facts = {c: {'prompt': f'{c} is located in', 'answer': a} for c, a in [
        ('France', ' Europe'), ('Japan', ' Asia'), ('Germany', ' Europe'),
        ('Italy', ' Europe'), ('Brazil', ' South'), ('Egypt', ' Africa'),
        ('Spain', ' Europe'), ('Canada', ' North'),
    ]}

    # ── Extract all fact data ──
    print("\n  Extracting capital facts...", flush=True)
    cap_data = extract_facts(engine, tokenizer, capital_facts)

    print("\n  Extracting language facts...", flush=True)
    lang_data = extract_facts(engine, tokenizer, language_facts)

    print("\n  Extracting continent facts...", flush=True)
    cont_data = extract_facts(engine, tokenizer, continent_facts)

    # ── Get LM head answer rows ──
    cap_answer_dirs = {}
    for c in countries:
        tid = tokenizer.encode(capital_facts[c]['answer'])[0]
        cap_answer_dirs[c] = decode_lm_row(engine, tid)

    lang_answer_dirs = {}
    for c in countries:
        tid = tokenizer.encode(language_facts[c]['answer'])[0]
        lang_answer_dirs[c] = decode_lm_row(engine, tid)

    # ══════════════════════════════════════════════════════════════════
    # PART A: ENTITY-ANSWER MAPPING
    # Learn: entity displacement → answer direction (in manifold subspace)
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part A: Entity-Answer Mapping (capital manifold)")
    print("─" * 72)

    # Compute grand mean attention for capital facts
    cap_grand_mean = {}
    for li in range(n_layers):
        cap_grand_mean[li] = np.mean(
            [cap_data[c]['attn_list'][li] for c in countries], axis=0)

    # Entity displacement = attention - grand_mean (at key layers, concat)
    entity_disps = {}  # country → concat of displacements at key layers
    for c in countries:
        vecs = [cap_data[c]['attn_list'][li] - cap_grand_mean[li] for li in KEY_LAYERS]
        entity_disps[c] = np.concatenate(vecs)

    # Answer directions
    answer_vecs = {}
    for c in countries:
        answer_vecs[c] = cap_answer_dirs[c] / np.linalg.norm(cap_answer_dirs[c])

    # Leave-one-out: for each held-out country, learn a mapping from
    # entity displacement to answer direction using the other 7 countries.
    #
    # Approach: in the entity displacement space, find the nearest
    # neighbors and use their answer directions as a weighted vote.
    #
    # But first, let's try something more principled:
    # Project entity displacements onto their principal components,
    # then learn a linear map from PC scores to answer directions.

    print(f"\n  Leave-one-out: entity displacement → answer prediction")

    for holdout in countries:
        train = [c for c in countries if c != holdout]

        # Build entity displacement matrix (7 × D)
        X_train = np.array([entity_disps[c] for c in train])  # (7, D)
        # Build answer direction matrix (7 × 3584)
        Y_train = np.array([answer_vecs[c] for c in train])  # (7, 3584)

        # SVD of entity displacements to get low-dim representation
        X_mean = X_train.mean(axis=0)
        X_centered = X_train - X_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Keep top-k components (90% variance)
        cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))
        
        # Project to k dimensions
        X_proj = X_centered @ Vt[:k].T  # (7, k)
        
        # Learn linear map: X_proj → Y_train
        # Y_train ≈ X_proj @ W + b
        # This is 7 equations in k unknowns (per output dim) — overdetermined if k < 7
        # Use least squares
        X_proj_bias = np.hstack([X_proj, np.ones((len(train), 1))])  # (7, k+1)
        W, residuals, rank_lstsq, sv = np.linalg.lstsq(X_proj_bias, Y_train, rcond=None)
        
        # Predict holdout
        x_holdout = entity_disps[holdout]
        x_centered = x_holdout - X_mean
        x_proj = x_centered @ Vt[:k].T  # (k,)
        x_proj_bias = np.append(x_proj, 1.0)  # (k+1,)
        y_pred = x_proj_bias @ W  # (3584,)
        y_pred_hat = y_pred / (np.linalg.norm(y_pred) + 1e-10)

        # Use predicted answer direction for manifold projection
        holdout_attn = cap_data[holdout]['attn_list']
        deltas_pred = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train), holdout_attn[li].shape[0]))
            for i, tn in enumerate(train):
                train_deltas[i] = cap_data[tn]['attn_list'][li] - holdout_attn[li]

            U_d, S_d, Vt_d = np.linalg.svd(train_deltas, full_matrices=False)
            cum = np.cumsum(S_d**2) / (np.sum(S_d**2) + 1e-10)
            kk = min(int(np.searchsorted(cum, 0.9)) + 1, len(S_d))
            basis = Vt_d[:kk]
            coeffs = basis @ y_pred_hat
            mean_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj = basis.T @ coeffs
            proj = proj / (np.linalg.norm(proj) + 1e-10) * mean_norm
            deltas_pred[li] = proj

        # Run with predicted deltas
        p_ids = tokenizer.encode(capital_facts[holdout]['prompt'])
        logits_pred = run_with_deltas(engine, p_ids, deltas_pred, all_layers)
        rank_pred, _ = get_rank(logits_pred, capital_facts[holdout]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits_pred, 5)

        # Also try: nearest neighbor in entity space
        dists = [np.linalg.norm(entity_disps[holdout] - entity_disps[c]) for c in train]
        nearest_idx = np.argmin(dists)
        nearest_country = train[nearest_idx]
        nearest_answer_dir = answer_vecs[nearest_country]

        # Manifold projection with nearest neighbor's answer direction
        deltas_nn = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train), holdout_attn[li].shape[0]))
            for i, tn in enumerate(train):
                train_deltas[i] = cap_data[tn]['attn_list'][li] - holdout_attn[li]

            U_d, S_d, Vt_d = np.linalg.svd(train_deltas, full_matrices=False)
            cum = np.cumsum(S_d**2) / (np.sum(S_d**2) + 1e-10)
            kk = min(int(np.searchsorted(cum, 0.9)) + 1, len(S_d))
            basis = Vt_d[:kk]
            coeffs = basis @ nearest_answer_dir
            mean_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj = basis.T @ coeffs
            proj = proj / (np.linalg.norm(proj) + 1e-10) * mean_norm
            deltas_nn[li] = proj

        logits_nn = run_with_deltas(engine, p_ids, deltas_nn, all_layers)
        rank_nn, _ = get_rank(logits_nn, capital_facts[holdout]['answer'], tokenizer)
        top5_nn = top_k_tokens(tokenizer, logits_nn, 5)

        # Also: cos between predicted direction and true answer direction
        true_dir = answer_vecs[holdout]
        cos_pred = float(np.dot(y_pred_hat, true_dir))

        print(f"\n  {holdout} → '{capital_facts[holdout]['answer']}' "
              f"(baseline rank={cap_data[holdout]['rank']})")
        print(f"    Learned map:  rank={rank_pred}, cos(pred,true)={cos_pred:.4f}, "
              f"top5={[t[0] for t in top5]}")
        print(f"    Nearest ({nearest_country}): rank={rank_nn}, "
              f"top5={[t[0] for t in top5_nn]}")

    # ══════════════════════════════════════════════════════════════════
    # PART B: CROSS-RELATIONSHIP TRANSFER
    # Use language manifold entity positions to predict capitals
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part B: Cross-Relationship Transfer (language → capital)")
    print("─" * 72)

    # The entity displacement from the LANGUAGE query should be usable
    # to predict CAPITALS, since entity positions are shared (cos=0.94).
    #
    # Method: use the language entity displacement to find the answer
    # direction, then inject into the capital query.

    lang_grand_mean = {}
    for li in range(n_layers):
        lang_grand_mean[li] = np.mean(
            [lang_data[c]['attn_list'][li] for c in countries], axis=0)

    lang_entity_disps = {}
    for c in countries:
        vecs = [lang_data[c]['attn_list'][li] - lang_grand_mean[li] for li in KEY_LAYERS]
        lang_entity_disps[c] = np.concatenate(vecs)

    print(f"\n  Cross-transfer: language entity position → capital prediction")

    for holdout in countries:
        train = [c for c in countries if c != holdout]

        # Use LANGUAGE entity displacements for training
        X_train = np.array([lang_entity_disps[c] for c in train])
        # But CAPITAL answer directions as targets
        Y_train = np.array([answer_vecs[c] for c in train])

        X_mean = X_train.mean(axis=0)
        X_centered = X_train - X_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))

        X_proj = X_centered @ Vt[:k].T
        X_proj_bias = np.hstack([X_proj, np.ones((len(train), 1))])
        W, _, _, _ = np.linalg.lstsq(X_proj_bias, Y_train, rcond=None)

        # Predict using holdout's LANGUAGE entity displacement
        x_holdout = lang_entity_disps[holdout]
        x_centered = x_holdout - X_mean
        x_proj = x_centered @ Vt[:k].T
        x_proj_bias = np.append(x_proj, 1.0)
        y_pred = x_proj_bias @ W
        y_pred_hat = y_pred / (np.linalg.norm(y_pred) + 1e-10)

        # Inject into capital query via manifold projection
        holdout_cap_attn = cap_data[holdout]['attn_list']
        deltas = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train), holdout_cap_attn[li].shape[0]))
            for i, tn in enumerate(train):
                train_deltas[i] = cap_data[tn]['attn_list'][li] - holdout_cap_attn[li]

            U_d, S_d, Vt_d = np.linalg.svd(train_deltas, full_matrices=False)
            cum = np.cumsum(S_d**2) / (np.sum(S_d**2) + 1e-10)
            kk = min(int(np.searchsorted(cum, 0.9)) + 1, len(S_d))
            basis = Vt_d[:kk]
            coeffs = basis @ y_pred_hat
            mean_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj = basis.T @ coeffs
            proj = proj / (np.linalg.norm(proj) + 1e-10) * mean_norm
            deltas[li] = proj

        p_ids = tokenizer.encode(capital_facts[holdout]['prompt'])
        logits_cross = run_with_deltas(engine, p_ids, deltas, all_layers)
        rank_cross, _ = get_rank(logits_cross, capital_facts[holdout]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits_cross, 5)

        true_dir = answer_vecs[holdout]
        cos_pred = float(np.dot(y_pred_hat, true_dir))

        print(f"    {holdout:12s}: rank={rank_cross}, cos={cos_pred:.4f}, "
              f"top5={[t[0] for t in top5]}")

    # ══════════════════════════════════════════════════════════════════
    # PART C: MULTI-MANIFOLD TRIANGULATION
    # Combine entity info from capital + language + continent
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part C: Multi-Manifold Triangulation")
    print("─" * 72)

    cont_grand_mean = {}
    for li in range(n_layers):
        cont_grand_mean[li] = np.mean(
            [cont_data[c]['attn_list'][li] for c in countries], axis=0)

    cont_entity_disps = {}
    for c in countries:
        vecs = [cont_data[c]['attn_list'][li] - cont_grand_mean[li] for li in KEY_LAYERS]
        cont_entity_disps[c] = np.concatenate(vecs)

    print(f"\n  Triangulation: capital + language + continent → capital prediction")

    for holdout in countries:
        train = [c for c in countries if c != holdout]

        # Stack entity displacements from ALL THREE manifolds
        X_parts = []
        for c in train:
            cap_disp = entity_disps[c]
            lang_disp = lang_entity_disps[c]
            cont_disp = cont_entity_disps[c]
            combined = np.concatenate([cap_disp, lang_disp, cont_disp])
            X_parts.append(combined)
        X_train = np.array(X_parts)
        Y_train = np.array([answer_vecs[c] for c in train])

        X_mean = X_train.mean(axis=0)
        X_centered = X_train - X_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))

        X_proj = X_centered @ Vt[:k].T
        X_proj_bias = np.hstack([X_proj, np.ones((len(train), 1))])
        W, _, _, _ = np.linalg.lstsq(X_proj_bias, Y_train, rcond=None)

        # Predict
        x_holdout = np.concatenate([
            entity_disps[holdout],
            lang_entity_disps[holdout],
            cont_entity_disps[holdout],
        ])
        x_centered = x_holdout - X_mean
        x_proj = x_centered @ Vt[:k].T
        x_proj_bias = np.append(x_proj, 1.0)
        y_pred = x_proj_bias @ W
        y_pred_hat = y_pred / (np.linalg.norm(y_pred) + 1e-10)

        # Inject into capital query
        holdout_cap_attn = cap_data[holdout]['attn_list']
        deltas = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train), holdout_cap_attn[li].shape[0]))
            for i, tn in enumerate(train):
                train_deltas[i] = cap_data[tn]['attn_list'][li] - holdout_cap_attn[li]

            U_d, S_d, Vt_d = np.linalg.svd(train_deltas, full_matrices=False)
            cum = np.cumsum(S_d**2) / (np.sum(S_d**2) + 1e-10)
            kk = min(int(np.searchsorted(cum, 0.9)) + 1, len(S_d))
            basis = Vt_d[:kk]
            coeffs = basis @ y_pred_hat
            mean_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj = basis.T @ coeffs
            proj = proj / (np.linalg.norm(proj) + 1e-10) * mean_norm
            deltas[li] = proj

        p_ids = tokenizer.encode(capital_facts[holdout]['prompt'])
        logits_tri = run_with_deltas(engine, p_ids, deltas, all_layers)
        rank_tri, _ = get_rank(logits_tri, capital_facts[holdout]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits_tri, 5)

        true_dir = answer_vecs[holdout]
        cos_pred = float(np.dot(y_pred_hat, true_dir))

        print(f"    {holdout:12s}: rank={rank_tri}, cos={cos_pred:.4f}, "
              f"top5={[t[0] for t in top5]}")

    # ══════════════════════════════════════════════════════════════════
    # PART D: DIRECT TRIANGULATION — no learned mapping
    # Use the entity position directly to weight training answers
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part D: Direct Triangulation (entity-weighted voting)")
    print("─" * 72)

    # For each held-out country, find its position in TruthSpace
    # (average entity displacement across all three manifolds),
    # then weight the training countries' answer directions by
    # proximity in TruthSpace.

    # Compute TruthSpace positions (average across manifolds)
    truthspace_pos = {}
    for c in countries:
        # Average the entity displacement across capital, language, continent
        cap_disp = entity_disps[c]
        lang_disp = lang_entity_disps[c]
        cont_disp = cont_entity_disps[c]
        # These are different sizes if KEY_LAYERS differs, but same structure
        # Average them element-wise (they're all concat of KEY_LAYERS)
        avg_disp = (cap_disp + lang_disp + cont_disp) / 3.0
        truthspace_pos[c] = avg_disp

    print(f"\n  TruthSpace position cosines between countries:")
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if j > i:
                d1 = truthspace_pos[c1] / (np.linalg.norm(truthspace_pos[c1]) + 1e-10)
                d2 = truthspace_pos[c2] / (np.linalg.norm(truthspace_pos[c2]) + 1e-10)
                cos = float(np.dot(d1, d2))
                if abs(cos) > 0.3:
                    print(f"    {c1:10s} × {c2:10s}: cos = {cos:+.4f}")

    print(f"\n  Entity-weighted capital prediction:")

    for holdout in countries:
        train = [c for c in countries if c != holdout]
        holdout_pos = truthspace_pos[holdout]
        holdout_hat = holdout_pos / (np.linalg.norm(holdout_pos) + 1e-10)

        # Weight each training country by cosine similarity in TruthSpace
        weights = []
        for c in train:
            c_hat = truthspace_pos[c] / (np.linalg.norm(truthspace_pos[c]) + 1e-10)
            cos = float(np.dot(holdout_hat, c_hat))
            # Use softmax-like weighting: exp(cos * temperature)
            weights.append(cos)

        # Convert to probability weights using softmax
        weights = np.array(weights)
        temp = 5.0  # temperature
        exp_weights = np.exp(weights * temp)
        prob_weights = exp_weights / exp_weights.sum()

        # Weighted average of answer directions
        y_weighted = sum(prob_weights[i] * answer_vecs[train[i]] for i in range(len(train)))
        y_weighted_hat = y_weighted / (np.linalg.norm(y_weighted) + 1e-10)

        # Inject
        holdout_cap_attn = cap_data[holdout]['attn_list']
        deltas = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train), holdout_cap_attn[li].shape[0]))
            for i, tn in enumerate(train):
                train_deltas[i] = cap_data[tn]['attn_list'][li] - holdout_cap_attn[li]

            U_d, S_d, Vt_d = np.linalg.svd(train_deltas, full_matrices=False)
            cum = np.cumsum(S_d**2) / (np.sum(S_d**2) + 1e-10)
            kk = min(int(np.searchsorted(cum, 0.9)) + 1, len(S_d))
            basis = Vt_d[:kk]
            coeffs = basis @ y_weighted_hat
            mean_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
            proj = basis.T @ coeffs
            proj = proj / (np.linalg.norm(proj) + 1e-10) * mean_norm
            deltas[li] = proj

        p_ids = tokenizer.encode(capital_facts[holdout]['prompt'])
        logits_ew = run_with_deltas(engine, p_ids, deltas, all_layers)
        rank_ew, _ = get_rank(logits_ew, capital_facts[holdout]['answer'], tokenizer)
        top5 = top_k_tokens(tokenizer, logits_ew, 5)

        true_dir = answer_vecs[holdout]
        cos_pred = float(np.dot(y_weighted_hat, true_dir))

        # Show nearest country and its weight
        nearest_idx = np.argmax(prob_weights)
        nearest = train[nearest_idx]

        print(f"    {holdout:12s}: rank={rank_ew}, cos={cos_pred:.4f}, "
              f"nearest={nearest}({prob_weights[nearest_idx]:.2f}), "
              f"top5={[t[0] for t in top5]}")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    elapsed = time.time() - t0
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z20_triangulation.json')
    with open(out_path, 'w') as f:
        json.dump({'total_time': elapsed}, f)
    print(f"  Saved to {out_path}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
