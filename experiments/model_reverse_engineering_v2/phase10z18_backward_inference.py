"""
Phase 10z18: Backward Inference — From Answer to Path
========================================================

Can we start with an answer and work backwards to find/construct paths?

Part A: Answer Anatomy
  - For each known capital fact, project per-layer attention deltas
    onto W_lm[answer] to measure each layer's contribution toward
    the answer direction.

Part B: Fact Manifold
  - Stack per-layer deltas across facts, PCA to find dimensionality.
  - Extract structural template T and fact-specific deviations S_i.

Part C: Backward Path Construction
  - Hold out one fact. Use the template from remaining facts to
    CONSTRUCT the per-layer deltas for the held-out answer.
  - Inject constructed deltas. Does the model predict correctly?

Part D: Knowledge Extension
  - Use the structural template to predict where a NOVEL answer
    should be (not by injecting W_lm[token] directly, but by
    geometric extrapolation from the fact manifold).
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


def extract_all_layer_attn(engine, prompt_ids):
    """Run forward pass, cache per-layer attention outputs (last token)."""
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
    """Decode a single row of the LM head weight matrix to float."""
    s = engine.lm_head.weight.signs[tid:tid+1, :]
    e = engine.lm_head.weight.exponents[tid:tid+1, :]
    return phi_to_float(s, e)[0]


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    all_layers = set(range(n_layers))
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z18: BACKWARD INFERENCE — FROM ANSWER TO PATH")
    print("=" * 72)

    # ── Facts ──
    facts = {
        'France':   {'prompt': 'The capital of France is',   'answer': ' Paris'},
        'Japan':    {'prompt': 'The capital of Japan is',    'answer': ' Tokyo'},
        'Germany':  {'prompt': 'The capital of Germany is',  'answer': ' Berlin'},
        'Italy':    {'prompt': 'The capital of Italy is',    'answer': ' Rome'},
        'Brazil':   {'prompt': 'The capital of Brazil is',   'answer': ' Brasilia'},
        'Egypt':    {'prompt': 'The capital of Egypt is',    'answer': ' Cairo'},
        'Spain':    {'prompt': 'The capital of Spain is',    'answer': ' Madrid'},
        'Canada':   {'prompt': 'The capital of Canada is',   'answer': ' Ottawa'},
    }

    # ══════════════════════════════════════════════════════════════════
    # PART A: ANSWER ANATOMY — per-layer contribution toward answer
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part A: Answer Anatomy — how each layer contributes to the answer")
    print("─" * 72)

    # Get LM head rows for all answers
    answer_dirs = {}
    for name, fact in facts.items():
        tid = tokenizer.encode(fact['answer'])[0]
        answer_dirs[name] = decode_lm_row(engine, tid)
        print(f"  W_lm['{fact['answer']}'] (id={tid}): norm={np.linalg.norm(answer_dirs[name]):.4f}")

    # Extract per-layer attention for each fact
    fact_data = {}
    for name, fact in facts.items():
        t1 = time.time()
        p_ids = tokenizer.encode(fact['prompt'])
        attn_list, logits = extract_all_layer_attn(engine, p_ids)
        rank, logit = get_rank(logits, fact['answer'], tokenizer)
        fact_data[name] = {
            'attn_list': attn_list,   # list of 28 vectors (3584,)
            'logits': logits,
            'rank': rank,
        }
        print(f"  {name:10s}: '{fact['answer']}' rank={rank}  ({time.time()-t1:.1f}s)")

    # For each fact, compute per-layer projection onto answer direction
    print(f"\n  Per-layer projection onto W_lm[answer] (cosine similarity × norm):")
    print(f"  {'Layer':<6}", end="")
    for name in facts:
        print(f"  {name:>8}", end="")
    print()

    # Store projections for manifold analysis
    projections = {}  # name → array of shape (28,)

    for name in facts:
        attn_list = fact_data[name]['attn_list']
        w = answer_dirs[name]
        w_hat = w / np.linalg.norm(w)
        projs = []
        for li in range(n_layers):
            # Projection of attn_L onto answer direction
            proj = float(np.dot(attn_list[li], w_hat))
            projs.append(proj)
        projections[name] = np.array(projs)

    # Print per-layer projections
    for li in range(n_layers):
        print(f"  L{li:<4d}", end="")
        for name in facts:
            print(f"  {projections[name][li]:>8.2f}", end="")
        print()

    # Summary: which layers contribute most (averaged across facts)
    mean_proj = np.mean([np.abs(projections[n]) for n in facts], axis=0)
    top_layers = np.argsort(mean_proj)[-7:][::-1]
    print(f"\n  Top contributing layers (mean |projection|):")
    for li in top_layers:
        print(f"    L{li}: {mean_proj[li]:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # PART B: FACT MANIFOLD — PCA of per-layer attention deltas
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part B: Fact Manifold — structure of capital city knowledge")
    print("─" * 72)

    # Use France as baseline, compute deltas for all others
    baseline_name = 'France'
    baseline_attn = fact_data[baseline_name]['attn_list']

    # Stack all per-layer deltas into a matrix
    # Each fact gets a vector: concat of all 28 layer deltas
    # Shape: (n_facts, 28 * 3584)
    fact_names = [n for n in facts if n != baseline_name]
    n_facts = len(fact_names)

    # But 28*3584 = 100,352 dims is too large for PCA interpretation.
    # Instead, analyze per-layer: for each layer, stack fact deltas
    # and find the manifold dimensionality per layer.

    print(f"\n  Per-layer manifold analysis (deltas from {baseline_name}):")
    print(f"  {'Layer':<6} {'σ[0]':>8} {'σ[1]':>8} {'σ[2]':>8} {'ratio':>8} {'rank90%':>8}")

    per_layer_deltas = {}  # layer → (n_facts, 3584) matrix
    per_layer_svd = {}

    for li in range(n_layers):
        deltas_mat = np.zeros((n_facts, baseline_attn[li].shape[0]))
        for fi, name in enumerate(fact_names):
            deltas_mat[fi] = fact_data[name]['attn_list'][li] - baseline_attn[li]
        per_layer_deltas[li] = deltas_mat

        # SVD
        U, S, Vt = np.linalg.svd(deltas_mat, full_matrices=False)
        per_layer_svd[li] = (U, S, Vt)

        # Rank at 90% variance
        cum_var = np.cumsum(S**2) / np.sum(S**2) if np.sum(S**2) > 0 else np.array([1.0])
        rank90 = int(np.searchsorted(cum_var, 0.9)) + 1

        s0 = S[0] if len(S) > 0 else 0
        s1 = S[1] if len(S) > 1 else 0
        s2 = S[2] if len(S) > 2 else 0
        ratio = s0/s1 if s1 > 1e-10 else float('inf')

        if li in top_layers or ratio > 10:
            print(f"  L{li:<4d} {s0:>8.2f} {s1:>8.2f} {s2:>8.2f} {ratio:>8.1f} {rank90:>8d}")

    # ══════════════════════════════════════════════════════════════════
    # PART C: BACKWARD PATH CONSTRUCTION — leave-one-out test
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part C: Backward Path Construction — leave-one-out")
    print("─" * 72)

    # For each fact (except baseline), hold it out and try to reconstruct
    # its per-layer deltas from the remaining facts' manifold.
    #
    # Method: project the held-out fact's answer direction onto the
    # manifold basis (from remaining facts), then construct deltas.

    print(f"\n  Leave-one-out reconstruction:")
    all_fact_names = list(facts.keys())

    for holdout_name in all_fact_names:
        # Training set: all facts except holdout and baseline
        train_names = [n for n in all_fact_names if n != holdout_name]

        holdout_prompt_ids = tokenizer.encode(facts[holdout_name]['prompt'])
        holdout_answer = facts[holdout_name]['answer']

        # Extract holdout's own attention (we need this as the "host")
        holdout_attn = fact_data[holdout_name]['attn_list']

        # For each layer, compute the mean delta from training facts
        # relative to the holdout's own attention (since we're injecting
        # into the holdout prompt).
        #
        # Strategy: the "capital city template" is the MEAN attention
        # pattern across training facts. We compute:
        #   constructed_delta[L] = mean(train_attn[L]) - holdout_attn[L]
        #
        # But we want to steer toward the holdout ANSWER, not the mean.
        # So we weight the training facts by how much their answer
        # direction aligns with the holdout's answer direction.

        # Get holdout answer direction
        holdout_tid = tokenizer.encode(holdout_answer)[0]
        holdout_w = decode_lm_row(engine, holdout_tid)
        holdout_w_hat = holdout_w / np.linalg.norm(holdout_w)

        # Strategy 1: Simple mean of training deltas
        constructed_deltas_mean = {}
        for li in range(n_layers):
            train_attns = [fact_data[n]['attn_list'][li] for n in train_names]
            mean_train = np.mean(train_attns, axis=0)
            constructed_deltas_mean[li] = mean_train - holdout_attn[li]

        # Strategy 2: Weighted by answer similarity
        # Weight each training fact by cos(W_lm[train_answer], W_lm[holdout_answer])
        weights = []
        for tn in train_names:
            tw = answer_dirs[tn]
            tw_hat = tw / np.linalg.norm(tw)
            cos_sim = float(np.dot(holdout_w_hat, tw_hat))
            weights.append(max(cos_sim, 0))  # only positive weights

        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
        else:
            weights = [1.0/len(train_names)] * len(train_names)

        constructed_deltas_weighted = {}
        for li in range(n_layers):
            weighted_attn = sum(
                weights[i] * fact_data[train_names[i]]['attn_list'][li]
                for i in range(len(train_names))
            )
            constructed_deltas_weighted[li] = weighted_attn - holdout_attn[li]

        # Strategy 3: Project holdout answer direction through manifold
        # Use the SVD basis from the training facts to find the best
        # decomposition that points toward the holdout answer.
        #
        # For each layer, the training deltas define a subspace.
        # Project W_lm[holdout_answer] onto this subspace at each layer.
        constructed_deltas_proj = {}
        for li in range(n_layers):
            # Training deltas relative to holdout
            train_deltas = np.zeros((len(train_names), holdout_attn[li].shape[0]))
            for i, tn in enumerate(train_names):
                train_deltas[i] = fact_data[tn]['attn_list'][li] - holdout_attn[li]

            if train_deltas.shape[0] > 1:
                U, S, Vt = np.linalg.svd(train_deltas, full_matrices=False)
                # Project answer direction onto the span of Vt
                # Use top-k components that capture 90% variance
                cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
                k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))
                basis = Vt[:k]  # (k, 3584)
                # Project holdout_w onto this basis
                coeffs = basis @ holdout_w_hat  # (k,)
                # Reconstruct delta in the subspace, scaled by mean delta norm
                mean_delta_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
                proj_delta = (basis.T @ coeffs)  # (3584,)
                proj_delta = proj_delta / (np.linalg.norm(proj_delta) + 1e-10) * mean_delta_norm
                constructed_deltas_proj[li] = proj_delta
            else:
                constructed_deltas_proj[li] = train_deltas[0]

        # Test all three strategies
        strategies = [
            ("mean", constructed_deltas_mean),
            ("weighted", constructed_deltas_weighted),
            ("manifold_proj", constructed_deltas_proj),
        ]

        print(f"\n  Holdout: {holdout_name} (answer: '{holdout_answer}')")
        print(f"    Baseline rank: {fact_data[holdout_name]['rank']}")

        for strat_name, deltas in strategies:
            logits_constructed = run_with_deltas(
                engine, holdout_prompt_ids, deltas, all_layers)
            rank, logit = get_rank(logits_constructed, holdout_answer, tokenizer)
            top1_idx = int(np.argmax(logits_constructed))
            top1_tok = tokenizer.decode([top1_idx])
            # Also check what other capitals come up
            other_ranks = {}
            for on, of in facts.items():
                if on != holdout_name:
                    or_rank, _ = get_rank(logits_constructed, of['answer'], tokenizer)
                    other_ranks[on] = or_rank

            print(f"    {strat_name:15s}: '{holdout_answer}' rank={rank}, "
                  f"top1='{top1_tok}' ({logit:.2f})")

    # ══════════════════════════════════════════════════════════════════
    # PART D: KNOWLEDGE EXTENSION — predict novel answer from structure
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part D: Knowledge Extension — predict from structure alone")
    print("─" * 72)

    # The ultimate test: can we find the answer to a question
    # WITHOUT using W_lm[answer]?
    #
    # Method: for a held-out fact, construct deltas using the manifold
    # projection BUT project onto the average answer direction instead
    # of the specific answer. Then see what token the model predicts.
    #
    # If the manifold structure alone encodes the relationship between
    # country and capital, the correct answer should emerge.

    # First, compute the "generic answer direction" — mean of all
    # answer directions (this encodes "capital city-ness" not any
    # specific capital)
    all_answer_vecs = np.array([answer_dirs[n] for n in facts])
    mean_answer_dir = np.mean(all_answer_vecs, axis=0)
    mean_answer_dir = mean_answer_dir / np.linalg.norm(mean_answer_dir)

    # Also compute the specific country→capital displacement vectors
    # (in the attention delta space)
    print(f"\n  Mean answer direction norm: {np.linalg.norm(np.mean(all_answer_vecs, axis=0)):.4f}")
    print(f"  Inter-answer cosines:")
    for i, n1 in enumerate(facts):
        for j, n2 in enumerate(facts):
            if j > i:
                d1 = answer_dirs[n1] / np.linalg.norm(answer_dirs[n1])
                d2 = answer_dirs[n2] / np.linalg.norm(answer_dirs[n2])
                cos = float(np.dot(d1, d2))
                if abs(cos) > 0.3:
                    print(f"    {n1}-{n2}: cos={cos:.4f}")

    # Leave-one-out: for each fact, construct deltas from template
    # WITHOUT using the specific answer direction
    print(f"\n  Structure-only prediction (no answer direction used):")

    for holdout_name in all_fact_names:
        train_names = [n for n in all_fact_names if n != holdout_name]
        holdout_prompt_ids = tokenizer.encode(facts[holdout_name]['prompt'])
        holdout_answer = facts[holdout_name]['answer']
        holdout_attn = fact_data[holdout_name]['attn_list']

        # Compute mean training attention per layer
        deltas_from_mean = {}
        for li in range(n_layers):
            train_attns = [fact_data[n]['attn_list'][li] for n in train_names]
            mean_train = np.mean(train_attns, axis=0)
            deltas_from_mean[li] = mean_train - holdout_attn[li]

        # Scale the deltas: instead of using the raw mean, project
        # through the manifold using the GENERIC answer direction
        deltas_generic = {}
        for li in range(n_layers):
            train_deltas = np.zeros((len(train_names), holdout_attn[li].shape[0]))
            for i, tn in enumerate(train_names):
                train_deltas[i] = fact_data[tn]['attn_list'][li] - holdout_attn[li]

            if train_deltas.shape[0] > 1:
                U, S, Vt = np.linalg.svd(train_deltas, full_matrices=False)
                cum_var = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
                k = min(int(np.searchsorted(cum_var, 0.9)) + 1, len(S))
                basis = Vt[:k]
                # Project GENERIC answer direction onto subspace
                coeffs = basis @ mean_answer_dir
                mean_delta_norm = np.mean(np.linalg.norm(train_deltas, axis=1))
                proj_delta = (basis.T @ coeffs)
                proj_delta = proj_delta / (np.linalg.norm(proj_delta) + 1e-10) * mean_delta_norm
                deltas_generic[li] = proj_delta
            else:
                deltas_generic[li] = train_deltas[0]

        # Test: mean strategy
        logits_mean = run_with_deltas(engine, holdout_prompt_ids, deltas_from_mean, all_layers)
        rank_mean, _ = get_rank(logits_mean, holdout_answer, tokenizer)
        top1_mean = tokenizer.decode([int(np.argmax(logits_mean))])

        # Test: generic manifold projection
        logits_generic = run_with_deltas(engine, holdout_prompt_ids, deltas_generic, all_layers)
        rank_generic, _ = get_rank(logits_generic, holdout_answer, tokenizer)
        top1_generic = tokenizer.decode([int(np.argmax(logits_generic))])

        # Show top-5 for the generic projection
        top5_idx = np.argsort(logits_generic)[-5:][::-1]
        top5 = [(tokenizer.decode([int(i)]), float(logits_generic[i])) for i in top5_idx]

        print(f"\n  {holdout_name} → '{holdout_answer}'")
        print(f"    Mean transfer:     top1='{top1_mean}', answer rank={rank_mean}")
        print(f"    Generic manifold:  top1='{top1_generic}', answer rank={rank_generic}")
        print(f"    Top-5: {[f'{t[0]}({t[1]:.1f})' for t in top5]}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z18_backward_inference.json')
    save_data = {
        'facts': {n: {'prompt': f['prompt'], 'answer': f['answer']}
                  for n, f in facts.items()},
        'projections': {n: projections[n].tolist() for n in projections},
        'top_layers': [int(x) for x in top_layers],
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
