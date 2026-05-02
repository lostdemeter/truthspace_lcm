"""
Phase 10z12: Value Orthogonality & Direct Injection
====================================================

Finding 114 showed ALL routing heads share ONE d_k direction.
Facts are differentiated by:
  1. RoPE frequencies (which position)
  2. V·W_o projections (what output)

This experiment tests the V·W_o pathway directly:

  PART A: VALUE ORTHOGONALITY
    - For "capital of France/Japan/Germany", extract V·W_o outputs at pos 3
    - Check angles between fact vectors in residual stream
    - Project onto vocabulary embeddings — does "France" → "Paris"?

  PART B: FACT VECTOR ANATOMY
    - Decompose the contribution: residual = Σ_h W_o_h @ V_h @ h[selected_pos]
    - Which heads contribute the ANSWER vs structure?
    - Is the answer concentrated in a few heads or distributed?

  PART C: DIRECT INJECTION
    - For a new fact (e.g. "The capital of Spain is"),
      compute what V·W_o SHOULD produce
    - Inject the computed vector into the residual stream
    - Test if the model outputs "Madrid" without ever being trained on it

If this works, we have demonstrated O(1) learning:
  knowledge = a vector in the V·W_o output space.
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

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# ═══════════════════════════════════════════════════════════════════════
# CAPITAL CITY PROMPTS — same structure, different facts
# ═══════════════════════════════════════════════════════════════════════

CAPITAL_PROMPTS = [
    ('The capital of France is', 'Paris', 3),      # country at pos 3
    ('The capital of Japan is', 'Tokyo', 3),
    ('The capital of Germany is', 'Berlin', 3),
    ('The capital of Italy is', 'Rome', 3),
    ('The capital of Brazil is', 'Brasilia', 3),
    ('The capital of Egypt is', 'Cairo', 3),
]

# For injection test — facts the model should know but we'll verify geometrically
INJECTION_PROMPTS = [
    ('The capital of Spain is', 'Madrid', 3),
    ('The capital of Australia is', 'Canberra', 3),
    ('The capital of Canada is', 'Ottawa', 3),
]


def run_to_layer(engine, prompt_ids, target_layer):
    """Run model forward pass up to (but not through) target_layer.
    Returns the hidden state just before target_layer's attention."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        if layer.layer_idx == target_layer:
            break
        h = layer(h)
    return h


def extract_head_outputs(engine, layer_idx, h_before_layer):
    """Extract per-head attention outputs for the LAST token position.

    For each head:
      1. Compute Q, K with RoPE
      2. Find which position each head attends to (argmax)
      3. Compute V at that position
      4. Project through W_o to get the head's contribution to residual stream

    Returns per-head outputs and metadata.
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads

    normed = rms_norm(h_before_layer, attn.norm_weight)
    seq_len = normed.shape[1]

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

    head_outputs = []
    for hi in range(num_heads):
        # Attention scores for last token
        scores = Q[0, hi, -1, :] @ K_exp[0, hi, :, :].T
        sel_pos = int(np.argmax(scores))

        # Softmax attention weights
        attn_weights = np.exp(scores - np.max(scores))
        attn_weights = attn_weights / np.sum(attn_weights)

        # Value at selected position (hard selection)
        v_selected = V_exp[0, hi, sel_pos, :]  # (head_dim,)

        # Value weighted by attention (soft selection)
        v_soft = attn_weights @ V_exp[0, hi, :, :]  # (head_dim,)

        # Project through W_o to get contribution to residual stream
        # W_o maps (num_heads * head_dim) → hidden_dim
        # For head hi, we need to place v in the right slot
        full_v = np.zeros(num_heads * head_dim, dtype=np.float32)
        full_v[hi * head_dim:(hi + 1) * head_dim] = v_selected
        o_hard = phi_linear(attn.W_o, full_v[np.newaxis, np.newaxis, :])[0, 0, :]

        full_v_soft = np.zeros(num_heads * head_dim, dtype=np.float32)
        full_v_soft[hi * head_dim:(hi + 1) * head_dim] = v_soft
        o_soft = phi_linear(attn.W_o, full_v_soft[np.newaxis, np.newaxis, :])[0, 0, :]

        head_outputs.append({
            'head': hi,
            'sel_pos': sel_pos,
            'attn_weights': attn_weights,
            'v_selected': v_selected,
            'v_soft': v_soft,
            'o_hard': o_hard,      # residual contribution (hard select)
            'o_soft': o_soft,      # residual contribution (soft attn)
            'score_margin': float(scores[sel_pos] - np.sort(scores)[-2]),
        })

    return head_outputs, normed


def get_vocab_projection(engine, vector):
    """Project a residual-stream vector onto the vocabulary.
    Returns top tokens and their logit scores."""
    # PhiLMHead is callable: (batch, seq_len, hidden_dim) → (batch, seq_len, vocab_size)
    logits = engine.lm_head(vector[np.newaxis, np.newaxis, :])[0, 0, :]
    return logits


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    hidden_dim = engine.hidden_dim
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23

    print("\n" + "=" * 72)
    print("  PHASE 10z12: VALUE ORTHOGONALITY & DIRECT INJECTION")
    print("=" * 72)
    print(f"  Layer: {target_layer}")
    print(f"  Hypothesis: facts live in V·W_o output space")
    print(f"  Test: orthogonality, vocab projection, direct injection")

    # ══════════════════════════════════════════════════════════════════
    # PART A: EXTRACT FACT VECTORS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part A: Extract V·W_o fact vectors for capital cities")
    print("─" * 72)

    fact_data = {}

    for prompt, expected, country_pos in CAPITAL_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]

        h_before = run_to_layer(engine, p_ids, target_layer)
        head_outputs, normed = extract_head_outputs(engine, target_layer, h_before)

        # Store per-head outputs
        country_token = tokens[country_pos]
        print(f"\n  \"{prompt}\"  (expect: {expected})")
        print(f"    Tokens: {tokens}")
        print(f"    Country token: pos {country_pos} = '{country_token}'")

        # Sum up all head contributions (this is what attention adds to residual)
        total_attn_output = np.zeros(hidden_dim, dtype=np.float32)
        for ho in head_outputs:
            total_attn_output += ho['o_soft']

        # Show Head 6 specifically (known capital router from F39-40)
        h6 = head_outputs[6]
        print(f"    Head 6: attends to pos {h6['sel_pos']} ('{tokens[h6['sel_pos']]}'), "
              f"margin={h6['score_margin']:.3f}")

        # Vocab projection of Head 6's output alone
        h6_logits = get_vocab_projection(engine, h6['o_soft'])
        top5_idx = np.argsort(h6_logits)[::-1][:5]
        print(f"    Head 6 → vocab top-5:")
        for idx in top5_idx:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {h6_logits[idx]:+8.3f}  '{tok}'")

        # Vocab projection of total attention output
        total_logits = get_vocab_projection(engine, total_attn_output)
        top5_total = np.argsort(total_logits)[::-1][:5]
        print(f"    Total L23 attn → vocab top-5:")
        for idx in top5_total:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {total_logits[idx]:+8.3f}  '{tok}'")

        # Also get the full model's next-token prediction for comparison
        h_full = h_before.copy()
        for layer in engine.layers:
            if layer.layer_idx >= target_layer:
                h_full = layer(h_full)
        full_logits = get_vocab_projection(engine, h_full[0, -1, :])
        top5_full = np.argsort(full_logits)[::-1][:5]
        print(f"    Full model → vocab top-5:")
        for idx in top5_full:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {full_logits[idx]:+8.3f}  '{tok}'")

        fact_data[prompt] = {
            'prompt': prompt,
            'expected': expected,
            'tokens': tokens,
            'country_pos': country_pos,
            'head_outputs': head_outputs,
            'total_attn_output': total_attn_output,
            'normed': normed,
            'h_before': h_before,
        }

    # ══════════════════════════════════════════════════════════════════
    # PART B: ORTHOGONALITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part B: Fact vector orthogonality")
    print("─" * 72)

    prompts = list(fact_data.keys())

    # Compare Head 6 outputs (the known capital-router)
    print(f"\n  Head 6 output vectors (o_soft) — angles between facts:")
    h6_vecs = {}
    for p in prompts:
        h6_vecs[p] = fact_data[p]['head_outputs'][6]['o_soft']

    for i, p1 in enumerate(prompts):
        for j, p2 in enumerate(prompts):
            if j <= i:
                continue
            v1 = h6_vecs[p1]
            v2 = h6_vecs[p2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-20)
            angle = np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi
            name1 = fact_data[p1]['expected']
            name2 = fact_data[p2]['expected']
            print(f"    {name1:>8s} ↔ {name2:<8s}: {angle:6.1f}°  cos={cos:+.4f}")

    # Compare total attention outputs
    print(f"\n  Total attention output — angles between facts:")
    total_vecs = {}
    for p in prompts:
        total_vecs[p] = fact_data[p]['total_attn_output']

    for i, p1 in enumerate(prompts):
        for j, p2 in enumerate(prompts):
            if j <= i:
                continue
            v1 = total_vecs[p1]
            v2 = total_vecs[p2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-20)
            angle = np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi
            name1 = fact_data[p1]['expected']
            name2 = fact_data[p2]['expected']
            print(f"    {name1:>8s} ↔ {name2:<8s}: {angle:6.1f}°  cos={cos:+.4f}")

    # SVD of fact vectors — how many dimensions do they span?
    print(f"\n  SVD of Head 6 fact vectors ({len(prompts)} vectors × {hidden_dim} dims):")
    h6_matrix = np.array([h6_vecs[p] for p in prompts])
    U_f, S_f, Vt_f = np.linalg.svd(h6_matrix, full_matrices=False)
    total_var = np.sum(S_f**2)
    for i in range(len(prompts)):
        cumvar = np.sum(S_f[:i+1]**2) / total_var * 100
        phi_level = np.log(S_f[i]) / LOG_PHI if S_f[i] > 1e-10 else float('-inf')
        print(f"    σ[{i}] = {S_f[i]:10.4f}  cumvar={cumvar:5.1f}%  φ-level={phi_level:+.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PART C: PER-HEAD CONTRIBUTION ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part C: Which heads contribute the ANSWER?")
    print("─" * 72)

    for prompt, expected, country_pos in CAPITAL_PROMPTS[:3]:  # France, Japan, Germany
        fd = fact_data[prompt]
        print(f"\n  \"{prompt}\"  → expected '{expected}'")

        # For each head, project its output onto vocab and check if expected appears
        expected_ids = tokenizer.encode(expected)
        if len(expected_ids) > 0:
            exp_id = expected_ids[0]  # first token of expected answer
            exp_token = tokenizer.decode_token(exp_id)
            print(f"    Expected token: '{exp_token}' (id={exp_id})")

            head_contributions = []
            for ho in fd['head_outputs']:
                logits = get_vocab_projection(engine, ho['o_soft'])
                exp_logit = float(logits[exp_id])
                top_id = int(np.argmax(logits))
                top_token = tokenizer.decode_token(top_id)
                top_logit = float(logits[top_id])
                head_contributions.append({
                    'head': ho['head'],
                    'sel_pos': ho['sel_pos'],
                    'exp_logit': exp_logit,
                    'top_token': top_token,
                    'top_logit': top_logit,
                })

            # Sort by contribution to expected answer
            head_contributions.sort(key=lambda x: -x['exp_logit'])
            print(f"    Top 10 heads contributing to '{exp_token}':")
            for hc in head_contributions[:10]:
                marker = "★" if hc['exp_logit'] == head_contributions[0]['exp_logit'] else " "
                print(f"      {marker} Head {hc['head']:2d}: "
                      f"logit('{exp_token}')={hc['exp_logit']:+7.3f}  "
                      f"sel_pos={hc['sel_pos']}  "
                      f"top='{hc['top_token']}' ({hc['top_logit']:+.3f})")

    # ══════════════════════════════════════════════════════════════════
    # PART D: FACT VECTOR ARITHMETIC
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part D: Fact vector arithmetic")
    print("─" * 72)

    # Can we do France - Japan + Tokyo → Paris?
    # Or more precisely: does total_attn(France) - total_attn(Japan) + embed(Tokyo) → Paris?
    if len(prompts) >= 3:
        # Difference vector between France and Japan attention outputs
        v_france = fact_data[prompts[0]]['total_attn_output']
        v_japan = fact_data[prompts[1]]['total_attn_output']
        v_germany = fact_data[prompts[2]]['total_attn_output']

        diff_fj = v_france - v_japan
        diff_fg = v_france - v_germany
        diff_jg = v_japan - v_germany

        print(f"\n  Difference vectors (total attention output):")
        print(f"    ||France - Japan||  = {np.linalg.norm(diff_fj):.4f}")
        print(f"    ||France - Germany||= {np.linalg.norm(diff_fg):.4f}")
        print(f"    ||Japan - Germany|| = {np.linalg.norm(diff_jg):.4f}")

        # Project differences onto vocab
        print(f"\n  France_vec - Japan_vec → vocab top-5:")
        diff_logits = get_vocab_projection(engine, diff_fj)
        top5 = np.argsort(diff_logits)[::-1][:5]
        for idx in top5:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {diff_logits[idx]:+8.3f}  '{tok}'")

        bot5 = np.argsort(diff_logits)[:5]
        print(f"    (bottom 5 — what Japan contributes more than France):")
        for idx in bot5:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {diff_logits[idx]:+8.3f}  '{tok}'")

    # ══════════════════════════════════════════════════════════════════
    # PART E: DIRECT INJECTION TEST
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part E: Direct injection — can we add a fact geometrically?")
    print("─" * 72)

    for prompt, expected, country_pos in INJECTION_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]

        print(f"\n  Test: \"{prompt}\"  → expected '{expected}'")
        print(f"    Tokens: {tokens}")

        # 1. Run full model normally — what does it predict?
        h_before = run_to_layer(engine, p_ids, target_layer)
        h_full = h_before.copy()
        for layer in engine.layers:
            if layer.layer_idx >= target_layer:
                h_full = layer(h_full)
        normal_logits = get_vocab_projection(engine, h_full[0, -1, :])
        top5_normal = np.argsort(normal_logits)[::-1][:5]
        print(f"    Normal model prediction:")
        for idx in top5_normal:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {normal_logits[idx]:+8.3f}  '{tok}'")

        expected_ids = tokenizer.encode(expected)
        if expected_ids:
            exp_id = expected_ids[0]
            exp_tok = tokenizer.decode_token(exp_id)
            exp_rank = int(np.sum(normal_logits > normal_logits[exp_id])) + 1
            print(f"    '{exp_tok}' rank: {exp_rank}, logit: {normal_logits[exp_id]:+.3f}")

        # 2. Extract this prompt's head outputs at Layer 23
        head_outputs, normed = extract_head_outputs(engine, target_layer, h_before)

        # 3. Compute the attention output that Layer 23 would produce
        actual_attn_output = np.zeros(hidden_dim, dtype=np.float32)
        for ho in head_outputs:
            actual_attn_output += ho['o_soft']

        # 4. Check: if we use a KNOWN fact's attention output instead,
        #    does the model produce THAT fact's answer?
        #    This tests whether the V·W_o vector IS the fact.
        if len(fact_data) >= 1:
            # Take France's attention output and inject it
            france_prompt = CAPITAL_PROMPTS[0][0]
            france_attn = fact_data[france_prompt]['total_attn_output']

            # Run remaining layers with France's attention output replacing
            # this prompt's Layer 23 attention output
            #
            # After layer attention:
            #   h_post_attn = h_before + attn_output (residual connection)
            # Then MLP:
            #   h_post_mlp = h_post_attn + mlp(h_post_attn)
            #
            # We replace attn_output with france_attn_output.

            layer23 = engine.layers[target_layer]
            attn23 = layer23.attention

            # Residual after attention with INJECTED fact vector
            normed_h = rms_norm(h_before, attn23.norm_weight)
            h_injected = h_before.copy()
            h_injected[0, -1, :] += france_attn - actual_attn_output  # swap attn output

            # Continue through rest of Layer 23 (MLP) and subsequent layers
            # We need to handle the MLP part of Layer 23
            mlp = layer23.mlp
            normed_for_mlp = rms_norm(h_injected, mlp.norm_weight)
            gate_out = phi_linear(mlp.W_gate, normed_for_mlp)
            up_out = phi_linear(mlp.W_up, normed_for_mlp)
            mlp_out = phi_linear(mlp.W_down, phi_silu(gate_out) * up_out)
            h_after_23 = h_injected + mlp_out

            # Run layers 24+
            for layer in engine.layers:
                if layer.layer_idx > target_layer:
                    h_after_23 = layer(h_after_23)

            injected_logits = get_vocab_projection(engine, h_after_23[0, -1, :])
            top5_inj = np.argsort(injected_logits)[::-1][:5]
            print(f"\n    WITH France's L23 attn vector injected:")
            for idx in top5_inj:
                tok = tokenizer.decode_token(int(idx))
                print(f"      {injected_logits[idx]:+8.3f}  '{tok}'")

            # Check if "Paris" now appears
            paris_ids = tokenizer.encode("Paris")
            if paris_ids:
                paris_id = paris_ids[0]
                paris_tok = tokenizer.decode_token(paris_id)
                paris_rank_normal = int(np.sum(normal_logits > normal_logits[paris_id])) + 1
                paris_rank_injected = int(np.sum(injected_logits > injected_logits[paris_id])) + 1
                print(f"    '{paris_tok}' rank: {paris_rank_normal} → {paris_rank_injected} "
                      f"(logit: {normal_logits[paris_id]:+.3f} → {injected_logits[paris_id]:+.3f})")

            # Also check the expected answer
            if expected_ids:
                exp_rank_inj = int(np.sum(injected_logits > injected_logits[exp_id])) + 1
                print(f"    '{exp_tok}' rank: {exp_rank} → {exp_rank_inj} "
                      f"(logit: {normal_logits[exp_id]:+.3f} → {injected_logits[exp_id]:+.3f})")

    # ══════════════════════════════════════════════════════════════════
    # PART F: CROSS-FACT INJECTION — swap fact vectors between prompts
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part F: Cross-fact injection — can we swap country→capital mappings?")
    print("─" * 72)

    # For each pair of known facts, swap their Layer 23 attention outputs
    # France with Japan's vector → should predict Tokyo instead of Paris
    swap_pairs = [
        (CAPITAL_PROMPTS[0], CAPITAL_PROMPTS[1]),  # France ↔ Japan
        (CAPITAL_PROMPTS[0], CAPITAL_PROMPTS[2]),  # France ↔ Germany
        (CAPITAL_PROMPTS[1], CAPITAL_PROMPTS[2]),  # Japan ↔ Germany
    ]

    for (prompt_a, expected_a, pos_a), (prompt_b, expected_b, pos_b) in swap_pairs:
        print(f"\n  Swap: \"{prompt_a}\" gets {expected_b}'s vector")

        fd_a = fact_data[prompt_a]
        fd_b = fact_data[prompt_b]

        h_before_a = fd_a['h_before'].copy()
        attn_a = fd_a['total_attn_output']
        attn_b = fd_b['total_attn_output']

        # Inject B's attention output into A's residual stream
        layer23 = engine.layers[target_layer]
        attn23 = layer23.attention

        h_swapped = h_before_a.copy()
        h_swapped[0, -1, :] += attn_b - attn_a  # replace A's attn with B's

        # Layer 23 MLP
        mlp = layer23.mlp
        normed_for_mlp = rms_norm(h_swapped, mlp.norm_weight)
        gate_out = phi_linear(mlp.W_gate, normed_for_mlp)
        up_out = phi_linear(mlp.W_up, normed_for_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate_out) * up_out)
        h_after = h_swapped + mlp_out

        # Run layers 24+
        for layer in engine.layers:
            if layer.layer_idx > target_layer:
                h_after = layer(h_after)

        swap_logits = get_vocab_projection(engine, h_after[0, -1, :])
        top5_swap = np.argsort(swap_logits)[::-1][:5]
        print(f"    Prediction with {expected_b}'s vector:")
        for idx in top5_swap:
            tok = tokenizer.decode_token(int(idx))
            print(f"      {swap_logits[idx]:+8.3f}  '{tok}'")

        # Check original and swapped answers
        for label, expected_str in [(f"Original ({expected_a})", expected_a),
                                     (f"Swapped  ({expected_b})", expected_b)]:
            e_ids = tokenizer.encode(expected_str)
            if e_ids:
                e_id = e_ids[0]
                e_tok = tokenizer.decode_token(e_id)
                e_rank = int(np.sum(swap_logits > swap_logits[e_id])) + 1
                print(f"    {label}: '{e_tok}' rank={e_rank} logit={swap_logits[e_id]:+.3f}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SAVING RESULTS")
    print("=" * 72)

    # Collect angle data
    h6_angles = {}
    for i, p1 in enumerate(prompts):
        for j, p2 in enumerate(prompts):
            if j <= i:
                continue
            v1 = h6_vecs[p1]
            v2 = h6_vecs[p2]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-20))
            angle = float(np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi)
            h6_angles[f"{fact_data[p1]['expected']}-{fact_data[p2]['expected']}"] = {
                'cos': cos, 'angle': angle
            }

    save_data = {
        'layer': target_layer,
        'prompts': [{'prompt': p, 'expected': e, 'pos': pos}
                    for p, e, pos in CAPITAL_PROMPTS],
        'h6_angles': h6_angles,
        'h6_svd': S_f.tolist(),
    }

    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z12_value_injection.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Final summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Layer {target_layer} V·W_o analysis complete")
    print(f"  Head 6 fact vector SVD: {S_f[:3].tolist()}")
    print(f"  Injection test: swap attention outputs between facts")
    print(f"  If swapping works → knowledge IS the V·W_o vector")


if __name__ == '__main__':
    main()
