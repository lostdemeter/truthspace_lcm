"""
Phase 10z17: Novel Memory Injection
=====================================

Can we make the model "remember" something it was never trained on?

Test fact: "NASA landed the first Tesla Model Y on Mars on February 27, 2026."

This fact cannot exist in training data. If we can inject it via V·W_o
manipulation and then query the model, we've demonstrated true memory creation.

Approaches:
  A) LM-head inverse: target token's LM head row IS the hidden-space direction
     that maximizes that token's logit. Inject it as a residual delta.
  B) Donor transfer: find a prompt that naturally produces target tokens,
     extract its per-layer V·W_o, transplant to a date-query prompt.
  C) Compositional: inject multiple token directions simultaneously.

Success = model predicts NASA/Tesla/Mars tokens for a date query.
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


def get_logits(engine, hidden_3d):
    """Project hidden state to logits WITH final RMS norm."""
    normed = rms_norm(hidden_3d, engine.final_norm_weight)
    logits = engine.lm_head(normed)
    return logits[0, -1, :]


def run_full(engine, prompt_ids):
    """Full forward pass, return logits and final hidden state."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        h = layer(h)
    logits = get_logits(engine, h)
    return logits, h


def extract_all_layer_attn(engine, prompt_ids):
    """Run forward pass, cache per-layer attention outputs.

    Returns:
        per_layer_attn: list of (1, seq_len, hidden_dim) attn deltas per layer
        final_hidden: (1, seq_len, hidden_dim) final hidden state
        logits: vocab logits
    """
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

        per_layer_attn.append(attn_output.copy())

        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    logits = get_logits(engine, h)
    return per_layer_attn, h, logits


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

        if li in delta_layers and li in deltas:
            attn_output[0, -1, :] += deltas[li]

        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    return get_logits(engine, h)


def get_rank(logits, token_str, tokenizer):
    tids = tokenizer.encode(token_str)
    if not tids:
        return None, None
    tid = tids[0]
    rank = int(np.sum(logits > logits[tid]))
    return rank, float(logits[tid])


def show_top_k(tokenizer, logits, k=10, prefix="    "):
    top_idx = np.argsort(logits)[-k:][::-1]
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        print(f"{prefix}{logits[idx]:+8.3f}  '{tok}'")


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    hidden_dim = engine.hidden_dim
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z17: NOVEL MEMORY INJECTION")
    print("=" * 72)

    # ── Target tokens for the novel fact ──
    target_tokens = {
        ' NASA': tokenizer.encode(' NASA'),
        ' Tesla': tokenizer.encode(' Tesla'),
        ' Mars': tokenizer.encode(' Mars'),
        ' landed': tokenizer.encode(' landed'),
    }
    print("\n  Target token IDs:")
    for tok_str, tids in target_tokens.items():
        print(f"    '{tok_str}' → {tids}")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: BASELINE — what does the model predict for date queries?
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 1: Baseline predictions for date queries")
    print("─" * 72)

    query_prompts = [
        "On February 27, 2026,",
        "The major event on February 27, 2026 was",
        "On February 27, 2026, NASA",
    ]

    baseline_data = {}
    for prompt in query_prompts:
        p_ids = tokenizer.encode(prompt)
        logits, _ = run_full(engine, p_ids)
        top1_idx = int(np.argmax(logits))
        top1_tok = tokenizer.decode([top1_idx])
        print(f"\n  '{prompt}'")
        print(f"    Top-1: '{top1_tok}' ({logits[top1_idx]:.2f})")
        show_top_k(tokenizer, logits, k=5)
        for tok_str, tids in target_tokens.items():
            if tids:
                rank, logit = get_rank(logits, tok_str, tokenizer)
                print(f"    '{tok_str}': rank={rank}, logit={logit:.2f}")
        baseline_data[prompt] = {'ids': p_ids, 'logits': logits}

    # ══════════════════════════════════════════════════════════════════
    # PART 2: DONOR EXTRACTION — get V·W_o from prompts that produce
    #         our target tokens naturally
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 2: Donor extraction — prompts that naturally produce targets")
    print("─" * 72)

    # These prompts should naturally produce our target tokens
    donor_prompts = {
        'nasa_mars': "The space agency that landed a rover on Mars is",
        'tesla': "The electric car company founded by Elon Musk is",
        'nasa': "The first person to walk on the Moon worked for",
    }

    donor_data = {}
    for label, prompt in donor_prompts.items():
        t1 = time.time()
        p_ids = tokenizer.encode(prompt)
        attn_list, h_final, logits = extract_all_layer_attn(engine, p_ids)
        top1_idx = int(np.argmax(logits))
        top1_tok = tokenizer.decode([top1_idx])
        print(f"\n  [{label}] '{prompt}'")
        print(f"    Top-1: '{top1_tok}' ({logits[top1_idx]:.2f})  ({time.time()-t1:.1f}s)")
        show_top_k(tokenizer, logits, k=5)
        for tok_str, tids in target_tokens.items():
            if tids:
                rank, logit = get_rank(logits, tok_str, tokenizer)
                print(f"    '{tok_str}': rank={rank}, logit={logit:.2f}")
        donor_data[label] = {
            'prompt': prompt, 'ids': p_ids,
            'attn_list': attn_list, 'logits': logits
        }

    # ══════════════════════════════════════════════════════════════════
    # PART 3: APPROACH A — LM head inverse injection
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 3A: LM head inverse — inject token direction into residual")
    print("─" * 72)

    # The LM head weight matrix: logits = hidden_normed @ W_lm^T
    # So the direction in hidden space that maximizes logit for token k
    # is W_lm[k, :] (the k-th row of the LM head weight).
    #
    # We decode the LM head weights to get these directions.
    from phi_geometric.inference.phi_integer import phi_to_float
    lm_signs = engine.lm_head.weight.signs
    lm_exps = engine.lm_head.weight.exponents
    # Decode a few target token rows
    target_directions = {}
    for tok_str, tids in target_tokens.items():
        if tids:
            tid = tids[0]
            # Decode just this row
            row_sign = lm_signs[tid:tid+1, :]
            row_exp = lm_exps[tid:tid+1, :]
            row_float = phi_to_float(row_sign, row_exp)[0]
            target_directions[tok_str] = row_float
            print(f"  LM head row for '{tok_str}' (id={tid}): "
                  f"norm={np.linalg.norm(row_float):.4f}, "
                  f"max={np.max(row_float):.4f}")

    # Now inject these directions into the query prompt
    host_prompt = "On February 27, 2026,"
    host_ids = tokenizer.encode(host_prompt)

    # Try different scales and target combinations
    # Combine NASA + Mars + landed directions
    combined_dir = (target_directions[' NASA'] +
                    target_directions[' Mars'] +
                    target_directions[' landed'])
    combined_dir = combined_dir / np.linalg.norm(combined_dir)

    # We need to figure out the right scale. Let's check what the
    # typical attention output magnitude is for this prompt.
    host_attn_list, host_h, host_logits = extract_all_layer_attn(engine, host_ids)
    attn_norms = [float(np.linalg.norm(a[0, -1, :])) for a in host_attn_list]
    mean_attn_norm = np.mean(attn_norms)
    print(f"\n  Host prompt: '{host_prompt}'")
    print(f"  Mean attn output norm: {mean_attn_norm:.2f}")
    print(f"  Per-layer attn norms: {[f'{n:.1f}' for n in attn_norms]}")

    # Key layers from F117
    key_layers = {9, 22, 23, 25, 27}
    all_layers = set(range(n_layers))

    # Try various scales
    print(f"\n  Injecting combined (NASA+Mars+landed) direction:")
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        delta_vec = combined_dir * scale * mean_attn_norm
        deltas = {li: delta_vec for li in range(n_layers)}

        logits_inj = run_with_deltas(engine, host_ids, deltas, all_layers)
        top1_idx = int(np.argmax(logits_inj))
        top1_tok = tokenizer.decode([top1_idx])

        nasa_rank, nasa_logit = get_rank(logits_inj, ' NASA', tokenizer)
        mars_rank, mars_logit = get_rank(logits_inj, ' Mars', tokenizer)
        landed_rank, landed_logit = get_rank(logits_inj, ' landed', tokenizer)

        print(f"    scale={scale:5.1f}: top1='{top1_tok}'  "
              f"NASA={nasa_rank}  Mars={mars_rank}  landed={landed_rank}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3B: Donor transfer — transplant attention from NASA prompt
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 3B: Donor transfer — transplant NASA prompt's attention")
    print("─" * 72)

    # Use the nasa_mars donor's per-layer attention as the injection source
    # Delta = donor_attn - host_attn at each layer (last token)
    best_donor = 'nasa_mars'
    donor_attn = donor_data[best_donor]['attn_list']

    # Compute deltas
    deltas_donor = {}
    for li in range(n_layers):
        d = donor_attn[li][0, -1, :] - host_attn_list[li][0, -1, :]
        deltas_donor[li] = d

    # Test with all layers
    print(f"\n  Donor: '{donor_data[best_donor]['prompt']}'")
    print(f"  Host:  '{host_prompt}'")

    for layer_set_name, layer_set in [
        ("Key 5 (9,22,23,25,27)", key_layers),
        ("Late (21-27)", set(range(21, 28))),
        ("ALL (0-27)", all_layers),
    ]:
        logits_donor = run_with_deltas(engine, host_ids, deltas_donor, layer_set)
        top1_idx = int(np.argmax(logits_donor))
        top1_tok = tokenizer.decode([top1_idx])

        print(f"\n  {layer_set_name}:")
        print(f"    Top-1: '{top1_tok}'")
        show_top_k(tokenizer, logits_donor, k=5)
        for tok_str in [' NASA', ' Mars', ' landed', ' Tesla']:
            rank, logit = get_rank(logits_donor, tok_str, tokenizer)
            print(f"    '{tok_str}': rank={rank}, logit={logit:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3C: Blended donor — combine NASA and Tesla donors
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 3C: Blended donors — NASA + Tesla attention combined")
    print("─" * 72)

    if 'tesla' in donor_data:
        tesla_attn = donor_data['tesla']['attn_list']

        for alpha in [0.5, 0.7, 1.0]:
            deltas_blend = {}
            for li in range(n_layers):
                d_nasa = donor_attn[li][0, -1, :] - host_attn_list[li][0, -1, :]
                d_tesla = tesla_attn[li][0, -1, :] - host_attn_list[li][0, -1, :]
                deltas_blend[li] = alpha * d_nasa + (1 - alpha) * d_tesla

            logits_blend = run_with_deltas(engine, host_ids, deltas_blend, all_layers)
            top1_idx = int(np.argmax(logits_blend))
            top1_tok = tokenizer.decode([top1_idx])

            print(f"\n  alpha={alpha:.1f} (NASA={alpha:.0%}, Tesla={1-alpha:.0%}):")
            print(f"    Top-1: '{top1_tok}'")
            show_top_k(tokenizer, logits_blend, k=5)
            for tok_str in [' NASA', ' Mars', ' landed', ' Tesla']:
                rank, logit = get_rank(logits_blend, tok_str, tokenizer)
                print(f"    '{tok_str}': rank={rank}, logit={logit:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: VERIFICATION — does it work on related queries?
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 4: Verification — test with varied query phrasings")
    print("─" * 72)

    # Use the best injection method found above (all-layer donor transfer)
    verify_prompts = [
        "On February 27, 2026,",
        "The major event on February 27, 2026 was",
        "On February 27, 2026, NASA",
    ]

    for prompt in verify_prompts:
        p_ids = tokenizer.encode(prompt)

        # We need to extract this prompt's own attention to compute deltas
        v_attn, _, v_logits_normal = extract_all_layer_attn(engine, p_ids)

        # Compute deltas from nasa_mars donor
        # Note: donor and host have different seq_len, so we use last-token only
        v_deltas = {}
        for li in range(n_layers):
            d = donor_attn[li][0, -1, :] - v_attn[li][0, -1, :]
            v_deltas[li] = d

        logits_injected = run_with_deltas(engine, p_ids, v_deltas, all_layers)

        print(f"\n  '{prompt}'")
        top1_n = tokenizer.decode([int(np.argmax(v_logits_normal))])
        top1_i = tokenizer.decode([int(np.argmax(logits_injected))])
        print(f"    Normal:   '{top1_n}'")
        show_top_k(tokenizer, v_logits_normal, k=3, prefix="      ")
        print(f"    Injected: '{top1_i}'")
        show_top_k(tokenizer, logits_injected, k=5, prefix="      ")

        for tok_str in [' NASA', ' Mars', ' landed', ' Tesla', ' rover']:
            rank_n, _ = get_rank(v_logits_normal, tok_str, tokenizer)
            rank_i, logit_i = get_rank(logits_injected, tok_str, tokenizer)
            if rank_n is not None:
                print(f"    '{tok_str}': rank {rank_n} → {rank_i}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z17_novel_memory.json')
    save_data = {
        'novel_fact': 'NASA landed the first Tesla Model Y on Mars on February 27, 2026',
        'donor_prompts': {k: v['prompt'] for k, v in donor_data.items()},
        'target_tokens': {k: v for k, v in target_tokens.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
