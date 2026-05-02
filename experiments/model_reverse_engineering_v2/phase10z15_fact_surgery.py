"""
Phase 10z15: Fact Surgery with Clean Inference
================================================

F115 showed V·W_o vectors encode facts, but the full-model predictions
were corrupted by a missing final RMS norm in get_vocab_projection.

Phase10z14 confirmed: the model WORKS. ' Paris' = rank 0.

This script:
  1. Fixes the vocab projection (includes final RMS norm)
  2. Re-validates injection with clean absolute predictions
  3. Tests true fact surgery:
     a) ADD a fact: inject V·W_o vector → does the model predict correctly?
     b) REMOVE a fact: zero V·W_o contribution → does the model forget?
     c) SWAP a fact: replace one country's vector with another → does it switch?

If fact surgery works with clean outputs, we have proven:
  knowledge = V·W_o vector on the d_k axis at specific RoPE frequency
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


CAPITAL_PROMPTS = [
    ('The capital of France is', 'Paris', 3),
    ('The capital of Japan is', 'Tokyo', 3),
    ('The capital of Germany is', 'Berlin', 3),
    ('The capital of Italy is', 'Rome', 3),
    ('The capital of Brazil is', 'Brasilia', 3),
    ('The capital of Egypt is', 'Cairo', 3),
]

INJECTION_PROMPTS = [
    ('The capital of Spain is', 'Madrid', 3),
    ('The capital of Australia is', 'Canberra', 3),
    ('The capital of Canada is', 'Ottawa', 3),
]


def get_logits(engine, hidden_3d):
    """Project hidden state to logits WITH final RMS norm (correct path)."""
    normed = rms_norm(hidden_3d, engine.final_norm_weight)
    logits = engine.lm_head(normed)
    return logits[0, -1, :]


def run_to_layer(engine, prompt_ids, target_layer):
    """Run model up to (not through) target_layer."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        if layer.layer_idx == target_layer:
            break
        h = layer(h)
    return h


def run_from_layer(engine, h, from_layer):
    """Run model from from_layer (inclusive) to end, return logits."""
    for layer in engine.layers:
        if layer.layer_idx >= from_layer:
            h = layer(h)
    return get_logits(engine, h)


def run_layer_attn_only(engine, layer_idx, h_before):
    """Run just the attention part of a layer (not MLP).
    Returns the attention output delta (what gets added to residual)."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    head_dim = attn.head_dim
    seq_len = h_before.shape[1]

    normed = rms_norm(h_before, attn.norm_weight)

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
    attn_weights = phi_softmax(scores, axis=-1)

    attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_exp)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    attn_output = phi_linear(attn.W_o, attn_output)

    return attn_output, attn_weights


def extract_per_head_attn_output(engine, layer_idx, h_before):
    """Extract attention output per head for the last token.

    Returns:
        per_head: list of (hidden_dim,) arrays — each head's W_o projection
        attn_weights: (1, num_heads, seq_len, seq_len) attention weight matrix
        total_attn: (1, seq_len, hidden_dim) total attention output
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    head_dim = attn.head_dim
    seq_len = h_before.shape[1]

    normed = rms_norm(h_before, attn.norm_weight)

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
    attn_weights = phi_softmax(scores, axis=-1)

    attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_exp)
    # attn_output shape: (1, num_heads, seq_len, head_dim)

    # Per-head W_o projections for last token
    per_head = []
    for hi in range(num_heads):
        v_head = attn_output[0, hi, -1, :]  # (head_dim,)
        full_v = np.zeros(num_heads * head_dim, dtype=np.float32)
        full_v[hi * head_dim:(hi + 1) * head_dim] = v_head
        o_head = phi_linear(attn.W_o, full_v[np.newaxis, np.newaxis, :])[0, 0, :]
        per_head.append(o_head)

    # Total attention output (all positions)
    attn_all = attn_output.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    total_attn = phi_linear(attn.W_o, attn_all)

    return per_head, attn_weights, total_attn


def run_layer_with_modified_attn(engine, layer_idx, h_before, attn_delta):
    """Run a layer but add attn_delta to the attention output at the last token.

    attn_delta: (hidden_dim,) — vector to ADD to the attention output for last token.

    Returns h_after (post-MLP with residual).
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    mlp = layer.mlp

    # Normal attention
    attn_output, _ = run_layer_attn_only(engine, layer_idx, h_before)

    # Modify attention output at last token
    attn_modified = attn_output.copy()
    attn_modified[0, -1, :] += attn_delta

    # Residual after modified attention
    h_post_attn = h_before + attn_modified

    # MLP (normal)
    normed_for_mlp = rms_norm(h_post_attn, mlp.norm_weight)
    gate = phi_linear(mlp.W_gate, normed_for_mlp)
    up = phi_linear(mlp.W_up, normed_for_mlp)
    mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
    h_after = h_post_attn + mlp_out

    return h_after


def show_top_k(tokenizer, logits, k=5, prefix=""):
    """Print top-k predictions."""
    top_idx = np.argsort(logits)[-k:][::-1]
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        print(f"    {prefix}{logits[idx]:+8.3f}  '{tok}'")


def show_target(tokenizer, logits, target_str, label=""):
    """Show rank and logit for a target token."""
    tids = tokenizer.encode(target_str)
    if not tids:
        return None, None
    tid = tids[0]
    tok = tokenizer.decode([tid])
    rank = int(np.sum(logits > logits[tid]))
    print(f"    {label}'{tok}': rank={rank}, logit={logits[tid]:+.3f}")
    return rank, float(logits[tid])


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    hidden_dim = engine.hidden_dim
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23

    print("\n" + "=" * 72)
    print("  PHASE 10z15: FACT SURGERY WITH CLEAN INFERENCE")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════
    # PART 1: VALIDATE — does the model predict correctly?
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print("  Part 1: Baseline validation — model predicts correctly?")
    print("─" * 72)

    baseline_results = {}
    for prompt, expected, pos in CAPITAL_PROMPTS + INJECTION_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        logits = run_from_layer(engine,
                                engine.embedding(p_ids)[np.newaxis, :, :],
                                from_layer=0)
        rank, logit = None, None
        for target in [' ' + expected, expected]:
            tids = tokenizer.encode(target)
            if tids:
                tid = tids[0]
                rank = int(np.sum(logits > logits[tid]))
                logit = float(logits[tid])
                break

        top1_idx = int(np.argmax(logits))
        top1_tok = tokenizer.decode([top1_idx])
        top1_logit = float(logits[top1_idx])
        marker = "✓" if rank is not None and rank == 0 else "✗"
        print(f"  {marker} '{prompt}' → '{top1_tok}' ({top1_logit:.2f})  "
              f"['{expected}' rank={rank}]")
        baseline_results[prompt] = {
            'expected': expected, 'rank': rank, 'logit': logit,
            'top1': top1_tok, 'correct': rank == 0
        }

    n_correct = sum(1 for v in baseline_results.values() if v['correct'])
    print(f"\n  Baseline: {n_correct}/{len(baseline_results)} correct")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: EXTRACT — per-head contributions at Layer 23
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print(f"  Part 2: Extract L{target_layer} attention outputs for known facts")
    print("─" * 72)

    fact_data = {}
    for prompt, expected, pos in CAPITAL_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([tid]) for tid in p_ids]

        h_before = run_to_layer(engine, p_ids, target_layer)
        per_head, attn_weights, total_attn = extract_per_head_attn_output(
            engine, target_layer, h_before)

        # Which head contributes most to the expected answer?
        target_tok = ' ' + expected
        target_ids = tokenizer.encode(target_tok)
        if not target_ids:
            target_tok = expected
            target_ids = tokenizer.encode(target_tok)
        target_id = target_ids[0] if target_ids else None

        head_scores = []
        if target_id is not None:
            for hi, h_out in enumerate(per_head):
                # Project through final norm + LM head to get this head's logit contribution
                # Note: for per-head, we project the raw vector (no norm — it's a delta)
                logits_h = engine.lm_head(h_out[np.newaxis, np.newaxis, :])[0, 0, :]
                head_scores.append((hi, float(logits_h[target_id])))
            head_scores.sort(key=lambda x: -x[1])

        print(f"\n  '{prompt}' → '{expected}'")
        print(f"    Top contributing heads to '{target_tok}':")
        for hi, score in head_scores[:5]:
            argmax_pos = int(np.argmax(attn_weights[0, hi, -1, :]))
            print(f"      Head {hi:2d}: logit={score:+7.3f}  "
                  f"attends_to=pos {argmax_pos} ('{tokens[argmax_pos]}')")

        fact_data[prompt] = {
            'expected': expected,
            'tokens': tokens,
            'h_before': h_before,
            'per_head': per_head,
            'attn_weights': attn_weights,
            'total_attn': total_attn,
            'head_scores': head_scores,
        }

    # ══════════════════════════════════════════════════════════════════
    # PART 3: SURGERY — swap, add, remove facts
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print(f"  Part 3: FACT SURGERY — swap country vectors at L{target_layer}")
    print("─" * 72)

    swap_tests = [
        # (host_prompt, donor_prompt) — donor's attn replaces host's
        (CAPITAL_PROMPTS[0], CAPITAL_PROMPTS[1]),  # France gets Japan's vector
        (CAPITAL_PROMPTS[0], CAPITAL_PROMPTS[2]),  # France gets Germany's vector
        (CAPITAL_PROMPTS[1], CAPITAL_PROMPTS[0]),  # Japan gets France's vector
        (CAPITAL_PROMPTS[2], CAPITAL_PROMPTS[0]),  # Germany gets France's vector
    ]

    swap_results = []
    for (host_prompt, host_exp, host_pos), (donor_prompt, donor_exp, donor_pos) in swap_tests:
        fd_host = fact_data[host_prompt]
        fd_donor = fact_data[donor_prompt]

        # Compute delta: donor's total attn - host's total attn (last token only)
        attn_delta = (fd_donor['total_attn'][0, -1, :] -
                      fd_host['total_attn'][0, -1, :])

        # Run layer 23 with modified attention, then layers 24-27
        h_before = fd_host['h_before'].copy()
        h_modified = run_layer_with_modified_attn(
            engine, target_layer, h_before, attn_delta)

        # Run remaining layers
        logits_swapped = run_from_layer(engine, h_modified, from_layer=target_layer + 1)

        # Also get normal prediction for comparison
        logits_normal = run_from_layer(engine, h_before, from_layer=target_layer)

        print(f"\n  SWAP: '{host_prompt}' gets {donor_exp}'s L{target_layer} vector")

        # Normal prediction
        top1_n = tokenizer.decode([int(np.argmax(logits_normal))])
        print(f"    Normal:  top-1 = '{top1_n}'")
        show_top_k(tokenizer, logits_normal, k=3, prefix="  ")

        # Swapped prediction
        top1_s = tokenizer.decode([int(np.argmax(logits_swapped))])
        print(f"    Swapped: top-1 = '{top1_s}'")
        show_top_k(tokenizer, logits_swapped, k=3, prefix="  ")

        # Track target tokens
        host_rank_n, _ = show_target(tokenizer, logits_normal, ' ' + host_exp,
                                     f"Normal {host_exp}: ")
        host_rank_s, _ = show_target(tokenizer, logits_swapped, ' ' + host_exp,
                                     f"Swapped {host_exp}: ")
        donor_rank_n, _ = show_target(tokenizer, logits_normal, ' ' + donor_exp,
                                      f"Normal {donor_exp}: ")
        donor_rank_s, _ = show_target(tokenizer, logits_swapped, ' ' + donor_exp,
                                      f"Swapped {donor_exp}: ")

        success = (donor_rank_s is not None and host_rank_s is not None and
                   donor_rank_s < host_rank_s)
        swap_results.append({
            'host': host_exp, 'donor': donor_exp,
            'host_rank_normal': host_rank_n, 'host_rank_swapped': host_rank_s,
            'donor_rank_normal': donor_rank_n, 'donor_rank_swapped': donor_rank_s,
            'success': success,
        })
        print(f"    {'✓ SWAP WORKS' if success else '✗ swap failed'}: "
              f"{donor_exp} rank {donor_rank_n} → {donor_rank_s}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: FACT REMOVAL — zero out Layer 23 attention
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print(f"  Part 4: FACT REMOVAL — zero L{target_layer} attention contribution")
    print("─" * 72)

    removal_results = []
    for prompt, expected, pos in CAPITAL_PROMPTS[:3]:
        fd = fact_data[prompt]
        h_before = fd['h_before'].copy()

        # Delta to REMOVE all of Layer 23's attention for the last token
        attn_delta = -fd['total_attn'][0, -1, :]

        h_removed = run_layer_with_modified_attn(
            engine, target_layer, h_before, attn_delta)
        logits_removed = run_from_layer(engine, h_removed, from_layer=target_layer + 1)

        # Normal comparison
        logits_normal = run_from_layer(engine, h_before, from_layer=target_layer)

        print(f"\n  REMOVE L{target_layer} attn: '{prompt}'")
        top1_n = tokenizer.decode([int(np.argmax(logits_normal))])
        top1_r = tokenizer.decode([int(np.argmax(logits_removed))])
        print(f"    Normal:  top-1 = '{top1_n}'")
        show_top_k(tokenizer, logits_normal, k=3, prefix="  ")
        print(f"    Removed: top-1 = '{top1_r}'")
        show_top_k(tokenizer, logits_removed, k=3, prefix="  ")
        rank_n, _ = show_target(tokenizer, logits_normal, ' ' + expected,
                                "Normal: ")
        rank_r, _ = show_target(tokenizer, logits_removed, ' ' + expected,
                                "Removed: ")

        forgotten = rank_r is not None and rank_r > 100
        removal_results.append({
            'prompt': prompt, 'expected': expected,
            'rank_normal': rank_n, 'rank_removed': rank_r,
            'forgotten': forgotten,
        })
        print(f"    {'✓ FACT FORGOTTEN' if forgotten else '○ fact persists'}: "
              f"rank {rank_n} → {rank_r}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: CROSS-PROMPT INJECTION — inject into unknown prompts
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 72)
    print(f"  Part 5: INJECTION — inject France's vector into new prompts")
    print("─" * 72)

    france_fd = fact_data[CAPITAL_PROMPTS[0][0]]
    france_attn_vec = france_fd['total_attn'][0, -1, :].copy()

    injection_results = []
    for prompt, expected, pos in INJECTION_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        h_before = run_to_layer(engine, p_ids, target_layer)

        # Get this prompt's actual L23 attention output
        _, _, actual_attn = extract_per_head_attn_output(engine, target_layer, h_before)
        actual_attn_vec = actual_attn[0, -1, :]

        # Delta: France's vector minus this prompt's vector
        attn_delta = france_attn_vec - actual_attn_vec

        h_injected = run_layer_with_modified_attn(
            engine, target_layer, h_before, attn_delta)
        logits_injected = run_from_layer(engine, h_injected, from_layer=target_layer + 1)

        # Normal comparison
        logits_normal = run_from_layer(engine, h_before, from_layer=target_layer)

        print(f"\n  INJECT France's vector: '{prompt}'")
        top1_n = tokenizer.decode([int(np.argmax(logits_normal))])
        top1_i = tokenizer.decode([int(np.argmax(logits_injected))])
        print(f"    Normal:   top-1 = '{top1_n}'")
        show_top_k(tokenizer, logits_normal, k=3, prefix="  ")
        print(f"    Injected: top-1 = '{top1_i}'")
        show_top_k(tokenizer, logits_injected, k=3, prefix="  ")

        paris_rank_n, _ = show_target(tokenizer, logits_normal, ' Paris',
                                      "Normal Paris: ")
        paris_rank_i, _ = show_target(tokenizer, logits_injected, ' Paris',
                                      "Injected Paris: ")
        exp_rank_n, _ = show_target(tokenizer, logits_normal, ' ' + expected,
                                    f"Normal {expected}: ")
        exp_rank_i, _ = show_target(tokenizer, logits_injected, ' ' + expected,
                                    f"Injected {expected}: ")

        paris_wins = paris_rank_i is not None and paris_rank_i < 10
        injection_results.append({
            'prompt': prompt, 'expected': expected,
            'paris_rank_normal': paris_rank_n, 'paris_rank_injected': paris_rank_i,
            'exp_rank_normal': exp_rank_n, 'exp_rank_injected': exp_rank_i,
            'paris_wins': paris_wins,
        })
        print(f"    {'✓ PARIS INJECTED' if paris_wins else '○ Paris not in top-10'}: "
              f"Paris rank {paris_rank_n} → {paris_rank_i}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    n_swap_ok = sum(1 for r in swap_results if r['success'])
    n_remove_ok = sum(1 for r in removal_results if r['forgotten'])
    n_inject_ok = sum(1 for r in injection_results if r['paris_wins'])

    print(f"\n  Baseline correctness: {n_correct}/{len(baseline_results)}")
    print(f"  Fact SWAP:    {n_swap_ok}/{len(swap_results)} "
          f"(donor answer rises above host answer)")
    print(f"  Fact REMOVE:  {n_remove_ok}/{len(removal_results)} "
          f"(answer rank drops below 100)")
    print(f"  Fact INJECT:  {n_inject_ok}/{len(injection_results)} "
          f"(Paris enters top-10)")

    # ── Save ──
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z15_fact_surgery.json')
    save_data = {
        'target_layer': target_layer,
        'baseline': baseline_results,
        'swap_results': swap_results,
        'removal_results': removal_results,
        'injection_results': injection_results,
        'summary': {
            'baseline_correct': n_correct,
            'baseline_total': len(baseline_results),
            'swap_success': n_swap_ok,
            'swap_total': len(swap_results),
            'remove_success': n_remove_ok,
            'remove_total': len(removal_results),
            'inject_success': n_inject_ok,
            'inject_total': len(injection_results),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
