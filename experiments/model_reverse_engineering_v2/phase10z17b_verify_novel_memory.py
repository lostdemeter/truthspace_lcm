"""
Phase 10z17b: Verify Novel Memory via LM Head Inverse
=======================================================

Phase 10z17 Part 3A showed that injecting LM head weight rows into the
residual stream at all layers makes the model predict target tokens:

  "On February 27, 2026," → Mars=rank 0, NASA=rank 1, landed=rank 2

This script verifies the result across multiple query phrasings and
also tests adding Tesla to the mix.
"""

import sys
import os
import numpy as np
import time
import gc

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
    """Run forward pass, cache per-layer attention outputs."""
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

    return per_layer_attn, h


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


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    all_layers = set(range(n_layers))
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z17b: VERIFY NOVEL MEMORY — LM HEAD INVERSE")
    print("=" * 72)

    # ── Decode LM head rows for target tokens ──
    target_tokens = {
        ' NASA': tokenizer.encode(' NASA')[0],
        ' Tesla': tokenizer.encode(' Tesla')[0],
        ' Mars': tokenizer.encode(' Mars')[0],
        ' landed': tokenizer.encode(' landed')[0],
        ' Model': tokenizer.encode(' Model')[0],
    }

    lm_signs = engine.lm_head.weight.signs
    lm_exps = engine.lm_head.weight.exponents

    lm_rows = {}
    for tok_str, tid in target_tokens.items():
        row = phi_to_float(lm_signs[tid:tid+1, :], lm_exps[tid:tid+1, :])[0]
        lm_rows[tok_str] = row
        print(f"  LM row '{tok_str}' (id={tid}): norm={np.linalg.norm(row):.4f}")

    # ── Injection combos ──
    combos = {
        'NASA+Mars+landed': [' NASA', ' Mars', ' landed'],
        'NASA+Tesla+Mars+landed': [' NASA', ' Tesla', ' Mars', ' landed'],
        'NASA+Mars': [' NASA', ' Mars'],
        'NASA only': [' NASA'],
        'Mars only': [' Mars'],
        'Tesla only': [' Tesla'],
        'Tesla+Mars': [' Tesla', ' Mars'],
    }

    # ── Query prompts ──
    query_prompts = [
        "On February 27, 2026,",
        "The major event on February 27, 2026 was",
        "On February 27, 2026, NASA",
        "Breaking news from February 27, 2026:",
        "What happened on February 27, 2026? On that day,",
    ]

    # Best scale from 10z17: 1.0 works, but let's also try a few
    best_scale = 2.0

    track_tokens = [' NASA', ' Tesla', ' Mars', ' landed', ' rover']

    print("\n" + "─" * 72)
    print(f"  Testing LM head inverse injection (scale={best_scale})")
    print("─" * 72)

    for combo_name, combo_toks in combos.items():
        # Build combined direction
        combined = sum(lm_rows[t] for t in combo_toks)
        combined = combined / np.linalg.norm(combined)

        print(f"\n  ═══ Combo: {combo_name} ═══")

        for prompt in query_prompts:
            p_ids = tokenizer.encode(prompt)

            # Get this prompt's mean attn norm for scaling
            attn_list, _ = extract_all_layer_attn(engine, p_ids)
            mean_norm = np.mean([float(np.linalg.norm(a[0, -1, :])) for a in attn_list])

            # Normal prediction
            logits_normal = get_logits(engine, _)  # _ is final hidden from extract
            # Wait, _ is not the right hidden state. Let me re-run properly.
            # Actually extract_all_layer_attn returns (attn_list, final_hidden)
            # and final_hidden went through all layers already.
            # But get_logits applies final norm + LM head.
            # So logits_normal = get_logits(engine, final_hidden) IS the normal logits.

            # Injected prediction
            delta_vec = combined * best_scale * mean_norm
            deltas = {li: delta_vec for li in range(n_layers)}
            logits_inj = run_with_deltas(engine, p_ids, deltas, all_layers)

            top1_n = tokenizer.decode([int(np.argmax(logits_normal))])
            top1_i = tokenizer.decode([int(np.argmax(logits_inj))])

            # Compact report
            rank_changes = []
            for tok in track_tokens:
                rn, _ = get_rank(logits_normal, tok, tokenizer)
                ri, li_val = get_rank(logits_inj, tok, tokenizer)
                if rn is not None:
                    rank_changes.append(f"{tok.strip()}:{rn}→{ri}")

            print(f"    '{prompt}'")
            print(f"      Normal: '{top1_n}'  →  Injected: '{top1_i}'")
            print(f"      {', '.join(rank_changes)}")

    # ── Layer ablation for best combo ──
    print("\n" + "─" * 72)
    print("  Layer ablation: which layers are needed?")
    print("─" * 72)

    combo_toks = [' NASA', ' Mars', ' landed']
    combined = sum(lm_rows[t] for t in combo_toks)
    combined = combined / np.linalg.norm(combined)

    prompt = "On February 27, 2026,"
    p_ids = tokenizer.encode(prompt)
    attn_list, h_final = extract_all_layer_attn(engine, p_ids)
    mean_norm = np.mean([float(np.linalg.norm(a[0, -1, :])) for a in attn_list])
    delta_vec = combined * best_scale * mean_norm

    layer_sets = [
        ("L27 only", {27}),
        ("L23 only", {23}),
        ("L22,23", {22, 23}),
        ("Key 5 (9,22,23,25,27)", {9, 22, 23, 25, 27}),
        ("Late (21-27)", set(range(21, 28))),
        ("L14-27", set(range(14, 28))),
        ("ALL", set(range(28))),
        ("Early (0-6)", set(range(0, 7))),
        ("Mid (7-20)", set(range(7, 21))),
    ]

    for label, layer_set in layer_sets:
        deltas = {li: delta_vec for li in range(n_layers)}
        logits_test = run_with_deltas(engine, p_ids, deltas, layer_set)
        top1 = tokenizer.decode([int(np.argmax(logits_test))])
        nasa_r, _ = get_rank(logits_test, ' NASA', tokenizer)
        mars_r, _ = get_rank(logits_test, ' Mars', tokenizer)
        landed_r, _ = get_rank(logits_test, ' landed', tokenizer)
        print(f"    {label:30s}: top1='{top1}'  NASA={nasa_r}  Mars={mars_r}  landed={landed_r}")

    # ── Scale sensitivity ──
    print("\n" + "─" * 72)
    print("  Scale sensitivity (all layers, NASA+Mars+landed)")
    print("─" * 72)

    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        delta_vec = combined * scale * mean_norm
        deltas = {li: delta_vec for li in range(n_layers)}
        logits_test = run_with_deltas(engine, p_ids, deltas, all_layers)
        top3_idx = np.argsort(logits_test)[-3:][::-1]
        top3 = [(tokenizer.decode([int(i)]), float(logits_test[i])) for i in top3_idx]
        nasa_r, _ = get_rank(logits_test, ' NASA', tokenizer)
        mars_r, _ = get_rank(logits_test, ' Mars', tokenizer)
        print(f"    scale={scale:6.1f}: "
              f"top3=['{top3[0][0]}','{top3[1][0]}','{top3[2][0]}']  "
              f"NASA={nasa_r}  Mars={mars_r}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
