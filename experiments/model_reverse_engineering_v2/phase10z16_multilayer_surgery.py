"""
Phase 10z16: Multi-Layer Fact Surgery
=======================================

Phase10z15 showed single-layer surgery at L23 shifts donor rank by 10-65×
but can't fully override the host answer because 27 other layers still
contribute the original fact.

This experiment applies V·W_o deltas at ALL layers simultaneously.

If the transformer is truly a Riemann-Siegel sum where each layer is
one term, then modifying all terms should give complete control:

  Σ_{layers} V·W_o_modified = new_answer

Strategy:
  1. Run France and Japan through all layers, cache per-layer attention outputs
  2. Compute delta_L = attn_france_L - attn_japan_L for each layer L
  3. Run Japan's prompt with ALL deltas applied → should predict Paris
  4. Test subsets: which layers matter most?
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


PROMPTS = {
    'France': ('The capital of France is', 'Paris', 3),
    'Japan':  ('The capital of Japan is', 'Tokyo', 3),
    'Germany':('The capital of Germany is', 'Berlin', 3),
    'Italy':  ('The capital of Italy is', 'Rome', 3),
    'Brazil': ('The capital of Brazil is', 'Brasilia', 3),
    'Egypt':  ('The capital of Egypt is', 'Cairo', 3),
}


def get_logits(engine, hidden_3d):
    """Project hidden state to logits WITH final RMS norm."""
    normed = rms_norm(hidden_3d, engine.final_norm_weight)
    logits = engine.lm_head(normed)
    return logits[0, -1, :]


def extract_all_layer_attn(engine, prompt_ids):
    """Run forward pass, returning per-layer attention outputs (deltas).

    Returns:
        per_layer_attn: list of (1, seq_len, hidden_dim) — attention delta per layer
        per_layer_hidden_before: list of (1, seq_len, hidden_dim) — input to each layer
        final_logits: vocab logits from full model
    """
    n_layers = len(engine.layers)
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]

    per_layer_attn = []
    per_layer_hidden_before = []

    for layer in engine.layers:
        per_layer_hidden_before.append(h.copy())

        attn = layer.attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        heads_per_kv = num_heads // num_kv_heads
        head_dim = attn.head_dim
        seq_len = h.shape[1]

        # Attention
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

        # Continue through residual + MLP
        h_post_attn = h + attn_output
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    logits = get_logits(engine, h)
    return per_layer_attn, per_layer_hidden_before, logits


def run_with_attn_deltas(engine, prompt_ids, deltas, delta_layers):
    """Run forward pass but add deltas to attention output at specified layers.

    Args:
        deltas: dict mapping layer_idx → (hidden_dim,) delta for last token
        delta_layers: set of layer indices to modify

    Returns:
        logits from the modified forward pass
    """
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]

    for layer in engine.layers:
        li = layer.layer_idx
        attn = layer.attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        heads_per_kv = num_heads // num_kv_heads
        head_dim = attn.head_dim
        seq_len = h.shape[1]

        # Attention
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

        # Apply delta if this layer is in the modification set
        if li in delta_layers and li in deltas:
            attn_output[0, -1, :] += deltas[li]

        h_post_attn = h + attn_output

        # MLP
        mlp = layer.mlp
        normed_mlp = rms_norm(h_post_attn, mlp.norm_weight)
        gate = phi_linear(mlp.W_gate, normed_mlp)
        up = phi_linear(mlp.W_up, normed_mlp)
        mlp_out = phi_linear(mlp.W_down, phi_silu(gate) * up)
        h = h_post_attn + mlp_out

    return get_logits(engine, h)


def show_top_k(tokenizer, logits, k=5, prefix=""):
    top_idx = np.argsort(logits)[-k:][::-1]
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        print(f"    {prefix}{logits[idx]:+8.3f}  '{tok}'")


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
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 72)
    print("  PHASE 10z16: MULTI-LAYER FACT SURGERY")
    print("=" * 72)

    # ── Extract per-layer attention for all prompts ──
    print("\n" + "─" * 72)
    print("  Extracting per-layer attention outputs...")
    print("─" * 72)

    prompt_data = {}
    for country, (prompt, expected, pos) in PROMPTS.items():
        t1 = time.time()
        p_ids = tokenizer.encode(prompt)
        attn_list, h_before_list, logits = extract_all_layer_attn(engine, p_ids)
        rank, logit = get_rank(logits, ' ' + expected, tokenizer)
        top1_idx = int(np.argmax(logits))
        top1_tok = tokenizer.decode([top1_idx])
        print(f"  {country:>8s}: '{top1_tok}' ({logits[top1_idx]:.2f})  "
              f"['{expected}' rank={rank}]  ({time.time()-t1:.1f}s)")
        prompt_data[country] = {
            'prompt': prompt, 'expected': expected, 'pos': pos,
            'p_ids': p_ids, 'attn_list': attn_list,
            'h_before_list': h_before_list, 'logits': logits,
        }

    # ── Compute per-layer deltas ──
    print("\n" + "─" * 72)
    print("  Computing per-layer attention deltas (last token)")
    print("─" * 72)

    # Per-layer delta magnitude: how much does each layer's attn differ?
    test_pairs = [
        ('France', 'Japan'),
        ('France', 'Germany'),
        ('Japan', 'Germany'),
    ]

    for country_a, country_b in test_pairs:
        da = prompt_data[country_a]
        db = prompt_data[country_b]
        print(f"\n  {country_a} vs {country_b} — per-layer delta norms:")

        deltas = {}
        delta_norms = []
        for li in range(n_layers):
            d = da['attn_list'][li][0, -1, :] - db['attn_list'][li][0, -1, :]
            deltas[li] = d
            norm = float(np.linalg.norm(d))
            delta_norms.append(norm)

        # Find layers with largest deltas
        sorted_layers = sorted(range(n_layers), key=lambda i: -delta_norms[i])
        for li in sorted_layers[:10]:
            bar = '█' * int(delta_norms[li] / max(delta_norms) * 20)
            print(f"    L{li:2d}: {delta_norms[li]:8.3f}  {bar}")

        # ── Test: apply deltas at ALL layers ──
        print(f"\n  FULL SWAP ({country_a} → {country_b}): "
              f"inject {country_a}'s pattern into {country_b}")
        logits_full = run_with_attn_deltas(
            engine, db['p_ids'], deltas, set(range(n_layers)))

        top1_tok = tokenizer.decode([int(np.argmax(logits_full))])
        print(f"    Top-1: '{top1_tok}'")
        show_top_k(tokenizer, logits_full, k=5)
        rank_a, _ = get_rank(logits_full, ' ' + da['expected'], tokenizer)
        rank_b, _ = get_rank(logits_full, ' ' + db['expected'], tokenizer)
        print(f"    '{da['expected']}' rank: {rank_a}  "
              f"'{db['expected']}' rank: {rank_b}")

        # ── Test: apply deltas at subsets ──
        # Top-N layers by delta magnitude
        for n_top in [1, 3, 5, 10, 14, 20, 28]:
            top_layers = set(sorted_layers[:n_top])
            logits_partial = run_with_attn_deltas(
                engine, db['p_ids'], deltas, top_layers)
            rank_a_p, _ = get_rank(logits_partial, ' ' + da['expected'], tokenizer)
            rank_b_p, _ = get_rank(logits_partial, ' ' + db['expected'], tokenizer)
            top1_p = tokenizer.decode([int(np.argmax(logits_partial))])
            marker = "✓" if rank_a_p is not None and rank_a_p < rank_b_p else " "
            layer_str = ','.join(str(l) for l in sorted(top_layers)[:5])
            if n_top > 5:
                layer_str += f"...+{n_top-5}"
            print(f"    {marker} Top-{n_top:2d} layers [{layer_str}]: "
                  f"'{da['expected']}'={rank_a_p}  "
                  f"'{db['expected']}'={rank_b_p}  top1='{top1_p}'")

        # ── Test: layer ranges ──
        print(f"\n  Layer range analysis:")
        ranges = [
            ("Early (0-6)", range(0, 7)),
            ("Mid-early (7-13)", range(7, 14)),
            ("Mid-late (14-20)", range(14, 21)),
            ("Late (21-27)", range(21, 28)),
            ("L23 only", [23]),
            ("L20-27", range(20, 28)),
            ("L0-13", range(0, 14)),
            ("L14-27", range(14, 28)),
            ("ALL", range(0, 28)),
        ]

        for label, layer_range in ranges:
            logits_r = run_with_attn_deltas(
                engine, db['p_ids'], deltas, set(layer_range))
            rank_a_r, _ = get_rank(logits_r, ' ' + da['expected'], tokenizer)
            rank_b_r, _ = get_rank(logits_r, ' ' + db['expected'], tokenizer)
            top1_r = tokenizer.decode([int(np.argmax(logits_r))])
            marker = "✓" if rank_a_r is not None and rank_a_r < rank_b_r else " "
            print(f"    {marker} {label:20s}: '{da['expected']}'={rank_a_r:5d}  "
                  f"'{db['expected']}'={rank_b_r:5d}  top1='{top1_r}'")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                            'phase10z16_multilayer_surgery.json')
    # Just save a summary since the full data is huge
    save_data = {
        'n_layers': n_layers,
        'test_pairs': test_pairs,
        'note': 'Full results printed to stdout',
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
