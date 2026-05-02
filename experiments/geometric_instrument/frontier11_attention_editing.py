"""
Frontier 11: Attention Editing — Writing Through the Reader
=============================================================
F153 showed the hologram is read-only at the MLP level:
  - MLP = amplifier (processes whatever attention presents)
  - Attention = reader (selects what to answer)

If attention is the reader, editing attention should redirect the answer.

Experiments:
  A: V·W_o binding swap at L23 H6 (entity position only)
  B: V·W_o binding swap at L23, ALL heads
  C: Full attention output swap at L23
  D: Attention output swap — layer sweep (which layers matter?)
  E: Entity-position hidden state swap at early layers
  F: Targeted V-injection at extraction layer (the surgical write)

"To redirect to a different zero, change the compressor's initial estimate."
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear


MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-20:
        return 0.0
    return float(np.dot(a, b) / d)


def run_layer_attn(engine, layer_idx, h):
    """Run JUST the attention sublayer, return h_post_attn."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    sl = h.shape[1]
    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    Ve = np.repeat(V, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if sl > 1:
        scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
    w = phi_softmax(scores, axis=-1)
    ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
    attn_out = phi_linear(attn.W_o, ao)
    return h + attn_out, attn_out


def run_layer_mlp(engine, layer_idx, h_post_attn):
    """Run JUST the MLP sublayer."""
    mlp = engine.layers[layer_idx].mlp
    nm = rms_norm(h_post_attn, mlp.norm_weight)
    gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
    return h_post_attn + phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))


def run_layer_full(engine, layer_idx, h):
    """Run one full layer."""
    h_pa, _ = run_layer_attn(engine, layer_idx, h)
    return run_layer_mlp(engine, layer_idx, h_pa)


def run_layers(engine, h, start, end):
    for li in range(start, end):
        h = run_layer_full(engine, li, h)
    return h


def predict_token(engine, tokenizer, h):
    h_last = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = phi_linear(engine.lm_head.weight, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def get_attn_output_per_head(engine, layer_idx, h):
    """Get per-head attention outputs (before W_o projection)."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    sl = h.shape[1]
    normed = rms_norm(h, attn.norm_weight)
    Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
    K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
    V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn.rope.apply(Q), attn.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    Ve = np.repeat(V, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
    if sl > 1:
        scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
    w = phi_softmax(scores, axis=-1)
    # Per-head output: (nh, sl, hd) — weighted V for each head
    head_outputs = np.einsum('bhqk,bhkd->bhqd', w, Ve)  # (1, nh, sl, hd)
    return head_outputs[0], w[0], normed  # (nh, sl, hd), (nh, sl, sl), (1, sl, hid)


def main():
    print("=" * 80)
    print("  Frontier 11: Attention Editing — Writing Through the Reader")
    print("  If attention is the reader, can we redirect by editing attention?")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    prompts = {
        'France': 'The capital of France is',
        'Germany': 'The capital of Germany is',
        'Japan': 'The capital of Japan is',
        'Italy': 'The capital of Italy is',
    }

    # ═══════════════════════════════════════════════════════════
    # Baseline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Baseline")
    print("=" * 80)

    baselines = {}
    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h)
        baselines[name] = tok[0]
        print(f"    {name}: {tok[0]:>10s}  (top5: {tok})")

    # Collect hidden states at every layer for France and Germany
    print("\n  Collecting per-layer hidden states...")
    layer_states = {name: {} for name in prompts}

    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        layer_states[name][-1] = h.copy()  # post-embedding
        for li in range(28):
            h = run_layer_full(engine, li, h)
            layer_states[name][li] = h.copy()

    # Find entity token position for each prompt
    entity_positions = {}
    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([int(t)]) for t in tids]
        # Entity name is typically token 3 (after "The capital of")
        for i, t in enumerate(tokens):
            if name.lower() in t.lower():
                entity_positions[name] = i
                break
        print(f"    {name}: entity at pos {entity_positions.get(name, '?')} "
              f"in {tokens}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT A: V·W_o binding swap at L23 H6
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment A: V·W_o binding swap at L23 Head 6")
    print("  Keep France's routing, use Germany's V at entity position")
    print("=" * 80)

    L23 = 23

    # Get France's state at L22 (input to L23)
    h_fr_pre23 = layer_states['France'][22].copy()
    h_de_pre23 = layer_states['Germany'][22].copy()

    # Get per-head attention info for France and Germany at L23
    heads_fr, weights_fr, normed_fr = get_attn_output_per_head(engine, L23, h_fr_pre23)
    heads_de, weights_de, normed_de = get_attn_output_per_head(engine, L23, h_de_pre23)

    # Which position does Head 6 attend to for France?
    head6_weights_fr = weights_fr[6, -1, :]  # last query → all keys
    head6_argmax_fr = int(np.argmax(head6_weights_fr))
    head6_weights_de = weights_de[6, -1, :]
    head6_argmax_de = int(np.argmax(head6_weights_de))

    fr_tids = tokenizer.encode(prompts['France'])
    de_tids = tokenizer.encode(prompts['Germany'])
    fr_tokens = [tokenizer.decode([int(t)]) for t in fr_tids]
    de_tokens = [tokenizer.decode([int(t)]) for t in de_tids]

    print(f"\n    France H6 attends to pos {head6_argmax_fr} "
          f"({fr_tokens[head6_argmax_fr]}) w={head6_weights_fr[head6_argmax_fr]:.3f}")
    print(f"    Germany H6 attends to pos {head6_argmax_de} "
          f"({de_tokens[head6_argmax_de]}) w={head6_weights_de[head6_argmax_de]:.3f}")

    # Swap: replace Head 6's output at last position with Germany's
    attn = engine.layers[L23].attention
    nh, hd = attn.num_heads, attn.head_dim
    sl = h_fr_pre23.shape[1]

    # Build modified attention output: all heads from France EXCEPT head 6 from Germany
    # Concatenate per-head outputs, project through W_o
    modified_heads = heads_fr.copy()  # (nh, sl, hd)
    modified_heads[6, -1, :] = heads_de[6, -1, :]  # swap head 6 at last position

    # Reshape to (1, sl, nh*hd) for W_o projection
    modified_concat = modified_heads.transpose(1, 0, 2).reshape(1, sl, nh * hd)
    modified_attn_out = phi_linear(attn.W_o, modified_concat)

    h_pa_modified = h_fr_pre23 + modified_attn_out
    h_modified = run_layer_mlp(engine, L23, h_pa_modified)
    h_modified = run_layers(engine, h_modified, 24, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_modified)

    berlin_tok = tokenizer.encode(" Berlin")[-1]
    paris_tok = tokenizer.encode(" Paris")[-1]
    print(f"\n    France + Germany's H6 at L23 → {tok[0]}  (top5: {tok})")
    print(f"    Berlin: {logits[berlin_tok]:.2f}, Paris: {logits[paris_tok]:.2f}, "
          f"gap: {logits[berlin_tok]-logits[paris_tok]:.2f}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT B: ALL heads swap at L23
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment B: ALL heads swap at L23")
    print("  Replace entire attention output at last position with Germany's")
    print("=" * 80)

    all_de_heads = heads_de.copy()
    # Swap ALL heads at last position
    swap_all = heads_fr.copy()
    swap_all[:, -1, :] = heads_de[:, -1, :]

    swap_concat = swap_all.transpose(1, 0, 2).reshape(1, sl, nh * hd)
    swap_attn_out = phi_linear(attn.W_o, swap_concat)

    h_pa_swap = h_fr_pre23 + swap_attn_out
    h_swap = run_layer_mlp(engine, L23, h_pa_swap)
    h_swap = run_layers(engine, h_swap, 24, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_swap)
    print(f"\n    France + ALL Germany heads at L23 → {tok[0]}  (top5: {tok})")
    print(f"    Berlin: {logits[berlin_tok]:.2f}, Paris: {logits[paris_tok]:.2f}, "
          f"gap: {logits[berlin_tok]-logits[paris_tok]:.2f}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT C: Full attention output swap at L23
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment C: Full attention output replacement at L23")
    print("  France's residual + Germany's full attention output")
    print("=" * 80)

    # Get Germany's full attention output at L23
    _, attn_out_de = run_layer_attn(engine, L23, h_de_pre23)
    _, attn_out_fr = run_layer_attn(engine, L23, h_fr_pre23)

    # Replace France's attention output with Germany's (at last position only)
    h_pa_full = h_fr_pre23.copy()
    h_pa_full[:, -1:, :] += (attn_out_de[:, -1:, :] - attn_out_fr[:, -1:, :])
    h_pa_full += attn_out_fr  # add original back (we subtracted above effectively)

    # Simpler: France residual + Germany's attention contribution at last pos
    h_pa_c = h_fr_pre23 + attn_out_fr  # normal post-attn
    h_pa_c[:, -1:, :] += (attn_out_de[:, -1:, :] - attn_out_fr[:, -1:, :])

    h_c = run_layer_mlp(engine, L23, h_pa_c)
    h_c = run_layers(engine, h_c, 24, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_c)
    print(f"\n    France + Germany attn_out delta at L23 last pos → {tok[0]}  (top5: {tok})")
    print(f"    Berlin: {logits[berlin_tok]:.2f}, Paris: {logits[paris_tok]:.2f}, "
          f"gap: {logits[berlin_tok]-logits[paris_tok]:.2f}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT D: Layer sweep — attention output swap at each layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment D: Attention output swap — layer sweep")
    print("  At which layer does swapping attention output redirect France→Berlin?")
    print("=" * 80)

    print(f"\n    {'Layer':>5s} {'Top-1':>10s} {'Berlin':>8s} {'Paris':>8s} {'Gap':>8s}")
    print(f"    {'-----':>5s} {'-----':>10s} {'------':>8s} {'-----':>8s} {'---':>8s}")

    for target_l in range(28):
        # Get France and Germany states at target_l input
        if target_l == 0:
            h_fr_in = layer_states['France'][-1].copy()
            h_de_in = layer_states['Germany'][-1].copy()
        else:
            h_fr_in = layer_states['France'][target_l - 1].copy()
            h_de_in = layer_states['Germany'][target_l - 1].copy()

        # Get attention outputs
        _, attn_out_fr_l = run_layer_attn(engine, target_l, h_fr_in)
        _, attn_out_de_l = run_layer_attn(engine, target_l, h_de_in)

        # France's post-attn with Germany's attention output at last position
        h_pa_d = h_fr_in + attn_out_fr_l
        h_pa_d[:, -1:, :] += (attn_out_de_l[:, -1:, :] - attn_out_fr_l[:, -1:, :])

        # Run MLP for this layer, then remaining layers
        h_d = run_layer_mlp(engine, target_l, h_pa_d)
        h_d = run_layers(engine, h_d, target_l + 1, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h_d)

        b_logit = float(logits[berlin_tok])
        p_logit = float(logits[paris_tok])
        marker = " ←←←" if tok[0].strip() == "Berlin" else ""
        print(f"    L{target_l:>3d} {tok[0]:>10s} {b_logit:>8.2f} {p_logit:>8.2f} "
              f"{b_logit-p_logit:>8.2f}{marker}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT E: Entity-position hidden state swap
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment E: Entity-position hidden state swap")
    print("  At entity token position, swap France's h with Germany's")
    print("=" * 80)

    fr_entity_pos = entity_positions['France']
    de_entity_pos = entity_positions['Germany']

    print(f"\n    France entity pos: {fr_entity_pos}, Germany entity pos: {de_entity_pos}")
    print(f"\n    {'Layer':>5s} {'Top-1':>10s} {'Berlin':>8s} {'Paris':>8s} {'Gap':>8s}")
    print(f"    {'-----':>5s} {'-----':>10s} {'------':>8s} {'-----':>8s} {'---':>8s}")

    for swap_after_layer in [-1, 0, 1, 2, 3, 4, 5, 10, 15, 20, 22, 25, 27]:
        h_edited = layer_states['France'][swap_after_layer].copy()
        h_de_source = layer_states['Germany'][swap_after_layer]

        # Swap just the entity position's hidden state
        h_edited[0, fr_entity_pos, :] = h_de_source[0, de_entity_pos, :]

        # Run remaining layers
        start_layer = swap_after_layer + 1
        h_out = run_layers(engine, h_edited, start_layer, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h_out)

        b_logit = float(logits[berlin_tok])
        p_logit = float(logits[paris_tok])
        marker = " ←←←" if tok[0].strip() == "Berlin" else ""
        label = f"emb" if swap_after_layer == -1 else f"L{swap_after_layer}"
        print(f"    {label:>5s} {tok[0]:>10s} {b_logit:>8.2f} {p_logit:>8.2f} "
              f"{b_logit-p_logit:>8.2f}{marker}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT F: Surgical V-injection at L23
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment F: Surgical V-injection — the minimal write")
    print("  Replace V at entity pos for specific head groups at L23")
    print("=" * 80)

    # Test replacing V at entity position for each KV group separately
    nkv = attn.num_kv_heads
    hpk = nh // nkv

    for kv_group in range(nkv):
        head_range = f"H{kv_group*hpk}-{(kv_group+1)*hpk-1}"
        swap_heads = heads_fr.copy()
        for hi in range(kv_group * hpk, (kv_group + 1) * hpk):
            swap_heads[hi, -1, :] = heads_de[hi, -1, :]

        swap_concat = swap_heads.transpose(1, 0, 2).reshape(1, sl, nh * hd)
        swap_out = phi_linear(attn.W_o, swap_concat)
        h_pa_f = h_fr_pre23 + swap_out
        h_f = run_layer_mlp(engine, L23, h_pa_f)
        h_f = run_layers(engine, h_f, 24, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h_f)
        b_logit = float(logits[berlin_tok])
        p_logit = float(logits[paris_tok])
        print(f"    KV group {kv_group} ({head_range}): → {tok[0]:>10s}  "
              f"Berlin={b_logit:.2f} Paris={p_logit:.2f} gap={b_logit-p_logit:.2f}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT G: Multi-layer attention output swap (cumulative)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment G: Cumulative attention swap across layers")
    print("  Swap attention output at last position for expanding layer ranges")
    print("=" * 80)

    ranges_to_test = [
        (23, 24, "L23 only"),
        (22, 24, "L22-23"),
        (20, 24, "L20-23"),
        (15, 24, "L15-23"),
        (15, 28, "L15-27"),
        (10, 28, "L10-27"),
        (5, 28, "L5-27"),
        (0, 28, "L0-27"),
    ]

    for start_l, end_l, label in ranges_to_test:
        # Start from the state BEFORE the swap range
        if start_l == 0:
            h = layer_states['France'][-1].copy()
        else:
            h = layer_states['France'][start_l - 1].copy()

        for li in range(start_l, 28):
            if li < end_l:
                # Get France's and Germany's states at this layer's input
                if li == 0:
                    h_de_in = layer_states['Germany'][-1]
                else:
                    h_de_in = layer_states['Germany'][li - 1]

                _, attn_out_de_li = run_layer_attn(engine, li, h_de_in)
                h_pa, attn_out_fr_li = run_layer_attn(engine, li, h)
                # Swap attention output at last position
                h_pa[:, -1:, :] += (attn_out_de_li[:, -1:, :] - attn_out_fr_li[:, -1:, :])
                h = run_layer_mlp(engine, li, h_pa)
            else:
                h = run_layer_full(engine, li, h)

        idx, tok, logits = predict_token(engine, tokenizer, h)
        b_logit = float(logits[berlin_tok])
        p_logit = float(logits[paris_tok])
        marker = " ←←←" if tok[0].strip() == "Berlin" else ""
        print(f"    {label:>12s}: → {tok[0]:>10s}  "
              f"Berlin={b_logit:.2f} Paris={p_logit:.2f} gap={b_logit-p_logit:.2f}{marker}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print()
    print("  A: V·W_o swap H6 only at L23     — single head extraction edit")
    print("  B: All heads swap at L23          — full extraction edit")
    print("  C: Full attn output delta at L23  — representation edit")
    print("  D: Layer sweep attn output swap   — which layers control the answer?")
    print("  E: Entity-position hidden swap    — early intervention")
    print("  F: KV-group targeted swap at L23  — which heads matter?")
    print("  G: Cumulative layer range swap    — how many layers needed?")
    print()
    print("  The question: is attention the reader? Can we write through it?")
    print()


if __name__ == '__main__':
    main()
