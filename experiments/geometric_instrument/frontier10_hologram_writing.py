"""
Frontier 10: Writing to the Hologram
======================================
Can we ADD or EDIT knowledge in the transformer by engineering
rank-1 components in the weight matrices?

Three experiments:
  A: Residual stream surgery — swap France→Germany pre-COMB
  B: Rank-1 weight edit — modify W_gate+W_up to redirect France→Berlin
  C: Novel fact injection — add a new rank-1 component for unknown fact

"A zero is where all rotations conspire to cancel.
 A correct answer is where all projectors conspire to contribute.
 Can we add a new rotation to the sum?"
"""

import sys, os, time, copy
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-20:
        return 0.0
    return float(np.dot(a, b) / d)


def run_layers(engine, h, start, end):
    for li in range(start, end):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
        h = h_pa + phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))
    return h


def run_layers_with_weight_edits(engine, h, start, end, gate_edits=None, up_edits=None):
    """Run layers but apply rank-1 weight edits to W_gate and W_up."""
    for li in range(start, end):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)

        # Apply weight edits: W_edited · x = W · x + Δf · (v₁ᵀ · x)
        gate_out = phi_linear(mlp.W_gate, nm)
        up_out = phi_linear(mlp.W_up, nm)

        if gate_edits and li in gate_edits:
            delta_f, v1 = gate_edits[li]
            # For last token only
            proj = np.dot(v1, nm[0, -1])
            gate_out[0, -1] += delta_f * proj

        if up_edits and li in up_edits:
            delta_f, v1 = up_edits[li]
            proj = np.dot(v1, nm[0, -1])
            up_out[0, -1] += delta_f * proj

        gate_act = phi_silu(gate_out)
        h = h_pa + phi_linear(mlp.W_down, gate_act * up_out)
    return h


def predict_token(engine, tokenizer, h):
    h_last = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = phi_linear(engine.lm_head.weight, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80)
    print("  Frontier 10: Writing to the Hologram")
    print("  Can we add a new rotation to the sum?")
    print("=" * 80)

    gc_mod.collect()
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
    # Baseline: verify model answers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Baseline Verification")
    print("=" * 80)

    h_states = {}
    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 28)
        h_states[name] = {}
        idx, tok, logits = predict_token(engine, tokenizer, h)
        print(f"    {name}: {tok[0]:>10s}  (top5: {tok})")

    # ═══════════════════════════════════════════════════════════
    # Collect intermediate states at COMB boundary
    # ═══════════════════════════════════════════════════════════
    print("\n  Collecting intermediate states...")

    pre_comb = {}   # h at layer 15 input
    post_comb = {}  # h at layer 21 input
    mlp_inputs = {} # {name: {layer: normed_last_token}}

    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)
        pre_comb[name] = h.copy()

        mlp_inputs[name] = {}
        h_comb = h.copy()
        for li in range(15, 21):
            layer = engine.layers[li]
            attn, mlp = layer.attention, layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            sl = h_comb.shape[1]
            normed = rms_norm(h_comb, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
            K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if sl > 1:
                scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
            w = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
            h_pa = h_comb + phi_linear(attn.W_o, ao)
            nm = rms_norm(h_pa, mlp.norm_weight)
            mlp_inputs[name][li] = nm[0, -1].copy()
            gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
            h_comb = h_pa + phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))

        post_comb[name] = h_comb.copy()

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT A: Residual Stream Surgery
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment A: Residual Stream Surgery")
    print("  Swap France's pre-COMB state with Germany's → expect Berlin")
    print("=" * 80)

    # Method 1: Full swap at pre-COMB (layer 15)
    # Take Germany's hidden state, run it through COMB+MUSIC
    h_france_body_germany_state = pre_comb['Germany'].copy()
    h_france_body_germany_state = run_layers(engine, h_france_body_germany_state, 15, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_france_body_germany_state)
    print(f"\n    Full swap (Germany pre-COMB → COMB+MUSIC):")
    print(f"    → {tok[0]}  (top5: {tok})")

    # Method 2: Additive edit — add the DIFFERENCE to France's state
    delta = pre_comb['Germany'][:, -1:, :] - pre_comb['France'][:, -1:, :]
    h_edited = pre_comb['France'].copy()
    h_edited[:, -1:, :] += delta  # Add Germany-France difference at last token
    h_edited = run_layers(engine, h_edited, 15, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_edited)
    print(f"\n    Additive edit (France + Δ(Germany-France) at last token):")
    print(f"    → {tok[0]}  (top5: {tok})")

    # Method 3: Post-COMB swap
    h_post_swap = post_comb['Germany'].copy()
    h_post_swap = run_layers(engine, h_post_swap, 21, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_post_swap)
    print(f"\n    Post-COMB swap (Germany post-COMB → MUSIC):")
    print(f"    → {tok[0]}  (top5: {tok})")

    # Method 4: Cross-test — Italy's state should → Rome
    h_italy_swap = pre_comb['Italy'].copy()
    h_italy_swap = run_layers(engine, h_italy_swap, 15, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h_italy_swap)
    print(f"\n    Control (Italy pre-COMB → COMB+MUSIC):")
    print(f"    → {tok[0]}  (top5: {tok})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT B: Rank-1 Weight Edit
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment B: Rank-1 Weight Edit")
    print("  Modify W_gate + W_up at COMB layers to redirect France → Berlin")
    print("=" * 80)

    # Extract v₁ for the capitals class
    cap_inputs = np.stack([mlp_inputs[n][17] for n in ['France', 'Germany', 'Japan', 'Italy']])
    U, S, Vt = np.linalg.svd(cap_inputs, full_matrices=False)
    v1 = Vt[0]  # Shared capitals direction
    print(f"\n    Capitals v₁: energy = {S[0]**2/np.sum(S**2)*100:.1f}%")

    # For each COMB layer, compute the rank-1 edit: Δf = W·x_germany - W·x_france
    # projected onto v₁ space
    gate_edits = {}
    up_edits = {}

    for li in range(15, 21):
        mlp = engine.layers[li].mlp
        x_fr = mlp_inputs['France'][li]
        x_de = mlp_inputs['Germany'][li]

        # Gate edit: we want the gate to respond to France as if it were Germany
        W_gate_dec = mlp.W_gate.decode_cached()
        f_fr = W_gate_dec @ x_fr
        f_de = W_gate_dec @ x_de
        delta_gate = f_de - f_fr

        # Up edit: same for W_up
        W_up_dec = mlp.W_up.decode_cached()
        u_fr = W_up_dec @ x_fr
        u_de = W_up_dec @ x_de
        delta_up = u_de - u_fr

        # Project the edit through v₁ (the rank-1 channel)
        proj_fr = np.dot(v1, x_fr)
        proj_de = np.dot(v1, x_de)

        # The edit Δf should be such that: Δf · (v₁ᵀ · x_france) = delta
        # So Δf = delta / (v₁ᵀ · x_france)
        if abs(proj_fr) > 1e-6:
            gate_edits[li] = (delta_gate / proj_fr, v1.copy())
            up_edits[li] = (delta_up / proj_fr, v1.copy())

        cos_gate = cosine(delta_gate, f_fr)
        print(f"    L{li}: |Δgate|/|gate| = {np.linalg.norm(delta_gate)/np.linalg.norm(f_fr):.4f}, "
              f"|Δup|/|up| = {np.linalg.norm(delta_up)/np.linalg.norm(u_fr):.4f}")

    # Run France with edited weights
    print(f"\n    Running France with rank-1 weight edits (gate+up, L15-L20)...")
    tids = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)
    h = run_layers_with_weight_edits(engine, h, 15, 21,
                                      gate_edits=gate_edits, up_edits=up_edits)
    h = run_layers(engine, h, 21, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h)
    print(f"    France + rank-1 edit → {tok[0]}  (top5: {tok})")

    # Check Berlin's rank
    berlin_tok = tokenizer.encode(" Berlin")[-1]
    paris_tok = tokenizer.encode(" Paris")[-1]
    berlin_logit = float(logits[berlin_tok])
    paris_logit = float(logits[paris_tok])
    print(f"    Berlin logit: {berlin_logit:.2f}, Paris logit: {paris_logit:.2f}, "
          f"gap: {berlin_logit - paris_logit:.2f}")

    # Control: Germany with edited weights should still say Berlin
    print(f"\n    Control: Germany with same edits (should still be Berlin)...")
    tids = tokenizer.encode(prompts['Germany'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)
    h = run_layers_with_weight_edits(engine, h, 15, 21,
                                      gate_edits=gate_edits, up_edits=up_edits)
    h = run_layers(engine, h, 21, 28)
    idx, tok, logits_de = predict_token(engine, tokenizer, h)
    print(f"    Germany + rank-1 edit → {tok[0]}  (top5: {tok})")

    # Control: Japan should still say Tokyo
    print(f"\n    Control: Japan with same edits (should still be Tokyo)...")
    tids = tokenizer.encode(prompts['Japan'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)
    h = run_layers_with_weight_edits(engine, h, 15, 21,
                                      gate_edits=gate_edits, up_edits=up_edits)
    h = run_layers(engine, h, 21, 28)
    idx, tok, logits_jp = predict_token(engine, tokenizer, h)
    print(f"    Japan + rank-1 edit → {tok[0]}  (top5: {tok})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT B2: Gate-only edit
    # ═══════════════════════════════════════════════════════════
    print(f"\n    --- Gate-only edit ---")
    tids = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)
    h = run_layers_with_weight_edits(engine, h, 15, 21,
                                      gate_edits=gate_edits, up_edits=None)
    h = run_layers(engine, h, 21, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h)
    print(f"    France + gate-only edit → {tok[0]}  (top5: {tok})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT B3: Up-only edit
    # ═══════════════════════════════════════════════════════════
    print(f"\n    --- Up-only edit ---")
    tids = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)
    h = run_layers_with_weight_edits(engine, h, 15, 21,
                                      gate_edits=None, up_edits=up_edits)
    h = run_layers(engine, h, 21, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h)
    print(f"    France + up-only edit → {tok[0]}  (top5: {tok})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT C: Novel Fact — Direct MLP Output Injection
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment C: Novel Fact Injection")
    print("  Can we teach a new fact by injecting MLP output deltas?")
    print("=" * 80)

    # Strategy: We want "The capital of Truthland is" → "Geometria"
    # But the model doesn't know "Truthland". So we use a proxy:
    # Take a country the model knows poorly and redirect it.

    # First, find what the model says for an obscure prompt
    test_prompts = [
        ("The capital of Nauru is", "Yaren"),
        ("The capital of Tuvalu is", "Funafuti"),
    ]

    for prompt_text, expected in test_prompts:
        tids = tokenizer.encode(prompt_text)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h)
        print(f"\n    '{prompt_text}' → {tok[0]}  (top5: {tok})")
        # Check if expected answer exists
        expected_tids = tokenizer.encode(f" {expected}")
        if expected_tids:
            eid = expected_tids[-1]
            rank = int(np.sum(logits > logits[eid])) + 1
            print(f"    Expected '{expected}' at rank {rank}, logit = {logits[eid]:.2f}")

    # Now the real test: redirect a known fact using the MLP output delta
    # from another fact. This is "writing a new entry in the hologram"
    # by composing existing rank-1 components.
    print(f"\n    --- Compositional hologram write ---")
    print(f"    Goal: Make 'capital of France' output 'Tokyo'")
    print(f"    Method: Add (Japan_MLP_output - France_MLP_output) at COMB layers")

    # Collect full MLP outputs at each COMB layer for France and Japan
    mlp_outputs = {name: {} for name in ['France', 'Japan']}

    for name in ['France', 'Japan']:
        tids = tokenizer.encode(prompts[name])
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        for li in range(15, 21):
            layer = engine.layers[li]
            attn, mlp = layer.attention, layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            sl = h.shape[1]
            normed = rms_norm(h, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
            K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if sl > 1:
                scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
            w = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
            h_pa = h + phi_linear(attn.W_o, ao)
            nm = rms_norm(h_pa, mlp.norm_weight)
            gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
            mlp_out = phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))
            mlp_outputs[name][li] = mlp_out[0, -1].copy()
            h = h_pa + mlp_out

    # Now run France but inject the Japan-France MLP output delta
    tids = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)

    for li in range(15, 21):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        sl = h.shape[1]
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
        K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if sl > 1:
            scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
        w = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
        h_pa = h + phi_linear(attn.W_o, ao)
        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
        mlp_out = phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))

        # INJECT: add the Japan-France delta at last token
        delta_mlp = mlp_outputs['Japan'][li] - mlp_outputs['France'][li]
        mlp_out[0, -1] += delta_mlp

        h = h_pa + mlp_out

    h = run_layers(engine, h, 21, 28)
    idx, tok, logits = predict_token(engine, tokenizer, h)
    tokyo_tok = tokenizer.encode(" Tokyo")[-1]
    paris_tok = tokenizer.encode(" Paris")[-1]
    print(f"\n    France + MLP delta(Japan-France) → {tok[0]}  (top5: {tok})")
    print(f"    Tokyo logit: {logits[tokyo_tok]:.2f}, Paris logit: {logits[paris_tok]:.2f}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT C2: Scaled injection — find the threshold
    # ═══════════════════════════════════════════════════════════
    print(f"\n    --- Scaling sweep: how much delta is needed? ---")

    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        tids = tokenizer.encode(prompts['France'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        for li in range(15, 21):
            layer = engine.layers[li]
            attn, mlp = layer.attention, layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            sl = h.shape[1]
            normed = rms_norm(h, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q).reshape(1, sl, nh, hd).transpose(0, 2, 1, 3)
            K = phi_linear(attn.W_k, normed, attn.b_k).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            V = phi_linear(attn.W_v, normed, attn.b_v).reshape(1, sl, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke, Ve = np.repeat(K, hpk, axis=1), np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if sl > 1:
                scores += np.triu(np.full((sl, sl), -1e9, np.float32), k=1)
            w = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', w, Ve).transpose(0, 2, 1, 3).reshape(1, sl, -1)
            h_pa = h + phi_linear(attn.W_o, ao)
            nm = rms_norm(h_pa, mlp.norm_weight)
            gate_act = phi_silu(phi_linear(mlp.W_gate, nm))
            mlp_out = phi_linear(mlp.W_down, gate_act * phi_linear(mlp.W_up, nm))
            delta_mlp = mlp_outputs['Japan'][li] - mlp_outputs['France'][li]
            mlp_out[0, -1] += alpha * delta_mlp
            h = h_pa + mlp_out

        h = run_layers(engine, h, 21, 28)
        idx, tok, logits = predict_token(engine, tokenizer, h)
        t_logit = float(logits[tokyo_tok])
        p_logit = float(logits[paris_tok])
        print(f"    α={alpha:.2f}: → {tok[0]:>10s}  "
              f"Tokyo={t_logit:.2f} Paris={p_logit:.2f} gap={t_logit-p_logit:.2f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print()
    print("  Exp A: Residual stream surgery (representation-level edit)")
    print("  Exp B: Rank-1 weight edit (parameter-level edit)")
    print("  Exp C: MLP output injection (compositional hologram write)")
    print()
    print("  The question: can we write to the hologram?")
    print()


if __name__ == '__main__':
    main()
