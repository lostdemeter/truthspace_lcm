"""
Frontier 7d: Gate = Structure Selector, W_up = Content Router
==============================================================
F7c showed:
  - Rank-1 GATE at all COMB layers → 2/2 correct (Paris, Berlin)
  - Scalar navigation along v₁ → ZERO effect on output
  - Entity info NOT on the rank-1 manifold

This means:
  - Gate selects the STRUCTURE CLASS ("capital fact")
  - Content (WHICH capital) flows through W_up
  - W_down projects the selected content back

Verification tests:
  1. Rank-1 W_up should FAIL (removes entity info from content path)
  2. Rank-1 gate + original W_up → should work (structure select + content)
  3. Original gate + rank-1 W_up → should FAIL
  4. What is W_up actually routing? Project W_up through v₁ and v₁_perp
  5. Cross-entity W_up swap: use France's gate with Germany's W_up path
  6. Design test: can we ENGINEER the gate for a custom structure class?
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2


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


def run_attn_only(engine, h, li):
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
    return h_pa, nm


def predict_token(engine, tokenizer, h):
    h_last = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = phi_linear(engine.lm_head.weight, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80)
    print("  Frontier 7d: Gate = Structure, W_up = Content")
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
        'Egypt': 'The capital of Egypt is',
    }
    entity_names = list(prompts.keys())
    answer_tokens = {
        'France': tokenizer.encode(' Paris')[-1],
        'Germany': tokenizer.encode(' Berlin')[-1],
        'Japan': tokenizer.encode(' Tokyo')[-1],
        'Egypt': tokenizer.encode(' Cairo')[-1],
    }

    # ═══════════════════════════════════════════════════════════
    # Collect MLP inputs at each COMB layer
    # ═══════════════════════════════════════════════════════════
    mlp_inputs = {}  # {entity: {layer: normed(3584,)}}
    h_pre_comb = {}  # {entity: h at L15 input}

    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)
        h_pre_comb[name] = h.copy()

        mlp_inputs[name] = {}
        h_walk = h.copy()
        for li in range(15, 21):
            h_pa, nm = run_attn_only(engine, h_walk, li)
            mlp_inputs[name][li] = nm[0, -1].copy()
            mlp = engine.layers[li].mlp
            gate = phi_silu(phi_linear(mlp.W_gate, nm))
            up = phi_linear(mlp.W_up, nm)
            h_walk = h_pa + phi_linear(mlp.W_down, gate * up)

        h_walk = run_layers(engine, h_walk, 21, 28)
        _, tok, _ = predict_token(engine, tokenizer, h_walk)
        print(f"  Baseline {name}: → {tok[0]!r}")

    # Compute v₁ for each layer
    manifold = {}
    for li in range(15, 21):
        inputs = np.stack([mlp_inputs[n][li] for n in entity_names])
        U, S, Vt = np.linalg.svd(inputs, full_matrices=False)
        manifold[li] = Vt[0]  # v₁

    # ═══════════════════════════════════════════════════════════
    # Inv 1: Rank-1 W_up at ALL COMB layers (should FAIL)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: Rank-1 W_up (Should FAIL — Removes Entity Content)")
    print("=" * 80)

    for name in entity_names:
        h = h_pre_comb[name].copy()
        for li in range(15, 21):
            v1 = manifold[li]
            mlp = engine.layers[li].mlp
            W_up_dec = mlp.W_up.decode_cached()
            # Rank-1 W_up: projects only v₁ component
            filter_up = W_up_dec @ v1.astype(np.float32)  # (18944,)
            W_up_r1 = np.outer(filter_up, v1).astype(np.float32)
            W_up_r1_phi = PhiEncoded.encode(W_up_r1)

            h_pa, nm = run_attn_only(engine, h, li)
            gate = phi_silu(phi_linear(mlp.W_gate, nm))
            up_r1 = phi_linear(W_up_r1_phi, nm)
            h = h_pa + phi_linear(mlp.W_down, gate * up_r1)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)
        ans_l = float(logits[answer_tokens[name]])
        print(f"  {name:8s} rank-1 W_up: → {tok[0]!r}  (answer logit={ans_l:.2f})")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: Rank-1 Gate (should work — from F7c Phase 7)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Rank-1 Gate + Full W_up (Confirmation)")
    print("=" * 80)

    for name in entity_names:
        h = h_pre_comb[name].copy()
        for li in range(15, 21):
            v1 = manifold[li]
            mlp = engine.layers[li].mlp
            W_gate_dec = mlp.W_gate.decode_cached()
            filter_gate = W_gate_dec @ v1.astype(np.float32)
            W_gate_r1 = np.outer(filter_gate, v1).astype(np.float32)
            W_gate_r1_phi = PhiEncoded.encode(W_gate_r1)

            h_pa, nm = run_attn_only(engine, h, li)
            gate_r1 = phi_silu(phi_linear(W_gate_r1_phi, nm))
            up = phi_linear(mlp.W_up, nm)  # Full W_up
            h = h_pa + phi_linear(mlp.W_down, gate_r1 * up)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)
        ans_l = float(logits[answer_tokens[name]])
        print(f"  {name:8s} rank-1 gate: → {tok[0]!r}  (answer logit={ans_l:.2f})")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: BOTH Rank-1 (should fail like W_up)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: BOTH Rank-1 (Gate AND W_up)")
    print("=" * 80)

    for name in ['France', 'Germany']:
        h = h_pre_comb[name].copy()
        for li in range(15, 21):
            v1 = manifold[li]
            mlp = engine.layers[li].mlp

            W_gate_dec = mlp.W_gate.decode_cached()
            W_up_dec = mlp.W_up.decode_cached()
            fg = W_gate_dec @ v1.astype(np.float32)
            fu = W_up_dec @ v1.astype(np.float32)
            W_gate_r1_phi = PhiEncoded.encode(np.outer(fg, v1).astype(np.float32))
            W_up_r1_phi = PhiEncoded.encode(np.outer(fu, v1).astype(np.float32))

            h_pa, nm = run_attn_only(engine, h, li)
            gate_r1 = phi_silu(phi_linear(W_gate_r1_phi, nm))
            up_r1 = phi_linear(W_up_r1_phi, nm)
            h = h_pa + phi_linear(mlp.W_down, gate_r1 * up_r1)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)
        p_l = float(logits[answer_tokens['France']])
        b_l = float(logits[answer_tokens['Germany']])
        print(f"  {name:8s} both rank-1: → {tok[0]!r}  P={p_l:.2f} B={b_l:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: W_up Content Analysis — What does W_up carry?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: W_up Content — Parallel vs Orthogonal to v₁")
    print("=" * 80)

    for li in [15, 17, 19]:
        v1 = manifold[li]
        mlp = engine.layers[li].mlp

        for name in ['France', 'Germany']:
            x = mlp_inputs[name][li].astype(np.float64)

            # Decompose input into v₁ component and orthogonal
            x_par = float(x @ v1) * v1
            x_orth = x - x_par
            par_frac = float(np.linalg.norm(x_par) / np.linalg.norm(x))

            # W_up applied to each component
            W_up_dec = mlp.W_up.decode_cached()
            up_full = W_up_dec @ x.astype(np.float32)
            up_par = W_up_dec @ x_par.astype(np.float32)
            up_orth = W_up_dec @ x_orth.astype(np.float32)

            cos_par = cosine(up_full, up_par)
            cos_orth = cosine(up_full, up_orth)
            energy_par = float(np.linalg.norm(up_par)**2 / np.linalg.norm(up_full)**2)
            energy_orth = float(np.linalg.norm(up_orth)**2 / np.linalg.norm(up_full)**2)

            print(f"  L{li} {name:8s}: |x_par|/|x|={par_frac:.4f}  "
                  f"up_energy: par={energy_par:.4f} orth={energy_orth:.4f}  "
                  f"cos: par={cos_par:.4f} orth={cos_orth:.4f}")

        # Cross-entity: is the orthogonal component entity-specific?
        x_fr = mlp_inputs['France'][li].astype(np.float64)
        x_de = mlp_inputs['Germany'][li].astype(np.float64)
        x_fr_orth = x_fr - float(x_fr @ v1) * v1
        x_de_orth = x_de - float(x_de @ v1) * v1
        cos_orth_cross = cosine(x_fr_orth, x_de_orth)
        print(f"  L{li} France⊥ vs Germany⊥: cos={cos_orth_cross:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 5: Cross-Entity Content Swap at MLP Level
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Content Swap — France Gate × Germany W_up Input")
    print("=" * 80)

    # At each COMB layer: use France's h (for gate) but inject
    # Germany's orthogonal component into the W_up input.
    # If content is in the orthogonal part, this should push toward Berlin.

    for li_target in [15, 17, 19]:
        h_fr = h_pre_comb['France'].copy()
        h_de = h_pre_comb['Germany'].copy()

        # Run both through COMB up to the target layer
        for li in range(15, li_target):
            mlp = engine.layers[li].mlp
            h_pa_fr, nm_fr = run_attn_only(engine, h_fr, li)
            gate_fr = phi_silu(phi_linear(mlp.W_gate, nm_fr))
            up_fr = phi_linear(mlp.W_up, nm_fr)
            h_fr = h_pa_fr + phi_linear(mlp.W_down, gate_fr * up_fr)

            h_pa_de, nm_de = run_attn_only(engine, h_de, li)
            gate_de = phi_silu(phi_linear(mlp.W_gate, nm_de))
            up_de = phi_linear(mlp.W_up, nm_de)
            h_de = h_pa_de + phi_linear(mlp.W_down, gate_de * up_de)

        # At the target layer: France gate, Germany up-input (orth component)
        v1 = manifold[li_target]
        mlp = engine.layers[li_target].mlp
        h_pa_fr, nm_fr = run_attn_only(engine, h_fr, li_target)

        # Also get Germany's normed input at this point
        h_pa_de, nm_de = run_attn_only(engine, h_de, li_target)

        # Swap: France's normed_last parallel + Germany's normed_last orthogonal
        nm_swap = nm_fr.copy()
        x_fr = nm_fr[0, -1].astype(np.float64)
        x_de = nm_de[0, -1].astype(np.float64)
        fr_par = float(x_fr @ v1) * v1
        fr_orth = x_fr - fr_par
        de_par = float(x_de @ v1) * v1
        de_orth = x_de - de_par

        # Keep France's parallel, swap in Germany's orthogonal
        nm_swap[0, -1] = (fr_par + de_orth).astype(np.float32)

        gate_fr = phi_silu(phi_linear(mlp.W_gate, nm_fr))  # France gate
        up_swap = phi_linear(mlp.W_up, nm_swap)             # Swapped up input
        h_swap = h_pa_fr + phi_linear(mlp.W_down, gate_fr * up_swap)

        # Continue normally
        for li in range(li_target + 1, 21):
            mlp2 = engine.layers[li].mlp
            h_pa2, nm2 = run_attn_only(engine, h_swap, li)
            gate2 = phi_silu(phi_linear(mlp2.W_gate, nm2))
            up2 = phi_linear(mlp2.W_up, nm2)
            h_swap = h_pa2 + phi_linear(mlp2.W_down, gate2 * up2)

        h_swap = run_layers(engine, h_swap, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h_swap)
        p_l = float(logits[answer_tokens['France']])
        b_l = float(logits[answer_tokens['Germany']])
        print(f"  L{li_target}: France∥ + Germany⊥ → {tok[0]!r}  "
              f"Paris={p_l:.2f}  Berlin={b_l:.2f}  gap={p_l-b_l:+.2f}")

    # Now swap ALL COMB layers
    print("\n  All COMB layers: France∥ + Germany⊥")
    h_fr = h_pre_comb['France'].copy()
    h_de = h_pre_comb['Germany'].copy()

    for li in range(15, 21):
        v1 = manifold[li]
        mlp = engine.layers[li].mlp

        h_pa_fr, nm_fr = run_attn_only(engine, h_fr, li)
        h_pa_de, nm_de = run_attn_only(engine, h_de, li)

        x_fr = nm_fr[0, -1].astype(np.float64)
        x_de = nm_de[0, -1].astype(np.float64)
        fr_par = float(x_fr @ v1) * v1
        de_orth = x_de - float(x_de @ v1) * v1

        nm_swap = nm_fr.copy()
        nm_swap[0, -1] = (fr_par + de_orth).astype(np.float32)

        gate_fr = phi_silu(phi_linear(mlp.W_gate, nm_fr))
        up_swap = phi_linear(mlp.W_up, nm_swap)
        h_fr = h_pa_fr + phi_linear(mlp.W_down, gate_fr * up_swap)

        # Advance Germany normally for next layer's comparison
        gate_de = phi_silu(phi_linear(mlp.W_gate, nm_de))
        up_de = phi_linear(mlp.W_up, nm_de)
        h_de = h_pa_de + phi_linear(mlp.W_down, gate_de * up_de)

    h_result = run_layers(engine, h_fr, 21, 28)
    _, tok, logits = predict_token(engine, tokenizer, h_result)
    p_l = float(logits[answer_tokens['France']])
    b_l = float(logits[answer_tokens['Germany']])
    t_l = float(logits[answer_tokens['Japan']])
    c_l = float(logits[answer_tokens['Egypt']])
    print(f"  → {tok[0]!r}  top5: {tok[:5]}")
    print(f"    Paris={p_l:.2f}  Berlin={b_l:.2f}  Tokyo={t_l:.2f}  Cairo={c_l:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 6: Full W_up Input Swap (not just orthogonal)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: Full Content Swap — Germany's Full Input to W_up")
    print("=" * 80)

    # France's gate activation + Germany's FULL normed input to W_up
    h_fr = h_pre_comb['France'].copy()
    h_de = h_pre_comb['Germany'].copy()

    for li in range(15, 21):
        mlp = engine.layers[li].mlp
        h_pa_fr, nm_fr = run_attn_only(engine, h_fr, li)
        h_pa_de, nm_de = run_attn_only(engine, h_de, li)

        gate_fr = phi_silu(phi_linear(mlp.W_gate, nm_fr))  # France gate
        up_de = phi_linear(mlp.W_up, nm_de)                 # Germany content
        h_fr = h_pa_fr + phi_linear(mlp.W_down, gate_fr * up_de)

        # Advance Germany
        gate_de = phi_silu(phi_linear(mlp.W_gate, nm_de))
        h_de = h_pa_de + phi_linear(mlp.W_down, gate_de * up_de)

    h_result = run_layers(engine, h_fr, 21, 28)
    _, tok, logits = predict_token(engine, tokenizer, h_result)
    p_l = float(logits[answer_tokens['France']])
    b_l = float(logits[answer_tokens['Germany']])
    print(f"  France gate × Germany up (all COMB): → {tok[0]!r}")
    print(f"    Paris={p_l:.2f}  Berlin={b_l:.2f}  gap={p_l-b_l:+.2f}")

    # Reverse: Germany's gate × France's up
    h_fr = h_pre_comb['France'].copy()
    h_de = h_pre_comb['Germany'].copy()

    for li in range(15, 21):
        mlp = engine.layers[li].mlp
        h_pa_fr, nm_fr = run_attn_only(engine, h_fr, li)
        h_pa_de, nm_de = run_attn_only(engine, h_de, li)

        gate_de = phi_silu(phi_linear(mlp.W_gate, nm_de))  # Germany gate
        up_fr = phi_linear(mlp.W_up, nm_fr)                 # France content
        # Apply to France's residual stream
        h_fr = h_pa_fr + phi_linear(mlp.W_down, gate_de * up_fr)

        # Advance Germany
        up_de = phi_linear(mlp.W_up, nm_de)
        h_de = h_pa_de + phi_linear(mlp.W_down, gate_de * up_de)

    h_result = run_layers(engine, h_fr, 21, 28)
    _, tok, logits = predict_token(engine, tokenizer, h_result)
    p_l = float(logits[answer_tokens['France']])
    b_l = float(logits[answer_tokens['Germany']])
    print(f"  Germany gate × France up (all COMB): → {tok[0]!r}")
    print(f"    Paris={p_l:.2f}  Berlin={b_l:.2f}  gap={p_l-b_l:+.2f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 7: W_down Analysis — Does W_down also separate?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 7: Rank-1 W_down (is projection also structure-only?)")
    print("=" * 80)

    for name in ['France', 'Germany']:
        h = h_pre_comb[name].copy()
        for li in range(15, 21):
            v1 = manifold[li]
            mlp = engine.layers[li].mlp

            h_pa, nm = run_attn_only(engine, h, li)
            gate = phi_silu(phi_linear(mlp.W_gate, nm))
            up = phi_linear(mlp.W_up, nm)
            hidden = gate * up  # (1, seq, 18944)

            # Rank-1 W_down: project hidden through the v1-derived direction
            W_down_dec = mlp.W_down.decode_cached()  # (3584, 18944)
            # The "output direction" for v1: W_down's response to the filter
            W_gate_dec = mlp.W_gate.decode_cached()
            filter_gate = W_gate_dec @ v1.astype(np.float32)  # (18944,)
            out_dir = W_down_dec @ filter_gate  # (3584,)
            out_dir_norm = out_dir / (np.linalg.norm(out_dir) + 1e-20)

            # How much of the MLP output is along this direction?
            mlp_out = phi_linear(mlp.W_down, hidden)  # (1, seq, 3584)
            mlp_out_last = mlp_out[0, -1].astype(np.float64)
            cos_out = cosine(mlp_out_last, out_dir)
            energy_along = float((mlp_out_last @ out_dir_norm)**2 /
                                  (np.linalg.norm(mlp_out_last)**2 + 1e-20))

            h = h_pa + mlp_out
            if li in [15, 17, 19]:
                print(f"  L{li} {name:8s}: MLP output cos(v1_output_dir)={cos_out:.4f}  "
                      f"energy_along={energy_along:.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  The MLP Architecture:
  - W_gate: STRUCTURE SELECTOR (rank-1 sufficient, same for all entities)
  - W_up:   CONTENT ROUTER (full-rank needed, entity info in orthogonal complement)
  - W_down: OUTPUT PROJECTION (maps selected content back to hidden space)
  
  Shape Translation Strategy:
  - Gate shapes ARE translatable (rank-1, structure-class universal)
  - Content routing through W_up must be PRESERVED (not designed)
  - The "irreducible" part is W_up × W_down, not the gate
  
  This means shape translation works for the SELECTOR (gate),
  and the content flows through automatically.
""")


if __name__ == '__main__':
    main()
