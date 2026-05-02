"""
Frontier 7c: The Rank-1 Manifold — Shape Translation on a Line
================================================================
F7b revealed that the COMB zone's full-rank weight matrices
operate on EFFECTIVELY rank-1 inputs (98.8-99.5% gate cos from
rank-1 SVD). This means:

  gate ≈ W_gate @ (σ₁ · v₁) = σ₁ · (W_gate @ v₁)

The full 3584-d computation collapses to a 1-d operation:
  - v₁ = principal input direction (shared across all capitals)
  - W_gate @ v₁ = "filter response" (which neurons activate)
  - σ₁ = entity-specific scalar (the 0.5% perturbation from F148)

This is EXACTLY the IPA converter's codepoint → RECT pair mapping,
but in the transformer's coordinate system.

Tests:
  1. Extract v₁ and σ₁ for each entity at each COMB layer
  2. Navigate by changing σ₁: France scalar → Germany scalar
  3. Design a custom "filter response" for the capital structure class
  4. Full pipeline: rank-1 MLP at ALL COMB layers → does prediction work?
  5. Cross-entity navigation: change σ₁ to target entity → answer change?
  6. Rank-1 weight replacement: W_gate ≈ (W_gate·v₁)⊗v₁ᵀ → does it work?
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
    """Run attention part of one layer, return h_post_attn and normed_for_mlp."""
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


def run_mlp_from_normed(engine, h_pa, nm, li):
    """Run MLP part of one layer from normed input."""
    mlp = engine.layers[li].mlp
    gate = phi_silu(phi_linear(mlp.W_gate, nm))
    up = phi_linear(mlp.W_up, nm)
    return h_pa + phi_linear(mlp.W_down, gate * up)


def predict_token(engine, tokenizer, h):
    h_last = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = phi_linear(engine.lm_head.weight, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80)
    print("  Frontier 7c: The Rank-1 Manifold — Shape Translation on a Line")
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

    # Get token IDs for answers
    answer_tokens = {
        'France': tokenizer.encode(' Paris')[-1],
        'Germany': tokenizer.encode(' Berlin')[-1],
        'Japan': tokenizer.encode(' Tokyo')[-1],
        'Egypt': tokenizer.encode(' Cairo')[-1],
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Extract the rank-1 manifold at each COMB layer
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 1: Extract the Rank-1 Manifold")
    print("=" * 80)

    # Collect MLP inputs at each COMB layer for all entities
    mlp_inputs = {}  # {entity: {layer: normed_last_token}}
    h_at_layer = {}  # {entity: {layer: (h_pa, nm)}}
    baselines = {}   # {entity: final_h}

    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        mlp_inputs[name] = {}
        h_at_layer[name] = {}
        for li in range(15, 21):
            h_pa, nm = run_attn_only(engine, h, li)
            mlp_inputs[name][li] = nm[0, -1].copy()  # (3584,)
            h_at_layer[name][li] = (h_pa.copy(), nm.copy())
            h = run_mlp_from_normed(engine, h_pa, nm, li)

        h = run_layers(engine, h, 21, 28)
        baselines[name] = h.copy()
        _, tok, _ = predict_token(engine, tokenizer, h)
        print(f"  {name}: → {tok[0]!r}")

    # SVD of the 4 entity inputs at each COMB layer
    print("\n  Rank-1 manifold extraction:")
    manifold = {}  # {layer: (v1, scalars_dict, S)}
    for li in range(15, 21):
        inputs = np.stack([mlp_inputs[n][li] for n in entity_names])  # (4, 3584)
        # Center? No — the structure-class shared component IS the principal direction
        U, S, Vt = np.linalg.svd(inputs, full_matrices=False)
        v1 = Vt[0]  # (3584,) — the principal input direction

        # Scalar projection of each entity onto v1
        scalars = {}
        for i, name in enumerate(entity_names):
            scalars[name] = float(inputs[i] @ v1)

        # Energy in rank-1
        energy_1 = float(S[0] ** 2 / np.sum(S ** 2))

        manifold[li] = (v1, scalars, S)
        print(f"  L{li}: rank-1 energy={energy_1:.6f}  "
              f"scalars: {' '.join(f'{n}={scalars[n]:.4f}' for n in entity_names)}")
        # Scalar differences
        print(f"       France-Germany scalar diff: {scalars['France']-scalars['Germany']:.6f}  "
              f"({abs(scalars['France']-scalars['Germany'])/abs(scalars['France'])*100:.3f}%)")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Navigate by Changing the Scalar
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 2: Navigate by Changing σ₁ (Scalar on the Manifold)")
    print("=" * 80)

    # At a single COMB layer: replace France's input with
    # France_projected_to_v1 + (Germany_scalar - France_scalar) * v1
    # This shifts France along the manifold to Germany's position

    for li in [15, 17, 19]:
        v1, scalars, S = manifold[li]
        fr_input = mlp_inputs['France'][li]  # (3584,)
        de_scalar = scalars['Germany']
        fr_scalar = scalars['France']

        # Decompose France's input: parallel to v1 + orthogonal
        fr_parallel = fr_scalar * v1
        fr_ortho = fr_input - fr_parallel

        # Navigate: replace parallel component with Germany's scalar
        navigated = de_scalar * v1 + fr_ortho

        # Run MLP with navigated input
        mlp = engine.layers[li].mlp
        nav_input = navigated.reshape(1, 1, -1).astype(np.float32)
        gate_nav = phi_silu(phi_linear(mlp.W_gate, nav_input))
        up_nav = phi_linear(mlp.W_up, nav_input)
        delta_nav = phi_linear(mlp.W_down, gate_nav * up_nav)[0, 0]

        # Normal France MLP delta at this layer
        fr_nm = h_at_layer['France'][li][1]
        fr_h_pa = h_at_layer['France'][li][0]
        normal_fr = run_mlp_from_normed(engine, fr_h_pa, fr_nm, li)
        delta_fr = (normal_fr - fr_h_pa)[0, -1]

        # Germany's MLP delta at this layer
        de_nm = h_at_layer['Germany'][li][1]
        de_h_pa = h_at_layer['Germany'][li][0]
        normal_de = run_mlp_from_normed(engine, de_h_pa, de_nm, li)
        delta_de = (normal_de - de_h_pa)[0, -1]

        cos_nav_de = cosine(delta_nav, delta_de)
        cos_nav_fr = cosine(delta_nav, delta_fr)
        print(f"  L{li}: scalar shift {fr_scalar:.4f}→{de_scalar:.4f}  "
              f"cos(nav,France)={cos_nav_fr:.4f}  cos(nav,Germany)={cos_nav_de:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Full Pipeline — Scalar Navigation at ALL COMB Layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 3: Full Pipeline — Navigate France→Germany via Scalars")
    print("=" * 80)

    tids_fr = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids_fr)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)

    for li in range(15, 21):
        v1, scalars, S = manifold[li]
        h_pa, nm = run_attn_only(engine, h, li)

        # Navigate ONLY the last token
        nm_mod = nm.copy()
        x_last = nm_mod[0, -1].astype(np.float64)
        x_parallel = float(x_last @ v1)
        x_ortho = x_last - x_parallel * v1
        target_scalar = scalars['Germany']
        nm_mod[0, -1] = (target_scalar * v1 + x_ortho).astype(np.float32)

        h = run_mlp_from_normed(engine, h_pa, nm_mod, li)

    h = run_layers(engine, h, 21, 28)
    _, tok, logits = predict_token(engine, tokenizer, h)

    paris_l = float(logits[answer_tokens['France']])
    berlin_l = float(logits[answer_tokens['Germany']])
    tokyo_l = float(logits[answer_tokens['Japan']])
    cairo_l = float(logits[answer_tokens['Egypt']])
    print(f"  France input + all scalars→Germany:")
    print(f"    → {tok[0]!r}  top5: {tok[:5]}")
    print(f"    Paris={paris_l:.2f}  Berlin={berlin_l:.2f}  "
          f"Tokyo={tokyo_l:.2f}  Cairo={cairo_l:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Navigate to ALL targets
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 4: Navigate France → Each Target")
    print("=" * 80)

    for target in entity_names:
        tids_fr = tokenizer.encode(prompts['France'])
        h = engine.embedding(tids_fr)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        for li in range(15, 21):
            v1, scalars, S = manifold[li]
            h_pa, nm = run_attn_only(engine, h, li)

            nm_mod = nm.copy()
            x_last = nm_mod[0, -1].astype(np.float64)
            x_parallel = float(x_last @ v1)
            x_ortho = x_last - x_parallel * v1
            nm_mod[0, -1] = (scalars[target] * v1 + x_ortho).astype(np.float32)
            h = run_mlp_from_normed(engine, h_pa, nm_mod, li)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)

        ans_logits = {n: float(logits[answer_tokens[n]]) for n in entity_names}
        print(f"  France→{target:8s}: → {tok[0]!r}  "
              f"P={ans_logits['France']:.1f} B={ans_logits['Germany']:.1f} "
              f"T={ans_logits['Japan']:.1f} C={ans_logits['Egypt']:.1f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Navigate Germany → Each Target (bidirectional test)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 5: Navigate Germany → Each Target")
    print("=" * 80)

    for target in entity_names:
        tids_de = tokenizer.encode(prompts['Germany'])
        h = engine.embedding(tids_de)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        for li in range(15, 21):
            v1, scalars, S = manifold[li]
            h_pa, nm = run_attn_only(engine, h, li)

            nm_mod = nm.copy()
            x_last = nm_mod[0, -1].astype(np.float64)
            x_parallel = float(x_last @ v1)
            x_ortho = x_last - x_parallel * v1
            nm_mod[0, -1] = (scalars[target] * v1 + x_ortho).astype(np.float32)
            h = run_mlp_from_normed(engine, h_pa, nm_mod, li)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)

        ans_logits = {n: float(logits[answer_tokens[n]]) for n in entity_names}
        print(f"  Germany→{target:8s}: → {tok[0]!r}  "
              f"P={ans_logits['France']:.1f} B={ans_logits['Germany']:.1f} "
              f"T={ans_logits['Japan']:.1f} C={ans_logits['Egypt']:.1f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Rank-1 Weight Replacement — Can We Replace W_gate?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 6: Rank-1 Weight Replacement")
    print("=" * 80)

    # Replace W_gate with its rank-1 projection through v1:
    # W_gate_approx = (W_gate @ v1) @ v1.T
    # This means the gate only sees the v1 component of any input.

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        v1 = manifold[li][0]
        W_gate_dec = mlp.W_gate.decode_cached()  # (18944, 3584)

        # Rank-1 gate: each row's response is proportional to v1
        filter_response = W_gate_dec @ v1.astype(np.float32)  # (18944,)
        # W_gate_rank1[i, j] = filter_response[i] * v1[j]
        # φ-encode: sign = sign(filter_response[i]) * sign(v1[j])
        #           exp = exp(filter_response[i]) + exp(v1[j])
        # But for simplicity, just build as float and φ-encode
        W_gate_rank1 = np.outer(filter_response, v1).astype(np.float32)
        W_gate_rank1_phi = PhiEncoded.encode(W_gate_rank1)

        for name in ['France', 'Germany']:
            normed_input = mlp_inputs[name][li].reshape(1, 1, -1).astype(np.float32)

            # Normal MLP
            gate_normal = phi_silu(phi_linear(mlp.W_gate, normed_input))
            up_normal = phi_linear(mlp.W_up, normed_input)
            delta_normal = phi_linear(mlp.W_down, gate_normal * up_normal)[0, 0]

            # Rank-1 gate MLP
            gate_r1 = phi_silu(phi_linear(W_gate_rank1_phi, normed_input))
            delta_r1 = phi_linear(mlp.W_down, gate_r1 * up_normal)[0, 0]

            cos_d = cosine(delta_normal, delta_r1)
            # Gate agreement
            cos_g = cosine(gate_normal[0, 0], gate_r1[0, 0])
            print(f"  L{li} {name:8s}: rank-1 gate: gate_cos={cos_g:.4f}  delta_cos={cos_d:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Rank-1 Gate at ALL COMB Layers — Full Pipeline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 7: Rank-1 Gate at ALL COMB Layers")
    print("=" * 80)

    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)

        for li in range(15, 21):
            mlp = engine.layers[li].mlp
            v1 = manifold[li][0]
            W_gate_dec = mlp.W_gate.decode_cached()
            filter_response = W_gate_dec @ v1.astype(np.float32)
            W_gate_rank1 = np.outer(filter_response, v1).astype(np.float32)
            W_gate_r1_phi = PhiEncoded.encode(W_gate_rank1)

            h_pa, nm = run_attn_only(engine, h, li)
            gate_r1 = phi_silu(phi_linear(W_gate_r1_phi, nm))
            up = phi_linear(mlp.W_up, nm)
            h = h_pa + phi_linear(mlp.W_down, gate_r1 * up)

        h = run_layers(engine, h, 21, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)
        ans_l = float(logits[answer_tokens[name]])
        print(f"  {name:8s}: → {tok[0]!r}  (answer logit={ans_l:.2f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 8: The Manifold's Language — Scalar → Answer Mapping
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 8: The Manifold's Language — σ₁ → Answer")
    print("=" * 80)

    # Interpolate along the manifold between France and Germany
    # Does the answer change smoothly?
    v1_17, scalars_17, _ = manifold[17]
    fr_s = scalars_17['France']
    de_s = scalars_17['Germany']

    print(f"  L17 manifold: France σ₁={fr_s:.6f}  Germany σ₁={de_s:.6f}")
    print(f"  Interpolating (only modifying L17):")

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        target_s = fr_s + alpha * (de_s - fr_s)

        tids_fr = tokenizer.encode(prompts['France'])
        h = engine.embedding(tids_fr)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 17)

        h_pa, nm = run_attn_only(engine, h, 17)
        nm_mod = nm.copy()
        x_last = nm_mod[0, -1].astype(np.float64)
        x_parallel = float(x_last @ v1_17)
        x_ortho = x_last - x_parallel * v1_17
        nm_mod[0, -1] = (target_s * v1_17 + x_ortho).astype(np.float32)
        h = run_mlp_from_normed(engine, h_pa, nm_mod, 17)

        h = run_layers(engine, h, 18, 28)
        _, tok, logits = predict_token(engine, tokenizer, h)
        paris_l = float(logits[answer_tokens['France']])
        berlin_l = float(logits[answer_tokens['Germany']])
        print(f"    α={alpha:.2f} σ₁={target_s:.6f}: → {tok[0]!r}  "
              f"Paris={paris_l:.2f}  Berlin={berlin_l:.2f}  gap={paris_l-berlin_l:+.2f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  The Rank-1 Manifold:
  - v₁ = principal input direction for the "capital" structure class
  - σ₁ = entity-specific scalar projection onto v₁
  - W_gate @ v₁ = the "filter response" — which neurons activate for capitals
  
  If scalar navigation works: entity identity IS a point on a line,
    and shape translation reduces to 1-d RECT-pair engineering.
  If rank-1 gate works: the full 18944×3584 W_gate collapses to
    one 18944-d vector + one 3584-d direction.
""")


if __name__ == '__main__':
    main()
