"""
Frontier 8: Multi-Class Rank-1 Test — The Superposition of Shapes
===================================================================
DC 280 predicts that every structure class has its own rank-1 manifold:
  W_gate ≈ Σ_c  f_c ⊗ v_c^T

Tests:
  P1: Different structure classes each have rank-1 energy > 90%
  P2: Filter responses f_c are unique per class (|cos| < 0.5)
  P3: v_c directions are nearly orthogonal (|cos| < 0.1)
  P4: Rank-1 gate replacement works for each class independently
  
Structure classes tested:
  - capitals:   "The capital of X is"
  - colors:     "The color of X is"
  - continents: "X is located in the continent of"
  - opposites:  "The opposite of X is"
  - languages:  "The official language of X is"
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
    print("  Frontier 8: Multi-Class Rank-1 Test — The Superposition of Shapes")
    print("=" * 80)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Define structure classes with 4 prompts each
    # ═══════════════════════════════════════════════════════════
    classes = {
        'capitals': {
            'France': 'The capital of France is',
            'Germany': 'The capital of Germany is',
            'Japan': 'The capital of Japan is',
            'Egypt': 'The capital of Egypt is',
        },
        'colors': {
            'grass': 'The color of grass is',
            'sky': 'The color of the sky is',
            'blood': 'The color of blood is',
            'snow': 'The color of snow is',
        },
        'continents': {
            'Brazil': 'Brazil is located in the continent of',
            'China': 'China is located in the continent of',
            'Nigeria': 'Nigeria is located in the continent of',
            'Sweden': 'Sweden is located in the continent of',
        },
        'opposites': {
            'hot': 'The opposite of hot is',
            'big': 'The opposite of big is',
            'fast': 'The opposite of fast is',
            'dark': 'The opposite of dark is',
        },
        'languages': {
            'France_L': 'The official language of France is',
            'Japan_L': 'The official language of Japan is',
            'Brazil_L': 'The official language of Brazil is',
            'Germany_L': 'The official language of Germany is',
        },
    }

    class_names = list(classes.keys())

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Collect MLP inputs at COMB layers for all classes
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 1: Baseline Predictions & MLP Input Collection")
    print("=" * 80)

    # {class_name: {entity: {layer: normed_last}}}
    all_mlp_inputs = {}
    # {class_name: {entity: h_pre_comb}}
    all_h_pre = {}

    for cname, prompts in classes.items():
        print(f"\n  [{cname}]")
        all_mlp_inputs[cname] = {}
        all_h_pre[cname] = {}

        for ename, prompt in prompts.items():
            tids = tokenizer.encode(prompt)
            h = engine.embedding(tids)[np.newaxis, :, :]
            h = run_layers(engine, h, 0, 15)
            all_h_pre[cname][ename] = h.copy()

            all_mlp_inputs[cname][ename] = {}
            for li in range(15, 21):
                h_pa, nm = run_attn_only(engine, h, li)
                all_mlp_inputs[cname][ename][li] = nm[0, -1].copy()
                mlp = engine.layers[li].mlp
                gate = phi_silu(phi_linear(mlp.W_gate, nm))
                up = phi_linear(mlp.W_up, nm)
                h = h_pa + phi_linear(mlp.W_down, gate * up)

            h = run_layers(engine, h, 21, 28)
            _, tok, _ = predict_token(engine, tokenizer, h)
            print(f"    {ename:12s}: → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: P1 — Rank-1 Energy per Structure Class
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 2: P1 — Rank-1 Energy per Structure Class")
    print("=" * 80)

    # {class_name: {layer: (v1, scalars, S, energy_1)}}
    manifolds = {}

    for cname in class_names:
        manifolds[cname] = {}
        enames = list(classes[cname].keys())
        print(f"\n  [{cname}]")

        for li in range(15, 21):
            inputs = np.stack([all_mlp_inputs[cname][e][li] for e in enames])
            U, S, Vt = np.linalg.svd(inputs, full_matrices=False)
            v1 = Vt[0]
            energy_1 = float(S[0]**2 / np.sum(S**2))

            scalars = {}
            for i, e in enumerate(enames):
                scalars[e] = float(inputs[i] @ v1)

            manifolds[cname][li] = (v1, scalars, S, energy_1)

        # Summary: average rank-1 energy across layers
        avg_e = np.mean([manifolds[cname][li][3] for li in range(15, 21)])
        e_by_layer = [f"{manifolds[cname][li][3]:.3f}" for li in [15, 17, 19]]
        print(f"    Rank-1 energy: L15={e_by_layer[0]}  L17={e_by_layer[1]}  "
              f"L19={e_by_layer[2]}  avg={avg_e:.3f}")

        # Scalar range
        for li in [17]:
            _, scalars, _, _ = manifolds[cname][li]
            vals = list(scalars.values())
            spread = max(vals) - min(vals)
            mean_s = np.mean(vals)
            print(f"    L17 scalars: {' '.join(f'{e[:6]}={s:.2f}' for e, s in scalars.items())}")
            print(f"    L17 spread: {spread:.4f} ({spread/abs(mean_s)*100:.2f}% of mean)")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: P3 — Cross-Class v₁ Orthogonality
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 3: P3 — Are v₁ Directions Orthogonal Across Classes?")
    print("=" * 80)

    for li in [15, 17, 19]:
        print(f"\n  Layer {li} — |cos(v₁_c1, v₁_c2)|:")
        print(f"  {'':14s}", end="")
        for c2 in class_names:
            print(f"  {c2:>10s}", end="")
        print()

        for c1 in class_names:
            print(f"    {c1:12s}", end="")
            v1_c1 = manifolds[c1][li][0]
            for c2 in class_names:
                v1_c2 = manifolds[c2][li][0]
                c = abs(cosine(v1_c1, v1_c2))
                print(f"  {c:10.4f}", end="")
            print()

    # ═══════════════════════════════════════════════════════════
    # Phase 4: P2 — Filter Response Uniqueness
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 4: P2 — Filter Response Uniqueness")
    print("=" * 80)

    # f_c = W_gate · v_c for each class
    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        W_gate_dec = mlp.W_gate.decode_cached()

        filters = {}
        for cname in class_names:
            v1 = manifolds[cname][li][0]
            filters[cname] = W_gate_dec @ v1.astype(np.float32)

        print(f"\n  Layer {li} — |cos(f_c1, f_c2)|:")
        print(f"  {'':14s}", end="")
        for c2 in class_names:
            print(f"  {c2:>10s}", end="")
        print()

        for c1 in class_names:
            print(f"    {c1:12s}", end="")
            for c2 in class_names:
                c = abs(cosine(filters[c1], filters[c2]))
                print(f"  {c:10.4f}", end="")
            print()

    # ═══════════════════════════════════════════════════════════
    # Phase 5: P1 — Rank-1 Gate Replacement per Class
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 5: P1 — Rank-1 Gate at ALL COMB Layers (Per Class)")
    print("=" * 80)

    for cname, prompts in classes.items():
        print(f"\n  [{cname}]")
        enames = list(prompts.keys())

        for ename in enames:
            h = all_h_pre[cname][ename].copy()

            for li in range(15, 21):
                v1 = manifolds[cname][li][0]
                mlp = engine.layers[li].mlp
                W_gate_dec = mlp.W_gate.decode_cached()
                filter_gate = W_gate_dec @ v1.astype(np.float32)
                W_gate_r1 = np.outer(filter_gate, v1).astype(np.float32)
                W_gate_r1_phi = PhiEncoded.encode(W_gate_r1)

                h_pa, nm = run_attn_only(engine, h, li)
                gate_r1 = phi_silu(phi_linear(W_gate_r1_phi, nm))
                up = phi_linear(mlp.W_up, nm)
                h = h_pa + phi_linear(mlp.W_down, gate_r1 * up)

            h = run_layers(engine, h, 21, 28)
            _, tok, logits = predict_token(engine, tokenizer, h)
            print(f"    {ename:12s} rank-1 gate: → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Cross-Class Interference — P3 detailed
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 6: P3 — Cross-Class Interference (Wrong v₁)")
    print("=" * 80)

    # Use capitals' v₁ for colors, and vice versa — should it fail?
    cross_tests = [
        ('capitals', 'colors'),
        ('capitals', 'opposites'),
        ('colors', 'continents'),
    ]

    for class_src, class_wrong in cross_tests:
        print(f"\n  [{class_src}] with {class_wrong}'s v₁:")
        prompts_src = classes[class_src]
        enames = list(prompts_src.keys())

        for ename in enames[:2]:  # Test 2 per pair
            h = all_h_pre[class_src][ename].copy()

            for li in range(15, 21):
                v1_wrong = manifolds[class_wrong][li][0]
                mlp = engine.layers[li].mlp
                W_gate_dec = mlp.W_gate.decode_cached()
                filter_wrong = W_gate_dec @ v1_wrong.astype(np.float32)
                W_gate_wrong = np.outer(filter_wrong, v1_wrong).astype(np.float32)
                W_gate_wrong_phi = PhiEncoded.encode(W_gate_wrong)

                h_pa, nm = run_attn_only(engine, h, li)
                gate_wrong = phi_silu(phi_linear(W_gate_wrong_phi, nm))
                up = phi_linear(mlp.W_up, nm)
                h = h_pa + phi_linear(mlp.W_down, gate_wrong * up)

            h = run_layers(engine, h, 21, 28)
            _, tok, _ = predict_token(engine, tokenizer, h)
            print(f"    {ename:12s}: → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Phase 7: BOTH Rank-1 (Gate AND W_up) per Class
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 7: BOTH Rank-1 per Class")
    print("=" * 80)

    for cname in class_names:
        print(f"\n  [{cname}]")
        prompts_c = classes[cname]
        enames = list(prompts_c.keys())

        for ename in enames[:2]:  # Test 2 per class
            h = all_h_pre[cname][ename].copy()

            for li in range(15, 21):
                v1 = manifolds[cname][li][0]
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
            print(f"    {ename:12s} both-r1: → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"\n  Structure classes tested: {len(class_names)}")
    print(f"  Prompts per class: 4")
    print(f"  COMB layers tested: L15-L20")
    print()
    print("  DC 280 Predictions:")
    print("  P1: Rank-1 energy > 90% per class?")
    print("  P2: Filter responses unique per class?")
    print("  P3: v₁ directions orthogonal across classes?")
    print("  P4: Rank-1 gate works per class?")
    print()


if __name__ == '__main__':
    main()
