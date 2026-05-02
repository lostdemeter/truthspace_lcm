"""
Frontier 7b: Shape Translation — The Language of Light
========================================================
The weight signs ARE the shapes. The shapes carry the computation.
Now we learn to READ the shapes and WRITE our own.

The IPA converter's language: "at codepoint X, add height H"
The transformer's language: "when input projects onto direction d_i, 
  neuron i fires, selects from W_up, projects through W_down"

Shape Translation means:
  1. Read what the trained shapes select (gate activation analysis)
  2. Understand the vocabulary (what directions do the shapes define?)
  3. Design custom shapes (create sign patterns for known features)
  4. Test the translation (do custom shapes produce correct computation?)

If this works, we can engineer the COMB zone — not by training,
but by DESIGNING binary hyperplane arrangements, just like the
IPA converter designs RECT pairs.
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
    from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
    from phi_geometric.inference.phi_matmul import phi_linear
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


def get_gate_activations(engine, h, layer_idx):
    """Get the MLP gate activations at a specific layer for the last token."""
    layer = engine.layers[layer_idx]
    attn, mlp = layer.attention, layer.mlp
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    sl = h.shape[1]

    # Run attention first
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

    # MLP gate computation
    nm = rms_norm(h_pa, mlp.norm_weight)
    gate_raw = phi_linear(mlp.W_gate, nm)  # (1, seq, 18944)
    gate_activated = phi_silu(gate_raw)     # (1, seq, 18944)
    up_raw = phi_linear(mlp.W_up, nm)      # (1, seq, 18944)

    # Full MLP for continuation
    mlp_out = phi_linear(mlp.W_down, gate_activated * up_raw)
    h_out = h_pa + mlp_out

    return {
        'h_before': h.copy(),
        'h_post_attn': h_pa.copy(),
        'normed': nm.copy(),
        'gate_raw': gate_raw[0, -1].copy(),      # (18944,) last token
        'gate_activated': gate_activated[0, -1].copy(),
        'up_raw': up_raw[0, -1].copy(),
        'h_out': h_out.copy(),
        'mlp_delta': mlp_out[0, -1].copy(),       # (3584,) last token
    }


def run_custom_mlp(normed_last, W_gate, W_up, W_down):
    """Run MLP on a single token with given weight matrices."""
    x = normed_last.reshape(1, 1, -1)
    gate = phi_silu(phi_linear(W_gate, x))
    up = phi_linear(W_up, x)
    return phi_linear(W_down, gate * up)[0, 0]


def predict_token(engine, tokenizer, h):
    h_last = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = phi_linear(engine.lm_head.weight, h_last)[0, 0]
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_tok = [tokenizer.decode([int(i)]) for i in top5_idx]
    return top5_idx, top5_tok, logits


def main():
    print("=" * 80)
    print("  Frontier 7b: Shape Translation — The Language of Light")
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

    # ═══════════════════════════════════════════════════════════
    # Collect gate activations for all prompts at COMB layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 1: Reading the Shapes — Gate Activation Patterns")
    print("=" * 80)

    all_data = {}  # {prompt_name: {layer_idx: gate_info}}
    for name, prompt in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, 15)  # Through pre-COMB

        all_data[name] = {}
        for li in range(15, 21):
            info = get_gate_activations(engine, h, li)
            all_data[name][li] = info
            h = info['h_out']

        # Verify prediction
        h_final = run_layers(engine, h, 21, 28)
        _, tok, _ = predict_token(engine, tokenizer, h_final)
        print(f"  {name}: → {tok[0]!r}")

    # ═══════════════════════════════════════════════════════════
    # Inv 1: Gate Sparsity — How many neurons actually fire?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 1: Gate Sparsity (How Many Neurons Fire?)")
    print("=" * 80)

    for li in range(15, 21):
        print(f"\n  Layer {li}:")
        for name in prompts:
            gate = all_data[name][li]['gate_activated']
            # SiLU: x * sigmoid(x). Positive inputs → positive output, negative → ~0
            n_active = int(np.sum(np.abs(gate) > 0.01))
            n_strong = int(np.sum(np.abs(gate) > 1.0))
            top10_mag = np.sort(np.abs(gate))[-10:][::-1]
            pct_active = 100.0 * n_active / len(gate)
            pct_strong = 100.0 * n_strong / len(gate)
            print(f"    {name:8s}: {n_active:5d}/{len(gate)} active ({pct_active:.1f}%)  "
                  f"{n_strong:5d} strong ({pct_strong:.1f}%)  "
                  f"top3: {top10_mag[0]:.1f}, {top10_mag[1]:.1f}, {top10_mag[2]:.1f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 2: Cross-Entity Gate Pattern Similarity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Cross-Entity Gate Patterns — Same Shape, Different Content?")
    print("=" * 80)

    entity_names = list(prompts.keys())
    for li in [15, 17, 19]:
        print(f"\n  Layer {li} — gate activation cosine:")
        print(f"          ", end="")
        for n2 in entity_names:
            print(f"  {n2:>8s}", end="")
        print()
        for n1 in entity_names:
            print(f"    {n1:8s}", end="")
            g1 = all_data[n1][li]['gate_activated']
            for n2 in entity_names:
                g2 = all_data[n2][li]['gate_activated']
                c = cosine(g1, g2)
                print(f"  {c:8.4f}", end="")
            print()

        # Also: what fraction of neurons fire for ALL entities?
        active_sets = {}
        for name in entity_names:
            gate = all_data[name][li]['gate_activated']
            active_sets[name] = set(np.where(np.abs(gate) > 0.01)[0])

        all_active = active_sets[entity_names[0]]
        for name in entity_names[1:]:
            all_active = all_active & active_sets[name]
        any_active = active_sets[entity_names[0]]
        for name in entity_names[1:]:
            any_active = any_active | active_sets[name]

        print(f"    Neurons active for ALL: {len(all_active)}/{len(any_active)} "
              f"({100*len(all_active)/max(1,len(any_active)):.1f}% Jaccard)")

    # ═══════════════════════════════════════════════════════════
    # Inv 3: Gate Sign Pattern — The Binary Shape
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Gate SIGN Pattern — The Binary Decision")
    print("=" * 80)

    for li in [15, 17, 19]:
        print(f"\n  Layer {li}:")
        # The gate_raw sign is the binary decision: which side of each hyperplane?
        gate_signs = {}
        for name in entity_names:
            gr = all_data[name][li]['gate_raw']
            gate_signs[name] = np.sign(gr)

        # Cross-entity gate sign agreement
        for i, n1 in enumerate(entity_names):
            for j, n2 in enumerate(entity_names):
                if j <= i:
                    continue
                agree = float(np.mean(gate_signs[n1] == gate_signs[n2]))
                diff_count = int(np.sum(gate_signs[n1] != gate_signs[n2]))
                print(f"    {n1:8s} vs {n2:8s}: {agree:.4f} agree  "
                      f"({diff_count} neurons differ)")

        # Entity-specific neurons: fire for one but not others
        for name in entity_names:
            unique_pos = np.ones(18944, dtype=bool)
            for other in entity_names:
                if other == name:
                    continue
                unique_pos &= (gate_signs[name] != gate_signs[other])
            n_unique = int(np.sum(unique_pos))
            print(f"    {name:8s} unique gate sign neurons: {n_unique}")

    # ═══════════════════════════════════════════════════════════
    # Inv 4: The Directions of Light — What Do Active Neurons "See"?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: The Directions of Light — What Do Active Neurons See?")
    print("=" * 80)

    # For each COMB layer, the gate rows define directions in 3584-d space.
    # A neuron "fires" when the input projects positively onto its direction.
    # What are the PRINCIPAL directions that the active neurons define?

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        W_gate_dec = mlp.W_gate.decode_cached()  # (18944, 3584) — full weight

        for name in ['France', 'Germany']:
            gate = all_data[name][li]['gate_activated']
            # Top-K most active neurons
            top_k = 100
            top_idx = np.argsort(np.abs(gate))[-top_k:][::-1]
            top_mag = gate[top_idx]

            # The directions these neurons "see"
            top_dirs = W_gate_dec[top_idx]  # (100, 3584)

            # Weighted average direction (weighted by activation magnitude)
            weights = np.abs(top_mag).reshape(-1, 1)
            avg_dir = (weights * top_dirs).sum(axis=0)  # (3584,)
            avg_dir_norm = avg_dir / (np.linalg.norm(avg_dir) + 1e-20)

            # Does the input align with this average direction?
            normed_input = all_data[name][li]['normed'][0, -1]
            cos_input = cosine(avg_dir, normed_input)

            # SVD of top neuron directions — how many principal directions?
            _, S_top, Vt_top = np.linalg.svd(top_dirs, full_matrices=False)
            energy = np.cumsum(S_top ** 2) / np.sum(S_top ** 2)
            rank_50 = int(np.searchsorted(energy, 0.50) + 1)
            rank_90 = int(np.searchsorted(energy, 0.90) + 1)

            print(f"  L{li} {name:8s}: top-{top_k} neuron dirs rank@50%={rank_50} "
                  f"@90%={rank_90}  avg_dir·input={cos_input:.4f}")

        # Cross-entity: do France and Germany's top neurons see the SAME directions?
        fr_gate = all_data['France'][li]['gate_activated']
        de_gate = all_data['Germany'][li]['gate_activated']
        fr_top = np.argsort(np.abs(fr_gate))[-100:]
        de_top = np.argsort(np.abs(de_gate))[-100:]
        overlap = len(set(fr_top) & set(de_top))
        print(f"  L{li} France↔Germany top-100 neuron overlap: {overlap}/100")

    # ═══════════════════════════════════════════════════════════
    # Inv 5: Shape Translation Attempt — Design a Custom Gate
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Shape Translation — Design Custom Gate Signs")
    print("=" * 80)

    # Strategy: For each COMB layer, compute the "consensus" gate direction
    # from all capital prompts. Then design a sign matrix where each row's
    # sign pattern matches the projection of the consensus direction.
    #
    # This is the geometric equivalent of designing a RECT pair:
    # instead of "activate at codepoint X", we say
    # "activate when input aligns with direction D"

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        W_gate_dec = mlp.W_gate.decode_cached()  # (18944, 3584)

        # Step A: Find consensus activation pattern across all entities
        all_gates = []
        all_normed = []
        for name in entity_names:
            all_gates.append(all_data[name][li]['gate_activated'])
            all_normed.append(all_data[name][li]['normed'][0, -1])
        gates_stack = np.stack(all_gates)  # (4, 18944)
        normed_stack = np.stack(all_normed)  # (4, 3584)

        # Consensus: average gate sign across entities
        avg_gate_sign = np.sign(np.mean(np.sign(gates_stack), axis=0))
        avg_gate_sign[avg_gate_sign == 0] = 1  # break ties
        consensus_frac = float(np.mean(avg_gate_sign == np.sign(gates_stack[0])))

        # Step B: Design custom sign matrix from consensus direction
        # The consensus input direction
        avg_input = normed_stack.mean(axis=0)
        avg_input /= np.linalg.norm(avg_input) + 1e-20

        # Projection of each gate row onto the consensus input direction
        proj = W_gate_dec @ avg_input  # (18944,)
        designed_sign = np.sign(proj).astype(np.int8)
        designed_sign[designed_sign == 0] = 1

        # How well does this match the actual gate signs?
        original_signs = mlp.W_gate.signs
        sign_match = float(np.mean(designed_sign == original_signs[:, 0]))  # column 0

        # Step C: Build the custom sign weight matrix
        # Keep the original exponents but replace signs with our design
        # For each row i, if designed_sign[i] agrees with the ORIGINAL row's
        # projection sign onto avg_input, keep the row sign. Otherwise, flip it.
        #
        # Actually, simpler: we want to preserve the rows that agree with
        # the consensus, and flip the rows that disagree.
        # But that's just: new_sign[i,j] = designed_sign[i] * abs(original_sign[i,j])
        # No — we want to test a DIFFERENT approach:
        #
        # The "consensus shape": a gate matrix where EVERY row points
        # in a direction related to the consensus input.
        # row_i_sign = sign(W_gate_row_i · avg_input) for each dimension j:
        # new_sign[i,j] = sign(avg_input[j]) × original_sign[i,j]
        #
        # Wait, let's think about this differently.
        # The trained shape: W_gate[i,:] has signs that define a hyperplane.
        # What if we design: row i should fire iff the ENTITY-SPECIFIC
        # part of the input has positive projection?

        # Step D: Test — replace gate signs with consensus-derived signs
        # For a simple first test: what if we SORT the neurons by their
        # projection onto the consensus direction, and keep the top/bottom
        # half with their original signs, but set the middle to zero?

        # Simpler first test: what if we use the SIGN of each row's
        # agreement with the mean input as the sign pattern?
        # This means: each row's sign in each dimension = sign(avg_input[dim])
        # So all rows have the SAME sign pattern = the input sign pattern.
        # This creates a "projector" onto the input direction.

        # Build the designed weight: all rows have sign = sign(avg_input)
        input_sign = np.sign(avg_input).astype(np.int8)
        input_sign[input_sign == 0] = 1
        custom_signs = np.tile(input_sign, (18944, 1))  # (18944, 3584)

        custom_W_gate = PhiEncoded(
            signs=custom_signs,
            exponents=mlp.W_gate.exponents.copy(),
        )

        # Test this custom gate
        print(f"\n  Layer {li}: Consensus shape test")
        print(f"    Consensus gate sign agreement: {consensus_frac:.4f}")

        for name in entity_names:
            normed_input = all_data[name][li]['normed'][0, -1]
            normal_delta = all_data[name][li]['mlp_delta']

            # Custom MLP with designed gate
            custom_delta = run_custom_mlp(normed_input, custom_W_gate,
                                           mlp.W_up, mlp.W_down)
            cos_d = cosine(normal_delta, custom_delta)

            # Also test: what if we keep original gate signs but
            # replace UP signs with consensus?
            print(f"    {name:8s}: cos(normal_delta, consensus_gate_delta)={cos_d:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 6: Rank-1 Shape Translation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: Rank-1 Shape — SVD-Based Translation")
    print("=" * 80)

    # The trained W_gate is full-rank. But what if the EFFECTIVE shape
    # (when applied to real inputs from this distribution) is low-rank?
    # Take the top-K SVD components of W_gate and test if they suffice.

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        W_gate_dec = mlp.W_gate.decode_cached()  # (18944, 3584)

        # Full SVD is too expensive for (18944, 3584).
        # Use randomized SVD: project to smaller space first.
        # Or just test with the activation-weighted subspace.

        # Approach: compute W_gate @ input for each entity,
        # then reconstruct from low-rank approximation
        for name in ['France', 'Germany']:
            normed_input = all_data[name][li]['normed'][0, -1]  # (3584,)
            gate_full = W_gate_dec @ normed_input  # (18944,) — real gate

            # What if we project through the TOP singular directions of the input?
            # SVD of the 4 inputs: find the principal input subspace
            input_stack = np.stack([all_data[n][li]['normed'][0, -1]
                                     for n in entity_names])  # (4, 3584)
            _, S_in, Vt_in = np.linalg.svd(input_stack, full_matrices=False)

            # Project W_gate through top-K input directions
            for K in [1, 2, 4]:
                V_k = Vt_in[:K].T  # (3584, K)
                # W_gate projected: W_gate @ V_k @ V_k.T @ input
                proj_input = V_k @ (V_k.T @ normed_input)  # (3584,)
                gate_proj = W_gate_dec @ proj_input  # (18944,)
                cos_g = cosine(gate_full, gate_proj)

                # Now test full MLP with projected input
                proj_delta = run_custom_mlp(proj_input.astype(np.float32),
                                             mlp.W_gate, mlp.W_up, mlp.W_down)
                normal_delta = all_data[name][li]['mlp_delta']
                cos_d = cosine(normal_delta, proj_delta)

                if K <= 2:
                    print(f"  L{li} {name:8s} rank-{K} input: "
                          f"gate_cos={cos_g:.4f}  delta_cos={cos_d:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 7: Entity-Specific Sign Flip — Minimal Shape Edit
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 7: Minimal Shape Edit — Flip Signs That Differ Between Entities")
    print("=" * 80)

    # From Inv 3, we know which gate neurons have different sign patterns
    # for France vs Germany. What if we take France's gate and flip ONLY
    # those neurons to match Germany's pattern? Does this change the output?

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        fr_gate_raw = all_data['France'][li]['gate_raw']
        de_gate_raw = all_data['Germany'][li]['gate_raw']

        fr_sign = np.sign(fr_gate_raw)
        de_sign = np.sign(de_gate_raw)
        diff_mask = fr_sign != de_sign
        n_diff = int(np.sum(diff_mask))

        # Build a modified W_gate: start with original signs,
        # but for each neuron where France and Germany disagree,
        # flip the sign of that gate row (negate all elements)
        modified_signs = mlp.W_gate.signs.copy()  # (18944, 3584)
        # For rows where the gate sign differs, flip the entire row
        flip_rows = np.where(diff_mask)[0]
        modified_signs[flip_rows] *= -1

        modified_W_gate = PhiEncoded(
            signs=modified_signs,
            exponents=mlp.W_gate.exponents.copy(),
        )

        # Run France's input through the modified gate
        fr_normed = all_data['France'][li]['normed'][0, -1]
        fr_normal_delta = all_data['France'][li]['mlp_delta']

        # Normal France delta
        fr_delta_modified = run_custom_mlp(fr_normed, modified_W_gate,
                                            mlp.W_up, mlp.W_down)

        # Compare with Germany's actual delta
        de_normal_delta = all_data['Germany'][li]['mlp_delta']
        cos_fr_mod_vs_de = cosine(fr_delta_modified, de_normal_delta)
        cos_fr_mod_vs_fr = cosine(fr_delta_modified, fr_normal_delta)

        print(f"  L{li}: {n_diff} gate neurons differ France↔Germany")
        print(f"    Modified France delta → cos(Germany actual): {cos_fr_mod_vs_de:.4f}")
        print(f"    Modified France delta → cos(France actual):  {cos_fr_mod_vs_fr:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 8: Full Pipeline Shape Edit — Does It Change the Answer?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 8: Full Pipeline — Shape Edit at Single Layer")
    print("=" * 80)

    # For each COMB layer, flip the entity-specific gate signs
    # (France→Germany mapping) and see if the answer changes.
    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        fr_gate_raw = all_data['France'][li]['gate_raw']
        de_gate_raw = all_data['Germany'][li]['gate_raw']
        diff_mask = np.sign(fr_gate_raw) != np.sign(de_gate_raw)
        flip_rows = np.where(diff_mask)[0]

        # Full forward: France input, normal L0-14, modified L_i, normal rest
        tids = tokenizer.encode(prompts['France'])
        h = engine.embedding(tids)[np.newaxis, :, :]
        h = run_layers(engine, h, 0, li)

        # Modified layer
        modified_signs = mlp.W_gate.signs.copy()
        modified_signs[flip_rows] *= -1
        modified_W_gate = PhiEncoded(
            signs=modified_signs,
            exponents=mlp.W_gate.exponents.copy(),
        )

        # Run this layer with modified gate
        info = get_gate_activations(engine, h, li)
        h_pa = info['h_post_attn']
        nm = rms_norm(h_pa, mlp.norm_weight)
        mod_delta = run_custom_mlp(nm[0, -1], modified_W_gate, mlp.W_up, mlp.W_down)

        # Replace only the last token's MLP delta
        h_mod = h_pa.copy()
        h_mod[0, -1, :] += mod_delta
        # For non-last tokens, use the normal output
        for pos in range(h.shape[1] - 1):
            normal_pos_delta = run_custom_mlp(nm[0, pos], mlp.W_gate, mlp.W_up, mlp.W_down)
            h_mod[0, pos, :] += normal_pos_delta

        h_mod = run_layers(engine, h_mod, li + 1, 28)
        _, tok, logits = predict_token(engine, tokenizer, h_mod)

        # Check what changed
        paris_logit = None
        berlin_logit = None
        paris_id = tokenizer.encode(' Paris')[-1]
        berlin_id = tokenizer.encode(' Berlin')[-1]
        paris_logit = float(logits[paris_id])
        berlin_logit = float(logits[berlin_id])

        print(f"  L{li}: France input + flipped gate ({len(flip_rows)} rows) → {tok[0]!r}")
        print(f"    Paris logit: {paris_logit:.2f}  Berlin logit: {berlin_logit:.2f}  "
              f"gap: {paris_logit - berlin_logit:+.2f}")

    # ═══════════════════════════════════════════════════════════
    # Inv 9: Cumulative Shape Edit — All COMB Layers
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 9: Cumulative Shape Edit — All COMB Layers (France→Germany)")
    print("=" * 80)

    # Flip France→Germany gate signs at ALL COMB layers
    tids = tokenizer.encode(prompts['France'])
    h = engine.embedding(tids)[np.newaxis, :, :]
    h = run_layers(engine, h, 0, 15)

    for li in range(15, 21):
        layer = engine.layers[li]
        attn, mlp = layer.attention, layer.mlp

        # Run attention normally
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

        # Need gate signs from both France and Germany runs at this layer
        # We pre-computed France gate at the original hidden state.
        # For the MODIFIED pipeline, we need to recompute.
        nm = rms_norm(h_pa, mlp.norm_weight)
        gate_raw_here = phi_linear(mlp.W_gate, nm)[0, -1]  # Current gate
        # We want to know what Germany's gate would be — but we only have
        # the pre-computed data from the original run. Let's use the
        # pre-computed Germany gate sign as target.
        de_gate_sign = np.sign(all_data['Germany'][li]['gate_raw'])
        fr_gate_sign = np.sign(gate_raw_here)
        diff_mask = fr_gate_sign != de_gate_sign
        flip_rows = np.where(diff_mask)[0]

        # Build modified gate
        modified_signs = mlp.W_gate.signs.copy()
        modified_signs[flip_rows] *= -1
        modified_W_gate = PhiEncoded(
            signs=modified_signs,
            exponents=mlp.W_gate.exponents.copy(),
        )

        # Run MLP with modified gate (all positions)
        gate_mod = phi_silu(phi_linear(modified_W_gate, nm))
        up = phi_linear(mlp.W_up, nm)
        h = h_pa + phi_linear(mlp.W_down, gate_mod * up)

        print(f"    L{li}: flipped {len(flip_rows)} gate rows")

    h = run_layers(engine, h, 21, 28)
    _, tok, logits = predict_token(engine, tokenizer, h)

    paris_logit = float(logits[tokenizer.encode(' Paris')[-1]])
    berlin_logit = float(logits[tokenizer.encode(' Berlin')[-1]])
    tokyo_logit = float(logits[tokenizer.encode(' Tokyo')[-1]])
    cairo_logit = float(logits[tokenizer.encode(' Cairo')[-1]])

    print(f"\n  France input + ALL COMB gate signs flipped to Germany pattern:")
    print(f"    → {tok[0]!r}  (top-5: {[t for t in tok[:5]]})")
    print(f"    Paris: {paris_logit:.2f}  Berlin: {berlin_logit:.2f}  "
          f"Tokyo: {tokyo_logit:.2f}  Cairo: {cairo_logit:.2f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("""
  Shape Translation results:
  - Gate sparsity: how selective is each shape?
  - Cross-entity patterns: same shape or different?
  - Gate sign agreement: the binary decision boundary
  - Directions of light: what the active neurons "see"
  - Consensus shape: can we design a universal gate?
  - Rank-1 input: how much of the gate is input-specific?
  - Minimal edit: flip entity-specific signs
  - Full pipeline: does shape editing change the answer?
  - Cumulative: all COMB layers edited → what happens?
""")


if __name__ == '__main__':
    main()
