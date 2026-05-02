"""
Frontier 5b: COMB Zone Investigation (L10-L20)
=================================================
F144 found gate codes are content-specific at L10-L20 (cross-struct 0.40-0.49).
This is where the hourglass OPENS and the model does genuine content processing.

Questions:
  1. What do the PRESERVE channels compute? (content-dependent routing)
  2. Does the MLP output match any of the 6 known structures from DC 276?
     - Gyroscope (stable angular displacement)
     - Spectrometer (per-dimension sign rules)
     - Selector (rank-1 direction)
     - Resonator (bias-dominated rank-1)
     - Lens (near-isometric projection)
     - Amplifier (orthogonal boost)
  3. How does MLP vs attention contribution change across the hourglass?
  4. What is the SVD structure of COMB-zone MLP outputs?
  5. Are the content-specific channels a small perturbation or fundamentally different?
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
LOG_PHI = np.log((1 + np.sqrt(5)) / 2)


def decode_weight(w):
    return w.decode() if isinstance(w, PhiEncoded) else w


def to_4state(gate_pre):
    codes = np.zeros_like(gate_pre, dtype=np.int8)
    codes[gate_pre < -LOG_PHI] = 0
    codes[(gate_pre >= -LOG_PHI) & (gate_pre < 0)] = 1
    codes[(gate_pre >= 0) & (gate_pre < LOG_PHI)] = 2
    codes[gate_pre >= LOG_PHI] = 3
    return codes


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def main():
    print("=" * 80, flush=True)
    print("  Frontier 5b: COMB Zone Investigation (L10-L20)", flush=True)
    print("=" * 80, flush=True)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    n_layers = len(engine.layers)
    print(f" done in {time.time()-t0:.1f}s", flush=True)

    prompts = [
        'The capital of France is',
        'The capital of Germany is',
        'The capital of Japan is',
        'I really love eating pizza',
        'Please help me find this',
        'Once upon a time there',
        'How does the engine work',
    ]

    working = []
    for prompt in prompts:
        tids = tokenizer.encode(prompt)
        if len(tids) == 5:
            working.append((prompt, tids))
    print(f"  Using {len(working)} N=5 prompts", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Capture: per-layer attention output, MLP output, gate, norms
    # ═══════════════════════════════════════════════════════════
    print("\n  Capturing per-layer decomposition...", end="", flush=True)

    all_data = {}
    for prompt, tids in working:
        h = engine.embedding(tids)[np.newaxis, :, :]
        layer_data = {}

        for li in range(n_layers):
            layer = engine.layers[li]
            attn = layer.attention
            mlp = layer.mlp
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            sl = h.shape[1]

            # Attention
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
            attn_out = phi_linear(attn.W_o, ao)
            h_pa = h + attn_out

            # MLP
            nm = rms_norm(h_pa, mlp.norm_weight)
            gate_pre = phi_linear(mlp.W_gate, nm)
            up = phi_linear(mlp.W_up, nm)
            gate_act = phi_silu(gate_pre)
            intermediate = gate_act * up
            mlp_out = phi_linear(mlp.W_down, intermediate)

            h_new = h_pa + mlp_out

            layer_data[li] = {
                'h_in': h[0].copy(),
                'attn_out': attn_out[0].copy(),
                'mlp_out': mlp_out[0].copy(),
                'h_out': h_new[0].copy(),
                'gate_pre': gate_pre[0].copy(),
                'gate_codes': to_4state(gate_pre[0]),
                'intermediate': intermediate[0].copy(),
                'attn_weights': w.copy(),
            }

            h = h_new

        all_data[prompt] = layer_data
        gc_mod.collect()

    print(" done", flush=True)
    prompt_names = [p for p, _ in working]
    capital_prompts = [p for p in prompt_names if 'capital' in p]
    diverse_prompts = [p for p in prompt_names if 'capital' not in p]

    # ═══════════════════════════════════════════════════════════
    # Investigation 1: MLP vs Attention Contribution Across Hourglass
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 1: MLP vs Attention Contribution (Amplifier test)", flush=True)
    print("=" * 80, flush=True)

    # DC 276: Amplifier has ||Δmlp|| / ||Δattn|| = 2.1-5.3× at L22-L27
    # and cos(Δattn, Δmlp) ≈ 0 (orthogonal). Does this hold in COMB zone?
    print(f"\n  At last position (averaged across all prompts):", flush=True)
    print(f"  {'Layer':>6} | {'||attn||':>10} {'||mlp||':>10} {'mlp/attn':>10} "
          f"{'cos(a,m)':>10} {'cos(m,Δ)':>10} | {'Gate C%':>7} {'P-%':>5} {'P+%':>5} {'X%':>5}", flush=True)

    for li in range(n_layers):
        a_norms, m_norms, ratios, am_cos, md_cos = [], [], [], [], []
        c_pcts, pn_pcts, pp_pcts, x_pcts = [], [], [], []

        for prompt in prompt_names:
            d = all_data[prompt][li]
            a = d['attn_out'][-1]  # last pos
            m = d['mlp_out'][-1]
            delta = a + m  # total residual update

            an = float(np.linalg.norm(a))
            mn = float(np.linalg.norm(m))
            a_norms.append(an)
            m_norms.append(mn)
            ratios.append(mn / (an + 1e-20))
            am_cos.append(cosine(a, m))
            md_cos.append(cosine(m, delta))

            gc = d['gate_codes'][-1]
            c_pcts.append(100 * np.mean(gc == 0))
            pn_pcts.append(100 * np.mean(gc == 1))
            pp_pcts.append(100 * np.mean(gc == 2))
            x_pcts.append(100 * np.mean(gc == 3))

        print(f"  L{li:4d} | {np.mean(a_norms):10.1f} {np.mean(m_norms):10.1f} "
              f"{np.mean(ratios):10.2f} {np.mean(am_cos):10.4f} {np.mean(md_cos):10.4f} | "
              f"{np.mean(c_pcts):6.1f}% {np.mean(pn_pcts):4.1f}% {np.mean(pp_pcts):4.1f}% "
              f"{np.mean(x_pcts):4.1f}%", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 2: MLP Output SVD Structure (Resonator/Lens test)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 2: MLP Output SVD Structure", flush=True)
    print("=" * 80, flush=True)

    # Resonator: rank-1 (S[0]/S[1] >> 1)
    # Lens: near-isometric (S[0]/S[1] ≈ 1)
    # Amplifier: orthogonal to attention, large norm
    # Stack MLP outputs from all prompts at last position → SVD
    print(f"\n  SVD of MLP outputs stacked across prompts (last pos):", flush=True)
    print(f"  {'Layer':>6} | {'S[0]':>10} {'S[1]':>10} {'S[0]/S[1]':>10} "
          f"{'rank@90%':>8} {'rank@99%':>8} | {'Type':>12}", flush=True)

    for li in range(n_layers):
        # Stack mlp outputs from all prompts at last position
        mlp_stack = np.array([all_data[p][li]['mlp_out'][-1] for p in prompt_names])
        # SVD of the (n_prompts × 3584) matrix
        U, S, Vt = np.linalg.svd(mlp_stack, full_matrices=False)
        s0s1 = S[0] / (S[1] + 1e-20) if len(S) > 1 else float('inf')

        # Rank at 90% and 99% energy
        energy = np.cumsum(S ** 2)
        total = energy[-1]
        rank90 = int(np.searchsorted(energy, 0.90 * total) + 1) if total > 0 else 0
        rank99 = int(np.searchsorted(energy, 0.99 * total) + 1) if total > 0 else 0

        # Classify
        if s0s1 > 100:
            stype = "RANK-1"
        elif s0s1 > 10:
            stype = "LOW-RANK"
        elif s0s1 < 1.5:
            stype = "ISOMETRIC"
        else:
            stype = "SPREAD"

        print(f"  L{li:4d} | {S[0]:10.1f} {S[1]:10.1f} {s0s1:10.1f} "
              f"{rank90:8d} {rank99:8d} | {stype:>12}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 3: COMB Zone PRESERVE Channel Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 3: PRESERVE Channel Analysis in COMB Zone", flush=True)
    print("=" * 80, flush=True)

    # At L10-L20, which channels are PRESERVE and what do they carry?
    # Compare PRESERVE channel masks across prompts
    print(f"\n  PRESERVE channel overlap between prompt pairs (Jaccard index):", flush=True)
    print(f"  {'Layer':>6} | {'Same-struct':>11} {'Cross-struct':>12} | "
          f"{'PRESERVE count':>14}", flush=True)

    for li in [5, 10, 12, 15, 18, 20, 23, 27]:
        same_j, cross_j = [], []
        preserve_counts = []

        for prompt in prompt_names:
            gc = all_data[prompt][li]['gate_codes'][-1]  # last pos
            preserve = (gc == 1) | (gc == 2)  # PRESERVE- or PRESERVE+
            preserve_counts.append(int(np.sum(preserve)))

        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                g1 = all_data[capital_prompts[i]][li]['gate_codes'][-1]
                g2 = all_data[capital_prompts[j]][li]['gate_codes'][-1]
                p1 = (g1 == 1) | (g1 == 2)
                p2 = (g2 == 1) | (g2 == 2)
                intersection = float(np.sum(p1 & p2))
                union = float(np.sum(p1 | p2))
                same_j.append(intersection / (union + 1e-20))

        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                g1 = all_data[p1][li]['gate_codes'][-1]
                g2 = all_data[p2][li]['gate_codes'][-1]
                pp1 = (g1 == 1) | (g1 == 2)
                pp2 = (g2 == 1) | (g2 == 2)
                intersection = float(np.sum(pp1 & pp2))
                union = float(np.sum(pp1 | pp2))
                cross_j.append(intersection / (union + 1e-20))

        print(f"  L{li:4d} | {np.mean(same_j):11.4f} {np.mean(cross_j):12.4f} | "
              f"{np.mean(preserve_counts):14.0f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 4: MLP Output Cross-Prompt Cosine (Content Independence)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 4: MLP Output Cross-Prompt Similarity", flush=True)
    print("=" * 80, flush=True)

    # If MLP output is content-independent (like attention templates),
    # cross-prompt cosine should be high.
    # If content-specific, it will be low.
    print(f"\n  MLP output cosine at last position:", flush=True)
    print(f"  {'Layer':>6} | {'Same-struct':>11} {'Cross-struct':>12} | "
          f"{'||mlp||':>10}", flush=True)

    for li in range(n_layers):
        same_cos, cross_cos = [], []
        mlp_norms = []

        for prompt in prompt_names:
            mlp_norms.append(float(np.linalg.norm(all_data[prompt][li]['mlp_out'][-1])))

        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                m1 = all_data[capital_prompts[i]][li]['mlp_out'][-1]
                m2 = all_data[capital_prompts[j]][li]['mlp_out'][-1]
                same_cos.append(cosine(m1, m2))

        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                m1 = all_data[p1][li]['mlp_out'][-1]
                m2 = all_data[p2][li]['mlp_out'][-1]
                cross_cos.append(cosine(m1, m2))

        print(f"  L{li:4d} | {np.mean(same_cos):11.4f} {np.mean(cross_cos):12.4f} | "
              f"{np.mean(mlp_norms):10.1f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 5: Attention Output Cross-Prompt Cosine
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 5: Attention Output Cross-Prompt Similarity", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Attention output cosine at last position:", flush=True)
    print(f"  {'Layer':>6} | {'Same-struct':>11} {'Cross-struct':>12} | "
          f"{'||attn||':>10}", flush=True)

    for li in range(n_layers):
        same_cos, cross_cos = [], []
        attn_norms = []

        for prompt in prompt_names:
            attn_norms.append(float(np.linalg.norm(all_data[prompt][li]['attn_out'][-1])))

        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                a1 = all_data[capital_prompts[i]][li]['attn_out'][-1]
                a2 = all_data[capital_prompts[j]][li]['attn_out'][-1]
                same_cos.append(cosine(a1, a2))

        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                a1 = all_data[p1][li]['attn_out'][-1]
                a2 = all_data[p2][li]['attn_out'][-1]
                cross_cos.append(cosine(a1, a2))

        print(f"  L{li:4d} | {np.mean(same_cos):11.4f} {np.mean(cross_cos):12.4f} | "
              f"{np.mean(attn_norms):10.1f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 6: Gyroscope Test — Angular Stability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 6: Angular Stability (Gyroscope Test)", flush=True)
    print("=" * 80, flush=True)

    # Gyroscope: cos(h_in, h_out) should be high and stable
    # If the residual stream is self-correcting, the angle between
    # input and output should be predictable
    print(f"\n  cos(h_in, h_out) at last position:", flush=True)
    print(f"  {'Layer':>6} | {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} | "
          f"{'||h_in||':>10} {'||h_out||':>10}", flush=True)

    for li in range(n_layers):
        coses, h_in_norms, h_out_norms = [], [], []
        for prompt in prompt_names:
            d = all_data[prompt][li]
            c = cosine(d['h_in'][-1], d['h_out'][-1])
            coses.append(c)
            h_in_norms.append(float(np.linalg.norm(d['h_in'][-1])))
            h_out_norms.append(float(np.linalg.norm(d['h_out'][-1])))

        print(f"  L{li:4d} | {np.mean(coses):8.4f} {np.std(coses):8.4f} "
              f"{np.min(coses):8.4f} {np.max(coses):8.4f} | "
              f"{np.mean(h_in_norms):10.1f} {np.mean(h_out_norms):10.1f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Investigation 7: MLP Intermediate Structure — Per-Channel Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Inv 7: COMB MLP Intermediate — PRESERVE Channel Content", flush=True)
    print("=" * 80, flush=True)

    # For a representative COMB layer (L15), examine the PRESERVE channels
    # Do they carry structured or random content?
    for test_layer in [10, 15, 20]:
        print(f"\n  Layer {test_layer}:", flush=True)

        # Get intermediate activations at PRESERVE channels
        preserve_vals = []
        for prompt in prompt_names:
            d = all_data[prompt][test_layer]
            gc = d['gate_codes'][-1]  # last pos, [18944]
            inter = d['intermediate'][-1]  # last pos, [18944]

            preserve_mask = (gc == 1) | (gc == 2)
            preserve_inter = inter[preserve_mask]
            preserve_vals.append(preserve_inter)

        # Cross-prompt cosine of PRESERVE-channel intermediates
        print(f"    PRESERVE intermediate cross-prompt cosine:", flush=True)
        same_cos, cross_cos = [], []
        for i in range(len(capital_prompts)):
            for j in range(i + 1, len(capital_prompts)):
                pi = prompt_names.index(capital_prompts[i])
                pj = prompt_names.index(capital_prompts[j])
                # Need shared PRESERVE channels
                gi = all_data[capital_prompts[i]][test_layer]['gate_codes'][-1]
                gj = all_data[capital_prompts[j]][test_layer]['gate_codes'][-1]
                shared = ((gi == 1) | (gi == 2)) & ((gj == 1) | (gj == 2))
                if np.sum(shared) > 10:
                    vi = all_data[capital_prompts[i]][test_layer]['intermediate'][-1][shared]
                    vj = all_data[capital_prompts[j]][test_layer]['intermediate'][-1][shared]
                    same_cos.append(cosine(vi, vj))

        for p1 in capital_prompts:
            for p2 in diverse_prompts:
                g1 = all_data[p1][test_layer]['gate_codes'][-1]
                g2 = all_data[p2][test_layer]['gate_codes'][-1]
                shared = ((g1 == 1) | (g1 == 2)) & ((g2 == 1) | (g2 == 2))
                if np.sum(shared) > 10:
                    v1 = all_data[p1][test_layer]['intermediate'][-1][shared]
                    v2 = all_data[p2][test_layer]['intermediate'][-1][shared]
                    cross_cos.append(cosine(v1, v2))

        if same_cos:
            print(f"      Same-struct: {np.mean(same_cos):.4f}", flush=True)
        if cross_cos:
            print(f"      Cross-struct: {np.mean(cross_cos):.4f}", flush=True)

        # SVD of the PRESERVE intermediates stacked across prompts
        # Get shared PRESERVE channels across ALL prompts
        all_gc = [all_data[p][test_layer]['gate_codes'][-1] for p in prompt_names]
        shared_preserve = np.ones(18944, dtype=bool)
        for gc in all_gc:
            shared_preserve &= ((gc == 1) | (gc == 2))

        n_shared = int(np.sum(shared_preserve))
        print(f"    Shared PRESERVE channels across ALL prompts: {n_shared}", flush=True)

        if n_shared > 5:
            stack = np.array([
                all_data[p][test_layer]['intermediate'][-1][shared_preserve]
                for p in prompt_names
            ])
            U, S, Vt = np.linalg.svd(stack, full_matrices=False)
            s0s1 = S[0] / (S[1] + 1e-20) if len(S) > 1 else float('inf')
            print(f"    SVD of shared PRESERVE intermediates: S[0]={S[0]:.1f} S[1]={S[1]:.1f} "
                  f"ratio={s0s1:.1f}", flush=True)
            energy = np.cumsum(S ** 2)
            total = energy[-1]
            for pct in [50, 90, 99]:
                r = int(np.searchsorted(energy, pct / 100 * total) + 1)
                print(f"    rank@{pct}% = {r}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  Summary: COMB Zone Structure", flush=True)
    print("=" * 80, flush=True)
    print("""
  Questions answered:
  1. How does MLP vs attention balance change across the hourglass?
  2. Is MLP output rank-1 (Resonator), isometric (Lens), or spread?
  3. Are PRESERVE channels shared or different across prompts?
  4. Is MLP output content-independent (like attention templates)?
  5. Is attention output content-independent in the COMB zone?
  6. Does the Gyroscope (angular stability) hold in the COMB zone?
  7. What's in the PRESERVE-channel intermediates?

  We're testing if the COMB zone matches any of DC 276's 6 structures
  or if it's something genuinely new.
""", flush=True)


if __name__ == '__main__':
    main()
