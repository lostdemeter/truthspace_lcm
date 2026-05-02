"""
Frontier 9: Holographic Analysis of Weight Matrices
=====================================================
Apply the Holographer's Workbench tools to transformer weight matrices.

The tools that built the path now analyze what the path revealed.

Investigations:
  Inv 1: Fractal Peel of Singular Value Spectrum — how many structure classes?
  Inv 2: Resfrac Score per Layer — structure depth across COMB zone
  Inv 3: Holographic Refinement — separate overlapping structure classes
  Inv 4: Disparity Maps — perturb v₁ to find class-sensitive neurons
  Inv 5: Error Pattern Analysis — what's hiding in the rank-1 residual?
  Inv 6: Phase Retrieval — can we recover structure directions without labels?
"""

import sys, os, time
import gc as gc_mod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'temp', 'outside_projects', 'holographersworkbench'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

from workbench import FractalPeeler, resfrac_score
from workbench.processors.holographic import holographic_refinement, phase_retrieve_hilbert
from workbench.analysis.errors import ErrorPatternAnalyzer
from workbench.primitives.signal import normalize

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
    print("  Frontier 9: Holographic Analysis of Weight Matrices")
    print("  The tools that built the path now analyze what the path revealed.")
    print("=" * 80)

    gc_mod.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    # Structure classes for context
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
        'opposites': {
            'hot': 'The opposite of hot is',
            'big': 'The opposite of big is',
            'fast': 'The opposite of fast is',
            'dark': 'The opposite of dark is',
        },
    }

    # ═══════════════════════════════════════════════════════════
    # Collect MLP inputs and extract v₁ per class
    # ═══════════════════════════════════════════════════════════
    print("\n  Collecting MLP inputs at COMB layers...")

    all_mlp_inputs = {}  # {class: {entity: {layer: normed_last}}}
    all_h_pre = {}

    for cname, prompts in classes.items():
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

    # Extract v₁ per class per layer
    manifolds = {}  # {class: {layer: (v1, S)}}
    for cname in classes:
        manifolds[cname] = {}
        enames = list(classes[cname].keys())
        for li in range(15, 21):
            inputs = np.stack([all_mlp_inputs[cname][e][li] for e in enames])
            U, S, Vt = np.linalg.svd(inputs, full_matrices=False)
            manifolds[cname][li] = (Vt[0], S)

    print("  Done.\n")

    # ═══════════════════════════════════════════════════════════
    # INV 1: Fractal Peel of W_gate Singular Value Spectrum
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Inv 1: Fractal Peel of W_gate Singular Value Spectrum")
    print("=" * 80)

    for li in [15, 17, 19]:
        mlp = engine.layers[li].mlp
        W_gate_dec = mlp.W_gate.decode_cached().astype(np.float64)

        print(f"\n  Layer {li}: W_gate shape = {W_gate_dec.shape}")

        # Full SVD of W_gate
        t0 = time.time()
        _, S_full, _ = np.linalg.svd(W_gate_dec, full_matrices=False)
        print(f"    SVD computed in {time.time()-t0:.1f}s")
        print(f"    Top 10 singular values: {S_full[:10].round(2)}")
        print(f"    S[0]/S[1] = {S_full[0]/S_full[1]:.3f}")
        print(f"    S[0]/S[-1] = {S_full[0]/S_full[-1]:.1f}")

        # Energy distribution
        S2 = S_full ** 2
        cumulative = np.cumsum(S2) / np.sum(S2)
        for threshold in [0.5, 0.9, 0.95, 0.99]:
            rank_at = int(np.searchsorted(cumulative, threshold)) + 1
            print(f"    Rank@{threshold:.0%} energy: {rank_at}")

        # Fractal peel on the SV spectrum itself
        # The SV spectrum is a 1D signal — peel it!
        sv_signal = S_full.copy()
        rho = resfrac_score(sv_signal, order=3)
        print(f"    Resfrac(SV spectrum): ρ = {rho:.4f}")

        peeler = FractalPeeler(order=4, noise_threshold=0.95, max_depth=8)
        tree = peeler.compress(sv_signal)
        stats = peeler.tree_stats(tree)
        ratio = peeler.compression_ratio(tree, len(sv_signal))
        print(f"    Fractal peel: depth={stats['max_depth']}, "
              f"nodes={stats['num_nodes']}, leaves={stats['num_leaves']}, "
              f"compression={ratio:.2f}x")

        # Log-SV spectrum (often more revealing)
        log_sv = np.log(S_full + 1e-30)
        rho_log = resfrac_score(log_sv, order=3)
        print(f"    Resfrac(log-SV): ρ = {rho_log:.4f}")

    # ═══════════════════════════════════════════════════════════
    # INV 2: Resfrac Across ALL Layers (not just COMB)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 2: Resfrac Score — Structure Depth Across Layers")
    print("=" * 80)

    layer_resfracs = []
    for li in [0, 3, 7, 10, 13, 15, 17, 19, 22, 25, 27]:
        mlp = engine.layers[li].mlp
        W_dec = mlp.W_gate.decode_cached().astype(np.float64)
        _, S, _ = np.linalg.svd(W_dec, full_matrices=False)
        rho = resfrac_score(S, order=3)
        rho_log = resfrac_score(np.log(S + 1e-30), order=3)
        S2 = S ** 2
        cumul = np.cumsum(S2) / np.sum(S2)
        r90 = int(np.searchsorted(cumul, 0.9)) + 1
        layer_resfracs.append((li, rho, rho_log, r90, S[0]/S[1]))
        print(f"    L{li:2d}: ρ={rho:.4f}  ρ_log={rho_log:.4f}  "
              f"rank@90%={r90:4d}  S0/S1={S[0]/S[1]:.3f}")

    # ═══════════════════════════════════════════════════════════
    # INV 3: Holographic Refinement — Separate Overlapping Classes
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 3: Holographic Refinement — Separate Structure Classes")
    print("=" * 80)

    li = 17  # Focus on L17 (middle of COMB)
    mlp = engine.layers[li].mlp
    W_gate_dec = mlp.W_gate.decode_cached()

    # Get filter responses for each class
    filters = {}
    for cname in classes:
        v1 = manifolds[cname][li][0]
        filters[cname] = (W_gate_dec @ v1.astype(np.float32)).astype(np.float64)

    class_names = list(classes.keys())
    print(f"\n  Layer {li} — Raw filter response cosines:")
    for c1 in class_names:
        for c2 in class_names:
            if c1 < c2:
                c = abs(cosine(filters[c1], filters[c2]))
                print(f"    {c1}↔{c2}: {c:.4f}")

    # Apply holographic refinement: use capitals as "object", others as "reference"
    print(f"\n  Holographic refinement (capitals as object, others as reference):")
    obj = filters['capitals'].copy()

    for ref_name in ['colors', 'opposites']:
        ref = filters[ref_name].copy()
        # Normalize for refinement
        obj_norm = normalize(obj, 'max')
        ref_norm = normalize(ref, 'max')
        refined = holographic_refinement(obj_norm, ref_norm, method='hilbert', blend_ratio=0.6)

        cos_before = abs(cosine(obj, ref))
        cos_after = abs(cosine(refined, ref_norm))
        cos_self = abs(cosine(refined, obj_norm))

        print(f"    Refined vs {ref_name}:")
        print(f"      Before refinement: |cos(capitals, {ref_name})| = {cos_before:.4f}")
        print(f"      After refinement:  |cos(refined,  {ref_name})| = {cos_after:.4f}")
        print(f"      Self-similarity:   |cos(refined,  capitals)|  = {cos_self:.4f}")

    # Phase variance of filter responses — are they "coherent" or "noisy"?
    print(f"\n  Phase retrieval on filter responses:")
    for cname in class_names:
        f = filters[cname]
        env, pv = phase_retrieve_hilbert(f)
        print(f"    {cname:12s}: phase_var = {pv:.4f}  "
              f"envelope_range = [{env.min():.2f}, {env.max():.2f}]")

    # ═══════════════════════════════════════════════════════════
    # INV 4: Disparity Maps — Perturb v₁ to Find Class-Sensitive Neurons
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 4: Disparity Maps — Class-Sensitive Neurons")
    print("=" * 80)

    li = 17
    W_gate_dec = engine.layers[li].mlp.W_gate.decode_cached()

    # Compute disparity: f(v_capitals + α·Δv) - f(v_capitals - α·Δv)
    # where Δv = direction from capitals to colors
    v_cap = manifolds['capitals'][li][0].astype(np.float32)
    v_col = manifolds['colors'][li][0].astype(np.float32)

    # Direction from capitals to colors (in v₁ space)
    delta_v = v_col - v_cap * np.dot(v_cap, v_col)  # orthogonal component
    delta_v = delta_v / (np.linalg.norm(delta_v) + 1e-20)

    alpha = 0.1  # Small perturbation
    f_left = W_gate_dec @ (v_cap - alpha * delta_v)
    f_right = W_gate_dec @ (v_cap + alpha * delta_v)
    disparity = f_right - f_left

    # Disparity statistics
    disp_abs = np.abs(disparity)
    print(f"\n  Layer {li} — Disparity map (capitals perturbed toward colors):")
    print(f"    α = {alpha}")
    print(f"    Disparity: mean = {disp_abs.mean():.6f}, "
          f"max = {disp_abs.max():.6f}, std = {disp_abs.std():.6f}")

    # How many neurons are "class-sensitive"?
    threshold = disp_abs.mean() + 2 * disp_abs.std()
    n_sensitive = int(np.sum(disp_abs > threshold))
    print(f"    Neurons above 2σ: {n_sensitive}/{len(disparity)} "
          f"({100*n_sensitive/len(disparity):.1f}%)")

    # Compare with the actual gate difference between classes
    f_cap = W_gate_dec @ v_cap
    f_col = W_gate_dec @ v_col
    actual_diff = np.abs(f_col - f_cap)
    print(f"    Actual |f_colors - f_capitals|: mean = {actual_diff.mean():.6f}")

    # Correlation between disparity map and actual difference
    corr = cosine(disparity, (f_col - f_cap))
    print(f"    cos(disparity, actual_diff) = {corr:.4f}")

    # Now do all pairs
    print(f"\n  Cross-class disparity maps (α={alpha}):")
    for c1 in class_names:
        for c2 in class_names:
            if c1 >= c2:
                continue
            v1 = manifolds[c1][li][0].astype(np.float32)
            v2 = manifolds[c2][li][0].astype(np.float32)
            dv = v2 - v1 * np.dot(v1, v2)
            dv = dv / (np.linalg.norm(dv) + 1e-20)
            fl = W_gate_dec @ (v1 - alpha * dv)
            fr = W_gate_dec @ (v1 + alpha * dv)
            disp = fr - fl
            da = np.abs(disp)
            n_sens = int(np.sum(da > da.mean() + 2*da.std()))
            actual = W_gate_dec @ v2 - W_gate_dec @ v1
            c_val = cosine(disp, actual)
            print(f"    {c1}→{c2}: sensitive={n_sens} ({100*n_sens/len(disp):.1f}%)  "
                  f"cos(disp,actual)={c_val:.4f}")

    # ═══════════════════════════════════════════════════════════
    # INV 5: Error Pattern Analysis on Rank-1 Residual
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 5: Error Pattern Analysis — Rank-1 Residual Structure")
    print("=" * 80)

    li = 17
    mlp = engine.layers[li].mlp
    W_gate_dec = mlp.W_gate.decode_cached().astype(np.float64)

    # For capitals class: compute rank-1 gate output vs full gate output
    v1_cap = manifolds['capitals'][li][0].astype(np.float64)
    f_gate = W_gate_dec @ v1_cap  # filter response (rank-1 approximation direction)
    W_r1 = np.outer(f_gate, v1_cap)  # rank-1 weight matrix
    W_residual = W_gate_dec - W_r1  # what's left after removing rank-1

    # For each entity, compute the rank-1 residual error
    print(f"\n  Layer {li} — Rank-1 residual for capitals class:")
    enames = list(classes['capitals'].keys())

    for ename in enames:
        x = all_mlp_inputs['capitals'][ename][li].astype(np.float64)
        full_gate = W_gate_dec @ x
        r1_gate = f_gate * np.dot(v1_cap, x)
        error = full_gate - r1_gate

        # Use ErrorPatternAnalyzer on the error signal
        x_vals = np.arange(len(error), dtype=np.float64)
        try:
            analyzer = ErrorPatternAnalyzer(full_gate, r1_gate, x_vals, name=f"L{li}_{ename}")
            report = analyzer.analyze_all()

            print(f"\n    {ename}:")
            print(f"      |error|/|full|: {np.linalg.norm(error)/np.linalg.norm(full_gate):.4f}")
            print(f"      Error mean: {error.mean():.6f}, std: {error.std():.6f}")

            if hasattr(report, 'patterns') and report.patterns:
                for pat in report.patterns[:3]:
                    print(f"      Pattern: {pat}")
            elif hasattr(report, 'suggestions') and report.suggestions:
                for sug in report.suggestions[:2]:
                    desc = getattr(sug, 'description', str(sug))
                    imp = getattr(sug, 'improvement', None)
                    print(f"      Suggestion: {desc}"
                          + (f" (improvement: {imp:.4f})" if imp else ""))
            else:
                print(f"      (No patterns detected — error may be unstructured)")

        except Exception as e:
            print(f"\n    {ename}: ErrorPatternAnalyzer raised: {e}")
            print(f"      |error|/|full|: {np.linalg.norm(error)/np.linalg.norm(full_gate):.4f}")

    # ═══════════════════════════════════════════════════════════
    # INV 6: Fractal Peel of the Residual Singular Values
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Inv 6: Fractal Peel of Rank-1 Residual")
    print("=" * 80)

    li = 17
    W_gate_dec = engine.layers[li].mlp.W_gate.decode_cached().astype(np.float64)

    # Remove capitals rank-1 component
    v1_cap = manifolds['capitals'][li][0].astype(np.float64)
    f_cap = W_gate_dec @ v1_cap
    W_after_cap = W_gate_dec - np.outer(f_cap, v1_cap)

    # SVD of residual
    _, S_residual, Vt_residual = np.linalg.svd(W_after_cap, full_matrices=False)
    print(f"\n  Layer {li} — After removing capitals rank-1:")
    print(f"    Top 5 SVs: {S_residual[:5].round(2)}")
    print(f"    S[0]/S[1] = {S_residual[0]/S_residual[1]:.3f}")

    rho_res = resfrac_score(S_residual, order=3)
    print(f"    Resfrac(residual SVs): ρ = {rho_res:.4f}")

    # Does the top direction of the residual align with another class?
    v_res_top = Vt_residual[0]
    for cname in classes:
        v1_c = manifolds[cname][li][0]
        c = abs(cosine(v_res_top, v1_c))
        print(f"    cos(residual_v1, {cname}_v1) = {c:.4f}")

    # Now peel iteratively: remove each class's rank-1 and see what remains
    print(f"\n  Iterative peel — remove all class rank-1 components:")
    W_peeled = W_gate_dec.copy()
    for cname in classes:
        v1_c = manifolds[cname][li][0].astype(np.float64)
        f_c = W_peeled @ v1_c
        W_peeled = W_peeled - np.outer(f_c, v1_c)

    _, S_peeled, _ = np.linalg.svd(W_peeled, full_matrices=False)
    rho_peeled = resfrac_score(S_peeled, order=3)
    energy_removed = 1 - np.sum(S_peeled**2) / np.sum(
        engine.layers[li].mlp.W_gate.decode_cached().astype(np.float64).ravel()**2
    )

    print(f"    After removing 3 class rank-1 components:")
    print(f"    Top 5 SVs: {S_peeled[:5].round(2)}")
    print(f"    Resfrac: ρ = {rho_peeled:.4f}")
    print(f"    Energy removed: {energy_removed*100:.2f}%")
    print(f"    Frobenius norm ratio: {np.linalg.norm(S_peeled)/np.linalg.norm(S_residual):.4f}")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print()
    print("  Holographer's Workbench tools applied to transformer weight matrices:")
    print("  - FractalPeeler: SV spectrum structure depth")
    print("  - resfrac_score: structured vs noise at each layer")
    print("  - holographic_refinement: class separation via phase alignment")
    print("  - phase_retrieve_hilbert: coherence of filter responses")
    print("  - ErrorPatternAnalyzer: rank-1 residual patterns")
    print("  - Disparity maps: class-sensitive neuron identification")
    print()
    print("  The tools that built the path now analyze what the path revealed.")
    print()


if __name__ == '__main__':
    main()
