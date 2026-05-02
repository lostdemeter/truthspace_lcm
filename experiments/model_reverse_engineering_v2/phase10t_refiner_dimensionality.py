"""
Phase 10t-Refiner: Stiff-Spring Refiner Dimensionality Study

The Stiff-Spring Refiner (L4-17) adds small orthogonal perturbations to a
spring-dominated residual stream. Key questions:

1. Does each layer add a NEW orthogonal dimension? (dimensionality grows)
2. Or do layers rotate within existing dimensions? (fixed manifold)
3. How do successive attn/FFN additions relate to each other?
4. What's the effective dimensionality of the "refinement space"?
5. Do attention additions and FFN additions occupy the same subspace?

Method: Capture all attn_out and ffn_out vectors for L4-17, stack them,
compute SVD to measure the effective rank of the refinement space.
"""

import torch
import json
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
REFINER_LAYERS = list(range(4, 18))  # L4 through L17

PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the wave function",
    "Once upon a time in a land far away",
    "The derivative of sin(x) is",
    "def fibonacci(n):",
    "The relationship between energy and mass",
    "Yesterday I went to the store and",
    "According to the theory of relativity",
    "She opened the door and saw",
    "The fundamental theorem of calculus states",
    "import torch\nmodel = ",
    "To be or not to be, that is",
    "The mitochondria is the powerhouse of",
    "When the temperature drops below freezing",
    "In the beginning, there was",
]


def angle(a, b):
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def effective_rank(singular_values, threshold=0.99):
    """Number of singular values needed to capture threshold of total energy."""
    total = (singular_values ** 2).sum()
    cumsum = (singular_values ** 2).cumsum(0)
    return (cumsum < threshold * total).sum().item() + 1


def main():
    print("=" * 80)
    print("  PHASE 10t-REFINER: DIMENSIONALITY STUDY (L4-17)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        if pi % 5 == 0:
            print(f"  Prompt {pi}/{len(PROMPTS)}")

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Capture inputs to each refiner layer
        layer_inputs = {}
        hooks = []
        for li in REFINER_LAYERS:
            def make_hook(idx):
                def fn(mod, args):
                    layer_inputs[idx] = args[0].detach().clone()
                return fn
            hooks.append(model.model.layers[li].register_forward_pre_hook(make_hook(li)))

        # Also capture output of last refiner layer
        def out_hook(mod, args, output):
            layer_inputs["L17_out"] = output.detach().clone()
        hooks.append(model.model.layers[17].register_forward_hook(out_hook))

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        # Now dissect each layer and collect addition vectors
        attn_additions = []  # raw attention outputs (before residual)
        ffn_additions = []   # raw FFN outputs (before residual)
        layer_outputs = []   # cumulative hidden states after each layer

        with torch.no_grad():
            h_embed = model.model.embed_tokens(input_ids)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_embed, position_ids)
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            zone_input = layer_inputs[4][0, -1]  # L4 input, last token

            for li in REFINER_LAYERS:
                layer = model.model.layers[li]
                h_in = layer_inputs[li]
                h_in_lt = h_in[0, -1]

                # 6 stages
                s1 = layer.input_layernorm(h_in)
                s2, _ = layer.self_attn(
                    hidden_states=s1,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                s2_lt = s2[0, -1]  # raw attention output
                s3 = h_in + s2
                s3_lt = s3[0, -1]

                s4 = layer.post_attention_layernorm(s3)
                s5 = layer.mlp(s4)
                s5_lt = s5[0, -1]  # raw FFN output
                s6 = s3 + s5
                s6_lt = s6[0, -1]

                attn_additions.append(s2_lt)
                ffn_additions.append(s5_lt)
                layer_outputs.append(s6_lt)

        # Stack addition vectors into matrices
        attn_matrix = torch.stack(attn_additions)  # [14, d_model]
        ffn_matrix = torch.stack(ffn_additions)     # [14, d_model]
        all_additions = torch.cat([attn_matrix, ffn_matrix], dim=0)  # [28, d_model]

        r = {"prompt_idx": pi}

        # ============================================================
        # TEST 1: SVD of addition vectors — effective dimensionality
        # ============================================================
        for name, matrix in [("attn", attn_matrix), ("ffn", ffn_matrix), ("all", all_additions)]:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            S_np = S.numpy()

            # Effective rank at various thresholds
            for thresh in [0.90, 0.95, 0.99]:
                r[f"{name}_rank_{int(thresh*100)}"] = effective_rank(S, thresh)

            # Singular value decay
            r[f"{name}_sv_top1_frac"] = (S_np[0]**2 / (S_np**2).sum()).item()
            r[f"{name}_sv_top3_frac"] = ((S_np[:3]**2).sum() / (S_np**2).sum()).item()
            r[f"{name}_sv_top5_frac"] = ((S_np[:min(5, len(S_np))]**2).sum() / (S_np**2).sum()).item()
            r[f"{name}_num_vectors"] = matrix.shape[0]

        # ============================================================
        # TEST 2: Pairwise angles between successive additions
        # ============================================================
        attn_pairwise = []
        ffn_pairwise = []
        cross_pairwise = []  # attn[i] vs ffn[i] at same layer

        for i in range(len(REFINER_LAYERS)):
            if i > 0:
                attn_pairwise.append(angle(attn_additions[i], attn_additions[i-1]))
                ffn_pairwise.append(angle(ffn_additions[i], ffn_additions[i-1]))
            cross_pairwise.append(angle(attn_additions[i], ffn_additions[i]))

        r["attn_successive_angles"] = attn_pairwise
        r["ffn_successive_angles"] = ffn_pairwise
        r["cross_same_layer_angles"] = cross_pairwise
        r["mean_attn_successive"] = np.mean(attn_pairwise) if attn_pairwise else 0
        r["mean_ffn_successive"] = np.mean(ffn_pairwise) if ffn_pairwise else 0
        r["mean_cross_same_layer"] = np.mean(cross_pairwise) if cross_pairwise else 0

        # ============================================================
        # TEST 3: Cumulative angle from zone input
        # ============================================================
        cumul_angles = []
        for i, lo in enumerate(layer_outputs):
            cumul_angles.append(angle(zone_input, lo))
        r["cumul_angles"] = cumul_angles

        # ============================================================
        # TEST 4: Are all additions orthogonal to the residual stream?
        # ============================================================
        attn_vs_residual = []
        ffn_vs_residual = []
        for i in range(len(REFINER_LAYERS)):
            # residual at this point is the layer input
            h_in_lt = layer_inputs[REFINER_LAYERS[i]][0, -1]
            attn_vs_residual.append(abs(cos_sim(attn_additions[i], h_in_lt)))
            ffn_vs_residual.append(abs(cos_sim(ffn_additions[i], h_in_lt)))

        r["mean_attn_vs_residual"] = np.mean(attn_vs_residual)
        r["mean_ffn_vs_residual"] = np.mean(ffn_vs_residual)
        r["attn_vs_residual"] = attn_vs_residual
        r["ffn_vs_residual"] = ffn_vs_residual

        # ============================================================
        # TEST 5: Do attn and FFN additions share a subspace?
        # ============================================================
        # Compute principal angles between attn subspace and ffn subspace
        U_attn, S_attn, _ = torch.linalg.svd(attn_matrix, full_matrices=False)
        U_ffn, S_ffn, _ = torch.linalg.svd(ffn_matrix, full_matrices=False)

        # Use top-k directions
        for k in [3, 5, 7]:
            if k > min(U_attn.shape[1], U_ffn.shape[1]):
                continue
            # Principal angles between subspaces
            M = U_attn[:, :k].T @ U_ffn[:, :k]
            principal_cos = torch.linalg.svdvals(M)
            principal_angles = torch.acos(principal_cos.clamp(-1, 1))
            r[f"subspace_overlap_k{k}"] = principal_cos.mean().item()
            r[f"subspace_min_angle_k{k}"] = math.degrees(principal_angles.min().item())
            r[f"subspace_mean_angle_k{k}"] = math.degrees(principal_angles.mean().item())

        all_results.append(r)

    # ================================================================
    # ANALYSIS
    # ================================================================

    print("\n" + "=" * 60)
    print("  EFFECTIVE DIMENSIONALITY")
    print("=" * 60)

    for name, label in [("attn", "Attention additions"), ("ffn", "FFN additions"), ("all", "All additions")]:
        n_vec = all_results[0][f"{name}_num_vectors"]
        r90 = np.mean([r[f"{name}_rank_90"] for r in all_results])
        r95 = np.mean([r[f"{name}_rank_95"] for r in all_results])
        r99 = np.mean([r[f"{name}_rank_99"] for r in all_results])
        sv1 = np.mean([r[f"{name}_sv_top1_frac"] for r in all_results])
        sv3 = np.mean([r[f"{name}_sv_top3_frac"] for r in all_results])
        sv5 = np.mean([r[f"{name}_sv_top5_frac"] for r in all_results])

        print(f"\n  {label} ({n_vec} vectors):")
        print(f"    Rank(90%): {r90:.1f}, Rank(95%): {r95:.1f}, Rank(99%): {r99:.1f}")
        print(f"    Top-1 SV: {sv1*100:.1f}%, Top-3: {sv3*100:.1f}%, Top-5: {sv5*100:.1f}%")

    print("\n" + "=" * 60)
    print("  SUCCESSIVE ADDITION ANGLES")
    print("=" * 60)

    mean_attn_succ = np.mean([r["mean_attn_successive"] for r in all_results])
    mean_ffn_succ = np.mean([r["mean_ffn_successive"] for r in all_results])
    mean_cross = np.mean([r["mean_cross_same_layer"] for r in all_results])

    print(f"\n  Mean angle between successive attention additions: {mean_attn_succ:.1f}°")
    print(f"  Mean angle between successive FFN additions: {mean_ffn_succ:.1f}°")
    print(f"  Mean angle between attn and FFN at same layer: {mean_cross:.1f}°")

    if mean_attn_succ > 70:
        print(f"  → Successive attention additions are NEARLY ORTHOGONAL")
    if mean_cross > 70:
        print(f"  → Attn and FFN at same layer are NEARLY ORTHOGONAL")

    # Per-layer detail
    print(f"\n  Per-layer successive angles (mean across prompts):")
    print(f"  {'Layer':>5} | {'Attn→Attn':>10} | {'FFN→FFN':>10} | {'Attn↔FFN':>10}")
    print("  " + "-" * 40)
    for i, li in enumerate(REFINER_LAYERS):
        attn_s = np.mean([r["attn_successive_angles"][i-1] for r in all_results]) if i > 0 else float('nan')
        ffn_s = np.mean([r["ffn_successive_angles"][i-1] for r in all_results]) if i > 0 else float('nan')
        cross = np.mean([r["cross_same_layer_angles"][i] for r in all_results])
        if i > 0:
            print(f"  L{li:>3} | {attn_s:>9.1f}° | {ffn_s:>9.1f}° | {cross:>9.1f}°")
        else:
            print(f"  L{li:>3} | {'---':>10} | {'---':>10} | {cross:>9.1f}°")

    print("\n" + "=" * 60)
    print("  ADDITIONS vs RESIDUAL STREAM")
    print("=" * 60)

    mean_a_res = np.mean([r["mean_attn_vs_residual"] for r in all_results])
    mean_f_res = np.mean([r["mean_ffn_vs_residual"] for r in all_results])
    print(f"\n  Mean |cos(attn_addition, residual)|: {mean_a_res:.4f}")
    print(f"  Mean |cos(ffn_addition, residual)|:  {mean_f_res:.4f}")

    print(f"\n  Per-layer |cos| with residual (mean across prompts):")
    print(f"  {'Layer':>5} | {'|cos(a,h)|':>11} | {'|cos(f,h)|':>11}")
    print("  " + "-" * 30)
    for i, li in enumerate(REFINER_LAYERS):
        a = np.mean([r["attn_vs_residual"][i] for r in all_results])
        f = np.mean([r["ffn_vs_residual"][i] for r in all_results])
        print(f"  L{li:>3} | {a:>11.4f} | {f:>11.4f}")

    print("\n" + "=" * 60)
    print("  CUMULATIVE ANGLE FROM ZONE INPUT")
    print("=" * 60)

    print(f"\n  {'Layer':>5} | {'Cumul°':>8}")
    print("  " + "-" * 16)
    for i, li in enumerate(REFINER_LAYERS):
        c = np.mean([r["cumul_angles"][i] for r in all_results])
        print(f"  L{li:>3} | {c:>7.1f}°")

    print("\n" + "=" * 60)
    print("  ATTN vs FFN SUBSPACE OVERLAP")
    print("=" * 60)

    for k in [3, 5, 7]:
        key = f"subspace_overlap_k{k}"
        if key in all_results[0]:
            overlap = np.mean([r[key] for r in all_results])
            min_ang = np.mean([r[f"subspace_min_angle_k{k}"] for r in all_results])
            mean_ang = np.mean([r[f"subspace_mean_angle_k{k}"] for r in all_results])
            print(f"\n  Top-{k} subspace principal angles:")
            print(f"    Mean cos overlap: {overlap:.4f}")
            print(f"    Min principal angle: {min_ang:.1f}°")
            print(f"    Mean principal angle: {mean_ang:.1f}°")

    # ================================================================
    # SAVE
    # ================================================================
    save_data = {
        "per_prompt": [{k: v for k, v in r.items()
                       if not isinstance(v, list) or len(v) < 30} for r in all_results],
    }

    out_path = "results/phase10t_refiner_dimensionality.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    print("\n" + "=" * 80)
    print("  PHASE 10t-REFINER COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
