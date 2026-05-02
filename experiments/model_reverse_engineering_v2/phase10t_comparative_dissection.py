"""
Phase 10t-comparative: Deep Dissection Across All Zones

Run the same 6-stage simple machine trace on representative layers from
each zone to discover if the Orthogonal Tripod pattern repeats or if
each machine has a unique composition.

Layers sampled:
  L0  — φ-Projector (known: Orthogonal Tripod)
  L2  — φ-Corrector (negative zero correction)
  L5  — Stabilizer early (deep CONTRACT, FFN-critical)
  L8  — Stabilizer late (transition to mixed)
  L12 — Equilibrium Core (nearly passive, spring-dominant?)
  L16 — Equilibrium Core (peak PRESERVE)
  L20 — Pre-Targeter (re-contracting)
  L24 — Pre-Targeter (near exit)
  L27 — φ-Filter (EXPAND/CONTRACT targeting)
"""

import torch
import json
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)

SAMPLE_LAYERS = [0, 2, 5, 8, 12, 16, 20, 24, 27]

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


def project_onto_frac(v, direction):
    """Fraction of ||v|| that lies along direction."""
    d_norm = direction / direction.norm()
    proj = (v @ d_norm)
    return abs(proj.item()) / max(v.norm().item(), 1e-12)


def project_perp_frac(v, direction):
    """Fraction of ||v|| perpendicular to direction."""
    d_norm = direction / direction.norm()
    proj_along = (v @ d_norm) * d_norm
    perp = v - proj_along
    return perp.norm().item() / max(v.norm().item(), 1e-12)


def main():
    print("=" * 80)
    print("  PHASE 10t-COMPARATIVE: DEEP DISSECTION ACROSS ALL ZONES")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # We need to capture hidden states at the INPUT of each sample layer.
    # Strategy: run the full model with hooks that capture input to each sample layer,
    # then manually run each sample layer's 6 stages.

    all_layer_data = {li: [] for li in SAMPLE_LAYERS}

    for pi, prompt in enumerate(PROMPTS):
        if pi % 5 == 0:
            print(f"  Prompt {pi}/{len(PROMPTS)}")

        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Capture hidden states at the input of each sampled layer using hooks
        layer_inputs = {}
        hooks = []

        def make_pre_hook(layer_idx):
            def hook_fn(module, args):
                # args[0] is hidden_states
                layer_inputs[layer_idx] = args[0].detach().clone()
            return hook_fn

        for li in SAMPLE_LAYERS:
            h = model.model.layers[li].register_forward_pre_hook(make_pre_hook(li))
            hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=False)

        for h in hooks:
            h.remove()

        # Now dissect each sampled layer
        with torch.no_grad():
            # Get position embeddings (needed for attention)
            h_embed = model.model.embed_tokens(input_ids)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_embed, position_ids)

            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            for li in SAMPLE_LAYERS:
                layer = model.model.layers[li]
                h_in = layer_inputs[li]  # [1, seq, d_model]
                h_in_lt = h_in[0, -1]   # last token

                # Stage 1: Damper 1
                s1 = layer.input_layernorm(h_in)
                s1_lt = s1[0, -1]

                # Stage 2: Lever (attention)
                s2, _ = layer.self_attn(
                    hidden_states=s1,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                s2_lt = s2[0, -1]

                # Stage 3: Spring 1
                s3 = h_in + s2
                s3_lt = s3[0, -1]

                # Stage 4: Damper 2
                s4 = layer.post_attention_layernorm(s3)
                s4_lt = s4[0, -1]

                # Stage 5: Wedge (FFN)
                s5 = layer.mlp(s4)
                s5_lt = s5[0, -1]

                # Stage 6: Spring 2
                s6 = s3 + s5
                s6_lt = s6[0, -1]

                # ---- Measurements ----
                d = {}

                # Norms
                d["input_norm"] = h_in_lt.norm().item()
                d["damper1_norm"] = s1_lt.norm().item()
                d["lever_norm"] = s2_lt.norm().item()
                d["spring1_norm"] = s3_lt.norm().item()
                d["damper2_norm"] = s4_lt.norm().item()
                d["wedge_norm"] = s5_lt.norm().item()
                d["output_norm"] = s6_lt.norm().item()

                # Norm ratios
                d["damper1_ratio"] = d["damper1_norm"] / max(d["input_norm"], 1e-12)
                d["lever_ratio"] = d["lever_norm"] / max(d["damper1_norm"], 1e-12)
                d["spring1_ratio"] = d["spring1_norm"] / max(d["input_norm"], 1e-12)
                d["damper2_ratio"] = d["damper2_norm"] / max(d["spring1_norm"], 1e-12)
                d["wedge_ratio"] = d["wedge_norm"] / max(d["damper2_norm"], 1e-12)
                d["spring2_ratio"] = d["output_norm"] / max(d["spring1_norm"], 1e-12)

                # Per-stage rotations
                d["damper1_rot"] = angle(h_in_lt, s1_lt)
                d["lever_rot"] = angle(s1_lt, s2_lt)
                d["spring1_cumul"] = angle(h_in_lt, s3_lt)  # cumul from input
                d["damper2_rot"] = angle(s3_lt, s4_lt)
                d["wedge_rot"] = angle(s4_lt, s5_lt)
                d["spring2_from_mid"] = angle(s3_lt, s6_lt)  # FFN's effect on h_mid
                d["total_rot"] = angle(h_in_lt, s6_lt)

                # Rotation budget
                d["lever_rot_pct"] = d["spring1_cumul"] / max(d["total_rot"], 1e-6) * 100
                d["wedge_rot_contribution"] = d["total_rot"] - d["spring1_cumul"]

                # Energy budget
                total_e = d["input_norm"] + d["lever_norm"] + d["wedge_norm"]
                d["energy_input"] = d["input_norm"] / total_e * 100
                d["energy_lever"] = d["lever_norm"] / total_e * 100
                d["energy_wedge"] = d["wedge_norm"] / total_e * 100

                # Cross-correlations (the key test)
                d["cos_input_attn"] = cos_sim(h_in_lt, s2_lt)
                d["cos_input_ffn"] = cos_sim(h_in_lt, s5_lt)
                d["cos_attn_ffn"] = cos_sim(s2_lt, s5_lt)
                d["cos_ffn_hmid"] = cos_sim(s5_lt, s3_lt)

                # Projection decomposition
                d["ffn_along_input"] = project_onto_frac(s5_lt, h_in_lt)
                d["ffn_perp_input"] = project_perp_frac(s5_lt, h_in_lt)
                d["ffn_along_hmid"] = project_onto_frac(s5_lt, s3_lt)
                d["ffn_perp_hmid"] = project_perp_frac(s5_lt, s3_lt)

                # Spring stiffness (how much does residual resist perturbation?)
                # k = ||h_in|| / (||h_in|| + ||sublayer_out||)
                d["spring1_k"] = d["input_norm"] / (d["input_norm"] + d["lever_norm"])
                d["spring2_k"] = d["spring1_norm"] / (d["spring1_norm"] + d["wedge_norm"])

                all_layer_data[li].append(d)

    # ================================================================
    # SUMMARY TABLES
    # ================================================================

    print("\n" + "=" * 80)
    print("  ROTATION BUDGET BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'Total°':>7} | {'Lever°':>7} | {'Spr1°':>7} | {'Wedge°':>7} | {'Spr2°':>7} | {'LevPct':>7} | {'WedgeΔ':>7}")
    print("  " + "-" * 72)

    layer_summaries = {}
    for li in SAMPLE_LAYERS:
        data = all_layer_data[li]
        s = {}
        for key in data[0].keys():
            vals = [d[key] for d in data]
            s[key] = np.mean(vals)
            s[f"{key}_std"] = np.std(vals)
        layer_summaries[li] = s

        print(f"  L{li:>3} | {s['total_rot']:>7.1f} | {s['lever_rot']:>7.1f} | "
              f"{s['spring1_cumul']:>7.1f} | {s['wedge_rot']:>7.1f} | "
              f"{s['spring2_from_mid']:>7.1f} | {s['lever_rot_pct']:>6.1f}% | "
              f"{s['wedge_rot_contribution']:>+6.1f}°")

    print("\n" + "=" * 80)
    print("  CROSS-CORRELATIONS BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'cos(in,a)':>9} | {'cos(in,f)':>9} | {'cos(a,f)':>9} | {'cos(f,hm)':>10} | Pattern")
    print("  " + "-" * 72)

    for li in SAMPLE_LAYERS:
        s = layer_summaries[li]
        ci = s["cos_input_attn"]
        cf = s["cos_input_ffn"]
        af = s["cos_attn_ffn"]
        fh = s["cos_ffn_hmid"]

        # Classify pattern
        if abs(ci) < 0.2 and abs(cf) < 0.2:
            pattern = "ORTHOGONAL TRIPOD"
        elif abs(fh) > 0.7:
            pattern = "ENERGY BOOSTER"
        elif abs(fh) < 0.3 and s["spring2_k"] > 0.8:
            pattern = "DIRECTION REFINER"
        elif s["spring1_k"] > 0.8 and s["spring2_k"] > 0.8:
            pattern = "EQUILIBRIUM MAINTAINER"
        else:
            pattern = f"MIXED (fh={fh:.2f}, k1={s['spring1_k']:.2f}, k2={s['spring2_k']:.2f})"

        print(f"  L{li:>3} | {ci:>+9.4f} | {cf:>+9.4f} | {af:>+9.4f} | {fh:>+10.4f} | {pattern}")

    print("\n" + "=" * 80)
    print("  ENERGY BUDGET BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'Input%':>7} | {'Lever%':>7} | {'Wedge%':>7} | {'Spr1_k':>7} | {'Spr2_k':>7}")
    print("  " + "-" * 52)

    for li in SAMPLE_LAYERS:
        s = layer_summaries[li]
        print(f"  L{li:>3} | {s['energy_input']:>6.1f}% | {s['energy_lever']:>6.1f}% | "
              f"{s['energy_wedge']:>6.1f}% | {s['spring1_k']:>7.4f} | {s['spring2_k']:>7.4f}")

    print("\n" + "=" * 80)
    print("  NORM FLOW BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'h_in':>8} | {'Damp1':>8} | {'Lever':>8} | {'h_mid':>8} | {'Damp2':>8} | {'Wedge':>8} | {'h_out':>8}")
    print("  " + "-" * 72)

    for li in SAMPLE_LAYERS:
        s = layer_summaries[li]
        print(f"  L{li:>3} | {s['input_norm']:>8.2f} | {s['damper1_norm']:>8.2f} | "
              f"{s['lever_norm']:>8.2f} | {s['spring1_norm']:>8.2f} | "
              f"{s['damper2_norm']:>8.2f} | {s['wedge_norm']:>8.2f} | {s['output_norm']:>8.2f}")

    print("\n" + "=" * 80)
    print("  FFN PROJECTION DECOMPOSITION BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'Along_in':>9} | {'Perp_in':>9} | {'Along_hm':>9} | {'Perp_hm':>9}")
    print("  " + "-" * 48)

    for li in SAMPLE_LAYERS:
        s = layer_summaries[li]
        print(f"  L{li:>3} | {s['ffn_along_input']*100:>8.1f}% | {s['ffn_perp_input']*100:>8.1f}% | "
              f"{s['ffn_along_hmid']*100:>8.1f}% | {s['ffn_perp_hmid']*100:>8.1f}%")

    print("\n" + "=" * 80)
    print("  DAMPER RATIOS BY LAYER")
    print("=" * 80)
    print(f"\n  {'Layer':>5} | {'Damp1_ratio':>11} | {'Damp2_ratio':>11} | {'Damp1_rot':>10} | {'Damp2_rot':>10}")
    print("  " + "-" * 52)

    for li in SAMPLE_LAYERS:
        s = layer_summaries[li]
        print(f"  L{li:>3} | {s['damper1_ratio']:>11.4f} | {s['damper2_ratio']:>11.4f} | "
              f"{s['damper1_rot']:>9.1f}° | {s['damper2_rot']:>9.1f}°")

    # ================================================================
    # SAVE
    # ================================================================
    save_data = {
        "layer_summaries": {str(li): {k: float(v) for k, v in s.items()} for li, s in layer_summaries.items()},
    }

    out_path = "results/phase10t_comparative.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {out_path}")

    print("\n" + "=" * 80)
    print("  PHASE 10t-COMPARATIVE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
