"""
Phase 10t: φ-Projector (L0) Deep Dissection

Trace all 6 simple machine operations within L0 and measure geometric
quantities at each stage. Goal: understand how 4 types of simple machine
(damper, lever, spring, wedge) compose into the compound 81° rotation.

The 6 stages of a Qwen2 decoder layer:
  1. Damper 1  (input_layernorm / RMSNorm)
  2. Lever     (self_attn / multi-head attention)
  3. Spring 1  (residual addition: h_mid = h_in + attn_out)
  4. Damper 2  (post_attention_layernorm / RMSNorm)
  5. Wedge     (mlp / gated FFN with SiLU)
  6. Spring 2  (residual addition: h_out = h_mid + ffn_out)
"""

import torch
import json
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)

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


def angle_between(a, b):
    """Angle in degrees between two vectors (last-token only)."""
    if a.dim() == 3:
        a, b = a[0, -1], b[0, -1]
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def norm_ratio(a, b):
    """||b|| / ||a|| for last token."""
    if a.dim() == 3:
        a, b = a[0, -1], b[0, -1]
    return b.norm().item() / max(a.norm().item(), 1e-12)


def cosine_sim(a, b):
    """Cosine similarity for last token."""
    if a.dim() == 3:
        a, b = a[0, -1], b[0, -1]
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def classify_gate(activations):
    """Classify gate channels into 4 states using ±log(φ) thresholds."""
    counts = {"EXPAND": 0, "PRESERVE+": 0, "PRESERVE-": 0, "CONTRACT": 0}
    for v in activations.flatten().tolist():
        if v > LOG_PHI:
            counts["EXPAND"] += 1
        elif v > 0:
            counts["PRESERVE+"] += 1
        elif v > -LOG_PHI:
            counts["PRESERVE-"] += 1
        else:
            counts["CONTRACT"] += 1
    total = sum(counts.values())
    return {k: (v, v / total * 100) for k, v in counts.items()}


def main():
    print("=" * 80)
    print("  PHASE 10t: φ-PROJECTOR (L0) DEEP DISSECTION")
    print("  Tracing all 6 simple machine operations")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    layer0 = model.model.layers[0]

    # ================================================================
    # PART 1: TRACE ALL 6 STAGES
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 1: STAGE-BY-STAGE TRACE THROUGH L0")
    print("=" * 60)

    all_traces = []

    for pi, prompt in enumerate(PROMPTS):
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Get embedding output (= L0 input)
            h_in = model.model.embed_tokens(input_ids)

            # Need position embeddings for attention
            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_in, position_ids)

            # Build causal mask
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

            # === Stage 0: Input (h_in) ===
            s0_input = h_in.clone()

            # === Stage 1: Damper 1 (input_layernorm) ===
            s1_damper1 = layer0.input_layernorm(h_in)

            # === Stage 2: Lever (self_attn) ===
            s2_lever_out, _ = layer0.self_attn(
                hidden_states=s1_damper1,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            # === Stage 3: Spring 1 (residual add) ===
            s3_spring1 = s0_input + s2_lever_out

            # === Stage 4: Damper 2 (post_attention_layernorm) ===
            s4_damper2 = layer0.post_attention_layernorm(s3_spring1)

            # === Stage 5: Wedge (MLP/FFN) ===
            # Also capture gate activations for analysis
            gate_act = layer0.mlp.act_fn(layer0.mlp.gate_proj(s4_damper2))
            up_act = layer0.mlp.up_proj(s4_damper2)
            s5_wedge = layer0.mlp.down_proj(gate_act * up_act)

            # === Stage 6: Spring 2 (residual add) ===
            s6_spring2 = s3_spring1 + s5_wedge

        # Collect all states for measurement
        stages = [
            ("0_input", s0_input),
            ("1_damper1", s1_damper1),
            ("2_lever", s2_lever_out),    # raw attention output (before residual)
            ("3_spring1", s3_spring1),     # h_in + attn_out
            ("4_damper2", s4_damper2),
            ("5_wedge", s5_wedge),         # raw FFN output (before residual)
            ("6_spring2", s6_spring2),     # h_mid + ffn_out = final output
        ]

        trace = {"prompt_idx": pi, "seq_len": seq_len}

        # Measure properties at each stage
        for name, state in stages:
            lt = state[0, -1]  # last token
            trace[f"{name}_norm"] = lt.norm().item()

        # Angle from input at each stage
        for name, state in stages[1:]:
            trace[f"{name}_angle_from_input"] = angle_between(s0_input, state)
            trace[f"{name}_cos_with_input"] = cosine_sim(s0_input, state)

        # Sequential angles (stage N vs stage N-1 input)
        # Damper1: how much does LN rotate the input?
        trace["damper1_rotation"] = angle_between(s0_input, s1_damper1)
        trace["damper1_contraction"] = norm_ratio(s0_input, s1_damper1)

        # Lever: how much does attention rotate the LN'd input?
        trace["lever_rotation"] = angle_between(s1_damper1, s2_lever_out)
        trace["lever_magnitude"] = norm_ratio(s1_damper1, s2_lever_out)

        # Spring1: angle between h_in and h_in+attn_out
        trace["spring1_rotation"] = angle_between(s0_input, s3_spring1)
        trace["spring1_dilution"] = norm_ratio(s0_input, s3_spring1)
        # How much of the lever's rotation survives the spring?
        trace["lever_survived_spring1"] = angle_between(s0_input, s3_spring1)

        # Damper2: how much does LN rotate the post-attention state?
        trace["damper2_rotation"] = angle_between(s3_spring1, s4_damper2)
        trace["damper2_contraction"] = norm_ratio(s3_spring1, s4_damper2)

        # Wedge: how much does FFN rotate the LN'd mid-state?
        trace["wedge_rotation"] = angle_between(s4_damper2, s5_wedge)
        trace["wedge_magnitude"] = norm_ratio(s4_damper2, s5_wedge)

        # Spring2: angle from h_mid to h_mid+ffn_out
        trace["spring2_rotation"] = angle_between(s3_spring1, s6_spring2)
        trace["spring2_dilution"] = norm_ratio(s3_spring1, s6_spring2)

        # Total rotation (input to output)
        trace["total_rotation"] = angle_between(s0_input, s6_spring2)

        # Gate state classification for L0's wedge
        gate_last = gate_act[0, -1]  # [intermediate_size]
        # Classify based on pre-activation (gate_proj output before SiLU)
        gate_pre = layer0.mlp.gate_proj(s4_damper2)[0, -1]
        trace["gate_states"] = classify_gate(gate_pre)

        # Energy decomposition: how much of the output is from lever vs wedge?
        # s6 = s0 + s2 + s5 (approximately, since s4 = LN(s0+s2))
        lever_energy = s2_lever_out[0, -1].norm().item()
        wedge_energy = s5_wedge[0, -1].norm().item()
        input_energy = s0_input[0, -1].norm().item()
        total_energy = lever_energy + wedge_energy + input_energy
        trace["energy_lever_frac"] = lever_energy / total_energy
        trace["energy_wedge_frac"] = wedge_energy / total_energy
        trace["energy_input_frac"] = input_energy / total_energy

        # Orthogonality: are lever output and wedge output orthogonal?
        trace["lever_wedge_cos"] = cosine_sim(s2_lever_out, s5_wedge)
        trace["lever_input_cos"] = cosine_sim(s0_input, s2_lever_out)
        trace["wedge_input_cos"] = cosine_sim(s0_input, s5_wedge)

        all_traces.append(trace)

    # Print summary table
    print(f"\n  {'Prompt':>3} | {'Total°':>7} | {'Damp1°':>7} | {'Lever°':>7} | {'Spr1°':>7} | {'Damp2°':>7} | {'Wedge°':>7} | {'Spr2°':>7}")
    print("  " + "-" * 75)
    for t in all_traces:
        print(f"  {t['prompt_idx']:>3} | {t['total_rotation']:>7.2f} | "
              f"{t['damper1_rotation']:>7.2f} | {t['lever_rotation']:>7.2f} | "
              f"{t['spring1_rotation']:>7.2f} | {t['damper2_rotation']:>7.2f} | "
              f"{t['wedge_rotation']:>7.2f} | {t['spring2_rotation']:>7.2f}")

    # Compute means
    keys = ["total_rotation", "damper1_rotation", "lever_rotation",
            "spring1_rotation", "damper2_rotation", "wedge_rotation", "spring2_rotation"]
    means = {k: np.mean([t[k] for t in all_traces]) for k in keys}
    stds = {k: np.std([t[k] for t in all_traces]) for k in keys}

    print("  " + "-" * 75)
    print(f"  {'MEAN':>3} | {means['total_rotation']:>7.2f} | "
          f"{means['damper1_rotation']:>7.2f} | {means['lever_rotation']:>7.2f} | "
          f"{means['spring1_rotation']:>7.2f} | {means['damper2_rotation']:>7.2f} | "
          f"{means['wedge_rotation']:>7.2f} | {means['spring2_rotation']:>7.2f}")
    print(f"  {'±':>3} | {stds['total_rotation']:>7.2f} | "
          f"{stds['damper1_rotation']:>7.2f} | {stds['lever_rotation']:>7.2f} | "
          f"{stds['spring1_rotation']:>7.2f} | {stds['damper2_rotation']:>7.2f} | "
          f"{stds['wedge_rotation']:>7.2f} | {stds['spring2_rotation']:>7.2f}")

    # ================================================================
    # PART 2: NORM FLOW (how magnitude changes through the pipeline)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 2: NORM FLOW THROUGH L0")
    print("=" * 60)

    print(f"\n  {'Prompt':>3} | {'Input':>8} | {'Damp1':>8} | {'Lever':>8} | {'Spr1':>8} | {'Damp2':>8} | {'Wedge':>8} | {'Output':>8}")
    print("  " + "-" * 72)
    for t in all_traces:
        print(f"  {t['prompt_idx']:>3} | {t['0_input_norm']:>8.2f} | "
              f"{t['1_damper1_norm']:>8.2f} | {t['2_lever_norm']:>8.2f} | "
              f"{t['3_spring1_norm']:>8.2f} | {t['4_damper2_norm']:>8.2f} | "
              f"{t['5_wedge_norm']:>8.2f} | {t['6_spring2_norm']:>8.2f}")

    norm_keys = ["0_input_norm", "1_damper1_norm", "2_lever_norm",
                 "3_spring1_norm", "4_damper2_norm", "5_wedge_norm", "6_spring2_norm"]
    norm_means = {k: np.mean([t[k] for t in all_traces]) for k in norm_keys}
    print("  " + "-" * 72)
    print(f"  {'MEAN':>3} | {norm_means['0_input_norm']:>8.2f} | "
          f"{norm_means['1_damper1_norm']:>8.2f} | {norm_means['2_lever_norm']:>8.2f} | "
          f"{norm_means['3_spring1_norm']:>8.2f} | {norm_means['4_damper2_norm']:>8.2f} | "
          f"{norm_means['5_wedge_norm']:>8.2f} | {norm_means['6_spring2_norm']:>8.2f}")

    # ================================================================
    # PART 3: DIRECTION ANALYSIS — What does each machine do to direction?
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 3: DIRECTION ANALYSIS")
    print("=" * 60)

    # Mean contractions
    d1_contract = np.mean([t["damper1_contraction"] for t in all_traces])
    lever_mag = np.mean([t["lever_magnitude"] for t in all_traces])
    s1_dilute = np.mean([t["spring1_dilution"] for t in all_traces])
    d2_contract = np.mean([t["damper2_contraction"] for t in all_traces])
    wedge_mag = np.mean([t["wedge_magnitude"] for t in all_traces])
    s2_dilute = np.mean([t["spring2_dilution"] for t in all_traces])

    print(f"\n  Stage-by-stage norm ratios (||out|| / ||in||):")
    print(f"    Damper 1 contraction:  {d1_contract:.4f}  (LN compresses by {(1-d1_contract)*100:.1f}%)")
    print(f"    Lever magnitude:       {lever_mag:.4f}  (attention output vs LN'd input)")
    print(f"    Spring 1 dilution:     {s1_dilute:.4f}  (residual grows by {(s1_dilute-1)*100:.1f}%)")
    print(f"    Damper 2 contraction:  {d2_contract:.4f}  (LN compresses by {(1-d2_contract)*100:.1f}%)")
    print(f"    Wedge magnitude:       {wedge_mag:.4f}  (FFN output vs LN'd mid-state)")
    print(f"    Spring 2 dilution:     {s2_dilute:.4f}  (residual grows by {(s2_dilute-1)*100:.1f}%)")

    # Orthogonality analysis
    lw_cos = np.mean([t["lever_wedge_cos"] for t in all_traces])
    li_cos = np.mean([t["lever_input_cos"] for t in all_traces])
    wi_cos = np.mean([t["wedge_input_cos"] for t in all_traces])

    print(f"\n  Cross-correlations (cosine similarity):")
    print(f"    cos(lever_out, wedge_out):  {lw_cos:+.4f}")
    print(f"    cos(input, lever_out):      {li_cos:+.4f}")
    print(f"    cos(input, wedge_out):      {wi_cos:+.4f}")

    if abs(lw_cos) < 0.3:
        print("    → Lever and Wedge outputs are NEARLY ORTHOGONAL")
    elif abs(lw_cos) < 0.6:
        print("    → Lever and Wedge outputs are MODERATELY correlated")
    else:
        print("    → Lever and Wedge outputs are STRONGLY correlated")

    # Energy decomposition
    e_lever = np.mean([t["energy_lever_frac"] for t in all_traces])
    e_wedge = np.mean([t["energy_wedge_frac"] for t in all_traces])
    e_input = np.mean([t["energy_input_frac"] for t in all_traces])

    print(f"\n  Energy budget (norm fractions of output components):")
    print(f"    Input (embedding):  {e_input*100:.1f}%")
    print(f"    Lever (attention):  {e_lever*100:.1f}%")
    print(f"    Wedge (FFN):        {e_wedge*100:.1f}%")

    # ================================================================
    # PART 4: GATE STATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 4: L0 GATE STATE (WEDGE CHARACTER)")
    print("=" * 60)

    # Aggregate gate states
    agg_states = {"EXPAND": [], "PRESERVE+": [], "PRESERVE-": [], "CONTRACT": []}
    for t in all_traces:
        for state, (count, pct) in t["gate_states"].items():
            agg_states[state].append(pct)

    for state in ["EXPAND", "PRESERVE+", "PRESERVE-", "CONTRACT"]:
        vals = agg_states[state]
        print(f"  {state:>12}: {np.mean(vals):6.2f}% ± {np.std(vals):5.2f}%")

    # ================================================================
    # PART 5: ROTATION BUDGET — How does 81° break down?
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 5: ROTATION BUDGET ANALYSIS")
    print("=" * 60)

    # The total rotation from input to output is ~81°.
    # But rotations don't simply add because they're in different subspaces.
    # Let's trace the cumulative angle from input at each stage.
    print(f"\n  Cumulative angle from input at each stage:")
    cum_keys = ["1_damper1_angle_from_input", "3_spring1_angle_from_input",
                "6_spring2_angle_from_input"]
    cum_labels = ["After Damper1", "After Spring1 (attn+resid)", "After Spring2 (final)"]

    # Also measure: angle from input after each major operation
    for label, key in zip(cum_labels, cum_keys):
        vals = [t[key] for t in all_traces]
        print(f"    {label:>30}: {np.mean(vals):6.2f}° ± {np.std(vals):.2f}°")

    # The key question: where does the rotation happen?
    # After Damper1: how much has LN already rotated?
    # After Spring1: how much total rotation from input → attention + residual?
    # After Spring2: final total (should be ~81°)
    d1_rot = np.mean([t["damper1_rotation"] for t in all_traces])
    s1_rot = np.mean([t["spring1_rotation"] for t in all_traces])
    s2_contribution = means["total_rotation"] - s1_rot

    print(f"\n  Rotation attribution:")
    print(f"    Damper1 (LN rotation):    {d1_rot:.2f}°")
    print(f"    Spring1 (after lever):    {s1_rot:.2f}° (cumulative from input)")
    print(f"    Spring2 (after wedge):    {means['total_rotation']:.2f}° (final)")
    print(f"    Wedge+Spring2 added:      {s2_contribution:.2f}° (from spring1 to final)")

    attn_frac = s1_rot / means['total_rotation'] * 100
    ffn_frac = s2_contribution / means['total_rotation'] * 100
    print(f"\n  → Attention path: {attn_frac:.1f}% of total rotation")
    print(f"  → FFN path:       {ffn_frac:.1f}% of total rotation")

    # ================================================================
    # PART 6: PER-HEAD LEVER ANALYSIS
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 6: PER-HEAD LEVER ANALYSIS")
    print("=" * 60)

    # Get attention config
    config = model.config
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // n_heads
    print(f"  Heads: {n_heads} Q-heads, {n_kv_heads} KV-heads, dim={head_dim}")

    # Extract per-head contributions by decomposing O-projection
    # attn_output = concat(head_outputs) @ W_o
    # We can look at the o_proj weight structure
    W_o = layer0.self_attn.o_proj.weight  # [hidden_size, hidden_size]
    print(f"  O-proj shape: {W_o.shape}")

    # Each head's contribution goes through a slice of W_o
    # Head i maps from [head_dim] at position [i*head_dim : (i+1)*head_dim]
    head_norms = []
    for h in range(n_heads):
        W_o_head = W_o[:, h * head_dim:(h + 1) * head_dim]
        head_norms.append(W_o_head.norm().item())

    head_norms = np.array(head_norms)
    top_heads = np.argsort(head_norms)[::-1][:5]
    bot_heads = np.argsort(head_norms)[:5]

    print(f"\n  O-proj weight norms by head (proxy for head influence):")
    print(f"    Mean: {head_norms.mean():.4f}, Std: {head_norms.std():.4f}")
    print(f"    Top 5:    {', '.join(f'H{h}={head_norms[h]:.3f}' for h in top_heads)}")
    print(f"    Bottom 5: {', '.join(f'H{h}={head_norms[h]:.3f}' for h in bot_heads)}")
    print(f"    Max/Min ratio: {head_norms.max() / head_norms.min():.2f}×")

    # ================================================================
    # PART 7: COMPOUND MACHINE SIGNATURE
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 7: COMPOUND MACHINE SIGNATURE")
    print("=" * 60)

    # The φ-Projector is: Damper → Lever → Spring → Damper → Wedge → Spring
    # Can we characterize it by a few key ratios?

    print("\n  Machine parameters (mean across prompts):")
    print(f"    Damper 1: contraction = {d1_contract:.4f}, rotation = {d1_rot:.2f}°")
    print(f"    Lever:    magnitude = {lever_mag:.4f}, rotation = {means['lever_rotation']:.2f}°")
    print(f"    Spring 1: dilution = {s1_dilute:.4f}")
    print(f"    Damper 2: contraction = {d2_contract:.4f}, rotation = {means['damper2_rotation']:.2f}°")
    print(f"    Wedge:    magnitude = {wedge_mag:.4f}, rotation = {means['wedge_rotation']:.2f}°")
    print(f"    Spring 2: dilution = {s2_dilute:.4f}")

    # Is there a simple formula?
    # If the spring halves the lever's rotation:
    # total ≈ lever_rotation * spring_factor + wedge_rotation * spring_factor
    # Let's check
    spring1_factor = s1_rot / means['lever_rotation'] if means['lever_rotation'] > 0 else 0
    print(f"\n  Spring 1 rotation survival: {spring1_factor:.4f}")
    print(f"    (Lever rotates {means['lever_rotation']:.1f}°, but only {s1_rot:.1f}° survives residual addition)")

    # What fraction of lever rotation gets through?
    # If h_out = h_in + attn_out, and ||attn_out|| << ||h_in||,
    # then angle ≈ arctan(||attn_out|| / ||h_in||)
    # But if ||attn_out|| >> ||h_in||, angle ≈ lever_rotation
    lever_to_input_ratio = norm_means['2_lever_norm'] / norm_means['0_input_norm']
    predicted_spring1_angle = math.degrees(math.atan(lever_to_input_ratio))
    print(f"\n  Lever/Input norm ratio: {lever_to_input_ratio:.4f}")
    print(f"  Predicted spring1 angle (arctan model): {predicted_spring1_angle:.2f}°")
    print(f"  Actual spring1 angle: {s1_rot:.2f}°")

    wedge_to_mid_ratio = norm_means['5_wedge_norm'] / norm_means['3_spring1_norm']
    predicted_spring2_contribution = math.degrees(math.atan(wedge_to_mid_ratio))
    print(f"\n  Wedge/Mid norm ratio: {wedge_to_mid_ratio:.4f}")
    print(f"  Predicted wedge contribution (arctan model): {predicted_spring2_contribution:.2f}°")
    print(f"  Actual wedge contribution: {s2_contribution:.2f}°")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results = {
        "means": {k: float(v) for k, v in means.items()},
        "stds": {k: float(v) for k, v in stds.items()},
        "norm_means": {k: float(v) for k, v in norm_means.items()},
        "direction_analysis": {
            "damper1_contraction": float(d1_contract),
            "lever_magnitude": float(lever_mag),
            "spring1_dilution": float(s1_dilute),
            "damper2_contraction": float(d2_contract),
            "wedge_magnitude": float(wedge_mag),
            "spring2_dilution": float(s2_dilute),
        },
        "cross_correlations": {
            "lever_wedge_cos": float(lw_cos),
            "lever_input_cos": float(li_cos),
            "wedge_input_cos": float(wi_cos),
        },
        "energy_budget": {
            "input_frac": float(e_input),
            "lever_frac": float(e_lever),
            "wedge_frac": float(e_wedge),
        },
        "rotation_attribution": {
            "attention_path_pct": float(attn_frac),
            "ffn_path_pct": float(ffn_frac),
        },
        "arctan_model": {
            "lever_to_input_ratio": float(lever_to_input_ratio),
            "predicted_spring1_angle": float(predicted_spring1_angle),
            "actual_spring1_angle": float(s1_rot),
            "wedge_to_mid_ratio": float(wedge_to_mid_ratio),
            "predicted_spring2_contribution": float(predicted_spring2_contribution),
            "actual_spring2_contribution": float(s2_contribution),
        },
        "head_norms": {f"H{i}": float(v) for i, v in enumerate(head_norms)},
        "traces": all_traces,
    }

    out_path = "results/phase10t_projector_dissection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    print("\n" + "=" * 80)
    print("  PHASE 10t COMPLETE — φ-PROJECTOR DISSECTION")
    print("=" * 80)


if __name__ == "__main__":
    main()
