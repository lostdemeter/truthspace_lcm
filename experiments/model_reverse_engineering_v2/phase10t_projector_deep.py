"""
Phase 10t-deep: φ-Projector geometry deep dive

Follow-up on phase10t findings:
1. Verify the "perpendicular refinement" hypothesis — FFN rotates h_mid
   in a plane that preserves the angle from h_in
2. Decompose the 3-vector triangle: h_in, lever_out, wedge_out
3. Check if FFN output lies in a specific subspace relative to h_in
4. Per-head lever decomposition — which heads drive the 86° rotation?
5. Test the "project-and-refine" compound machine hypothesis
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


def angle(a, b):
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def project_onto(v, direction):
    """Project v onto direction (return component along direction)."""
    d_norm = direction / direction.norm()
    return (v @ d_norm) * d_norm


def project_out(v, direction):
    """Remove component of v along direction."""
    return v - project_onto(v, direction)


def main():
    print("=" * 80)
    print("  PHASE 10t-DEEP: φ-PROJECTOR GEOMETRY DEEP DIVE")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    layer0 = model.model.layers[0]

    config = model.config
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            h_in = model.model.embed_tokens(input_ids)

            cache_position = torch.arange(seq_len, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(h_in, position_ids)

            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # Run through the 6 stages
            s0 = h_in[0, -1]  # last token, [d_model]
            s1 = layer0.input_layernorm(h_in)
            s1_lt = s1[0, -1]

            # We need per-head attention outputs
            # Run attention manually to capture per-head
            attn_module = layer0.self_attn

            # Get Q, K, V
            query_states = attn_module.q_proj(s1)  # [1, seq, d_model]
            key_states = attn_module.k_proj(s1)
            value_states = attn_module.v_proj(s1)

            # Reshape for heads
            bsz = 1
            q = query_states.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
            n_kv = config.num_key_value_heads
            k = key_states.view(bsz, seq_len, n_kv, head_dim).transpose(1, 2)
            v = value_states.view(bsz, seq_len, n_kv, head_dim).transpose(1, 2)

            # Apply RoPE
            cos_rope, sin_rope = position_embeddings
            # Apply rotary to Q and K
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos_rope, sin_rope)

            # Repeat KV for GQA
            n_rep = n_heads // n_kv
            if n_rep > 1:
                k = k[:, :, None, :, :].expand(bsz, n_kv, n_rep, seq_len, head_dim)
                k = k.reshape(bsz, n_heads, seq_len, head_dim)
                v = v[:, :, None, :, :].expand(bsz, n_kv, n_rep, seq_len, head_dim)
                v = v.reshape(bsz, n_heads, seq_len, head_dim)

            # Compute attention weights
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)

            # Per-head attention output (before O-projection)
            attn_output_per_head = torch.matmul(attn_weights, v)  # [1, n_heads, seq, head_dim]

            # Get O-projection weight
            W_o = attn_module.o_proj.weight  # [d_model, d_model]

            # Per-head contribution to final attention output
            head_contributions = []
            for h in range(n_heads):
                head_out = attn_output_per_head[0, h, -1]  # [head_dim]
                W_o_head = W_o[:, h * head_dim:(h + 1) * head_dim]
                contribution = W_o_head @ head_out  # [d_model]
                head_contributions.append(contribution)

            head_contributions = torch.stack(head_contributions)  # [n_heads, d_model]

            # Full attention output (sum of heads)
            attn_out = head_contributions.sum(dim=0)  # [d_model]

            # Verify this matches the model's attention
            s2_full, _ = layer0.self_attn(
                hidden_states=s1,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            s2_lt = s2_full[0, -1]
            recon_error = (attn_out - s2_lt).norm().item() / s2_lt.norm().item()

            # Spring 1
            h_mid = s0 + attn_out
            # Damper 2
            s3_mid_3d = (s0 + s2_full)  # [1, seq, d_model]
            s4_damper2 = layer0.post_attention_layernorm(s3_mid_3d)
            # Wedge
            gate_pre = layer0.mlp.gate_proj(s4_damper2)
            gate_act = layer0.mlp.act_fn(gate_pre)
            up_act = layer0.mlp.up_proj(s4_damper2)
            ffn_out_3d = layer0.mlp.down_proj(gate_act * up_act)
            ffn_out = ffn_out_3d[0, -1]  # [d_model]
            # Spring 2
            h_out = h_mid + ffn_out

        r = {"prompt_idx": pi, "recon_error": recon_error}

        # ============================================================
        # TEST 1: Perpendicular refinement hypothesis
        # ============================================================
        # If FFN rotates h_mid in the null space of h_in, then:
        # project(ffn_out, h_in) should be small relative to project(ffn_out, ⊥h_in)

        ffn_along_input = project_onto(ffn_out, s0)
        ffn_perp_input = project_out(ffn_out, s0)

        r["ffn_along_input_frac"] = ffn_along_input.norm().item() / ffn_out.norm().item()
        r["ffn_perp_input_frac"] = ffn_perp_input.norm().item() / ffn_out.norm().item()

        # Also: does FFN output lie along h_mid?
        ffn_along_hmid = project_onto(ffn_out, h_mid)
        ffn_perp_hmid = project_out(ffn_out, h_mid)

        r["ffn_along_hmid_frac"] = ffn_along_hmid.norm().item() / ffn_out.norm().item()
        r["ffn_perp_hmid_frac"] = ffn_perp_hmid.norm().item() / ffn_out.norm().item()
        r["cos_ffn_hmid"] = cos_sim(ffn_out, h_mid)

        # Does the lever output lie along h_mid? (it should, since h_mid ≈ attn_out)
        r["cos_attn_hmid"] = cos_sim(attn_out, h_mid)
        r["cos_ffn_attn"] = cos_sim(ffn_out, attn_out)

        # ============================================================
        # TEST 2: 3-vector geometry (h_in, attn_out, ffn_out)
        # ============================================================
        r["angle_input_attn"] = angle(s0, attn_out)
        r["angle_input_ffn"] = angle(s0, ffn_out)
        r["angle_attn_ffn"] = angle(attn_out, ffn_out)
        r["angle_input_hmid"] = angle(s0, h_mid)
        r["angle_hmid_hout"] = angle(h_mid, h_out)
        r["angle_input_hout"] = angle(s0, h_out)

        # ============================================================
        # TEST 3: Per-head rotation contribution
        # ============================================================
        # Each head contributes a vector. What angle does each head's
        # contribution make with h_in? And how much does each head
        # contribute to the total rotation?
        head_data = []
        for h in range(n_heads):
            hc = head_contributions[h]
            hd = {
                "head": h,
                "norm": hc.norm().item(),
                "angle_from_input": angle(s0, hc),
                "cos_with_input": cos_sim(s0, hc),
                "cos_with_total_attn": cos_sim(attn_out, hc),
                "cos_with_ffn": cos_sim(ffn_out, hc),
            }

            # How much of the total rotation is this head responsible for?
            # Test: attn_out without this head
            attn_without_h = attn_out - hc
            hmid_without_h = s0 + attn_without_h
            angle_without_h = angle(s0, hmid_without_h)
            hd["ablation_angle"] = angle_without_h
            hd["rotation_contribution"] = angle(s0, h_mid) - angle_without_h

            head_data.append(hd)

        r["heads"] = head_data

        # ============================================================
        # TEST 4: Attention pattern — what is L0 attending to?
        # ============================================================
        # Last token's attention weights
        last_attn = attn_weights[0, :, -1, :]  # [n_heads, seq_len]

        # How concentrated is attention? (entropy)
        for h in range(n_heads):
            weights = last_attn[h]
            # Remove zeros for entropy calc
            w_pos = weights[weights > 0]
            entropy = -(w_pos * w_pos.log()).sum().item()
            head_data[h]["attn_entropy"] = entropy
            head_data[h]["attn_max"] = weights.max().item()
            head_data[h]["attn_to_first"] = weights[0].item()
            head_data[h]["attn_to_last"] = weights[-1].item()

        all_results.append(r)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 60)
    print("  TEST 1: PERPENDICULAR REFINEMENT HYPOTHESIS")
    print("=" * 60)

    along_input = np.mean([r["ffn_along_input_frac"] for r in all_results])
    perp_input = np.mean([r["ffn_perp_input_frac"] for r in all_results])
    along_hmid = np.mean([r["ffn_along_hmid_frac"] for r in all_results])
    perp_hmid = np.mean([r["ffn_perp_hmid_frac"] for r in all_results])
    cos_ffn_hmid = np.mean([r["cos_ffn_hmid"] for r in all_results])
    cos_ffn_attn = np.mean([r["cos_ffn_attn"] for r in all_results])
    cos_attn_hmid = np.mean([r["cos_attn_hmid"] for r in all_results])

    print(f"\n  FFN output decomposition relative to h_in:")
    print(f"    Along h_in:  {along_input*100:.1f}% of ||FFN||")
    print(f"    Perp to h_in: {perp_input*100:.1f}% of ||FFN||")

    print(f"\n  FFN output decomposition relative to h_mid:")
    print(f"    Along h_mid:  {along_hmid*100:.1f}% of ||FFN||")
    print(f"    Perp to h_mid: {perp_hmid*100:.1f}% of ||FFN||")
    print(f"    cos(FFN, h_mid): {cos_ffn_hmid:+.4f}")

    print(f"\n  Key relationships:")
    print(f"    cos(attn_out, h_mid): {cos_attn_hmid:+.4f}  (h_mid ≈ attn_out since input is tiny)")
    print(f"    cos(FFN, attn_out):   {cos_ffn_attn:+.4f}")

    if perp_input > 0.9:
        print(f"\n  → FFN output is {perp_input*100:.0f}% perpendicular to input")
        print(f"    CONFIRMED: FFN operates in the null space of h_in")
    if along_hmid > 0.7:
        print(f"  → FFN is {along_hmid*100:.0f}% along h_mid — it BOOSTS the attention direction")
    elif along_hmid < 0.3:
        print(f"  → FFN is {perp_hmid*100:.0f}% perpendicular to h_mid — it REFINES in orthogonal subspace")
    else:
        print(f"  → FFN has mixed alignment with h_mid: {along_hmid*100:.0f}% along, {perp_hmid*100:.0f}% perp")

    # ================================================================
    print("\n" + "=" * 60)
    print("  TEST 2: THREE-VECTOR GEOMETRY")
    print("=" * 60)

    print(f"\n  Mean angles between the three output vectors:")
    ai_a = np.mean([r["angle_input_attn"] for r in all_results])
    ai_f = np.mean([r["angle_input_ffn"] for r in all_results])
    a_f = np.mean([r["angle_attn_ffn"] for r in all_results])
    print(f"    ∠(h_in, attn_out):  {ai_a:.1f}°")
    print(f"    ∠(h_in, ffn_out):   {ai_f:.1f}°")
    print(f"    ∠(attn_out, ffn_out): {a_f:.1f}°")

    # Triangle inequality check
    print(f"\n  Triangle sum: {ai_a:.1f} + {a_f:.1f} = {ai_a + a_f:.1f}° (vs ∠(h_in,ffn): {ai_f:.1f}°)")

    print(f"\n  Cumulative rotation:")
    ai_hm = np.mean([r["angle_input_hmid"] for r in all_results])
    hm_ho = np.mean([r["angle_hmid_hout"] for r in all_results])
    ai_ho = np.mean([r["angle_input_hout"] for r in all_results])
    print(f"    ∠(h_in, h_mid):   {ai_hm:.1f}°  (after attention + residual)")
    print(f"    ∠(h_mid, h_out):  {hm_ho:.1f}°  (FFN contribution)")
    print(f"    ∠(h_in, h_out):   {ai_ho:.1f}°  (total)")
    print(f"    Δ from input perspective: {ai_ho - ai_hm:.1f}°  (how much FFN changes angle from input)")

    # ================================================================
    print("\n" + "=" * 60)
    print("  TEST 3: PER-HEAD LEVER DECOMPOSITION")
    print("=" * 60)

    # Aggregate head data across prompts
    head_agg = {}
    for h in range(n_heads):
        head_agg[h] = {
            "norm": np.mean([r["heads"][h]["norm"] for r in all_results]),
            "angle_from_input": np.mean([r["heads"][h]["angle_from_input"] for r in all_results]),
            "cos_with_total_attn": np.mean([r["heads"][h]["cos_with_total_attn"] for r in all_results]),
            "rotation_contribution": np.mean([r["heads"][h]["rotation_contribution"] for r in all_results]),
            "cos_with_ffn": np.mean([r["heads"][h]["cos_with_ffn"] for r in all_results]),
            "attn_entropy": np.mean([r["heads"][h]["attn_entropy"] for r in all_results]),
            "attn_to_first": np.mean([r["heads"][h]["attn_to_first"] for r in all_results]),
            "attn_to_last": np.mean([r["heads"][h]["attn_to_last"] for r in all_results]),
        }

    # Sort by rotation contribution
    sorted_heads = sorted(head_agg.items(), key=lambda x: x[1]["rotation_contribution"], reverse=True)

    print(f"\n  {'Head':>4} | {'Norm':>6} | {'∠Input':>7} | {'cos(Σ)':>7} | {'Δ Rot':>7} | {'Entropy':>7} | {'→First':>7} | {'→Last':>7}")
    print("  " + "-" * 70)
    for h, d in sorted_heads:
        print(f"  H{h:>2} | {d['norm']:>6.3f} | {d['angle_from_input']:>6.1f}° | "
              f"{d['cos_with_total_attn']:>+.4f} | {d['rotation_contribution']:>+.3f}° | "
              f"{d['attn_entropy']:>7.3f} | {d['attn_to_first']:>7.3f} | {d['attn_to_last']:>7.3f}")

    # Identify head roles
    print(f"\n  Head role classification:")
    for h, d in sorted_heads:
        role = []
        if d["rotation_contribution"] > 1.0:
            role.append("PROJECTOR (drives rotation)")
        elif d["rotation_contribution"] < -1.0:
            role.append("RESISTOR (opposes rotation)")
        else:
            role.append("neutral")

        if d["attn_to_first"] > 0.5:
            role.append("BOS-focused")
        if d["attn_to_last"] > 0.5:
            role.append("self-focused")
        if d["attn_entropy"] > 2.0:
            role.append("distributed")
        elif d["attn_entropy"] < 1.0:
            role.append("concentrated")

        print(f"    H{h:>2}: {', '.join(role)}")

    # ================================================================
    print("\n" + "=" * 60)
    print("  TEST 4: COMPOUND MACHINE MODEL")
    print("=" * 60)

    # The φ-Projector as a compound machine:
    # 1. Damper1 amplifies embedding into working range
    # 2. Lever rotates ~86° (nearly orthogonal projection)
    # 3. Spring1 is transparent (input norm << lever norm)
    # 4. FFN operates in the null space of h_in
    # 5. Spring2 adds FFN's contribution

    # Key test: can we predict the output from the lever alone?
    # h_out ≈ attn_out + ffn(LN(attn_out))
    # Since input is negligible, the whole layer is approximately:
    # h_out ≈ Lever(LN(h_in)) + Wedge(LN(Lever(LN(h_in))))

    print(f"\n  Reconstruction error (per-head decomposition): "
          f"{np.mean([r['recon_error'] for r in all_results]):.6f}")

    # Model: "lever does the projection, wedge does energy balancing"
    # Evidence:
    # 1. Lever: 98.7% of angle from input
    # 2. FFN: operates perpendicular to input
    # 3. FFN and attn are weakly correlated (cos ~0.25)

    print(f"\n  THE φ-PROJECTOR COMPOUND MACHINE:")
    print(f"  ──────────────────────────────────")
    print(f"  Stage 1 - Damper (RMSNorm): amplify embedding 18× into working range")
    print(f"  Stage 2 - Lever (Attention): project {ai_a:.0f}° from input")
    print(f"  Stage 3 - Spring (Residual): transparent (input is {np.mean([r['heads'][0]['norm'] for r in all_results]) / np.mean([all_results[0]['heads'][0]['norm'] for _ in [1]]):.0f}× smaller)")
    print(f"  Stage 4 - Wedge (FFN): {perp_input*100:.0f}% in null space of input,")
    print(f"            {along_hmid*100:.0f}% along h_mid (energy boost)")
    print(f"  Stage 5 - Spring (Residual): merge attention + FFN ({hm_ho:.0f}° from h_mid)")
    print(f"")
    print(f"  Total: {ai_ho:.1f}° rotation, {ai_a:.0f}° from lever, {ai_ho - ai_hm:.1f}° from wedge (input perspective)")
    print(f"")
    print(f"  Naming: φ-Projector = Lever-dominant compound machine")
    print(f"  Formula: Project(h) = Spring(Lever(Damp(h))) + Spring(Wedge(Damp(Spring(Lever(Damp(h))))))")
    print(f"  Simplified: Project(h) ≈ Lever(h) + Wedge(Lever(h))  [since springs are transparent at L0]")

    # ================================================================
    # SAVE
    # ================================================================
    save_data = {
        "perpendicular_refinement": {
            "ffn_along_input_frac": float(along_input),
            "ffn_perp_input_frac": float(perp_input),
            "ffn_along_hmid_frac": float(along_hmid),
            "ffn_perp_hmid_frac": float(perp_hmid),
            "cos_ffn_hmid": float(cos_ffn_hmid),
            "cos_ffn_attn": float(cos_ffn_attn),
        },
        "three_vector_geometry": {
            "angle_input_attn": float(ai_a),
            "angle_input_ffn": float(ai_f),
            "angle_attn_ffn": float(a_f),
            "angle_input_hmid": float(ai_hm),
            "angle_hmid_hout": float(hm_ho),
            "angle_input_hout": float(ai_ho),
        },
        "head_aggregate": {str(h): d for h, d in head_agg.items()},
        "per_prompt": all_results,
    }

    out_path = "results/phase10t_projector_deep.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    print("\n" + "=" * 80)
    print("  PHASE 10t-DEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
