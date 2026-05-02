#!/usr/bin/env python3
"""
Ternary MLP Decomposition — Negative Zero as 4th Dimension
============================================================

Key insight from Phase 17C: "dead" channels contribute 31.6% of output energy.
They're anti-correlated with alive channels (cos ≈ -0.19). GELU/SiLU leakage IS signal.

Doc 247 ternary φ-map:
  EXPAND   (g > +log(φ)):  SiLU(g) ≈ g        (full fire)
  PRESERVE (|g| ≤ log(φ)): SiLU(g) ≈ g/2      (linear regime)
  CONTRACT (g < -log(φ)):  SiLU(g) ≈ g·exp(g)  (negative leakage)

The hypothesis: negative zero is meaningful.
- SiLU(+ε) → +ε/2  (positive zero)
- SiLU(-ε) → -ε·σ(-ε) (negative zero — carries sign information!)

In 4D φ-space, the SIGN at zero magnitude IS the fourth coordinate.
This gives us 4 states, not 2:
  +1: strongly positive (EXPAND)
  +0: weakly positive (PRESERVE+)
  -0: weakly negative (PRESERVE-)
  -1: strongly negative (CONTRACT)

This maps to 2 bits per channel — sign bit + magnitude bit.
Phase 17D already showed: "sign pattern > magnitude for information (5/6 blocks)"

This script tests:
1. Energy contribution from each ternary region
2. Whether CONTRACT (negative-zero) contribution is structured/low-rank
3. Whether sign-at-zero carries information the magnitude doesn't
4. Whether a ternary approximation preserves more than binary
5. Anti-correlation between positive and negative contributions
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)  # ≈ 0.481


def silu(x):
    return x * torch.sigmoid(x)


def main():
    print("=" * 70)
    print("TERNARY MLP — Negative Zero as 4th Dimension")
    print(f"φ = {PHI:.6f}, log(φ) = {LOG_PHI:.6f}")
    print("=" * 70)

    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    hidden_dim = model.config.hidden_size       # 3584
    inter_dim = model.config.intermediate_size   # 18944

    # Capture MLP inputs via hooks
    prompt = (
        "The Riemann hypothesis states that all non-trivial zeros of the "
        "Riemann zeta function have real part equal to one half. This is one "
        "of the most important unsolved problems in mathematics. The golden "
        "ratio phi appears in the distribution of prime numbers."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"\nPrompt tokens: {seq_len}")

    test_layers = [0, 7, 14, 21, 27]
    mlp_inputs = {L: [] for L in test_layers}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0][0].detach()
            for pos in range(x.shape[0]):
                mlp_inputs[layer_idx].append(x[pos].clone())
        return hook_fn

    hooks = []
    for L in test_layers:
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(L))
        hooks.append(h)
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    # ================================================================
    # Step 1: Ternary decomposition — energy from each region
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Ternary energy decomposition (EXPAND / PRESERVE / CONTRACT)")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        expand_energies = []
        preserve_energies = []
        contract_energies = []
        total_energies = []

        # Also track anti-correlation
        pos_outputs = []
        neg_outputs = []

        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)  # (inter_dim,)
                up_out = F.linear(x, W_up)
                activated = silu(gate_out) * up_out  # (inter_dim,)

                # Ternary classification
                expand_mask = gate_out > LOG_PHI
                contract_mask = gate_out < -LOG_PHI
                preserve_mask = ~expand_mask & ~contract_mask

                # Per-region activated values (zero out other regions)
                act_expand = activated.clone()
                act_expand[~expand_mask] = 0
                act_preserve = activated.clone()
                act_preserve[~preserve_mask] = 0
                act_contract = activated.clone()
                act_contract[~contract_mask] = 0

                # Project through W_down to get per-region OUTPUT contributions
                out_expand = F.linear(act_expand, W_down)
                out_preserve = F.linear(act_preserve, W_down)
                out_contract = F.linear(act_contract, W_down)
                out_full = F.linear(activated, W_down)

                # Energy (L2 norm squared)
                expand_energies.append(torch.norm(out_expand).item()**2)
                preserve_energies.append(torch.norm(out_preserve).item()**2)
                contract_energies.append(torch.norm(out_contract).item()**2)
                total_energies.append(torch.norm(out_full).item()**2)

                # Store for anti-correlation analysis
                # "positive" = EXPAND + PRESERVE+, "negative" = CONTRACT + PRESERVE-
                pos_mask = gate_out > 0
                neg_mask = gate_out <= 0

                act_pos = activated.clone()
                act_pos[~pos_mask] = 0
                act_neg = activated.clone()
                act_neg[~neg_mask] = 0

                pos_outputs.append(F.linear(act_pos, W_down).numpy())
                neg_outputs.append(F.linear(act_neg, W_down).numpy())

        avg_expand = np.mean(expand_energies)
        avg_preserve = np.mean(preserve_energies)
        avg_contract = np.mean(contract_energies)
        avg_total = np.mean(total_energies)

        # Channel counts
        sample_gate = F.linear(mlp_inputs[layer_idx][0], W_gate)
        n_expand = (sample_gate > LOG_PHI).sum().item()
        n_preserve = ((sample_gate >= -LOG_PHI) & (sample_gate <= LOG_PHI)).sum().item()
        n_contract = (sample_gate < -LOG_PHI).sum().item()

        print(f"\n  Layer {layer_idx}:")
        print(f"    Channel counts (sample): E={n_expand} P={n_preserve} C={n_contract}")
        print(f"    Energy fractions:")
        print(f"      EXPAND   (g > +log(φ)):  {avg_expand/avg_total*100:6.2f}%")
        print(f"      PRESERVE (|g| ≤ log(φ)): {avg_preserve/avg_total*100:6.2f}%")
        print(f"      CONTRACT (g < -log(φ)):  {avg_contract/avg_total*100:6.2f}%")
        print(f"      Sum vs total:            {(avg_expand+avg_preserve+avg_contract)/avg_total*100:6.2f}%")
        # Note: sum > 100% because cross-terms can be negative

        # Anti-correlation between positive and negative contributions
        pos_arr = np.array(pos_outputs)
        neg_arr = np.array(neg_outputs)

        anti_corrs = []
        for i in range(len(pos_arr)):
            if np.std(pos_arr[i]) > 0 and np.std(neg_arr[i]) > 0:
                c = np.corrcoef(pos_arr[i], neg_arr[i])[0, 1]
                anti_corrs.append(c)

        print(f"    Anti-correlation (pos vs neg output): {np.mean(anti_corrs):.4f}")

    # ================================================================
    # Step 2: Does the sign at zero carry information?
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Sign at zero — does negative zero carry information?")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        # For channels in the PRESERVE region (|g| < log(φ)):
        # Compare: keeping their actual SiLU value vs replacing with |SiLU| (losing sign)
        corrs_with_sign = []
        corrs_no_sign = []
        corrs_sign_only = []

        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                up_out = F.linear(x, W_up)
                activated = silu(gate_out) * up_out
                out_full = F.linear(activated, W_down)

                # PRESERVE region
                preserve_mask = (gate_out.abs() <= LOG_PHI)

                # Test 1: Full output (reference)
                # (already have out_full)

                # Test 2: In PRESERVE region, replace SiLU(g) with |SiLU(g)|
                # This removes sign-at-zero information
                act_nosign = activated.clone()
                act_nosign[preserve_mask] = act_nosign[preserve_mask].abs()
                out_nosign = F.linear(act_nosign, W_down)

                # Test 3: In PRESERVE region, keep only the SIGN (replace magnitude with constant)
                # This keeps ONLY sign information
                act_signonly = activated.clone()
                preserve_vals = act_signonly[preserve_mask]
                act_signonly[preserve_mask] = torch.sign(preserve_vals) * preserve_vals.abs().mean()
                out_signonly = F.linear(act_signonly, W_down)

                # Test 4: Zero out the PRESERVE region entirely
                act_no_preserve = activated.clone()
                act_no_preserve[preserve_mask] = 0
                out_no_preserve = F.linear(act_no_preserve, W_down)

                full_np = out_full.numpy()
                corrs_with_sign.append(1.0)  # reference
                corrs_no_sign.append(np.corrcoef(full_np, out_nosign.numpy())[0, 1])
                corrs_sign_only.append(np.corrcoef(full_np, out_signonly.numpy())[0, 1])

        print(f"\n  Layer {layer_idx}:")
        print(f"    Full MLP (reference):                   1.000000")
        print(f"    Remove sign in PRESERVE (|SiLU|):       {np.mean(corrs_no_sign):.6f}")
        print(f"    Keep only sign in PRESERVE:             {np.mean(corrs_sign_only):.6f}")

    # ================================================================
    # Step 3: Ternary approximation — linearize each region differently
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Ternary approximation (different approx per region)")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        corrs_binary = []      # active/dead binary (skip CONTRACT)
        corrs_ternary = []     # different approx per region
        corrs_ternary_neg = [] # ternary + negative zero contribution

        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                up_out = F.linear(x, W_up)
                activated = silu(gate_out) * up_out
                out_full = F.linear(activated, W_down)

                expand_mask = gate_out > LOG_PHI
                contract_mask = gate_out < -LOG_PHI
                preserve_mask = ~expand_mask & ~contract_mask

                # === Binary: skip CONTRACT entirely ===
                act_binary = activated.clone()
                act_binary[contract_mask] = 0
                out_binary = F.linear(act_binary, W_down)

                # === Ternary: approximate each region ===
                act_ternary = torch.zeros_like(activated)
                # EXPAND: SiLU(g) ≈ g (identity)
                act_ternary[expand_mask] = gate_out[expand_mask] * up_out[expand_mask]
                # PRESERVE: SiLU(g) ≈ g/2 (linear)
                act_ternary[preserve_mask] = (gate_out[preserve_mask] / 2) * up_out[preserve_mask]
                # CONTRACT: skip (binary behavior)
                out_ternary = F.linear(act_ternary, W_down)

                # === Ternary + negative zero: include CONTRACT leakage ===
                act_ternary_neg = act_ternary.clone()
                # CONTRACT: SiLU(g) ≈ g * exp(g) for g << 0
                # More accurate: keep SiLU but flag these as "negative zero"
                g_contract = gate_out[contract_mask]
                # Actual SiLU leakage (exact, just categorized)
                act_ternary_neg[contract_mask] = silu(g_contract) * up_out[contract_mask]
                out_ternary_neg = F.linear(act_ternary_neg, W_down)

                full_np = out_full.numpy()
                corrs_binary.append(np.corrcoef(full_np, out_binary.numpy())[0, 1])
                corrs_ternary.append(np.corrcoef(full_np, out_ternary.numpy())[0, 1])
                corrs_ternary_neg.append(np.corrcoef(full_np, out_ternary_neg.numpy())[0, 1])

        print(f"\n  Layer {layer_idx}:")
        print(f"    Binary (skip CONTRACT):                 {np.mean(corrs_binary):.6f}")
        print(f"    Ternary (approx each region, no CONTRACT): {np.mean(corrs_ternary):.6f}")
        print(f"    Ternary + negative zero (full):         {np.mean(corrs_ternary_neg):.6f}")

    # ================================================================
    # Step 4: Is CONTRACT contribution low-rank?
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Is the CONTRACT (negative zero) contribution low-rank?")
    print("=" * 70)

    for layer_idx in [0, 14, 27]:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        # Collect CONTRACT-only outputs across all tokens
        contract_outputs = []

        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                up_out = F.linear(x, W_up)
                activated = silu(gate_out) * up_out

                contract_mask = gate_out < -LOG_PHI
                act_contract = torch.zeros_like(activated)
                act_contract[contract_mask] = activated[contract_mask]
                out_contract = F.linear(act_contract, W_down)
                contract_outputs.append(out_contract.numpy())

        contract_matrix = np.array(contract_outputs)  # (n_tokens, hidden_dim)

        # SVD of contract outputs
        U, S, Vh = np.linalg.svd(contract_matrix, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)

        rank_90 = np.searchsorted(cumvar, 0.90) + 1
        rank_95 = np.searchsorted(cumvar, 0.95) + 1
        rank_99 = np.searchsorted(cumvar, 0.99) + 1

        print(f"\n  Layer {layer_idx}:")
        print(f"    CONTRACT output matrix: ({contract_matrix.shape[0]}, {contract_matrix.shape[1]})")
        print(f"    S[0]/S[1] = {S[0]/S[1]:.3f}")
        print(f"    Rank for 90% var: {rank_90}/{min(contract_matrix.shape)}")
        print(f"    Rank for 95% var: {rank_95}/{min(contract_matrix.shape)}")
        print(f"    Rank for 99% var: {rank_99}/{min(contract_matrix.shape)}")
        print(f"    CONTRACT energy / total energy: ", end="")

        # Also measure ratio
        total_outputs = []
        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                up_out = F.linear(x, W_up)
                out = F.linear(silu(gate_out) * up_out, W_down)
                total_outputs.append(out.numpy())
        total_matrix = np.array(total_outputs)

        contract_energy = np.sum(contract_matrix**2)
        total_energy = np.sum(total_matrix**2)
        print(f"{contract_energy/total_energy*100:.2f}%")

    # ================================================================
    # Step 5: 4-state encoding: +1, +0, -0, -1
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: 4-state gate encoding (+1, +0, -0, -1)")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data

        # Classify every channel for every token into 4 states
        state_counts = {'+1': 0, '+0': 0, '-0': 0, '-1': 0}
        total = 0

        # Track uniqueness of 4-state patterns vs 2-state (binary)
        binary_patterns = set()
        quad_patterns = set()

        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)

                # 4-state classification
                expand = gate_out > LOG_PHI           # +1
                preserve_pos = (gate_out > 0) & (gate_out <= LOG_PHI)  # +0
                preserve_neg = (gate_out <= 0) & (gate_out >= -LOG_PHI)  # -0
                contract = gate_out < -LOG_PHI        # -1

                state_counts['+1'] += expand.sum().item()
                state_counts['+0'] += preserve_pos.sum().item()
                state_counts['-0'] += preserve_neg.sum().item()
                state_counts['-1'] += contract.sum().item()
                total += gate_out.numel()

                # Binary: just positive/negative
                binary_pat = tuple((gate_out > 0).numpy().astype(np.int8))
                binary_patterns.add(binary_pat)

                # Quad: 4-state (2 bits)
                quad = np.zeros(gate_out.shape[0], dtype=np.int8)
                quad[expand.numpy()] = 3
                quad[preserve_pos.numpy()] = 2
                quad[preserve_neg.numpy()] = 1
                quad[contract.numpy()] = 0
                quad_patterns.add(tuple(quad))

        print(f"\n  Layer {layer_idx}:")
        for state, count in state_counts.items():
            print(f"    {state}: {count/total*100:6.2f}%")

        print(f"    Unique binary patterns:  {len(binary_patterns)}/{len(mlp_inputs[layer_idx])}")
        print(f"    Unique 4-state patterns: {len(quad_patterns)}/{len(mlp_inputs[layer_idx])}")

        # Information content: how much more information in 4-state vs binary?
        # Binary: 1 bit per channel
        # 4-state: 2 bits per channel
        # But if the extra bit is always predictable from the first, no extra info
        # Measure: given binary sign, how predictable is the magnitude bit?
        predictable = 0
        total_check = 0
        for x in mlp_inputs[layer_idx]:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                sign_bit = (gate_out > 0)  # binary
                mag_bit = (gate_out.abs() > LOG_PHI)  # magnitude threshold

                # For positive channels: is it +1 or +0?
                # For negative channels: is it -1 or -0?
                # If sign_bit predicts mag_bit, then 4-state = binary + redundant
                pos_channels = sign_bit
                neg_channels = ~sign_bit

                # Among positive: what fraction are +1 (expand)?
                if pos_channels.sum() > 0:
                    pos_expand_rate = (gate_out[pos_channels] > LOG_PHI).float().mean().item()
                else:
                    pos_expand_rate = 0

                # Among negative: what fraction are -1 (contract)?
                if neg_channels.sum() > 0:
                    neg_contract_rate = (gate_out[neg_channels] < -LOG_PHI).float().mean().item()
                else:
                    neg_contract_rate = 0

        print(f"    Among positive channels: {pos_expand_rate*100:.1f}% are EXPAND (+1), {(1-pos_expand_rate)*100:.1f}% are PRESERVE+ (+0)")
        print(f"    Among negative channels: {neg_contract_rate*100:.1f}% are CONTRACT (-1), {(1-neg_contract_rate)*100:.1f}% are PRESERVE- (-0)")

    # ================================================================
    # Step 6: End-to-end ternary MLP with negative zero
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: End-to-end test — ternary + negative zero")
    print("=" * 70)

    def make_ternary_hook(include_contract=True):
        """Hook that replaces MLP with ternary approximation."""
        def hook_fn(module, inp, out):
            x = inp[0]
            W_g = module.gate_proj.weight.data
            W_u = module.up_proj.weight.data
            W_d = module.down_proj.weight.data

            with torch.no_grad():
                gate_out = F.linear(x, W_g)
                up_out = F.linear(x, W_u)

                expand_mask = gate_out > LOG_PHI
                contract_mask = gate_out < -LOG_PHI
                preserve_mask = ~expand_mask & ~contract_mask

                activated = torch.zeros_like(gate_out)

                # EXPAND: SiLU(g) ≈ g (identity for large positive)
                activated[expand_mask] = gate_out[expand_mask] * up_out[expand_mask]

                # PRESERVE: SiLU(g) ≈ g/2 (linear regime)
                activated[preserve_mask] = (gate_out[preserve_mask] / 2) * up_out[preserve_mask]

                if include_contract:
                    # CONTRACT: keep actual SiLU leakage (negative zero)
                    activated[contract_mask] = silu(gate_out[contract_mask]) * up_out[contract_mask]
                # else: CONTRACT channels stay at 0 (skip negative zero)

                return F.linear(activated, W_d)
        return hook_fn

    test_prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("def hello():\n    print(", '"'),
        ("1 + 1 =", "2"),
        ("Water freezes at", "0"),
    ]

    for mode_name, include_contract in [("Ternary (no negative zero)", False),
                                          ("Ternary + negative zero", True)]:
        correct = 0
        match = 0

        for prompt, expected in test_prompts:
            test_inp = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                # Full model
                out_full = model(**test_inp)
                token_full = torch.argmax(out_full.logits[0, -1]).item()
                word_full = tokenizer.decode([token_full])

                # Ternary model
                ternary_hooks = []
                for L in range(28):
                    h = model.model.layers[L].mlp.register_forward_hook(
                        make_ternary_hook(include_contract)
                    )
                    ternary_hooks.append(h)

                out_ternary = model(**test_inp)
                token_ternary = torch.argmax(out_ternary.logits[0, -1]).item()
                word_ternary = tokenizer.decode([token_ternary])

                for h in ternary_hooks:
                    h.remove()

            full_ok = expected.lower() in word_full.lower().strip()
            ternary_ok = expected.lower() in word_ternary.lower().strip()
            same = token_full == token_ternary

            if ternary_ok: correct += 1
            if same: match += 1

        print(f"\n  {mode_name}:")
        print(f"    Correct: {correct}/{len(test_prompts)}")
        print(f"    Same argmax as full: {match}/{len(test_prompts)}")

    del model
    print("\nDone.")


if __name__ == "__main__":
    main()
