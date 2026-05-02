#!/usr/bin/env python3
"""
Scaffold MLP Test — Can we eliminate computation in the MLP?
=============================================================

From our prior work:
- Doc 132: SiLU operates in linear regime, SiLU(x) ≈ x/2, 99.99% correlation
- Doc 245: Mean Jacobian IMPROVES on full GELU for DDColor (-1.3% RMSE)
- Doc 247: Scaffold = (1/2) R @ H collapses expand→gate→compress to ONE matmul

Qwen2's MLP is GATED (unlike DDColor's single-expand):
    output = W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))

This script tests:
1. Full MLP (reference)
2. Linearized SiLU: W_down @ ((W_gate @ x / 2) ⊙ (W_up @ x))  [3 matmuls, no SiLU]
3. Naive scaffold: (1/2) W_down @ W_gate @ x  [1 matmul, ignores up_proj]
4. Mean Jacobian: average dMLP/dx over calibration inputs  [1 matmul at runtime]
5. Low-rank Jacobian: SVD of mean Jacobian, keep top-k  [1 smaller matmul]

We measure: correlation, max error, argmax preservation through full model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

PHI = (1 + np.sqrt(5)) / 2


def silu(x):
    """Standard SiLU/Swish activation."""
    return x * torch.sigmoid(x)


def silu_derivative(x):
    """SiLU'(x) = σ(x) + x·σ(x)·(1-σ(x)) = σ(x)·(1 + x·(1-σ(x)))"""
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))


def main():
    print("=" * 70)
    print("SCAFFOLD MLP TEST — Qwen2-7B")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    config = model.config
    hidden_dim = config.hidden_size        # 3584
    inter_dim = config.intermediate_size   # 18944

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Intermediate dim: {inter_dim}")
    print(f"  Expansion ratio: {inter_dim/hidden_dim:.1f}x")

    # ================================================================
    # Step 1: Get real hidden states from multiple prompts
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Capturing real hidden states")
    print("=" * 70)

    prompts = [
        "The capital of France is",
        "In quantum mechanics, the wave function",
        "def fibonacci(n):",
        "The quick brown fox jumps over the",
        "Water boils at a temperature of",
        "According to Einstein's theory of",
        "The largest planet in our solar system is",
        "Machine learning algorithms can be",
    ]

    # We'll test on multiple layers
    test_layers = [0, 7, 14, 21, 27]

    # Collect hidden states at each test layer using hooks
    hidden_states_per_layer = {L: [] for L in test_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0] is the normalized hidden state entering the MLP
            # We want what post_attention_layernorm produces
            pass
        return hook_fn

    # Hook into post_attention_layernorm to capture MLP input
    def make_norm_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is the normalized hidden state that feeds into MLP
            hidden_states_per_layer[layer_idx].append(
                output[0, -1].detach().clone()  # batch=0, last token
            )
        return hook_fn

    for L in test_layers:
        h = model.model.layers[L].post_attention_layernorm.register_forward_hook(
            make_norm_hook(L)
        )
        hooks.append(h)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    for L in test_layers:
        n = len(hidden_states_per_layer[L])
        print(f"  Layer {L:2d}: {n} hidden states captured")

    # ================================================================
    # Step 2: Test MLP approximations per layer
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Testing MLP approximations")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]

        # Extract MLP weights
        W_gate = layer.mlp.gate_proj.weight.data  # (inter_dim, hidden_dim)
        W_up = layer.mlp.up_proj.weight.data      # (inter_dim, hidden_dim)
        W_down = layer.mlp.down_proj.weight.data   # (hidden_dim, inter_dim)

        test_inputs = hidden_states_per_layer[layer_idx]

        print(f"\n{'─' * 70}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 70}")

        # -----------------------------------------------------------
        # Test each input
        # -----------------------------------------------------------
        corrs_lin = []
        corrs_scaffold = []
        max_errs_lin = []
        max_errs_scaffold = []
        gate_stats = []

        # For mean Jacobian computation
        jacobian_sum = torch.zeros(hidden_dim, hidden_dim)
        n_calib = 0

        for x in test_inputs:
            with torch.no_grad():
                # === FULL MLP (reference) ===
                gate_out = F.linear(x, W_gate)      # (inter_dim,)
                up_out = F.linear(x, W_up)           # (inter_dim,)
                activated = silu(gate_out) * up_out   # (inter_dim,)
                full_output = F.linear(activated, W_down)  # (hidden_dim,)

                # === Gate statistics ===
                gate_np = gate_out.numpy()
                log_phi = np.log(PHI)
                pct_preserve = np.mean(np.abs(gate_np) <= log_phi) * 100
                pct_contract = np.mean(gate_np < -log_phi) * 100
                pct_expand = np.mean(gate_np > log_phi) * 100
                gate_stats.append((pct_preserve, pct_contract, pct_expand))

                # === LINEARIZED SiLU: SiLU(gate) ≈ gate/2 ===
                lin_activated = (gate_out / 2) * up_out
                lin_output = F.linear(lin_activated, W_down)

                # === NAIVE SCAFFOLD: (1/2) W_down @ W_gate @ x ===
                # This ignores up_proj entirely — treats MLP as single-path
                scaffold_hidden = F.linear(x, W_gate) / 2
                scaffold_output = F.linear(scaffold_hidden, W_down)

                # === Compute Jacobian for this input ===
                # J(x) = W_down @ [diag(SiLU'(g) ⊙ u) @ W_gate + diag(SiLU(g)) @ W_up]
                silu_prime_g = silu_derivative(gate_out)  # (inter_dim,)
                silu_g = silu(gate_out)                    # (inter_dim,)

                # Term 1: diag(SiLU'(g) * u) @ W_gate
                term1_diag = silu_prime_g * up_out  # (inter_dim,)
                # Term 2: diag(SiLU(g)) @ W_up
                term2_diag = silu_g                 # (inter_dim,)

                # J = W_down @ (diag1 @ W_gate + diag2 @ W_up)
                # = W_down @ diag1 @ W_gate + W_down @ diag2 @ W_up
                # For mean Jacobian, accumulate
                J = (W_down * term1_diag.unsqueeze(0)) @ W_gate + \
                    (W_down * term2_diag.unsqueeze(0)) @ W_up  # (hidden, hidden)
                jacobian_sum += J
                n_calib += 1

                # === Metrics ===
                full_np = full_output.numpy()
                lin_np = lin_output.numpy()
                scaffold_np = scaffold_output.numpy()

                corr_lin = np.corrcoef(full_np, lin_np)[0, 1]
                corr_scaffold = np.corrcoef(full_np, scaffold_np)[0, 1]

                corrs_lin.append(corr_lin)
                corrs_scaffold.append(corr_scaffold)
                max_errs_lin.append(np.max(np.abs(full_np - lin_np)))
                max_errs_scaffold.append(np.max(np.abs(full_np - scaffold_np)))

        # Gate statistics
        avg_preserve = np.mean([s[0] for s in gate_stats])
        avg_contract = np.mean([s[1] for s in gate_stats])
        avg_expand = np.mean([s[2] for s in gate_stats])

        print(f"\n  Gate distribution (±log(φ) = ±{np.log(PHI):.3f} boundaries):")
        print(f"    PRESERVE (|g| ≤ log(φ)): {avg_preserve:.1f}%")
        print(f"    CONTRACT (g < -log(φ)):   {avg_contract:.1f}%")
        print(f"    EXPAND   (g > +log(φ)):   {avg_expand:.1f}%")

        print(f"\n  Linearized SiLU (gate/2, 3 matmuls):")
        print(f"    Correlation:  {np.mean(corrs_lin):.6f} (min {np.min(corrs_lin):.6f})")
        print(f"    Max error:    {np.mean(max_errs_lin):.4f}")

        print(f"\n  Naive scaffold ((1/2) W_down @ W_gate, 1 matmul):")
        print(f"    Correlation:  {np.mean(corrs_scaffold):.6f} (min {np.min(corrs_scaffold):.6f})")
        print(f"    Max error:    {np.mean(max_errs_scaffold):.4f}")

        # -----------------------------------------------------------
        # Mean Jacobian test
        # -----------------------------------------------------------
        J_mean = jacobian_sum / n_calib  # (hidden_dim, hidden_dim)

        corrs_jac = []
        max_errs_jac = []

        for x in test_inputs:
            with torch.no_grad():
                # Full MLP output (recompute)
                gate_out = F.linear(x, W_gate)
                up_out = F.linear(x, W_up)
                activated = silu(gate_out) * up_out
                full_output = F.linear(activated, W_down)

                # Mean Jacobian output
                jac_output = J_mean @ x  # (hidden_dim,)

                full_np = full_output.numpy()
                jac_np = jac_output.numpy()

                corr_jac = np.corrcoef(full_np, jac_np)[0, 1]
                corrs_jac.append(corr_jac)
                max_errs_jac.append(np.max(np.abs(full_np - jac_np)))

        print(f"\n  Mean Jacobian (1 matmul, {hidden_dim}×{hidden_dim}):")
        print(f"    Correlation:  {np.mean(corrs_jac):.6f} (min {np.min(corrs_jac):.6f})")
        print(f"    Max error:    {np.mean(max_errs_jac):.4f}")

        # -----------------------------------------------------------
        # Low-rank Jacobian test
        # -----------------------------------------------------------
        # SVD of J_mean
        U, S, Vh = torch.linalg.svd(J_mean, full_matrices=False)
        S_np = S.numpy()

        # Find effective rank
        cumvar = np.cumsum(S_np**2) / np.sum(S_np**2)
        rank_90 = np.searchsorted(cumvar, 0.90) + 1
        rank_95 = np.searchsorted(cumvar, 0.95) + 1
        rank_99 = np.searchsorted(cumvar, 0.99) + 1

        print(f"\n  Jacobian SVD spectrum:")
        print(f"    S[0]/S[1] = {S_np[0]/S_np[1]:.3f}")
        print(f"    Rank for 90% var: {rank_90}/{hidden_dim}")
        print(f"    Rank for 95% var: {rank_95}/{hidden_dim}")
        print(f"    Rank for 99% var: {rank_99}/{hidden_dim}")

        for rank in [rank_90, rank_95, 256, 512, 1024]:
            if rank > hidden_dim:
                continue
            # Low-rank approximation
            J_lr = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]

            corrs_lr = []
            for x in test_inputs:
                with torch.no_grad():
                    gate_out = F.linear(x, W_gate)
                    up_out = F.linear(x, W_up)
                    activated = silu(gate_out) * up_out
                    full_output = F.linear(activated, W_down)

                    lr_output = J_lr @ x
                    full_np = full_output.numpy()
                    lr_np = lr_output.numpy()
                    corrs_lr.append(np.corrcoef(full_np, lr_np)[0, 1])

            print(f"    Rank {rank:4d}: corr = {np.mean(corrs_lr):.6f}")

    # ================================================================
    # Step 3: Full model end-to-end test with linearized MLP
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: End-to-end argmax test (linearized SiLU)")
    print("=" * 70)

    test_prompts = [
        ("The capital of France is", "Paris"),
        ("Water boils at", "100"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("def hello():\n    print(", '"'),
        ("1 + 1 =", "2"),
    ]

    correct_full = 0
    correct_lin = 0
    match_full_lin = 0

    # Patch strategy: hook into each MLP to replace the act_fn with linearized version
    def make_mlp_hook():
        """Hook that replaces SiLU(gate) with gate/2 inside MLP forward."""
        def hook_fn(module, input, output):
            # Recompute MLP output with linearized SiLU
            x = input[0]
            gate_out = F.linear(x, module.gate_proj.weight.data)
            up_out = F.linear(x, module.up_proj.weight.data)
            lin_activated = (gate_out / 2) * up_out
            return F.linear(lin_activated, module.down_proj.weight.data)
        return hook_fn

    for prompt, expected in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            # === FULL MODEL (reference) ===
            outputs_full = model(**inputs)
            logits_full = outputs_full.logits[0, -1]
            token_full = torch.argmax(logits_full).item()
            word_full = tokenizer.decode([token_full])

            # === LINEARIZED SiLU MODEL (via hooks) ===
            lin_hooks = []
            for layer in model.model.layers:
                h = layer.mlp.register_forward_hook(make_mlp_hook())
                lin_hooks.append(h)

            outputs_lin = model(**inputs)
            logits_lin = outputs_lin.logits[0, -1]
            token_lin = torch.argmax(logits_lin).item()
            word_lin = tokenizer.decode([token_lin])

            for h in lin_hooks:
                h.remove()

        # Check
        full_match = expected.lower() in word_full.lower().strip()
        lin_match = expected.lower() in word_lin.lower().strip()
        same_token = (token_full == token_lin)

        if full_match:
            correct_full += 1
        if lin_match:
            correct_lin += 1
        if same_token:
            match_full_lin += 1

        status_full = "✓" if full_match else "✗"
        status_lin = "✓" if lin_match else "✗"
        status_same = "=" if same_token else "≠"

        print(f"  {prompt[:40]:<40s}")
        print(f"    Full:       {word_full.strip():<20s} {status_full}")
        print(f"    Linearized: {word_lin.strip():<20s} {status_lin}  {status_same}")

    print(f"\n  Summary:")
    print(f"    Full model correct:   {correct_full}/{len(test_prompts)}")
    print(f"    Linearized correct:   {correct_lin}/{len(test_prompts)}")
    print(f"    Same argmax:          {match_full_lin}/{len(test_prompts)}")

    # ================================================================
    # Step 4: Timing comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Timing comparison (single layer MLP)")
    print("=" * 70)

    layer = model.model.layers[14]
    W_gate = layer.mlp.gate_proj.weight.data
    W_up = layer.mlp.up_proj.weight.data
    W_down = layer.mlp.down_proj.weight.data

    x = hidden_states_per_layer[14][0]

    # Time full MLP
    start = time.time()
    for _ in range(100):
        g = F.linear(x, W_gate)
        u = F.linear(x, W_up)
        a = silu(g) * u
        out = F.linear(a, W_down)
    full_time = (time.time() - start) / 100

    # Time linearized MLP
    start = time.time()
    for _ in range(100):
        g = F.linear(x, W_gate)
        u = F.linear(x, W_up)
        a = (g / 2) * u
        out = F.linear(a, W_down)
    lin_time = (time.time() - start) / 100

    # Time naive scaffold
    start = time.time()
    for _ in range(100):
        g = F.linear(x, W_gate) / 2
        out = F.linear(g, W_down)
    scaffold_time = (time.time() - start) / 100

    # Precompute J_mean for layer 14
    J_sum = torch.zeros(hidden_dim, hidden_dim)
    for xi in hidden_states_per_layer[14]:
        with torch.no_grad():
            g = F.linear(xi, W_gate)
            u = F.linear(xi, W_up)
            sp = silu_derivative(g)
            sg = silu(g)
            J = (W_down * (sp * u).unsqueeze(0)) @ W_gate + \
                (W_down * sg.unsqueeze(0)) @ W_up
            J_sum += J
    J_mean = J_sum / len(hidden_states_per_layer[14])

    # Time mean Jacobian
    start = time.time()
    for _ in range(100):
        out = J_mean @ x
    jac_time = (time.time() - start) / 100

    print(f"\n  Full MLP (3 matmuls + SiLU):  {full_time*1000:.2f}ms")
    print(f"  Linearized (3 matmuls):        {lin_time*1000:.2f}ms")
    print(f"  Naive scaffold (2 matmuls):    {scaffold_time*1000:.2f}ms")
    print(f"  Mean Jacobian (1 matmul):      {jac_time*1000:.2f}ms")
    print(f"\n  Speedups vs full:")
    print(f"    Linearized:    {full_time/lin_time:.2f}x")
    print(f"    Scaffold:      {full_time/scaffold_time:.2f}x")
    print(f"    Mean Jacobian: {full_time/jac_time:.2f}x")

    del model
    print("\nDone.")


if __name__ == "__main__":
    main()
