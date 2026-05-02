#!/usr/bin/env python3
"""
Low-Rank Gate Sign Predictor — Path A
=======================================

Finding 57 showed the MLP gate is a 4-state encoder (+1, +0, -0, -1) and that
the SIGN carries 4× more information than magnitude. The question:

  Can a low-rank approximation of W_gate predict the 4-state code?

If rank-k SVD of W_gate predicts >95% of channel states correctly, we can:
  - Replace 18944×3584 gate matmul with O(k×d) projection
  - Only compute exact gate/up for misclassified or EXPAND channels
  - Use linearized approximation for correctly-classified PRESERVE channels
  - Use low-rank correction for CONTRACT channels

Cost comparison (per token):
  Full gate:  18944 × 3584 = 67.9M multiplies
  Rank-32:    2 × 32 × max(18944, 3584) ≈ 1.2M multiplies (56× reduction)
  Rank-128:   2 × 128 × max(18944, 3584) ≈ 4.8M multiplies (14× reduction)
  Rank-256:   2 × 256 × max(18944, 3584) ≈ 9.7M multiplies (7× reduction)

This script tests:
1. SVD spectrum of W_gate — how fast do singular values decay?
2. 4-state classification accuracy vs rank k
3. Per-state accuracy (which states are hardest to predict?)
4. Binary sign accuracy (just positive vs negative)
5. End-to-end MLP quality when using low-rank gate to drive sparse computation
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)  # ≈ 0.481


def classify_gate(gate_vals, log_phi=LOG_PHI):
    """Classify gate values into 4 states: +1, +0, -0, -1."""
    states = np.zeros_like(gate_vals, dtype=np.int8)
    states[gate_vals > log_phi] = 1       # +1 EXPAND
    states[(gate_vals >= 0) & (gate_vals <= log_phi)] = 2   # +0 PRESERVE+
    states[(gate_vals < 0) & (gate_vals >= -log_phi)] = 3   # -0 PRESERVE-
    states[gate_vals < -log_phi] = 4      # -1 CONTRACT
    return states


def main():
    print("=" * 70)
    print("PATH A: Low-Rank Gate Sign Predictor")
    print("=" * 70)

    model_name = "Qwen/Qwen2-7B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "In quantum mechanics, the uncertainty principle states that",
        "The color of the sky is",
        "One plus one equals",
        "The chemical symbol for gold is",
        "To solve a quadratic equation, you can use the",
        "The speed of light in a vacuum is approximately",
    ]

    test_layers = [0, 7, 14, 21, 27]
    ranks_to_test = [8, 16, 32, 64, 128, 256, 512, 1024]

    # ================================================================
    # Step 0: Capture hidden states at each test layer
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 0: Capturing hidden states")
    print("=" * 70)

    hidden_states = {L: [] for L in test_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states[layer_idx].append(input[0].detach())
        return hook_fn

    for L in test_layers:
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(L))
        hooks.append(h)

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            model(ids)

    for h in hooks:
        h.remove()

    for L in test_layers:
        hidden_states[L] = torch.cat(hidden_states[L], dim=1).squeeze(0)
        print(f"  Layer {L:2d}: {hidden_states[L].shape[0]} tokens × {hidden_states[L].shape[1]} dims")

    # ================================================================
    # Step 1: SVD spectrum of W_gate
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: SVD spectrum of W_gate")
    print("=" * 70)

    gate_svd = {}
    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data  # (18944, 3584)
        t0 = time.time()
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)
        svd_time = time.time() - t0

        S_np = S.numpy()
        gate_svd[L] = (U, S, Vt)

        # Cumulative variance
        var_cum = np.cumsum(S_np**2) / np.sum(S_np**2)
        rank90 = np.searchsorted(var_cum, 0.90) + 1
        rank95 = np.searchsorted(var_cum, 0.95) + 1
        rank99 = np.searchsorted(var_cum, 0.99) + 1

        print(f"\n  Layer {L:2d}: W_gate shape {tuple(W_gate.shape)}, SVD in {svd_time:.1f}s")
        print(f"    S[0]={S_np[0]:.3f}, S[1]={S_np[1]:.3f}, ratio={S_np[0]/S_np[1]:.3f}")
        print(f"    S[0]/S[-1] = {S_np[0]/S_np[-1]:.1f} (condition number)")
        print(f"    Rank for 90% var: {rank90}/{len(S_np)}")
        print(f"    Rank for 95% var: {rank95}/{len(S_np)}")
        print(f"    Rank for 99% var: {rank99}/{len(S_np)}")
        print(f"    First 10 S: {' '.join(f'{s:.2f}' for s in S_np[:10])}")

    # ================================================================
    # Step 2: 4-state classification accuracy vs rank k
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: 4-state classification accuracy vs rank k")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        U, S, Vt = gate_svd[L]
        X = hidden_states[L]  # (n_tokens, 3584)

        # Full gate output
        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()  # (n_tokens, 18944)

        states_full = classify_gate(gate_full)
        sign_full = (gate_full >= 0).astype(np.int8)  # binary: positive vs negative

        print(f"\n  Layer {L:2d}:")
        print(f"    {'Rank':>6s}  {'4-state%':>9s}  {'Sign%':>7s}  {'+1 acc':>7s}  {'+0 acc':>7s}  {'-0 acc':>7s}  {'-1 acc':>7s}  {'Speedup':>8s}")
        print(f"    {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")

        for k in ranks_to_test:
            if k > min(W_gate.shape):
                continue

            # Low-rank gate: W_gate_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
            # gate_k = X @ Vt[:k,:].T @ diag(S[:k]) @ U[:,:k].T
            # But more efficiently: project = X @ Vt[:k,:].T  (n_tokens, k)
            # then gate_k = project @ diag(S[:k]) @ U[:,:k].T  (n_tokens, 18944)
            with torch.no_grad():
                # Efficient: first project to k dims, then expand
                projected = X @ Vt[:k, :].T  # (n_tokens, k)
                gate_k = (projected * S[:k].unsqueeze(0)) @ U[:, :k].T  # (n_tokens, 18944)
                gate_k = gate_k.numpy()

            states_k = classify_gate(gate_k)
            sign_k = (gate_k >= 0).astype(np.int8)

            # Overall accuracy
            acc_4state = np.mean(states_full == states_k) * 100
            acc_sign = np.mean(sign_full == sign_k) * 100

            # Per-state accuracy
            per_state_acc = {}
            for state_val, state_name in [(1, "+1"), (2, "+0"), (3, "-0"), (4, "-1")]:
                mask = (states_full == state_val)
                if mask.sum() > 0:
                    per_state_acc[state_name] = np.mean(states_k[mask] == state_val) * 100
                else:
                    per_state_acc[state_name] = float('nan')

            # Speedup estimate
            full_ops = W_gate.shape[0] * W_gate.shape[1]  # 18944 × 3584
            lowrank_ops = k * (W_gate.shape[0] + W_gate.shape[1])
            speedup = full_ops / lowrank_ops

            print(f"    {k:6d}  {acc_4state:8.2f}%  {acc_sign:6.2f}%  "
                  f"{per_state_acc.get('+1', float('nan')):6.1f}%  "
                  f"{per_state_acc.get('+0', float('nan')):6.1f}%  "
                  f"{per_state_acc.get('-0', float('nan')):6.1f}%  "
                  f"{per_state_acc.get('-1', float('nan')):6.1f}%  "
                  f"{speedup:7.1f}×")

    # ================================================================
    # Step 3: Where do misclassifications concentrate?
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Misclassification analysis at rank-128")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        U, S, Vt = gate_svd[L]
        X = hidden_states[L]

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            projected = X @ Vt[:128, :].T
            gate_128 = (projected * S[:128].unsqueeze(0)) @ U[:, :128].T
            gate_128 = gate_128.numpy()

        states_full = classify_gate(gate_full)
        states_128 = classify_gate(gate_128)

        misclass = (states_full != states_128)
        gate_mag_at_misclass = np.abs(gate_full[misclass])

        # Are misclassifications near the boundaries?
        near_boundary = np.abs(np.abs(gate_full) - LOG_PHI)
        near_zero = np.abs(gate_full)

        print(f"\n  Layer {L:2d}:")
        print(f"    Total misclassifications: {misclass.sum()}/{misclass.size} ({np.mean(misclass)*100:.2f}%)")
        if misclass.sum() > 0:
            print(f"    |gate| at misclass:  median={np.median(gate_mag_at_misclass):.3f}, "
                  f"mean={np.mean(gate_mag_at_misclass):.3f}, "
                  f"P90={np.percentile(gate_mag_at_misclass, 90):.3f}")
            print(f"    Distance to ±log(φ) boundary at misclass: "
                  f"median={np.median(near_boundary[misclass]):.3f}")
            print(f"    Distance to zero at misclass: "
                  f"median={np.median(near_zero[misclass]):.3f}")

            # Confusion: what states are confused with what?
            confusion = {}
            for true_s in [1, 2, 3, 4]:
                for pred_s in [1, 2, 3, 4]:
                    if true_s != pred_s:
                        mask = (states_full == true_s) & (states_128 == pred_s)
                        count = mask.sum()
                        if count > 0:
                            names = {1: "+1", 2: "+0", 3: "-0", 4: "-1"}
                            confusion[(names[true_s], names[pred_s])] = count

            if confusion:
                sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])[:5]
                print(f"    Top confusions:")
                for (true_s, pred_s), count in sorted_conf:
                    pct = count / misclass.sum() * 100
                    print(f"      {true_s} → {pred_s}: {count} ({pct:.1f}% of errors)")

    # ================================================================
    # Step 4: Sign-only accuracy (the critical metric)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Binary sign accuracy — minimum rank for >99% sign correctness")
    print("=" * 70)

    fine_ranks = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        U, S, Vt = gate_svd[L]
        X = hidden_states[L]

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()

        sign_full = (gate_full >= 0)

        print(f"\n  Layer {L:2d}:")
        threshold_ranks = {}

        for k in fine_ranks:
            if k > min(W_gate.shape):
                continue

            with torch.no_grad():
                projected = X @ Vt[:k, :].T
                gate_k = (projected * S[:k].unsqueeze(0)) @ U[:, :k].T
                gate_k = gate_k.numpy()

            sign_k = (gate_k >= 0)
            acc = np.mean(sign_full == sign_k) * 100

            marker = ""
            for thresh in [95, 99, 99.5, 99.9]:
                if acc >= thresh and thresh not in threshold_ranks:
                    threshold_ranks[thresh] = k
                    marker += f" ← >{thresh}%"

            full_ops = W_gate.shape[0] * W_gate.shape[1]
            lowrank_ops = k * (W_gate.shape[0] + W_gate.shape[1])
            speedup = full_ops / lowrank_ops

            print(f"    rank {k:4d}: sign accuracy {acc:7.3f}%  (speedup {speedup:6.1f}×){marker}")

        print(f"    Summary:")
        for thresh in [95, 99, 99.5, 99.9]:
            if thresh in threshold_ranks:
                k = threshold_ranks[thresh]
                full_ops = W_gate.shape[0] * W_gate.shape[1]
                lowrank_ops = k * (W_gate.shape[0] + W_gate.shape[1])
                speedup = full_ops / lowrank_ops
                print(f"      >{thresh}% sign accuracy at rank {k} ({speedup:.1f}× speedup)")
            else:
                print(f"      >{thresh}% sign accuracy: NOT REACHED in tested ranks")

    # ================================================================
    # Step 5: End-to-end test — low-rank gate predictor + sparse MLP
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: End-to-end — low-rank gate code drives sparse MLP")
    print("=" * 70)

    # Pick a rank that gives good sign accuracy (will pick based on step 4 results)
    # For now test rank-128 and rank-256
    test_e2e_ranks = [128, 256]

    test_prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "The color of the sky is",
        "One plus one equals",
        "The chemical symbol for gold is",
    ]

    for test_rank in test_e2e_ranks:
        print(f"\n  === Testing rank-{test_rank} gate predictor end-to-end ===")

        # Precompute low-rank factors for all layers
        lowrank_factors = {}
        for layer_idx in range(model.config.num_hidden_layers):
            W_gate = model.model.layers[layer_idx].mlp.gate_proj.weight.data
            U, S_vals, Vt = torch.linalg.svd(W_gate, full_matrices=False)
            # Store: V_k (3584, k), S_k (k,), U_k (18944, k)
            lowrank_factors[layer_idx] = (
                Vt[:test_rank, :].T.contiguous(),   # (3584, k)
                S_vals[:test_rank].contiguous(),     # (k,)
                U[:, :test_rank].contiguous(),       # (18944, k)
            )

        def make_sparse_mlp_hook(layer_idx, rank):
            """Hook: use low-rank gate to predict 4-state code, then sparse compute."""
            V_k, S_k, U_k = lowrank_factors[layer_idx]

            def hook_fn(module, input, output):
                x = input[0]  # (batch, seq, hidden)
                shape = x.shape
                x_flat = x.view(-1, shape[-1])  # (B*S, hidden)

                # 1. Low-rank gate prediction (cheap)
                projected = x_flat @ V_k  # (B*S, k)
                gate_approx = (projected * S_k.unsqueeze(0)) @ U_k.T  # (B*S, 18944)

                # 2. Classify into 4 states
                log_phi = LOG_PHI
                expand_mask = gate_approx > log_phi
                preserve_pos_mask = (gate_approx >= 0) & (gate_approx <= log_phi)
                preserve_neg_mask = (gate_approx < 0) & (gate_approx >= -log_phi)
                contract_mask = gate_approx < -log_phi

                # 3. Compute full gate and up for ALL channels (we need exact values)
                #    But use the predicted CODE to determine approximation strategy
                gate_out = F.linear(x_flat, module.gate_proj.weight.data)
                up_out = F.linear(x_flat, module.up_proj.weight.data)

                # 4. Apply SiLU with ternary approximation based on predicted code
                activated = torch.zeros_like(gate_out)

                # EXPAND: exact SiLU
                if expand_mask.any():
                    activated[expand_mask] = F.silu(gate_out[expand_mask])

                # PRESERVE+: linearized SiLU(g) ≈ g/2
                if preserve_pos_mask.any():
                    activated[preserve_pos_mask] = gate_out[preserve_pos_mask] / 2

                # PRESERVE-: linearized SiLU(g) ≈ g/2 (preserves sign!)
                if preserve_neg_mask.any():
                    activated[preserve_neg_mask] = gate_out[preserve_neg_mask] / 2

                # CONTRACT: exact SiLU (negative zero contribution is essential)
                if contract_mask.any():
                    activated[contract_mask] = F.silu(gate_out[contract_mask])

                # 5. Combine and project
                intermediate = activated * up_out
                result = F.linear(intermediate, module.down_proj.weight.data)
                return result.view(shape)

            return hook_fn

        # Run end-to-end with hooks
        for prompt in test_prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids

            # Baseline: full model
            with torch.no_grad():
                logits_full = model(ids).logits[0, -1, :]
                top_full = torch.argmax(logits_full).item()
                tok_full = tokenizer.decode([top_full])

            # Hook: sparse MLP
            hooks = []
            for layer_idx in range(model.config.num_hidden_layers):
                h = model.model.layers[layer_idx].mlp.register_forward_hook(
                    make_sparse_mlp_hook(layer_idx, test_rank)
                )
                hooks.append(h)

            with torch.no_grad():
                logits_sparse = model(ids).logits[0, -1, :]
                top_sparse = torch.argmax(logits_sparse).item()
                tok_sparse = tokenizer.decode([top_sparse])

            for h in hooks:
                h.remove()

            match = "✓" if top_full == top_sparse else "✗"
            corr = torch.corrcoef(torch.stack([logits_full, logits_sparse]))[0, 1].item()
            print(f"    {match} \"{prompt}\" → full={tok_full!r}, sparse={tok_sparse!r}, "
                  f"logit_corr={corr:.6f}")

    # ================================================================
    # Step 6: Timing comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Timing — low-rank gate prediction vs full gate matmul")
    print("=" * 70)

    L = 14
    W_gate = model.model.layers[L].mlp.gate_proj.weight.data
    X = hidden_states[L][:1]  # single token

    # Full gate
    times_full = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            _ = F.linear(X, W_gate)
        times_full.append(time.time() - t0)

    # Low-rank gate (rank 128)
    V_k, S_k, U_k = lowrank_factors[L]
    times_lr = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            proj = X @ V_k
            gate_approx = (proj * S_k.unsqueeze(0)) @ U_k.T
        times_lr.append(time.time() - t0)

    t_full = np.median(times_full) * 1000
    t_lr = np.median(times_lr) * 1000

    print(f"\n  Layer {L}, single token:")
    print(f"    Full gate matmul (18944×3584): {t_full:.2f} ms")
    print(f"    Low-rank gate (rank {test_rank}):     {t_lr:.2f} ms")
    print(f"    Speedup: {t_full/t_lr:.1f}×")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
