#!/usr/bin/env python3
"""
Sparse MLP Test — Eliminate Invalid Positions (Tetromino + rhzeros approach)
============================================================================

From our findings:
- 28-85% of MLP channels are gated off (CONTRACT region)
- Phase 17: Bias predicts default gate pattern with 98-100% accuracy
- rhzeros: cached derivative works because ζ' changes slowly → cache what's stable
- sublinear_clock: predict what matters, only compute that

Strategy: Instead of computing ALL 18944 intermediate channels:
1. Use bias to predict which channels will be active (FREE — bias is known)
2. Only compute gate/up for predicted-active channels (SPARSE matmul)
3. Only compute down_proj from active channels (SPARSE matmul)

Also test: cached Jacobian between adjacent tokens (rhzeros analog)

This is the "eliminate invalid positions like tetrominoes" approach.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

PHI = (1 + np.sqrt(5)) / 2


def silu(x):
    return x * torch.sigmoid(x)


def main():
    print("=" * 70)
    print("SPARSE MLP — Eliminate Invalid Positions")
    print("=" * 70)

    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    hidden_dim = model.config.hidden_size       # 3584
    inter_dim = model.config.intermediate_size   # 18944

    # ================================================================
    # Step 1: Capture real MLP inputs from a long prompt
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Capturing real MLP inputs")
    print("=" * 70)

    prompt = (
        "The Riemann hypothesis states that all non-trivial zeros of the "
        "Riemann zeta function have real part equal to one half. This is one "
        "of the most important unsolved problems in mathematics."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt tokens: {seq_len}")

    test_layers = [0, 7, 14, 21, 27]

    # Capture ALL token positions' MLP inputs per layer
    mlp_inputs_per_layer = {L: [] for L in test_layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0] shape: (batch, seq_len, hidden_dim)
            x = input[0][0].detach()  # (seq_len, hidden_dim)
            for pos in range(x.shape[0]):
                mlp_inputs_per_layer[layer_idx].append(x[pos].clone())
        return hook_fn

    hooks = []
    for L in test_layers:
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(L))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    for L in test_layers:
        print(f"  Layer {L:2d}: {len(mlp_inputs_per_layer[L])} token positions")

    # ================================================================
    # Step 2: Analyze gate sparsity and bias prediction
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Gate sparsity & bias prediction accuracy")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        test_inputs = mlp_inputs_per_layer[layer_idx]

        # Analyze gate patterns
        all_gate_signs = []
        all_active_masks = []

        for x in test_inputs:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                activated = silu(gate_out)

                # "Active" = |SiLU(gate)| > threshold
                # SiLU(x) ≈ 0 when x << 0
                active_mask = (gate_out > -2.0)  # channels likely to contribute
                all_gate_signs.append((gate_out > 0).numpy())
                all_active_masks.append(active_mask.numpy())

        gate_signs = np.array(all_gate_signs)  # (n_tokens, inter_dim)
        active_masks = np.array(all_active_masks)

        # Default pattern from bias (no input needed)
        # For Qwen2, gate_proj has no bias, but the distribution is pushed
        # negative by the weight structure. Use the mean gate output as proxy.
        mean_gate = np.zeros(inter_dim)
        for x in test_inputs:
            with torch.no_grad():
                mean_gate += F.linear(x, W_gate).numpy()
        mean_gate /= len(test_inputs)

        # Bias prediction: predict sign from mean gate output
        bias_prediction = (mean_gate > 0)  # predicted "active" channels

        # Per-token accuracy of bias prediction
        accuracies = []
        for signs in gate_signs:
            acc = np.mean(signs == bias_prediction)
            accuracies.append(acc)

        avg_active = np.mean(active_masks) * 100
        avg_positive = np.mean(gate_signs) * 100
        bias_acc = np.mean(accuracies) * 100

        # Channel stability: how many channels ALWAYS have the same sign?
        always_same = np.all(gate_signs == gate_signs[0], axis=0)
        pct_stable = np.mean(always_same) * 100

        # How many channels are ALWAYS negative (never fire)?
        always_neg = np.all(~gate_signs, axis=0)
        pct_always_dead = np.mean(always_neg) * 100

        # How many are ALWAYS positive (always fire)?
        always_pos = np.all(gate_signs, axis=0)
        pct_always_alive = np.mean(always_pos) * 100

        # "Volatile" channels that switch
        pct_volatile = 100 - pct_always_dead - pct_always_alive

        print(f"\n  Layer {layer_idx}:")
        print(f"    Avg channels positive (gate > 0): {avg_positive:.1f}%")
        print(f"    Avg channels active (gate > -2):  {avg_active:.1f}%")
        print(f"    Always dead (all tokens):         {pct_always_dead:.1f}%")
        print(f"    Always alive (all tokens):        {pct_always_alive:.1f}%")
        print(f"    Volatile (input-dependent):       {pct_volatile:.1f}%")
        print(f"    Mean-gate prediction accuracy:    {bias_acc:.1f}%")
        print(f"    Channel stability:                {pct_stable:.1f}%")

    # ================================================================
    # Step 3: Sparse MLP — only compute active channels
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Sparse MLP — skip dead channels")
    print("=" * 70)

    for layer_idx in test_layers:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        test_inputs = mlp_inputs_per_layer[layer_idx]

        print(f"\n  Layer {layer_idx}:")

        # Determine always-dead channels (across all tokens for this prompt)
        all_positive = []
        for x in test_inputs:
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)
                all_positive.append((gate_out > 0).numpy())

        gate_patterns = np.array(all_positive)
        always_dead = np.all(~gate_patterns, axis=0)
        always_alive = np.all(gate_patterns, axis=0)
        volatile = ~always_dead & ~always_alive

        n_dead = np.sum(always_dead)
        n_alive = np.sum(always_alive)
        n_volatile = np.sum(volatile)

        print(f"    Channels: {n_dead} dead + {n_alive} alive + {n_volatile} volatile = {inter_dim}")

        # Test 1: Skip always-dead channels (oracle — knows which are dead)
        active_idx = torch.tensor(np.where(~always_dead)[0])
        n_active = len(active_idx)

        # Extract sub-matrices for active channels only
        W_gate_active = W_gate[active_idx]  # (n_active, hidden_dim)
        W_up_active = W_up[active_idx]      # (n_active, hidden_dim)
        W_down_active = W_down[:, active_idx]  # (hidden_dim, n_active)

        corrs_sparse = []
        for x in test_inputs:
            with torch.no_grad():
                # Full MLP
                g_full = F.linear(x, W_gate)
                u_full = F.linear(x, W_up)
                act_full = silu(g_full) * u_full
                out_full = F.linear(act_full, W_down)

                # Sparse MLP (skip dead channels)
                g_sparse = F.linear(x, W_gate_active)
                u_sparse = F.linear(x, W_up_active)
                act_sparse = silu(g_sparse) * u_sparse
                out_sparse = F.linear(act_sparse, W_down_active)

                corr = np.corrcoef(out_full.numpy(), out_sparse.numpy())[0, 1]
                corrs_sparse.append(corr)

        pct_compute = n_active / inter_dim * 100
        speedup_est = inter_dim / n_active

        print(f"    Skip-dead: {n_active}/{inter_dim} channels ({pct_compute:.1f}%)")
        print(f"    Correlation: {np.mean(corrs_sparse):.6f} (min {np.min(corrs_sparse):.6f})")
        print(f"    Estimated speedup: {speedup_est:.1f}×")

        # Test 2: More aggressive — skip channels where gate < -1 for ALL tokens
        all_gates = []
        for x in test_inputs:
            with torch.no_grad():
                all_gates.append(F.linear(x, W_gate).numpy())
        all_gates = np.array(all_gates)  # (n_tokens, inter_dim)

        for threshold in [-1.0, -2.0, -3.0, -5.0]:
            below_thresh = np.all(all_gates < threshold, axis=0)
            active_mask = ~below_thresh
            n_active_t = np.sum(active_mask)
            active_idx_t = torch.tensor(np.where(active_mask)[0])

            W_g_t = W_gate[active_idx_t]
            W_u_t = W_up[active_idx_t]
            W_d_t = W_down[:, active_idx_t]

            corrs_t = []
            for x in test_inputs:
                with torch.no_grad():
                    g_full = F.linear(x, W_gate)
                    u_full = F.linear(x, W_up)
                    out_full = F.linear(silu(g_full) * u_full, W_down)

                    g_t = F.linear(x, W_g_t)
                    u_t = F.linear(x, W_u_t)
                    out_t = F.linear(silu(g_t) * u_t, W_d_t)

                    corrs_t.append(np.corrcoef(out_full.numpy(), out_t.numpy())[0, 1])

            pct = n_active_t / inter_dim * 100
            print(f"    gate < {threshold:5.1f} pruned: {n_active_t}/{inter_dim} ({pct:.1f}%), "
                  f"corr = {np.mean(corrs_t):.6f}")

    # ================================================================
    # Step 4: Cached Jacobian between adjacent tokens (rhzeros analog)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Cached Jacobian — rhzeros analog")
    print("=" * 70)
    print("  (Does J(x_i) ≈ J(x_{i-1}) for adjacent tokens?)")

    for layer_idx in [0, 14, 27]:
        layer = model.model.layers[layer_idx]
        W_gate = layer.mlp.gate_proj.weight.data
        W_up = layer.mlp.up_proj.weight.data
        W_down = layer.mlp.down_proj.weight.data

        test_inputs = mlp_inputs_per_layer[layer_idx]

        if len(test_inputs) < 3:
            continue

        def compute_jacobian(x):
            """Compute MLP Jacobian at input x."""
            with torch.no_grad():
                g = F.linear(x, W_gate)
                u = F.linear(x, W_up)
                sp = torch.sigmoid(g) * (1 + g * (1 - torch.sigmoid(g)))  # SiLU'
                sg = silu(g)
                # J = W_down @ [diag(sp*u) @ W_gate + diag(sg) @ W_up]
                J = (W_down * (sp * u).unsqueeze(0)) @ W_gate + \
                    (W_down * sg.unsqueeze(0)) @ W_up
                return J

        # Compute Jacobian at each token position
        # Compare J(token_i) vs J(token_{i+1})
        corrs_cached = []
        corrs_self = []

        prev_J = None
        for i, x in enumerate(test_inputs[:min(20, len(test_inputs))]):
            with torch.no_grad():
                # Full MLP output (reference)
                g = F.linear(x, W_gate)
                u = F.linear(x, W_up)
                out_full = F.linear(silu(g) * u, W_down)

                # Current Jacobian
                J_curr = compute_jacobian(x)

                # Self-check: J_curr @ x should approximate MLP output
                out_self = J_curr @ x
                corr_self = np.corrcoef(out_full.numpy(), out_self.numpy())[0, 1]
                corrs_self.append(corr_self)

                # Cached Jacobian from previous token
                if prev_J is not None:
                    out_cached = prev_J @ x
                    corr_cached = np.corrcoef(out_full.numpy(), out_cached.numpy())[0, 1]
                    corrs_cached.append(corr_cached)

                prev_J = J_curr

        print(f"\n  Layer {layer_idx}:")
        print(f"    J(x_i) @ x_i correlation (self):     {np.mean(corrs_self):.6f}")
        print(f"    J(x_{'{i-1}'}) @ x_i correlation (cached): {np.mean(corrs_cached):.6f}")
        print(f"    Degradation from caching:             {np.mean(corrs_self) - np.mean(corrs_cached):.6f}")

    # ================================================================
    # Step 5: End-to-end sparse MLP test
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: End-to-end test with sparse MLP")
    print("=" * 70)

    # Strategy: for each layer, determine always-dead channels from a
    # calibration pass, then test with sparse MLP on new prompts

    # Calibration: use first prompt to determine dead channels per layer
    print("  Calibrating dead channels from prompt...")

    dead_channels = {}
    calib_hooks = []

    def make_calib_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0][0]  # (seq_len, hidden_dim)
            W_gate = module.gate_proj.weight.data
            with torch.no_grad():
                gate_out = F.linear(x, W_gate)  # (seq_len, inter_dim)
                # A channel is "dead" if gate < threshold for ALL positions
                max_gate_per_channel = gate_out.max(dim=0).values
                dead_channels[layer_idx] = (max_gate_per_channel < -2.0).numpy()
        return hook_fn

    for L in range(28):
        h = model.model.layers[L].mlp.register_forward_hook(make_calib_hook(L))
        calib_hooks.append(h)

    calib_prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "In mathematics, the golden ratio appears everywhere. "
        "Machine learning models process text sequentially."
    )
    calib_inputs = tokenizer(calib_prompt, return_tensors="pt")
    with torch.no_grad():
        model(**calib_inputs)

    for h in calib_hooks:
        h.remove()

    total_dead = 0
    total_channels = 0
    for L in range(28):
        n_dead = np.sum(dead_channels[L])
        total_dead += n_dead
        total_channels += inter_dim
        if L in [0, 7, 14, 21, 27]:
            pct = n_dead / inter_dim * 100
            print(f"    Layer {L:2d}: {n_dead}/{inter_dim} dead ({pct:.1f}%)")

    print(f"    Overall: {total_dead}/{total_channels} dead ({total_dead/total_channels*100:.1f}%)")

    # Now test with sparse MLP hooks
    def make_sparse_hook(layer_idx):
        dead_mask = dead_channels[layer_idx]
        active_mask = ~dead_mask
        active_idx = torch.tensor(np.where(active_mask)[0])

        def hook_fn(module, input, output):
            x = input[0]  # (batch, seq, hidden)
            W_g = module.gate_proj.weight.data[active_idx]
            W_u = module.up_proj.weight.data[active_idx]
            W_d = module.down_proj.weight.data[:, active_idx]

            with torch.no_grad():
                g = F.linear(x, W_g)
                u = F.linear(x, W_u)
                return F.linear(silu(g) * u, W_d)
        return hook_fn

    test_prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("def hello():\n    print(", '"'),
        ("1 + 1 =", "2"),
        ("Water freezes at", "0"),
    ]

    correct_full = 0
    correct_sparse = 0
    match_count = 0

    for prompt, expected in test_prompts:
        test_inp = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            # Full model
            out_full = model(**test_inp)
            token_full = torch.argmax(out_full.logits[0, -1]).item()
            word_full = tokenizer.decode([token_full])

            # Sparse model
            sparse_hooks = []
            for L in range(28):
                h = model.model.layers[L].mlp.register_forward_hook(
                    make_sparse_hook(L)
                )
                sparse_hooks.append(h)

            out_sparse = model(**test_inp)
            token_sparse = torch.argmax(out_sparse.logits[0, -1]).item()
            word_sparse = tokenizer.decode([token_sparse])

            for h in sparse_hooks:
                h.remove()

        full_ok = expected.lower() in word_full.lower().strip()
        sparse_ok = expected.lower() in word_sparse.lower().strip()
        same = token_full == token_sparse

        if full_ok: correct_full += 1
        if sparse_ok: correct_sparse += 1
        if same: match_count += 1

        sf = "✓" if full_ok else "✗"
        ss = "✓" if sparse_ok else "✗"
        sm = "=" if same else "≠"

        print(f"  {prompt[:40]:<40s}")
        print(f"    Full:   {word_full.strip():<20s} {sf}")
        print(f"    Sparse: {word_sparse.strip():<20s} {ss}  {sm}")

    print(f"\n  Summary:")
    print(f"    Full correct:   {correct_full}/{len(test_prompts)}")
    print(f"    Sparse correct: {correct_sparse}/{len(test_prompts)}")
    print(f"    Same argmax:    {match_count}/{len(test_prompts)}")

    # ================================================================
    # Step 6: Timing estimate
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Timing — sparse vs full")
    print("=" * 70)

    layer = model.model.layers[14]
    W_gate = layer.mlp.gate_proj.weight.data
    W_up = layer.mlp.up_proj.weight.data
    W_down = layer.mlp.down_proj.weight.data

    dead = dead_channels[14]
    active = torch.tensor(np.where(~dead)[0])
    W_g_s = W_gate[active]
    W_u_s = W_up[active]
    W_d_s = W_down[:, active]

    x = mlp_inputs_per_layer[14][0]

    # Full
    start = time.time()
    for _ in range(100):
        g = F.linear(x, W_gate)
        u = F.linear(x, W_up)
        out = F.linear(silu(g) * u, W_down)
    t_full = (time.time() - start) / 100

    # Sparse
    start = time.time()
    for _ in range(100):
        g = F.linear(x, W_g_s)
        u = F.linear(x, W_u_s)
        out = F.linear(silu(g) * u, W_d_s)
    t_sparse = (time.time() - start) / 100

    n_active = len(active)
    print(f"\n  Layer 14: {n_active}/{inter_dim} active channels ({n_active/inter_dim*100:.1f}%)")
    print(f"  Full MLP:   {t_full*1000:.2f}ms")
    print(f"  Sparse MLP: {t_sparse*1000:.2f}ms")
    print(f"  Speedup:    {t_full/t_sparse:.2f}×")

    del model
    print("\nDone.")


if __name__ == "__main__":
    main()
