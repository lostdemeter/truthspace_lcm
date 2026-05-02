#!/usr/bin/env python3
"""
Gate Sign Structure — Why Does Low-Rank Get 94%?
==================================================

Finding 58 showed low-rank gate prediction gets 77-94% sign accuracy but never 95%.
The user's intuition: 94% isn't random — there must be a deeper relationship.

This script investigates:
1. Trivial baseline: what accuracy does "always predict majority sign" give?
2. Bias baseline: what accuracy does the per-channel mean sign give?
3. Per-channel analysis: which channels are predictable vs unpredictable?
4. Margin analysis: does the low-rank prediction have high confidence for correct predictions?
5. Volatile vs stable: does low-rank succeed on stable channels and fail on volatile?
6. What's SPECIAL about the correctly-predicted channels?
7. The residual: what structure exists in the ERRORS?
8. Does the sign have φ-structure? (e.g., channels whose sign flips cluster at φ-boundaries)
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def main():
    print("=" * 70)
    print("INVESTIGATING: Why Does Low-Rank Gate Get 94%?")
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

    # ================================================================
    # Capture hidden states
    # ================================================================
    print("\nCapturing hidden states...")
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
        print(f"  Layer {L:2d}: {hidden_states[L].shape[0]} tokens")

    # ================================================================
    # Test 1: Baselines — Trivial, Bias, and Low-Rank
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Baselines — Trivial vs Bias vs Low-Rank")
    print("=" * 70)
    print("\nFor each layer, compare:")
    print("  Trivial: predict ALL channels as the majority sign")
    print("  Bias:    predict each channel by its mean sign across tokens")
    print("  Rank-4:  SVD rank-4 approximation")
    print("  Rank-32: SVD rank-32 approximation")
    print("  Rank-128: SVD rank-128 approximation")

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]

        # Full gate
        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()  # (n_tokens, 18944)

        sign_full = (gate_full >= 0)  # True = positive
        n_tokens, n_channels = sign_full.shape

        # Trivial baseline: predict majority sign globally
        frac_positive = sign_full.mean()
        trivial_acc = max(frac_positive, 1 - frac_positive) * 100

        # Bias baseline: per-channel majority sign
        channel_frac_pos = sign_full.mean(axis=0)  # fraction of tokens where channel is positive
        bias_prediction = (channel_frac_pos >= 0.5)  # predict positive if majority is positive
        bias_correct = (sign_full == bias_prediction[np.newaxis, :])
        bias_acc = bias_correct.mean() * 100

        # Channel-level bias strength
        channel_bias_strength = np.abs(channel_frac_pos - 0.5)  # 0 = balanced, 0.5 = always same sign
        strongly_biased = (channel_bias_strength > 0.4).sum()
        moderately_biased = ((channel_bias_strength > 0.1) & (channel_bias_strength <= 0.4)).sum()
        weakly_biased = (channel_bias_strength <= 0.1).sum()

        # SVD
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        lr_accs = {}
        for k in [4, 32, 128]:
            with torch.no_grad():
                projected = X @ Vt[:k, :].T
                gate_k = (projected * S[:k].unsqueeze(0)) @ U[:, :k].T
                gate_k = gate_k.numpy()
            sign_k = (gate_k >= 0)
            lr_accs[k] = np.mean(sign_full == sign_k) * 100

        # Low-rank ABOVE bias: how much does rank-k add over just knowing the bias?
        # For channels where bias predicts correctly, rank-k doesn't help
        # For channels where bias is wrong, does rank-k fix them?

        print(f"\n  Layer {L:2d}:")
        print(f"    Fraction positive:  {frac_positive*100:.1f}%")
        print(f"    Trivial baseline:   {trivial_acc:.1f}%")
        print(f"    Bias baseline:      {bias_acc:.1f}%")
        print(f"    Rank-4:             {lr_accs[4]:.1f}%")
        print(f"    Rank-32:            {lr_accs[32]:.1f}%")
        print(f"    Rank-128:           {lr_accs[128]:.1f}%")
        print(f"    Channel bias strength:")
        print(f"      Strongly biased (>0.4):   {strongly_biased:6d} / {n_channels} ({strongly_biased/n_channels*100:.1f}%)")
        print(f"      Moderately biased (0.1-0.4): {moderately_biased:6d} / {n_channels} ({moderately_biased/n_channels*100:.1f}%)")
        print(f"      Weakly biased (<0.1):     {weakly_biased:6d} / {n_channels} ({weakly_biased/n_channels*100:.1f}%)")
        print(f"    Improvement over bias:")
        print(f"      Rank-4:    {lr_accs[4] - bias_acc:+.2f} pts")
        print(f"      Rank-32:   {lr_accs[32] - bias_acc:+.2f} pts")
        print(f"      Rank-128:  {lr_accs[128] - bias_acc:+.2f} pts")

    # ================================================================
    # Test 2: Per-Channel Accuracy Decomposition
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Per-Channel Sign Accuracy (Rank-128)")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            projected = X @ Vt[:128, :].T
            gate_128 = (projected * S[:128].unsqueeze(0)) @ U[:, :128].T
            gate_128 = gate_128.numpy()

        sign_full = (gate_full >= 0)
        sign_128 = (gate_128 >= 0)

        # Per-channel accuracy
        per_channel_acc = np.mean(sign_full == sign_128, axis=0)  # (18944,)

        # Categorize channels by accuracy
        perfect = (per_channel_acc == 1.0).sum()
        high = ((per_channel_acc >= 0.9) & (per_channel_acc < 1.0)).sum()
        medium = ((per_channel_acc >= 0.7) & (per_channel_acc < 0.9)).sum()
        low = ((per_channel_acc >= 0.5) & (per_channel_acc < 0.7)).sum()
        terrible = (per_channel_acc < 0.5).sum()

        # Correlation with gate magnitude: are near-zero gates harder to predict?
        mean_gate_mag = np.mean(np.abs(gate_full), axis=0)
        from scipy import stats
        corr_mag_acc, p_mag = stats.pearsonr(mean_gate_mag, per_channel_acc)

        # Correlation with channel bias strength
        channel_frac_pos = sign_full.mean(axis=0)
        channel_bias = np.abs(channel_frac_pos - 0.5)
        corr_bias_acc, p_bias = stats.pearsonr(channel_bias, per_channel_acc)

        print(f"\n  Layer {L:2d}: Per-channel rank-128 sign accuracy")
        print(f"    Perfect (100%):   {perfect:6d} ({perfect/18944*100:.1f}%)")
        print(f"    High (90-99%):    {high:6d} ({high/18944*100:.1f}%)")
        print(f"    Medium (70-89%):  {medium:6d} ({medium/18944*100:.1f}%)")
        print(f"    Low (50-69%):     {low:6d} ({low/18944*100:.1f}%)")
        print(f"    Terrible (<50%):  {terrible:6d} ({terrible/18944*100:.1f}%)")
        print(f"    Corr(|gate|, accuracy):     r={corr_mag_acc:.3f} (p={p_mag:.2e})")
        print(f"    Corr(bias_strength, accuracy): r={corr_bias_acc:.3f} (p={p_bias:.2e})")

    # ================================================================
    # Test 3: Margin Analysis — Confidence of Predictions
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Margin Analysis — How Confident Are Correct vs Wrong Predictions?")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            projected = X @ Vt[:128, :].T
            gate_128 = (projected * S[:128].unsqueeze(0)) @ U[:, :128].T
            gate_128 = gate_128.numpy()

        sign_full = (gate_full >= 0)
        sign_128 = (gate_128 >= 0)
        correct = (sign_full == sign_128)

        # "Margin" = how far the approximate gate is from zero (confidence in its prediction)
        margin = np.abs(gate_128)

        # Also: where are the REAL gate values for misclassified channels?
        real_gate_at_error = np.abs(gate_full[~correct])
        approx_gate_at_error = np.abs(gate_128[~correct])
        real_gate_at_correct = np.abs(gate_full[correct])

        print(f"\n  Layer {L:2d}:")
        print(f"    Correct predictions: {correct.sum()}/{correct.size} ({correct.mean()*100:.1f}%)")
        print(f"    Approximate margin (|gate_128|):")
        print(f"      Correct: median={np.median(margin[correct]):.3f}, mean={np.mean(margin[correct]):.3f}")
        print(f"      Wrong:   median={np.median(margin[~correct]):.3f}, mean={np.mean(margin[~correct]):.3f}")
        print(f"    REAL gate magnitude:")
        print(f"      At correct: median={np.median(real_gate_at_correct):.3f}")
        print(f"      At errors:  median={np.median(real_gate_at_error):.3f}")
        if len(real_gate_at_error) > 0:
            near_zero_errors = (real_gate_at_error < LOG_PHI).sum()
            print(f"    Errors where |real gate| < log(φ): {near_zero_errors}/{len(real_gate_at_error)} "
                  f"({near_zero_errors/len(real_gate_at_error)*100:.1f}%)")
            deep_errors = (real_gate_at_error > 2 * LOG_PHI).sum()
            print(f"    Errors where |real gate| > 2·log(φ): {deep_errors}/{len(real_gate_at_error)} "
                  f"({deep_errors/len(real_gate_at_error)*100:.1f}%)")

    # ================================================================
    # Test 4: The Residual — What's in the Tail?
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: The Residual — Structure in the Low-Rank Tail")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            projected = X @ Vt[:128, :].T
            gate_128 = (projected * S[:128].unsqueeze(0)) @ U[:, :128].T
            gate_128 = gate_128.numpy()

        # The residual: what the rank-128 approximation misses
        residual = gate_full - gate_128

        # How large is the residual relative to the full gate?
        rel_residual = np.std(residual, axis=0) / (np.std(gate_full, axis=0) + 1e-10)

        # For channels where the sign prediction FAILS, how large is the residual?
        sign_full = (gate_full >= 0)
        sign_128 = (gate_128 >= 0)
        errors = (sign_full != sign_128)

        # At error locations: is the residual larger than the prediction?
        residual_at_errors = np.abs(residual[errors])
        prediction_at_errors = np.abs(gate_128[errors])

        print(f"\n  Layer {L:2d}:")
        print(f"    Residual std / full std:")
        print(f"      mean ratio: {np.mean(rel_residual):.3f}")
        print(f"      median ratio: {np.median(rel_residual):.3f}")
        print(f"      min ratio: {np.min(rel_residual):.3f}, max: {np.max(rel_residual):.3f}")

        if errors.sum() > 0:
            flipped = (residual_at_errors > prediction_at_errors).sum()
            print(f"    At error locations:")
            print(f"      |residual| > |prediction|: {flipped}/{errors.sum()} "
                  f"({flipped/errors.sum()*100:.1f}%) — the tail FLIPS the sign")
            print(f"      Mean |residual|: {np.mean(residual_at_errors):.4f}")
            print(f"      Mean |prediction|: {np.mean(prediction_at_errors):.4f}")

    # ================================================================
    # Test 5: φ-Structure in Sign Boundaries
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: φ-Structure — Do Sign Errors Cluster at φ-Boundaries?")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()

        # The gate values near zero are where sign is most uncertain
        # Question: are the "near zero" channels at φ-related positions?

        # Per-channel mean gate value
        mean_gate = gate_full.mean(axis=0)
        gate_std = gate_full.std(axis=0)

        # Channels where sign is uncertain (low |mean| relative to std)
        snr = np.abs(mean_gate) / (gate_std + 1e-10)  # signal-to-noise for sign prediction

        # Is SNR distributed with φ-structure?
        log_snr = np.log(snr + 1e-10)

        # Check: do channel indices with low SNR cluster?
        low_snr_channels = np.where(snr < 0.5)[0]
        high_snr_channels = np.where(snr > 2.0)[0]

        # Check if the SVD singular values have φ-ratios
        S_np = S.numpy()
        ratios = S_np[:-1] / S_np[1:]

        # Find where ratio ≈ φ
        phi_ratios = np.abs(ratios - PHI) < 0.05
        phi_inv_ratios = np.abs(ratios - 1/PHI) < 0.05

        print(f"\n  Layer {L:2d}:")
        print(f"    Sign-to-noise ratio (|mean_gate| / std_gate):")
        print(f"      Low SNR (<0.5):  {len(low_snr_channels):6d} channels — sign is UNPREDICTABLE")
        print(f"      High SNR (>2.0): {len(high_snr_channels):6d} channels — sign is PREDICTABLE")
        print(f"      Median SNR: {np.median(snr):.3f}")
        print(f"    SVD S ratios near φ: {phi_ratios.sum()}/{len(ratios)} "
              f"(expected by chance: {len(ratios) * 0.05 / PHI:.0f})")
        print(f"    SVD S ratios near 1/φ: {phi_inv_ratios.sum()}/{len(ratios)}")
        print(f"    S[0:5] ratios: {' '.join(f'{r:.4f}' for r in ratios[:5])}")

        # KEY QUESTION: What fraction of the "predictable" accuracy comes from
        # high-SNR channels vs low-SNR channels?
        with torch.no_grad():
            projected = X @ Vt[:128, :].T
            gate_128 = (projected * S[:128].unsqueeze(0)) @ U[:, :128].T
            gate_128 = gate_128.numpy()

        sign_full = (gate_full >= 0)
        sign_128 = (gate_128 >= 0)

        if len(low_snr_channels) > 0:
            acc_low_snr = np.mean(sign_full[:, low_snr_channels] == sign_128[:, low_snr_channels]) * 100
        else:
            acc_low_snr = float('nan')

        if len(high_snr_channels) > 0:
            acc_high_snr = np.mean(sign_full[:, high_snr_channels] == sign_128[:, high_snr_channels]) * 100
        else:
            acc_high_snr = float('nan')

        print(f"    Rank-128 accuracy by SNR:")
        print(f"      Low SNR channels:  {acc_low_snr:.1f}%")
        print(f"      High SNR channels: {acc_high_snr:.1f}%")

    # ================================================================
    # Test 6: The "Bias-Corrected" Accuracy — What Does Low-Rank Add?
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Bias-Corrected — What Does Low-Rank Actually Contribute?")
    print("=" * 70)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()

        sign_full = (gate_full >= 0)
        n_tokens, n_channels = sign_full.shape

        # Per-channel bias prediction
        channel_frac_pos = sign_full.mean(axis=0)
        bias_prediction = (channel_frac_pos >= 0.5)
        bias_correct = (sign_full == bias_prediction[np.newaxis, :])

        # Cases where bias is WRONG — these are the only cases rank-k could improve
        bias_wrong = ~bias_correct
        n_bias_wrong = bias_wrong.sum()

        for k in [4, 32, 128, 256]:
            if k > min(W_gate.shape):
                continue

            with torch.no_grad():
                projected = X @ Vt[:k, :].T
                gate_k = (projected * S[:k].unsqueeze(0)) @ U[:, :k].T
                gate_k = gate_k.numpy()

            sign_k = (gate_k >= 0)
            lr_correct = (sign_full == sign_k)

            # How many of the bias-wrong cases does rank-k fix?
            fixed_by_lr = (bias_wrong & lr_correct).sum()
            # How many of the bias-correct cases does rank-k break?
            broken_by_lr = (bias_correct & ~lr_correct).sum()

            overall_acc = lr_correct.mean() * 100
            bias_acc = bias_correct.mean() * 100

            print(f"\n  Layer {L:2d}, Rank-{k}:")
            print(f"    Bias accuracy:    {bias_acc:.2f}%  ({bias_correct.sum()} correct)")
            print(f"    Rank-{k} accuracy:  {overall_acc:.2f}%  ({lr_correct.sum()} correct)")
            print(f"    Bias-wrong cases: {n_bias_wrong}")
            print(f"      Fixed by rank-{k}:  {fixed_by_lr} ({fixed_by_lr/max(n_bias_wrong,1)*100:.1f}%)")
            print(f"      Broken by rank-{k}: {broken_by_lr} ({broken_by_lr/max(bias_correct.sum(),1)*100:.2f}%)")
            print(f"    Net improvement:  {fixed_by_lr - broken_by_lr:+d} predictions")

    # ================================================================
    # Test 7: The Key Question — Is 94% Explained by Bias?
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 7: SUMMARY — Is the 94% Just Bias?")
    print("=" * 70)

    print("\n  Layer | Trivial | Bias  | Rank-4 | Rank-128 | Rank-512 | Bias explains?")
    print("  " + "─" * 75)

    for L in test_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()

        sign_full = (gate_full >= 0)
        frac_pos = sign_full.mean()
        trivial = max(frac_pos, 1 - frac_pos) * 100

        channel_frac_pos = sign_full.mean(axis=0)
        bias_pred = (channel_frac_pos >= 0.5)
        bias_acc = np.mean(sign_full == bias_pred[np.newaxis, :]) * 100

        accs = {}
        for k in [4, 128, 512]:
            if k > min(W_gate.shape):
                accs[k] = float('nan')
                continue
            with torch.no_grad():
                projected = X @ Vt[:k, :].T
                gate_k = (projected * S[:k].unsqueeze(0)) @ U[:, :k].T
                gate_k = gate_k.numpy()
            sign_k = (gate_k >= 0)
            accs[k] = np.mean(sign_full == sign_k) * 100

        # Does bias explain rank-128?
        gap_bias_to_r128 = accs[128] - bias_acc
        gap_trivial_to_bias = bias_acc - trivial

        explained = "YES — bias dominates" if gap_bias_to_r128 < 2.0 else \
                    "PARTIAL" if gap_bias_to_r128 < gap_trivial_to_bias else \
                    "NO — rank adds signal"

        print(f"  {L:5d} | {trivial:5.1f}%  | {bias_acc:5.1f}% | {accs[4]:5.1f}%  | "
              f"{accs[128]:6.1f}%  | {accs.get(512, float('nan')):6.1f}%  | {explained}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
