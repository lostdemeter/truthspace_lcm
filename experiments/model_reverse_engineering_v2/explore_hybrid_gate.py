#!/usr/bin/env python3
"""
Hybrid Gate: 3-Tier Selective Computation
==========================================

The gate has 3-tier information structure:
  Tier 1 (bias):      ~75-93% of signs — channels with fixed polarity (FREE)
  Tier 2 (low-rank):  ~0-7% additional signs — scaffold, input-dependent (CHEAP)
  Tier 3 (full-rank): ~7-25% of signs — PRESERVE region, negative zero (EXPENSIVE)

This script:
1. Investigates WHY Layer 21 has strongest low-rank signal (+7pts over bias)
   - Connection to 4-zone architecture (DRUM/TRANSITION/COMB-early/COMB-late/MUSIC)
   - Per-layer profile of bias vs low-rank vs full-rank contribution
2. Builds the hybrid gate predictor:
   - Precompute per-channel bias (Tier 1)
   - Low-rank projection for Tier 2 scaffold
   - Full gate matmul ONLY for channels predicted to be in PRESERVE region
3. Tests end-to-end with the hybrid approach
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
    print("HYBRID GATE: 3-Tier Selective Computation")
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

    # ================================================================
    # Part 1: ALL-LAYER profile — where is low-rank signal strongest?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: All-Layer Tier Decomposition")
    print("=" * 70)
    print("Zones: DRUM(0-2), TRANSITION(3), COMB-early(4-6), COMB-late(7-25), MUSIC(26-27)")

    # Capture hidden states for ALL 28 layers
    all_layers = list(range(28))
    hidden_states = {L: [] for L in all_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states[layer_idx].append(input[0].detach())
        return hook_fn

    for L in all_layers:
        h = model.model.layers[L].mlp.register_forward_hook(make_hook(L))
        hooks.append(h)

    print("\nCapturing hidden states for all 28 layers...")
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            model(ids)

    for h in hooks:
        h.remove()

    for L in all_layers:
        hidden_states[L] = torch.cat(hidden_states[L], dim=1).squeeze(0)

    n_tokens = hidden_states[0].shape[0]
    print(f"  {n_tokens} tokens captured per layer")

    # Profile each layer
    zone_map = {}
    for L in range(28):
        if L <= 2:
            zone_map[L] = "DRUM"
        elif L == 3:
            zone_map[L] = "TRANS"
        elif L <= 6:
            zone_map[L] = "COMB-E"
        elif L <= 25:
            zone_map[L] = "COMB-L"
        else:
            zone_map[L] = "MUSIC"

    print(f"\n  {'L':>2s} {'Zone':>6s} | {'Trivial':>7s} {'Bias':>7s} {'Rk-4':>7s} {'Rk-32':>7s} {'Rk-128':>7s} | "
          f"{'Rk4-Bias':>8s} {'LR signal':>9s} | {'Preserve%':>9s} {'Volatile%':>9s}")
    print("  " + "─" * 105)

    layer_profiles = {}

    for L in all_layers:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data
        X = hidden_states[L]

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()

        sign_full = (gate_full >= 0)
        n_ch = sign_full.shape[1]

        # Trivial
        frac_pos = sign_full.mean()
        trivial = max(frac_pos, 1 - frac_pos) * 100

        # Bias
        ch_frac_pos = sign_full.mean(axis=0)
        bias_pred = (ch_frac_pos >= 0.5)
        bias_acc = np.mean(sign_full == bias_pred[np.newaxis, :]) * 100

        # Low-rank
        U, S, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        accs = {}
        for k in [4, 32, 128]:
            with torch.no_grad():
                proj = X @ Vt[:k, :].T
                gate_k = (proj * S[:k].unsqueeze(0)) @ U[:, :k].T
                gate_k = gate_k.numpy()
            accs[k] = np.mean(sign_full == (gate_k >= 0)) * 100

        # PRESERVE fraction
        preserve_frac = (np.abs(gate_full) <= LOG_PHI).mean() * 100

        # Volatile channels (sign changes across tokens)
        ch_bias = np.abs(ch_frac_pos - 0.5)
        volatile_frac = (ch_bias < 0.1).mean() * 100  # channels where sign is nearly 50/50

        lr_signal = accs[4] - bias_acc
        zone = zone_map[L]

        layer_profiles[L] = {
            'trivial': trivial, 'bias': bias_acc,
            'rk4': accs[4], 'rk32': accs[32], 'rk128': accs[128],
            'lr_signal': lr_signal, 'preserve_frac': preserve_frac,
            'volatile_frac': volatile_frac, 'zone': zone,
            'U': U, 'S': S, 'Vt': Vt, 'ch_frac_pos': ch_frac_pos,
        }

        marker = " ◄" if lr_signal > 3.0 else ""
        print(f"  {L:2d} {zone:>6s} | {trivial:6.1f}% {bias_acc:6.1f}% {accs[4]:6.1f}% "
              f"{accs[32]:6.1f}% {accs[128]:6.1f}% | {lr_signal:+7.1f}pt "
              f"{'STRONG' if lr_signal > 3 else 'weak':>9s} | {preserve_frac:8.1f}% {volatile_frac:8.1f}%{marker}")

    # ================================================================
    # Part 2: What makes Layer 21 special?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: Why is Layer 21 special?")
    print("=" * 70)

    # Find all layers with strong low-rank signal
    strong_lr = [(L, p['lr_signal']) for L, p in layer_profiles.items() if p['lr_signal'] > 3.0]
    strong_lr.sort(key=lambda x: -x[1])

    if strong_lr:
        print(f"\n  Layers with strong low-rank signal (>3pts over bias):")
        for L, sig in strong_lr:
            print(f"    Layer {L:2d} ({layer_profiles[L]['zone']:>6s}): {sig:+.1f}pts, "
                  f"preserve={layer_profiles[L]['preserve_frac']:.1f}%, "
                  f"volatile={layer_profiles[L]['volatile_frac']:.1f}%")
    else:
        print("\n  No layers have >3pts of low-rank signal")

    # For strongest layer: what does the low-rank projection capture?
    if strong_lr:
        best_L = strong_lr[0][0]
        print(f"\n  Analyzing strongest layer: {best_L}")

        W_gate = model.model.layers[best_L].mlp.gate_proj.weight.data
        X = hidden_states[best_L]
        p = layer_profiles[best_L]

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            proj4 = X @ p['Vt'][:4, :].T
            gate_4 = (proj4 * p['S'][:4].unsqueeze(0)) @ p['U'][:, :4].T
            gate_4 = gate_4.numpy()

        sign_full = (gate_full >= 0)
        sign_4 = (gate_4 >= 0)
        bias_pred = (p['ch_frac_pos'] >= 0.5)

        # Which channels does rank-4 fix that bias gets wrong?
        bias_wrong = (sign_full != bias_pred[np.newaxis, :])
        lr4_correct = (sign_full == sign_4)
        fixed = (bias_wrong & lr4_correct)

        # Per-channel: which channels are "fixed" most often?
        fix_rate = fixed.mean(axis=0)  # fraction of tokens where rank-4 fixes bias error
        top_fixed = np.argsort(fix_rate)[::-1][:20]

        print(f"\n  Top 20 channels where rank-4 fixes bias errors:")
        print(f"    {'Channel':>8s} {'Fix rate':>8s} {'Bias':>6s} {'|mean gate|':>11s} {'gate std':>8s}")
        for ch in top_fixed:
            mg = np.abs(gate_full[:, ch].mean())
            gs = gate_full[:, ch].std()
            print(f"    {ch:8d} {fix_rate[ch]:7.1%} {p['ch_frac_pos'][ch]:5.1%} {mg:10.4f} {gs:7.4f}")

        # Are the fixed channels clustered in the weight matrix?
        high_fix_channels = np.where(fix_rate > 0.1)[0]
        if len(high_fix_channels) > 1:
            spacings = np.diff(high_fix_channels)
            print(f"\n  High-fix channels (fix_rate > 10%): {len(high_fix_channels)}")
            print(f"    Spacing stats: mean={spacings.mean():.0f}, std={spacings.std():.0f}, "
                  f"min={spacings.min()}, max={spacings.max()}")
            # Check for φ-spacing
            phi_spaced = np.abs(spacings / PHI - np.round(spacings / PHI)) < 0.1
            print(f"    φ-spaced: {phi_spaced.sum()}/{len(spacings)}")

    # ================================================================
    # Part 3: Build and test the hybrid gate
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: Hybrid Gate — Selective Computation")
    print("=" * 70)

    # Strategy:
    # 1. Use per-channel bias to predict Tier 1 channels (confidence > threshold)
    # 2. Use rank-32 projection for Tier 2 channels
    # 3. Compute FULL gate row ONLY for channels where the rank-32 prediction
    #    falls near the zero boundary (uncertain channels)
    #
    # The key insight: we don't need to compute ALL 18944 rows of W_gate @ x.
    # We only need the rows where the sign is uncertain.

    print("\n  Testing selective gate computation...")
    print("  For each layer, measure how many rows need full computation")

    # Threshold for "confident" prediction from low-rank
    confidence_thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]

    for L in [0, 7, 14, 21, 27]:
        W_gate = model.model.layers[L].mlp.gate_proj.weight.data  # (18944, 3584)
        X = hidden_states[L]
        p = layer_profiles[L]
        n_ch = W_gate.shape[0]

        with torch.no_grad():
            gate_full = F.linear(X, W_gate).numpy()
            proj = X @ p['Vt'][:32, :].T
            gate_32 = (proj * p['S'][:32].unsqueeze(0)) @ p['U'][:, :32].T
            gate_32 = gate_32.numpy()

        sign_full = (gate_full >= 0)

        print(f"\n  Layer {L:2d} ({layer_profiles[L]['zone']}):")
        print(f"    {'Threshold':>10s} {'Uncertain%':>10s} {'Need full':>10s} "
              f"{'Sign acc':>9s} {'Savings':>8s}")

        for thresh in confidence_thresholds:
            # Channels where |gate_32| < threshold → uncertain, need full computation
            uncertain = (np.abs(gate_32) < thresh)  # per token-channel
            frac_uncertain = uncertain.mean() * 100

            # For confident channels, use sign of gate_32
            # For uncertain channels, use sign of gate_full
            hybrid_sign = np.where(uncertain, sign_full, (gate_32 >= 0))
            hybrid_acc = np.mean(sign_full == hybrid_sign) * 100

            # Savings: we skip (100 - frac_uncertain)% of the full gate rows
            savings = (100 - frac_uncertain)

            print(f"    {thresh:10.2f} {frac_uncertain:9.1f}% {int(n_ch * frac_uncertain / 100):10d} "
                  f"{hybrid_acc:8.2f}% {savings:7.1f}%")

    # ================================================================
    # Part 4: End-to-end test with hybrid gate
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: End-to-End Hybrid Gate Test")
    print("=" * 70)

    test_prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "The color of the sky is",
        "One plus one equals",
        "The chemical symbol for gold is",
    ]

    # Precompute low-rank factors and bias for ALL layers
    hybrid_data = {}
    print("\n  Precomputing hybrid gate data for all 28 layers...")
    for layer_idx in range(model.config.num_hidden_layers):
        W_gate = model.model.layers[layer_idx].mlp.gate_proj.weight.data
        U, S_vals, Vt = torch.linalg.svd(W_gate, full_matrices=False)

        # Per-channel bias: use the mean gate sign from our calibration data
        # In practice, this could be computed from any calibration set
        X_cal = hidden_states.get(layer_idx)
        if X_cal is not None:
            with torch.no_grad():
                gate_cal = F.linear(X_cal, W_gate)
                ch_bias = (gate_cal >= 0).float().mean(dim=0)  # fraction positive
        else:
            ch_bias = torch.ones(W_gate.shape[0]) * 0.5  # unknown, assume 50/50

        hybrid_data[layer_idx] = {
            'Vt_k': Vt[:32, :].T.contiguous(),     # (3584, 32)
            'S_k': S_vals[:32].contiguous(),         # (32,)
            'U_k': U[:, :32].contiguous(),           # (18944, 32)
            'ch_bias': ch_bias,                       # (18944,) fraction positive
            'W_gate': W_gate,                         # full weight for uncertain rows
        }

    # Confidence threshold for the hybrid approach
    HYBRID_THRESH = 0.5  # |gate_approx| < this → compute exact

    def make_hybrid_mlp_hook(layer_idx):
        """
        Hybrid MLP hook:
        1. Low-rank gate prediction (rank-32)
        2. Identify uncertain channels (|gate_approx| < threshold)
        3. Compute FULL gate only for uncertain channels
        4. Use bias for channels where bias is very confident AND rank-32 agrees
        5. Compute SiLU with the hybrid gate
        """
        hd = hybrid_data[layer_idx]

        def hook_fn(module, input, output):
            x = input[0]
            shape = x.shape
            x_flat = x.view(-1, shape[-1])  # (B*S, 3584)
            n_tok = x_flat.shape[0]
            n_ch = module.gate_proj.weight.shape[0]  # 18944

            # Step 1: Low-rank gate prediction (CHEAP: 32 × 3584 + 32 × 18944)
            proj = x_flat @ hd['Vt_k']  # (n_tok, 32)
            gate_approx = (proj * hd['S_k'].unsqueeze(0)) @ hd['U_k'].T  # (n_tok, 18944)

            # Step 2: Identify uncertain channels per token
            confident = (torch.abs(gate_approx) >= HYBRID_THRESH)  # True = confident

            # Step 3: For uncertain channels, compute FULL gate row
            # We need to compute W_gate[uncertain_rows, :] @ x for each token
            # This is the selective part — only compute rows we're not confident about

            # Use the low-rank prediction as the base gate value
            gate_hybrid = gate_approx.clone()

            # For uncertain channels, overwrite with exact computation
            uncertain_mask = ~confident
            if uncertain_mask.any():
                # Find which channels need full computation for ANY token
                any_uncertain = uncertain_mask.any(dim=0)  # (18944,) — which channels ever uncertain
                uncertain_indices = torch.where(any_uncertain)[0]

                if len(uncertain_indices) > 0:
                    # Compute full gate for just these rows
                    W_uncertain = module.gate_proj.weight.data[uncertain_indices, :]  # (n_uncertain, 3584)
                    gate_exact_subset = x_flat @ W_uncertain.T  # (n_tok, n_uncertain)

                    # Overwrite uncertain positions with exact values
                    # Only overwrite where that specific (token, channel) was uncertain
                    for i, ch_idx in enumerate(uncertain_indices):
                        mask_col = uncertain_mask[:, ch_idx]
                        if mask_col.any():
                            gate_hybrid[mask_col, ch_idx] = gate_exact_subset[mask_col, i]

            # Step 4: Apply SiLU and compute MLP
            activated = F.silu(gate_hybrid)
            up_out = F.linear(x_flat, module.up_proj.weight.data)
            intermediate = activated * up_out
            result = F.linear(intermediate, module.down_proj.weight.data)
            return result.view(shape)

        return hook_fn

    # --- Test: Measure what fraction of channels are "uncertain" ---
    print(f"\n  Threshold = {HYBRID_THRESH}")
    print(f"  Measuring fraction of gate rows requiring full computation...")

    # Quick measurement on one prompt
    test_id = tokenizer("The capital of France is", return_tensors="pt").input_ids

    uncertain_stats = {}
    measure_hooks = []

    def make_measure_hook(layer_idx):
        hd = hybrid_data[layer_idx]

        def hook_fn(module, input, output):
            x = input[0].view(-1, input[0].shape[-1])
            proj = x @ hd['Vt_k']
            gate_approx = (proj * hd['S_k'].unsqueeze(0)) @ hd['U_k'].T
            uncertain = (torch.abs(gate_approx) < HYBRID_THRESH)
            any_uncertain = uncertain.any(dim=0)
            uncertain_stats[layer_idx] = {
                'frac_per_element': uncertain.float().mean().item(),
                'channels_ever_uncertain': any_uncertain.sum().item(),
                'total_channels': gate_approx.shape[1],
            }
        return hook_fn

    for layer_idx in range(model.config.num_hidden_layers):
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_measure_hook(layer_idx))
        measure_hooks.append(h)

    with torch.no_grad():
        model(test_id)

    for h in measure_hooks:
        h.remove()

    total_uncertain = 0
    total_channels = 0
    print(f"\n  {'Layer':>5s} {'Zone':>6s} {'Uncertain%':>10s} {'Channels':>10s} {'Savings':>8s}")
    for L in range(28):
        s = uncertain_stats[L]
        pct = s['frac_per_element'] * 100
        ch = s['channels_ever_uncertain']
        savings = (1 - s['frac_per_element']) * 100
        total_uncertain += s['frac_per_element']
        total_channels += 1
        print(f"  {L:5d} {zone_map[L]:>6s} {pct:9.1f}% {ch:9d}/{s['total_channels']} {savings:7.1f}%")

    avg_uncertain = total_uncertain / total_channels * 100
    avg_savings = 100 - avg_uncertain
    print(f"\n  Average: {avg_uncertain:.1f}% uncertain, {avg_savings:.1f}% gate rows skipped")

    # --- End-to-end accuracy test ---
    print(f"\n  End-to-end accuracy test:")

    for prompt in test_prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Baseline: full model
        with torch.no_grad():
            logits_full = model(ids).logits[0, -1, :]
            top_full = torch.argmax(logits_full).item()
            tok_full = tokenizer.decode([top_full])

        # Hybrid gate
        hooks = []
        for layer_idx in range(model.config.num_hidden_layers):
            h = model.model.layers[layer_idx].mlp.register_forward_hook(
                make_hybrid_mlp_hook(layer_idx)
            )
            hooks.append(h)

        with torch.no_grad():
            logits_hybrid = model(ids).logits[0, -1, :]
            top_hybrid = torch.argmax(logits_hybrid).item()
            tok_hybrid = tokenizer.decode([top_hybrid])

        for h in hooks:
            h.remove()

        match = "✓" if top_full == top_hybrid else "✗"
        corr = torch.corrcoef(torch.stack([logits_full, logits_hybrid]))[0, 1].item()
        print(f"    {match} \"{prompt}\" → full={tok_full!r}, hybrid={tok_hybrid!r}, "
              f"logit_corr={corr:.6f}")

    # --- Timing comparison ---
    print(f"\n  Timing comparison (single prompt):")

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    # Full model timing
    times_full = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            model(ids)
        times_full.append(time.time() - t0)

    # Hybrid timing
    hooks = []
    for layer_idx in range(model.config.num_hidden_layers):
        h = model.model.layers[layer_idx].mlp.register_forward_hook(
            make_hybrid_mlp_hook(layer_idx)
        )
        hooks.append(h)

    times_hybrid = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            model(ids)
        times_hybrid.append(time.time() - t0)

    for h in hooks:
        h.remove()

    t_full = np.median(times_full)
    t_hybrid = np.median(times_hybrid)
    print(f"    Full model:   {t_full:.3f}s")
    print(f"    Hybrid model: {t_hybrid:.3f}s")
    print(f"    Ratio: {t_hybrid/t_full:.2f}× ({'faster' if t_hybrid < t_full else 'SLOWER'})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
