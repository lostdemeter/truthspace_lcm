#!/usr/bin/env python3
"""
Phase 4 Deep Dive: The Selector-Spectrometer Connection

Layers 12 and 23 are sign-heavy (41-57% sign rules) and fail top-1 on
narrow-margin decisions. The selector (exp5b) showed Layer 1 acts as a
geometric discriminator with rank-1 concentrated MESH.

Questions:
  1. Do layers 12/23 have elevated rank-1 MESH energy? (selector-like)
  2. Is sign-heaviness correlated with selector-like attention across all layers?
  3. Can we build a selector-based replacement that works better than per-dim rules?
  4. Can a geometric "bias correction" on the sign dims recover the margin?
  5. What about a low-rank residual correction: rules + rank-k correction term?
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI
from phi_geometric.inference.phi_spectrometer import (
    SpectrometerRules, SpectrometerLayer, load_all_rules,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")
RULES_DIR = str(Path(__file__).parent / "results" / "phase4_rules_full")

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS


def efficient_mesh_svd(W_q_head, W_k_head):
    """Compute MESH singular values via QR + small SVD."""
    A = W_q_head.T.astype(np.float64)
    B = W_k_head.astype(np.float64)
    Q, R = np.linalg.qr(A)
    C = R @ B
    U_c, S, Vt = np.linalg.svd(C, full_matrices=False)
    U = Q @ U_c
    return U, S, Vt


def load_layer_weights(layer_idx):
    """Load Q/K weights for a layer."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    q_phi = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    k_phi = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
    return q_phi.decode(), k_phi.decode()


def get_rank1_energy(layer_idx):
    """Get average rank-1 energy fraction for a layer's MESH."""
    W_q, W_k = load_layer_weights(layer_idx)
    W_q_heads = W_q.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_k_heads = W_k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

    r1_fracs = []
    for h in range(NUM_HEADS):
        kv = h // HEADS_PER_KV
        _, S, _ = efficient_mesh_svd(W_q_heads[h], W_k_heads[kv])
        total = (S ** 2).sum()
        r1_fracs.append((S[0] ** 2) / total)
    return np.mean(r1_fracs), r1_fracs


def get_rule_composition(rules_dir, layer_idx, hidden_dim):
    """Load rule file and compute composition percentages."""
    rule_path = os.path.join(rules_dir, f'layer_{layer_idx:02d}.json')
    if not os.path.exists(rule_path):
        return {}
    with open(rule_path) as f:
        data = json.load(f)

    # Use pre-computed rule_distribution if available
    counts = data.get('rule_distribution', {})
    if not counts:
        # Fallback: compute from dim_rules list
        c = Counter()
        for rule in data.get('dim_rules', []):
            c[rule['rule_type']] += 1
        counts = dict(c)

    total = sum(counts.values())
    if total == 0:
        return {'sign_pct': 0, 'affine_pct': 0, 'unstruct_pct': 0,
                'counts': {}, 'total': 0}

    pcts = {k: v / total for k, v in counts.items()}
    sign_pct = sum(pcts.get(t, 0) for t in
                   ['sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate'])
    affine_pct = pcts.get('affine', 0)
    unstruct_pct = pcts.get('unstructured', 0)

    return {
        'sign_pct': sign_pct,
        'affine_pct': affine_pct,
        'unstruct_pct': unstruct_pct,
        'counts': counts,
        'total': total,
    }


def main():
    print("=" * 90)
    print("  Phase 4 Deep Dive: Selector ↔ Sign-Heavy Layer Connection")
    print("=" * 90)
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Part 1: Rank-1 MESH energy vs sign composition across ALL layers
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("  Part 1: Is sign-heaviness correlated with selector-like MESH?")
    print("=" * 90)
    print()

    print(f"{'Layer':>5} {'Zone':>7} {'Rank1%':>7} {'Sign%':>6} {'Affine%':>8} "
          f"{'Unstruct%':>10} {'Selector?':>10}")
    print("-" * 65)

    rank1_energies = []
    sign_pcts = []
    affine_pcts = []

    for li in range(28):
        r1_mean, _ = get_rank1_energy(li)
        comp = get_rule_composition(RULES_DIR, li, HIDDEN_DIM)

        rank1_energies.append(r1_mean)
        sign_pcts.append(comp.get('sign_pct', 0))
        affine_pcts.append(comp.get('affine_pct', 0))

        zone = ""
        if li <= 2: zone = "DRUM"
        elif li == 3: zone = "TRANS"
        elif li <= 6: zone = "COMB-E"
        elif li <= 25: zone = "COMB-L"
        else: zone = "MUSIC"

        sel_mark = "★ SEL" if r1_mean > 0.10 else ""
        fail_mark = " ← FAIL" if li in [12, 23] else ""

        print(f"{li:>5} {zone:>7} {r1_mean*100:>6.1f}% "
              f"{comp.get('sign_pct', 0)*100:>5.0f}% "
              f"{comp.get('affine_pct', 0)*100:>7.0f}% "
              f"{comp.get('unstruct_pct', 0)*100:>9.0f}% "
              f"{sel_mark}{fail_mark}")

    # Correlation between rank-1 energy and sign fraction
    r1_arr = np.array(rank1_energies)
    sign_arr = np.array(sign_pcts)
    aff_arr = np.array(affine_pcts)

    # Exclude layer 0 (no rules) and layers 26-27 (MUSIC, different regime)
    mask = np.array([1 <= i <= 25 for i in range(28)])
    corr_r1_sign = np.corrcoef(r1_arr[mask], sign_arr[mask])[0, 1]
    corr_r1_aff = np.corrcoef(r1_arr[mask], aff_arr[mask])[0, 1]

    print()
    print(f"  Correlation (layers 1-25):")
    print(f"    rank-1 energy ↔ sign%:   r = {corr_r1_sign:.3f}")
    print(f"    rank-1 energy ↔ affine%: r = {corr_r1_aff:.3f}")
    print()

    if abs(corr_r1_sign) > 0.3:
        direction = "positive" if corr_r1_sign > 0 else "negative"
        print(f"  → {direction.upper()} correlation: sign-heavy layers "
              f"{'ARE' if corr_r1_sign > 0 else 'are NOT'} more selector-like")
    else:
        print(f"  → No strong correlation: sign-heaviness and selector-ness are independent")

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: Detailed selector analysis of layers 12 and 23
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  Part 2: Detailed Selector Analysis of Layers 12 and 23")
    print(f"{'='*90}\n")

    for li in [1, 5, 12, 23]:
        r1_mean, r1_fracs = get_rank1_energy(li)
        r1_std = np.std(r1_fracs)
        r1_max = max(r1_fracs)
        r1_min = min(r1_fracs)

        # How many heads are selector-like (>10% rank-1)?
        n_selector_heads = sum(1 for f in r1_fracs if f > 0.10)

        label = {1: "DRUM (known selector)", 5: "COMB-E (peak structured)",
                 12: "COMB-L (FAIL)", 23: "COMB-L (FAIL)"}[li]
        print(f"  Layer {li} [{label}]:")
        print(f"    Rank-1 energy: {r1_mean*100:.1f}% ± {r1_std*100:.1f}% "
              f"(min={r1_min*100:.1f}%, max={r1_max*100:.1f}%)")
        print(f"    Selector-like heads (>10%): {n_selector_heads}/28")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # Part 3: Geometric Bias Correction
    # For sign dims, learn a per-dim magnitude correction from data
    # ═══════════════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print("  Part 3: Geometric Bias Correction for Sign-Heavy Layers")
    print("  Can a per-dim additive/multiplicative bias fix the margin?")
    print(f"{'='*90}\n")

    # Load engine for hidden state extraction
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)

    # Extract hidden states for correction calibration
    calibration_prompts = [
        "The capital of France is",
        "1 + 1 =",
        "Once upon a time",
        "The quick brown fox",
        "Water boils at",
        "The largest planet is",
    ]

    for target_layer in [12, 23]:
        print(f"\n  Layer {target_layer}: Learning bias correction...")

        # Collect pre/post hidden states for this layer
        pre_states = []
        post_states_full = []
        post_states_spec = []

        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        for prompt in calibration_prompts:
            ids = tokenizer.encode(prompt)

            # Forward to just before target layer
            hidden = engine.embedding(ids)[np.newaxis, :, :]
            for layer in engine.layers:
                if layer.layer_idx == target_layer:
                    pre_states.append(hidden.copy())
                    # Full layer output
                    full_out = layer(hidden.copy())
                    post_states_full.append(full_out)
                    # Spec layer output
                    spec_out = spec_layer(hidden.copy())
                    post_states_spec.append(spec_out)
                    break
                hidden = layer(hidden)

        # Stack all: (N_total, D)
        all_pre = np.concatenate([s[0] for s in pre_states], axis=0)
        all_full = np.concatenate([s[0] for s in post_states_full], axis=0)
        all_spec = np.concatenate([s[0] for s in post_states_spec], axis=0)

        # Per-dim error: spec - full
        error = all_spec - all_full  # (N, D)
        mean_error = error.mean(axis=0)  # (D,)  -- the bias
        std_error = error.std(axis=0)    # (D,)

        # Per-dim scale correction: spec/full
        safe_full = np.where(np.abs(all_full) > 1e-10, all_full, 1.0)
        scale_ratio = (all_spec / safe_full)
        mean_scale = np.median(scale_ratio, axis=0)  # median is more robust

        print(f"    Mean abs bias: {np.abs(mean_error).mean():.6f}")
        print(f"    Mean scale deviation from 1.0: {np.abs(mean_scale - 1.0).mean():.4f}")
        print(f"    Dims with |bias| > 0.1: {(np.abs(mean_error) > 0.1).sum()}/3584")
        print(f"    Dims with |scale-1| > 0.1: {(np.abs(mean_scale - 1.0) > 0.1).sum()}/3584")

        # Apply additive bias correction and test
        test_prompt = "The capital of France is"
        test_ids = tokenizer.encode(test_prompt)

        # Forward to target layer
        hidden = engine.embedding(test_ids)[np.newaxis, :, :]
        for layer in engine.layers:
            if layer.layer_idx == target_layer:
                # Apply spec layer + bias correction
                spec_out = spec_layer(hidden.copy())
                corrected_out = spec_out - mean_error[np.newaxis, np.newaxis, :]
                # Continue with corrected output
                hidden_corrected = corrected_out
                hidden_full = layer(hidden.copy())
                break
            hidden = layer(hidden)

        # Finish forward from target_layer+1 with both
        def finish_forward(engine, hidden_start, start_layer):
            h = hidden_start
            for layer in engine.layers:
                if layer.layer_idx > start_layer:
                    h = layer(h)
            h = rms_norm(h, engine.final_norm_weight)
            return engine.lm_head(h)

        logits_full = finish_forward(engine, hidden_full, target_layer)
        logits_corrected = finish_forward(engine, hidden_corrected, target_layer)
        logits_uncorrected = finish_forward(engine, spec_out, target_layer)

        # Compare
        last_full = logits_full[0, -1, :]
        last_corr = logits_corrected[0, -1, :]
        last_uncorr = logits_uncorrected[0, -1, :]

        corr_r = np.corrcoef(last_full, last_corr)[0, 1]
        uncorr_r = np.corrcoef(last_full, last_uncorr)[0, 1]

        full_top1 = int(np.argmax(last_full))
        corr_top1 = int(np.argmax(last_corr))
        uncorr_top1 = int(np.argmax(last_uncorr))

        full_tok = tokenizer.decode_token(full_top1)
        corr_tok = tokenizer.decode_token(corr_top1)
        uncorr_tok = tokenizer.decode_token(uncorr_top1)

        # Margins
        full_sorted = np.sort(last_full)[::-1]
        corr_sorted = np.sort(last_corr)[::-1]

        print(f"\n    Test: \"{test_prompt}\"")
        print(f"    {'':>20s} {'r':>8} {'Top-1':>12} {'Margin':>8}")
        print(f"    {'Full engine':>20s} {'1.0000':>8} {full_tok:>12} "
              f"{full_sorted[0]-full_sorted[1]:>8.3f}")
        print(f"    {'Spec (no correction)':>20s} {uncorr_r:>8.4f} {uncorr_tok:>12}")
        print(f"    {'Spec + bias corr':>20s} {corr_r:>8.4f} {corr_tok:>12} "
              f"{corr_sorted[0]-corr_sorted[1]:>8.3f}")

        match_uncorr = "✓" if uncorr_top1 == full_top1 else "✗"
        match_corr = "✓" if corr_top1 == full_top1 else "✗"
        print(f"\n    Uncorrected: {match_uncorr}  Corrected: {match_corr}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 4: Low-Rank Residual Correction
    # rules + rank-k SVD of the residual error
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  Part 4: Low-Rank Residual Correction")
    print("  rules_output + U @ S @ Vt (rank-k correction from calibration)")
    print(f"{'='*90}\n")

    for target_layer in [12, 23]:
        print(f"  Layer {target_layer}:")

        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        # Collect error matrix from calibration prompts
        errors = []
        inputs = []
        for prompt in calibration_prompts:
            ids = tokenizer.encode(prompt)
            hidden = engine.embedding(ids)[np.newaxis, :, :]
            for layer in engine.layers:
                if layer.layer_idx == target_layer:
                    full_out = layer(hidden.copy())
                    spec_out = spec_layer(hidden.copy())
                    # Each token position is a sample
                    for t in range(hidden.shape[1]):
                        errors.append((full_out[0, t] - spec_out[0, t]))
                        inputs.append(hidden[0, t])
                    break
                hidden = layer(hidden)

        E = np.array(errors)   # (N, D)  -- error matrix
        X = np.array(inputs)   # (N, D)  -- input matrix

        # SVD of error matrix
        U_e, S_e, Vt_e = np.linalg.svd(E, full_matrices=False)
        total_error_var = (S_e ** 2).sum()

        print(f"    Error matrix: {E.shape}")
        print(f"    Top-10 singular values: [{', '.join(f'{s:.2f}' for s in S_e[:10])}]")

        # How much error variance is captured by rank-k?
        for k in [1, 3, 5, 10, 20]:
            captured = (S_e[:k] ** 2).sum() / total_error_var
            print(f"    Rank-{k:2d} captures {captured*100:.1f}% of error variance")

        # Try learning a correction: error ≈ X @ W_correction
        # where W_correction is low-rank: W_corr = A @ B.T
        # Least squares: W_corr = (X.T @ X)^-1 @ X.T @ E
        # But X may be ill-conditioned, use pseudoinverse
        for k in [5, 10, 20]:
            # Truncated SVD of X
            U_x, S_x, Vt_x = np.linalg.svd(X, full_matrices=False)
            # Use top-k components of X to predict E
            X_k = U_x[:, :k] * S_x[:k]  # (N, k)
            # Least squares: E ≈ X_k @ W  => W = pinv(X_k) @ E
            W_k = np.linalg.lstsq(X_k, E, rcond=None)[0]  # (k, D)

            E_pred = X_k @ W_k
            residual = E - E_pred
            residual_var = np.sum(residual ** 2)
            explained = 1.0 - residual_var / np.sum(E ** 2)

            print(f"    Input-rank-{k:2d} linear correction explains "
                  f"{explained*100:.1f}% of error")

        # Apply the best correction (rank-20 input-space) and test
        k = 20
        U_x, S_x, Vt_x = np.linalg.svd(X, full_matrices=False)
        X_k = U_x[:, :k] * S_x[:k]
        W_k = np.linalg.lstsq(X_k, E, rcond=None)[0]

        # The correction for a new input x is:
        # correction = (x @ Vt_x[:k].T) @ diag(1/S_x[:k]) @ ... but simpler:
        # Project input onto the k components, apply W_k
        # correction(x) = (x @ Vt_x[:k].T / S_x[:k]) @ W_k

        # Build the full correction matrix: D_in → D_out
        # correction = input @ Vt_x[:k].T @ diag(1/S_x[:k]) @ W_k
        # = input @ (Vt_x[:k].T @ diag(1/S_x[:k]) @ W_k)
        # = input @ C  where C is (D, D) but rank-k
        C = Vt_x[:k].T @ (np.diag(1.0 / S_x[:k]) @ W_k)  # (D, D)

        # Test on "The capital of France is"
        test_ids = tokenizer.encode("The capital of France is")
        hidden = engine.embedding(test_ids)[np.newaxis, :, :]
        for layer in engine.layers:
            if layer.layer_idx == target_layer:
                full_out = layer(hidden.copy())
                spec_out = spec_layer(hidden.copy())
                # Apply low-rank correction
                corrected = spec_out.copy()
                for t in range(hidden.shape[1]):
                    correction = hidden[0, t] @ C
                    corrected[0, t] += correction
                break
            hidden = layer(hidden)

        logits_full = finish_forward(engine, full_out, target_layer)
        logits_corr = finish_forward(engine, corrected, target_layer)
        logits_uncorr = finish_forward(engine, spec_out, target_layer)

        last_full = logits_full[0, -1, :]
        last_corr = logits_corr[0, -1, :]
        last_uncorr = logits_uncorrected[0, -1, :]

        full_top1 = int(np.argmax(last_full))
        corr_top1 = int(np.argmax(last_corr))
        uncorr_top1 = int(np.argmax(last_uncorr))

        corr_r = np.corrcoef(last_full, last_corr)[0, 1]

        full_tok = tokenizer.decode_token(full_top1)
        corr_tok = tokenizer.decode_token(corr_top1)

        corr_sorted = np.sort(last_corr)[::-1]

        print(f"\n    Low-rank-{k} correction test: \"{test_prompt}\"")
        print(f"      Full engine top-1:  {full_tok}")
        print(f"      Corrected top-1:    {corr_tok}  (r={corr_r:.4f}, "
              f"margin={corr_sorted[0]-corr_sorted[1]:.3f})")
        match = "✓" if corr_top1 == full_top1 else "✗"
        print(f"      Match: {match}")

        # Show cost: the correction is rank-k, so it's k*D + k*D parameters
        correction_params = k * HIDDEN_DIM * 2
        full_layer_params = HIDDEN_DIM * HIDDEN_DIM  # rough
        print(f"      Correction cost: {correction_params:,} params "
              f"(vs ~{full_layer_params:,} for full layer)")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # Part 5: What Would "Ripping Out" Look Like?
    # Can we replace the whole layer with selector + rules + correction?
    # ═══════════════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print("  Part 5: Layer Replacement Cost Analysis")
    print("  Full layer vs spec-rules + low-rank correction")
    print(f"{'='*90}\n")

    for target_layer in [12, 23]:
        comp = get_rule_composition(RULES_DIR, target_layer, HIDDEN_DIM)
        rules = all_rules[target_layer]
        n_structured = sum(1 for d, r in rules.rules.items()
                          if r.r_squared >= 0.7)
        n_total = len(rules.rules)

        # Count parameters needed for rules
        n_affine_params = comp['counts'].get('affine', 0) * 2  # slope + intercept
        n_quad_params = comp['counts'].get('quadratic', 0) * 3  # a, b, c
        n_gate_params = comp['counts'].get('gating', 0) * 5    # threshold + 2 slopes + 2 intercepts
        n_sign_params = comp['counts'].get('sign_preserve', 0) * 0  # no params!
        n_sign_params += comp['counts'].get('sign_flip', 0) * 0
        n_sign_params += comp['counts'].get('sign_xor', 0) * 1  # xor_dim
        n_sign_params += comp['counts'].get('sign_gate', 0) * 1  # threshold
        n_scale_params = comp['counts'].get('scale', 0) * 1

        total_rule_params = (n_affine_params + n_quad_params +
                            n_gate_params + n_sign_params + n_scale_params)

        # Full transformer layer params (approximate)
        # Q, K, V, O projections + MLP (gate, up, down) + norms
        full_attn_params = (HIDDEN_DIM * HIDDEN_DIM * 2 +  # Q, O
                           HIDDEN_DIM * 512 * 2)  # K, V (GQA)
        full_mlp_params = HIDDEN_DIM * 18944 * 3  # gate, up, down
        full_norm_params = HIDDEN_DIM * 2
        full_total = full_attn_params + full_mlp_params + full_norm_params

        print(f"  Layer {target_layer}:")
        print(f"    Full layer:        ~{full_total:>12,} params")
        print(f"    Rule parameters:    {total_rule_params:>12,} params "
              f"({total_rule_params/full_total*100:.3f}%)")
        print(f"    + rank-20 corr:     {20*HIDDEN_DIM*2:>12,} params "
              f"({20*HIDDEN_DIM*2/full_total*100:.3f}%)")
        print(f"    Total replacement:  {total_rule_params + 20*HIDDEN_DIM*2:>12,} params")
        print(f"    Compression:        {full_total / (total_rule_params + 20*HIDDEN_DIM*2):.0f}×")
        print()


if __name__ == '__main__':
    main()
