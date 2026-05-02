#!/usr/bin/env python3
"""
Phase 4: Rank-1 Error Correction for Sign-Heavy Layers

The spectrometer error concentrates 99.6% in rank-1 for layers 12 and 23.
This means the entire mismatch between spec and full layer lives in ONE
direction in 3584-d space.

Plan:
  1. Generate ~200+ calibration samples from diverse prompts
  2. Compute error matrix: E[i] = full_layer(x_i) - spec_layer(x_i)
  3. SVD of E to extract dominant error direction (u, v, σ)
  4. Learn rank-1 correction: corrected(x) = spec(x) + (x · v_in) · v_out
  5. Test on HELD-OUT prompts
  6. Check stability: does the error direction generalize?
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_spectrometer import (
    SpectrometerRules, SpectrometerLayer, load_all_rules,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")
RULES_DIR = str(Path(__file__).parent / "results" / "phase4_rules_full")

# Diverse calibration prompts — NOT including the test prompt
CALIBRATION_PROMPTS = [
    # Simple facts
    "1 + 1 =",
    "2 + 2 =",
    "The sky is",
    "Water is made of",
    "The sun rises in the",
    "Gravity makes things fall",
    # Narrative
    "Once upon a time",
    "She walked into the room and",
    "He said that he would",
    "They decided to go to the",
    "The old man sat on the",
    "After the rain stopped",
    # Technical
    "The quick brown fox",
    "In machine learning",
    "Python is a programming",
    "The function returns",
    "An algorithm that sorts",
    # Knowledge
    "The largest planet is",
    "Albert Einstein developed the",
    "Shakespeare wrote many",
    "The speed of light is",
    "DNA stands for",
    "The Pacific Ocean is",
    # Conversational
    "I think that we should",
    "She said that she would",
    "It is important to note that",
    "The reason for this is",
    "According to the latest",
    # Mixed structure
    "In 2024, the world",
    "If you want to learn",
    "The best way to",
    "One of the most important",
    "As a result of the",
    "Between the two options",
    "Despite the challenges",
    "For example, consider",
    "However, it is worth",
    "In conclusion, the",
]

# Held-out test prompts — these are NOT in calibration
TEST_PROMPTS = [
    "The capital of France is",       # The original failure case
    "The largest ocean is the",       # Similar: factual with close alternatives
    "Water boils at",                 # Numeric
    "The color of grass is",          # Simple fact
    "Barack Obama was the",           # Proper noun
    "To be or not to",                # Famous quote
    "The square root of 144 is",      # Math
    "Roses are red, violets are",     # Pattern completion
]


def collect_layer_data(engine, tokenizer, spec_layer, target_layer, prompts):
    """Collect (input, full_output, spec_output) triples for a layer."""
    inputs = []
    full_outputs = []
    spec_outputs = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        hidden = engine.embedding(ids)[np.newaxis, :, :]
        for layer in engine.layers:
            if layer.layer_idx == target_layer:
                full_out = layer(hidden.copy())
                spec_out = spec_layer(hidden.copy())
                # Each token position is a sample
                for t in range(hidden.shape[1]):
                    inputs.append(hidden[0, t])
                    full_outputs.append(full_out[0, t])
                    spec_outputs.append(spec_out[0, t])
                break
            hidden = layer(hidden)

    return np.array(inputs), np.array(full_outputs), np.array(spec_outputs)


def finish_forward(engine, hidden_start, start_layer):
    """Continue forward pass from a given layer."""
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def test_replacement(engine, tokenizer, spec_layer, target_layer, prompt,
                     correction_fn=None):
    """Test single-layer replacement with optional correction, return logits."""
    ids = tokenizer.encode(prompt)
    hidden = engine.embedding(ids)[np.newaxis, :, :]

    for layer in engine.layers:
        if layer.layer_idx == target_layer:
            full_out = layer(hidden.copy())
            spec_out = spec_layer(hidden.copy())
            if correction_fn is not None:
                corrected_out = correction_fn(hidden, spec_out)
            else:
                corrected_out = spec_out
            break
        hidden = layer(hidden)

    logits_full = finish_forward(engine, full_out, target_layer)
    logits_corrected = finish_forward(engine, corrected_out, target_layer)

    return logits_full[0, -1, :], logits_corrected[0, -1, :]


def main():
    print("=" * 90)
    print("  Phase 4: Rank-1 Error Correction")
    print("  Can we fix sign-heavy layers with a single geometric correction?")
    print("=" * 90)
    print()

    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)

    for target_layer in [12, 23]:
        print(f"\n{'='*90}")
        print(f"  Layer {target_layer}")
        print(f"{'='*90}\n")

        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        # ═════════════════════════════════════════════════════════
        # Step 1: Collect calibration data
        # ═════════════════════════════════════════════════════════
        print(f"  Step 1: Collecting calibration data from {len(CALIBRATION_PROMPTS)} prompts...")
        X_cal, Y_full_cal, Y_spec_cal = collect_layer_data(
            engine, tokenizer, spec_layer, target_layer, CALIBRATION_PROMPTS)

        E_cal = Y_full_cal - Y_spec_cal  # Error matrix: what spec gets wrong
        print(f"    Samples: {E_cal.shape[0]} token positions × {E_cal.shape[1]} dims")
        print(f"    Mean |error|: {np.abs(E_cal).mean():.4f}")

        # ═════════════════════════════════════════════════════════
        # Step 2: SVD of error matrix
        # ═════════════════════════════════════════════════════════
        print(f"\n  Step 2: SVD of error matrix...")
        U_e, S_e, Vt_e = np.linalg.svd(E_cal, full_matrices=False)
        total_var = (S_e ** 2).sum()

        for k in [1, 2, 3, 5, 10]:
            captured = (S_e[:k] ** 2).sum() / total_var * 100
            print(f"    Rank-{k:2d}: {captured:.2f}% of error variance")

        print(f"    S[0]/S[1] ratio: {S_e[0]/S_e[1]:.1f}×")

        # The dominant error direction
        v_out = Vt_e[0]  # (D,) — output direction of error
        # The error is E = U @ diag(S) @ Vt
        # For rank-1: E ≈ S[0] * U[:,0] @ Vt[0] = S[0] * u_0 ⊗ v_0
        # But u_0 lives in sample space, not input space
        # We need: error(x) ≈ (x · v_in) * v_out
        # So: v_in = X.T @ U[:,0] * S[0] / (something)
        # Better: do regression E ≈ X @ W, then SVD of W

        # ═════════════════════════════════════════════════════════
        # Step 3: Learn input-conditioned correction
        # error(x) ≈ (x · v_in) · v_out
        # ═════════════════════════════════════════════════════════
        print(f"\n  Step 3: Learning rank-1 input-conditioned correction...")

        # Method A: Direct regression E ≈ X @ W, then rank-1 of W
        # W = pinv(X) @ E, but X is (N, D) with N < D, so ill-conditioned
        # Use SVD of X for stable pseudoinverse
        U_x, S_x, Vt_x = np.linalg.svd(X_cal, full_matrices=False)
        # Effective rank of X
        s_ratio = S_x / S_x[0]
        eff_rank_x = (s_ratio > 0.01).sum()
        print(f"    Input matrix effective rank: {eff_rank_x} (of {min(X_cal.shape)})")

        # Truncated pseudoinverse: use top-k singular components
        for k in [1, 5, 10, 20, 50]:
            if k > len(S_x):
                continue
            # W_k = Vt_x[:k].T @ diag(1/S_x[:k]) @ U_x[:,:k].T @ E
            X_proj = U_x[:, :k] * S_x[:k]  # (N, k)
            W_k_proj = np.linalg.lstsq(X_proj, E_cal, rcond=None)[0]  # (k, D)
            E_pred = X_proj @ W_k_proj
            explained = 1.0 - np.sum((E_cal - E_pred)**2) / np.sum(E_cal**2)
            print(f"    Input-rank-{k:2d} explains {explained*100:.2f}% of error (train)")

        # ═════════════════════════════════════════════════════════
        # Step 4: Stability analysis — split data and check direction
        # ═════════════════════════════════════════════════════════
        print(f"\n  Step 4: Stability analysis — does error direction generalize?")

        n = len(E_cal)
        perm = np.random.RandomState(42).permutation(n)
        half = n // 2
        E_a = E_cal[perm[:half]]
        E_b = E_cal[perm[half:]]

        _, S_a, Vt_a = np.linalg.svd(E_a, full_matrices=False)
        _, S_b, Vt_b = np.linalg.svd(E_b, full_matrices=False)

        # Cosine similarity of dominant error direction across halves
        cos_sim_1 = abs(np.dot(Vt_a[0], Vt_b[0]))
        cos_sim_2 = abs(np.dot(Vt_a[1], Vt_b[1]))
        cos_sim_3 = abs(np.dot(Vt_a[2], Vt_b[2]))

        # Also check: is the direction stable across DIFFERENT prompts?
        # Split by prompt (odd/even) rather than random
        E_odd = []
        E_even = []
        idx = 0
        for pi, prompt in enumerate(CALIBRATION_PROMPTS):
            n_tok = len(tokenizer.encode(prompt))
            if pi % 2 == 0:
                E_even.extend(range(idx, idx + n_tok))
            else:
                E_odd.extend(range(idx, idx + n_tok))
            idx += n_tok

        E_prompt_a = E_cal[E_even]
        E_prompt_b = E_cal[E_odd]

        _, _, Vt_pa = np.linalg.svd(E_prompt_a, full_matrices=False)
        _, _, Vt_pb = np.linalg.svd(E_prompt_b, full_matrices=False)
        cos_prompt = abs(np.dot(Vt_pa[0], Vt_pb[0]))

        print(f"    Random split — direction cosine similarity:")
        print(f"      Mode 1: {cos_sim_1:.6f}")
        print(f"      Mode 2: {cos_sim_2:.6f}")
        print(f"      Mode 3: {cos_sim_3:.6f}")
        print(f"    Prompt split — mode-1 cosine: {cos_prompt:.6f}")

        if cos_sim_1 > 0.99:
            print(f"    → Mode-1 direction is STABLE across data splits")
        elif cos_sim_1 > 0.9:
            print(f"    → Mode-1 direction is mostly stable")
        else:
            print(f"    → Mode-1 direction is UNSTABLE — rank-1 correction may not generalize")

        if cos_prompt > 0.99:
            print(f"    → Direction is STABLE across different prompts")
        elif cos_prompt > 0.9:
            print(f"    → Direction is mostly stable across prompts")
        else:
            print(f"    → Direction VARIES by prompt — correction needs to be prompt-dependent")

        # ═════════════════════════════════════════════════════════
        # Step 5: Build and test rank-1 correction
        # ═════════════════════════════════════════════════════════
        print(f"\n  Step 5: Testing rank-1 correction on held-out prompts...")

        # The correction: for input x, error ≈ (x · v_in) * v_out
        # v_out = Vt_e[0]  (dominant output direction)
        # Projection coefficients: c_i = E_i · v_out (how much error along v_out)
        # These should be linear in x: c_i ≈ x_i · v_in
        c = E_cal @ v_out          # (N,) — error magnitude per sample
        # v_in = argmin sum_i (x_i · v_in - c_i)^2
        # = (X.T X)^-1 X.T c ... but use lstsq
        v_in = np.linalg.lstsq(X_cal, c, rcond=None)[0]  # (D,)
        # Normalize: v_in_hat * scale
        v_in_norm = np.linalg.norm(v_in)
        v_in_hat = v_in / v_in_norm

        # Train fit quality
        c_pred = X_cal @ v_in
        train_r2 = 1.0 - np.sum((c - c_pred)**2) / np.sum((c - c.mean())**2)
        print(f"    v_in regression R² (train): {train_r2:.4f}")
        print(f"    v_in norm: {v_in_norm:.4f}")
        print(f"    |v_out|: {np.linalg.norm(v_out):.4f}")

        # Build correction function
        def rank1_correction(hidden_in, spec_out):
            """Apply rank-1 correction: spec_out + (hidden_in · v_in) * v_out"""
            corrected = spec_out.copy()
            for t in range(hidden_in.shape[1]):
                coeff = np.dot(hidden_in[0, t], v_in)
                corrected[0, t] += coeff * v_out
            return corrected

        # Also try rank-k correction using top components
        # For rank-k: error ≈ sum_j (x · v_in_j) * v_out_j
        k_corr = 5
        V_out_k = Vt_e[:k_corr]  # (k, D)
        C_k = E_cal @ V_out_k.T   # (N, k) — coefficients
        V_in_k = np.linalg.lstsq(X_cal, C_k, rcond=None)[0]  # (D, k)

        C_k_pred = X_cal @ V_in_k
        train_r2_k = 1.0 - np.sum((C_k - C_k_pred)**2) / np.sum((C_k - C_k.mean(0))**2)
        print(f"    Rank-{k_corr} correction R² (train): {train_r2_k:.4f}")

        def rankk_correction(hidden_in, spec_out):
            """Apply rank-k correction."""
            corrected = spec_out.copy()
            for t in range(hidden_in.shape[1]):
                coeffs = hidden_in[0, t] @ V_in_k  # (k,)
                corrected[0, t] += coeffs @ V_out_k
            return corrected

        # Test on held-out prompts
        print(f"\n    Held-out test results:")
        print(f"    {'Prompt':>40s}  {'Full':>8s}  {'Spec':>8s}  {'R1-Corr':>8s}  {'R5-Corr':>8s}")
        print(f"    {'-'*80}")

        n_pass_uncorr = 0
        n_pass_r1 = 0
        n_pass_rk = 0

        for prompt in TEST_PROMPTS:
            # Uncorrected
            logits_full_u, logits_uncorr = test_replacement(
                engine, tokenizer, spec_layer, target_layer, prompt)
            # Rank-1 corrected
            _, logits_r1 = test_replacement(
                engine, tokenizer, spec_layer, target_layer, prompt,
                correction_fn=rank1_correction)
            # Rank-k corrected
            _, logits_rk = test_replacement(
                engine, tokenizer, spec_layer, target_layer, prompt,
                correction_fn=rankk_correction)

            full_tok = tokenizer.decode_token(int(np.argmax(logits_full_u)))
            uncorr_tok = tokenizer.decode_token(int(np.argmax(logits_uncorr)))
            r1_tok = tokenizer.decode_token(int(np.argmax(logits_r1)))
            rk_tok = tokenizer.decode_token(int(np.argmax(logits_rk)))

            full_id = int(np.argmax(logits_full_u))
            match_u = "✓" if int(np.argmax(logits_uncorr)) == full_id else "✗"
            match_r1 = "✓" if int(np.argmax(logits_r1)) == full_id else "✗"
            match_rk = "✓" if int(np.argmax(logits_rk)) == full_id else "✗"

            if int(np.argmax(logits_uncorr)) == full_id:
                n_pass_uncorr += 1
            if int(np.argmax(logits_r1)) == full_id:
                n_pass_r1 += 1
            if int(np.argmax(logits_rk)) == full_id:
                n_pass_rk += 1

            # Show margin for the key test case
            extra = ""
            if "France" in prompt:
                full_sorted = np.sort(logits_full_u)[::-1]
                r1_sorted = np.sort(logits_r1)[::-1]
                rk_sorted = np.sort(logits_rk)[::-1]
                extra = (f"  margins: full={full_sorted[0]-full_sorted[1]:.3f} "
                        f"r1={r1_sorted[0]-r1_sorted[1]:.3f} "
                        f"r5={rk_sorted[0]-rk_sorted[1]:.3f}")

            print(f"    {prompt:>40s}  {full_tok:>8s}  "
                  f"{match_u}{uncorr_tok:>7s}  "
                  f"{match_r1}{r1_tok:>7s}  "
                  f"{match_rk}{rk_tok:>7s}{extra}")

        print(f"\n    Score: uncorrected={n_pass_uncorr}/{len(TEST_PROMPTS)}  "
              f"rank-1={n_pass_r1}/{len(TEST_PROMPTS)}  "
              f"rank-{k_corr}={n_pass_rk}/{len(TEST_PROMPTS)}")

        # ═════════════════════════════════════════════════════════
        # Step 6: Diagnostic — What IS the error direction?
        # ═════════════════════════════════════════════════════════
        print(f"\n  Step 6: What is the dominant error direction?")

        # Which dimensions contribute most to v_out?
        top_dims = np.argsort(np.abs(v_out))[::-1][:20]
        print(f"    Top-20 dimensions of v_out (error output direction):")
        for d in top_dims:
            print(f"      dim {d:4d}: {v_out[d]:+.6f}")

        # Is v_out sparse or dense?
        v_sorted = np.sort(np.abs(v_out))[::-1]
        cum_energy = np.cumsum(v_sorted**2) / (v_sorted**2).sum()
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            n_dims = (cum_energy < threshold).sum() + 1
            print(f"    {threshold*100:.0f}% of v_out energy in {n_dims}/3584 dims "
                  f"({n_dims/3584*100:.1f}%)")

        # Correlation of v_out with layer norms, residual stream properties
        v_out_abs = np.abs(v_out)
        print(f"    v_out sparsity: {(v_out_abs > 0.01).sum()}/3584 dims active")
        print(f"    v_out max: {v_out.max():.6f} at dim {np.argmax(v_out)}")
        print(f"    v_out min: {v_out.min():.6f} at dim {np.argmin(v_out)}")

        # Does v_out align with the layer norm weight?
        norm_weight = engine.layers[target_layer].attention.norm_weight
        cos_with_norm = abs(np.dot(v_out, norm_weight) /
                          (np.linalg.norm(v_out) * np.linalg.norm(norm_weight)))
        print(f"    |cos(v_out, attn_norm_weight)| = {cos_with_norm:.4f}")

        # What about the MLP norm?
        mlp_norm = engine.layers[target_layer].mlp.norm_weight
        cos_with_mlp = abs(np.dot(v_out, mlp_norm) /
                          (np.linalg.norm(v_out) * np.linalg.norm(mlp_norm)))
        print(f"    |cos(v_out, ffn_norm_weight)|  = {cos_with_mlp:.4f}")

        # Is v_in aligned with v_out? (would suggest the error is a projection)
        cos_in_out = abs(np.dot(v_in_hat, v_out / np.linalg.norm(v_out)))
        print(f"    |cos(v_in, v_out)| = {cos_in_out:.4f}")

    # ═════════════════════════════════════════════════════════
    # Final summary
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}\n")

    print("  The rank-1 correction captures the dominant error mode of the")
    print("  spectrometer layer approximation. Results above show whether")
    print("  this correction generalizes to unseen prompts.")
    print()
    print("  Key questions answered:")
    print("  1. Is the error direction stable? (Step 4)")
    print("  2. Does rank-1 correction fix the top-1 failures? (Step 5)")
    print("  3. What IS the error direction geometrically? (Step 6)")
    print()


if __name__ == '__main__':
    main()
