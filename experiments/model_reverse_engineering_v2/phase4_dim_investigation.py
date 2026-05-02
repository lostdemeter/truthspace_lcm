#!/usr/bin/env python3
"""
Investigate WHY the rank-1 correction failed.

Key discovery: position 0 has ~750x more error than other positions in
dims 2718/2730, dominating the SVD. The rank-1 correction was learning
to fix pos-0's catastrophe, NOT the last-token's subtle error.

This script:
  A) Confirms pos-0 catastrophe is universal across prompts
  B) Analyzes last-token error structure (the actual failure cause)
  C) Tests position-aware bias correction (exclude pos 0)
  D) Tests per-rule-type correction (fix unstructured dims only)
"""

import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_spectrometer import (
    SpectrometerLayer, load_all_rules,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")
RULES_DIR = str(Path(__file__).parent / "results" / "phase4_rules_full")


def finish_forward(engine, hidden_start, start_layer):
    h = hidden_start
    for layer in engine.layers:
        if layer.layer_idx > start_layer:
            h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    return engine.lm_head(h)


def main():
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)

    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "Once upon a time",
        "The quick brown fox",
        "Water boils at",
        "The largest planet is",
    ]

    # ═══════════════════════════════════════════════════════════
    # Part A: Pos-0 catastrophe — universal?
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Part A: Position-0 error catastrophe")
    print("=" * 80)
    print()

    for target_layer in [12, 23]:
        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        print(f"  Layer {target_layer}:")
        print(f"  {'Prompt':>30s} {'Tok0':>10s} {'Err pos0':>10s} "
              f"{'Err last':>10s} {'Ratio':>8s}")
        print(f"  {'-'*70}")

        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            hidden = engine.embedding(ids)[np.newaxis, :, :]
            for layer in engine.layers:
                if layer.layer_idx == target_layer:
                    full_out = layer(hidden.copy())
                    spec_out = spec_layer(hidden.copy())
                    break
                hidden = layer(hidden)

            err = spec_out - full_out
            err_0 = np.linalg.norm(err[0, 0, :])
            err_last = np.linalg.norm(err[0, -1, :])
            tok0 = tokenizer.decode_token(ids[0])
            ratio = err_0 / err_last if err_last > 0 else float('inf')
            print(f"  {prompt:>30s} {tok0:>10s} {err_0:>10.1f} "
                  f"{err_last:>10.1f} {ratio:>8.1f}x")
        print()

    # ═══════════════════════════════════════════════════════════
    # Part B: Last-token error structure
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Part B: Last-token error — where does it live?")
    print("=" * 80)
    print()

    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt)

    for target_layer in [12, 23, 5, 14]:
        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        hidden = engine.embedding(ids)[np.newaxis, :, :]
        for layer in engine.layers:
            if layer.layer_idx == target_layer:
                full_out = layer(hidden.copy())
                spec_out = spec_layer(hidden.copy())
                break
            hidden = layer(hidden)

        err = (spec_out - full_out)[0, -1, :]
        status = "FAIL" if target_layer in [12, 23] else "PASS"
        print(f"  Layer {target_layer} ({status}): last-token err norm={np.linalg.norm(err):.2f}")

        # Error concentration
        sorted_err2 = np.sort(err**2)[::-1]
        cum = np.cumsum(sorted_err2) / sorted_err2.sum()
        for pct in [0.5, 0.8, 0.9]:
            n = (cum < pct).sum() + 1
            print(f"    {pct*100:.0f}% of error² in {n}/3584 dims")

        # Top error dims with rule types
        with open(f'{RULES_DIR}/layer_{target_layer:02d}.json') as f:
            data = json.load(f)
        dim_rules = {r['global_dim']: r for r in data['dim_rules']}

        top_dims = np.argsort(np.abs(err))[::-1][:10]
        print(f"    Top-10 error dims:")
        for d in top_dims:
            rule = dim_rules.get(d, {})
            rt = rule.get('rule_type', 'N/A')
            r2 = rule.get('r_squared', 0)
            print(f"      dim {d:4d}: err={err[d]:+8.4f}  "
                  f"rule={rt:>15s}  R²={r2:.3f}")
        print()

    # ═══════════════════════════════════════════════════════════
    # Part C: Position-aware bias correction (exclude pos 0)
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Part C: Bias correction excluding position 0")
    print("=" * 80)
    print()

    cal_prompts = [
        "1 + 1 =", "2 + 2 =", "The sky is", "Water is made of",
        "The sun rises in the", "Once upon a time",
        "She walked into the room and", "He said that he would",
        "The quick brown fox", "In machine learning",
        "Python is a programming", "The function returns",
        "The largest planet is", "Albert Einstein developed the",
        "Shakespeare wrote many", "The speed of light is",
        "I think that we should", "She said that she would",
    ]

    test_prompts = [
        "The capital of France is",
        "The largest ocean is the",
        "The color of grass is",
        "Barack Obama was the",
        "To be or not to",
        "Roses are red, violets are",
    ]

    for target_layer in [12, 23]:
        rules = all_rules[target_layer]
        spec_layer = SpectrometerLayer(
            rules=rules, full_layer=engine.layers[target_layer],
            r2_threshold=0.7, mode='rules_only',
        )

        # Collect per-dim mean error excluding pos 0
        errors_nopos0 = []
        for p in cal_prompts:
            ids_p = tokenizer.encode(p)
            hidden = engine.embedding(ids_p)[np.newaxis, :, :]
            for layer in engine.layers:
                if layer.layer_idx == target_layer:
                    full_out = layer(hidden.copy())
                    spec_out = spec_layer(hidden.copy())
                    break
                hidden = layer(hidden)

            for t in range(1, len(ids_p)):  # SKIP position 0
                errors_nopos0.append((full_out - spec_out)[0, t, :])

        E_np0 = np.array(errors_nopos0)
        mean_err = E_np0.mean(axis=0)

        # SVD of pos-0-free errors
        _, S_np0, Vt_np0 = np.linalg.svd(E_np0, full_matrices=False)
        total = (S_np0**2).sum()
        r1_pct = (S_np0[0]**2) / total * 100
        print(f"  Layer {target_layer}: {E_np0.shape[0]} calibration samples (excl pos 0)")
        print(f"    Error rank-1: {r1_pct:.1f}% (S0/S1={S_np0[0]/S_np0[1]:.1f}x)")
        print(f"    Mean |bias|: {np.abs(mean_err).mean():.4f}")

        # Test on held-out prompts
        print(f"\n    {'Prompt':>35s}  {'Full':>8s}  {'Uncorr':>8s}  {'Corrected':>10s}")
        print(f"    {'-'*70}")

        n_pass_u = 0
        n_pass_c = 0

        for prompt in test_prompts:
            test_ids = tokenizer.encode(prompt)
            hidden = engine.embedding(test_ids)[np.newaxis, :, :]
            for layer in engine.layers:
                if layer.layer_idx == target_layer:
                    full_out = layer(hidden.copy())
                    spec_out = spec_layer(hidden.copy())
                    corrected = spec_out.copy()
                    for t in range(spec_out.shape[1]):
                        corrected[0, t, :] += mean_err
                    break
                hidden = layer(hidden)

            logits_full = finish_forward(engine, full_out, target_layer)
            logits_corr = finish_forward(engine, corrected, target_layer)
            logits_uncorr = finish_forward(engine, spec_out, target_layer)

            last_full = logits_full[0, -1, :]
            last_corr = logits_corr[0, -1, :]
            last_uncorr = logits_uncorr[0, -1, :]

            full_id = int(np.argmax(last_full))
            full_tok = tokenizer.decode_token(full_id)
            uncorr_tok = tokenizer.decode_token(int(np.argmax(last_uncorr)))
            corr_tok = tokenizer.decode_token(int(np.argmax(last_corr)))

            match_u = "✓" if int(np.argmax(last_uncorr)) == full_id else "✗"
            match_c = "✓" if int(np.argmax(last_corr)) == full_id else "✗"

            if int(np.argmax(last_uncorr)) == full_id: n_pass_u += 1
            if int(np.argmax(last_corr)) == full_id: n_pass_c += 1

            extra = ""
            if "France" in prompt:
                cs = np.sort(last_corr)[::-1]
                extra = f"  margin={cs[0]-cs[1]:.3f}"

            print(f"    {prompt:>35s}  {full_tok:>8s}  "
                  f"{match_u}{uncorr_tok:>7s}  "
                  f"{match_c}{corr_tok:>9s}{extra}")

        print(f"\n    Score: uncorr={n_pass_u}/{len(test_prompts)} "
              f"corrected={n_pass_c}/{len(test_prompts)}")
        print()


if __name__ == '__main__':
    main()
