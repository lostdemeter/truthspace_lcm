#!/usr/bin/env python3
"""
Phase 4 Investigation: Why do layers 12 and 23 fail top-1 in single-layer replacement?

Both layers have high logit correlation (r=0.992, r=0.974) but still miss top-1.
This script investigates:
  1. What token does the full engine predict vs the spectrometer?
  2. How close are the top-1 and top-2 logits? (margin analysis)
  3. Which dimensions diverge most when these layers are replaced?
  4. What rule types dominate the diverging dimensions?
  5. Are there specific dimension patterns unique to these layers?
  6. Test across multiple prompts — is the failure consistent?
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
from phi_geometric.inference.phi_spectrometer import (
    SpectrometerRules, SpectrometerLayer, load_all_rules,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")
RULES_DIR = str(Path(__file__).parent / "results" / "phase4_rules_full")

FAIL_LAYERS = [12, 23]
# Include some passing layers for comparison
PASS_LAYERS = [5, 13, 14, 16, 17]

TEST_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Once upon a time",
    "The quick brown fox",
    "Water boils at",
    "The largest planet is",
    "Albert Einstein developed the",
    "She said that she would",
]


def run_single_replacement(engine, all_rules, layer_idx, token_ids):
    """Run forward with a single layer replaced. Return full & spec logits."""
    # Full forward
    hidden_full = engine.embedding(token_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        hidden_full = layer(hidden_full)
    hidden_full = rms_norm(hidden_full, engine.final_norm_weight)
    logits_full = engine.lm_head(hidden_full)

    # Spectrometer forward (replace one layer)
    hidden_spec = engine.embedding(token_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        li = layer.layer_idx
        if li == layer_idx and li in all_rules:
            rules = all_rules[li]
            spec_layer = SpectrometerLayer(
                rules=rules, full_layer=layer,
                r2_threshold=0.7, mode='rules_only',
            )
            hidden_spec = spec_layer(hidden_spec)
        else:
            hidden_spec = layer(hidden_spec)
    hidden_spec = rms_norm(hidden_spec, engine.final_norm_weight)
    logits_spec = engine.lm_head(hidden_spec)

    return logits_full, logits_spec


def analyze_hidden_divergence(engine, all_rules, layer_idx, token_ids):
    """
    Compute hidden state JUST after the replaced layer vs full.
    Returns per-dimension absolute difference.
    """
    # Full forward up to and including target layer
    hidden = engine.embedding(token_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        hidden = layer(hidden)
        if layer.layer_idx == layer_idx:
            hidden_full_post = hidden.copy()
            break

    # Spec forward up to and including target layer
    hidden = engine.embedding(token_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        li = layer.layer_idx
        if li == layer_idx and li in all_rules:
            rules = all_rules[li]
            spec_layer = SpectrometerLayer(
                rules=rules, full_layer=layer,
                r2_threshold=0.7, mode='rules_only',
            )
            hidden = spec_layer(hidden)
            hidden_spec_post = hidden.copy()
            break
        else:
            hidden = layer(hidden)

    # Per-dimension divergence (averaged over seq positions)
    diff = np.abs(hidden_full_post - hidden_spec_post)  # (1, seq, D)
    per_dim_diff = diff[0].mean(axis=0)  # (D,)

    # Relative divergence
    full_mag = np.abs(hidden_full_post[0]).mean(axis=0) + 1e-20
    per_dim_rel = per_dim_diff / full_mag

    return per_dim_diff, per_dim_rel, hidden_full_post, hidden_spec_post


def main():
    print("=" * 90)
    print("  Phase 4 Investigation: Why Layers 12 and 23 Fail Top-1")
    print("=" * 90)
    print()

    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)
    print(f"Loaded {len(engine.layers)} layers, {len(all_rules)} rule sets\n")

    # ═══════════════════════════════════════════════════════════════════
    # Test 1: Margin analysis — how close are top-1 and top-2?
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("  Test 1: Top-1/Top-2 Margin Analysis")
    print("=" * 90)

    test_ids = tokenizer.encode("The capital of France is")

    print(f"\n{'Layer':>5} {'Status':>6} {'Logit r':>8} "
          f"{'Full top1':>12} {'Spec top1':>12} "
          f"{'Full margin':>12} {'Spec margin':>12}")
    print("-" * 80)

    for li in FAIL_LAYERS + PASS_LAYERS:
        logits_full, logits_spec = run_single_replacement(
            engine, all_rules, li, test_ids)

        last_full = logits_full[0, -1, :]
        last_spec = logits_spec[0, -1, :]

        corr = np.corrcoef(last_full, last_spec)[0, 1]

        # Full engine margins
        full_sorted = np.sort(last_full)[::-1]
        full_top1_id = int(np.argmax(last_full))
        full_margin = full_sorted[0] - full_sorted[1]

        # Spec engine margins
        spec_sorted = np.sort(last_spec)[::-1]
        spec_top1_id = int(np.argmax(last_spec))
        spec_margin = spec_sorted[0] - spec_sorted[1]

        full_tok = tokenizer.decode_token(full_top1_id)
        spec_tok = tokenizer.decode_token(spec_top1_id)
        status = "PASS" if full_top1_id == spec_top1_id else "FAIL"

        print(f"{li:>5} {status:>6} {corr:>8.4f} "
              f"{full_tok:>12} {spec_tok:>12} "
              f"{full_margin:>12.3f} {spec_margin:>12.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Test 2: Multi-prompt consistency — is failure prompt-specific?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  Test 2: Multi-Prompt Consistency")
    print(f"{'='*90}\n")

    for li in FAIL_LAYERS:
        pass_count = 0
        fail_count = 0
        print(f"  Layer {li}:")
        for prompt in TEST_PROMPTS:
            ids = tokenizer.encode(prompt)
            logits_full, logits_spec = run_single_replacement(
                engine, all_rules, li, ids)
            full_top1 = int(np.argmax(logits_full[0, -1, :]))
            spec_top1 = int(np.argmax(logits_spec[0, -1, :]))
            match = full_top1 == spec_top1
            if match:
                pass_count += 1
            else:
                fail_count += 1
            status = "PASS" if match else "FAIL"
            f_tok = tokenizer.decode_token(full_top1)
            s_tok = tokenizer.decode_token(spec_top1)
            corr = np.corrcoef(logits_full[0, -1, :], logits_spec[0, -1, :])[0, 1]
            print(f"    [{status}] \"{prompt}\" -> "
                  f"full:\"{f_tok.strip()}\" spec:\"{s_tok.strip()}\" r={corr:.4f}")
        print(f"    Score: {pass_count}/{pass_count + fail_count}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # Test 3: Hidden state divergence — which dims diverge most?
    # ═══════════════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print("  Test 3: Hidden State Divergence Analysis")
    print(f"{'='*90}\n")

    test_ids = tokenizer.encode("The capital of France is")

    for li in FAIL_LAYERS + [5, 14]:  # Compare fail vs pass
        per_dim_diff, per_dim_rel, h_full, h_spec = analyze_hidden_divergence(
            engine, all_rules, li, test_ids)

        rules = all_rules.get(li)
        if rules is None:
            continue

        # Overall stats
        overall_corr = np.corrcoef(h_full.flatten(), h_spec.flatten())[0, 1]

        # Top 20 most divergent dimensions
        top_div_dims = np.argsort(per_dim_rel)[-20:][::-1]

        # Classify divergent dims by rule type
        rule_types_divergent = Counter()
        rule_types_all = Counter()
        for d, rule in rules.rules.items():
            rule_types_all[rule.rule_type] += 1
        for d in top_div_dims:
            if d in rules.rules:
                rule_types_divergent[rules.rules[d].rule_type] += 1
            else:
                rule_types_divergent['no_rule'] += 1

        print(f"  Layer {li} ({'FAIL' if li in FAIL_LAYERS else 'PASS'}):")
        print(f"    Hidden state corr after replacement: {overall_corr:.6f}")
        print(f"    Mean relative divergence: {per_dim_rel.mean():.4f}")
        print(f"    Max relative divergence:  {per_dim_rel.max():.4f}")
        print(f"    Dims with >50% relative error: "
              f"{(per_dim_rel > 0.5).sum()}/3584")
        print(f"    Dims with >100% relative error: "
              f"{(per_dim_rel > 1.0).sum()}/3584")

        # Rule type distribution in top-20 most divergent
        print(f"    Top-20 divergent dims rule types: {dict(rule_types_divergent)}")

        # Compare to overall distribution
        print(f"    Overall rule distribution: ", end="")
        total = sum(rule_types_all.values())
        top3 = rule_types_all.most_common(3)
        print(", ".join(f"{rt}={cnt/total:.0%}" for rt, cnt in top3))

        # Check if divergent dims cluster spatially
        if len(top_div_dims) > 1:
            diffs = np.diff(sorted(top_div_dims))
            print(f"    Top-20 dim spacing: min={diffs.min()}, "
                  f"max={diffs.max()}, mean={diffs.mean():.0f}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # Test 4: Rule quality comparison — fail vs pass layers
    # ═══════════════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print("  Test 4: Rule Quality Comparison (Fail vs Pass)")
    print(f"{'='*90}\n")

    print(f"{'Layer':>5} {'Status':>6} {'Coverage':>8} {'R2':>6} "
          f"{'Affine%':>8} {'Sign%':>8} {'Quad%':>8} {'Unstruct%':>10} "
          f"{'MeanAffR2':>10}")
    print("-" * 80)

    for li in FAIL_LAYERS + PASS_LAYERS:
        rules = all_rules.get(li)
        if not rules:
            continue

        cov = rules.coverage(0.7)
        status = "FAIL" if li in FAIL_LAYERS else "PASS"

        # Count rule types
        type_counts = Counter()
        r2_by_type = {}
        for d, rule in rules.rules.items():
            type_counts[rule.rule_type] += 1
            r2_by_type.setdefault(rule.rule_type, []).append(rule.r_squared)

        total = sum(type_counts.values())
        affine_pct = type_counts.get('affine', 0) / total
        sign_pct = (type_counts.get('sign_preserve', 0) +
                    type_counts.get('sign_flip', 0) +
                    type_counts.get('sign_xor', 0) +
                    type_counts.get('sign_gate', 0)) / total
        quad_pct = type_counts.get('quadratic', 0) / total
        unstruct_pct = type_counts.get('unstructured', 0) / total

        # Mean R2 for affine rules specifically
        affine_r2s = r2_by_type.get('affine', [0])
        mean_aff_r2 = np.mean(affine_r2s) if affine_r2s else 0

        print(f"{li:>5} {status:>6} {cov:>7.0%} {rules.mean_r2:>6.3f} "
              f"{affine_pct:>7.0%} {sign_pct:>7.0%} {quad_pct:>7.0%} "
              f"{unstruct_pct:>9.0%} {mean_aff_r2:>10.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Test 5: Logit rank analysis — where does the correct token end up?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  Test 5: Where Does the Correct Token Rank in Spec Output?")
    print(f"{'='*90}\n")

    test_ids = tokenizer.encode("The capital of France is")

    for li in FAIL_LAYERS:
        logits_full, logits_spec = run_single_replacement(
            engine, all_rules, li, test_ids)

        last_full = logits_full[0, -1, :]
        last_spec = logits_spec[0, -1, :]

        correct_id = int(np.argmax(last_full))
        correct_tok = tokenizer.decode_token(correct_id)

        # Where does the correct token rank in spec?
        spec_sorted_ids = np.argsort(last_spec)[::-1]
        correct_rank = int(np.where(spec_sorted_ids == correct_id)[0][0]) + 1

        # Show top-5 from both
        print(f"  Layer {li}:")
        print(f"    Correct token: \"{correct_tok.strip()}\" (id={correct_id})")
        print(f"    Correct token rank in spec: #{correct_rank}")
        print()
        print(f"    Full engine top-5:")
        for rank, tid in enumerate(np.argsort(last_full)[::-1][:5]):
            tok = tokenizer.decode_token(int(tid))
            print(f"      #{rank+1}: \"{tok.strip()}\" logit={last_full[tid]:.3f}")
        print(f"    Spec engine top-5:")
        for rank, tid in enumerate(np.argsort(last_spec)[::-1][:5]):
            tid = int(tid)
            tok = tokenizer.decode_token(tid)
            marker = " <<<" if tid == correct_id else ""
            print(f"      #{rank+1}: \"{tok.strip()}\" logit={last_spec[tid]:.3f}{marker}")
        print()


if __name__ == '__main__':
    main()
