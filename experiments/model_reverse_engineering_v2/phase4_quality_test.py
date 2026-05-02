#!/usr/bin/env python3
"""
Phase 4 Steps 4-6: Build spectrometer-guided engine and test quality.

Tests three modes:
  1. 'full':       standard engine (baseline)
  2. 'rules_only': spectrometer rules for structured dims, identity for rest
  3. 'hybrid':     full layer for unstructured, rules for structured

Compares:
  - Hidden state correlation per layer
  - Logit correlation
  - Top-1 token agreement
  - Per-token timing
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_spectrometer import (
    SpectrometerRules, SpectrometerLayer, load_all_rules,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")
RULES_DIR = str(Path(__file__).parent / "results" / "phase4_rules_full")


def build_spectrometer_engine(engine, rules_dir, mode='rules_only',
                               r2_threshold=0.7, verbose=True):
    """
    Replace engine layers with SpectrometerLayers.

    Returns the modified engine (in-place modification of layer list)
    and a report of coverage per layer.
    """
    all_rules = load_all_rules(rules_dir, engine.hidden_dim)

    report = []
    spec_layers = []

    for layer in engine.layers:
        li = layer.layer_idx
        if li in all_rules:
            rules = all_rules[li]
            coverage = rules.coverage(r2_threshold)
            spec_layer = SpectrometerLayer(
                rules=rules,
                full_layer=layer,
                r2_threshold=r2_threshold,
                mode=mode,
            )
            spec_layers.append(spec_layer)
            report.append({
                'layer': li,
                'coverage': coverage,
                'structured_dims': len(spec_layer.structured_dims),
                'unstructured_dims': len(spec_layer.unstructured_dims),
                'archetype': rules.archetype,
                'mean_r2': rules.mean_r2,
            })
            if verbose:
                print(f"  Layer {li:2d}: {coverage:.0%} coverage "
                      f"({len(spec_layer.structured_dims)} struct / "
                      f"{len(spec_layer.unstructured_dims)} unstruct)  "
                      f"R2={rules.mean_r2:.3f}")
        else:
            # No rules for this layer — use full computation
            spec_layers.append(layer)
            report.append({
                'layer': li,
                'coverage': 0.0,
                'structured_dims': 0,
                'unstructured_dims': engine.hidden_dim,
                'archetype': 'none',
                'mean_r2': 0.0,
            })
            if verbose:
                print(f"  Layer {li:2d}: no rules — full computation")

    return spec_layers, report


def compare_outputs(engine, spec_layers, token_ids, label=""):
    """
    Compare full engine vs spectrometer engine on the same input.

    Returns comparison metrics.
    """
    from phi_geometric.inference.phi_components import rms_norm

    # --- Full engine forward ---
    t0 = time.time()
    hidden_full = engine.embedding(token_ids)
    hidden_full = hidden_full[np.newaxis, :, :]

    full_states = [hidden_full.copy()]
    for layer in engine.layers:
        hidden_full = layer(hidden_full)
        full_states.append(hidden_full.copy())

    hidden_full = rms_norm(hidden_full, engine.final_norm_weight)
    logits_full = engine.lm_head(hidden_full)
    full_time = time.time() - t0

    # --- Spectrometer forward ---
    t0 = time.time()
    hidden_spec = engine.embedding(token_ids)
    hidden_spec = hidden_spec[np.newaxis, :, :]

    spec_states = [hidden_spec.copy()]
    for spec_layer in spec_layers:
        hidden_spec = spec_layer(hidden_spec)
        spec_states.append(hidden_spec.copy())

    hidden_spec = rms_norm(hidden_spec, engine.final_norm_weight)
    logits_spec = engine.lm_head(hidden_spec)
    spec_time = time.time() - t0

    # --- Compare ---
    results = {
        'label': label,
        'full_time': full_time,
        'spec_time': spec_time,
        'speedup': full_time / max(spec_time, 0.001),
    }

    # Per-layer hidden state correlation
    layer_corrs = []
    for i in range(1, min(len(full_states), len(spec_states))):
        f = full_states[i].flatten()
        s = spec_states[i].flatten()
        corr = float(np.corrcoef(f, s)[0, 1])
        layer_corrs.append(corr)
    results['layer_correlations'] = layer_corrs

    # Final logit comparison
    last_full = logits_full[0, -1, :]
    last_spec = logits_spec[0, -1, :]
    results['logit_correlation'] = float(np.corrcoef(last_full, last_spec)[0, 1])
    results['logit_max_diff'] = float(np.max(np.abs(last_full - last_spec)))

    # Top-k agreement
    full_top1 = int(np.argmax(last_full))
    spec_top1 = int(np.argmax(last_spec))
    results['top1_match'] = full_top1 == spec_top1
    results['full_top1'] = full_top1
    results['spec_top1'] = spec_top1

    full_top10 = set(np.argsort(last_full)[-10:])
    spec_top10 = set(np.argsort(last_spec)[-10:])
    results['top10_agreement'] = len(full_top10 & spec_top10) / 10

    return results


def main():
    print("=" * 90)
    print("  Phase 4: Progressive Layer Replacement Test")
    print("=" * 90)
    print()

    # Load engine + tokenizer
    print("Loading engine...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"  {len(engine.layers)} layers, {engine.hidden_dim}d\n")

    # Load all rules and rank layers by coverage
    all_rules = load_all_rules(RULES_DIR, engine.hidden_dim)
    layer_coverage = []
    for li, rules in sorted(all_rules.items()):
        cov = rules.coverage(0.7)
        layer_coverage.append((li, cov, rules.mean_r2))
    layer_coverage.sort(key=lambda x: -x[1])  # Most structured first

    print("Layer ranking by coverage (most structured first):")
    for li, cov, r2 in layer_coverage[:10]:
        print(f"  Layer {li:2d}: {cov:.0%} coverage, R2={r2:.3f}")
    print()

    test_prompt = "The capital of France is"
    test_ids = tokenizer.encode(test_prompt)

    # Progressive replacement: replace 0, 1, 3, 5, 10, 15, 20, 28 layers
    replace_counts = [0, 1, 3, 5, 10, 15, 20, 28]

    print(f"Test prompt: \"{test_prompt}\" -> {test_ids}")
    print(f"\n{'Replaced':>8} {'Layers':>30} {'Logit r':>8} {'Top1':>6} "
          f"{'Top10':>6} {'Full(s)':>8} {'Spec(s)':>8} {'Speedup':>8}")
    print("-" * 90)

    for n_replace in replace_counts:
        # Build layer list: replace the top-N most structured layers
        replaced_set = set()
        for i in range(min(n_replace, len(layer_coverage))):
            replaced_set.add(layer_coverage[i][0])

        spec_layers = []
        for layer in engine.layers:
            li = layer.layer_idx
            if li in replaced_set and li in all_rules:
                rules = all_rules[li]
                spec_layer = SpectrometerLayer(
                    rules=rules, full_layer=layer,
                    r2_threshold=0.7, mode='rules_only',
                )
                spec_layers.append(spec_layer)
            else:
                spec_layers.append(layer)

        results = compare_outputs(engine, spec_layers, test_ids,
                                  label=f"{n_replace} layers")

        # Format replaced layer list
        replaced_str = ",".join(str(x) for x in sorted(replaced_set)[:8])
        if len(replaced_set) > 8:
            replaced_str += f"...+{len(replaced_set)-8}"

        logit_r = results['logit_correlation']
        logit_str = f"{logit_r:.4f}" if not np.isnan(logit_r) else "NaN"

        full_tok = tokenizer.decode_token(results['full_top1'])
        spec_tok = tokenizer.decode_token(results['spec_top1'])
        match = "YES" if results['top1_match'] else f"NO({spec_tok.strip()[:8]})"

        print(f"{n_replace:>8} {replaced_str:>30} {logit_str:>8} {match:>6} "
              f"{results['top10_agreement']:>5.0%} "
              f"{results['full_time']:>8.1f} {results['spec_time']:>8.2f} "
              f"{results['speedup']:>7.0f}x")

    # Detailed single-layer replacement sweep
    print(f"\n{'='*90}")
    print("  Single-layer replacement: which layer hurts most?")
    print(f"{'='*90}\n")

    print(f"{'Layer':>5} {'Coverage':>8} {'R2':>6} {'Logit r':>8} {'Top1':>6} "
          f"{'Top10':>6} {'Spec(s)':>8}")
    print("-" * 60)

    for li, cov, r2 in layer_coverage[:15]:
        spec_layers = []
        for layer in engine.layers:
            if layer.layer_idx == li and li in all_rules:
                rules = all_rules[li]
                spec_layer = SpectrometerLayer(
                    rules=rules, full_layer=layer,
                    r2_threshold=0.7, mode='rules_only',
                )
                spec_layers.append(spec_layer)
            else:
                spec_layers.append(layer)

        results = compare_outputs(engine, spec_layers, test_ids,
                                  label=f"layer {li}")

        logit_r = results['logit_correlation']
        logit_str = f"{logit_r:.6f}" if not np.isnan(logit_r) else "NaN"

        spec_tok = tokenizer.decode_token(results['spec_top1'])
        match = "YES" if results['top1_match'] else f"NO({spec_tok.strip()[:8]})"

        print(f"{li:>5} {cov:>7.0%} {r2:>6.3f} {logit_str:>8} {match:>6} "
              f"{results['top10_agreement']:>5.0%} "
              f"{results['spec_time']:>8.1f}")


if __name__ == '__main__':
    main()
