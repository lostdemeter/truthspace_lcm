#!/usr/bin/env python3
"""
Phase 4 Step 1-3: Extract hidden states from phi-engine + discover per-layer rules.

No PyTorch needed. Uses our phi-engine to extract hidden states, then runs
ContinuousPhaseDiscovery to classify each dimension's transformation rule.

Outputs:
  - results/phase4_rules/layer_XX.json  (per-layer rule files)
  - results/phase4_rules/summary.json   (aggregate statistics)
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
from phi_geometric.core.continuous_discovery import (
    ContinuousPhaseDiscovery,
    to_phi_levels,
)

MODEL_DIR = str(Path(__file__).parent / "phi_model")

# Diverse prompts for broad coverage of the hidden state space
PROMPTS = [
    "The capital of France is",
    "I went to the store and",
    "She said that she would",
    "The book is on the",
    "The largest planet is",
    "Water boils at",
    "Albert Einstein developed the",
    "The speed of light is",
    "In the beginning there was",
    "Once upon a time in a",
    "The quick brown fox jumps",
    "To be or not to be",
    "All that glitters is not",
    "The meaning of life is",
    "A journey of a thousand miles",
]


def extract_hidden_states(engine, tokenizer, prompts, verbose=True):
    """
    Run prompts through phi-engine and collect per-layer hidden states.

    Returns:
        hidden_states: dict[layer_idx] -> list of (seq_len, hidden_dim) arrays
        Layer 0 = post-embedding, layer i+1 = post-transformer-layer-i
    """
    all_hidden = {}
    n_layers = len(engine.layers)

    for idx, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            continue

        t0 = time.time()
        states = engine.forward_with_hidden_states(token_ids)
        dt = time.time() - t0

        for li, hs in enumerate(states):
            # hs shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            hs_2d = hs[0]
            all_hidden.setdefault(li, []).append(hs_2d)

        if verbose:
            print(f"  Prompt {idx+1:2d}/{len(prompts)}: "
                  f"{len(token_ids)} tokens, {dt:.1f}s  "
                  f"\"{prompt[:40]}...\"" if len(prompt) > 40 else
                  f"  Prompt {idx+1:2d}/{len(prompts)}: "
                  f"{len(token_ids)} tokens, {dt:.1f}s  "
                  f"\"{prompt}\"")

    return all_hidden


def run_spectrometer(hidden_states, num_dims=0, phi_scale=64,
                     context_radius=5, n_sample=200, verbose=True):
    """
    Run ContinuousPhaseDiscovery on each layer transition.

    Args:
        hidden_states: dict[layer_idx] -> list of arrays
        num_dims: dimensions to analyze (0 = ALL dimensions)
        phi_scale: phi-level resolution
        n_sample: max observation pairs per layer

    Returns:
        list of (layer_idx, ContinuousDiscoveryResult, selected_dims)
    """
    num_layers = max(hidden_states.keys())
    results = []

    for li in range(num_layers):
        if li not in hidden_states or (li + 1) not in hidden_states:
            continue

        t0 = time.time()

        # Stack all tokens from all prompts into (N, D) matrices
        all_in = np.concatenate(hidden_states[li], axis=0)    # (N, 3584)
        all_out = np.concatenate(hidden_states[li + 1], axis=0)  # (N, 3584)

        N, D = all_in.shape

        if num_dims > 0 and num_dims < D:
            # Sample dimensions (original behavior)
            delta = all_out - all_in
            dim_var = np.var(delta, axis=0)
            top_dims = np.argsort(dim_var)[-num_dims // 2:]
            rand_dims = np.random.choice(D, num_dims // 2, replace=False)
            selected = np.unique(np.concatenate([top_dims, rand_dims]))
            selected.sort()
        else:
            # ALL dimensions
            selected = np.arange(D)

        # Subsample observations if too many
        n = min(N, n_sample)
        idx = np.random.choice(N, n, replace=False) if N > n_sample else np.arange(N)

        # Run CPD
        cpd = ContinuousPhaseDiscovery(
            phi_scale=phi_scale,
            context_radius=context_radius,
            identity_threshold=1.0,
            affine_threshold=0.7,
        )
        for i in idx:
            cpd.add_pair(all_in[i, selected], all_out[i, selected])

        result = cpd.discover()
        results.append((li, result, selected))

        dt = time.time() - t0

        # Print summary
        rd = result.rule_distribution
        total = sum(rd.values())
        structured = 1.0 - rd.get('unstructured', 0) / total

        if verbose:
            zone = ""
            if li <= 2: zone = "DRUM"
            elif li == 3: zone = "TRANS"
            elif li <= 6: zone = "C-early"
            elif li <= 25: zone = "C-late"
            else: zone = "MUSIC"

            print(f"  Layer {li:2d} [{zone:>7s}]: "
                  f"R2={result.mean_r_squared:.3f}  "
                  f"struct={structured:.0%}  "
                  f"arch={result.archetype:<20s}  "
                  f"({dt:.1f}s)")

    return results


def save_rules(results, output_dir, verbose=True):
    """Save per-layer rules as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ALL_RULES = ['identity', 'scale', 'affine', 'quadratic', 'gating',
                 'sigmoid', 'cross_dim', 'context', 'collapse',
                 'sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate',
                 'unstructured']
    LINEAR = {'identity', 'scale', 'affine'}
    NONLINEAR = {'quadratic', 'gating', 'sigmoid'}
    SIGN = {'sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate'}

    summary_rows = []

    for layer_idx, result, selected_dims in results:
        # Build per-dimension rule data
        dim_rules = []
        for local_d, rule in sorted(result.dim_results.items()):
            global_d = int(selected_dims[local_d]) if local_d < len(selected_dims) else local_d
            dim_rules.append({
                'local_dim': local_d,
                'global_dim': global_d,
                'rule_type': rule.rule_type,
                'r_squared': float(rule.r_squared),
                'params': {k: (float(v) if isinstance(v, (np.floating, float)) else
                              int(v) if isinstance(v, (np.integer, int)) else
                              bool(v) if isinstance(v, (np.bool_, bool)) else
                              str(v))
                          for k, v in rule.params.items()},
            })

        # Save layer file
        layer_data = {
            'layer': layer_idx,
            'archetype': result.archetype,
            'mean_r_squared': float(result.mean_r_squared),
            'rule_distribution': {k: int(v) for k, v in result.rule_distribution.items()},
            'num_dims_analyzed': len(result.dim_results),
            'selected_dims': selected_dims.tolist(),
            'dim_rules': dim_rules,
        }

        layer_file = output_dir / f"layer_{layer_idx:02d}.json"
        with open(layer_file, 'w') as f:
            json.dump(layer_data, f, indent=2)

        # Build summary row
        rd = result.rule_distribution
        total = sum(rd.values())
        row = {
            'layer': layer_idx,
            'archetype': result.archetype,
            'r_squared': float(result.mean_r_squared),
            'num_dims': len(result.dim_results),
        }
        for rt in ALL_RULES:
            row[f'{rt}_pct'] = rd.get(rt, 0) / total
        row['linear_pct'] = sum(row.get(f'{t}_pct', 0) for t in LINEAR)
        row['nonlinear_pct'] = sum(row.get(f'{t}_pct', 0) for t in NONLINEAR)
        row['sign_pct'] = sum(row.get(f'{t}_pct', 0) for t in SIGN)
        row['structured_pct'] = 1.0 - row['unstructured_pct']
        summary_rows.append(row)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    if verbose:
        print(f"\n  Saved {len(results)} layer rule files to {output_dir}/")

    return summary_rows


def print_summary_table(summary_rows):
    """Print a formatted summary table."""
    print(f"\n{'='*100}")
    print("  SPECTROMETER RESULTS (from phi-engine hidden states)")
    print(f"{'='*100}")
    print(f"{'Lyr':>3} {'Archetype':<22} {'R2':>5} "
          f"{'lin%':>5} {'quad':>5} {'gate':>5} "
          f"{'sign':>5} {'unk%':>5} {'STRUC':>6}")
    print("-" * 100)

    for row in summary_rows:
        li = row['layer']
        zone = ""
        if li <= 2: zone = " DRM"
        elif li == 3: zone = " TRN"
        elif li <= 6: zone = " Ce"
        elif li <= 25: zone = " Cl"
        else: zone = " MUS"

        print(f"{li:>3} {row['archetype']:<22} {row['r_squared']:>5.3f} "
              f"{row['linear_pct']:>5.0%} {row['quadratic_pct']:>5.0%} "
              f"{row['gating_pct']:>5.0%} "
              f"{row['sign_pct']:>5.0%} "
              f"{row['unstructured_pct']:>5.0%} "
              f"{row['structured_pct']:>5.0%}{zone}")

    # Zone averages
    print(f"\n{'='*100}")
    print("  ZONE AVERAGES")
    print(f"{'='*100}")

    zones = {
        'DRUM (0-2)': [r for r in summary_rows if r['layer'] <= 2],
        'TRANSITION (3)': [r for r in summary_rows if r['layer'] == 3],
        'COMB-early (4-6)': [r for r in summary_rows if 4 <= r['layer'] <= 6],
        'COMB-late (7-25)': [r for r in summary_rows if 7 <= r['layer'] <= 25],
        'MUSIC (26-27)': [r for r in summary_rows if r['layer'] >= 26],
    }

    print(f"\n{'Zone':<20} {'R2':>5} {'linear':>7} {'nonlin':>7} "
          f"{'sign':>7} {'struct':>7} {'unk':>7}")
    print("-" * 70)
    for zname, rows in zones.items():
        if not rows:
            continue
        print(f"{zname:<20} "
              f"{np.mean([r['r_squared'] for r in rows]):>5.3f} "
              f"{np.mean([r['linear_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['nonlinear_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['sign_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['structured_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['unstructured_pct'] for r in rows]):>7.0%}")

    overall_struct = np.mean([r['structured_pct'] for r in summary_rows])
    overall_r2 = np.mean([r['r_squared'] for r in summary_rows])
    peak = max(summary_rows, key=lambda r: r['structured_pct'])

    print(f"\n  Overall: {overall_struct:.0%} structured, R2={overall_r2:.3f}")
    print(f"  Peak layer {peak['layer']}: {peak['structured_pct']:.0%} structured, "
          f"R2={peak['r_squared']:.3f}")


def main():
    print("=" * 100)
    print("  Phase 4: Spectrometer-Guided Rule Extraction")
    print("  (Using phi-engine hidden states — no PyTorch)")
    print("=" * 100)
    print()

    np.random.seed(42)

    # Load engine
    print("Step 1: Loading phi-engine...")
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=True)
    n_layers = len(engine.layers)

    # Load tokenizer
    tokenizer = Qwen2Tokenizer()
    print(f"  Tokenizer: {tokenizer.vocab_size} tokens\n")

    # Extract hidden states
    print(f"Step 2: Extracting hidden states ({len(PROMPTS)} prompts, "
          f"{n_layers} layers)...")
    t0 = time.time()
    hidden_states = extract_hidden_states(engine, tokenizer, PROMPTS)
    extract_time = time.time() - t0

    # Count total observations
    total_tokens = sum(hs.shape[0] for hs in hidden_states[0])
    print(f"\n  Extracted {total_tokens} token positions across "
          f"{len(PROMPTS)} prompts in {extract_time:.0f}s")

    # Run spectrometer on ALL 3584 dimensions
    print(f"\nStep 3: Running spectrometer on {n_layers} layer transitions "
          f"(ALL {engine.hidden_dim} dims)...")
    t0 = time.time()
    results = run_spectrometer(hidden_states, num_dims=0,
                               phi_scale=64, n_sample=200)
    spec_time = time.time() - t0
    print(f"\n  Spectrometer complete in {spec_time:.0f}s")

    # Save rules
    output_dir = Path(__file__).parent / "results" / "phase4_rules_full"
    print(f"\nStep 4: Saving rules...")
    summary_rows = save_rules(results, output_dir)

    # Print summary
    print_summary_table(summary_rows)

    print(f"\n  Total time: {extract_time + spec_time:.0f}s "
          f"(extract: {extract_time:.0f}s, spectrometer: {spec_time:.0f}s)")


if __name__ == '__main__':
    main()
