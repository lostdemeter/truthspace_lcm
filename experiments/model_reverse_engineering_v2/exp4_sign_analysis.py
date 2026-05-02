#!/usr/bin/env python3
"""
Experiment 4: Sign-Aware Spectrometer on Qwen2-7B

Tests the hypothesis that the "35% unstructured" dimensions are actually
structured in their SIGN patterns — the irreducible 1-bit boundary decisions
(doc 141: "Which side of hyperplane N?").

XOR of signs = boundary crossing computation. Even when magnitudes are
chaotic, signs may follow structured patterns.

New rule types:
  - sign_preserve:  output sign always matches input sign
  - sign_flip:      output sign always opposite to input sign
  - sign_xor:       output sign = XOR of input signs (cross-dim boundary)
  - sign_gate:      sign behavior depends on input magnitude threshold
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.continuous_discovery import (
    ContinuousPhaseDiscovery,
    to_phi_levels,
)

PHI = (1 + np.sqrt(5)) / 2

ALL_RULES = ['identity', 'scale', 'affine', 'quadratic', 'gating',
             'sigmoid', 'cross_dim', 'context', 'collapse',
             'sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate',
             'unstructured']
LINEAR = {'identity', 'scale', 'affine'}
NONLINEAR = {'quadratic', 'gating', 'sigmoid'}
CROSSDIM = {'cross_dim', 'context'}
SIGN = {'sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate'}


def load_and_extract(prompts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2-7B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    all_hidden = {}
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for li, hs in enumerate(outputs.hidden_states):
            hs_np = hs[0].cpu().float().numpy()
            all_hidden.setdefault(li, []).append(hs_np)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(prompts)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_hidden


def run_cpd(inputs_matrix, outputs_matrix, num_dims=256, phi_scale=64,
            context_radius=5, n_sample=100):
    N, D = inputs_matrix.shape
    delta = outputs_matrix - inputs_matrix
    dim_var = np.var(delta, axis=0)
    top_dims = np.argsort(dim_var)[-num_dims // 2:]
    rand_dims = np.random.choice(D, num_dims // 2, replace=False)
    selected = np.unique(np.concatenate([top_dims, rand_dims]))
    selected.sort()

    n = min(N, n_sample)
    idx = np.random.choice(N, n, replace=False)

    cpd = ContinuousPhaseDiscovery(
        phi_scale=phi_scale,
        context_radius=context_radius,
        identity_threshold=1.0,
        affine_threshold=0.7,
    )
    for i in idx:
        cpd.add_pair(inputs_matrix[i, selected], outputs_matrix[i, selected])

    return cpd.discover()


def make_row(result, layer_idx):
    rd = result.rule_distribution
    total = sum(rd.values())
    row = {'layer': layer_idx,
           'archetype': result.archetype,
           'r_squared': result.mean_r_squared}
    for rt in ALL_RULES:
        row[f'{rt}_pct'] = rd.get(rt, 0) / total
    row['linear_pct'] = sum(row[f'{t}_pct'] for t in LINEAR)
    row['nonlinear_pct'] = sum(row[f'{t}_pct'] for t in NONLINEAR)
    row['crossdim_pct'] = sum(row[f'{t}_pct'] for t in CROSSDIM)
    row['sign_pct'] = sum(row[f'{t}_pct'] for t in SIGN)
    row['structured_pct'] = 1.0 - row['unstructured_pct']
    return row


PROMPTS = [
    "I went to the store and",
    "She said that she would",
    "The book is on the",
    "The capital of France is",
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


def main():
    import json

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    np.random.seed(42)

    print("=" * 105)
    print("Experiment 4: Sign-Aware Spectrometer on Qwen2-7B")
    print("=" * 105)

    hidden_states = load_and_extract(PROMPTS)
    num_layers = len(hidden_states) - 1
    print(f"\n{num_layers} layers, {hidden_states[0][0].shape[-1]} dims\n")

    print(f"{'Lyr':>3} {'Archetype':<20} {'R²':>5} "
          f"{'lin%':>5} {'quad':>5} {'gate':>5} "
          f"{'s_pr':>5} {'s_fl':>5} {'s_xr':>5} {'s_gt':>5} "
          f"{'unk%':>5} {'STRUC':>6}")
    print("-" * 105)

    all_rows = []
    for li in range(num_layers):
        all_in = np.concatenate(hidden_states[li], axis=0)
        all_out = np.concatenate(hidden_states[li + 1], axis=0)
        result = run_cpd(all_in, all_out, num_dims=256, phi_scale=64,
                         context_radius=5, n_sample=100)
        row = make_row(result, li)
        all_rows.append(row)

        zone = ""
        if li <= 2: zone = " DRM"
        elif li == 3: zone = " TRN"
        elif li <= 6: zone = " Ce"
        elif li <= 25: zone = " Cl"
        else: zone = " MUS"

        print(f"{li:>3} {row['archetype']:<20} {row['r_squared']:>5.3f} "
              f"{row['linear_pct']:>5.0%} {row['quadratic_pct']:>5.0%} "
              f"{row['gating_pct']:>5.0%} "
              f"{row['sign_preserve_pct']:>5.0%} {row['sign_flip_pct']:>5.0%} "
              f"{row['sign_xor_pct']:>5.0%} {row['sign_gate_pct']:>5.0%} "
              f"{row['unstructured_pct']:>5.0%} "
              f"{row['structured_pct']:>5.0%}{zone}")

    # Summary
    print(f"\n{'='*105}")
    print("SUMMARY BY ZONE")
    print(f"{'='*105}")

    zones = {
        'DRUM (0-2)': [r for r in all_rows if r['layer'] <= 2],
        'TRANSITION (3)': [r for r in all_rows if r['layer'] == 3],
        'COMB-early (4-6)': [r for r in all_rows if 4 <= r['layer'] <= 6],
        'COMB-late (7-25)': [r for r in all_rows if 7 <= r['layer'] <= 25],
        'MUSIC (26-27)': [r for r in all_rows if r['layer'] >= 26],
    }

    print(f"\n{'Zone':<20} {'R²':>5} {'linear':>7} {'nonlin':>7} "
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

    # vs previous experiments
    total_struct = np.mean([r['structured_pct'] for r in all_rows])
    total_sign = np.mean([r['sign_pct'] for r in all_rows])
    total_r2 = np.mean([r['r_squared'] for r in all_rows])

    print(f"\nOverall: {total_struct:.0%} structured")
    print(f"  Sign rules contribute: {total_sign:.0%}")
    print(f"  R²: {total_r2:.3f}")
    print(f"\nComparison:")
    print(f"  Exp1c (affine-only):        ~30% structured")
    print(f"  Exp3  (+ nonlinear):        ~40% structured")
    print(f"  Exp4  (+ sign patterns):    {total_struct:.0%} structured")

    # Detailed sign analysis for peak layer
    peak_row = max(all_rows, key=lambda r: r['structured_pct'])
    print(f"\nPeak layer {peak_row['layer']}:")
    print(f"  Linear:      {peak_row['linear_pct']:.0%}")
    print(f"  Nonlinear:   {peak_row['nonlinear_pct']:.0%}")
    print(f"  Sign:        {peak_row['sign_pct']:.0%}")
    print(f"  Unstructured:{peak_row['unstructured_pct']:.0%}")
    print(f"  TOTAL:       {peak_row['structured_pct']:.0%}")

    out_file = output_dir / "exp4_sign_analysis.json"
    with open(out_file, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
