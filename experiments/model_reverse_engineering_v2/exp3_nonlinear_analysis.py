#!/usr/bin/env python3
"""
Experiment 3: Full Non-Linear Spectrometer on Qwen2-7B

Re-runs the layer analysis and attention head analysis with the upgraded
ContinuousPhaseDiscovery that includes non-linear rule types:
- quadratic (MLP curvature)
- gating (SiLU/GELU piecewise linear)
- sigmoid (softmax-like saturation)
- cross_dim (attention mixing)

Goal: Capture the 55-80% of dimensions that were "unstructured" before.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.continuous_discovery import (
    ContinuousPhaseDiscovery,
    ContinuousDiscoveryResult,
    to_phi_levels,
)

PHI = (1 + np.sqrt(5)) / 2

RULE_TYPES = ['identity', 'scale', 'affine', 'quadratic', 'gating',
              'sigmoid', 'cross_dim', 'context', 'collapse', 'unstructured']
LINEAR_TYPES = {'identity', 'scale', 'affine'}
NONLINEAR_TYPES = {'quadratic', 'gating', 'sigmoid'}
CROSSDIM_TYPES = {'cross_dim', 'context'}


# ---------------------------------------------------------------------------
# Model loading + hidden state extraction
# ---------------------------------------------------------------------------

def load_and_extract(prompts: List[str]) -> Dict[int, List[np.ndarray]]:
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
        for layer_idx, hs in enumerate(outputs.hidden_states):
            hs_np = hs[0].cpu().float().numpy()
            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(hs_np)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(prompts)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_hidden


# ---------------------------------------------------------------------------
# Attention hook capture
# ---------------------------------------------------------------------------

class AttentionCapture:
    def __init__(self, model, target_layers):
        self.captures = {}
        self.hooks = []
        for li in target_layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(attn.register_forward_pre_hook(
                self._pre(li), with_kwargs=True))
            self.hooks.append(attn.register_forward_hook(self._post(li)))

    def _pre(self, li):
        def hook(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                self.captures.setdefault((li, 'pre'), []).append(
                    hs.detach().cpu().float())
        return hook

    def _post(self, li):
        def hook(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captures.setdefault((li, 'post'), []).append(
                out.detach().cpu().float())
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

    def get(self, li):
        pre = self.captures.get((li, 'pre'), [])
        post = self.captures.get((li, 'post'), [])
        if not pre or not post:
            return np.array([]), np.array([])
        return (torch.cat(pre, dim=1)[0].numpy(),
                torch.cat(post, dim=1)[0].numpy())


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def run_cpd(inputs_matrix, outputs_matrix, num_dims=256, phi_scale=64,
            context_radius=5, n_sample=100):
    """Run ContinuousPhaseDiscovery on (inputs, outputs) matrices."""
    N, D = inputs_matrix.shape

    # Select dimensions: half highest-variance delta, half random
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


def result_row(result, layer_idx, label=""):
    rd = result.rule_distribution
    total = sum(rd.values())
    row = {'layer': layer_idx, 'label': label,
           'archetype': result.archetype,
           'r_squared': result.mean_r_squared}
    for rt in RULE_TYPES:
        row[f'{rt}_pct'] = rd.get(rt, 0) / total
    row['linear_pct'] = sum(row[f'{t}_pct'] for t in LINEAR_TYPES)
    row['nonlinear_pct'] = sum(row[f'{t}_pct'] for t in NONLINEAR_TYPES)
    row['crossdim_pct'] = sum(row[f'{t}_pct'] for t in CROSSDIM_TYPES)
    row['structured_pct'] = 1.0 - row['unstructured_pct']
    return row


def print_header():
    print(f"{'Lyr':>3} {'Archetype':<20} {'R²':>5} "
          f"{'lin%':>5} {'quad':>5} {'gate':>5} {'sigm':>5} "
          f"{'xdim':>5} {'unk%':>5} {'STRUC':>6}")
    print("-" * 90)


def print_row(r):
    zone = ""
    li = r['layer']
    if li <= 2: zone = " DRM"
    elif li == 3: zone = " TRN"
    elif li <= 6: zone = " Ce"
    elif li <= 25: zone = " Cl"
    else: zone = " MUS"

    print(f"{li:>3} {r['archetype']:<20} {r['r_squared']:>5.3f} "
          f"{r['linear_pct']:>5.0%} {r['quadratic_pct']:>5.0%} "
          f"{r['gating_pct']:>5.0%} {r['sigmoid_pct']:>5.0%} "
          f"{r['crossdim_pct']:>5.0%} {r['unstructured_pct']:>5.0%} "
          f"{r['structured_pct']:>5.0%}{zone}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

ATTN_LAYERS = [0, 3, 5, 6, 14, 27]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import json

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    np.random.seed(42)

    print("=" * 90)
    print("Experiment 3: Full Non-Linear Spectrometer on Qwen2-7B")
    print("=" * 90)

    # --- Part A: Layer analysis ---
    hidden_states = load_and_extract(PROMPTS)
    num_layers = len(hidden_states) - 1
    print(f"\n{num_layers} layers, {hidden_states[0][0].shape[-1]} dims\n")

    print("PART A: FULL-LAYER ANALYSIS (hidden[L] → hidden[L+1])")
    print("=" * 90)
    print_header()

    layer_results = []
    for li in range(num_layers):
        all_in = np.concatenate(hidden_states[li], axis=0)
        all_out = np.concatenate(hidden_states[li + 1], axis=0)
        result = run_cpd(all_in, all_out, num_dims=256, phi_scale=64,
                         context_radius=5, n_sample=100)
        row = result_row(result, li)
        layer_results.append(row)
        print_row(row)

    # Summary
    print(f"\n{'='*90}")
    print("LAYER SUMMARY")
    print(f"{'='*90}")

    zones = {
        'DRUM (0-2)': [r for r in layer_results if r['layer'] <= 2],
        'TRANSITION (3)': [r for r in layer_results if r['layer'] == 3],
        'COMB-early (4-6)': [r for r in layer_results if 4 <= r['layer'] <= 6],
        'COMB-late (7-25)': [r for r in layer_results if 7 <= r['layer'] <= 25],
        'MUSIC (26-27)': [r for r in layer_results if r['layer'] >= 26],
    }

    print(f"\n{'Zone':<20} {'R²':>5} {'linear':>7} {'nonlin':>7} {'xdim':>7} {'struct':>7} {'unk':>7}")
    print("-" * 65)
    for zname, rows in zones.items():
        if not rows:
            continue
        print(f"{zname:<20} "
              f"{np.mean([r['r_squared'] for r in rows]):>5.3f} "
              f"{np.mean([r['linear_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['nonlinear_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['crossdim_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['structured_pct'] for r in rows]):>7.0%} "
              f"{np.mean([r['unstructured_pct'] for r in rows]):>7.0%}")

    # Improvement vs exp1c (affine-only)
    total_structured = np.mean([r['structured_pct'] for r in layer_results])
    total_r2 = np.mean([r['r_squared'] for r in layer_results])
    print(f"\nOverall: {total_structured:.0%} structured (was ~30% with affine-only)")
    print(f"Overall R²: {total_r2:.3f} (was ~0.456 with affine-only)")

    # --- Part B: Attention head analysis ---
    print(f"\n{'='*90}")
    print("PART B: ATTENTION HEAD ANALYSIS (selected layers)")
    print(f"{'='*90}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2-7B"
    print(f"\nReloading {model_name} for attention hooks...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    capture = AttentionCapture(model, ATTN_LAYERS)
    for idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    capture.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    head_results = []
    for li in ATTN_LAYERS:
        pre, post = capture.get(li)
        if pre.size == 0:
            continue

        zone = {0: "DRUM", 3: "TRANSITION", 5: "COMB-e", 6: "COMB-e",
                14: "COMB-l", 27: "MUSIC"}.get(li, "")

        print(f"\n--- Layer {li} ({zone}): {num_heads} heads ---")
        print(f"  {'Head':>4} {'Archetype':<20} {'R²':>5} "
              f"{'lin%':>5} {'quad':>5} {'gate':>5} {'sigm':>5} "
              f"{'xdim':>5} {'unk%':>5} {'STRUC':>6}")
        print(f"  {'-'*85}")

        layer_head_results = []
        for h in range(num_heads):
            h_in = pre[:, h*head_dim:(h+1)*head_dim]
            h_out = post[:, h*head_dim:(h+1)*head_dim]

            cpd = ContinuousPhaseDiscovery(
                phi_scale=64, context_radius=5,
                identity_threshold=1.0, affine_threshold=0.7,
            )
            n = min(h_in.shape[0], 80)
            idx = np.random.choice(h_in.shape[0], n, replace=False)
            for i in idx:
                cpd.add_pair(h_in[i], h_out[i])
            result = cpd.discover()

            row = result_row(result, li, label=f"head_{h}")
            row['head'] = h
            layer_head_results.append(row)
            head_results.append(row)

            print(f"  {h:>4} {row['archetype']:<20} {row['r_squared']:>5.3f} "
                  f"{row['linear_pct']:>5.0%} {row['quadratic_pct']:>5.0%} "
                  f"{row['gating_pct']:>5.0%} {row['sigmoid_pct']:>5.0%} "
                  f"{row['crossdim_pct']:>5.0%} {row['unstructured_pct']:>5.0%} "
                  f"{row['structured_pct']:>5.0%}")

        mean_struct = np.mean([r['structured_pct'] for r in layer_head_results])
        mean_r2 = np.mean([r['r_squared'] for r in layer_head_results])
        print(f"\n  Layer {li} mean: R²={mean_r2:.3f}, structured={mean_struct:.0%}")

    # --- Save ---
    save_data = {
        'layer_results': layer_results,
        'head_results': head_results,
    }
    out_file = output_dir / "exp3_nonlinear_analysis.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
