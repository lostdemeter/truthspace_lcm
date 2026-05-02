#!/usr/bin/env python3
"""
Experiment 1c: Continuous Phase Discovery on Qwen2-7B Layers

Uses ContinuousPhaseDiscovery to analyze each layer transition
with proper φ-level rules (identity, scale, affine, context, collapse)
instead of discrete token mapping.

Key differences from exp1/exp1b:
- Works with φ-levels directly (no lossy quantization)
- Per-dimension rule discovery (not sequence-as-dimensions)
- Context rules look at neighboring dimensions
- Reports R² fit quality per dimension
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.continuous_discovery import (
    ContinuousPhaseDiscovery,
    ContinuousDiscoveryResult,
    to_phi_levels,
    from_phi_levels,
)

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Extract hidden states
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
            print(f"  Processed {idx + 1}/{len(prompts)}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_hidden


# ---------------------------------------------------------------------------
# Analysis: Direct layer transformation
# ---------------------------------------------------------------------------

def analyze_direct_transformation(
    hidden_in: List[np.ndarray],
    hidden_out: List[np.ndarray],
    layer_idx: int,
    num_dims: int = 256,
    phi_scale: int = 64,
    context_radius: int = 2,
) -> ContinuousDiscoveryResult:
    """
    Run ContinuousPhaseDiscovery on hidden[L] → hidden[L+1].
    
    Samples dimensions and tokens for tractability.
    """
    # Stack: (total_tokens, hidden_dim)
    all_in = np.concatenate(hidden_in, axis=0)
    all_out = np.concatenate(hidden_out, axis=0)
    
    N_tokens = all_in.shape[0]
    D = all_in.shape[1]
    
    # Select dimensions: mix of highest-variance delta and random
    delta = all_out - all_in
    dim_var = np.var(delta, axis=0)
    top_var_dims = np.argsort(dim_var)[-num_dims // 2:]
    rand_dims = np.random.choice(D, num_dims // 2, replace=False)
    selected_dims = np.unique(np.concatenate([top_var_dims, rand_dims]))
    selected_dims.sort()
    num_dims = len(selected_dims)
    
    # Subsample tokens
    n_sample = min(N_tokens, 100)
    token_indices = np.random.choice(N_tokens, n_sample, replace=False)
    
    in_subset = all_in[token_indices][:, selected_dims]   # (n_sample, num_dims)
    out_subset = all_out[token_indices][:, selected_dims]
    
    # Run ContinuousPhaseDiscovery
    cpd = ContinuousPhaseDiscovery(
        phi_scale=phi_scale,
        context_radius=context_radius,
        identity_threshold=1.0,   # φ-levels: ±1 level is close enough to identity
        affine_threshold=0.7,
    )
    
    for i in range(n_sample):
        cpd.add_pair(in_subset[i], out_subset[i])
    
    return cpd.discover()


# ---------------------------------------------------------------------------
# Analysis: Residual transformation
# ---------------------------------------------------------------------------

def analyze_residual_transformation(
    hidden_in: List[np.ndarray],
    hidden_out: List[np.ndarray],
    layer_idx: int,
    num_dims: int = 256,
    phi_scale: int = 64,
    context_radius: int = 2,
) -> ContinuousDiscoveryResult:
    """
    Run ContinuousPhaseDiscovery on hidden[L] → (hidden[L+1] - hidden[L]).
    This isolates what each layer ADDS to the representation.
    """
    all_in = np.concatenate(hidden_in, axis=0)
    all_out = np.concatenate(hidden_out, axis=0)
    residual = all_out - all_in
    
    N_tokens = all_in.shape[0]
    D = all_in.shape[1]
    
    dim_var = np.var(residual, axis=0)
    top_var_dims = np.argsort(dim_var)[-num_dims // 2:]
    rand_dims = np.random.choice(D, num_dims // 2, replace=False)
    selected_dims = np.unique(np.concatenate([top_var_dims, rand_dims]))
    selected_dims.sort()
    
    n_sample = min(N_tokens, 100)
    token_indices = np.random.choice(N_tokens, n_sample, replace=False)
    
    in_subset = all_in[token_indices][:, selected_dims]
    res_subset = residual[token_indices][:, selected_dims]
    
    cpd = ContinuousPhaseDiscovery(
        phi_scale=phi_scale,
        context_radius=context_radius,
        identity_threshold=1.0,
        affine_threshold=0.7,
    )
    
    for i in range(n_sample):
        cpd.add_pair(in_subset[i], res_subset[i])
    
    return cpd.discover()


# ---------------------------------------------------------------------------
# φ-Zipf analysis
# ---------------------------------------------------------------------------

def analyze_phi_zipf(hidden_in: List[np.ndarray], hidden_out: List[np.ndarray]) -> Dict:
    """Compute Zipf exponent of delta magnitude distribution."""
    all_in = np.concatenate(hidden_in, axis=0)
    all_out = np.concatenate(hidden_out, axis=0)
    
    # Mean delta per dimension
    mean_delta = np.mean(all_out - all_in, axis=0)
    abs_delta = np.abs(mean_delta)
    sorted_delta = np.sort(abs_delta)[::-1]
    
    top_k = min(200, len(sorted_delta))
    ranks = np.arange(1, top_k + 1)
    top_vals = sorted_delta[:top_k]
    
    if top_vals[0] > 1e-10:
        log_ranks = np.log(ranks)
        log_vals = np.log(top_vals + 1e-20)
        coeffs = np.polyfit(log_ranks, log_vals, 1)
        alpha = -coeffs[0]
    else:
        alpha = 0.0
    
    total = np.sum(abs_delta)
    top10 = np.sum(sorted_delta[:len(sorted_delta)//10])
    concentration = top10 / (total + 1e-20)
    
    return {
        'alpha': float(alpha),
        'concentration_top10': float(concentration),
        'delta_var': float(np.var(mean_delta)),
    }


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import json
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    print("=" * 70)
    print("Experiment 1c: Continuous Phase Discovery on Qwen2-7B")
    print("=" * 70)
    
    hidden_states = load_and_extract(PROMPTS)
    num_layers = len(hidden_states) - 1
    hidden_dim = hidden_states[0][0].shape[-1]
    print(f"\n{num_layers} layer transitions, {hidden_dim} dims\n")
    
    # Analyze each layer
    print("=" * 70)
    print("DIRECT: hidden[L] → hidden[L+1]")
    print("=" * 70)
    
    print(f"\n{'Layer':>5} {'Archetype':<22} {'R²':>6} "
          f"{'id%':>5} {'scl%':>5} {'aff%':>5} {'ctx%':>5} {'col%':>5} {'unk%':>5} "
          f"{'α':>6} {'δ_var':>10}")
    print("-" * 100)
    
    all_results = []
    
    for layer_idx in range(num_layers):
        result = analyze_direct_transformation(
            hidden_states[layer_idx],
            hidden_states[layer_idx + 1],
            layer_idx,
            num_dims=256,
            phi_scale=64,
            context_radius=2,
        )
        
        zipf = analyze_phi_zipf(
            hidden_states[layer_idx],
            hidden_states[layer_idx + 1],
        )
        
        rd = result.rule_distribution
        total = sum(rd.values())
        
        row = {
            'layer': layer_idx,
            'archetype': result.archetype,
            'r_squared': result.mean_r_squared,
            'identity_pct': rd.get('identity', 0) / total,
            'scale_pct': rd.get('scale', 0) / total,
            'affine_pct': rd.get('affine', 0) / total,
            'context_pct': rd.get('context', 0) / total,
            'collapse_pct': rd.get('collapse', 0) / total,
            'unstructured_pct': rd.get('unstructured', 0) / total,
            'zipf_alpha': zipf['alpha'],
            'delta_var': zipf['delta_var'],
        }
        all_results.append(row)
        
        zone = ""
        if layer_idx <= 2:
            zone = " DRUM"
        elif layer_idx == 3:
            zone = " TRANSITION"
        elif layer_idx >= 26:
            zone = " MUSIC"
        
        phi_zipf = "φ" if abs(zipf['alpha'] - 1/PHI) < 0.1 else " "
        
        print(f"{layer_idx:>5} {result.archetype:<22} {result.mean_r_squared:>6.3f} "
              f"{row['identity_pct']:>5.0%} {row['scale_pct']:>5.0%} "
              f"{row['affine_pct']:>5.0%} {row['context_pct']:>5.0%} "
              f"{row['collapse_pct']:>5.0%} {row['unstructured_pct']:>5.0%} "
              f"{zipf['alpha']:>5.2f}{phi_zipf} {zipf['delta_var']:>10.4f}"
              f"{zone}")
    
    # Residual analysis
    print(f"\n{'='*70}")
    print("RESIDUAL: hidden[L] → (hidden[L+1] - hidden[L])")
    print(f"{'='*70}")
    
    print(f"\n{'Layer':>5} {'Archetype':<22} {'R²':>6} "
          f"{'id%':>5} {'scl%':>5} {'aff%':>5} {'ctx%':>5} {'col%':>5} {'unk%':>5}")
    print("-" * 80)
    
    res_results = []
    
    for layer_idx in range(num_layers):
        result = analyze_residual_transformation(
            hidden_states[layer_idx],
            hidden_states[layer_idx + 1],
            layer_idx,
            num_dims=256,
            phi_scale=64,
            context_radius=2,
        )
        
        rd = result.rule_distribution
        total = sum(rd.values())
        
        row = {
            'layer': layer_idx,
            'archetype': result.archetype,
            'r_squared': result.mean_r_squared,
            'identity_pct': rd.get('identity', 0) / total,
            'scale_pct': rd.get('scale', 0) / total,
            'affine_pct': rd.get('affine', 0) / total,
            'context_pct': rd.get('context', 0) / total,
            'collapse_pct': rd.get('collapse', 0) / total,
            'unstructured_pct': rd.get('unstructured', 0) / total,
        }
        res_results.append(row)
        
        zone = ""
        if layer_idx <= 2:
            zone = " DRUM"
        elif layer_idx == 3:
            zone = " TRANSITION"
        elif layer_idx >= 26:
            zone = " MUSIC"
        
        print(f"{layer_idx:>5} {result.archetype:<22} {result.mean_r_squared:>6.3f} "
              f"{row['identity_pct']:>5.0%} {row['scale_pct']:>5.0%} "
              f"{row['affine_pct']:>5.0%} {row['context_pct']:>5.0%} "
              f"{row['collapse_pct']:>5.0%} {row['unstructured_pct']:>5.0%}"
              f"{zone}")
    
    # Meta-structure summary
    print(f"\n{'='*70}")
    print("META-STRUCTURE SUMMARY")
    print(f"{'='*70}")
    
    # Find archetype boundaries
    archetypes = [r['archetype'] for r in all_results]
    for i in range(1, len(archetypes)):
        if archetypes[i] != archetypes[i-1]:
            print(f"  Boundary at layer {i}: {archetypes[i-1]} → {archetypes[i]}")
    
    # R² profile
    r2_values = [r['r_squared'] for r in all_results]
    print(f"\n  R² range: {min(r2_values):.3f} - {max(r2_values):.3f}")
    print(f"  R² mean:  {np.mean(r2_values):.3f}")
    
    # Identity fraction profile
    id_fracs = [r['identity_pct'] for r in all_results]
    print(f"\n  Identity fraction range: {min(id_fracs):.0%} - {max(id_fracs):.0%}")
    
    # Zipf profile
    alphas = [r['zipf_alpha'] for r in all_results]
    phi_zipf_layers = [r['layer'] for r in all_results
                       if abs(r['zipf_alpha'] - 1/PHI) < 0.1]
    print(f"\n  Zipf α range: {min(alphas):.3f} - {max(alphas):.3f}")
    print(f"  Layers near φ-Zipf (α ≈ {1/PHI:.3f}): {phi_zipf_layers}")
    
    # Save
    save_data = {
        'direct': all_results,
        'residual': res_results,
    }
    results_file = output_dir / "exp1c_continuous_analysis.json"
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {results_file}")


if __name__ == "__main__":
    main()
