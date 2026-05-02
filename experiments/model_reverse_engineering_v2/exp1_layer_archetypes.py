#!/usr/bin/env python3
"""
Experiment 1: Layer Archetype Identification

Captures hidden states at each layer boundary of Qwen2-7B,
quantizes them to φ-lattice discrete tokens, and runs PhaseDiscovery
to classify what transformation each layer performs.

The hypothesis: The 28 layers decompose into a small number of
transformation archetypes that PhaseDiscovery can identify.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric import PhaseDiscovery

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# ---------------------------------------------------------------------------
# Step 1: φ-lattice quantization (continuous → discrete tokens)
# ---------------------------------------------------------------------------

def phi_quantize(values: np.ndarray, num_levels: int = 64) -> np.ndarray:
    """
    Quantize continuous values to discrete φ-lattice tokens.
    
    Maps each value to: sign_bit * level
    where level = round(log_φ(|value|) * scale)
    
    Returns integer tokens in range [0, num_levels).
    """
    flat = values.flatten().astype(np.float64)
    
    # Encode sign as high bit
    signs = np.sign(flat)
    signs[signs == 0] = 1
    
    magnitudes = np.abs(flat) + 1e-20
    log_phi_vals = np.log(magnitudes) / LOG_PHI
    
    # Normalize to [0, num_levels/2) range
    p5, p95 = np.percentile(log_phi_vals, [5, 95])
    normalized = (log_phi_vals - p5) / (p95 - p5 + 1e-10)
    normalized = np.clip(normalized, 0, 1)
    
    half = num_levels // 2
    level = np.round(normalized * (half - 1)).astype(int)
    
    # Combine: positive → [0, half), negative → [half, num_levels)
    tokens = np.where(signs > 0, level, level + half)
    return tokens.reshape(values.shape)


# ---------------------------------------------------------------------------
# Step 2: Extract hidden states from Qwen2-7B
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2-7B"):
    """Load model with hooks to capture hidden states."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def extract_hidden_states(model, tokenizer, prompts: List[str]) -> Dict[str, np.ndarray]:
    """
    Run prompts through model and capture hidden states at every layer.
    
    Returns dict mapping layer_idx → (num_prompts, seq_len, hidden_dim) arrays.
    """
    all_hidden = {}
    
    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
        
        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        # Shape: (batch=1, seq_len, hidden_dim)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            hs_np = hs[0].cpu().float().numpy()  # (seq_len, hidden_dim)
            
            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(hs_np)
        
        if (prompt_idx + 1) % 5 == 0:
            print(f"  Processed {prompt_idx + 1}/{len(prompts)} prompts")
    
    return all_hidden


# ---------------------------------------------------------------------------
# Step 3: Run PhaseDiscovery on layer pairs
# ---------------------------------------------------------------------------

@dataclass
class LayerArchetype:
    """Result of PhaseDiscovery on a single layer's transformation."""
    layer_idx: int
    archetype: str
    num_phases: int
    rule_types: Dict[str, int]   # e.g. {"identity": 40, "consistent": 15, ...}
    accuracy: float
    sample_rules: List[str]


def analyze_layer_transformation(
    hidden_in: List[np.ndarray],   # list of (seq_len, hidden_dim) per prompt
    hidden_out: List[np.ndarray],
    layer_idx: int,
    num_dims: int = 64,
    num_levels: int = 64,
) -> LayerArchetype:
    """
    Run PhaseDiscovery on the transformation from layer_idx to layer_idx+1.
    
    We sample dimensions and quantize to make it tractable for PhaseDiscovery
    (which expects discrete token sequences).
    """
    # Select a subset of dimensions (the most varying ones)
    all_in = np.concatenate(hidden_in, axis=0)   # (total_tokens, hidden_dim)
    all_out = np.concatenate(hidden_out, axis=0)
    
    # Pick dimensions with highest variance in the delta
    delta = all_out - all_in  # residual
    dim_variance = np.var(delta, axis=0)
    top_dims = np.argsort(dim_variance)[-num_dims:]
    top_dims.sort()
    
    # Also track identity dimensions (lowest delta variance)
    bot_dims = np.argsort(dim_variance)[:num_dims]
    
    # Subsample tokens to keep PhaseDiscovery tractable
    n_tokens = min(all_in.shape[0], 200)
    indices = np.random.choice(all_in.shape[0], n_tokens, replace=False)
    
    in_subset = all_in[indices][:, top_dims]    # (n_tokens, num_dims)
    out_subset = all_out[indices][:, top_dims]
    
    # Quantize to discrete tokens
    in_tokens = phi_quantize(in_subset, num_levels)
    out_tokens = phi_quantize(out_subset, num_levels)
    
    # Run PhaseDiscovery
    pd = PhaseDiscovery(context_window=1, geometric=False)
    
    for i in range(n_tokens):
        inp = list(in_tokens[i])
        out = list(out_tokens[i])
        pd.add_pair(inp, out)
    
    result = pd.discover()
    nav = result.to_navigator()
    
    # Classify rules
    rule_types = {}
    sample_rules = []
    for phase in nav.phases:
        for rule in phase.rules:
            rtype = rule.rule_type
            rule_types[rtype] = rule_types.get(rtype, 0) + 1
            if len(sample_rules) < 5:
                out_val = rule.params.get('output', rule.input_value)
                sample_rules.append(f"{rtype}: {rule.input_value} → {out_val}")
    
    # Test accuracy
    correct = 0
    total = 0
    for i in range(min(n_tokens, 50)):
        inp = list(in_tokens[i])
        predicted = nav.execute(inp)
        expected = list(out_tokens[i])
        if predicted == expected:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return LayerArchetype(
        layer_idx=layer_idx,
        archetype=result.archetype,
        num_phases=len(nav.phases),
        rule_types=rule_types,
        accuracy=accuracy,
        sample_rules=sample_rules,
    )


# ---------------------------------------------------------------------------
# Step 4: Analyze residual vs direct transformation
# ---------------------------------------------------------------------------

def analyze_residual_transformation(
    hidden_in: List[np.ndarray],
    hidden_out: List[np.ndarray],
    layer_idx: int,
    num_dims: int = 64,
    num_levels: int = 64,
) -> LayerArchetype:
    """
    Instead of mapping hidden_in → hidden_out directly,
    map hidden_in → (hidden_out - hidden_in), i.e. the RESIDUAL.
    
    Transformers use residual connections: out = in + f(in).
    The residual IS the layer's actual contribution.
    """
    all_in = np.concatenate(hidden_in, axis=0)
    all_out = np.concatenate(hidden_out, axis=0)
    residual = all_out - all_in
    
    # Pick dims where residual has most structure
    dim_variance = np.var(residual, axis=0)
    top_dims = np.argsort(dim_variance)[-num_dims:]
    top_dims.sort()
    
    n_tokens = min(all_in.shape[0], 200)
    indices = np.random.choice(all_in.shape[0], n_tokens, replace=False)
    
    in_subset = all_in[indices][:, top_dims]
    res_subset = residual[indices][:, top_dims]
    
    in_tokens = phi_quantize(in_subset, num_levels)
    res_tokens = phi_quantize(res_subset, num_levels)
    
    pd = PhaseDiscovery(context_window=1, geometric=False)
    for i in range(n_tokens):
        pd.add_pair(list(in_tokens[i]), list(res_tokens[i]))
    
    result = pd.discover()
    nav = result.to_navigator()
    
    rule_types = {}
    sample_rules = []
    for phase in nav.phases:
        for rule in phase.rules:
            rtype = rule.rule_type
            rule_types[rtype] = rule_types.get(rtype, 0) + 1
            if len(sample_rules) < 5:
                out_val = rule.params.get('output', rule.input_value)
                sample_rules.append(f"{rtype}: {rule.input_value} → {out_val}")
    
    correct = 0
    total = 0
    for i in range(min(n_tokens, 50)):
        predicted = nav.execute(list(in_tokens[i]))
        expected = list(res_tokens[i])
        if predicted == expected:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return LayerArchetype(
        layer_idx=layer_idx,
        archetype=f"residual_{result.archetype}",
        num_phases=len(nav.phases),
        rule_types=rule_types,
        accuracy=accuracy,
        sample_rules=sample_rules,
    )


# ---------------------------------------------------------------------------
# Calibration prompts
# ---------------------------------------------------------------------------

CALIBRATION_PROMPTS = [
    # Scaffolding (syntactic, predictable)
    "I went to the store and",
    "She said that she would",
    "The book is on the",
    "He walked to the",
    "They were going to the",
    "It was a very nice",
    "We need to find a",
    "The cat sat on the",
    "I think that we should",
    "Please pass me the",
    
    # Content (semantic, requires world knowledge)
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The Mona Lisa was painted by",
    "The chemical symbol for gold is",
    "Albert Einstein developed the theory of",
    "The speed of light is approximately",
    "DNA stands for deoxyribonucleic",
    "The Great Wall of China was built to",
    "Shakespeare wrote Romeo and",
    
    # Mixed
    "In the beginning there was",
    "Once upon a time in a land far",
    "The quick brown fox jumps over the lazy",
    "To be or not to be that is the",
    "All that glitters is not",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import json
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Experiment 1: Layer Archetype Identification")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Extract hidden states
    print(f"\nExtracting hidden states from {len(CALIBRATION_PROMPTS)} prompts...")
    hidden_states = extract_hidden_states(model, tokenizer, CALIBRATION_PROMPTS)
    
    num_layers = len(hidden_states) - 1  # subtract 1 because we have input embedding too
    print(f"  Captured {num_layers} layer transitions")
    print(f"  Hidden dim: {hidden_states[0][0].shape[-1]}")
    
    # Free model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Analyze each layer
    print(f"\n{'='*70}")
    print("DIRECT TRANSFORMATION: hidden[L] → hidden[L+1]")
    print(f"{'='*70}")
    
    direct_results = []
    for layer_idx in range(num_layers):
        print(f"\n--- Layer {layer_idx} → {layer_idx + 1} ---")
        result = analyze_layer_transformation(
            hidden_states[layer_idx],
            hidden_states[layer_idx + 1],
            layer_idx,
            num_dims=48,
            num_levels=48,
        )
        direct_results.append(result)
        print(f"  Archetype: {result.archetype}")
        print(f"  Phases: {result.num_phases}")
        print(f"  Rule types: {result.rule_types}")
        print(f"  Accuracy: {result.accuracy:.1%}")
    
    # Analyze residuals
    print(f"\n{'='*70}")
    print("RESIDUAL TRANSFORMATION: hidden[L] → (hidden[L+1] - hidden[L])")
    print(f"{'='*70}")
    
    residual_results = []
    for layer_idx in range(num_layers):
        print(f"\n--- Layer {layer_idx} residual ---")
        result = analyze_residual_transformation(
            hidden_states[layer_idx],
            hidden_states[layer_idx + 1],
            layer_idx,
            num_dims=48,
            num_levels=48,
        )
        residual_results.append(result)
        print(f"  Archetype: {result.archetype}")
        print(f"  Phases: {result.num_phases}")
        print(f"  Rule types: {result.rule_types}")
        print(f"  Accuracy: {result.accuracy:.1%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print("\nDirect transformation archetypes:")
    print(f"{'Layer':>6} {'Archetype':<30} {'Phases':>6} {'Accuracy':>8}")
    print("-" * 55)
    for r in direct_results:
        print(f"{r.layer_idx:>6} {r.archetype:<30} {r.num_phases:>6} {r.accuracy:>8.1%}")
    
    print("\nResidual transformation archetypes:")
    print(f"{'Layer':>6} {'Archetype':<30} {'Phases':>6} {'Accuracy':>8}")
    print("-" * 55)
    for r in residual_results:
        print(f"{r.layer_idx:>6} {r.archetype:<30} {r.num_phases:>6} {r.accuracy:>8.1%}")
    
    # Look for the DRUM/COMB boundary
    print("\n\nMeta-structure analysis:")
    archetypes = [r.archetype for r in direct_results]
    for i in range(1, len(archetypes)):
        if archetypes[i] != archetypes[i-1]:
            print(f"  Archetype CHANGE at layer {i}: {archetypes[i-1]} → {archetypes[i]}")
    
    # Count archetype frequencies
    from collections import Counter
    arch_counts = Counter(archetypes)
    print(f"\n  Archetype distribution: {dict(arch_counts)}")
    
    # Save results
    save_data = {
        "direct": [
            {
                "layer": r.layer_idx,
                "archetype": r.archetype,
                "num_phases": r.num_phases,
                "rule_types": r.rule_types,
                "accuracy": r.accuracy,
            }
            for r in direct_results
        ],
        "residual": [
            {
                "layer": r.layer_idx,
                "archetype": r.archetype,
                "num_phases": r.num_phases,
                "rule_types": r.rule_types,
                "accuracy": r.accuracy,
            }
            for r in residual_results
        ],
    }
    
    results_file = output_dir / "exp1_layer_archetypes.json"
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
