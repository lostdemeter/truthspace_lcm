#!/usr/bin/env python3
"""
Experiment 1b: Token Trajectory Analysis

REFRAME: Instead of treating dimensions as sequence positions,
treat LAYERS as the sequence. Each token travels through 28 layers.
The trajectory IS the transformation.

For each dimension d, we have a sequence of 29 values (embedding + 28 layers).
Quantized to φ-levels, this gives us a discrete trajectory.

We then ask PhaseDiscovery:
- Given trajectory[0:L] as input and trajectory[1:L+1] as output,
  what transformation pattern describes the layer-to-layer evolution?

We also group dimensions by their trajectory pattern to find
natural clusters (potential meta-structures).
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric import PhaseDiscovery

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# ---------------------------------------------------------------------------
# φ-level quantization
# ---------------------------------------------------------------------------

def to_phi_level(value: float) -> int:
    """Convert a single float to its nearest φ-lattice level."""
    if abs(value) < 1e-20:
        return 0
    sign = 1 if value > 0 else -1
    level = round(np.log(abs(value)) / LOG_PHI)
    return sign * level


def phi_quantize_trajectory(trajectory: np.ndarray, num_levels: int = 32) -> np.ndarray:
    """
    Quantize a trajectory (num_layers,) to discrete tokens.
    Uses rank-based quantization within each trajectory for robustness.
    """
    ranks = np.argsort(np.argsort(trajectory))  # rank transform
    tokens = (ranks * num_levels // len(trajectory)).astype(int)
    return np.clip(tokens, 0, num_levels - 1)


def phi_delta_quantize(trajectory: np.ndarray, num_levels: int = 16) -> np.ndarray:
    """
    Quantize the DELTAS (layer-to-layer changes) of a trajectory.
    This captures the transformation pattern rather than absolute values.
    """
    deltas = np.diff(trajectory)
    if np.std(deltas) < 1e-10:
        return np.zeros(len(deltas), dtype=int)
    
    # Normalize deltas to [-1, 1] range
    max_abs = np.max(np.abs(deltas)) + 1e-20
    normalized = deltas / max_abs
    
    # Quantize to num_levels bins
    half = num_levels // 2
    tokens = np.round(normalized * (half - 1)).astype(int) + half
    return np.clip(tokens, 0, num_levels - 1)


# ---------------------------------------------------------------------------
# Trajectory signature (for clustering)
# ---------------------------------------------------------------------------

def trajectory_signature(trajectory: np.ndarray) -> str:
    """
    Create a compact signature for a dimension's trajectory.
    
    Captures:
    - Direction of change at each layer (up/down/flat)
    - Magnitude class (small/medium/large change)
    """
    deltas = np.diff(trajectory)
    std = np.std(deltas) + 1e-20
    
    sig_chars = []
    for d in deltas:
        normalized = d / std
        if normalized > 1.0:
            sig_chars.append('U')   # big up
        elif normalized > 0.3:
            sig_chars.append('u')   # small up
        elif normalized < -1.0:
            sig_chars.append('D')   # big down
        elif normalized < -0.3:
            sig_chars.append('d')   # small down
        else:
            sig_chars.append('.')   # flat
    
    return ''.join(sig_chars)


# ---------------------------------------------------------------------------
# Extract hidden states
# ---------------------------------------------------------------------------

def load_model_and_extract(prompts: List[str]) -> Dict[int, List[np.ndarray]]:
    """Load Qwen2-7B and extract hidden states for all prompts."""
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
            print(f"  Processed {idx + 1}/{len(prompts)} prompts")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_hidden


# ---------------------------------------------------------------------------
# Analysis 1: Dimension trajectory clustering
# ---------------------------------------------------------------------------

def analyze_trajectory_clusters(hidden_states: Dict[int, List[np.ndarray]]):
    """
    For each dimension, compute its trajectory across layers.
    Cluster dimensions by trajectory signature.
    """
    num_layers = len(hidden_states)
    hidden_dim = hidden_states[0][0].shape[-1]
    
    # Average across all tokens and prompts to get mean trajectory per dimension
    mean_per_layer = []
    for layer_idx in range(num_layers):
        all_tokens = np.concatenate(hidden_states[layer_idx], axis=0)  # (total_tokens, hidden_dim)
        mean_per_layer.append(np.mean(all_tokens, axis=0))  # (hidden_dim,)
    
    trajectories = np.array(mean_per_layer)  # (num_layers, hidden_dim)
    
    print(f"\nTrajectory matrix: {trajectories.shape} (layers × dims)")
    
    # Compute signatures
    sig_counts = Counter()
    dim_signatures = {}
    for d in range(hidden_dim):
        traj = trajectories[:, d]
        sig = trajectory_signature(traj)
        sig_counts[sig] += 1
        dim_signatures[d] = sig
    
    # Top signature patterns
    print(f"\nTop 20 trajectory signatures (out of {len(sig_counts)} unique):")
    print(f"{'Signature':<35} {'Count':>6} {'%':>6}")
    print("-" * 50)
    for sig, count in sig_counts.most_common(20):
        pct = 100 * count / hidden_dim
        print(f"{sig:<35} {count:>6} {pct:>5.1f}%")
    
    # Where do signature changes concentrate?
    print(f"\nPer-layer change frequency (which layers cause the most change):")
    layer_change_counts = np.zeros(num_layers - 1)
    for d in range(hidden_dim):
        traj = trajectories[:, d]
        deltas = np.abs(np.diff(traj))
        # Count which layers have above-median change
        med = np.median(deltas)
        layer_change_counts += (deltas > med).astype(float)
    
    for l in range(num_layers - 1):
        bar = '#' * int(layer_change_counts[l] / hidden_dim * 50)
        print(f"  Layer {l:>2}→{l+1:>2}: {layer_change_counts[l]/hidden_dim:>5.1%} {bar}")
    
    return trajectories, dim_signatures


# ---------------------------------------------------------------------------
# Analysis 2: PhaseDiscovery on trajectory subsequences
# ---------------------------------------------------------------------------

def analyze_layer_phases(trajectories: np.ndarray, num_levels: int = 24):
    """
    Use PhaseDiscovery on dimension trajectories to find transformation phases.
    
    Framing: treat each dimension as a "word" that transforms through layers.
    Input = φ-quantized values at layers [0..N-1]
    Output = φ-quantized values at layers [1..N]
    
    This asks: "What is the one-step transformation rule for layer trajectories?"
    """
    num_layers, hidden_dim = trajectories.shape
    
    print(f"\n{'='*60}")
    print("PhaseDiscovery on trajectory shifts")
    print(f"{'='*60}")
    
    # Sample dimensions
    num_sample_dims = min(hidden_dim, 500)
    sample_dims = np.random.choice(hidden_dim, num_sample_dims, replace=False)
    
    # For each dimension, create (input, output) pairs from consecutive windows
    # Input:  quantized trajectory[t : t+window]
    # Output: quantized trajectory[t+1 : t+window+1]
    window = 5  # Look at 5-layer windows
    
    pairs = []
    for d in sample_dims:
        traj = trajectories[:, d]
        qtraj = phi_quantize_trajectory(traj, num_levels)
        
        for t in range(num_layers - window):
            inp = list(qtraj[t:t+window])
            out = list(qtraj[t+1:t+window+1])
            pairs.append((inp, out))
    
    print(f"  Generated {len(pairs)} (input, output) pairs")
    print(f"  Window size: {window} layers")
    print(f"  Vocabulary: {num_levels} φ-levels")
    
    # Run PhaseDiscovery
    pd = PhaseDiscovery(context_window=1, geometric=False)
    for inp, out in pairs:
        pd.add_pair(inp, out)
    
    result = pd.discover()
    nav = result.to_navigator()
    
    print(f"\n  Archetype: {result.archetype}")
    print(f"  Phases: {len(nav.phases)}")
    
    # Rule breakdown
    rule_types = Counter()
    for phase in nav.phases:
        for rule in phase.rules:
            rule_types[rule.rule_type] += 1
    print(f"  Rule types: {dict(rule_types)}")
    
    # Test accuracy
    correct = 0
    total = min(len(pairs), 200)
    for i in range(total):
        inp, expected = pairs[i]
        predicted = nav.execute(inp)
        if predicted == expected:
            correct += 1
    
    print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return result, nav


# ---------------------------------------------------------------------------
# Analysis 3: Per-layer delta patterns via PhaseDiscovery
# ---------------------------------------------------------------------------

def analyze_delta_patterns(trajectories: np.ndarray, num_levels: int = 24):
    """
    For each layer transition, ask PhaseDiscovery what the delta pattern is.
    
    Framing: For a group of dimensions, the INPUT is their φ-level at layer L,
    and the OUTPUT is the φ-level of the DELTA (change) at that layer.
    
    This asks: "Given a dimension's current value, what change does this layer apply?"
    """
    num_layers, hidden_dim = trajectories.shape
    
    print(f"\n{'='*60}")
    print("Per-layer delta patterns")
    print(f"{'='*60}")
    
    # Sample dimensions for tractability
    num_sample = min(hidden_dim, 300)
    sample_dims = np.random.choice(hidden_dim, num_sample, replace=False)
    sample_dims.sort()
    
    results_by_layer = []
    
    for layer_idx in range(num_layers - 1):
        vals_in = trajectories[layer_idx, sample_dims]
        vals_out = trajectories[layer_idx + 1, sample_dims]
        deltas = vals_out - vals_in
        
        # Quantize input values and deltas separately
        in_tokens = phi_quantize_trajectory(vals_in, num_levels)
        delta_tokens = phi_delta_quantize(
            np.concatenate([[0], deltas]),  # pad for diff
            num_levels,
        )
        # delta_quantize returns len-1, but we padded so it's correct length
        # Actually phi_delta_quantize takes diff internally, so let's just quantize deltas directly
        
        # Simpler: quantize deltas by rank
        delta_ranks = np.argsort(np.argsort(deltas))
        delta_tok = (delta_ranks * num_levels // num_sample).astype(int)
        delta_tok = np.clip(delta_tok, 0, num_levels - 1)
        
        # Group dimensions into windows of 8 for PhaseDiscovery
        group_size = 8
        pd = PhaseDiscovery(context_window=1, geometric=False)
        n_groups = num_sample // group_size
        
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            inp = list(in_tokens[start:end])
            out = list(delta_tok[start:end])
            pd.add_pair(inp, out)
        
        result = pd.discover()
        nav = result.to_navigator()
        
        # Count rule types
        rule_types = Counter()
        for phase in nav.phases:
            for rule in phase.rules:
                rule_types[rule.rule_type] += 1
        
        # Accuracy
        correct = 0
        for g in range(min(n_groups, 20)):
            start = g * group_size
            end = start + group_size
            inp = list(in_tokens[start:end])
            expected = list(delta_tok[start:end])
            predicted = nav.execute(inp)
            if predicted == expected:
                correct += 1
        
        acc = correct / min(n_groups, 20) if n_groups > 0 else 0
        
        # Compute some statistics about the deltas
        delta_std = np.std(deltas)
        identity_frac = rule_types.get('identity', 0) / max(sum(rule_types.values()), 1)
        
        results_by_layer.append({
            'layer': layer_idx,
            'archetype': result.archetype,
            'num_phases': len(nav.phases),
            'rule_types': dict(rule_types),
            'accuracy': acc,
            'delta_std': float(delta_std),
            'identity_fraction': identity_frac,
        })
        
        marker = ""
        if layer_idx in [0, 1, 2]:
            marker = " ← DRUM"
        elif layer_idx == 3:
            marker = " ← PHASE TRANSITION?"
        
        print(f"  Layer {layer_idx:>2}: {result.archetype:<20} "
              f"id={identity_frac:.0%} "
              f"δ_std={delta_std:.4f} "
              f"acc={acc:.0%}"
              f"{marker}")
    
    return results_by_layer


# ---------------------------------------------------------------------------
# Analysis 4: Variance explained per layer (φ-Zipf check)
# ---------------------------------------------------------------------------

def analyze_variance_structure(trajectories: np.ndarray):
    """
    For each layer transition, compute SVD of the delta matrix
    and check if singular values follow φ-Zipf.
    """
    num_layers, hidden_dim = trajectories.shape
    
    print(f"\n{'='*60}")
    print("Singular value structure of layer deltas")
    print(f"{'='*60}")
    
    for layer_idx in range(num_layers - 1):
        delta = trajectories[layer_idx + 1] - trajectories[layer_idx]
        
        # SVD of the delta (treating it as a 1×D matrix doesn't help)
        # Instead: compute variance profile
        var = np.var(delta)
        
        # More useful: magnitude distribution across dims
        abs_delta = np.abs(delta)
        sorted_delta = np.sort(abs_delta)[::-1]
        
        # Check Zipf: does delta[i] ∝ 1/i^α?
        top_k = min(100, hidden_dim)
        ranks = np.arange(1, top_k + 1)
        top_vals = sorted_delta[:top_k]
        
        if top_vals[0] > 1e-10:
            log_ranks = np.log(ranks)
            log_vals = np.log(top_vals + 1e-20)
            
            # Fit power law: log(val) = -α * log(rank) + c
            coeffs = np.polyfit(log_ranks, log_vals, 1)
            alpha = -coeffs[0]
            
            # What fraction of total delta is in top 10%?
            total = np.sum(abs_delta)
            top10pct = np.sum(sorted_delta[:hidden_dim//10])
            concentration = top10pct / (total + 1e-20)
        else:
            alpha = 0.0
            concentration = 0.0
        
        marker = ""
        if layer_idx in [0, 1, 2]:
            marker = " DRUM"
        elif layer_idx == 3:
            marker = " TRANSITION?"
        elif alpha > 0.5:
            marker = f" ← α > 1/φ!"
        
        phi_zipf_match = "φ-Zipf" if abs(alpha - 1/PHI) < 0.1 else ""
        
        print(f"  Layer {layer_idx:>2}: α={alpha:.3f} "
              f"top10%={concentration:.1%} "
              f"δ_var={var:.6f}"
              f" {phi_zipf_match}{marker}")


# ---------------------------------------------------------------------------
# Calibration prompts
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
    
    print("=" * 60)
    print("Experiment 1b: Token Trajectory Analysis")
    print("=" * 60)
    
    # Extract hidden states
    hidden_states = load_model_and_extract(PROMPTS)
    num_layers = len(hidden_states)
    hidden_dim = hidden_states[0][0].shape[-1]
    print(f"\n{num_layers} layers, {hidden_dim} dims")
    
    # Build trajectory matrix (average over all tokens)
    print("\nBuilding trajectory matrix...")
    trajectories, dim_sigs = analyze_trajectory_clusters(hidden_states)
    
    # Analysis 2: PhaseDiscovery on sliding windows
    analyze_layer_phases(trajectories)
    
    # Analysis 3: Per-layer delta patterns
    delta_results = analyze_delta_patterns(trajectories)
    
    # Analysis 4: Variance structure
    analyze_variance_structure(trajectories)
    
    # Save
    save_data = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "delta_analysis": delta_results,
    }
    results_file = output_dir / "exp1b_token_trajectories.json"
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
