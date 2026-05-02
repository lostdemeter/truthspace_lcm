#!/usr/bin/env python3
"""
Sign Flip Pattern Analysis
===========================

Hypothesis: Sign flips have a pattern that traverses the model.
If the pattern aligns with model structure, it becomes implicit knowledge.

Like the Fibonacci sequence is implicit in φ:
  φ^n = F_n × φ + F_{n-1}

Maybe sign flips follow:
  flip_pattern[dim] = f(φ, dim_index)

We'll look for:
1. Which dimensions flip for each semantic axis
2. Is there a pattern to the flip indices?
3. Do the flip positions follow φ-scaling?
4. Does the pattern traverse layers consistently?
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)


def compute_flip_patterns(embeds: torch.Tensor, tokenizer, dimensions: Dict[str, List[Tuple[str, str]]]) -> Dict[str, torch.Tensor]:
    """Compute flip patterns for each semantic dimension."""
    flip_patterns = {}
    
    for dim_name, pairs in dimensions.items():
        flip_counts = torch.zeros(embeds.shape[1])
        n_pairs = 0
        
        for neg_word, pos_word in pairs:
            neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
            pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
            
            if not neg_ids or not pos_ids:
                continue
            
            neg_signs = torch.sign(embeds[neg_ids[0]])
            pos_signs = torch.sign(embeds[pos_ids[0]])
            
            flips = (neg_signs != pos_signs).float()
            flip_counts += flips
            n_pairs += 1
        
        if n_pairs > 0:
            flip_prob = flip_counts / n_pairs
            flip_patterns[dim_name] = flip_prob
    
    return flip_patterns


def analyze_flip_indices(flip_patterns: Dict[str, torch.Tensor], threshold: float = 0.5):
    """
    Analyze which dimension indices flip for each semantic axis.
    
    Look for patterns in the indices themselves.
    """
    print(f"\n--- FLIP INDEX ANALYSIS ---")
    
    all_flip_indices = {}
    
    for dim_name, flip_prob in flip_patterns.items():
        # Get indices where flip probability > threshold
        flip_mask = flip_prob > threshold
        flip_indices = torch.where(flip_mask)[0].tolist()
        
        all_flip_indices[dim_name] = flip_indices
        
        n_flips = len(flip_indices)
        total_dims = len(flip_prob)
        
        print(f"\n  {dim_name}:")
        print(f"    Flips: {n_flips} / {total_dims} ({n_flips/total_dims*100:.1f}%)")
        
        if n_flips > 0:
            # Analyze the indices
            indices = np.array(flip_indices)
            
            # Check for patterns
            # 1. Are they evenly spaced?
            if len(indices) > 1:
                diffs = np.diff(indices)
                mean_diff = diffs.mean()
                std_diff = diffs.std()
                print(f"    Mean spacing: {mean_diff:.1f} ± {std_diff:.1f}")
                
                # Check if spacing follows φ
                phi_ratio = mean_diff / PHI
                print(f"    Spacing / φ: {phi_ratio:.2f}")
            
            # 2. Do indices cluster at certain positions?
            # Divide into bins and check distribution
            n_bins = 10
            bin_size = total_dims // n_bins
            bin_counts = [0] * n_bins
            for idx in indices:
                bin_idx = min(idx // bin_size, n_bins - 1)
                bin_counts[bin_idx] += 1
            
            print(f"    Distribution across {n_bins} bins: {bin_counts}")
            
            # 3. First and last flip indices
            print(f"    First flip: {indices[0]}, Last flip: {indices[-1]}")
    
    return all_flip_indices


def analyze_flip_overlap(flip_patterns: Dict[str, torch.Tensor], threshold: float = 0.5):
    """
    Analyze overlap between flip patterns.
    
    The 70% overlap we saw - is it the SAME dimensions that flip?
    """
    print(f"\n--- FLIP OVERLAP ANALYSIS ---")
    
    dim_names = list(flip_patterns.keys())
    n_dims = len(dim_names)
    
    # Get flip masks
    flip_masks = {}
    for name, prob in flip_patterns.items():
        flip_masks[name] = prob > threshold
    
    # Compute overlap matrix
    print(f"\n  Overlap matrix (% of dims that flip in BOTH):")
    print(f"  {'':12s}", end="")
    for name in dim_names:
        print(f"{name[:8]:>10s}", end="")
    print()
    
    for name1 in dim_names:
        print(f"  {name1:12s}", end="")
        mask1 = flip_masks[name1]
        for name2 in dim_names:
            mask2 = flip_masks[name2]
            both = (mask1 & mask2).sum().item()
            either = (mask1 | mask2).sum().item()
            jaccard = both / either if either > 0 else 0
            print(f"{jaccard*100:10.1f}", end="")
        print()
    
    # Find the COMMON flip dimensions (flip in ALL semantic axes)
    common_mask = torch.ones(flip_masks[dim_names[0]].shape, dtype=torch.bool)
    for name in dim_names:
        common_mask &= flip_masks[name]
    
    common_indices = torch.where(common_mask)[0].tolist()
    print(f"\n  Dimensions that flip in ALL semantic axes: {len(common_indices)}")
    if len(common_indices) > 0 and len(common_indices) < 50:
        print(f"    Indices: {common_indices[:20]}...")
    
    # Find UNIQUE flip dimensions (flip in only ONE semantic axis)
    print(f"\n  Unique flip dimensions per axis:")
    for name in dim_names:
        unique_mask = flip_masks[name].clone()
        for other_name in dim_names:
            if other_name != name:
                unique_mask &= ~flip_masks[other_name]
        unique_count = unique_mask.sum().item()
        print(f"    {name}: {unique_count} unique flips")
    
    return common_indices


def analyze_phi_structure_in_flips(flip_patterns: Dict[str, torch.Tensor], threshold: float = 0.5):
    """
    Look for φ-structure in the flip patterns.
    
    Hypothesis: Flip positions might follow φ^k spacing.
    """
    print(f"\n--- φ-STRUCTURE IN FLIP PATTERNS ---")
    
    for dim_name, flip_prob in flip_patterns.items():
        flip_indices = torch.where(flip_prob > threshold)[0].float()
        
        if len(flip_indices) < 10:
            continue
        
        print(f"\n  {dim_name}:")
        
        # Check if indices follow φ^k pattern
        # If flip_index[i] ≈ c × φ^i, then log(flip_index) ≈ log(c) + i × log(φ)
        log_indices = torch.log(flip_indices + 1)  # +1 to avoid log(0)
        
        # Fit linear regression: log_index = a + b × i
        i_values = torch.arange(len(flip_indices), dtype=torch.float32)
        
        # Simple linear fit
        mean_i = i_values.mean()
        mean_log = log_indices.mean()
        
        numerator = ((i_values - mean_i) * (log_indices - mean_log)).sum()
        denominator = ((i_values - mean_i) ** 2).sum()
        
        if denominator > 0:
            slope = numerator / denominator
            intercept = mean_log - slope * mean_i
            
            # If slope ≈ log(φ), indices follow φ^k
            phi_ratio = slope.item() / LOG_PHI
            
            print(f"    Slope of log(index) vs position: {slope.item():.4f}")
            print(f"    log(φ) = {LOG_PHI:.4f}")
            print(f"    Slope / log(φ) = {phi_ratio:.2f}")
            
            if 0.8 < phi_ratio < 1.2:
                print(f"    *** POSSIBLE φ-SCALING DETECTED ***")


def analyze_layer_consistency(model, tokenizer, pairs: List[Tuple[str, str]]):
    """
    Check if flip patterns are consistent across layers.
    
    If the pattern traverses the model, it should be similar in each layer's attention.
    """
    print(f"\n--- LAYER CONSISTENCY ANALYSIS ---")
    
    # Get embeddings
    embeds = model.model.embed_tokens.weight.detach().float().cpu()
    
    # Compute flip pattern from embeddings
    embed_flips = torch.zeros(embeds.shape[1])
    n_pairs = 0
    
    for neg_word, pos_word in pairs:
        neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
        pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
        
        if not neg_ids or not pos_ids:
            continue
        
        neg_signs = torch.sign(embeds[neg_ids[0]])
        pos_signs = torch.sign(embeds[pos_ids[0]])
        
        flips = (neg_signs != pos_signs).float()
        embed_flips += flips
        n_pairs += 1
    
    if n_pairs > 0:
        embed_flips /= n_pairs
    
    # Now check attention weights in each layer
    # Do the attention weights have similar flip structure?
    print(f"\n  Comparing embedding flips to attention weight structure:")
    
    for layer_idx in [0, 7, 14, 21, 27]:
        attn = model.model.layers[layer_idx].self_attn
        W_q = attn.q_proj.weight.detach().float().cpu()
        
        # Get signs of W_q
        W_q_signs = torch.sign(W_q)
        
        # How many dimensions have consistent sign across the weight matrix?
        # (This would indicate the dimension is "stable" in that layer)
        sign_consistency = W_q_signs.float().mean(dim=0).abs()  # [hidden_dim]
        
        # Correlation between embedding flip pattern and layer sign consistency
        corr = torch.corrcoef(torch.stack([embed_flips, sign_consistency]))[0, 1]
        
        print(f"    Layer {layer_idx:2d}: corr(embed_flips, sign_consistency) = {corr.item():.4f}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("SIGN FLIP PATTERN ANALYSIS")
    print("="*70)
    print("""
Hypothesis: Sign flips have a pattern that traverses the model.
If aligned with model structure, it becomes implicit (like Fibonacci in φ).

Looking for:
1. Pattern in flip indices
2. φ-scaling in flip positions
3. Common vs unique flips across semantic axes
4. Consistency across layers
""")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    embeds = model.model.embed_tokens.weight.detach().float().cpu()
    
    # Semantic dimensions
    dimensions = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("leisurely", "swift")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive")],
        "age": [("young", "old"), ("new", "ancient")],
        "brightness": [("dark", "bright"), ("dim", "light")],
    }
    
    # Compute flip patterns
    flip_patterns = compute_flip_patterns(embeds, tokenizer, dimensions)
    
    # Analyze flip indices
    all_flip_indices = analyze_flip_indices(flip_patterns)
    
    # Analyze overlap
    common_indices = analyze_flip_overlap(flip_patterns)
    
    # Look for φ-structure
    analyze_phi_structure_in_flips(flip_patterns)
    
    # Check layer consistency
    all_pairs = []
    for pairs in dimensions.values():
        all_pairs.extend(pairs)
    analyze_layer_consistency(model, tokenizer, all_pairs)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)


if __name__ == "__main__":
    main()
