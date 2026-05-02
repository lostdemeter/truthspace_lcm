#!/usr/bin/env python3
"""
Semantic Axes: One Axis Per Dimension, Not Per Word
====================================================

The insight: Instead of per-word flip patterns, store ONE AXIS per semantic dimension.

For each dimension (temperature, size, speed, etc.):
  - Learn the axis direction from training pairs
  - Navigation: project onto axis, flip sign, find nearest

This is the minimal representation:
  - 10 dimensions × 3584 floats = 35,840 values
  - vs 152,064 tokens × 3584 floats = 545M values

The axis IS the structure. Everything else is implicit.
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2


class SemanticAxes:
    """
    Store one axis per semantic dimension.
    
    Navigation:
    1. Project word onto the dimension's axis
    2. Flip the projection (negate)
    3. Find the word with closest projection
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach().float()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        # Per-dimension axes
        self.axes: Dict[str, torch.Tensor] = {}
        
        # Precompute projections for fast lookup
        self.projections: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def learn_axis(self, dim_name: str, pairs: List[Tuple[str, str]]):
        """
        Learn the axis for a semantic dimension from word pairs.
        
        The axis is the average direction from negative to positive words.
        """
        directions = []
        
        for neg_word, pos_word in pairs:
            e_neg = self.get_embedding(neg_word)
            e_pos = self.get_embedding(pos_word)
            
            if e_neg is None or e_pos is None:
                continue
            
            diff = (e_pos - e_neg).cpu()
            norm = diff.norm()
            if norm > 1e-6:
                directions.append(diff / norm)
        
        if not directions:
            return
        
        # Average direction
        axis = torch.stack(directions).mean(dim=0)
        axis = axis / axis.norm()
        self.axes[dim_name] = axis
        
        # Precompute projections for all tokens
        axis_device = axis.to(self.device)
        self.projections[dim_name] = (self.all_embeds @ axis_device).cpu()
        
        print(f"  {dim_name}: axis learned from {len(directions)} pairs")
    
    def find_opposite(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite by flipping projection on the dimension's axis.
        """
        if dim_name not in self.axes:
            return None
        
        word_id = self.get_token_id(word)
        if word_id is None:
            return None
        
        # Get word's projection on this axis
        word_proj = self.projections[dim_name][word_id].item()
        
        # Target projection is the negative
        target_proj = -word_proj
        
        # Find word with closest projection to target
        all_projs = self.projections[dim_name]
        distances = (all_projs - target_proj).abs()
        distances[word_id] = float('inf')  # Exclude source
        
        # Get top candidates
        top_indices = distances.topk(100, largest=False).indices
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 3 and result_word.islower():
                return (result_word, all_projs[idx].item())
        
        # Fallback
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 2:
                return (result_word, all_projs[idx].item())
        
        return None
    
    def find_opposite_auto(self, word: str) -> Optional[Tuple[str, float, str]]:
        """
        Automatically detect which dimension the word belongs to and find opposite.
        """
        word_id = self.get_token_id(word)
        if word_id is None:
            return None
        
        # Find dimension with largest absolute projection
        best_dim = None
        best_proj = 0
        
        for dim_name, projs in self.projections.items():
            proj = abs(projs[word_id].item())
            if proj > best_proj:
                best_proj = proj
                best_dim = dim_name
        
        if best_dim is None:
            return None
        
        result = self.find_opposite(word, best_dim)
        if result:
            return (result[0], result[1], best_dim)
        return None


def demo_semantic_axes(model, tokenizer):
    """Demo semantic axes."""
    print("="*70)
    print("SEMANTIC AXES: ONE AXIS PER DIMENSION")
    print("="*70)
    print("""
The minimal representation:
  - 10 dimensions × 3584 floats = 35,840 values
  - vs 152,064 tokens × 3584 floats = 545M values

The axis IS the structure. Everything else is implicit.
""")
    
    axes = SemanticAxes(model, tokenizer)
    
    # Define dimensions with training pairs
    dimensions = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery"), ("chilly", "scorching")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant"), ("petite", "massive")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift"), ("plodding", "brisk")],
        "height": [("short", "tall"), ("low", "high"), ("squat", "towering"), ("stumpy", "lofty")],
        "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant"), ("murky", "luminous")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale"), ("youthful", "elderly")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "virtuous")],
        "weight": [("light", "heavy"), ("weightless", "weighty"), ("airy", "dense")],
        "hardness": [("soft", "hard"), ("tender", "tough"), ("gentle", "harsh"), ("delicate", "rigid")],
        "moisture": [("dry", "wet"), ("arid", "damp"), ("parched", "moist"), ("dusty", "soggy")],
    }
    
    print("\n--- LEARNING AXES ---")
    for dim_name, pairs in dimensions.items():
        axes.learn_axis(dim_name, pairs)
    
    # Test
    print("\n--- TESTING ---")
    test_cases = [
        ("hot", "cold", "temperature"),
        ("big", "small", "size"),
        ("fast", "slow", "speed"),
        ("tall", "short", "height"),
        ("bright", "dark", "brightness"),
        ("old", "young", "age"),
        ("good", "bad", "valence"),
        ("heavy", "light", "weight"),
        ("hard", "soft", "hardness"),
        ("wet", "dry", "moisture"),
    ]
    
    correct = 0
    for source, expected, dim_name in test_cases:
        result = axes.find_opposite(source, dim_name)
        if result:
            got, proj = result
            match = expected.lower() in got.lower()
            if match:
                correct += 1
            marker = "✓" if match else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    # Generalization
    print("\n--- GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("swift", "leisurely", "speed"),
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
        ("ancient", "new", "age"),
        ("soggy", "dusty", "moisture"),
    ]
    
    gen_correct = 0
    for source, expected, dim_name in gen_tests:
        result = axes.find_opposite(source, dim_name)
        if result:
            got, proj = result
            match = expected.lower() in got.lower()
            if match:
                gen_correct += 1
            marker = "✓" if match else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nGeneralization: {gen_correct}/{len(gen_tests)} ({gen_correct/len(gen_tests)*100:.0f}%)")
    
    # Auto-detection
    print("\n--- AUTO-DETECTION ---")
    auto_tests = ["hot", "big", "fast", "tall", "bright", "old", "good", "heavy", "hard", "wet"]
    
    for word in auto_tests:
        result = axes.find_opposite_auto(word)
        if result:
            got, proj, dim = result
            print(f"  {word:10s} → {got:12s} (dim: {dim})")
        else:
            print(f"  {word:10s} → [no result]")
    
    # Storage comparison
    print("\n--- STORAGE ---")
    n_dims = len(dimensions)
    axis_storage = n_dims * axes.hidden_dim * 4  # 4 bytes per float
    full_storage = axes.vocab_size * axes.hidden_dim * 2  # 2 bytes per bfloat16
    print(f"Axes only: {axis_storage / 1e6:.2f} MB ({n_dims} dims × {axes.hidden_dim} floats)")
    print(f"Full embeddings: {full_storage / 1e9:.2f} GB")
    print(f"Compression: {full_storage / axis_storage:.0f}x")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    demo_semantic_axes(model, tokenizer)


if __name__ == "__main__":
    main()
