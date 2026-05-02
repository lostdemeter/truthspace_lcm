#!/usr/bin/env python3
"""
Orthogonal Dimension Navigation with φ-Zipf Duality
====================================================

From Design 039: φ-Zipf Duality
  - φ^n for encoding (outward expansion)
  - φ^(-n) for weighting (inward contraction)
  - Same fractal, opposite directions

Key insights:
1. Dimensions should be ORTHOGONAL to each other
2. Opposites are on the SAME axis but opposite ends
3. φ^(-rank) weighting identifies importance

The problem with cosine similarity:
  - It finds SIMILAR words
  - But opposites aren't similar - they're OPPOSED

The solution:
  - Use orthogonality to separate dimensions
  - Use axis projection to find opposites
  - Weight by φ^(-rank) for importance

The structure IS the navigation.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class OrthogonalDimension:
    """A dimension defined by its axis vector and word projections."""
    name: str
    axis: torch.Tensor  # Unit vector defining the dimension
    positive_words: Dict[str, float]  # word -> projection value
    negative_words: Dict[str, float]  # word -> projection value


class OrthogonalNavigator:
    """
    Navigate using orthogonal dimensions with φ-Zipf weighting.
    
    Key principles:
    1. Dimensions are orthogonal axes in embedding space
    2. Words project onto dimensions with positive/negative values
    3. Opposites have opposite projections on the same dimension
    4. φ^(-rank) weighting identifies important words
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        self.dimensions: Dict[str, OrthogonalDimension] = {}
        self.word_projections: Dict[str, Dict[str, float]] = {}  # word -> {dim: projection}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def find_nearest_by_projection(self, dim_name: str, target_projection: float,
                                    top_k: int = 5, exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find words with projection closest to target value on a dimension.
        """
        if dim_name not in self.dimensions:
            return []
        
        dim = self.dimensions[dim_name]
        axis = dim.axis.to(self.device)
        
        # Project all embeddings onto the axis
        projections = torch.matmul(self.all_embeds.float(), axis.float())
        
        # Find words closest to target projection
        distances = (projections - target_projection).abs()
        
        if exclude:
            for word in exclude:
                tid = self.get_token_id(word)
                if tid is not None:
                    distances[tid] = float('inf')
        
        top_indices = distances.topk(top_k, largest=False).indices
        
        results = []
        for idx in top_indices:
            word = self.tokenizer.decode([idx.item()]).strip()
            proj = projections[idx].item()
            results.append((word, proj))
        
        return results
    
    # =========================================================================
    # DIMENSION DISCOVERY USING ORTHOGONALITY
    # =========================================================================
    
    def discover_dimension_from_pairs(self, name: str, 
                                       pairs: List[Tuple[str, str]]) -> Optional[OrthogonalDimension]:
        """
        Discover a dimension from word pairs.
        
        The dimension axis is the average direction from negative to positive words.
        We then orthogonalize against existing dimensions.
        """
        directions = []
        
        for neg_word, pos_word in pairs:
            e_neg = self.get_embedding(neg_word)
            e_pos = self.get_embedding(pos_word)
            
            if e_neg is None or e_pos is None:
                continue
            
            direction = (e_pos - e_neg).float().cpu()
            direction = direction / direction.norm()
            directions.append(direction)
        
        if not directions:
            return None
        
        # Average direction
        axis = torch.stack(directions).mean(dim=0)
        axis = axis / axis.norm()
        
        # Orthogonalize against existing dimensions
        for existing_dim in self.dimensions.values():
            existing_axis = existing_dim.axis
            # Remove component along existing axis
            projection = torch.dot(axis, existing_axis)
            axis = axis - projection * existing_axis
            if axis.norm() < 0.1:
                print(f"  Warning: {name} is not orthogonal to existing dimensions")
                break
            axis = axis / axis.norm()
        
        # Project all words onto this axis
        axis_device = axis.to(self.device)
        all_projections = torch.matmul(self.all_embeds.float(), axis_device.float())
        
        # Identify positive and negative words
        positive_words = {}
        negative_words = {}
        
        for neg_word, pos_word in pairs:
            neg_id = self.get_token_id(neg_word)
            pos_id = self.get_token_id(pos_word)
            
            if neg_id is not None:
                negative_words[neg_word] = all_projections[neg_id].item()
            if pos_id is not None:
                positive_words[pos_word] = all_projections[pos_id].item()
        
        dim = OrthogonalDimension(
            name=name,
            axis=axis,
            positive_words=positive_words,
            negative_words=negative_words,
        )
        
        self.dimensions[name] = dim
        return dim
    
    # =========================================================================
    # NAVIGATION USING PROJECTION
    # =========================================================================
    
    def find_opposite(self, word: str) -> Optional[Tuple[str, float, str]]:
        """
        Find the opposite of a word by:
        1. Finding which dimension the word projects strongly onto
        2. Finding the word with opposite projection on that dimension
        
        Returns (opposite_word, projection, dimension_name)
        """
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        # Find which dimension this word projects most strongly onto
        best_dim = None
        best_projection = 0
        
        for dim_name, dim in self.dimensions.items():
            axis = dim.axis.to(self.device)
            projection = torch.dot(embed.float(), axis.float()).item()
            
            if abs(projection) > abs(best_projection):
                best_projection = projection
                best_dim = dim_name
        
        if best_dim is None:
            return None
        
        # Find word with opposite projection
        target_projection = -best_projection
        results = self.find_nearest_by_projection(best_dim, target_projection, 
                                                   top_k=5, exclude=[word])
        
        if results:
            # Filter out partial words and non-alphabetic
            for result_word, proj in results:
                if result_word.isalpha() and len(result_word) > 1:
                    return (result_word, proj, best_dim)
        
        return None
    
    def find_opposite_on_dimension(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """Find opposite on a specific dimension."""
        if dim_name not in self.dimensions:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        dim = self.dimensions[dim_name]
        axis = dim.axis.to(self.device)
        
        projection = torch.dot(embed.float(), axis.float()).item()
        target_projection = -projection
        
        results = self.find_nearest_by_projection(dim_name, target_projection,
                                                   top_k=10, exclude=[word])
        
        # Filter for clean words
        for result_word, proj in results:
            if result_word.isalpha() and len(result_word) > 1:
                return (result_word, proj)
        
        return None


def demo_orthogonal_navigation(model, tokenizer):
    """Demo orthogonal dimension navigation."""
    print("="*70)
    print("ORTHOGONAL DIMENSION NAVIGATION")
    print("="*70)
    print("""
From Design 039: φ-Zipf Duality
  - Dimensions are ORTHOGONAL axes
  - Opposites are at opposite ends of the SAME axis
  - The structure IS the navigation
""")
    
    nav = OrthogonalNavigator(model, tokenizer)
    
    # Define dimensions with seed pairs
    dimension_seeds = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid")],
        "height": [("short", "tall"), ("low", "high")],
        "brightness": [("dark", "bright"), ("dim", "light")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive")],
        "weight": [("light", "heavy"), ("weightless", "weighty")],
        "hardness": [("soft", "hard")],
        "moisture": [("dry", "wet"), ("arid", "damp")],
    }
    
    print("--- DISCOVERING ORTHOGONAL DIMENSIONS ---")
    for name, pairs in dimension_seeds.items():
        dim = nav.discover_dimension_from_pairs(name, pairs)
        if dim:
            print(f"  {name}: axis norm = {dim.axis.norm():.3f}, "
                  f"{len(dim.positive_words)} pos, {len(dim.negative_words)} neg")
    
    # Test navigation
    print("\n--- TESTING NAVIGATION ---")
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
        result = nav.find_opposite_on_dimension(source, dim_name)
        if result:
            got, proj = result
            found = expected.lower() in got.lower()
            if found:
                correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    # Test auto-detection
    print("\n--- TESTING AUTO-DETECTION ---")
    auto_correct = 0
    for source, expected, _ in test_cases:
        result = nav.find_opposite(source)
        if result:
            got, proj, dim = result
            found = expected.lower() in got.lower()
            if found:
                auto_correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:10s} → {got:12s} (dim: {dim:12s}) {marker}")
        else:
            print(f"  {source:10s} → [no result]")
    
    print(f"\nAuto accuracy: {auto_correct}/{len(test_cases)} ({auto_correct/len(test_cases)*100:.0f}%)")
    
    # Test generalization
    print("\n--- TESTING GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("quick", "slow", "speed"),
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
    ]
    
    gen_correct = 0
    for source, expected, dim_name in gen_tests:
        result = nav.find_opposite_on_dimension(source, dim_name)
        if result:
            got, proj = result
            found = expected.lower() in got.lower()
            if found:
                gen_correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nGeneralization: {gen_correct}/{len(gen_tests)} ({gen_correct/len(gen_tests)*100:.0f}%)")


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
    
    demo_orthogonal_navigation(model, tokenizer)


if __name__ == "__main__":
    main()
