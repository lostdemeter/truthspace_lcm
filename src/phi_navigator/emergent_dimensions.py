#!/usr/bin/env python3
"""
Emergent Dimension Navigation
==============================

From Design 114: Dimensions EMERGE from transformation pairs.

Key insight:
  - "hot" and "cold" define the TEMPERATURE dimension
  - "tall" and "short" define the HEIGHT dimension
  - These are DIFFERENT dimensions, not one "opposite" dimension

The approach:
  1. Given a word, find which semantic dimension it belongs to
  2. Find the opposite end of THAT dimension
  3. The opposite is at the other end of the axis

How to find the dimension:
  - Look at the word's nearest neighbors
  - Find words that are semantically related but opposed
  - The axis connecting them IS the dimension

The Platonic Ideal insight:
  - Some concepts sit at the origin (neutral)
  - Variations are φ distance along specific axes
  - Opposites are at opposite ends of the same axis
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

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class SemanticDimension:
    """A semantic dimension defined by word pairs."""
    name: str
    pairs: List[Tuple[str, str]]
    axis_vector: torch.Tensor  # The direction of this dimension
    

class EmergentDimensionNavigator:
    """
    Navigate using emergent dimensions.
    
    Each word belongs to one or more semantic dimensions.
    The opposite is found by moving to the other end of the dimension.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.coordinates = PhiCoordinates()
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.vocab_size = self.all_embeds.shape[0]
        self.hidden_dim = self.all_embeds.shape[1]
        
        # Discovered dimensions
        self.dimensions: Dict[str, SemanticDimension] = {}
        
        # Word to dimension mapping
        self.word_to_dim: Dict[str, str] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   self.all_embeds.float())
        if exclude:
            for word in exclude:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    sims[ids[0]] = -1
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item())
                for idx in top_indices]
    
    # =========================================================================
    # DIMENSION DISCOVERY
    # =========================================================================
    
    def add_dimension(self, name: str, pairs: List[Tuple[str, str]]) -> SemanticDimension:
        """
        Add a semantic dimension defined by word pairs.
        
        The dimension is the axis connecting the pairs.
        """
        # Compute the axis vector as the average direction
        directions = []
        
        for w1, w2 in pairs:
            e1 = self.get_embedding(w1)
            e2 = self.get_embedding(w2)
            
            if e1 is None or e2 is None:
                continue
            
            direction = (e2 - e1).float().cpu()
            direction = direction / direction.norm()  # Normalize
            directions.append(direction)
            
            # Map words to this dimension
            self.word_to_dim[w1] = name
            self.word_to_dim[w2] = name
        
        if not directions:
            raise ValueError(f"No valid pairs for dimension {name}")
        
        # Average direction
        axis_vector = torch.stack(directions).mean(dim=0)
        axis_vector = axis_vector / axis_vector.norm()
        
        dim = SemanticDimension(
            name=name,
            pairs=pairs,
            axis_vector=axis_vector,
        )
        
        self.dimensions[name] = dim
        return dim
    
    # =========================================================================
    # DIMENSION-AWARE NAVIGATION
    # =========================================================================
    
    def find_opposite_on_dimension(self, word: str, dim_name: str,
                                    scale: float = 1.0) -> Optional[Tuple[str, float]]:
        """
        Find the opposite of a word along a specific dimension.
        
        The opposite is found by:
        1. Project word onto the dimension axis
        2. Move to the opposite end
        3. Find nearest word
        """
        if dim_name not in self.dimensions:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        dim = self.dimensions[dim_name]
        axis = dim.axis_vector.to(self.device)
        
        # Project onto axis
        projection = torch.dot(embed.float(), axis.float())
        
        # Move to opposite end (flip the projection)
        # The opposite is at -2 * projection along the axis
        opposite_embed = embed.float() - 2 * scale * projection * axis
        
        nearest = self.find_nearest(opposite_embed, top_k=5, exclude=[word])
        
        if nearest:
            return nearest[0]
        return None
    
    def find_opposite_auto(self, word: str) -> Optional[Tuple[str, float, str]]:
        """
        Automatically find the opposite by detecting which dimension the word is on.
        
        Returns (opposite, confidence, dimension_name)
        """
        # Check if word is in a known dimension
        if word in self.word_to_dim:
            dim_name = self.word_to_dim[word]
            result = self.find_opposite_on_dimension(word, dim_name)
            if result:
                return (result[0], result[1], dim_name)
        
        # Try all dimensions, return best result
        best_result = None
        best_confidence = -1
        best_dim = None
        
        for dim_name in self.dimensions:
            result = self.find_opposite_on_dimension(word, dim_name)
            if result and result[1] > best_confidence:
                best_result = result
                best_confidence = result[1]
                best_dim = dim_name
        
        if best_result:
            return (best_result[0], best_result[1], best_dim)
        
        return None


def demo_emergent_dimensions(model, tokenizer):
    """Demo emergent dimension navigation."""
    print("="*70)
    print("EMERGENT DIMENSION NAVIGATION")
    print("="*70)
    print("""
From Design 114: Dimensions EMERGE from transformation pairs.

Each word pair defines its own semantic dimension:
  - hot/cold → TEMPERATURE dimension
  - tall/short → HEIGHT dimension
  - young/old → AGE dimension

The opposite is found by moving along THAT dimension.
""")
    
    nav = EmergentDimensionNavigator(model, tokenizer)
    
    # Define semantic dimensions with their pairs
    dimension_definitions = {
        "temperature": [("hot", "cold"), ("warm", "cool"), ("burning", "freezing")],
        "size": [("big", "small"), ("large", "tiny"), ("huge", "little")],
        "speed": [("fast", "slow"), ("quick", "sluggish"), ("rapid", "gradual")],
        "height": [("tall", "short"), ("high", "low")],
        "brightness": [("bright", "dark"), ("light", "dim")],
        "age": [("young", "old"), ("new", "ancient")],
        "wealth": [("rich", "poor"), ("wealthy", "impoverished")],
        "volume": [("loud", "quiet"), ("noisy", "silent")],
        "thickness": [("thick", "thin"), ("fat", "skinny")],
        "depth": [("deep", "shallow")],
        "hardness": [("hard", "soft")],
        "moisture": [("wet", "dry")],
        "valence": [("good", "bad"), ("happy", "sad")],
    }
    
    print("--- ADDING DIMENSIONS ---")
    for name, pairs in dimension_definitions.items():
        try:
            dim = nav.add_dimension(name, pairs)
            print(f"  {name}: {len(pairs)} pairs")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    # Test on words from known dimensions
    print("\n--- TESTING ON KNOWN DIMENSION WORDS ---")
    test_cases = [
        ("hot", "cold", "temperature"),
        ("big", "small", "size"),
        ("fast", "slow", "speed"),
        ("tall", "short", "height"),
        ("bright", "dark", "brightness"),
        ("young", "old", "age"),
        ("rich", "poor", "wealth"),
        ("loud", "quiet", "volume"),
        ("thick", "thin", "thickness"),
        ("deep", "shallow", "depth"),
        ("hard", "soft", "hardness"),
        ("wet", "dry", "moisture"),
        ("good", "bad", "valence"),
        ("happy", "sad", "valence"),
    ]
    
    correct = 0
    for source, expected, dim_name in test_cases:
        result = nav.find_opposite_on_dimension(source, dim_name)
        if result:
            got = result[0]
            found = expected.lower() in got.lower()
            if found:
                correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:8s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:8s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    # Test auto-detection
    print("\n--- TESTING AUTO-DETECTION ---")
    auto_correct = 0
    for source, expected, _ in test_cases:
        result = nav.find_opposite_auto(source)
        if result:
            got, conf, dim = result
            found = expected.lower() in got.lower()
            if found:
                auto_correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:8s} → {got:12s} (dim: {dim:12s}) {marker}")
        else:
            print(f"  {source:8s} → [no result]")
    
    print(f"\nAuto accuracy: {auto_correct}/{len(test_cases)} ({auto_correct/len(test_cases)*100:.0f}%)")
    
    # Test on words NOT in training
    print("\n--- TESTING GENERALIZATION ---")
    generalization_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("rapid", "gradual", "speed"),
        ("high", "low", "height"),
        ("light", "dim", "brightness"),
        ("new", "ancient", "age"),
    ]
    
    gen_correct = 0
    for source, expected, dim_name in generalization_tests:
        result = nav.find_opposite_on_dimension(source, dim_name)
        if result:
            got = result[0]
            found = expected.lower() in got.lower()
            if found:
                gen_correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:8s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:8s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nGeneralization: {gen_correct}/{len(generalization_tests)} ({gen_correct/len(generalization_tests)*100:.0f}%)")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    demo_emergent_dimensions(model, tokenizer)


if __name__ == "__main__":
    main()
