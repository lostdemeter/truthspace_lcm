#!/usr/bin/env python3
"""
Kerr Projection: Adding Twist and Polarity
===========================================

From the Kerr Truth Space discovery:
  - Helicity flips at the horizon
  - Polarization: P(k) = (-1)^k × φ^(-k)
  - Frame dragging causes values to spiral

We're missing the TWIST in our projection.

The insight:
  - Linear projection captures the "radial" component
  - But embeddings also have a "rotational" component
  - Opposites might differ in BOTH position AND polarity

Approach:
  1. Project to orthogonal space (radial component)
  2. Add a polarity/helicity term (rotational component)
  3. Opposites have flipped polarity, not just flipped position

The polarity could be encoded in:
  - The SIGN pattern of the embedding
  - The phase of the embedding relative to the axis
  - A separate "spin" dimension
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2


class KerrProjector:
    """
    Projection with Kerr-like twist/polarity.
    
    Key insight: Opposites differ in POLARITY, not just position.
    
    We decompose the embedding into:
    1. Radial component (projection onto dimension axis)
    2. Rotational component (phase/polarity relative to axis)
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.coordinates = PhiCoordinates()
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        
        # Dimension axes
        self.axes: Dict[str, torch.Tensor] = {}
        
        # Polarity vectors (perpendicular to axes)
        self.polarities: Dict[str, torch.Tensor] = {}
        
        # Word mappings
        self.word_to_dim: Dict[str, Tuple[str, int]] = {}  # word -> (dim, polarity)
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    # =========================================================================
    # KERR DECOMPOSITION
    # =========================================================================
    
    def learn_dimension_with_polarity(self, name: str, 
                                       pairs: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn both the axis AND the polarity vector for a dimension.
        
        The axis is the direction from negative to positive.
        The polarity is the perpendicular component that differs between opposites.
        """
        # Collect embeddings
        neg_embeds = []
        pos_embeds = []
        
        for neg_word, pos_word in pairs:
            e_neg = self.get_embedding(neg_word)
            e_pos = self.get_embedding(pos_word)
            
            if e_neg is None or e_pos is None:
                continue
            
            neg_embeds.append(e_neg.float().cpu())
            pos_embeds.append(e_pos.float().cpu())
            
            self.word_to_dim[neg_word] = (name, -1)
            self.word_to_dim[pos_word] = (name, +1)
        
        if not neg_embeds:
            raise ValueError(f"No valid pairs for {name}")
        
        neg_stack = torch.stack(neg_embeds)
        pos_stack = torch.stack(pos_embeds)
        
        # Axis: average direction from negative to positive
        directions = pos_stack - neg_stack
        axis = directions.mean(dim=0)
        axis = axis / axis.norm()
        
        # Polarity: the component that's DIFFERENT between neg and pos
        # but perpendicular to the axis
        # This is like the "spin" component
        
        # Compute the perpendicular residuals
        neg_proj = torch.einsum('nd,d->n', neg_stack, axis).unsqueeze(1) * axis
        pos_proj = torch.einsum('nd,d->n', pos_stack, axis).unsqueeze(1) * axis
        
        neg_perp = neg_stack - neg_proj
        pos_perp = pos_stack - pos_proj
        
        # The polarity is the direction that maximally separates neg_perp from pos_perp
        perp_diff = pos_perp - neg_perp
        polarity = perp_diff.mean(dim=0)
        
        if polarity.norm() > 1e-6:
            polarity = polarity / polarity.norm()
        else:
            # If no perpendicular difference, use a random perpendicular direction
            polarity = torch.randn(self.hidden_dim)
            polarity = polarity - torch.dot(polarity, axis) * axis
            polarity = polarity / polarity.norm()
        
        self.axes[name] = axis
        self.polarities[name] = polarity
        
        return axis, polarity
    
    def decompose(self, word: str, dim_name: str) -> Optional[Tuple[float, float]]:
        """
        Decompose a word into (radial, rotational) components on a dimension.
        
        Returns (projection_on_axis, projection_on_polarity)
        """
        if dim_name not in self.axes:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        axis = self.axes[dim_name]
        polarity = self.polarities[dim_name]
        
        radial = torch.dot(embed.float().cpu(), axis).item()
        rotational = torch.dot(embed.float().cpu(), polarity).item()
        
        return (radial, rotational)
    
    # =========================================================================
    # KERR NAVIGATION
    # =========================================================================
    
    def find_opposite_kerr(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite using Kerr decomposition.
        
        Flip BOTH the radial AND rotational components.
        """
        if dim_name not in self.axes:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        axis = self.axes[dim_name].to(self.device)
        polarity = self.polarities[dim_name].to(self.device)
        
        # Decompose
        radial = torch.dot(embed.float(), axis)
        rotational = torch.dot(embed.float(), polarity)
        
        # Flip both components
        target = embed.float() - 2 * radial * axis - 2 * rotational * polarity
        
        # Find nearest
        sims = F.cosine_similarity(target.unsqueeze(0), self.all_embeds.float())
        
        word_id = self.get_token_id(word)
        if word_id is not None:
            sims[word_id] = -1
        
        top_indices = sims.topk(10).indices
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) > 1:
                return (result_word, sims[idx].item())
        
        return None
    
    def find_opposite_radial_only(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """Find opposite using only radial component (for comparison)."""
        if dim_name not in self.axes:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        axis = self.axes[dim_name].to(self.device)
        
        radial = torch.dot(embed.float(), axis)
        target = embed.float() - 2 * radial * axis
        
        sims = F.cosine_similarity(target.unsqueeze(0), self.all_embeds.float())
        
        word_id = self.get_token_id(word)
        if word_id is not None:
            sims[word_id] = -1
        
        top_indices = sims.topk(10).indices
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) > 1:
                return (result_word, sims[idx].item())
        
        return None


def demo_kerr_projection(model, tokenizer):
    """Demo Kerr projection with twist/polarity."""
    print("="*70)
    print("KERR PROJECTION: ADDING TWIST AND POLARITY")
    print("="*70)
    print("""
From Kerr Truth Space:
  - Helicity flips at horizon
  - Polarization: P(k) = (-1)^k × φ^(-k)
  - Values spiral through levels (frame dragging)

The insight:
  - Linear projection captures RADIAL component
  - But embeddings also have ROTATIONAL component
  - Opposites differ in BOTH position AND polarity
""")
    
    proj = KerrProjector(model, tokenizer)
    
    # Define dimensions
    dimension_pairs = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery"), ("chilly", "scorching")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant"), ("petite", "massive")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift")],
        "height": [("short", "tall"), ("low", "high"), ("squat", "towering")],
        "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale"), ("youthful", "elderly")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "virtuous")],
        "weight": [("light", "heavy"), ("weightless", "weighty"), ("feathery", "leaden")],
        "hardness": [("soft", "hard"), ("tender", "tough"), ("gentle", "harsh")],
        "moisture": [("dry", "wet"), ("arid", "damp"), ("parched", "moist")],
    }
    
    print("\n--- LEARNING DIMENSIONS WITH POLARITY ---")
    for name, pairs in dimension_pairs.items():
        axis, polarity = proj.learn_dimension_with_polarity(name, pairs)
        # Check orthogonality
        dot = torch.dot(axis, polarity).abs().item()
        print(f"  {name}: axis-polarity dot = {dot:.4f}")
    
    # Test cases
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
    
    print("\n--- COMPARING RADIAL-ONLY vs KERR (RADIAL+ROTATIONAL) ---")
    print(f"{'Word':<10} {'Dim':<12} {'Radial Only':<15} {'Kerr':<15} {'Expected':<10}")
    print("-"*70)
    
    radial_correct = 0
    kerr_correct = 0
    
    for source, expected, dim_name in test_cases:
        radial_result = proj.find_opposite_radial_only(source, dim_name)
        kerr_result = proj.find_opposite_kerr(source, dim_name)
        
        radial_word = radial_result[0] if radial_result else "?"
        kerr_word = kerr_result[0] if kerr_result else "?"
        
        radial_match = expected.lower() in radial_word.lower()
        kerr_match = expected.lower() in kerr_word.lower()
        
        if radial_match:
            radial_correct += 1
        if kerr_match:
            kerr_correct += 1
        
        r_mark = "✓" if radial_match else "✗"
        k_mark = "✓" if kerr_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {radial_word:<12} {r_mark}  {kerr_word:<12} {k_mark}  {expected}")
    
    print(f"\nRadial-only accuracy: {radial_correct}/{len(test_cases)} ({radial_correct/len(test_cases)*100:.0f}%)")
    print(f"Kerr accuracy:        {kerr_correct}/{len(test_cases)} ({kerr_correct/len(test_cases)*100:.0f}%)")
    
    # Test generalization
    print("\n--- GENERALIZATION TEST ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("quick", "slow", "speed"),
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
        ("ancient", "new", "age"),
        ("damp", "dry", "moisture"),
    ]
    
    radial_gen = 0
    kerr_gen = 0
    
    for source, expected, dim_name in gen_tests:
        radial_result = proj.find_opposite_radial_only(source, dim_name)
        kerr_result = proj.find_opposite_kerr(source, dim_name)
        
        radial_word = radial_result[0] if radial_result else "?"
        kerr_word = kerr_result[0] if kerr_result else "?"
        
        radial_match = expected.lower() in radial_word.lower()
        kerr_match = expected.lower() in kerr_word.lower()
        
        if radial_match:
            radial_gen += 1
        if kerr_match:
            kerr_gen += 1
        
        r_mark = "✓" if radial_match else "✗"
        k_mark = "✓" if kerr_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {radial_word:<12} {r_mark}  {kerr_word:<12} {k_mark}  {expected}")
    
    print(f"\nRadial-only generalization: {radial_gen}/{len(gen_tests)} ({radial_gen/len(gen_tests)*100:.0f}%)")
    print(f"Kerr generalization:        {kerr_gen}/{len(gen_tests)} ({kerr_gen/len(gen_tests)*100:.0f}%)")


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
    
    demo_kerr_projection(model, tokenizer)


if __name__ == "__main__":
    main()
