#!/usr/bin/env python3
"""
Sign-Based Projection: Combining Constrained Projection with φ-Lattice Signs
=============================================================================

Earlier discovery: Signs encode semantics in φ-lattice.
  - ~44% of dimensions flip between opposites
  - But WHICH dimensions varies per pair

New approach:
  1. Learn the constrained projection (radial component)
  2. Also learn which SIGN DIMENSIONS flip for each semantic dimension
  3. Apply BOTH: project + flip signs

The twist/polarity might be encoded in the SIGN PATTERN, not a perpendicular direction.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    tensor = tensor.float()
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi(levels, signs):
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


class SignProjector:
    """
    Combine constrained projection with sign flipping.
    
    For each semantic dimension, we learn:
    1. The axis direction (for projection)
    2. The sign flip pattern (which dimensions flip)
    
    Navigation uses BOTH components.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        
        # Per-dimension data
        self.axes: Dict[str, torch.Tensor] = {}
        self.sign_flip_masks: Dict[str, torch.Tensor] = {}  # Which dims to flip
        self.word_to_dim: Dict[str, Tuple[str, int]] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def learn_dimension(self, name: str, pairs: List[Tuple[str, str]]):
        """
        Learn both axis and sign flip pattern for a dimension.
        """
        directions = []
        sign_flips = []
        
        for neg_word, pos_word in pairs:
            e_neg = self.get_embedding(neg_word)
            e_pos = self.get_embedding(pos_word)
            
            if e_neg is None or e_pos is None:
                continue
            
            # Direction for axis
            direction = (e_pos - e_neg).float().cpu()
            direction = direction / direction.norm()
            directions.append(direction)
            
            # Sign flip pattern
            _, s_neg = encode_phi(e_neg.cpu())
            _, s_pos = encode_phi(e_pos.cpu())
            flip = (s_neg != s_pos).float()
            sign_flips.append(flip)
            
            self.word_to_dim[neg_word] = (name, -1)
            self.word_to_dim[pos_word] = (name, +1)
        
        if not directions:
            return
        
        # Average axis
        axis = torch.stack(directions).mean(dim=0)
        axis = axis / axis.norm()
        self.axes[name] = axis
        
        # Sign flip probability per dimension
        flip_prob = torch.stack(sign_flips).mean(dim=0)
        # Keep dimensions that flip in >60% of pairs (higher threshold = more consistent)
        self.sign_flip_masks[name] = (flip_prob > 0.6)
        
        n_flip = self.sign_flip_masks[name].sum().item()
        print(f"  {name}: {n_flip} sign flip dims ({n_flip/self.hidden_dim*100:.1f}%)")
    
    def find_opposite(self, word: str, dim_name: str, 
                      use_signs: bool = True) -> Optional[Tuple[str, float]]:
        """
        Find opposite using projection + optional sign flipping.
        """
        if dim_name not in self.axes:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        axis = self.axes[dim_name].to(self.device)
        
        # Radial flip
        radial = torch.dot(embed.float(), axis)
        target = embed.float() - 2 * radial * axis
        
        if use_signs and dim_name in self.sign_flip_masks:
            # Also flip signs
            flip_mask = self.sign_flip_masks[dim_name].to(self.device)
            
            # Encode to φ-space
            levels, signs = encode_phi(target.cpu())
            
            # Flip specified dimensions
            signs_new = signs.clone()
            signs_new[flip_mask.cpu()] *= -1
            
            # Decode back
            target = decode_phi(levels, signs_new).to(self.device)
        
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


def demo_sign_projection(model, tokenizer):
    """Demo sign-based projection."""
    print("="*70)
    print("SIGN-BASED PROJECTION")
    print("="*70)
    print("""
Combining:
  1. Constrained projection (radial component)
  2. Sign flipping (φ-lattice semantics)

The twist might be in the SIGN PATTERN.
""")
    
    proj = SignProjector(model, tokenizer)
    
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
    
    print("\n--- LEARNING DIMENSIONS ---")
    for name, pairs in dimension_pairs.items():
        proj.learn_dimension(name, pairs)
    
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
    
    print("\n--- COMPARING PROJECTION-ONLY vs PROJECTION+SIGNS ---")
    print(f"{'Word':<10} {'Dim':<12} {'Proj Only':<12} {'Proj+Signs':<12} {'Expected':<10}")
    print("-"*70)
    
    proj_correct = 0
    sign_correct = 0
    
    for source, expected, dim_name in test_cases:
        proj_result = proj.find_opposite(source, dim_name, use_signs=False)
        sign_result = proj.find_opposite(source, dim_name, use_signs=True)
        
        proj_word = proj_result[0] if proj_result else "?"
        sign_word = sign_result[0] if sign_result else "?"
        
        proj_match = expected.lower() in proj_word.lower()
        sign_match = expected.lower() in sign_word.lower()
        
        if proj_match:
            proj_correct += 1
        if sign_match:
            sign_correct += 1
        
        p_mark = "✓" if proj_match else "✗"
        s_mark = "✓" if sign_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {proj_word:<10} {p_mark}  {sign_word:<10} {s_mark}  {expected}")
    
    print(f"\nProjection-only: {proj_correct}/{len(test_cases)} ({proj_correct/len(test_cases)*100:.0f}%)")
    print(f"Projection+Signs: {sign_correct}/{len(test_cases)} ({sign_correct/len(test_cases)*100:.0f}%)")
    
    # Generalization
    print("\n--- GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("quick", "slow", "speed"),
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
        ("ancient", "new", "age"),
        ("damp", "dry", "moisture"),
    ]
    
    proj_gen = 0
    sign_gen = 0
    
    for source, expected, dim_name in gen_tests:
        proj_result = proj.find_opposite(source, dim_name, use_signs=False)
        sign_result = proj.find_opposite(source, dim_name, use_signs=True)
        
        proj_word = proj_result[0] if proj_result else "?"
        sign_word = sign_result[0] if sign_result else "?"
        
        proj_match = expected.lower() in proj_word.lower()
        sign_match = expected.lower() in sign_word.lower()
        
        if proj_match:
            proj_gen += 1
        if sign_match:
            sign_gen += 1
        
        p_mark = "✓" if proj_match else "✗"
        s_mark = "✓" if sign_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {proj_word:<10} {p_mark}  {sign_word:<10} {s_mark}  {expected}")
    
    print(f"\nProjection-only gen: {proj_gen}/{len(gen_tests)} ({proj_gen/len(gen_tests)*100:.0f}%)")
    print(f"Projection+Signs gen: {sign_gen}/{len(gen_tests)} ({sign_gen/len(gen_tests)*100:.0f}%)")


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
    
    demo_sign_projection(model, tokenizer)


if __name__ == "__main__":
    main()
