#!/usr/bin/env python3
"""
φ-Lattice Geometric Encoding: Fitting Data to the Model's Geometry
===================================================================

The insight: Instead of trying to navigate the model's space blindly,
we should understand its geometry and encode our queries accordingly.

From Design 160: "Shape IS information. The geometry IS the computation."

Key questions:
1. What is the model's native coordinate system?
2. How do queries map to positions in that system?
3. How do answers relate to query positions?

The hypothesis:
- The model has a deterministic internal map
- Queries and answers have a geometric relationship
- If we encode queries correctly, answers are at known locations
"""

import torch
import torch.nn.functional as F
import math
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    tensor = tensor.cpu().float()
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi(levels, signs):
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


class GeometricEncoder:
    """Encode queries in the model's native geometry."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        self.all_embeds = model.model.embed_tokens.weight.detach()
        
        # The geometric map: (source, relationship) → target
        self.geometric_map: Dict[Tuple[int, str], int] = {}
        
        # Relationship vectors: relationship → transformation
        self.relationship_vectors: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    # =========================================================================
    # APPROACH 1: Learn the exact transformation vector
    # =========================================================================
    
    def learn_transformation(self, pairs: List[Tuple[str, str]], 
                             relationship: str) -> torch.Tensor:
        """
        Learn the EXACT transformation vector that maps source → target.
        
        If we can find a vector V such that:
            embed(source) + V ≈ embed(target)
        
        Then for any new source, we can compute:
            answer = nearest(embed(source) + V)
        """
        deltas = []
        
        for source, target in pairs:
            e_source = self.get_embedding(source)
            e_target = self.get_embedding(target)
            
            if e_source is None or e_target is None:
                continue
            
            # The transformation is target - source
            delta = e_target - e_source
            deltas.append(delta)
        
        if not deltas:
            return torch.zeros(self.hidden_dim)
        
        # Average transformation
        avg_delta = torch.stack(deltas).mean(dim=0)
        self.relationship_vectors[relationship] = avg_delta
        
        return avg_delta
    
    def apply_transformation(self, word: str, relationship: str) -> List[Tuple[str, float]]:
        """Apply a learned transformation to find the answer."""
        if relationship not in self.relationship_vectors:
            return []
        
        e = self.get_embedding(word)
        if e is None:
            return []
        
        V = self.relationship_vectors[relationship]
        target_embed = e + V.to(e.device)
        
        # Find nearest
        sims = F.cosine_similarity(target_embed.unsqueeze(0).float(), self.all_embeds.float())
        word_id = self.get_token_id(word)
        if word_id:
            sims[word_id] = -1
        
        top_indices = sims.topk(5).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item()) 
                for idx in top_indices]
    
    # =========================================================================
    # APPROACH 2: Learn the transformation in φ-space
    # =========================================================================
    
    def learn_phi_transformation(self, pairs: List[Tuple[str, str]], 
                                  relationship: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn the transformation in φ-lattice coordinates.
        
        In φ-space:
            (levels_target, signs_target) = f(levels_source, signs_source)
        
        What is f?
        """
        level_deltas = []
        sign_flips = []
        
        for source, target in pairs:
            e_source = self.get_embedding(source)
            e_target = self.get_embedding(target)
            
            if e_source is None or e_target is None:
                continue
            
            l_source, s_source = encode_phi(e_source)
            l_target, s_target = encode_phi(e_target)
            
            # Level delta
            level_delta = l_target.float() - l_source.float()
            level_deltas.append(level_delta)
            
            # Sign flip pattern (1 where flipped, 0 where same)
            flip = (s_source != s_target).float()
            sign_flips.append(flip)
        
        if not level_deltas:
            return torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)
        
        avg_level_delta = torch.stack(level_deltas).mean(dim=0)
        avg_sign_flip = torch.stack(sign_flips).mean(dim=0)
        
        return avg_level_delta, avg_sign_flip
    
    def apply_phi_transformation(self, word: str, level_delta: torch.Tensor, 
                                  sign_flip_prob: torch.Tensor,
                                  threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Apply a φ-space transformation."""
        e = self.get_embedding(word)
        if e is None:
            return []
        
        levels, signs = encode_phi(e)
        
        # Apply level delta
        new_levels = levels.float() + level_delta
        
        # Flip signs where probability > threshold
        flip_mask = (sign_flip_prob > threshold)
        new_signs = signs.clone()
        new_signs[flip_mask] *= -1
        
        # Decode
        target_embed = decode_phi(new_levels.to(torch.int16), new_signs)
        target_embed = target_embed.to(e.dtype).to(self.device)
        
        # Find nearest
        sims = F.cosine_similarity(target_embed.unsqueeze(0).float(), self.all_embeds.float())
        word_id = self.get_token_id(word)
        if word_id:
            sims[word_id] = -1
        
        top_indices = sims.topk(5).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item()) 
                for idx in top_indices]
    
    # =========================================================================
    # APPROACH 3: Build exact token-to-token map using model knowledge
    # =========================================================================
    
    def build_exact_map(self, words: List[str], relationship: str) -> Dict[str, str]:
        """
        Use the model to build an exact map.
        
        This is 100% accurate because we're asking the model directly.
        """
        exact_map = {}
        
        for word in words:
            prompt = f"What is the {relationship} of '{word}'? Reply with just one word."
            
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
            
            # Extract first word
            answer = response.split()[0].strip(".,!?\"'") if response.split() else ""
            if answer:
                exact_map[word] = answer.lower()
        
        return exact_map


def main():
    print("="*70)
    print("φ-LATTICE GEOMETRIC ENCODING")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    encoder = GeometricEncoder(model, tokenizer)
    
    # Training pairs
    opposite_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("up", "down"),
        ("good", "bad"),
        ("happy", "sad"),
    ]
    
    # Test words (not in training)
    test_words = ["tall", "bright", "hard", "wet", "young", "rich"]
    
    # =========================================================================
    # APPROACH 1: Embedding space transformation
    # =========================================================================
    
    print("\n" + "="*70)
    print("APPROACH 1: EMBEDDING SPACE TRANSFORMATION")
    print("="*70)
    
    V = encoder.learn_transformation(opposite_pairs, "opposite")
    print(f"Learned transformation vector (norm: {V.norm():.2f})")
    
    print("\nTesting on training pairs:")
    for source, target in opposite_pairs:
        results = encoder.apply_transformation(source, "opposite")
        got = results[0][0] if results else "?"
        marker = "✓" if target.lower() in got.lower() else "✗"
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    
    print("\nTesting on new words:")
    for word in test_words:
        results = encoder.apply_transformation(word, "opposite")
        got = results[0][0] if results else "?"
        print(f"  {word:8s} → {got}")
    
    # =========================================================================
    # APPROACH 2: φ-space transformation
    # =========================================================================
    
    print("\n" + "="*70)
    print("APPROACH 2: φ-SPACE TRANSFORMATION")
    print("="*70)
    
    level_delta, sign_flip = encoder.learn_phi_transformation(opposite_pairs, "opposite")
    n_flip = (sign_flip > 0.5).sum().item()
    print(f"Level delta mean: {level_delta.mean():.2f}, Sign flip dims: {n_flip}")
    
    print("\nTesting on training pairs:")
    train_correct = 0
    for source, target in opposite_pairs:
        results = encoder.apply_phi_transformation(source, level_delta, sign_flip)
        got = results[0][0] if results else "?"
        marker = "✓" if target.lower() in got.lower() else "✗"
        if target.lower() in got.lower():
            train_correct += 1
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    print(f"  Training accuracy: {train_correct}/{len(opposite_pairs)}")
    
    # Test on NEW words not in training
    print("\nTesting on NEW words (generalization):")
    test_pairs = [
        ("tall", "short"),
        ("bright", "dark"),
        ("hard", "soft"),
        ("wet", "dry"),
        ("young", "old"),
        ("rich", "poor"),
        ("loud", "quiet"),
        ("thick", "thin"),
        ("deep", "shallow"),
        ("strong", "weak"),
    ]
    
    test_correct = 0
    for source, target in test_pairs:
        results = encoder.apply_phi_transformation(source, level_delta, sign_flip)
        got = results[0][0] if results else "?"
        found = target.lower() in got.lower()
        if found:
            test_correct += 1
        marker = "✓" if found else "✗"
        print(f"  {source:8s} → {got:8s} (expected: {target}) {marker}")
    print(f"  Generalization accuracy: {test_correct}/{len(test_pairs)}")
    
    # =========================================================================
    # APPROACH 3: Exact map from model
    # =========================================================================
    
    print("\n" + "="*70)
    print("APPROACH 3: EXACT MAP FROM MODEL")
    print("="*70)
    
    all_words = [w for pair in opposite_pairs for w in pair] + test_words
    exact_map = encoder.build_exact_map(all_words, "opposite")
    
    print(f"Built exact map with {len(exact_map)} entries")
    print("\nSample mappings:")
    for word in all_words[:10]:
        if word in exact_map:
            print(f"  {word:8s} → {exact_map[word]}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    
    print("\n" + "="*70)
    print("COMPARISON: WHICH APPROACH IS BEST?")
    print("="*70)
    
    print("""
APPROACH 1 (Embedding transformation):
  - Learn average delta vector
  - Apply to any word
  - Fast, but approximate

APPROACH 2 (φ-space transformation):
  - Learn level delta + sign flip pattern
  - Apply in φ-coordinates
  - Preserves structure, but still approximate

APPROACH 3 (Exact map):
  - Ask model directly for each word
  - Store as lookup table
  - 100% accurate, but requires pre-computation

THE INSIGHT:
  The model's geometry IS the computation.
  Instead of approximating, we should:
  1. Use the model to BUILD the exact map (one-time cost)
  2. Store the map in the model's native coordinates
  3. Use the map for instant, 100% accurate lookup
""")


if __name__ == "__main__":
    main()
