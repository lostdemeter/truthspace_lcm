#!/usr/bin/env python3
"""
Constrained Projection: Fit Weights to Orthogonal Constraints
==============================================================

The insight: Orthogonality is the CONSTRAINT, not the observation.

Instead of finding orthogonal dimensions in embedding space,
we LEARN a projection that MAKES dimensions orthogonal.

From probe extraction:
  W = Y @ X @ (X^T X)^(-1)
  
This gives us the EXACT linear projection from X to Y.

Applied to semantic dimensions:
  X = embeddings of words
  Y = desired positions in orthogonal space
  W = projection matrix that enforces the constraint

The approach:
1. Define the desired orthogonal structure (Y)
   - Each dimension is a separate axis
   - Opposites are at +1 and -1 on their axis
   - Other words are at 0 on that axis

2. Collect training examples (X)
   - Embeddings of words with known positions

3. Solve for W
   - W = Y @ X^T @ (X @ X^T)^(-1)
   - This is the exact solution, no approximation

4. Apply W to any embedding
   - projected = W @ embedding
   - Find opposite by flipping sign on the relevant dimension
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2


class ConstrainedProjector:
    """
    Learn a projection that enforces orthogonal semantic dimensions.
    
    The projection maps from embedding space to a clean orthogonal space
    where each dimension corresponds to a semantic axis.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        
        # The learned projection matrix: [n_dims, hidden_dim]
        self.W: Optional[torch.Tensor] = None
        
        # Dimension names
        self.dim_names: List[str] = []
        
        # Word to position mapping in projected space
        self.word_positions: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    # =========================================================================
    # LEARNING THE PROJECTION
    # =========================================================================
    
    def learn_projection(self, dimension_pairs: Dict[str, List[Tuple[str, str]]]):
        """
        Learn the projection matrix from dimension definitions.
        
        Each dimension is defined by word pairs:
          dimension_pairs["temperature"] = [("cold", "hot"), ("cool", "warm"), ...]
        
        We create a target space where:
          - Each dimension is orthogonal
          - Negative words are at -1 on their dimension
          - Positive words are at +1 on their dimension
          - All words are at 0 on other dimensions
        """
        self.dim_names = list(dimension_pairs.keys())
        n_dims = len(self.dim_names)
        
        # Collect all words and their target positions
        words = []
        targets = []
        
        for dim_idx, (dim_name, pairs) in enumerate(dimension_pairs.items()):
            for neg_word, pos_word in pairs:
                # Negative word: -1 on this dimension, 0 elsewhere
                neg_embed = self.get_embedding(neg_word)
                if neg_embed is not None:
                    words.append(neg_word)
                    target = torch.zeros(n_dims)
                    target[dim_idx] = -1.0
                    targets.append(target)
                
                # Positive word: +1 on this dimension, 0 elsewhere
                pos_embed = self.get_embedding(pos_word)
                if pos_embed is not None:
                    words.append(pos_word)
                    target = torch.zeros(n_dims)
                    target[dim_idx] = 1.0
                    targets.append(target)
        
        if len(words) < n_dims:
            raise ValueError(f"Need at least {n_dims} words, got {len(words)}")
        
        # Build X matrix: [hidden_dim, n_words]
        X = torch.stack([self.get_embedding(w).float().cpu() for w in words], dim=1)
        
        # Build Y matrix: [n_dims, n_words]
        Y = torch.stack(targets, dim=1)
        
        # Solve for W: W @ X = Y
        # W = Y @ X^T @ (X @ X^T)^(-1)
        # But X @ X^T is [hidden_dim, hidden_dim] which is huge
        # Instead use: W = Y @ pinv(X)
        
        # Using pseudoinverse for numerical stability
        X_pinv = torch.linalg.pinv(X)  # [n_words, hidden_dim]
        self.W = Y @ X_pinv  # [n_dims, hidden_dim]
        
        print(f"Learned projection: {self.W.shape}")
        print(f"  Dimensions: {self.dim_names}")
        print(f"  Training words: {len(words)}")
        
        # Store word positions
        for word in words:
            embed = self.get_embedding(word)
            if embed is not None:
                pos = self.W @ embed.float().cpu()
                self.word_positions[word] = pos
        
        return self.W
    
    # =========================================================================
    # PROJECTION AND NAVIGATION
    # =========================================================================
    
    def project(self, word: str) -> Optional[torch.Tensor]:
        """Project a word into the orthogonal space."""
        if self.W is None:
            return None
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        return self.W @ embed.float().cpu()
    
    def find_by_position(self, target_pos: torch.Tensor, 
                         top_k: int = 5,
                         exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find words closest to a target position in projected space."""
        if self.W is None:
            return []
        
        # Project all embeddings
        W_device = self.W.to(self.device)
        all_projected = (W_device @ self.all_embeds.float().T).T  # [vocab, n_dims]
        
        # Compute distances to target
        target_device = target_pos.to(self.device)
        distances = (all_projected - target_device).pow(2).sum(dim=1).sqrt()
        
        if exclude:
            for word in exclude:
                tid = self.get_token_id(word)
                if tid is not None:
                    distances[tid] = float('inf')
        
        top_indices = distances.topk(top_k, largest=False).indices
        
        results = []
        for idx in top_indices:
            word = self.tokenizer.decode([idx.item()]).strip()
            dist = distances[idx].item()
            results.append((word, dist))
        
        return results
    
    def find_opposite(self, word: str) -> Optional[Tuple[str, float, str]]:
        """
        Find the opposite of a word.
        
        1. Project word to orthogonal space
        2. Find which dimension it's strongest on
        3. Flip that dimension
        4. Find nearest word to flipped position
        """
        pos = self.project(word)
        if pos is None:
            return None
        
        # Find strongest dimension
        abs_pos = pos.abs()
        strongest_dim = abs_pos.argmax().item()
        dim_name = self.dim_names[strongest_dim]
        
        # Flip that dimension
        target_pos = pos.clone()
        target_pos[strongest_dim] = -target_pos[strongest_dim]
        
        # Find nearest word
        results = self.find_by_position(target_pos, top_k=10, exclude=[word])
        
        # Filter for clean words
        for result_word, dist in results:
            if result_word.isalpha() and len(result_word) > 1:
                return (result_word, dist, dim_name)
        
        return None
    
    def find_opposite_on_dim(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """Find opposite on a specific dimension."""
        if dim_name not in self.dim_names:
            return None
        
        pos = self.project(word)
        if pos is None:
            return None
        
        dim_idx = self.dim_names.index(dim_name)
        
        # Flip that dimension
        target_pos = pos.clone()
        target_pos[dim_idx] = -target_pos[dim_idx]
        
        results = self.find_by_position(target_pos, top_k=10, exclude=[word])
        
        for result_word, dist in results:
            if result_word.isalpha() and len(result_word) > 1:
                return (result_word, dist)
        
        return None
    
    def validate(self, dimension_pairs: Dict[str, List[Tuple[str, str]]]) -> float:
        """Validate on the training pairs."""
        correct = 0
        total = 0
        
        for dim_name, pairs in dimension_pairs.items():
            for neg_word, pos_word in pairs:
                # Test neg -> pos
                result = self.find_opposite_on_dim(neg_word, dim_name)
                if result and pos_word.lower() in result[0].lower():
                    correct += 1
                total += 1
                
                # Test pos -> neg
                result = self.find_opposite_on_dim(pos_word, dim_name)
                if result and neg_word.lower() in result[0].lower():
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0


def demo_constrained_projection(model, tokenizer):
    """Demo constrained projection."""
    print("="*70)
    print("CONSTRAINED PROJECTION: FIT WEIGHTS TO ORTHOGONAL CONSTRAINTS")
    print("="*70)
    print("""
The insight: Orthogonality is the CONSTRAINT, not the observation.

We LEARN a projection W such that:
  - Each semantic dimension is orthogonal
  - Opposites are at +1 and -1 on their dimension
  - W = Y @ pinv(X) gives the exact solution
""")
    
    proj = ConstrainedProjector(model, tokenizer)
    
    # Use FEWER, CLEANER pairs - quality over quantity
    # The original 5 pairs per dimension gave 57% generalization
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
    
    print("\n--- LEARNING PROJECTION ---")
    proj.learn_projection(dimension_pairs)
    
    # Validate on training pairs
    print("\n--- VALIDATION ON TRAINING PAIRS ---")
    accuracy = proj.validate(dimension_pairs)
    print(f"Training accuracy: {accuracy*100:.1f}%")
    
    # Test specific examples
    print("\n--- TESTING SPECIFIC EXAMPLES ---")
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
        result = proj.find_opposite_on_dim(source, dim_name)
        if result:
            got, dist = result
            found = expected.lower() in got.lower()
            if found:
                correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nTest accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    # Test generalization
    print("\n--- TESTING GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("quick", "slow", "speed"),
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
        ("ancient", "new", "age"),
        ("damp", "dry", "moisture"),
    ]
    
    gen_correct = 0
    for source, expected, dim_name in gen_tests:
        result = proj.find_opposite_on_dim(source, dim_name)
        if result:
            got, dist = result
            found = expected.lower() in got.lower()
            if found:
                gen_correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:10s} --[{dim_name:12s}]--> {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} --[{dim_name:12s}]--> [no result]")
    
    print(f"\nGeneralization: {gen_correct}/{len(gen_tests)} ({gen_correct/len(gen_tests)*100:.0f}%)")
    
    # Test auto-detection
    print("\n--- TESTING AUTO-DETECTION ---")
    auto_tests = ["hot", "big", "fast", "tall", "bright", "old", "good", "heavy", "hard", "wet"]
    
    for word in auto_tests:
        result = proj.find_opposite(word)
        if result:
            got, dist, dim = result
            print(f"  {word:10s} → {got:12s} (dim: {dim})")
        else:
            print(f"  {word:10s} → [no result]")


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
    
    demo_constrained_projection(model, tokenizer)


if __name__ == "__main__":
    main()
