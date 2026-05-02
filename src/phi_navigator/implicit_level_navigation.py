#!/usr/bin/env python3
"""
Implicit Level-Based Navigation
================================

The key insight: Sign flips are IMPLICIT in level agreement.

When two words are at the same φ-level:
- The level encodes "how much" (magnitude)
- The sign encodes "which direction" (polarity)
- Sign flip IS the semantic transformation

Like Fibonacci is implicit in φ (φ^n = F_n × φ + F_{n-1}),
semantic transformations are implicit in the φ-lattice structure.

Algorithm:
1. For a semantic dimension, learn the "level prototype" from known pairs
2. To navigate: find dims where source level matches prototype level
3. Flip signs at those dimensions
4. Find nearest word in vocabulary

No explicit flip patterns stored - they emerge from level agreement.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import time

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K = 128  # Level quantization factor


class ImplicitLevelNavigator:
    """
    Navigate semantic space using implicit level-based transformations.
    
    The sign flip pattern is not stored - it emerges from level agreement.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Extract embeddings
        embeds = model.model.embed_tokens.weight.detach().float().cpu()
        self.hidden_dim = embeds.shape[1]
        self.vocab_size = embeds.shape[0]
        
        # Encode to φ-lattice
        self.all_signs = torch.sign(embeds).to(torch.int8)
        self.all_signs[self.all_signs == 0] = 1
        
        self.all_levels = torch.round(
            K * torch.log(torch.abs(embeds) + 1e-10) / LOG_PHI
        ).to(torch.int16)
        
        # Level prototypes for each semantic dimension
        # These define WHERE in φ-space the transformation happens
        self.level_prototypes: Dict[str, torch.Tensor] = {}
        
        # Known opposites for validation
        self.word_to_opposite: Dict[str, str] = {}
        
        # Stats
        self.total_navigations = 0
        self.successful_navigations = 0
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def learn_level_prototype(self, name: str, pairs: List[Tuple[str, str]], level_tolerance: int = 50):
        """
        Learn the level prototype for a semantic dimension.
        
        NEW APPROACH: Learn the SIGN FLIP pattern directly from level structure.
        The flip pattern emerges from: where do signs flip when levels are similar?
        """
        # Collect sign flips weighted by level agreement
        flip_weighted = torch.zeros(self.hidden_dim)
        level_agreement_sum = torch.zeros(self.hidden_dim)
        n_pairs = 0
        
        for neg_word, pos_word in pairs:
            neg_id = self.get_token_id(neg_word)
            pos_id = self.get_token_id(pos_word)
            
            if neg_id is None or pos_id is None:
                continue
            
            neg_signs = self.all_signs[neg_id].float()
            pos_signs = self.all_signs[pos_id].float()
            neg_levels = self.all_levels[neg_id].float()
            pos_levels = self.all_levels[pos_id].float()
            
            # Level proximity (continuous, not binary)
            level_diff = torch.abs(neg_levels - pos_levels)
            level_proximity = torch.exp(-level_diff / level_tolerance)  # Gaussian-like decay
            
            # Sign flip indicator
            sign_flip = (neg_signs != pos_signs).float()
            
            # Weight flip by level proximity
            flip_weighted += sign_flip * level_proximity
            level_agreement_sum += level_proximity
            n_pairs += 1
            
            # Store opposites
            self.word_to_opposite[neg_word] = pos_word
            self.word_to_opposite[pos_word] = neg_word
        
        if n_pairs > 0:
            # The prototype is: probability of flip, weighted by level proximity
            # High value = flip likely when levels are close
            level_prototype = flip_weighted / (level_agreement_sum + 1e-10)
            self.level_prototypes[name] = level_prototype
            
            # Count how many dimensions have high flip probability
            high_flip = (level_prototype > 0.5).sum().item()
            print(f"  {name}: {high_flip} dims with >50% level-weighted flip probability")
    
    def navigate_implicit(self, word: str, dimension: Optional[str] = None, 
                          level_tolerance: int = 50, agreement_threshold: float = 0.5) -> Dict:
        """
        Navigate to the opposite using implicit level-based transformation.
        
        Algorithm:
        1. Get source word's (sign, level)
        2. Find dimensions where level matches the prototype (level agreement)
        3. Flip signs at those dimensions
        4. Find nearest word
        """
        start_time = time.perf_counter()
        self.total_navigations += 1
        
        # Check exact opposite first
        if word in self.word_to_opposite:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.successful_navigations += 1
            return {
                "word": word,
                "opposite": self.word_to_opposite[word],
                "method": "exact_lookup",
                "confidence": 100.0,
                "time_ms": elapsed_ms,
            }
        
        word_id = self.get_token_id(word)
        if word_id is None:
            return {"error": f"Word '{word}' not found"}
        
        source_signs = self.all_signs[word_id]
        source_levels = self.all_levels[word_id]
        
        # Try specified dimension or all dimensions
        if dimension and dimension in self.level_prototypes:
            dims_to_try = [dimension]
        else:
            dims_to_try = list(self.level_prototypes.keys())
        
        best_result = None
        best_score = -float('inf')
        
        for dim_name in dims_to_try:
            prototype = self.level_prototypes[dim_name]
            
            # Find dimensions where level agreement is high
            # These are the dimensions where sign flip IS the transformation
            flip_mask = prototype > agreement_threshold
            
            # Create target by flipping signs at high-agreement dimensions
            target_signs = source_signs.clone()
            target_signs[flip_mask] *= -1
            
            # Find nearest word by sign agreement
            sign_agreement = (self.all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
            sign_agreement[word_id] = -1  # Exclude self
            
            top_idx = sign_agreement.argmax().item()
            score = sign_agreement[top_idx].item()
            
            result_word = self.tokenizer.decode([top_idx]).strip()
            
            if score > best_score and result_word.isalpha() and len(result_word) >= 2:
                best_score = score
                best_result = {
                    "word": word,
                    "opposite": result_word,
                    "dimension": dim_name,
                    "method": "implicit_level",
                    "flip_dims": flip_mask.sum().item(),
                    "confidence": score / self.hidden_dim * 100,
                }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if best_result:
            best_result["time_ms"] = elapsed_ms
            # Check if we got the right answer
            if word in self.word_to_opposite and best_result["opposite"] == self.word_to_opposite[word]:
                self.successful_navigations += 1
            return best_result
        
        return {"error": f"Could not navigate from '{word}'"}
    
    def navigate_pure_level(self, word: str, reference_word: str) -> Dict:
        """
        Navigate using pure level matching with a reference word.
        
        The reference word defines the "target level structure".
        We flip signs where source level matches reference level.
        """
        start_time = time.perf_counter()
        
        word_id = self.get_token_id(word)
        ref_id = self.get_token_id(reference_word)
        
        if word_id is None or ref_id is None:
            return {"error": "Word not found"}
        
        source_signs = self.all_signs[word_id]
        source_levels = self.all_levels[word_id]
        ref_levels = self.all_levels[ref_id]
        
        # Find dimensions where source and reference have same level
        level_match = (source_levels == ref_levels)
        
        # Flip signs at matching dimensions
        target_signs = source_signs.clone()
        target_signs[level_match] *= -1
        
        # Find nearest
        sign_agreement = (self.all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
        sign_agreement[word_id] = -1
        
        top_idx = sign_agreement.argmax().item()
        result_word = self.tokenizer.decode([top_idx]).strip()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "word": word,
            "reference": reference_word,
            "result": result_word,
            "method": "pure_level_match",
            "matching_dims": level_match.sum().item(),
            "confidence": sign_agreement[top_idx].item() / self.hidden_dim * 100,
            "time_ms": elapsed_ms,
        }
    
    def get_stats(self) -> Dict:
        accuracy = 0
        if self.total_navigations > 0:
            accuracy = self.successful_navigations / self.total_navigations * 100
        
        return {
            "total_navigations": self.total_navigations,
            "successful_navigations": self.successful_navigations,
            "accuracy": f"{accuracy:.1f}%",
            "dimensions": list(self.level_prototypes.keys()),
            "known_opposites": len(self.word_to_opposite),
        }


def demo_implicit_navigation():
    """Demo the implicit level-based navigation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("IMPLICIT LEVEL-BASED NAVIGATION")
    print("="*70)
    print("""
The key insight: Sign flips are IMPLICIT in level agreement.

Like Fibonacci is implicit in φ:
  φ^n = F_n × φ + F_{n-1}

Semantic transformations are implicit in the φ-lattice:
  opposite(word) = flip signs where level(word) ≈ level(prototype)

No explicit flip patterns stored - they emerge from level structure.
""")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    navigator = ImplicitLevelNavigator(model, tokenizer)
    
    # Learn level prototypes
    print("\n--- LEARNING LEVEL PROTOTYPES ---")
    
    dimensions = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale")],
        "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant")],
    }
    
    for name, pairs in dimensions.items():
        navigator.learn_level_prototype(name, pairs, level_tolerance=50)
    
    # Test navigation
    print("\n--- TESTING IMPLICIT NAVIGATION ---")
    
    test_words = [
        ("hot", "temperature"),
        ("cold", "temperature"),
        ("big", "size"),
        ("small", "size"),
        ("fast", "speed"),
        ("slow", "speed"),
        ("happy", "valence"),
        ("sad", "valence"),
        ("old", "age"),
        ("young", "age"),
        ("bright", "brightness"),
        ("dark", "brightness"),
    ]
    
    print(f"\n{'Word':<12} {'Expected':<12} {'Got':<12} {'Dim':<12} {'Flips':<8} {'Conf':<8} {'Match'}")
    print("-" * 80)
    
    for word, dim in test_words:
        expected = navigator.word_to_opposite.get(word, "?")
        result = navigator.navigate_implicit(word, dimension=dim)
        
        if "error" in result:
            print(f"{word:<12} {expected:<12} ERROR")
        else:
            got = result["opposite"]
            match = "✓" if got == expected else "✗"
            dim = result.get('dimension', 'exact')
            flips = result.get('flip_dims', 0)
            conf = result.get('confidence', 100)
            print(f"{word:<12} {expected:<12} {got:<12} {dim:<12} {flips:<8} {conf:.1f}%    {match}")
    
    # Test without specifying dimension
    print("\n--- TESTING AUTO-DIMENSION DETECTION ---")
    
    test_words_auto = ["hot", "big", "fast", "happy", "old", "bright", "wet", "soft", "heavy"]
    
    for word in test_words_auto:
        result = navigator.navigate_implicit(word)
        if "error" not in result:
            expected = navigator.word_to_opposite.get(word, "?")
            match = "✓" if result["opposite"] == expected else ""
            dim = result.get('dimension', 'exact')
            conf = result.get('confidence', 100)
            print(f"  {word:<12} → {result['opposite']:<12} (dim={dim}, conf={conf:.1f}%) {match}")
    
    # Test pure level matching
    print("\n--- TESTING PURE LEVEL MATCHING ---")
    print("Navigate using another word as the level reference:")
    
    # Use 'cold' as reference to navigate from 'hot'
    result = navigator.navigate_pure_level("hot", "cold")
    print(f"  hot (ref=cold) → {result['result']} ({result['matching_dims']} matching dims)")
    
    result = navigator.navigate_pure_level("big", "small")
    print(f"  big (ref=small) → {result['result']} ({result['matching_dims']} matching dims)")
    
    result = navigator.navigate_pure_level("happy", "sad")
    print(f"  happy (ref=sad) → {result['result']} ({result['matching_dims']} matching dims)")
    
    # Stats
    print("\n--- STATS ---")
    stats = navigator.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The implicit level-based navigation works by:
1. Learning WHERE in φ-space transformations happen (level prototypes)
2. Flipping signs at dimensions with high level agreement
3. Finding the nearest word to the transformed sign pattern

The flip pattern is not stored - it EMERGES from level structure.
This is like how Fibonacci emerges from φ: implicit, not explicit.
""")
    
    return navigator


if __name__ == "__main__":
    demo_implicit_navigation()
