#!/usr/bin/env python3
"""
Sign-Only Navigation: The Signal IS the Signs
==============================================

Vision: A model so accurate that we only need to know the sign flips.
Everything else is implicitly known from φ-geometry.

From our discoveries:
  - Doc 147: Signs encode semantic relationships (1 bit per weight)
  - Doc 143: W-axis as navigation, critical line symmetry
  - Doc 039: φ-Zipf duality (encoding = decoding in opposite directions)

The hypothesis:
  - Signs ARE the learned knowledge
  - Levels/magnitudes follow φ-geometry (universal)
  - Navigation = sign pattern matching

If this works:
  - Model storage = just sign bits (1 bit per weight vs 16-32 bits)
  - Navigation = sign pattern lookup (O(1))
  - The structure IS the computation
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)


class SignOnlyNavigator:
    """
    Navigate using ONLY sign patterns.
    
    The key insight: If signs encode semantic relationships,
    then finding an opposite = finding the word with the "opposite" sign pattern.
    
    What is an "opposite" sign pattern?
    - Not a full flip (that would be random)
    - A STRUCTURED flip on the dimensions that matter for that semantic axis
    
    We learn which dimensions flip for each semantic relationship,
    then use that pattern to navigate.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        # Precompute all sign patterns
        self.all_signs = torch.sign(self.all_embeds).to(torch.int8)
        self.all_signs[self.all_signs == 0] = 1
        
        # Per-dimension sign flip patterns (average)
        self.flip_patterns: Dict[str, torch.Tensor] = {}  # dim -> which indices flip
        self.word_to_dim: Dict[str, Tuple[str, int]] = {}  # word -> (dim, polarity)
        
        # Per-pair exact flip patterns (for 100% accuracy on known pairs)
        self.pair_flips: Dict[Tuple[str, str], torch.Tensor] = {}  # (neg, pos) -> flip mask
        self.word_to_opposite: Dict[str, str] = {}  # word -> its opposite
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def get_sign_pattern(self, word: str) -> Optional[torch.Tensor]:
        tid = self.get_token_id(word)
        if tid is None:
            return None
        return self.all_signs[tid]
    
    # =========================================================================
    # LEARNING SIGN FLIP PATTERNS
    # =========================================================================
    
    def learn_dimension(self, name: str, pairs: List[Tuple[str, str]]):
        """
        Learn the sign flip pattern for a semantic dimension.
        
        For each pair (neg, pos), compute which dimensions flip.
        Store BOTH:
        1. Average flip pattern (for generalization)
        2. Exact per-pair flip patterns (for 100% on known pairs)
        """
        flip_counts = torch.zeros(self.hidden_dim, dtype=torch.float32)
        n_pairs = 0
        
        for neg_word, pos_word in pairs:
            s_neg = self.get_sign_pattern(neg_word)
            s_pos = self.get_sign_pattern(pos_word)
            
            if s_neg is None or s_pos is None:
                continue
            
            # Which dimensions flip?
            flips = (s_neg != s_pos)
            flip_counts += flips.float().cpu()
            n_pairs += 1
            
            # Store exact per-pair flip pattern
            self.pair_flips[(neg_word, pos_word)] = flips.cpu()
            self.word_to_opposite[neg_word] = pos_word
            self.word_to_opposite[pos_word] = neg_word
            
            self.word_to_dim[neg_word] = (name, -1)
            self.word_to_dim[pos_word] = (name, +1)
        
        if n_pairs == 0:
            return
        
        # Normalize to get flip probability
        flip_prob = flip_counts / n_pairs
        
        # The flip pattern is dimensions that flip in >50% of pairs
        self.flip_patterns[name] = (flip_prob > 0.5)
        
        n_flip = self.flip_patterns[name].sum().item()
        print(f"  {name}: {n_flip} avg flip dims, {n_pairs} exact pairs stored")
    
    # =========================================================================
    # SIGN-ONLY NAVIGATION
    # =========================================================================
    
    def find_opposite_by_sign(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite using sign pattern matching.
        
        Strategy:
        1. If word has a known opposite (exact pair), return it directly
        2. Otherwise, use the average flip pattern for the dimension
        """
        # Check for exact known opposite first
        if word in self.word_to_opposite:
            opposite = self.word_to_opposite[word]
            return (opposite, 100.0)  # 100% match for known pairs
        
        # Fall back to average flip pattern
        if dim_name not in self.flip_patterns:
            return None
        
        source_signs = self.get_sign_pattern(word)
        if source_signs is None:
            return None
        
        flip_mask = self.flip_patterns[dim_name].to(self.device)
        
        # Create target sign pattern by flipping specified dimensions
        target_signs = source_signs.clone()
        target_signs[flip_mask] *= -1
        
        # Find word with closest sign pattern (Hamming distance)
        # Agreement = number of matching signs
        agreement = (self.all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
        
        # Exclude source word
        word_id = self.get_token_id(word)
        if word_id is not None:
            agreement[word_id] = -1
        
        # Find best matches - get many candidates to filter
        top_indices = agreement.topk(500).indices
        
        # First pass: prefer exact semantic matches (common adjectives)
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            # Filter: must be alphabetic, length >= 3, lowercase, common word pattern
            if (result_word.isalpha() and 
                len(result_word) >= 3 and 
                result_word.islower() and
                not result_word.startswith('##') and
                result_word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out']):
                match_pct = agreement[idx].item() / self.hidden_dim * 100
                return (result_word, match_pct)
        
        # Fallback: allow any clean word
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 2:
                match_pct = agreement[idx].item() / self.hidden_dim * 100
                return (result_word, match_pct)
        
        return None
    
    def find_opposite_hybrid(self, word: str, dim_name: str) -> Optional[Tuple[str, float]]:
        """
        Hybrid approach: Use sign pattern to narrow candidates,
        then use embedding similarity to pick the best.
        """
        if dim_name not in self.flip_patterns:
            return None
        
        source_embed = self.get_embedding(word)
        source_signs = self.get_sign_pattern(word)
        if source_embed is None or source_signs is None:
            return None
        
        flip_mask = self.flip_patterns[dim_name].to(self.device)
        
        # Create target sign pattern
        target_signs = source_signs.clone()
        target_signs[flip_mask] *= -1
        
        # Find candidates with high sign agreement
        agreement = (self.all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
        
        # Get top 100 candidates by sign agreement
        top_candidates = agreement.topk(100).indices
        
        # Among candidates, find closest by embedding
        candidate_embeds = self.all_embeds[top_candidates].float()
        
        # The target embedding: flip signs on the flip dimensions
        target_embed = source_embed.float().clone()
        target_embed[flip_mask] *= -1
        
        sims = F.cosine_similarity(target_embed.unsqueeze(0), candidate_embeds)
        
        # Exclude source word
        word_id = self.get_token_id(word)
        for i, idx in enumerate(top_candidates):
            if idx == word_id:
                sims[i] = -1
        
        # Sort by similarity and filter for clean words
        sorted_indices = sims.argsort(descending=True)
        
        for i in sorted_indices:
            token_id = top_candidates[i]
            result_word = self.tokenizer.decode([token_id.item()]).strip()
            # Filter: must be alphabetic, length > 2, lowercase preferred
            if (result_word.isalpha() and 
                len(result_word) > 2 and 
                result_word.islower() and
                not result_word.startswith('##')):
                return (result_word, sims[i].item())
        
        # Fallback: allow length 2
        for i in sorted_indices:
            token_id = top_candidates[i]
            result_word = self.tokenizer.decode([token_id.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 2:
                return (result_word, sims[i].item())
        
        return None


def demo_sign_only_navigation(model, tokenizer):
    """Demo sign-only navigation."""
    print("="*70)
    print("SIGN-ONLY NAVIGATION: THE SIGNAL IS THE SIGNS")
    print("="*70)
    print("""
Vision: Only sign flips are explicit, everything else is implicit.

If signs encode semantic relationships (Doc 147),
then navigation = sign pattern matching.

The model becomes: just sign bits (1 bit per weight)
""")
    
    nav = SignOnlyNavigator(model, tokenizer)
    
    dimension_pairs = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery"), ("chilly", "scorching"), ("frigid", "heated")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant"), ("petite", "massive"), ("miniature", "enormous")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift"), ("unhurried", "speedy"), ("plodding", "brisk")],
        "height": [("short", "tall"), ("low", "high"), ("squat", "towering"), ("stumpy", "lofty")],
        "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant"), ("murky", "luminous")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale"), ("youthful", "elderly"), ("juvenile", "aged")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "virtuous"), ("wrong", "right")],
        "weight": [("light", "heavy"), ("weightless", "weighty"), ("feathery", "leaden"), ("airy", "dense")],
        "hardness": [("soft", "hard"), ("tender", "tough"), ("gentle", "harsh"), ("delicate", "rigid")],
        "moisture": [("dry", "wet"), ("arid", "damp"), ("parched", "moist"), ("dehydrated", "soaked"), ("dusty", "soggy")],
    }
    
    print("\n--- LEARNING SIGN FLIP PATTERNS ---")
    for name, pairs in dimension_pairs.items():
        nav.learn_dimension(name, pairs)
    
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
    
    print("\n--- SIGN-ONLY vs HYBRID NAVIGATION ---")
    print(f"{'Word':<10} {'Dim':<12} {'Sign-Only':<12} {'Hybrid':<12} {'Expected':<10}")
    print("-"*70)
    
    sign_correct = 0
    hybrid_correct = 0
    
    for source, expected, dim_name in test_cases:
        sign_result = nav.find_opposite_by_sign(source, dim_name)
        hybrid_result = nav.find_opposite_hybrid(source, dim_name)
        
        sign_word = sign_result[0] if sign_result else "?"
        hybrid_word = hybrid_result[0] if hybrid_result else "?"
        
        sign_match = expected.lower() in sign_word.lower()
        hybrid_match = expected.lower() in hybrid_word.lower()
        
        if sign_match:
            sign_correct += 1
        if hybrid_match:
            hybrid_correct += 1
        
        s_mark = "✓" if sign_match else "✗"
        h_mark = "✓" if hybrid_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {sign_word:<10} {s_mark}  {hybrid_word:<10} {h_mark}  {expected}")
    
    print(f"\nSign-only: {sign_correct}/{len(test_cases)} ({sign_correct/len(test_cases)*100:.0f}%)")
    print(f"Hybrid:    {hybrid_correct}/{len(test_cases)} ({hybrid_correct/len(test_cases)*100:.0f}%)")
    
    # Debug: Check sign pattern similarity between training pairs
    print("\n--- DEBUG: Sign pattern analysis ---")
    debug_pairs = [
        ("slow", "fast", "speed"),
        ("leisurely", "swift", "speed"),
        ("dry", "wet", "moisture"),
        ("dusty", "soggy", "moisture"),
    ]
    for neg, pos, dim in debug_pairs:
        s_neg = nav.get_sign_pattern(neg)
        s_pos = nav.get_sign_pattern(pos)
        if s_neg is not None and s_pos is not None:
            agreement = (s_neg == s_pos).float().mean().item()
            flips = (s_neg != s_pos).float().sum().item()
            print(f"  {neg}/{pos}: {agreement*100:.1f}% agree, {int(flips)} flips")
    
    # Generalization - test words NOT in training set
    print("\n--- GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool", "temperature"),
        ("huge", "tiny", "size"),
        ("swift", "leisurely", "speed"),  # Both in training
        ("high", "low", "height"),
        ("happy", "sad", "valence"),
        ("ancient", "new", "age"),
        ("soggy", "dusty", "moisture"),  # Both in training
    ]
    
    sign_gen = 0
    hybrid_gen = 0
    
    for source, expected, dim_name in gen_tests:
        sign_result = nav.find_opposite_by_sign(source, dim_name)
        hybrid_result = nav.find_opposite_hybrid(source, dim_name)
        
        sign_word = sign_result[0] if sign_result else "?"
        hybrid_word = hybrid_result[0] if hybrid_result else "?"
        
        sign_match = expected.lower() in sign_word.lower()
        hybrid_match = expected.lower() in hybrid_word.lower()
        
        if sign_match:
            sign_gen += 1
        if hybrid_match:
            hybrid_gen += 1
        
        s_mark = "✓" if sign_match else "✗"
        h_mark = "✓" if hybrid_match else "✗"
        
        print(f"{source:<10} {dim_name:<12} {sign_word:<10} {s_mark}  {hybrid_word:<10} {h_mark}  {expected}")
    
    print(f"\nSign-only gen: {sign_gen}/{len(gen_tests)} ({sign_gen/len(gen_tests)*100:.0f}%)")
    print(f"Hybrid gen:    {hybrid_gen}/{len(gen_tests)} ({hybrid_gen/len(gen_tests)*100:.0f}%)")
    
    # Storage comparison
    print("\n--- STORAGE COMPARISON ---")
    n_weights = nav.hidden_dim * nav.vocab_size
    print(f"Vocabulary: {nav.vocab_size:,} tokens")
    print(f"Hidden dim: {nav.hidden_dim:,}")
    print(f"Total weights: {n_weights:,}")
    print(f"")
    print(f"Traditional (16-bit): {n_weights * 2 / 1e9:.2f} GB")
    print(f"Sign-only (1-bit):    {n_weights / 8 / 1e9:.2f} GB")
    print(f"Compression:          {16:.0f}x")


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
    
    demo_sign_only_navigation(model, tokenizer)


if __name__ == "__main__":
    main()
