#!/usr/bin/env python3
"""
Local Structure Navigation
==========================

The holographic insight applied correctly:
  - Don't learn a global template
  - Extract the relationship from LOCAL structure

In stereo: E = I_synth - I encodes depth gradients LOCALLY
In semantics: The neighborhood of a word encodes its relationships

Hypothesis:
  The opposite of "hot" is the word that is:
  1. Semantically related (close in some dimensions)
  2. Semantically opposed (flipped in other dimensions)
  
The LOCAL structure around "hot" should reveal "cold".

Approach:
  1. Find words that are "structurally similar" to the source
  2. Among those, find the one that is "maximally opposed"
  3. The opposition pattern is LOCAL, not global
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates, PhiPoint

PHI = (1 + math.sqrt(5)) / 2


class LocalStructureNavigator:
    """
    Navigate using local structure instead of global templates.
    
    Key insight: The relationship is encoded in the LOCAL neighborhood,
    not in a global transformation.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.coordinates = PhiCoordinates()
        
        # Pre-compute all embeddings and φ-points
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.vocab_size = self.all_embeds.shape[0]
        
        # Cache
        self._phi_cache: Dict[int, PhiPoint] = {}
    
    def get_phi_point(self, token_id: int) -> PhiPoint:
        if token_id in self._phi_cache:
            return self._phi_cache[token_id]
        embed = self.all_embeds[token_id].cpu()
        point = self.coordinates.encode(embed)
        self._phi_cache[token_id] = point
        return point
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id]).strip()
    
    # =========================================================================
    # LOCAL STRUCTURE ANALYSIS
    # =========================================================================
    
    def find_semantic_neighbors(self, word: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Find semantically similar words (neighbors in embedding space).
        """
        token_id = self.get_token_id(word)
        if token_id is None:
            return []
        
        embed = self.all_embeds[token_id]
        sims = F.cosine_similarity(embed.unsqueeze(0), self.all_embeds)
        sims[token_id] = -1  # Exclude self
        
        top_indices = sims.topk(top_k).indices
        return [(idx.item(), sims[idx].item()) for idx in top_indices]
    
    def compute_opposition_score(self, source_id: int, target_id: int) -> float:
        """
        Compute how "opposed" two words are in φ-space.
        
        Opposition = high sign flip rate in "important" dimensions
        """
        p1 = self.get_phi_point(source_id)
        p2 = self.get_phi_point(target_id)
        
        # Sign flips
        sign_flips = (p1.signs != p2.signs).float()
        
        # Weight by level magnitude (important dimensions)
        level_importance = (p1.levels.float().abs() + p2.levels.float().abs()) / 2
        level_importance = level_importance / level_importance.max()
        
        # Weighted opposition score
        weighted_flips = (sign_flips * level_importance).sum()
        
        return weighted_flips.item()
    
    def find_opposite_local(self, word: str, n_neighbors: int = 200) -> Optional[Tuple[str, float]]:
        """
        Find the opposite using local structure.
        
        Algorithm:
        1. Find semantic neighbors (similar words)
        2. Among neighbors, find the one with highest opposition score
        3. This is the "locally opposed" word
        
        The intuition: "cold" is similar to "hot" (both temperature words)
        but maximally opposed in the temperature dimension.
        """
        source_id = self.get_token_id(word)
        if source_id is None:
            return None
        
        # Get neighbors
        neighbors = self.find_semantic_neighbors(word, n_neighbors)
        
        if not neighbors:
            return None
        
        # Score each neighbor by opposition
        scored = []
        for neighbor_id, sim in neighbors:
            opposition = self.compute_opposition_score(source_id, neighbor_id)
            # Combined score: similar but opposed
            combined = sim * opposition
            scored.append((neighbor_id, combined, sim, opposition))
        
        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top result
        if scored:
            best_id, combined, sim, opp = scored[0]
            return self.decode_token(best_id), combined
        
        return None
    
    def find_opposite_balanced(self, word: str, n_neighbors: int = 500,
                                sim_weight: float = 0.3,
                                opp_weight: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Find opposite with balanced similarity/opposition weighting.
        """
        source_id = self.get_token_id(word)
        if source_id is None:
            return None
        
        neighbors = self.find_semantic_neighbors(word, n_neighbors)
        
        if not neighbors:
            return None
        
        scored = []
        for neighbor_id, sim in neighbors:
            opposition = self.compute_opposition_score(source_id, neighbor_id)
            # Normalize opposition to [0, 1] range approximately
            opp_norm = opposition / 1000  # Rough normalization
            combined = sim_weight * sim + opp_weight * opp_norm
            scored.append((neighbor_id, combined, sim, opposition))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if scored:
            best_id, combined, sim, opp = scored[0]
            return self.decode_token(best_id), combined
        
        return None


def demo_local_structure(model, tokenizer):
    """Demo local structure navigation."""
    print("="*70)
    print("LOCAL STRUCTURE NAVIGATION")
    print("="*70)
    
    nav = LocalStructureNavigator(model, tokenizer)
    
    test_words = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("tall", "short"),
        ("bright", "dark"),
        ("young", "old"),
        ("rich", "poor"),
        ("good", "bad"),
        ("happy", "sad"),
        ("up", "down"),
    ]
    
    print("\nFinding opposites using LOCAL structure:")
    print("-"*70)
    
    correct = 0
    for source, expected in test_words:
        result = nav.find_opposite_local(source, n_neighbors=200)
        if result:
            got, score = result
            found = expected.lower() in got.lower()
            if found:
                correct += 1
            marker = "✓" if found else "✗"
            print(f"  {source:8s} → {got:12s} (expected: {expected:8s}) {marker}")
        else:
            print(f"  {source:8s} → [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_words)} ({correct/len(test_words)*100:.0f}%)")
    
    # Try balanced approach
    print("\n" + "-"*70)
    print("Trying BALANCED weighting (0.3 sim + 0.7 opp):")
    print("-"*70)
    
    correct_balanced = 0
    for source, expected in test_words:
        result = nav.find_opposite_balanced(source, n_neighbors=500)
        if result:
            got, score = result
            found = expected.lower() in got.lower()
            if found:
                correct_balanced += 1
            marker = "✓" if found else "✗"
            print(f"  {source:8s} → {got:12s} (expected: {expected:8s}) {marker}")
        else:
            print(f"  {source:8s} → [no result]")
    
    print(f"\nBalanced accuracy: {correct_balanced}/{len(test_words)} ({correct_balanced/len(test_words)*100:.0f}%)")


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
    
    demo_local_structure(model, tokenizer)


if __name__ == "__main__":
    main()
