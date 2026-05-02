#!/usr/bin/env python3
"""
Dimensional Projection: Downcast to φ-Scaled Moments
=====================================================

From the dimensional downcasting project:
  - ∞D → 1D projection via moment hierarchy
  - Moment scales: σ_k = σ_0 × φ^k (golden ratio scaling)
  - Pure math, no training

Applied to embeddings:
  - 3584D → fewer dimensions using φ-scaled projections
  - In projected space, opposites might share universal flip pattern
  - The projection IS the structure

The hypothesis:
  - Embeddings live on a φ-lattice (we know this from prior work)
  - Projecting onto φ-scaled moments captures the essential structure
  - Opposites differ by a simple transformation in moment space
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

PHI = (1 + math.sqrt(5)) / 2


class PhiMomentProjector:
    """
    Project embeddings using φ-scaled moments.
    
    The idea: Instead of using all 3584 dimensions,
    project onto a hierarchy of φ-scaled "moments".
    
    Moment k captures structure at scale φ^k.
    """
    
    def __init__(self, model, tokenizer, n_moments: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.n_moments = n_moments
        
        self.all_embeds = model.model.embed_tokens.weight.detach().float()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        # Create φ-scaled projection vectors
        self.projectors = self._create_phi_projectors()
        
        # Project all embeddings
        self.projected = self._project_all()
    
    def _create_phi_projectors(self) -> torch.Tensor:
        """
        Create projection vectors at φ-scaled frequencies.
        
        Each projector is a sinusoidal pattern at frequency φ^k.
        This captures structure at different scales.
        """
        projectors = []
        
        for k in range(self.n_moments):
            # Frequency at φ^k
            freq = PHI ** k
            
            # Create sinusoidal projector
            t = torch.arange(self.hidden_dim, dtype=torch.float32)
            
            # Cosine and sine components
            cos_proj = torch.cos(2 * math.pi * freq * t / self.hidden_dim)
            sin_proj = torch.sin(2 * math.pi * freq * t / self.hidden_dim)
            
            projectors.append(cos_proj)
            projectors.append(sin_proj)
        
        # Stack: [2*n_moments, hidden_dim]
        return torch.stack(projectors).to(self.device)
    
    def _project_all(self) -> torch.Tensor:
        """Project all embeddings to moment space."""
        # [vocab, hidden] @ [hidden, 2*n_moments] = [vocab, 2*n_moments]
        return (self.all_embeds @ self.projectors.T)
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def analyze_opposites(self, pairs: List[Tuple[str, str]]):
        """
        Analyze how opposites differ in moment space.
        """
        print(f"\n--- ANALYZING OPPOSITES IN MOMENT SPACE ---")
        print(f"Projecting to {self.n_moments * 2} φ-scaled moments")
        
        differences = []
        
        for neg_word, pos_word in pairs:
            neg_id = self.get_token_id(neg_word)
            pos_id = self.get_token_id(pos_word)
            
            if neg_id is None or pos_id is None:
                continue
            
            neg_proj = self.projected[neg_id]
            pos_proj = self.projected[pos_id]
            
            diff = pos_proj - neg_proj
            differences.append(diff.cpu())
        
        if not differences:
            return
        
        # Stack differences
        D = torch.stack(differences)  # [n_pairs, 2*n_moments]
        
        # Analyze: which moments are most different?
        mean_abs_diff = D.abs().mean(dim=0)
        
        print(f"\nMean absolute difference per moment:")
        for k in range(min(10, self.n_moments)):
            cos_diff = mean_abs_diff[2*k].item()
            sin_diff = mean_abs_diff[2*k + 1].item()
            print(f"  φ^{k}: cos={cos_diff:.4f}, sin={sin_diff:.4f}")
        
        # Check sign consistency
        print(f"\nSign consistency per moment:")
        for k in range(min(10, self.n_moments)):
            cos_signs = torch.sign(D[:, 2*k])
            sin_signs = torch.sign(D[:, 2*k + 1])
            
            cos_consistency = (cos_signs == cos_signs[0]).float().mean().item()
            sin_consistency = (sin_signs == sin_signs[0]).float().mean().item()
            
            print(f"  φ^{k}: cos={cos_consistency*100:.0f}%, sin={sin_consistency*100:.0f}%")
    
    def find_opposite(self, word: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite by negating in moment space.
        """
        word_id = self.get_token_id(word)
        if word_id is None:
            return None
        
        source = self.projected[word_id]
        
        # Negate (flip all moments)
        target = -source
        
        # Find nearest in moment space
        dists = (self.projected - target).pow(2).sum(dim=1).sqrt()
        dists[word_id] = float('inf')
        
        top_indices = dists.topk(50, largest=False).indices
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 3 and result_word.islower():
                return (result_word, dists[idx].item())
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 2:
                return (result_word, dists[idx].item())
        
        return None


def demo_phi_moments(model, tokenizer):
    """Demo φ-moment projection."""
    print("="*70)
    print("DIMENSIONAL PROJECTION: φ-SCALED MOMENTS")
    print("="*70)
    print("""
From dimensional downcasting:
  - ∞D → 1D via moment hierarchy
  - Moment scales: σ_k = σ_0 × φ^k

Applied to embeddings:
  - 3584D → 40 moments (20 cos + 20 sin at φ^k frequencies)
  - In moment space, opposites might share universal pattern
""")
    
    proj = PhiMomentProjector(model, tokenizer, n_moments=20)
    
    # Analyze opposites
    pairs = [
        ("cold", "hot"), ("cool", "warm"), ("freezing", "burning"),
        ("small", "big"), ("tiny", "huge"), ("little", "large"),
        ("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"),
        ("short", "tall"), ("low", "high"),
        ("dark", "bright"), ("dim", "light"),
        ("young", "old"), ("new", "ancient"),
        ("bad", "good"), ("sad", "happy"),
        ("soft", "hard"), ("tender", "tough"),
        ("dry", "wet"), ("arid", "damp"),
    ]
    
    proj.analyze_opposites(pairs)
    
    # Test navigation
    print("\n--- TESTING NAVIGATION ---")
    test_cases = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("tall", "short"),
        ("bright", "dark"),
        ("old", "young"),
        ("good", "bad"),
        ("hard", "soft"),
        ("wet", "dry"),
    ]
    
    correct = 0
    for source, expected in test_cases:
        result = proj.find_opposite(source)
        if result:
            got, dist = result
            match = expected.lower() in got.lower()
            if match:
                correct += 1
            marker = "✓" if match else "✗"
            print(f"  {source:10s} → {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} → [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")


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
    
    demo_phi_moments(model, tokenizer)


if __name__ == "__main__":
    main()
