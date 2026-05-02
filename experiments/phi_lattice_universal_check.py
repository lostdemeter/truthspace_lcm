#!/usr/bin/env python3
"""
φ-Lattice Universal Transformation Check
=========================================

Design 137 says φ is a universal adapter for VALUES.
But our opposite transformation doesn't generalize.

Question: Is there a STRUCTURAL similarity between different opposite pairs
that we're missing?

Hypothesis: The transformation might be universal at a STRUCTURAL level,
not at a dimension-by-dimension level.

Let's check:
1. What's the STRUCTURE of each pair's transformation?
2. Are there patterns that ARE universal?
3. What makes φ universal for weights but not for semantics?
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from collections import Counter

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


class UniversalChecker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        self.all_embeds = model.model.embed_tokens.weight.detach()
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def analyze_transformation(self, word1: str, word2: str) -> Dict:
        """Analyze the transformation between two words in φ-space."""
        e1 = self.get_embedding(word1)
        e2 = self.get_embedding(word2)
        
        if e1 is None or e2 is None:
            return {"error": "Word not found"}
        
        l1, s1 = encode_phi(e1)
        l2, s2 = encode_phi(e2)
        
        # Level analysis
        level_delta = (l2.float() - l1.float())
        
        # Sign analysis
        sign_flip = (s1 != s2)
        
        return {
            "word1": word1,
            "word2": word2,
            "level_delta_mean": level_delta.mean().item(),
            "level_delta_std": level_delta.std().item(),
            "level_delta_min": level_delta.min().item(),
            "level_delta_max": level_delta.max().item(),
            "n_sign_flips": sign_flip.sum().item(),
            "pct_sign_flips": sign_flip.float().mean().item() * 100,
            "flip_dims": sign_flip.nonzero().squeeze().tolist(),
        }
    
    def compare_transformations(self, pairs: List[Tuple[str, str]]) -> Dict:
        """Compare transformations across multiple pairs."""
        analyses = []
        for w1, w2 in pairs:
            a = self.analyze_transformation(w1, w2)
            if "error" not in a:
                analyses.append(a)
        
        if not analyses:
            return {}
        
        # Statistics across pairs
        level_means = [a["level_delta_mean"] for a in analyses]
        level_stds = [a["level_delta_std"] for a in analyses]
        n_flips = [a["n_sign_flips"] for a in analyses]
        pct_flips = [a["pct_sign_flips"] for a in analyses]
        
        # Find common flip dimensions
        all_flip_sets = []
        for a in analyses:
            flips = a["flip_dims"]
            if isinstance(flips, int):
                flips = [flips]
            all_flip_sets.append(set(flips))
        
        # Intersection (dims that flip for ALL pairs)
        common_flips = all_flip_sets[0]
        for s in all_flip_sets[1:]:
            common_flips = common_flips & s
        
        # Union (dims that flip for ANY pair)
        any_flips = set()
        for s in all_flip_sets:
            any_flips = any_flips | s
        
        return {
            "n_pairs": len(analyses),
            "level_delta_mean": np.mean(level_means),
            "level_delta_std_of_means": np.std(level_means),
            "avg_n_flips": np.mean(n_flips),
            "std_n_flips": np.std(n_flips),
            "avg_pct_flips": np.mean(pct_flips),
            "n_common_flips": len(common_flips),
            "n_any_flips": len(any_flips),
            "common_flip_dims": sorted(list(common_flips))[:20],
        }
    
    def check_structural_similarity(self, pairs: List[Tuple[str, str]]) -> None:
        """Check if transformations have structural similarity."""
        print("\n" + "="*70)
        print("STRUCTURAL SIMILARITY ANALYSIS")
        print("="*70)
        
        # Analyze each pair
        print("\nPer-pair analysis:")
        for w1, w2 in pairs:
            a = self.analyze_transformation(w1, w2)
            if "error" in a:
                continue
            print(f"  {w1:8s} → {w2:8s}: Δlevel={a['level_delta_mean']:+.1f}, "
                  f"flips={a['n_sign_flips']} ({a['pct_sign_flips']:.1f}%)")
        
        # Compare across pairs
        print("\nCross-pair comparison:")
        comparison = self.compare_transformations(pairs)
        print(f"  Level delta mean: {comparison['level_delta_mean']:.1f} ± {comparison['level_delta_std_of_means']:.1f}")
        print(f"  Sign flips: {comparison['avg_n_flips']:.0f} ± {comparison['std_n_flips']:.0f}")
        print(f"  Common flip dims (ALL pairs): {comparison['n_common_flips']}")
        print(f"  Any flip dims (ANY pair): {comparison['n_any_flips']}")
        
        if comparison['n_common_flips'] > 0:
            print(f"  Common dims: {comparison['common_flip_dims']}")
    
    def check_phi_universality(self, pairs: List[Tuple[str, str]]) -> None:
        """Check if φ-encoding reveals universal structure."""
        print("\n" + "="*70)
        print("φ-UNIVERSALITY CHECK")
        print("="*70)
        
        # For each pair, compute the RATIO of embeddings
        print("\nChecking if e2/e1 has universal structure...")
        
        ratios = []
        for w1, w2 in pairs:
            e1 = self.get_embedding(w1)
            e2 = self.get_embedding(w2)
            if e1 is None or e2 is None:
                continue
            
            # Ratio in original space (element-wise)
            # Avoid division by zero
            safe_e1 = e1.clone()
            safe_e1[safe_e1.abs() < 1e-10] = 1e-10
            ratio = e2 / safe_e1
            
            # In φ-space, ratio becomes level difference
            l1, s1 = encode_phi(e1)
            l2, s2 = encode_phi(e2)
            level_ratio = l2.float() - l1.float()  # This IS the log-ratio in φ-base
            
            ratios.append({
                "pair": (w1, w2),
                "level_ratio_mean": level_ratio.mean().item(),
                "level_ratio_std": level_ratio.std().item(),
            })
        
        print("\nLevel ratio (log_φ(e2/e1)) per pair:")
        for r in ratios:
            print(f"  {r['pair'][0]:8s}/{r['pair'][1]:8s}: mean={r['level_ratio_mean']:+.1f}, std={r['level_ratio_std']:.1f}")
        
        # Check if the DISTRIBUTION of level ratios is similar
        print("\nAre the level ratio distributions similar?")
        means = [r["level_ratio_mean"] for r in ratios]
        stds = [r["level_ratio_std"] for r in ratios]
        print(f"  Mean of means: {np.mean(means):.1f}")
        print(f"  Std of means: {np.std(means):.1f}")
        print(f"  Mean of stds: {np.mean(stds):.1f}")
        
        if np.std(means) < 10:
            print("  → Level ratio means are SIMILAR across pairs!")
        else:
            print("  → Level ratio means VARY across pairs")


def main():
    print("="*70)
    print("φ-LATTICE UNIVERSAL TRANSFORMATION CHECK")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    checker = UniversalChecker(model, tokenizer)
    
    # All opposite pairs
    all_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("up", "down"),
        ("good", "bad"),
        ("happy", "sad"),
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
    
    # Check structural similarity
    checker.check_structural_similarity(all_pairs)
    
    # Check φ-universality
    checker.check_phi_universality(all_pairs)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The question: Is there a UNIVERSAL transformation for opposites?

Design 137 says φ encodes VALUES universally.
But semantic RELATIONSHIPS might work differently.

Key insight: φ-encoding is universal for REPRESENTATION,
but the TRANSFORMATION between concepts is concept-specific.

This is like: 
  - GPS coordinates can represent ANY location (universal)
  - But the PATH from A to B depends on A and B (specific)

The φ-lattice is the coordinate system.
The relationships are the paths.
Each path is different, even in the same coordinate system.
""")


if __name__ == "__main__":
    main()
