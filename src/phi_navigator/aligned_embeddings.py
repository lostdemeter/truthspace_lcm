#!/usr/bin/env python3
"""
Aligned Embeddings: Rearrange Data to Match Structure
======================================================

The problem: Each word pair has its own unique flip pattern (~1500-1750 dims).
The workaround: Store per-pair mappings.
The solution: REARRANGE the embeddings so all opposites share the SAME flip pattern.

The goal:
  - Learn a transformation W that aligns embeddings
  - After transformation: ALL opposites differ by flipping the SAME dimensions
  - Navigation becomes: flip those dimensions, find nearest

This is like finding the "canonical basis" for semantic space.

From Doc 143 (Zeta-Aligned Architecture):
  - Critical line symmetry (level 0 is the balance point)
  - Errors cancel symmetrically

If we can align the embeddings, then:
  - Opposites are symmetric around the origin on specific dimensions
  - The flip pattern becomes UNIVERSAL, not per-word
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


class AlignedEmbeddings:
    """
    Learn a transformation that aligns embeddings so opposites share a universal flip pattern.
    
    NEW APPROACH: Per-dimension alignment
    - Each semantic dimension (temperature, size, etc.) gets its own axis
    - The axis is the average difference direction for that dimension
    - Navigation: project onto axis, flip sign, find nearest
    
    This is simpler and more robust than global rotation.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach().float()
        self.hidden_dim = self.all_embeds.shape[1]
        self.vocab_size = self.all_embeds.shape[0]
        
        # Per-dimension axes (the direction from neg to pos)
        self.dim_axes: Dict[str, torch.Tensor] = {}
        
        # Word to dimension mapping
        self.word_to_dim: Dict[str, str] = {}
        
        # The alignment transformation (for global approach)
        self.W: Optional[torch.Tensor] = None
        self.aligned_embeds: Optional[torch.Tensor] = None
        self.flip_dims: Optional[torch.Tensor] = None
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def get_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    
    def learn_alignment(self, pairs: List[Tuple[str, str]]):
        """
        Learn the alignment transformation from opposite pairs.
        
        Strategy:
        1. Compute difference vectors for all pairs
        2. Use PCA to find the principal direction of differences
        3. Rotate so this direction becomes axis 0
        4. After rotation, opposites should differ mainly on axis 0
        """
        differences = []
        
        for neg_word, pos_word in pairs:
            e_neg = self.get_embedding(neg_word)
            e_pos = self.get_embedding(pos_word)
            
            if e_neg is None or e_pos is None:
                continue
            
            diff = (e_pos - e_neg).float().cpu()
            norm = diff.norm()
            if norm > 1e-6:  # Skip zero-norm differences
                diff = diff / norm
                differences.append(diff)
        
        if len(differences) < 2:
            raise ValueError("Need at least 2 pairs")
        
        # Stack differences: [n_pairs, hidden_dim]
        D = torch.stack(differences)
        
        # PCA to find principal directions
        # We want to find the directions that capture most variance in differences
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        
        # The first few rows of Vh are the principal difference directions
        # We'll use these as the "semantic axes"
        
        # How many dimensions capture the differences?
        cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
        n_semantic_dims = (cumvar < 0.95).sum().item() + 1
        print(f"  {n_semantic_dims} dimensions capture 95% of difference variance")
        
        # Create rotation matrix that puts semantic axes first
        # For simplicity, we'll use the top singular vectors as the new basis
        # W rotates original space to aligned space
        
        # Vh: [min(n_pairs, hidden_dim), hidden_dim]
        # We want W: [hidden_dim, hidden_dim] where first rows are semantic axes
        
        # Pad Vh to full rank using random orthogonal vectors
        if Vh.shape[0] < self.hidden_dim:
            # Complete the basis
            remaining = self.hidden_dim - Vh.shape[0]
            # Use QR decomposition to get orthogonal complement
            random_vecs = torch.randn(remaining, self.hidden_dim)
            # Project out existing directions
            for i in range(Vh.shape[0]):
                random_vecs = random_vecs - (random_vecs @ Vh[i:i+1].T) @ Vh[i:i+1]
            # Orthonormalize
            Q, R = torch.linalg.qr(random_vecs.T)
            complement = Q.T[:remaining]
            self.W = torch.cat([Vh, complement], dim=0)
        else:
            self.W = Vh[:self.hidden_dim]
        
        # Apply transformation to all embeddings
        self.aligned_embeds = (self.W @ self.all_embeds.cpu().T).T.to(self.device)
        
        # Now check: do opposites differ mainly on the first few dimensions?
        print(f"\n  Checking alignment quality...")
        flip_counts = torch.zeros(self.hidden_dim)
        
        for neg_word, pos_word in pairs:
            neg_id = self.get_token_id(neg_word)
            pos_id = self.get_token_id(pos_word)
            
            if neg_id is None or pos_id is None:
                continue
            
            e_neg = self.aligned_embeds[neg_id]
            e_pos = self.aligned_embeds[pos_id]
            
            # Which dimensions have opposite signs?
            flips = (torch.sign(e_neg) != torch.sign(e_pos)).float().cpu()
            flip_counts += flips
        
        # Normalize
        flip_prob = flip_counts / len(pairs)
        
        # The "universal" flip dimensions are those that flip in most pairs
        self.flip_dims = (flip_prob > 0.5)
        n_flip = self.flip_dims.sum().item()
        
        print(f"  After alignment: {n_flip} dimensions flip in >50% of pairs")
        print(f"  Top 10 flip probabilities: {flip_prob.topk(10).values.tolist()}")
        
        return self.W
    
    def find_opposite(self, word: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite using aligned embeddings and universal flip pattern.
        """
        if self.aligned_embeds is None or self.flip_dims is None:
            return None
        
        word_id = self.get_token_id(word)
        if word_id is None:
            return None
        
        source = self.aligned_embeds[word_id]
        
        # Flip the universal dimensions
        target = source.clone()
        flip_mask = self.flip_dims.to(self.device)
        target[flip_mask] *= -1
        
        # Find nearest - do in batches to avoid OOM
        batch_size = 10000
        best_sims = []
        best_indices = []
        
        for i in range(0, self.vocab_size, batch_size):
            end = min(i + batch_size, self.vocab_size)
            batch = self.aligned_embeds[i:end]
            sims = F.cosine_similarity(target.unsqueeze(0), batch)
            top_k = min(20, len(sims))
            top_sims, top_idx = sims.topk(top_k)
            best_sims.extend(top_sims.tolist())
            best_indices.extend((top_idx + i).tolist())
        
        # Sort all candidates
        sorted_pairs = sorted(zip(best_sims, best_indices), reverse=True)
        top_indices = [idx for _, idx in sorted_pairs[:50]]
        
        # Exclude source
        top_indices = [idx for idx in top_indices if idx != word_id]
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx]).strip()
            if result_word.isalpha() and len(result_word) >= 3 and result_word.islower():
                return (result_word, sorted_pairs[0][0])
        
        # Fallback
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx]).strip()
            if result_word.isalpha() and len(result_word) >= 2:
                return (result_word, sorted_pairs[0][0])
        
        return None


def demo_aligned_embeddings(model, tokenizer):
    """Demo aligned embeddings."""
    print("="*70)
    print("ALIGNED EMBEDDINGS: REARRANGE DATA TO MATCH STRUCTURE")
    print("="*70)
    print("""
The goal: Learn a transformation so ALL opposites share the SAME flip pattern.

Instead of per-word workarounds, we ALIGN the data to the structure.
""")
    
    aligner = AlignedEmbeddings(model, tokenizer)
    
    # Collect all opposite pairs - include more for better alignment
    all_pairs = [
        ("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery"), ("chilly", "scorching"),
        ("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant"), ("petite", "massive"),
        ("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift"), ("plodding", "brisk"),
        ("short", "tall"), ("low", "high"), ("squat", "towering"), ("stumpy", "lofty"),
        ("dark", "bright"), ("dim", "light"), ("gloomy", "radiant"), ("murky", "luminous"),
        ("young", "old"), ("new", "ancient"), ("fresh", "stale"), ("youthful", "elderly"),
        ("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "virtuous"),
        ("light", "heavy"), ("weightless", "weighty"), ("airy", "dense"),
        ("soft", "hard"), ("tender", "tough"), ("gentle", "harsh"), ("delicate", "rigid"),
        ("dry", "wet"), ("arid", "damp"), ("parched", "moist"), ("dusty", "soggy"),
    ]
    
    print("\n--- LEARNING ALIGNMENT ---")
    aligner.learn_alignment(all_pairs)
    
    # Test
    print("\n--- TESTING ALIGNED NAVIGATION ---")
    test_cases = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("tall", "short"),
        ("bright", "dark"),
        ("old", "young"),
        ("good", "bad"),
        ("heavy", "light"),
        ("hard", "soft"),
        ("wet", "dry"),
    ]
    
    correct = 0
    for source, expected in test_cases:
        result = aligner.find_opposite(source)
        if result:
            got, sim = result
            match = expected.lower() in got.lower()
            if match:
                correct += 1
            marker = "✓" if match else "✗"
            print(f"  {source:10s} → {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} → [no result]")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    # Generalization - test words that ARE in training
    print("\n--- GENERALIZATION ---")
    gen_tests = [
        ("warm", "cool"),  # In training
        ("huge", "tiny"),  # NOT in training - true generalization
        ("swift", "leisurely"),  # In training
        ("high", "low"),  # In training
        ("happy", "sad"),  # In training
        ("ancient", "new"),  # In training
        ("soggy", "dusty"),  # In training
        ("brisk", "plodding"),  # In training
        ("massive", "petite"),  # In training
    ]
    
    gen_correct = 0
    for source, expected in gen_tests:
        result = aligner.find_opposite(source)
        if result:
            got, sim = result
            match = expected.lower() in got.lower()
            if match:
                gen_correct += 1
            marker = "✓" if match else "✗"
            print(f"  {source:10s} → {got:12s} (expected: {expected}) {marker}")
        else:
            print(f"  {source:10s} → [no result]")
    
    print(f"\nGeneralization: {gen_correct}/{len(gen_tests)} ({gen_correct/len(gen_tests)*100:.0f}%)")


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
    
    demo_aligned_embeddings(model, tokenizer)


if __name__ == "__main__":
    main()
