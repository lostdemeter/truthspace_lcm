#!/usr/bin/env python3
"""
Sign Self-Assembly: Discover Structure in Sign Patterns
========================================================

From Doc 122: "Self-assembly discovers structure that EXISTS."

Instead of statistical filtering, we:
1. Build similarity matrix from sign patterns
2. Eigendecompose to find natural dimensions
3. Project words into low-dimensional space
4. Navigate geometrically in emergent space

The structure IS in the signs - we just need to self-assemble it.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import time

PHI = (1 + math.sqrt(5)) / 2


class SignSelfAssembler:
    """
    Self-assemble structure from sign patterns.
    
    From Doc 122: "Weights → Self-assemble structure → Understand geometry directly"
    """
    
    def __init__(self, model, tokenizer, n_components: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.n_components = n_components
        
        # Extract embeddings
        embeds = model.model.embed_tokens.weight.detach().float()
        self.hidden_dim = embeds.shape[1]
        self.vocab_size = embeds.shape[0]
        
        # Signs
        self.all_signs = torch.sign(embeds).to(torch.int8)
        self.all_signs[self.all_signs == 0] = 1
        
        # Self-assembled structure
        self.positions: Optional[torch.Tensor] = None  # [vocab_size, n_components]
        self.eigenvalues: Optional[torch.Tensor] = None
        self.eigenvectors: Optional[torch.Tensor] = None
        
        # Word cache for fast lookup
        self.word_to_id: Dict[str, int] = {}
    
    def get_token_id(self, word: str) -> Optional[int]:
        if word in self.word_to_id:
            return self.word_to_id[word]
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if ids:
            self.word_to_id[word] = ids[0]
            return ids[0]
        return None
    
    def self_assemble(self, sample_size: int = 10000):
        """
        Self-assemble structure from sign patterns.
        
        1. Sample words to build similarity matrix
        2. Eigendecompose to find natural dimensions
        3. Project all words into this space
        """
        print(f"\n--- SELF-ASSEMBLY ON SIGN PATTERNS ---")
        start_time = time.perf_counter()
        
        # Sample indices for similarity matrix (full vocab too large)
        torch.manual_seed(42)
        sample_indices = torch.randperm(self.vocab_size)[:sample_size]
        sample_signs = self.all_signs[sample_indices].float().to(self.device)
        
        print(f"  Building similarity matrix ({sample_size}x{sample_size})...")
        
        # Sign agreement as similarity: S[i,j] = (signs_i · signs_j) / hidden_dim
        # This is correlation of sign patterns
        S = (sample_signs @ sample_signs.T) / self.hidden_dim
        
        # Eigendecompose
        print(f"  Eigendecomposing...")
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top components
        self.eigenvalues = eigenvalues[:self.n_components].cpu()
        self.eigenvectors = eigenvectors[:, :self.n_components].cpu()
        
        # Variance explained
        total_var = eigenvalues.sum().item()
        top_var = eigenvalues[:self.n_components].sum().item()
        print(f"  Top {self.n_components} components: {top_var/total_var*100:.1f}% variance")
        print(f"  Top 3 eigenvalues: {eigenvalues[:3].tolist()}")
        
        # Project sample into eigenspace
        # Position = V @ sqrt(Λ) (from Doc 122)
        sqrt_eigenvalues = torch.sqrt(torch.clamp(self.eigenvalues, min=0))
        sample_positions = self.eigenvectors * sqrt_eigenvalues.unsqueeze(0)
        
        # Now project ALL words using the learned basis
        # For new words: position = (signs · sample_signs.T) @ eigenvectors @ sqrt(Λ)
        print(f"  Projecting all {self.vocab_size} words...")
        
        # Do in batches to avoid OOM
        batch_size = 10000
        all_positions = []
        
        for i in range(0, self.vocab_size, batch_size):
            end = min(i + batch_size, self.vocab_size)
            batch_signs = self.all_signs[i:end].float().to(self.device)
            
            # Similarity to sample
            batch_sim = (batch_signs @ sample_signs.T) / self.hidden_dim
            
            # Project
            batch_pos = batch_sim @ self.eigenvectors.to(self.device) * sqrt_eigenvalues.to(self.device).unsqueeze(0)
            all_positions.append(batch_pos.cpu())
        
        self.positions = torch.cat(all_positions, dim=0)
        
        elapsed = time.perf_counter() - start_time
        print(f"  Self-assembly complete in {elapsed:.1f}s")
        print(f"  Position shape: {self.positions.shape}")
        
        return self.positions
    
    def find_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar words by distance in self-assembled space.
        
        This is GEOMETRIC navigation, not statistical filtering.
        """
        if self.positions is None:
            self.self_assemble()
        
        word_id = self.get_token_id(word)
        if word_id is None:
            return []
        
        word_pos = self.positions[word_id]
        
        # Euclidean distance in assembled space
        distances = torch.norm(self.positions - word_pos.unsqueeze(0), dim=1)
        distances[word_id] = float('inf')  # Exclude self
        
        # Get closest
        top_indices = distances.argsort()[:top_k * 10]
        
        results = []
        seen = {word.lower()}
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            result_lower = result_word.lower()
            
            if (result_word.isalpha() and 
                len(result_word) >= 3 and 
                result_word.islower() and
                result_lower not in seen):
                
                seen.add(result_lower)
                results.append((result_word, distances[idx].item()))
                if len(results) >= top_k:
                    break
        
        return results
    
    def find_opposite(self, word: str) -> Optional[Tuple[str, float]]:
        """
        Find opposite by navigating to antipodal position.
        
        In self-assembled space, opposites should be far apart.
        """
        if self.positions is None:
            self.self_assemble()
        
        word_id = self.get_token_id(word)
        if word_id is None:
            return None
        
        word_pos = self.positions[word_id]
        
        # Antipodal: negate position (flip through origin)
        target_pos = -word_pos
        
        # Find closest to antipodal
        distances = torch.norm(self.positions - target_pos.unsqueeze(0), dim=1)
        distances[word_id] = float('inf')
        
        top_indices = distances.argsort()[:50]
        
        for idx in top_indices:
            result_word = self.tokenizer.decode([idx.item()]).strip()
            if result_word.isalpha() and len(result_word) >= 3 and result_word.islower():
                return (result_word, distances[idx].item())
        
        return None
    
    def analyze_structure(self):
        """Analyze the self-assembled structure."""
        if self.positions is None:
            self.self_assemble()
        
        print(f"\n--- STRUCTURE ANALYSIS ---")
        
        # Eigenvalue spectrum
        print(f"\n  Eigenvalue spectrum:")
        for i in range(min(10, len(self.eigenvalues))):
            var_pct = self.eigenvalues[i].item() / self.eigenvalues.sum().item() * 100
            print(f"    λ_{i}: {self.eigenvalues[i].item():.4f} ({var_pct:.1f}%)")
        
        # Check for φ-scaling
        print(f"\n  φ-scaling analysis:")
        ratios = []
        for i in range(min(9, len(self.eigenvalues) - 1)):
            ratio = self.eigenvalues[i].item() / self.eigenvalues[i+1].item()
            ratios.append(ratio)
            phi_diff = abs(ratio - PHI)
            marker = "≈φ" if phi_diff < 0.2 else ""
            print(f"    λ_{i}/λ_{i+1} = {ratio:.4f} {marker}")
        
        # Position statistics
        print(f"\n  Position statistics:")
        print(f"    Mean: {self.positions.mean().item():.4f}")
        print(f"    Std:  {self.positions.std().item():.4f}")
        print(f"    Min:  {self.positions.min().item():.4f}")
        print(f"    Max:  {self.positions.max().item():.4f}")


def demo_self_assembly(model, tokenizer):
    """Demo self-assembly on sign patterns."""
    print("="*70)
    print("SIGN SELF-ASSEMBLY: DISCOVER STRUCTURE")
    print("="*70)
    print("""
From Doc 122: "Self-assembly discovers structure that EXISTS."

Instead of statistical filtering, we:
  1. Build similarity matrix from sign patterns
  2. Eigendecompose to find natural dimensions
  3. Navigate geometrically in emergent space
""")
    
    assembler = SignSelfAssembler(model, tokenizer, n_components=64)
    assembler.self_assemble(sample_size=10000)
    assembler.analyze_structure()
    
    # Test similar words
    print(f"\n--- SIMILAR WORDS (geometric navigation) ---")
    test_words = ["happiness", "love", "hot", "big", "fast"]
    for word in test_words:
        results = assembler.find_similar(word, top_k=5)
        if results:
            words = [f"{w} ({d:.3f})" for w, d in results]
            print(f"  {word}: {', '.join(words)}")
    
    # Test opposites
    print(f"\n--- OPPOSITES (antipodal navigation) ---")
    test_pairs = ["hot", "cold", "big", "small", "happy", "sad"]
    for word in test_pairs:
        result = assembler.find_opposite(word)
        if result:
            opp, dist = result
            print(f"  {word} → {opp} (dist={dist:.3f})")
    
    return assembler


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
    
    demo_self_assembly(model, tokenizer)


if __name__ == "__main__":
    main()
