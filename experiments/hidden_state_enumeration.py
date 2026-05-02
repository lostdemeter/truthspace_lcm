#!/usr/bin/env python3
"""
Hidden State Enumeration: Can We Exhaustively Scan?
====================================================

Key insights:
1. Tetromino: Weights have only ~300 unique (level, sign) patterns
2. Vocabulary: 152K tokens
3. Hidden states = f(tokens, weights)

If weights are constrained, hidden states should be too.

Question: How many UNIQUE hidden states exist in practice?
Can we enumerate them all?

The 12D clock from ribbon_attention.py provides a systematic
way to probe the space.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import hashlib
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def hidden_to_signature(hidden: np.ndarray, k: int = 32) -> str:
    """
    Convert hidden state to a discrete signature.
    
    Uses φ-lattice quantization to create a hashable signature.
    """
    # Quantize to φ-levels
    signs = np.sign(hidden)
    magnitudes = np.abs(hidden) + 1e-10
    levels = np.round(k * np.log(magnitudes) / LOG_PHI).astype(int)
    
    # Combine into signature
    # Use coarse binning to group similar states
    coarse_levels = levels // 4  # Bin into groups of 4 levels
    
    # Hash the (sign, coarse_level) pairs
    sig_data = np.stack([signs, coarse_levels], axis=1).tobytes()
    return hashlib.md5(sig_data).hexdigest()[:16]


class HiddenStateEnumerator:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _get_hidden(self, token_ids: List[int]) -> np.ndarray:
        """Get hidden state for token sequence."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def enumerate_single_tokens(self, n_tokens: int = 1000) -> Dict:
        """
        Enumerate hidden states for single tokens.
        
        This is the base case - how many unique hidden states
        do we get from single tokens?
        """
        print(f"\n--- Enumerating {n_tokens} single tokens ---")
        
        signatures = set()
        sig_to_tokens = defaultdict(list)
        
        # Sample tokens (or enumerate all if small enough)
        if n_tokens >= self.vocab_size:
            token_ids = list(range(self.vocab_size))
        else:
            token_ids = np.random.choice(self.vocab_size, n_tokens, replace=False)
        
        for i, token_id in enumerate(token_ids):
            if i % 100 == 0:
                print(f"  Processing token {i}/{len(token_ids)}...")
            
            try:
                hidden = self._get_hidden([token_id])
                sig = hidden_to_signature(hidden)
                signatures.add(sig)
                sig_to_tokens[sig].append(token_id)
            except Exception as e:
                continue
        
        # Analyze
        n_unique = len(signatures)
        compression = len(token_ids) / n_unique if n_unique > 0 else 0
        
        print(f"\n  Tokens processed: {len(token_ids)}")
        print(f"  Unique signatures: {n_unique}")
        print(f"  Compression ratio: {compression:.1f}x")
        
        # Show some signature clusters
        print(f"\n  Largest signature clusters:")
        sorted_sigs = sorted(sig_to_tokens.items(), key=lambda x: -len(x[1]))[:5]
        for sig, tokens in sorted_sigs:
            token_strs = [self.tokenizer.decode([t]) for t in tokens[:5]]
            print(f"    {sig}: {len(tokens)} tokens - {token_strs}")
        
        return {
            'n_tokens': len(token_ids),
            'n_unique': n_unique,
            'compression': compression,
            'sig_to_tokens': dict(sig_to_tokens),
        }
    
    def enumerate_token_pairs(self, n_pairs: int = 1000) -> Dict:
        """
        Enumerate hidden states for token pairs.
        
        Does the number of unique states grow linearly or sublinearly?
        """
        print(f"\n--- Enumerating {n_pairs} token pairs ---")
        
        signatures = set()
        
        for i in range(n_pairs):
            if i % 100 == 0:
                print(f"  Processing pair {i}/{n_pairs}...")
            
            # Random token pair
            t1 = np.random.randint(0, self.vocab_size)
            t2 = np.random.randint(0, self.vocab_size)
            
            try:
                hidden = self._get_hidden([t1, t2])
                sig = hidden_to_signature(hidden)
                signatures.add(sig)
            except Exception as e:
                continue
        
        n_unique = len(signatures)
        compression = n_pairs / n_unique if n_unique > 0 else 0
        
        print(f"\n  Pairs processed: {n_pairs}")
        print(f"  Unique signatures: {n_unique}")
        print(f"  Compression ratio: {compression:.1f}x")
        
        return {
            'n_pairs': n_pairs,
            'n_unique': n_unique,
            'compression': compression,
        }
    
    def analyze_hidden_state_structure(self, n_samples: int = 500) -> Dict:
        """
        Analyze the structure of hidden states.
        
        Questions:
        1. How many unique φ-levels are used?
        2. What's the distribution of signs?
        3. Are there "tetromino-like" patterns?
        """
        print(f"\n--- Analyzing hidden state structure ({n_samples} samples) ---")
        
        all_levels = []
        all_signs = []
        level_counts = defaultdict(int)
        sign_patterns = defaultdict(int)
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  Processing sample {i}/{n_samples}...")
            
            # Random token sequence (1-5 tokens)
            seq_len = np.random.randint(1, 6)
            tokens = np.random.randint(0, self.vocab_size, seq_len).tolist()
            
            try:
                hidden = self._get_hidden(tokens)
                
                # Convert to φ-levels
                signs = np.sign(hidden)
                magnitudes = np.abs(hidden) + 1e-10
                levels = np.round(32 * np.log(magnitudes) / LOG_PHI).astype(int)
                
                all_levels.extend(levels)
                all_signs.extend(signs)
                
                # Count level occurrences
                for level in levels:
                    level_counts[level] += 1
                
                # Analyze sign patterns in blocks of 4 (like tetromino)
                for j in range(0, len(signs) - 3, 4):
                    pattern = tuple(signs[j:j+4].astype(int))
                    sign_patterns[pattern] += 1
                    
            except Exception as e:
                continue
        
        # Analyze
        all_levels = np.array(all_levels)
        all_signs = np.array(all_signs)
        
        unique_levels = len(level_counts)
        
        # Top levels
        top_levels = sorted(level_counts.items(), key=lambda x: -x[1])[:10]
        
        # Sign pattern distribution
        n_sign_patterns = len(sign_patterns)
        top_patterns = sorted(sign_patterns.items(), key=lambda x: -x[1])[:10]
        
        print(f"\n  Unique φ-levels: {unique_levels}")
        print(f"  Level range: [{all_levels.min()}, {all_levels.max()}]")
        print(f"  Level std: {all_levels.std():.1f}")
        
        print(f"\n  Top 10 levels:")
        for level, count in top_levels:
            pct = count / len(all_levels) * 100
            print(f"    Level {level}: {pct:.2f}%")
        
        print(f"\n  Unique sign patterns (4D blocks): {n_sign_patterns}")
        print(f"  Top 10 patterns:")
        total_patterns = sum(sign_patterns.values())
        for pattern, count in top_patterns:
            pct = count / total_patterns * 100
            print(f"    {pattern}: {pct:.2f}%")
        
        return {
            'unique_levels': unique_levels,
            'level_range': (all_levels.min(), all_levels.max()),
            'n_sign_patterns': n_sign_patterns,
            'level_counts': dict(level_counts),
            'sign_patterns': dict(sign_patterns),
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE ENUMERATION")
    print("=" * 70)
    print("""
Goal: Determine if hidden states can be exhaustively enumerated.

If weights are constrained (Tetromino: ~300 patterns),
and tokens are finite (152K),
then hidden states should also be finite.

Can we precompute ALL of them?
""")
    
    enumerator = HiddenStateEnumerator()
    
    # 1. Single token enumeration
    single_results = enumerator.enumerate_single_tokens(n_tokens=500)
    
    # 2. Token pair enumeration
    pair_results = enumerator.enumerate_token_pairs(n_pairs=500)
    
    # 3. Structure analysis
    structure_results = enumerator.analyze_hidden_state_structure(n_samples=300)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Single tokens:
  - {single_results['n_tokens']} tokens → {single_results['n_unique']} unique signatures
  - Compression: {single_results['compression']:.1f}x

Token pairs:
  - {pair_results['n_pairs']} pairs → {pair_results['n_unique']} unique signatures
  - Compression: {pair_results['compression']:.1f}x

Structure:
  - {structure_results['unique_levels']} unique φ-levels
  - {structure_results['n_sign_patterns']} unique sign patterns (4D blocks)

IMPLICATION:
""")
    
    # Calculate theoretical bounds
    if single_results['compression'] > 1:
        print(f"  Hidden states ARE clustered! Not every token has unique state.")
        print(f"  Estimated unique states for full vocab: ~{int(single_results['n_tokens'] / single_results['compression'] * 152064 / single_results['n_tokens'])}")
    else:
        print(f"  Each token has unique hidden state - enumeration may be large.")


if __name__ == "__main__":
    main()
