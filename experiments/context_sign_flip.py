#!/usr/bin/env python3
"""
Context as Sign Flips in Lattice Space
=======================================

Key insight from Doc 141: The irreducible shape is the SIGN PATTERN.
- 3584 critical lines divide semantic space
- Each hidden state is defined by which side of each line it's on
- The signs encode the region of the lattice

Hypothesis: Context FLIPS SIGNS in the lattice.

When we add prefix A to suffix B:
- Some dimensions flip sign (cross a critical line)
- Other dimensions stay the same

If the sign flips are CONSISTENT for a given prefix,
we can precompute them!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ContextSignFlipAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def get_final_hidden(self, token_ids: List[int]) -> np.ndarray:
        """Get final hidden state for token sequence."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def analyze_sign_flips(self, n_samples: int = 100):
        """
        Analyze how many dimensions flip sign when adding context.
        """
        print(f"\n--- Sign Flip Analysis ({n_samples} pairs) ---")
        
        flip_counts = []
        flip_fractions = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                # Count sign flips
                signs_B = np.sign(h_B)
                signs_AB = np.sign(h_AB)
                
                flips = (signs_B != signs_AB).sum()
                flip_counts.append(flips)
                flip_fractions.append(flips / self.hidden_dim)
            except:
                continue
        
        print(f"\n  Sign flip statistics:")
        print(f"    Mean flips: {np.mean(flip_counts):.0f} / {self.hidden_dim} ({np.mean(flip_fractions)*100:.1f}%)")
        print(f"    Std flips: {np.std(flip_counts):.0f}")
        print(f"    Min flips: {np.min(flip_counts)}")
        print(f"    Max flips: {np.max(flip_counts)}")
        
        return flip_counts, flip_fractions
    
    def analyze_prefix_specific_flips(self, n_prefixes: int = 10, n_suffixes: int = 30):
        """
        For a FIXED prefix, which dimensions consistently flip?
        
        If certain dimensions ALWAYS flip for a given prefix,
        we can precompute the flip pattern!
        """
        print(f"\n--- Prefix-Specific Sign Flip Analysis ---")
        
        results = []
        
        for p in range(n_prefixes):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            A_text = self.tokenizer.decode([A])
            
            flip_patterns = []
            
            for s in range(n_suffixes):
                B = np.random.randint(0, self.tokenizer.vocab_size)
                
                try:
                    h_B = self.get_final_hidden([B])
                    h_AB = self.get_final_hidden([A, B])
                    
                    signs_B = np.sign(h_B)
                    signs_AB = np.sign(h_AB)
                    
                    flips = (signs_B != signs_AB).astype(int)
                    flip_patterns.append(flips)
                except:
                    continue
            
            if len(flip_patterns) < 5:
                continue
            
            flip_patterns = np.array(flip_patterns)
            
            # For each dimension, what fraction of suffixes have a flip?
            flip_rates = flip_patterns.mean(axis=0)
            
            # Dimensions that ALWAYS flip (rate > 0.9)
            always_flip = (flip_rates > 0.9).sum()
            
            # Dimensions that NEVER flip (rate < 0.1)
            never_flip = (flip_rates < 0.1).sum()
            
            # Dimensions that are inconsistent (0.1 < rate < 0.9)
            inconsistent = self.hidden_dim - always_flip - never_flip
            
            results.append({
                'prefix': A_text[:15],
                'always_flip': always_flip,
                'never_flip': never_flip,
                'inconsistent': inconsistent,
            })
            
            print(f"  Prefix '{A_text[:10]}': always_flip={always_flip}, never_flip={never_flip}, inconsistent={inconsistent}")
        
        # Summary
        mean_always = np.mean([r['always_flip'] for r in results])
        mean_never = np.mean([r['never_flip'] for r in results])
        mean_inconsistent = np.mean([r['inconsistent'] for r in results])
        
        print(f"\n  Summary:")
        print(f"    Mean always flip: {mean_always:.0f}")
        print(f"    Mean never flip: {mean_never:.0f}")
        print(f"    Mean inconsistent: {mean_inconsistent:.0f}")
        
        return results
    
    def test_sign_based_reconstruction(self, n_samples: int = 100):
        """
        Test: Can we reconstruct h(A,B) by flipping signs in h(B)?
        
        h(A,B) ≈ h(B) * flip_pattern(A)
        
        where flip_pattern(A) is +1 or -1 for each dimension.
        """
        print(f"\n--- Sign-Based Reconstruction Test ---")
        
        # First, learn the "average" flip pattern
        # (This is a simplification - in practice we'd learn per-prefix)
        
        flip_patterns = []
        h_B_list = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                # The "flip" is the sign of (h_AB / h_B)
                # But we need to handle zeros
                ratio = h_AB / (h_B + 1e-10 * np.sign(h_B + 1e-20))
                flip = np.sign(ratio)
                
                flip_patterns.append(flip)
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        flip_patterns = np.array(flip_patterns)
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        
        # Average flip pattern
        mean_flip = np.sign(flip_patterns.mean(axis=0))
        
        # Test reconstruction
        correct = 0
        for i in range(len(h_B)):
            # Reconstruct by flipping signs
            reconstructed = h_B[i] * mean_flip
            
            # Also need to adjust magnitudes
            # Let's try: keep magnitude of h_B, just flip signs
            
            true_logits = np.dot(self.lm_head, h_AB[i])
            recon_logits = np.dot(self.lm_head, reconstructed)
            
            if np.argmax(true_logits) == np.argmax(recon_logits):
                correct += 1
        
        accuracy = correct / len(h_B)
        print(f"\n  Sign-flip reconstruction accuracy: {accuracy*100:.1f}%")
        
        return accuracy
    
    def analyze_magnitude_change(self, n_samples: int = 100):
        """
        Separate analysis: How do MAGNITUDES change with context?
        
        From Doc 141: magnitudes are on the φ-lattice.
        Maybe the transformation is simpler in magnitude space?
        """
        print(f"\n--- Magnitude Change Analysis ---")
        
        mag_ratios = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                # Per-dimension magnitude ratio
                ratio = np.abs(h_AB) / (np.abs(h_B) + 1e-10)
                mag_ratios.append(ratio)
            except:
                continue
        
        mag_ratios = np.array(mag_ratios)
        
        # Statistics
        mean_ratio = mag_ratios.mean(axis=0)
        std_ratio = mag_ratios.std(axis=0)
        
        print(f"\n  Per-dimension magnitude ratio:")
        print(f"    Mean of means: {mean_ratio.mean():.3f}")
        print(f"    Std of means: {mean_ratio.std():.3f}")
        print(f"    Mean of stds: {std_ratio.mean():.3f}")
        
        # Is the ratio consistent across dimensions?
        # If so, it's just a global scaling
        print(f"\n  Ratio consistency:")
        print(f"    Min mean ratio: {mean_ratio.min():.3f}")
        print(f"    Max mean ratio: {mean_ratio.max():.3f}")
        
        # Test: What if we use a SINGLE scale factor?
        global_scale = mag_ratios.mean()
        print(f"\n  Global scale factor: {global_scale:.3f}")


def main():
    print("=" * 70)
    print("CONTEXT AS SIGN FLIPS IN LATTICE SPACE")
    print("=" * 70)
    print("""
From Doc 141: The irreducible shape is the SIGN PATTERN.

Hypothesis: Context changes which side of critical lines we're on.
This manifests as SIGN FLIPS in certain dimensions.

If the flip pattern is consistent for a given prefix,
we can precompute it!
""")
    
    analyzer = ContextSignFlipAnalyzer()
    
    # 1. Basic sign flip analysis
    flip_counts, flip_fractions = analyzer.analyze_sign_flips(n_samples=100)
    
    # 2. Prefix-specific flip patterns
    prefix_results = analyzer.analyze_prefix_specific_flips(n_prefixes=10, n_suffixes=30)
    
    # 3. Test sign-based reconstruction
    accuracy = analyzer.test_sign_based_reconstruction(n_samples=100)
    
    # 4. Magnitude change analysis
    analyzer.analyze_magnitude_change(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
