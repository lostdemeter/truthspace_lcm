#!/usr/bin/env python3
"""
Context Geometry: How does the shape change with context?
==========================================================

Key insight from Doc 141: Shape = lattice of 3584 critical lines.
Each hidden state is a POINT in this space.

Question: When we add a token, how does the point MOVE?

Hypotheses:
1. TRANSLATION: h(A,B) = h(B) + offset(A)
2. ROTATION: h(A,B) = R(A) @ h(B)
3. SCALING: h(A,B) = s(A) * h(B)
4. COMBINATION: h(A,B) = R(A) @ h(B) + t(A)

If the transformation is simple, we can precompute it!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ContextGeometryAnalyzer:
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
    
    def analyze_context_transformation(self, n_samples: int = 100):
        """
        Analyze how adding a prefix token transforms the hidden state.
        
        Compare:
        - h(B): hidden state of token B alone
        - h(A,B): hidden state of B with prefix A
        
        What is the transformation from h(B) to h(A,B)?
        """
        print(f"\n--- Analyzing context transformation ({n_samples} pairs) ---")
        
        # Collect pairs
        h_B_list = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            # Random tokens A and B
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        
        # Test different transformation hypotheses
        print(f"\n  Testing transformation hypotheses:")
        
        # 1. TRANSLATION: h(A,B) = h(B) + offset
        # If true, h(A,B) - h(B) should be similar across samples
        deltas = h_AB - h_B
        delta_mean = deltas.mean(axis=0)
        delta_std = deltas.std(axis=0).mean()
        
        # How well does mean delta explain the transformation?
        predicted_AB = h_B + delta_mean
        translation_error = np.linalg.norm(h_AB - predicted_AB, axis=1).mean()
        baseline_error = np.linalg.norm(h_AB - h_B, axis=1).mean()
        
        print(f"\n  1. TRANSLATION hypothesis:")
        print(f"     Delta std: {delta_std:.2f}")
        print(f"     Baseline error (h_B → h_AB): {baseline_error:.2f}")
        print(f"     Translation error: {translation_error:.2f}")
        print(f"     Improvement: {(1 - translation_error/baseline_error)*100:.1f}%")
        
        # 2. SCALING: h(A,B) = s * h(B)
        # Compute optimal scale per sample
        scales = []
        for i in range(len(h_B)):
            s = np.dot(h_AB[i], h_B[i]) / (np.dot(h_B[i], h_B[i]) + 1e-10)
            scales.append(s)
        
        mean_scale = np.mean(scales)
        predicted_AB_scale = h_B * mean_scale
        scale_error = np.linalg.norm(h_AB - predicted_AB_scale, axis=1).mean()
        
        print(f"\n  2. SCALING hypothesis:")
        print(f"     Mean scale: {mean_scale:.3f}")
        print(f"     Scale error: {scale_error:.2f}")
        print(f"     Improvement: {(1 - scale_error/baseline_error)*100:.1f}%")
        
        # 3. LINEAR: h(A,B) = W @ h(B) + b
        # Solve least squares
        # Add bias term
        h_B_bias = np.hstack([h_B, np.ones((len(h_B), 1))])
        
        # Solve for W (using pseudo-inverse, but only for small subset)
        # This is expensive, so we'll use a low-rank approximation
        print(f"\n  3. LINEAR hypothesis (low-rank approximation):")
        
        # Use SVD to find the best low-rank linear transformation
        # h_AB ≈ h_B @ V @ V.T + mean_delta
        
        centered_AB = h_AB - h_AB.mean(axis=0)
        centered_B = h_B - h_B.mean(axis=0)
        
        # Cross-covariance
        C = centered_B.T @ centered_AB / len(h_B)
        
        # SVD of cross-covariance
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        
        print(f"     Top 10 singular values: {S[:10].round(2)}")
        
        # Test different ranks
        for k in [1, 5, 10, 20, 50]:
            # Low-rank transformation
            W_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            predicted_AB_linear = centered_B @ W_k + h_AB.mean(axis=0)
            linear_error = np.linalg.norm(h_AB - predicted_AB_linear, axis=1).mean()
            
            print(f"     Rank-{k} error: {linear_error:.2f} ({(1 - linear_error/baseline_error)*100:.1f}% improvement)")
        
        # 4. COSINE SIMILARITY: Are h(B) and h(A,B) pointing in similar directions?
        cos_sims = []
        for i in range(len(h_B)):
            cos = np.dot(h_B[i], h_AB[i]) / (np.linalg.norm(h_B[i]) * np.linalg.norm(h_AB[i]) + 1e-10)
            cos_sims.append(cos)
        
        print(f"\n  4. DIRECTION analysis:")
        print(f"     Mean cosine similarity: {np.mean(cos_sims):.4f}")
        print(f"     Min cosine similarity: {np.min(cos_sims):.4f}")
        print(f"     Max cosine similarity: {np.max(cos_sims):.4f}")
        
        return {
            'h_B': h_B,
            'h_AB': h_AB,
            'deltas': deltas,
            'cos_sims': cos_sims,
        }
    
    def analyze_prefix_specific_transformation(self, n_prefixes: int = 10, n_suffixes: int = 20):
        """
        For a FIXED prefix A, how does it transform different suffixes B?
        
        If the transformation is prefix-specific but consistent across suffixes,
        we can precompute one transformation per prefix token.
        """
        print(f"\n--- Prefix-specific transformation analysis ---")
        
        results = []
        
        for p in range(n_prefixes):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            A_text = self.tokenizer.decode([A])
            
            h_B_list = []
            h_AB_list = []
            
            for s in range(n_suffixes):
                B = np.random.randint(0, self.tokenizer.vocab_size)
                
                try:
                    h_B = self.get_final_hidden([B])
                    h_AB = self.get_final_hidden([A, B])
                    
                    h_B_list.append(h_B)
                    h_AB_list.append(h_AB)
                except:
                    continue
            
            if len(h_B_list) < 5:
                continue
            
            h_B = np.array(h_B_list)
            h_AB = np.array(h_AB_list)
            
            # For this prefix, what's the transformation?
            deltas = h_AB - h_B
            delta_mean = deltas.mean(axis=0)
            delta_std = deltas.std(axis=0).mean()
            
            # How consistent is the delta?
            consistency = 1 - delta_std / (np.linalg.norm(delta_mean) + 1e-10)
            
            # Test translation with this prefix's mean delta
            predicted = h_B + delta_mean
            error = np.linalg.norm(h_AB - predicted, axis=1).mean()
            baseline = np.linalg.norm(h_AB - h_B, axis=1).mean()
            improvement = (1 - error/baseline) * 100
            
            results.append({
                'prefix': A,
                'prefix_text': A_text,
                'consistency': consistency,
                'improvement': improvement,
            })
            
            print(f"  Prefix '{A_text[:10]}': consistency={consistency:.3f}, improvement={improvement:.1f}%")
        
        # Summary
        mean_consistency = np.mean([r['consistency'] for r in results])
        mean_improvement = np.mean([r['improvement'] for r in results])
        
        print(f"\n  Summary:")
        print(f"    Mean consistency: {mean_consistency:.3f}")
        print(f"    Mean improvement with prefix-specific delta: {mean_improvement:.1f}%")
        
        return results
    
    def analyze_position_effect(self, n_samples: int = 50):
        """
        How does position affect the hidden state?
        
        Compare h(B) at position 0 vs position 1 vs position 2...
        """
        print(f"\n--- Position effect analysis ---")
        
        for pos in range(5):
            h_list = []
            
            for i in range(n_samples):
                B = np.random.randint(0, self.tokenizer.vocab_size)
                
                # Create sequence with B at position `pos`
                prefix = [np.random.randint(0, self.tokenizer.vocab_size) for _ in range(pos)]
                sequence = prefix + [B]
                
                try:
                    h = self.get_final_hidden(sequence)
                    h_list.append(h)
                except:
                    continue
            
            h = np.array(h_list)
            
            # Statistics
            mean_norm = np.linalg.norm(h, axis=1).mean()
            
            print(f"  Position {pos}: mean norm = {mean_norm:.1f}")


def main():
    print("=" * 70)
    print("CONTEXT GEOMETRY ANALYSIS")
    print("=" * 70)
    print("""
Key question: How does adding context TRANSFORM the hidden state?

If the transformation is simple (translation, rotation, scaling),
we can precompute it and avoid running the transformer!
""")
    
    analyzer = ContextGeometryAnalyzer()
    
    # 1. General transformation analysis
    results = analyzer.analyze_context_transformation(n_samples=100)
    
    # 2. Prefix-specific analysis
    prefix_results = analyzer.analyze_prefix_specific_transformation(n_prefixes=10, n_suffixes=20)
    
    # 3. Position effect
    analyzer.analyze_position_effect(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
