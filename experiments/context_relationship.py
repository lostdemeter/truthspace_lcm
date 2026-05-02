#!/usr/bin/env python3
"""
Context as Relationship Geometry
=================================

Key insight: The transformer computes WHERE IN THE LATTICE the 
combination of A and B lands.

This isn't about transforming h(A) or h(B) - it's about computing
the RELATIONSHIP between them.

From Doc 180 (Platonic Ideals): Relationships are ROTATIONS toward
Platonic Ideals. The rotation angle is universal (~77° for capital-of).

Hypothesis: The context transformation IS a rotation, but the axis
depends on the RELATIONSHIP between A and B, not just A.

Questions:
1. What is the "relationship" between two tokens geometrically?
2. Can we characterize it with a small number of parameters?
3. Can we precompute relationship types?

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ContextRelationshipAnalyzer:
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
    
    def compute_relationship_vector(self, h_A: np.ndarray, h_B: np.ndarray) -> np.ndarray:
        """
        Compute a vector that characterizes the relationship between A and B.
        
        Options:
        1. Difference: h_B - h_A
        2. Concatenation: [h_A, h_B]
        3. Outer product features
        4. Attention-like: softmax(h_A @ h_B) * h_B
        """
        # Simple difference
        return h_B - h_A
    
    def analyze_relationship_structure(self, n_samples: int = 100):
        """
        Analyze the structure of relationship vectors.
        
        If relationships cluster into a small number of types,
        we can precompute transformations per type.
        """
        print(f"\n--- Relationship Structure Analysis ({n_samples} pairs) ---")
        
        relationships = []
        h_AB_list = []
        h_B_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                rel = self.compute_relationship_vector(h_A, h_B)
                relationships.append(rel)
                h_AB_list.append(h_AB)
                h_B_list.append(h_B)
            except:
                continue
        
        relationships = np.array(relationships)
        h_AB = np.array(h_AB_list)
        h_B = np.array(h_B_list)
        
        # SVD of relationships
        _, S, Vt = np.linalg.svd(relationships, full_matrices=False)
        
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Relationship vector SVD:")
        print(f"    Top 10 singular values: {S[:10].round(1)}")
        print(f"    Dims for 50% variance: {np.searchsorted(cumvar, 0.50) + 1}")
        print(f"    Dims for 90% variance: {np.searchsorted(cumvar, 0.90) + 1}")
        print(f"    Dims for 99% variance: {np.searchsorted(cumvar, 0.99) + 1}")
        
        # Can we predict h_AB from relationship + h_B?
        print(f"\n  Testing: h_AB = f(relationship, h_B)")
        
        # Concatenate relationship and h_B as features
        features = np.hstack([relationships, h_B])
        
        # Simple linear regression (too many params, but let's see the upper bound)
        # Use low-rank approximation
        for k in [10, 50, 100]:
            # Project features to k dimensions
            _, _, Vt_feat = np.linalg.svd(features, full_matrices=False)
            features_k = features @ Vt_feat[:k].T
            
            # Solve for W: h_AB ≈ features_k @ W
            W, _, _, _ = np.linalg.lstsq(features_k, h_AB, rcond=None)
            
            pred = features_k @ W
            
            correct = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred[i]) 
                         for i in range(len(h_AB)))
            
            print(f"    k={k}: {correct}/{len(h_AB)} = {correct/len(h_AB)*100:.1f}%")
        
        return {
            'relationships': relationships,
            'h_AB': h_AB,
            'h_B': h_B,
            'Vt': Vt,
        }
    
    def analyze_delta_structure(self, n_samples: int = 100):
        """
        Analyze the structure of the DELTA: h_AB - h_B
        
        This is what the context ADDS to the suffix.
        """
        print(f"\n--- Delta Structure Analysis ---")
        
        deltas = []
        h_A_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                delta = h_AB - h_B
                deltas.append(delta)
                h_A_list.append(h_A)
            except:
                continue
        
        deltas = np.array(deltas)
        h_A = np.array(h_A_list)
        
        # SVD of deltas
        _, S, Vt = np.linalg.svd(deltas, full_matrices=False)
        
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Delta (h_AB - h_B) SVD:")
        print(f"    Top 10 singular values: {S[:10].round(1)}")
        print(f"    Dims for 50% variance: {np.searchsorted(cumvar, 0.50) + 1}")
        print(f"    Dims for 90% variance: {np.searchsorted(cumvar, 0.90) + 1}")
        print(f"    Dims for 99% variance: {np.searchsorted(cumvar, 0.99) + 1}")
        
        # Can we predict delta from h_A?
        print(f"\n  Testing: delta = f(h_A)")
        
        for k in [10, 50, 100]:
            # Project h_A to k dimensions
            _, _, Vt_A = np.linalg.svd(h_A, full_matrices=False)
            h_A_k = h_A @ Vt_A[:k].T
            
            # Solve for W: delta ≈ h_A_k @ W
            W, _, _, _ = np.linalg.lstsq(h_A_k, deltas, rcond=None)
            
            pred_delta = h_A_k @ W
            
            # Reconstruction error
            error = np.linalg.norm(deltas - pred_delta, axis=1).mean()
            baseline = np.linalg.norm(deltas, axis=1).mean()
            
            print(f"    k={k}: error={error:.1f} (baseline={baseline:.1f}, {(1-error/baseline)*100:.1f}% explained)")
        
        # Correlation between delta and h_A
        correlations = []
        for i in range(len(deltas)):
            corr = np.corrcoef(deltas[i], h_A[i])[0, 1]
            correlations.append(corr)
        
        print(f"\n  Correlation between delta and h_A:")
        print(f"    Mean: {np.mean(correlations):.3f}")
        print(f"    Std: {np.std(correlations):.3f}")
        
        return {
            'deltas': deltas,
            'h_A': h_A,
        }
    
    def analyze_attention_pattern(self, n_samples: int = 50):
        """
        Look at what attention actually does.
        
        The transformer uses attention to combine tokens.
        What is the attention pattern for 2-token sequences?
        """
        print(f"\n--- Attention Pattern Analysis ---")
        
        device = next(self.model.parameters()).device
        
        attention_to_first = []
        attention_to_second = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            input_ids = torch.tensor([[A, B]]).to(device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
                
                # Get attention from last layer, last token
                # Shape: (batch, heads, seq_len, seq_len)
                attn = outputs.attentions[-1][0]  # (heads, 2, 2)
                
                # Attention from position 1 (B) to positions 0 (A) and 1 (B)
                attn_to_A = attn[:, 1, 0].mean().item()  # Average across heads
                attn_to_B = attn[:, 1, 1].mean().item()
                
                attention_to_first.append(attn_to_A)
                attention_to_second.append(attn_to_B)
        
        print(f"\n  Attention from B to A and B:")
        print(f"    Mean attention to A (prefix): {np.mean(attention_to_first):.3f}")
        print(f"    Mean attention to B (self): {np.mean(attention_to_second):.3f}")
        print(f"    Std attention to A: {np.std(attention_to_first):.3f}")
        
        return {
            'attention_to_first': attention_to_first,
            'attention_to_second': attention_to_second,
        }


def main():
    print("=" * 70)
    print("CONTEXT AS RELATIONSHIP GEOMETRY")
    print("=" * 70)
    print("""
Key question: What IS the shape change geometrically?

The transformer computes WHERE IN THE LATTICE the combination lands.
This depends on the RELATIONSHIP between the tokens.
""")
    
    analyzer = ContextRelationshipAnalyzer()
    
    # 1. Relationship structure
    rel_results = analyzer.analyze_relationship_structure(n_samples=100)
    
    # 2. Delta structure
    delta_results = analyzer.analyze_delta_structure(n_samples=100)
    
    # 3. Attention pattern
    attn_results = analyzer.analyze_attention_pattern(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
