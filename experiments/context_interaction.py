#!/usr/bin/env python3
"""
Context as Interaction: h(A,B) = f(h(A), h(B))
===============================================

Key finding: Sign flips are NOT consistent for a given prefix.
They depend on BOTH prefix AND suffix.

This means context is an INTERACTION, not a transformation.

Hypotheses for f:
1. ADDITION: h(A,B) = h(A) + h(B)
2. HADAMARD: h(A,B) = h(A) * h(B)  (element-wise)
3. BILINEAR: h(A,B) = h(A) @ W @ h(B)
4. ATTENTION: h(A,B) = softmax(h(A) @ h(B)) * h(B)

If one of these works, we can precompute h(A) for all tokens
and compute h(A,B) = f(h(A), h(B)) at inference time!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ContextInteractionAnalyzer:
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
    
    def test_interaction_hypotheses(self, n_samples: int = 100):
        """
        Test different interaction hypotheses.
        """
        print(f"\n--- Testing Interaction Hypotheses ({n_samples} pairs) ---")
        
        h_A_list = []
        h_B_list = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                h_A_list.append(h_A)
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        h_A = np.array(h_A_list)
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        
        baseline_error = np.linalg.norm(h_AB - h_B, axis=1).mean()
        print(f"\n  Baseline error (h_B → h_AB): {baseline_error:.2f}")
        
        # 1. ADDITION: h(A,B) = h(A) + h(B)
        pred_add = h_A + h_B
        error_add = np.linalg.norm(h_AB - pred_add, axis=1).mean()
        print(f"\n  1. ADDITION: h(A) + h(B)")
        print(f"     Error: {error_add:.2f} ({(1 - error_add/baseline_error)*100:.1f}% improvement)")
        
        # Test token prediction accuracy
        correct_add = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_add[i]) 
                         for i in range(len(h_AB)))
        print(f"     Token accuracy: {correct_add}/{len(h_AB)} = {correct_add/len(h_AB)*100:.1f}%")
        
        # 2. WEIGHTED ADDITION: h(A,B) = a*h(A) + b*h(B)
        # Solve for optimal a, b
        # h_AB = a*h_A + b*h_B
        # Stack and solve least squares
        X = np.column_stack([h_A.reshape(-1), h_B.reshape(-1)])
        y = h_AB.reshape(-1)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs
        
        pred_weighted = a * h_A + b * h_B
        error_weighted = np.linalg.norm(h_AB - pred_weighted, axis=1).mean()
        print(f"\n  2. WEIGHTED ADDITION: {a:.3f}*h(A) + {b:.3f}*h(B)")
        print(f"     Error: {error_weighted:.2f} ({(1 - error_weighted/baseline_error)*100:.1f}% improvement)")
        
        correct_weighted = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_weighted[i]) 
                              for i in range(len(h_AB)))
        print(f"     Token accuracy: {correct_weighted}/{len(h_AB)} = {correct_weighted/len(h_AB)*100:.1f}%")
        
        # 3. HADAMARD: h(A,B) = h(A) * h(B)
        pred_hadamard = h_A * h_B
        error_hadamard = np.linalg.norm(h_AB - pred_hadamard, axis=1).mean()
        print(f"\n  3. HADAMARD: h(A) * h(B)")
        print(f"     Error: {error_hadamard:.2f} ({(1 - error_hadamard/baseline_error)*100:.1f}% improvement)")
        
        correct_hadamard = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_hadamard[i]) 
                              for i in range(len(h_AB)))
        print(f"     Token accuracy: {correct_hadamard}/{len(h_AB)} = {correct_hadamard/len(h_AB)*100:.1f}%")
        
        # 4. SCALED HADAMARD: h(A,B) = c * h(A) * h(B)
        # Find optimal c
        c = np.sum(h_AB * (h_A * h_B)) / (np.sum((h_A * h_B)**2) + 1e-10)
        pred_scaled_hadamard = c * h_A * h_B
        error_scaled_hadamard = np.linalg.norm(h_AB - pred_scaled_hadamard, axis=1).mean()
        print(f"\n  4. SCALED HADAMARD: {c:.3f} * h(A) * h(B)")
        print(f"     Error: {error_scaled_hadamard:.2f} ({(1 - error_scaled_hadamard/baseline_error)*100:.1f}% improvement)")
        
        # 5. COMBINATION: h(A,B) = a*h(A) + b*h(B) + c*h(A)*h(B)
        X = np.column_stack([h_A.reshape(-1), h_B.reshape(-1), (h_A * h_B).reshape(-1)])
        y = h_AB.reshape(-1)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs
        
        pred_combo = a * h_A + b * h_B + c * h_A * h_B
        error_combo = np.linalg.norm(h_AB - pred_combo, axis=1).mean()
        print(f"\n  5. COMBINATION: {a:.3f}*h(A) + {b:.3f}*h(B) + {c:.3f}*h(A)*h(B)")
        print(f"     Error: {error_combo:.2f} ({(1 - error_combo/baseline_error)*100:.1f}% improvement)")
        
        correct_combo = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_combo[i]) 
                           for i in range(len(h_AB)))
        print(f"     Token accuracy: {correct_combo}/{len(h_AB)} = {correct_combo/len(h_AB)*100:.1f}%")
        
        # 6. Per-dimension weights: h(A,B)[i] = w_A[i]*h(A)[i] + w_B[i]*h(B)[i]
        print(f"\n  6. PER-DIMENSION WEIGHTS:")
        
        # Solve for w_A, w_B per dimension
        w_A = np.zeros(self.hidden_dim)
        w_B = np.zeros(self.hidden_dim)
        
        for d in range(self.hidden_dim):
            X_d = np.column_stack([h_A[:, d], h_B[:, d]])
            y_d = h_AB[:, d]
            coeffs_d, _, _, _ = np.linalg.lstsq(X_d, y_d, rcond=None)
            w_A[d], w_B[d] = coeffs_d
        
        pred_perdim = h_A * w_A + h_B * w_B
        error_perdim = np.linalg.norm(h_AB - pred_perdim, axis=1).mean()
        print(f"     Error: {error_perdim:.2f} ({(1 - error_perdim/baseline_error)*100:.1f}% improvement)")
        
        correct_perdim = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_perdim[i]) 
                            for i in range(len(h_AB)))
        print(f"     Token accuracy: {correct_perdim}/{len(h_AB)} = {correct_perdim/len(h_AB)*100:.1f}%")
        
        # Weight statistics
        print(f"\n     w_A stats: mean={w_A.mean():.3f}, std={w_A.std():.3f}")
        print(f"     w_B stats: mean={w_B.mean():.3f}, std={w_B.std():.3f}")
        
        return {
            'h_A': h_A,
            'h_B': h_B,
            'h_AB': h_AB,
            'w_A': w_A,
            'w_B': w_B,
        }
    
    def test_generalization(self, w_A: np.ndarray, w_B: np.ndarray, n_test: int = 50):
        """
        Test if learned weights generalize to new token pairs.
        """
        print(f"\n--- Generalization Test ({n_test} new pairs) ---")
        
        correct = 0
        
        for i in range(n_test):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                # Predict using learned weights
                pred = h_A * w_A + h_B * w_B
                
                true_token = np.argmax(self.lm_head @ h_AB)
                pred_token = np.argmax(self.lm_head @ pred)
                
                if true_token == pred_token:
                    correct += 1
            except:
                continue
        
        accuracy = correct / n_test
        print(f"  Generalization accuracy: {correct}/{n_test} = {accuracy*100:.1f}%")
        
        return accuracy


def main():
    print("=" * 70)
    print("CONTEXT AS INTERACTION")
    print("=" * 70)
    print("""
Key finding: Context is an INTERACTION between prefix and suffix.
h(A,B) = f(h(A), h(B))

If we can find a simple f, we can:
1. Precompute h(token) for all 152K tokens (1.09 GB)
2. Compute h(A,B) = f(h(A), h(B)) at inference time
3. No transformer needed!
""")
    
    analyzer = ContextInteractionAnalyzer()
    
    # 1. Test interaction hypotheses
    results = analyzer.test_interaction_hypotheses(n_samples=100)
    
    # 2. Test generalization
    analyzer.test_generalization(results['w_A'], results['w_B'], n_test=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
