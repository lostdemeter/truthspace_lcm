#!/usr/bin/env python3
"""
Layer 0 Injection Hypothesis
=============================

Key finding: After layer 0, coeff_A ≈ 0!

This means:
- Layer 0 injects context from A into B
- Layers 1-27 transform B's hidden state independently

If true, we can:
1. Precompute how layer 0 transforms each token
2. Precompute how layers 1-27 transform each hidden state
3. Combine them at inference time!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class Layer0InjectionAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_all_hidden_states(self, token_ids: List[int]) -> List[np.ndarray]:
        """Get hidden states at all layers."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # Return hidden states at position -1 (last token)
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def get_hidden_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get hidden state at specific layer for all positions."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[layer][0].float().cpu().numpy()  # (seq_len, hidden)
    
    def analyze_layer0_output(self, n_samples: int = 100):
        """
        Analyze the output of layer 0 for 2-token sequences.
        
        Key question: Can we predict h_1(B) from h_0(A) and h_0(B)?
        Where h_1(B) is the hidden state at position B after layer 0.
        """
        print(f"\n--- Layer 0 Output Analysis ({n_samples} pairs) ---")
        
        h0_A_list = []
        h0_B_list = []
        h1_B_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get hidden states at layer 0 (embeddings) and layer 1 (after first layer)
                h0 = self.get_hidden_at_layer([A, B], 0)  # (2, hidden)
                h1 = self.get_hidden_at_layer([A, B], 1)  # (2, hidden)
                
                h0_A_list.append(h0[0])
                h0_B_list.append(h0[1])
                h1_B_list.append(h1[1])
            except:
                continue
        
        h0_A = np.array(h0_A_list)
        h0_B = np.array(h0_B_list)
        h1_B = np.array(h1_B_list)
        
        # Test: h1_B = w_A × h0_A + w_B × h0_B
        print(f"\n  Testing: h1_B = w_A × h0_A + w_B × h0_B")
        
        # Global weights
        X = np.column_stack([h0_A.reshape(-1), h0_B.reshape(-1)])
        y = h1_B.reshape(-1)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        w_A, w_B = coeffs
        
        print(f"    Global: w_A={w_A:.3f}, w_B={w_B:.3f}")
        
        # Per-dimension weights
        w_A_dim = np.zeros(self.hidden_dim)
        w_B_dim = np.zeros(self.hidden_dim)
        
        for d in range(self.hidden_dim):
            X_d = np.column_stack([h0_A[:, d], h0_B[:, d]])
            y_d = h1_B[:, d]
            coeffs_d, _, _, _ = np.linalg.lstsq(X_d, y_d, rcond=None)
            w_A_dim[d], w_B_dim[d] = coeffs_d
        
        pred = h0_A * w_A_dim + h0_B * w_B_dim
        
        # Reconstruction error
        error = np.linalg.norm(h1_B - pred, axis=1).mean()
        baseline = np.linalg.norm(h1_B, axis=1).mean()
        
        print(f"    Per-dim: w_A mean={w_A_dim.mean():.3f}, w_B mean={w_B_dim.mean():.3f}")
        print(f"    Reconstruction error: {error:.1f} (baseline={baseline:.1f})")
        print(f"    Explained: {(1 - error/baseline)*100:.1f}%")
        
        return {
            'w_A_dim': w_A_dim,
            'w_B_dim': w_B_dim,
            'h0_A': h0_A,
            'h0_B': h0_B,
            'h1_B': h1_B,
        }
    
    def test_layer0_then_cache(self, n_samples: int = 100):
        """
        Test: If we compute layer 0 output, can we use the single-token cache
        for the rest?
        
        Strategy:
        1. Compute h1_B = layer0(h0_A, h0_B)
        2. Use precomputed single-token transformation for layers 1-27
        
        This requires knowing how layers 1-27 transform h1_B.
        """
        print(f"\n--- Layer 0 + Cache Test ---")
        
        # First, let's see if layers 1-27 are the same for single tokens
        # Compare: h_final(B alone) vs h_final(B in context of A)
        
        h_B_alone_list = []
        h_B_context_list = []
        h1_B_context_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # B alone
                h_B_alone = self.get_all_hidden_states([B])
                
                # B in context of A
                h_AB = self.get_hidden_at_layer([A, B], -1)  # Final layer, all positions
                h_B_context = h_AB[1]  # Position B
                
                # Layer 1 output for B in context
                h1_AB = self.get_hidden_at_layer([A, B], 1)
                h1_B_context = h1_AB[1]
                
                h_B_alone_list.append(h_B_alone[-1])  # Final hidden for B alone
                h_B_context_list.append(h_B_context)
                h1_B_context_list.append(h1_B_context)
            except:
                continue
        
        h_B_alone = np.array(h_B_alone_list)
        h_B_context = np.array(h_B_context_list)
        h1_B_context = np.array(h1_B_context_list)
        
        # How different are h_B_alone and h_B_context?
        cos_sims = []
        for i in range(len(h_B_alone)):
            cos = np.dot(h_B_alone[i], h_B_context[i]) / (
                np.linalg.norm(h_B_alone[i]) * np.linalg.norm(h_B_context[i]) + 1e-10)
            cos_sims.append(cos)
        
        print(f"\n  Comparison: h_final(B alone) vs h_final(B in context)")
        print(f"    Mean cosine similarity: {np.mean(cos_sims):.4f}")
        print(f"    Min: {np.min(cos_sims):.4f}, Max: {np.max(cos_sims):.4f}")
        
        # Key test: If we start from h1_B_context (layer 0 output with context),
        # can we predict h_B_context (final output)?
        # This would require knowing how layers 1-27 transform the hidden state.
        
        # The transformation from h1 to h_final is what we cached for single tokens!
        # But does it generalize?
        
        return {
            'h_B_alone': h_B_alone,
            'h_B_context': h_B_context,
            'h1_B_context': h1_B_context,
            'cos_sims': cos_sims,
        }
    
    def analyze_layer1_to_final_transformation(self, n_samples: int = 100):
        """
        Analyze the transformation from layer 1 to final layer.
        
        For single tokens: h1 → h_final is deterministic (can cache)
        For 2 tokens: Is h1_B → h_final_B the same transformation?
        """
        print(f"\n--- Layer 1 to Final Transformation Analysis ---")
        
        # Collect layer 1 and final hidden states for single tokens
        h1_single_list = []
        hf_single_list = []
        
        print(f"  Collecting single-token data...")
        for i in range(n_samples):
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                states = self.get_all_hidden_states([B])
                h1_single_list.append(states[1])  # After layer 0
                hf_single_list.append(states[-1])  # Final
            except:
                continue
        
        h1_single = np.array(h1_single_list)
        hf_single = np.array(hf_single_list)
        
        # Learn transformation: hf = f(h1)
        # Try linear: hf ≈ h1 @ W + b
        # Use low-rank approximation
        
        print(f"\n  Learning h1 → h_final transformation (single tokens):")
        
        # Center the data
        h1_mean = h1_single.mean(axis=0)
        hf_mean = hf_single.mean(axis=0)
        
        h1_centered = h1_single - h1_mean
        hf_centered = hf_single - hf_mean
        
        # SVD-based low-rank linear transformation
        # hf_centered ≈ h1_centered @ W
        # W = h1_centered.T @ hf_centered @ (h1_centered @ h1_centered.T)^-1
        # Use pseudo-inverse
        
        for k in [10, 50, 100]:
            # Project h1 to k dimensions
            _, _, Vt_h1 = np.linalg.svd(h1_centered, full_matrices=False)
            h1_k = h1_centered @ Vt_h1[:k].T
            
            # Solve for W: hf_centered ≈ h1_k @ W
            W, _, _, _ = np.linalg.lstsq(h1_k, hf_centered, rcond=None)
            
            pred_hf = h1_k @ W + hf_mean
            
            # Token accuracy
            correct = sum(np.argmax(self.lm_head @ hf_single[i]) == np.argmax(self.lm_head @ pred_hf[i]) 
                         for i in range(len(hf_single)))
            
            print(f"    k={k}: {correct}/{len(hf_single)} = {correct/len(hf_single)*100:.1f}%")
        
        # Now test on 2-token sequences
        print(f"\n  Testing on 2-token sequences:")
        
        correct_2tok = 0
        total_2tok = 0
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 1 and final for B in context
                h1_AB = self.get_hidden_at_layer([A, B], 1)
                hf_AB = self.get_hidden_at_layer([A, B], -1)
                
                h1_B = h1_AB[1]
                hf_B = hf_AB[1]
                
                # Apply learned transformation
                h1_B_centered = h1_B - h1_mean
                h1_B_k = h1_B_centered @ Vt_h1[:100].T
                pred_hf_B = h1_B_k @ W + hf_mean
                
                true_token = np.argmax(self.lm_head @ hf_B)
                pred_token = np.argmax(self.lm_head @ pred_hf_B)
                
                if true_token == pred_token:
                    correct_2tok += 1
                total_2tok += 1
            except:
                continue
        
        accuracy_2tok = correct_2tok / total_2tok if total_2tok > 0 else 0
        print(f"    2-token accuracy: {correct_2tok}/{total_2tok} = {accuracy_2tok*100:.1f}%")


def main():
    print("=" * 70)
    print("LAYER 0 INJECTION HYPOTHESIS")
    print("=" * 70)
    print("""
Key finding: After layer 0, coeff_A ≈ 0!

This means layer 0 injects context, then layers 1-27 transform independently.
If true, we can decompose the computation!
""")
    
    analyzer = Layer0InjectionAnalyzer()
    
    # 1. Analyze layer 0 output
    layer0_results = analyzer.analyze_layer0_output(n_samples=100)
    
    # 2. Test layer 0 + cache approach
    cache_results = analyzer.test_layer0_then_cache(n_samples=100)
    
    # 3. Analyze layer 1 to final transformation
    analyzer.analyze_layer1_to_final_transformation(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
