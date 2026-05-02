#!/usr/bin/env python3
"""
Attention Decomposition: Understanding the Shape Change
=========================================================

Key insight: For 2 tokens [A, B], the output at B is:
    h(A,B) = attn_to_A × V(A) + attn_to_B × V(B)

Where:
- attn_to_A = softmax(Q(B) @ K(A).T)
- V(A) = W_v @ embedding(A)

The shape change is determined by:
1. The attention weights (how much of A vs B)
2. The value vectors (what A and B contribute)

If we can precompute V(token) for all tokens, and predict attention weights,
we can compute h(A,B) without the full transformer!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AttentionDecompositionAnalyzer:
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
        self.n_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Heads: {self.n_heads}")
        print(f"  Head dim: {self.head_dim}")
    
    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Get token embeddings."""
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([token_ids]).to(device)
        
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(input_ids)
            return embeddings[0].float().cpu().numpy()
    
    def get_final_hidden(self, token_ids: List[int]) -> np.ndarray:
        """Get final hidden state."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def test_embedding_combination(self, n_samples: int = 100):
        """
        Test: Can we predict h(A,B) from embeddings?
        
        h(A,B) ≈ w_A × emb(A) + w_B × emb(B)
        """
        print(f"\n--- Embedding Combination Test ({n_samples} pairs) ---")
        
        emb_A_list = []
        emb_B_list = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb = self.get_embeddings([A, B])
                emb_A = emb[0]
                emb_B = emb[1]
                
                h_AB = self.get_final_hidden([A, B])
                
                emb_A_list.append(emb_A)
                emb_B_list.append(emb_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        emb_A = np.array(emb_A_list)
        emb_B = np.array(emb_B_list)
        h_AB = np.array(h_AB_list)
        
        # Test: h_AB = w_A × emb_A + w_B × emb_B
        # Solve for global w_A, w_B
        X = np.column_stack([emb_A.reshape(-1), emb_B.reshape(-1)])
        y = h_AB.reshape(-1)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        w_A, w_B = coeffs
        
        pred = w_A * emb_A + w_B * emb_B
        
        correct = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred[i]) 
                     for i in range(len(h_AB)))
        
        print(f"\n  Global weights: w_A={w_A:.3f}, w_B={w_B:.3f}")
        print(f"  Token accuracy: {correct}/{len(h_AB)} = {correct/len(h_AB)*100:.1f}%")
        
        # Test per-dimension weights
        w_A_dim = np.zeros(self.hidden_dim)
        w_B_dim = np.zeros(self.hidden_dim)
        
        for d in range(self.hidden_dim):
            X_d = np.column_stack([emb_A[:, d], emb_B[:, d]])
            y_d = h_AB[:, d]
            coeffs_d, _, _, _ = np.linalg.lstsq(X_d, y_d, rcond=None)
            w_A_dim[d], w_B_dim[d] = coeffs_d
        
        pred_dim = emb_A * w_A_dim + emb_B * w_B_dim
        
        correct_dim = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred_dim[i]) 
                        for i in range(len(h_AB)))
        
        print(f"\n  Per-dimension weights:")
        print(f"    w_A mean: {w_A_dim.mean():.3f}, std: {w_A_dim.std():.3f}")
        print(f"    w_B mean: {w_B_dim.mean():.3f}, std: {w_B_dim.std():.3f}")
        print(f"    Token accuracy: {correct_dim}/{len(h_AB)} = {correct_dim/len(h_AB)*100:.1f}%")
        
        return {
            'w_A': w_A,
            'w_B': w_B,
            'w_A_dim': w_A_dim,
            'w_B_dim': w_B_dim,
        }
    
    def test_generalization(self, w_A_dim: np.ndarray, w_B_dim: np.ndarray, n_test: int = 50):
        """Test if learned weights generalize."""
        print(f"\n--- Generalization Test ({n_test} new pairs) ---")
        
        correct = 0
        
        for i in range(n_test):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb = self.get_embeddings([A, B])
                emb_A = emb[0]
                emb_B = emb[1]
                
                h_AB = self.get_final_hidden([A, B])
                
                pred = emb_A * w_A_dim + emb_B * w_B_dim
                
                true_token = np.argmax(self.lm_head @ h_AB)
                pred_token = np.argmax(self.lm_head @ pred)
                
                if true_token == pred_token:
                    correct += 1
            except:
                continue
        
        accuracy = correct / n_test
        print(f"  Generalization accuracy: {correct}/{n_test} = {accuracy*100:.1f}%")
        
        return accuracy
    
    def analyze_layer_contributions(self, n_samples: int = 50):
        """
        Analyze how each layer transforms the hidden state.
        
        The transformer is a sequence of transformations:
        h_0 = embedding
        h_1 = layer_0(h_0)
        ...
        h_28 = layer_27(h_27)
        
        For 2 tokens, how does each layer combine the information?
        """
        print(f"\n--- Layer Contribution Analysis ---")
        
        device = next(self.model.parameters()).device
        
        # For each layer, measure how much the hidden state at B
        # depends on the hidden state at A vs B from the previous layer
        
        for i in range(min(5, n_samples)):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            input_ids = torch.tensor([[A, B]]).to(device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                
                # hidden_states[0] = embedding, [1] = after layer 0, etc.
                print(f"\n  Sample {i}: A={A}, B={B}")
                
                for layer_idx in range(0, self.n_layers, 7):  # Every 7th layer
                    h_prev = outputs.hidden_states[layer_idx]  # (1, 2, hidden)
                    h_curr = outputs.hidden_states[layer_idx + 1]
                    
                    # At position 1 (B), how much did it change?
                    h_prev_B = h_prev[0, 1, :].float().cpu().numpy()
                    h_curr_B = h_curr[0, 1, :].float().cpu().numpy()
                    
                    # Also look at position 0 (A)
                    h_prev_A = h_prev[0, 0, :].float().cpu().numpy()
                    
                    # How much of h_curr_B can be explained by h_prev_B vs h_prev_A?
                    X = np.column_stack([h_prev_A, h_prev_B])
                    coeffs, _, _, _ = np.linalg.lstsq(X, h_curr_B, rcond=None)
                    
                    print(f"    Layer {layer_idx}: coeff_A={coeffs[0]:.3f}, coeff_B={coeffs[1]:.3f}")


def main():
    print("=" * 70)
    print("ATTENTION DECOMPOSITION")
    print("=" * 70)
    print("""
Key insight: h(A,B) is a weighted combination of transformed embeddings.

If we can learn the weights, we can compute h(A,B) from embeddings alone!
""")
    
    analyzer = AttentionDecompositionAnalyzer()
    
    # 1. Test embedding combination
    results = analyzer.test_embedding_combination(n_samples=100)
    
    # 2. Test generalization
    analyzer.test_generalization(results['w_A_dim'], results['w_B_dim'], n_test=50)
    
    # 3. Analyze layer contributions
    analyzer.analyze_layer_contributions(n_samples=5)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
