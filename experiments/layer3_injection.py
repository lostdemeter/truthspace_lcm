#!/usr/bin/env python3
"""
Layer 3 Injection: The Click Point
===================================

Key finding: Context injection happens at layer 3!
- Layers 0-2: cosine similarity 0.60-0.99 (similar to single token)
- Layer 3: cosine similarity drops to 0.11 (massive divergence)
- Layers 4+: stays low (0.10-0.47)

This is the "click" in the safe dial analogy.

Hypothesis: If we can characterize WHAT happens at layer 3,
we can predict the shape change.

Questions:
1. What causes the layer 3 divergence?
2. Can we predict the layer 3 output from layer 2?
3. Can we precompute the "click" transformation?

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class Layer3InjectionAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_hidden_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get hidden states at specific layer for all positions."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[layer][0].float().cpu().numpy()
    
    def get_attention_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get attention weights at specific layer."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            # (batch, heads, seq, seq) -> (heads, seq, seq)
            return outputs.attentions[layer][0].float().cpu().numpy()
    
    def analyze_layer3_transformation(self, n_samples: int = 100):
        """
        Analyze what happens at layer 3.
        
        Compare:
        - h3_alone = layer3(h2_alone) for single token
        - h3_context = layer3(h2_context) for token in context
        
        What's the difference?
        """
        print(f"\n--- Layer 3 Transformation Analysis ({n_samples} pairs) ---")
        
        # Collect data
        h2_B_alone_list = []
        h3_B_alone_list = []
        h2_B_context_list = []
        h3_B_context_list = []
        h2_A_context_list = []
        attn3_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # B alone
                h2_alone = self.get_hidden_at_layer([B], 3)  # Layer index 3 = after layer 2
                h3_alone = self.get_hidden_at_layer([B], 4)  # After layer 3
                
                # B in context
                h2_context = self.get_hidden_at_layer([A, B], 3)
                h3_context = self.get_hidden_at_layer([A, B], 4)
                
                # Attention at layer 3
                attn3 = self.get_attention_at_layer([A, B], 3)  # (heads, 2, 2)
                
                h2_B_alone_list.append(h2_alone[0])
                h3_B_alone_list.append(h3_alone[0])
                h2_B_context_list.append(h2_context[1])
                h3_B_context_list.append(h3_context[1])
                h2_A_context_list.append(h2_context[0])
                attn3_list.append(attn3[:, 1, :].mean(axis=0))  # Attention from B, avg across heads
            except:
                continue
        
        h2_B_alone = np.array(h2_B_alone_list)
        h3_B_alone = np.array(h3_B_alone_list)
        h2_B_context = np.array(h2_B_context_list)
        h3_B_context = np.array(h3_B_context_list)
        h2_A_context = np.array(h2_A_context_list)
        attn3 = np.array(attn3_list)  # (n_samples, 2)
        
        print(f"\n  Layer 2 comparison (before the click):")
        cos_h2 = [np.dot(h2_B_alone[i], h2_B_context[i]) / 
                  (np.linalg.norm(h2_B_alone[i]) * np.linalg.norm(h2_B_context[i]) + 1e-10)
                  for i in range(len(h2_B_alone))]
        print(f"    Cosine(h2_B_alone, h2_B_context): {np.mean(cos_h2):.4f}")
        
        print(f"\n  Layer 3 comparison (after the click):")
        cos_h3 = [np.dot(h3_B_alone[i], h3_B_context[i]) / 
                  (np.linalg.norm(h3_B_alone[i]) * np.linalg.norm(h3_B_context[i]) + 1e-10)
                  for i in range(len(h3_B_alone))]
        print(f"    Cosine(h3_B_alone, h3_B_context): {np.mean(cos_h3):.4f}")
        
        print(f"\n  Attention at layer 3:")
        print(f"    Mean attention to A: {attn3[:, 0].mean():.4f}")
        print(f"    Mean attention to B: {attn3[:, 1].mean():.4f}")
        
        # Key question: Can we predict h3_B_context from h2_B_alone, h2_A_context, and attention?
        # h3_B_context ≈ f(h2_B_alone, h2_A_context, attn3)
        
        print(f"\n  Testing: h3_B_context = attn_A * h2_A + attn_B * h2_B")
        
        # Simple weighted combination
        pred_h3 = attn3[:, 0:1] * h2_A_context + attn3[:, 1:2] * h2_B_context
        
        cos_pred = [np.dot(pred_h3[i], h3_B_context[i]) / 
                    (np.linalg.norm(pred_h3[i]) * np.linalg.norm(h3_B_context[i]) + 1e-10)
                    for i in range(len(pred_h3))]
        print(f"    Cosine(predicted, actual): {np.mean(cos_pred):.4f}")
        
        # The layer also has MLP - let's see if the residual matters
        delta_attn = h3_B_context - pred_h3
        delta_norm = np.linalg.norm(delta_attn, axis=1).mean()
        h3_norm = np.linalg.norm(h3_B_context, axis=1).mean()
        print(f"    Residual norm: {delta_norm:.1f} (h3 norm: {h3_norm:.1f})")
        
        return {
            'h2_B_alone': h2_B_alone,
            'h3_B_alone': h3_B_alone,
            'h2_B_context': h2_B_context,
            'h3_B_context': h3_B_context,
            'h2_A_context': h2_A_context,
            'attn3': attn3,
        }
    
    def test_attention_prediction(self, n_samples: int = 100):
        """
        Test: Can we predict the layer 3 attention from layer 2 hidden states?
        
        Attention = softmax(Q @ K.T / sqrt(d))
        Q = W_q @ h2_B
        K = W_k @ h2_A
        
        If we can predict attention, we can predict the "click".
        """
        print(f"\n--- Attention Prediction at Layer 3 ---")
        
        # Extract Q, K matrices from layer 3
        layer3 = self.model.model.layers[3]
        W_q = layer3.self_attn.q_proj.weight.data.float().cpu().numpy()
        W_k = layer3.self_attn.k_proj.weight.data.float().cpu().numpy()
        
        print(f"  W_q shape: {W_q.shape}")
        print(f"  W_k shape: {W_k.shape}")
        
        n_heads = self.model.config.num_attention_heads
        n_kv_heads = self.model.config.num_key_value_heads
        head_dim = self.hidden_dim // n_heads
        
        print(f"  n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
        
        predicted_attns = []
        actual_attns = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states (input to layer 3)
                h2 = self.get_hidden_at_layer([A, B], 3)  # (2, hidden)
                h2_A, h2_B = h2[0], h2[1]
                
                # Compute Q and K
                Q_B = h2_B @ W_q.T  # (hidden,) -> (q_dim,)
                K_A = h2_A @ W_k.T  # (hidden,) -> (k_dim,)
                K_B = h2_B @ W_k.T
                
                # Reshape for heads
                # Q has n_heads, K has n_kv_heads (GQA)
                Q_B = Q_B.reshape(n_heads, head_dim)
                K_A = K_A.reshape(n_kv_heads, head_dim)
                K_B = K_B.reshape(n_kv_heads, head_dim)
                
                # For GQA, each K head serves multiple Q heads
                heads_per_kv = n_heads // n_kv_heads
                
                # Compute attention scores
                scores_to_A = []
                scores_to_B = []
                
                for h in range(n_heads):
                    kv_idx = h // heads_per_kv
                    score_A = np.dot(Q_B[h], K_A[kv_idx]) / np.sqrt(head_dim)
                    score_B = np.dot(Q_B[h], K_B[kv_idx]) / np.sqrt(head_dim)
                    scores_to_A.append(score_A)
                    scores_to_B.append(score_B)
                
                scores = np.array([scores_to_A, scores_to_B]).T  # (n_heads, 2)
                
                # Softmax
                exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                pred_attn = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                pred_attn_to_A = pred_attn[:, 0].mean()
                
                # Get actual attention
                actual_attn = self.get_attention_at_layer([A, B], 3)
                actual_attn_to_A = actual_attn[:, 1, 0].mean()
                
                predicted_attns.append(pred_attn_to_A)
                actual_attns.append(actual_attn_to_A)
            except Exception as e:
                continue
        
        predicted_attns = np.array(predicted_attns)
        actual_attns = np.array(actual_attns)
        
        # Correlation
        correlation = np.corrcoef(predicted_attns, actual_attns)[0, 1]
        error = np.abs(predicted_attns - actual_attns).mean()
        
        print(f"\n  Results:")
        print(f"    Predicted mean: {predicted_attns.mean():.4f}")
        print(f"    Actual mean: {actual_attns.mean():.4f}")
        print(f"    Correlation: {correlation:.4f}")
        print(f"    Mean absolute error: {error:.4f}")
        
        return {
            'predicted': predicted_attns,
            'actual': actual_attns,
            'correlation': correlation,
        }
    
    def test_click_prediction(self, n_samples: int = 50):
        """
        Test: Can we predict the final token using the "click" mechanism?
        
        Strategy:
        1. Use single-token cache for layers 0-2
        2. Predict layer 3 attention from layer 2 hidden states
        3. Compute layer 3 output using predicted attention
        4. Use single-token cache for layers 4-27 (adjusted for context)
        
        This is complex - let's start with a simpler test:
        If we know the TRUE layer 3 output, can we predict the final token?
        """
        print(f"\n--- Click Prediction Test ---")
        
        # Test: Use true h3_context, then single-token transformation for layers 4-27
        # This measures: How much does context matter AFTER the click?
        
        correct_with_context = 0
        correct_single_token = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get true final hidden state
                h_final_context = self.get_hidden_at_layer([A, B], -1)[1]
                
                # Get single-token final hidden state
                h_final_alone = self.get_hidden_at_layer([B], -1)[0]
                
                true_token = np.argmax(self.lm_head @ h_final_context)
                pred_token_alone = np.argmax(self.lm_head @ h_final_alone)
                
                if true_token == pred_token_alone:
                    correct_single_token += 1
            except:
                continue
        
        accuracy_single = correct_single_token / n_samples
        print(f"\n  Single-token accuracy: {correct_single_token}/{n_samples} = {accuracy_single*100:.1f}%")
        
        return accuracy_single


def main():
    print("=" * 70)
    print("LAYER 3 INJECTION: THE CLICK POINT")
    print("=" * 70)
    print("""
Key finding: Context injection happens at layer 3!
- Cosine similarity drops from 0.60 to 0.11

This is the "click" in the safe dial analogy.
""")
    
    analyzer = Layer3InjectionAnalyzer()
    
    # 1. Analyze layer 3 transformation
    layer3_results = analyzer.analyze_layer3_transformation(n_samples=100)
    
    # 2. Test attention prediction
    attn_results = analyzer.test_attention_prediction(n_samples=100)
    
    # 3. Test click prediction
    accuracy = analyzer.test_click_prediction(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
