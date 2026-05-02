#!/usr/bin/env python3
"""
Precompute the Click: Can We Replace Layer 3?
==============================================

Key insight: The "click" at layer 3 is determined by:
1. Q from current token
2. K from context tokens
3. V from all tokens
4. MLP transformation

All of these are deterministic from the layer 2 hidden states.

Test: Can we precompute layer 3 and get correct predictions?

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class PrecomputeClickAnalyzer:
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
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        # Extract layer 3 weights
        self.extract_layer3_weights()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def extract_layer3_weights(self):
        """Extract Q, K, V, O projection matrices from layer 3."""
        layer3 = self.model.model.layers[3]
        
        self.W_q = layer3.self_attn.q_proj.weight.data.float().cpu().numpy()
        self.W_k = layer3.self_attn.k_proj.weight.data.float().cpu().numpy()
        self.W_v = layer3.self_attn.v_proj.weight.data.float().cpu().numpy()
        self.W_o = layer3.self_attn.o_proj.weight.data.float().cpu().numpy()
        
        # MLP weights
        self.W_gate = layer3.mlp.gate_proj.weight.data.float().cpu().numpy()
        self.W_up = layer3.mlp.up_proj.weight.data.float().cpu().numpy()
        self.W_down = layer3.mlp.down_proj.weight.data.float().cpu().numpy()
        
        # Layer norms
        self.ln_attn_weight = layer3.input_layernorm.weight.data.float().cpu().numpy()
        self.ln_mlp_weight = layer3.post_attention_layernorm.weight.data.float().cpu().numpy()
        
        print(f"  Extracted layer 3 weights")
        print(f"    W_q: {self.W_q.shape}, W_k: {self.W_k.shape}, W_v: {self.W_v.shape}")
        print(f"    W_gate: {self.W_gate.shape}, W_up: {self.W_up.shape}, W_down: {self.W_down.shape}")
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation."""
        return x * (1 / (1 + np.exp(-x)))
    
    def compute_layer3_manual(self, h2_A: np.ndarray, h2_B: np.ndarray) -> np.ndarray:
        """
        Manually compute layer 3 output for position B given context A.
        
        This simulates what the transformer does, using extracted weights.
        """
        # Stack inputs
        h2 = np.stack([h2_A, h2_B])  # (2, hidden)
        
        # Input layer norm
        h2_normed = np.array([self.rms_norm(h2[i], self.ln_attn_weight) for i in range(2)])
        
        # Q, K, V projections
        Q = h2_normed @ self.W_q.T  # (2, q_dim)
        K = h2_normed @ self.W_k.T  # (2, k_dim)
        V = h2_normed @ self.W_v.T  # (2, v_dim)
        
        # Reshape for heads
        Q = Q.reshape(2, self.n_heads, self.head_dim)
        K = K.reshape(2, self.n_kv_heads, self.head_dim)
        V = V.reshape(2, self.n_kv_heads, self.head_dim)
        
        # Compute attention for position 1 (B)
        heads_per_kv = self.n_heads // self.n_kv_heads
        
        attn_output = np.zeros(self.hidden_dim)
        
        for h in range(self.n_heads):
            kv_idx = h // heads_per_kv
            
            # Attention scores
            scores = np.array([
                np.dot(Q[1, h], K[0, kv_idx]) / np.sqrt(self.head_dim),  # to A
                np.dot(Q[1, h], K[1, kv_idx]) / np.sqrt(self.head_dim),  # to B
            ])
            
            # Softmax
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            
            # Weighted sum of V
            v_out = attn[0] * V[0, kv_idx] + attn[1] * V[1, kv_idx]
            
            # Add to output
            attn_output[h * self.head_dim:(h+1) * self.head_dim] = v_out
        
        # Output projection
        attn_output = attn_output @ self.W_o.T
        
        # Residual connection
        h3_pre_mlp = h2[1] + attn_output
        
        # MLP layer norm
        h3_normed = self.rms_norm(h3_pre_mlp, self.ln_mlp_weight)
        
        # MLP: SiLU(gate) * up, then down
        gate = h3_normed @ self.W_gate.T
        up = h3_normed @ self.W_up.T
        mlp_out = self.silu(gate) * up
        mlp_out = mlp_out @ self.W_down.T
        
        # Residual connection
        h3 = h3_pre_mlp + mlp_out
        
        return h3
    
    def get_hidden_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get hidden states at specific layer."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[layer][0].float().cpu().numpy()
    
    def test_manual_layer3(self, n_samples: int = 50):
        """
        Test: Does our manual layer 3 computation match the model?
        """
        print(f"\n--- Testing Manual Layer 3 Computation ({n_samples} pairs) ---")
        
        cos_sims = []
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states (input to layer 3)
                h2 = self.get_hidden_at_layer([A, B], 3)  # After layer 2
                h2_A, h2_B = h2[0], h2[1]
                
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]  # After layer 3, position B
                
                # Compute manual layer 3
                h3_manual = self.compute_layer3_manual(h2_A, h2_B)
                
                # Compare
                cos = np.dot(h3_manual, h3_actual) / (
                    np.linalg.norm(h3_manual) * np.linalg.norm(h3_actual) + 1e-10)
                cos_sims.append(cos)
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Results:")
        print(f"    Mean cosine similarity: {np.mean(cos_sims):.4f}")
        print(f"    Min: {np.min(cos_sims):.4f}, Max: {np.max(cos_sims):.4f}")
        
        return cos_sims
    
    def test_precomputed_qkv(self, n_samples: int = 50):
        """
        Test: Can we precompute Q, K, V and get correct layer 3 output?
        
        Strategy:
        1. Precompute Q, K, V for single tokens (from layer 2 single-token hidden states)
        2. At inference: use precomputed Q, K, V to compute attention
        3. Apply attention and MLP
        """
        print(f"\n--- Testing Precomputed Q, K, V ({n_samples} pairs) ---")
        
        # First, precompute Q, K, V for a sample of tokens
        n_cache = 1000
        print(f"  Precomputing Q, K, V for {n_cache} tokens...")
        
        Q_cache = {}
        K_cache = {}
        V_cache = {}
        h2_cache = {}  # Also cache layer 2 hidden states
        
        for token_id in range(n_cache):
            try:
                # Get layer 2 hidden state for single token
                h2 = self.get_hidden_at_layer([token_id], 3)[0]
                h2_cache[token_id] = h2
                
                # Apply layer norm
                h2_normed = self.rms_norm(h2, self.ln_attn_weight)
                
                # Compute Q, K, V
                Q_cache[token_id] = h2_normed @ self.W_q.T
                K_cache[token_id] = h2_normed @ self.W_k.T
                V_cache[token_id] = h2_normed @ self.W_v.T
            except:
                continue
        
        print(f"  Cached {len(Q_cache)} tokens")
        
        # Now test on pairs
        correct = 0
        cos_sims = []
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, min(n_cache, self.tokenizer.vocab_size))
            B = np.random.randint(0, min(n_cache, self.tokenizer.vocab_size))
            
            if A not in Q_cache or B not in Q_cache:
                continue
            
            try:
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                
                # Use precomputed Q, K, V
                # BUT: The layer 2 hidden state for B in context is DIFFERENT from B alone!
                # So we can't just use the cached values...
                
                # Let's test: What if we use the TRUE layer 2 hidden states?
                h2 = self.get_hidden_at_layer([A, B], 3)
                h2_A, h2_B = h2[0], h2[1]
                
                # Compare cached vs actual layer 2 hidden states
                h2_A_cached = h2_cache.get(A)
                h2_B_cached = h2_cache.get(B)
                
                if h2_A_cached is not None and h2_B_cached is not None:
                    cos_A = np.dot(h2_A, h2_A_cached) / (
                        np.linalg.norm(h2_A) * np.linalg.norm(h2_A_cached) + 1e-10)
                    cos_B = np.dot(h2_B, h2_B_cached) / (
                        np.linalg.norm(h2_B) * np.linalg.norm(h2_B_cached) + 1e-10)
                    
                    # The issue: h2_B in context is different from h2_B alone!
                    # This is the "pre-click" divergence we saw earlier
                
                # Compute layer 3 using TRUE h2 values
                h3_manual = self.compute_layer3_manual(h2_A, h2_B)
                
                cos = np.dot(h3_manual, h3_actual) / (
                    np.linalg.norm(h3_manual) * np.linalg.norm(h3_actual) + 1e-10)
                cos_sims.append(cos)
                
                # Token prediction
                true_token = np.argmax(self.lm_head @ self.get_hidden_at_layer([A, B], -1)[1])
                
                # If we could get h3 right, would we get the final token right?
                # This requires running layers 4-27 too...
                
            except Exception as e:
                continue
        
        print(f"\n  Results:")
        print(f"    Mean cosine (manual vs actual layer 3): {np.mean(cos_sims):.4f}")
        
        return cos_sims


def main():
    print("=" * 70)
    print("PRECOMPUTE THE CLICK")
    print("=" * 70)
    print("""
Test: Can we manually compute layer 3 using extracted weights?

If yes, we can potentially:
1. Precompute Q, K, V for all tokens
2. Compute attention at inference
3. Apply MLP
4. Skip the transformer for layer 3
""")
    
    analyzer = PrecomputeClickAnalyzer()
    
    # 1. Test manual layer 3 computation
    cos_sims = analyzer.test_manual_layer3(n_samples=50)
    
    # 2. Test precomputed Q, K, V
    analyzer.test_precomputed_qkv(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
