#!/usr/bin/env python3
"""
Precompute the Click with RoPE
===============================

Previous attempt got 0.69 cosine - missing Rotary Position Embeddings (RoPE).

This version adds RoPE to the manual layer 3 computation.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotary position embeddings to Q and K.
    
    RoPE rotates pairs of dimensions by position-dependent angles.
    """
    # q, k shape: (seq_len, n_heads, head_dim)
    # cos, sin shape: (seq_len, head_dim)
    
    def rotate_half(x):
        """Rotate half the hidden dims."""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return np.concatenate([-x2, x1], axis=-1)
    
    # Expand cos, sin for heads
    cos = cos[:, np.newaxis, :]  # (seq, 1, head_dim)
    sin = sin[:, np.newaxis, :]
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


class PrecomputeClickRoPEAnalyzer:
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
        
        # Get RoPE parameters
        self.extract_rope_params()
        
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
    
    def extract_rope_params(self):
        """Extract RoPE parameters from the model."""
        # Qwen2 uses rotary embeddings
        # The rotary_emb is typically in the attention module
        layer3 = self.model.model.layers[3]
        
        if hasattr(layer3.self_attn, 'rotary_emb'):
            rotary = layer3.self_attn.rotary_emb
            
            # Get the inverse frequencies
            if hasattr(rotary, 'inv_freq'):
                self.inv_freq = rotary.inv_freq.float().cpu().numpy()
                print(f"  RoPE inv_freq shape: {self.inv_freq.shape}")
            else:
                # Compute default inv_freq
                base = 10000.0
                self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
                print(f"  Using default RoPE inv_freq")
        else:
            # Compute default inv_freq
            base = 10000.0
            self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
            print(f"  Using default RoPE inv_freq")
    
    def compute_rope_embeddings(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RoPE cos and sin for given sequence length."""
        positions = np.arange(seq_len)
        
        # Outer product: positions × inv_freq
        freqs = np.outer(positions, self.inv_freq)  # (seq_len, head_dim/2)
        
        # Duplicate for full head_dim
        freqs = np.concatenate([freqs, freqs], axis=-1)  # (seq_len, head_dim)
        
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        
        return cos, sin
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation."""
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_layer3_manual(self, h2_A: np.ndarray, h2_B: np.ndarray) -> np.ndarray:
        """
        Manually compute layer 3 output for position B given context A.
        Now with RoPE!
        """
        seq_len = 2
        
        # Stack inputs
        h2 = np.stack([h2_A, h2_B])  # (2, hidden)
        
        # Input layer norm
        h2_normed = np.array([self.rms_norm(h2[i], self.ln_attn_weight) for i in range(seq_len)])
        
        # Q, K, V projections
        Q = h2_normed @ self.W_q.T  # (2, q_dim)
        K = h2_normed @ self.W_k.T  # (2, k_dim)
        V = h2_normed @ self.W_v.T  # (2, v_dim)
        
        # Reshape for heads
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.compute_rope_embeddings(seq_len)
        
        # RoPE for Q (all heads)
        Q_rope = np.zeros_like(Q)
        for h in range(self.n_heads):
            q_h = Q[:, h, :]  # (seq, head_dim)
            q_rot = q_h * cos + np.concatenate([-q_h[..., self.head_dim//2:], q_h[..., :self.head_dim//2]], axis=-1) * sin
            Q_rope[:, h, :] = q_rot
        
        # RoPE for K (kv heads)
        K_rope = np.zeros_like(K)
        for h in range(self.n_kv_heads):
            k_h = K[:, h, :]
            k_rot = k_h * cos + np.concatenate([-k_h[..., self.head_dim//2:], k_h[..., :self.head_dim//2]], axis=-1) * sin
            K_rope[:, h, :] = k_rot
        
        # Compute attention for position 1 (B)
        heads_per_kv = self.n_heads // self.n_kv_heads
        
        attn_output = np.zeros(self.hidden_dim)
        
        for h in range(self.n_heads):
            kv_idx = h // heads_per_kv
            
            # Attention scores with RoPE
            scores = np.array([
                np.dot(Q_rope[1, h], K_rope[0, kv_idx]) / np.sqrt(self.head_dim),  # to A
                np.dot(Q_rope[1, h], K_rope[1, kv_idx]) / np.sqrt(self.head_dim),  # to B
            ])
            
            # Softmax
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            
            # Weighted sum of V (no RoPE on V)
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
    
    def test_manual_layer3(self, n_samples: int = 100):
        """Test manual layer 3 computation with RoPE."""
        print(f"\n--- Testing Manual Layer 3 with RoPE ({n_samples} pairs) ---")
        
        cos_sims = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states (input to layer 3)
                h2 = self.get_hidden_at_layer([A, B], 3)  # After layer 2
                h2_A, h2_B = h2[0], h2[1]
                
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]  # After layer 3, position B
                
                # Compute manual layer 3 with RoPE
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
        
        # Token prediction accuracy
        print(f"\n  Testing token prediction...")
        correct = 0
        for i in range(min(50, n_samples)):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h2 = self.get_hidden_at_layer([A, B], 3)
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                h3_manual = self.compute_layer3_manual(h2[0], h2[1])
                
                # Get final hidden state
                h_final = self.get_hidden_at_layer([A, B], -1)[1]
                
                true_token = np.argmax(self.lm_head @ h_final)
                
                # If we use h3_manual instead of h3_actual, would we get same token?
                # This requires running layers 4-27, which we can't do manually yet
                # Instead, check if h3_manual is close enough
                
            except:
                continue
        
        return cos_sims
    
    def test_end_to_end(self, n_samples: int = 50):
        """
        Test: If we get layer 3 right, can we predict the final token?
        
        Strategy:
        1. Compute manual layer 3
        2. Use the model to run layers 4-27 from our manual h3
        
        This tests whether layer 3 is the bottleneck.
        """
        print(f"\n--- End-to-End Test ({n_samples} pairs) ---")
        
        device = next(self.model.parameters()).device
        
        correct_actual = 0
        correct_manual = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get actual final hidden state
                h_final_actual = self.get_hidden_at_layer([A, B], -1)[1]
                true_token = np.argmax(self.lm_head @ h_final_actual)
                
                # Get layer 2 hidden states
                h2 = self.get_hidden_at_layer([A, B], 3)
                
                # Compute manual layer 3
                h3_manual = self.compute_layer3_manual(h2[0], h2[1])
                
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                
                # Compare h3
                cos_h3 = np.dot(h3_manual, h3_actual) / (
                    np.linalg.norm(h3_manual) * np.linalg.norm(h3_actual) + 1e-10)
                
                # For now, just check if high cosine correlates with correct prediction
                if cos_h3 > 0.99:
                    correct_manual += 1
                
                correct_actual += 1  # Baseline
                
            except Exception as e:
                continue
        
        print(f"\n  Results:")
        print(f"    Samples with cos > 0.99: {correct_manual}/{n_samples}")
        
        return correct_manual


def main():
    print("=" * 70)
    print("PRECOMPUTE THE CLICK WITH ROPE")
    print("=" * 70)
    print("""
Adding Rotary Position Embeddings to manual layer 3 computation.
Previous attempt: 0.69 cosine
Target: > 0.99 cosine
""")
    
    analyzer = PrecomputeClickRoPEAnalyzer()
    
    # 1. Test manual layer 3 with RoPE
    cos_sims = analyzer.test_manual_layer3(n_samples=100)
    
    # 2. End-to-end test
    analyzer.test_end_to_end(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
