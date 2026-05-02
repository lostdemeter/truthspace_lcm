#!/usr/bin/env python3
"""
MESH with Bias and RoPE: The Complete Fix
==========================================

Deep diagnosis found:
- Q, K, V all have BIASES (I was missing these!)
- Manual Q = Actual Q when bias is included (1.0 cosine)

Now let's add both biases AND RoPE to get the full computation correct.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class MeshWithBiasRoPEAnalyzer:
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
        self.device = next(self.model.parameters()).device
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        # Extract layer 3 weights WITH BIASES
        self.extract_layer3()
        
        # Extract RoPE parameters
        self.extract_rope()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def extract_layer3(self):
        """Extract layer 3 weights including biases."""
        layer3 = self.model.model.layers[3]
        attn = layer3.self_attn
        
        # Weights
        self.W_q = attn.q_proj.weight.data.float().cpu().numpy()
        self.W_k = attn.k_proj.weight.data.float().cpu().numpy()
        self.W_v = attn.v_proj.weight.data.float().cpu().numpy()
        self.W_o = attn.o_proj.weight.data.float().cpu().numpy()
        
        # Biases
        self.b_q = attn.q_proj.bias.data.float().cpu().numpy() if attn.q_proj.bias is not None else None
        self.b_k = attn.k_proj.bias.data.float().cpu().numpy() if attn.k_proj.bias is not None else None
        self.b_v = attn.v_proj.bias.data.float().cpu().numpy() if attn.v_proj.bias is not None else None
        
        # Layer norms
        self.ln_weight = layer3.input_layernorm.weight.data.float().cpu().numpy()
        self.ln_mlp_weight = layer3.post_attention_layernorm.weight.data.float().cpu().numpy()
        
        # MLP weights
        self.W_gate = layer3.mlp.gate_proj.weight.data.float().cpu().numpy()
        self.W_up = layer3.mlp.up_proj.weight.data.float().cpu().numpy()
        self.W_down = layer3.mlp.down_proj.weight.data.float().cpu().numpy()
        
        # Per-head weights
        self.heads_per_kv = self.n_heads // self.n_kv_heads
        self.W_q_heads = self.W_q.reshape(self.n_heads, self.head_dim, self.hidden_dim)
        self.W_k_heads = self.W_k.reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
        self.W_v_heads = self.W_v.reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
        
        # Per-head biases
        if self.b_q is not None:
            self.b_q_heads = self.b_q.reshape(self.n_heads, self.head_dim)
        if self.b_k is not None:
            self.b_k_heads = self.b_k.reshape(self.n_kv_heads, self.head_dim)
        if self.b_v is not None:
            self.b_v_heads = self.b_v.reshape(self.n_kv_heads, self.head_dim)
        
        print(f"  Extracted layer 3 weights")
        print(f"    Q bias: {self.b_q.shape if self.b_q is not None else None}")
        print(f"    K bias: {self.b_k.shape if self.b_k is not None else None}")
        print(f"    V bias: {self.b_v.shape if self.b_v is not None else None}")
    
    def extract_rope(self):
        """Extract RoPE parameters."""
        layer3 = self.model.model.layers[3]
        
        if hasattr(layer3.self_attn, 'rotary_emb'):
            rotary = layer3.self_attn.rotary_emb
            if hasattr(rotary, 'inv_freq'):
                self.inv_freq = rotary.inv_freq.float().cpu().numpy()
            else:
                base = 10000.0
                self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        else:
            base = 10000.0
            self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        
        print(f"  Extracted RoPE inv_freq: {self.inv_freq.shape}")
    
    def compute_rope_embeddings(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RoPE cos and sin for given positions."""
        freqs = np.outer(positions, self.inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Apply RoPE to a vector."""
        x1 = x[:self.head_dim//2]
        x2 = x[self.head_dim//2:]
        x_rotated = np.concatenate([-x2, x1])
        return x * cos + x_rotated * sin
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation."""
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_attention_with_bias_rope(self, h_A: np.ndarray, h_B: np.ndarray,
                                          pos_A: int = 0, pos_B: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention with biases and RoPE."""
        # Apply layer norm
        h_A_norm = self.rms_norm(h_A, self.ln_weight)
        h_B_norm = self.rms_norm(h_B, self.ln_weight)
        
        # Compute RoPE embeddings
        positions = np.array([pos_A, pos_B])
        cos, sin = self.compute_rope_embeddings(positions)
        
        attn_output = np.zeros(self.hidden_dim)
        attn_weights = []
        
        for h in range(self.n_heads):
            kv_idx = h // self.heads_per_kv
            
            # Q projection with bias
            q_B = h_B_norm @ self.W_q_heads[h].T + self.b_q_heads[h]
            
            # K projections with bias
            k_A = h_A_norm @ self.W_k_heads[kv_idx].T + self.b_k_heads[kv_idx]
            k_B = h_B_norm @ self.W_k_heads[kv_idx].T + self.b_k_heads[kv_idx]
            
            # Apply RoPE
            q_B_rope = self.apply_rope(q_B, cos[1], sin[1])
            k_A_rope = self.apply_rope(k_A, cos[0], sin[0])
            k_B_rope = self.apply_rope(k_B, cos[1], sin[1])
            
            # Attention scores
            score_to_A = np.dot(q_B_rope, k_A_rope) / np.sqrt(self.head_dim)
            score_to_B = np.dot(q_B_rope, k_B_rope) / np.sqrt(self.head_dim)
            
            # Softmax
            scores = np.array([score_to_A, score_to_B])
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            attn_weights.append(attn[0])
            
            # V projections with bias (no RoPE on V)
            v_A = h_A_norm @ self.W_v_heads[kv_idx].T + self.b_v_heads[kv_idx]
            v_B = h_B_norm @ self.W_v_heads[kv_idx].T + self.b_v_heads[kv_idx]
            
            # Weighted sum
            v_out = attn[0] * v_A + attn[1] * v_B
            
            attn_output[h * self.head_dim:(h+1) * self.head_dim] = v_out
        
        # Output projection (no bias on o_proj)
        attn_output = attn_output @ self.W_o.T
        
        return attn_output, np.array(attn_weights)
    
    def compute_layer3_complete(self, h2_A: np.ndarray, h2_B: np.ndarray) -> np.ndarray:
        """Compute full layer 3 with biases and RoPE."""
        # Attention
        attn_output, _ = self.compute_attention_with_bias_rope(h2_A, h2_B)
        
        # Residual
        h3_pre_mlp = h2_B + attn_output
        
        # MLP
        h3_norm = self.rms_norm(h3_pre_mlp, self.ln_mlp_weight)
        gate = h3_norm @ self.W_gate.T
        up = h3_norm @ self.W_up.T
        mlp_out = self.silu(gate) * up
        mlp_out = mlp_out @ self.W_down.T
        
        # Residual
        h3 = h3_pre_mlp + mlp_out
        
        return h3
    
    def get_hidden_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get hidden states at specific layer."""
        input_ids = torch.tensor([token_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[layer][0].float().cpu().numpy()
    
    def test_complete_computation(self, n_samples: int = 100):
        """Test complete layer 3 computation with biases and RoPE."""
        print(f"\n--- Testing Complete Computation ({n_samples} pairs) ---")
        
        cos_sims = []
        attn_corrs = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states
                h2 = self.get_hidden_at_layer([A, B], 3)
                h2_A, h2_B = h2[0], h2[1]
                
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                
                # Compute with biases + RoPE
                h3_computed = self.compute_layer3_complete(h2_A, h2_B)
                
                # Compare
                cos = np.dot(h3_computed, h3_actual) / (
                    np.linalg.norm(h3_computed) * np.linalg.norm(h3_actual) + 1e-10)
                cos_sims.append(cos)
                
                # Compare attention weights
                input_ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_attentions=True)
                    actual_attn = outputs.attentions[3][0, :, 1, 0].cpu().numpy()
                
                _, computed_attn = self.compute_attention_with_bias_rope(h2_A, h2_B)
                
                corr = np.corrcoef(actual_attn, computed_attn)[0, 1]
                attn_corrs.append(corr)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Results:")
        print(f"    h3 cosine (computed vs actual): {np.mean(cos_sims):.4f}")
        print(f"    Attention weight correlation:   {np.mean(attn_corrs):.4f}")
        
        print(f"\n  Comparison to previous attempts:")
        print(f"    Without bias, without RoPE: 0.70 cosine, 0.33 attn corr")
        print(f"    With bias + RoPE:           {np.mean(cos_sims):.2f} cosine, {np.mean(attn_corrs):.2f} attn corr")
        
        return cos_sims, attn_corrs
    
    def diagnose_remaining_gap(self, n_samples: int = 20):
        """Diagnose any remaining gap."""
        print(f"\n--- Diagnosing Remaining Gap ({n_samples} pairs) ---")
        
        layer3 = self.model.model.layers[3]
        
        q_cos_list = []
        k_cos_list = []
        attn_score_errors = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            input_ids = torch.tensor([[A, B]]).to(self.device)
            
            captured = {}
            
            def capture_q(module, input, output):
                captured['q'] = output.detach().float().cpu().numpy()
            
            def capture_k(module, input, output):
                captured['k'] = output.detach().float().cpu().numpy()
            
            h1 = layer3.self_attn.q_proj.register_forward_hook(capture_q)
            h2 = layer3.self_attn.k_proj.register_forward_hook(capture_k)
            
            try:
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
                
                h2_states = outputs.hidden_states[3][0].float().cpu().numpy()
                h2_A, h2_B = h2_states[0], h2_states[1]
                
                # Actual Q and K (after projection, before RoPE)
                q_actual = captured['q'][0, 1]
                k_actual_A = captured['k'][0, 0]
                
                # Manual Q and K with bias
                h_B_norm = self.rms_norm(h2_B, self.ln_weight)
                h_A_norm = self.rms_norm(h2_A, self.ln_weight)
                
                q_manual = h_B_norm @ self.W_q.T + self.b_q
                k_manual_A = h_A_norm @ self.W_k.T + self.b_k
                
                # Compare (before RoPE)
                q_cos = np.dot(q_actual, q_manual) / (
                    np.linalg.norm(q_actual) * np.linalg.norm(q_manual) + 1e-10)
                k_cos = np.dot(k_actual_A, k_manual_A) / (
                    np.linalg.norm(k_actual_A) * np.linalg.norm(k_manual_A) + 1e-10)
                
                q_cos_list.append(q_cos)
                k_cos_list.append(k_cos)
                
                # Compare attention weights
                actual_attn = outputs.attentions[3][0, :, 1, 0].cpu().numpy()
                _, computed_attn = self.compute_attention_with_bias_rope(h2_A, h2_B)
                
                attn_error = np.abs(actual_attn - computed_attn).mean()
                attn_score_errors.append(attn_error)
                
            finally:
                h1.remove()
                h2.remove()
        
        print(f"  Q projection (with bias, before RoPE): {np.mean(q_cos_list):.4f}")
        print(f"  K projection (with bias, before RoPE): {np.mean(k_cos_list):.4f}")
        print(f"  Attention weight MAE: {np.mean(attn_score_errors):.4f}")
        
        return q_cos_list, k_cos_list


def main():
    print("=" * 70)
    print("MESH WITH BIAS AND ROPE: THE COMPLETE FIX")
    print("=" * 70)
    print("""
Deep diagnosis found:
- Q, K, V all have BIASES (I was missing these!)
- Manual Q = Actual Q when bias is included (1.0 cosine)

Adding both biases AND RoPE for complete computation.
Expected: >0.95 cosine
""")
    
    analyzer = MeshWithBiasRoPEAnalyzer()
    
    # 1. Diagnose Q/K with bias
    analyzer.diagnose_remaining_gap(n_samples=20)
    
    # 2. Test complete computation
    cos_sims, attn_corrs = analyzer.test_complete_computation(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
