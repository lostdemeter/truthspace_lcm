#!/usr/bin/env python3
"""
MESH Context: Use Actual MESH to Compute h3
=============================================

We found:
- MESH exists: 28 heads × (3584, 3584)
- Singular values follow φ-structure
- k=233 for 90% variance

Now let's test: Can we compute h3 using the MESH directly?

From Doc 129, the unraveled attention is:
    score = input @ MESH @ input.T

For 2 tokens [A, B], the attention from B to A is:
    score_BA = h_B @ MESH @ h_A.T

Then the output is:
    attn_out = softmax(scores) @ V

Let's implement this and test if it matches the actual h3.

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


class MeshContextAnalyzer:
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
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        # Extract layer 3 weights
        self.extract_layer3()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def extract_layer3(self):
        """Extract and precompute MESH for layer 3."""
        layer3 = self.model.model.layers[3]
        
        self.W_q = layer3.self_attn.q_proj.weight.data.float().cpu().numpy()
        self.W_k = layer3.self_attn.k_proj.weight.data.float().cpu().numpy()
        self.W_v = layer3.self_attn.v_proj.weight.data.float().cpu().numpy()
        self.W_o = layer3.self_attn.o_proj.weight.data.float().cpu().numpy()
        
        self.ln_weight = layer3.input_layernorm.weight.data.float().cpu().numpy()
        
        # MLP weights
        self.W_gate = layer3.mlp.gate_proj.weight.data.float().cpu().numpy()
        self.W_up = layer3.mlp.up_proj.weight.data.float().cpu().numpy()
        self.W_down = layer3.mlp.down_proj.weight.data.float().cpu().numpy()
        self.ln_mlp_weight = layer3.post_attention_layernorm.weight.data.float().cpu().numpy()
        
        # Compute per-head MESH
        heads_per_kv = self.n_heads // self.n_kv_heads
        
        W_q_heads = self.W_q.reshape(self.n_heads, self.head_dim, self.hidden_dim)
        W_k_heads = self.W_k.reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
        W_v_heads = self.W_v.reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
        
        self.mesh_qk = []
        self.W_v_per_head = []
        
        for h in range(self.n_heads):
            kv_idx = h // heads_per_kv
            # MESH_h = W_q_h.T @ W_k_kv
            mesh_h = W_q_heads[h].T @ W_k_heads[kv_idx]
            self.mesh_qk.append(mesh_h)
            self.W_v_per_head.append(W_v_heads[kv_idx])
        
        self.mesh_qk = np.array(self.mesh_qk)  # (28, 3584, 3584)
        self.W_v_per_head = np.array(self.W_v_per_head)  # (28, 128, 3584)
        
        print(f"  Extracted MESH: {self.mesh_qk.shape}")
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation."""
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_attention_with_mesh(self, h_A: np.ndarray, h_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention output using precomputed MESH.
        
        Returns attention output and attention weights.
        """
        # Apply layer norm
        h_A_norm = self.rms_norm(h_A, self.ln_weight)
        h_B_norm = self.rms_norm(h_B, self.ln_weight)
        
        # Compute attention scores using MESH
        # score_BA = h_B @ MESH @ h_A.T (for each head)
        # score_BB = h_B @ MESH @ h_B.T
        
        attn_output = np.zeros(self.hidden_dim)
        attn_weights = []
        
        for h in range(self.n_heads):
            # Attention scores
            score_to_A = h_B_norm @ self.mesh_qk[h] @ h_A_norm / np.sqrt(self.head_dim)
            score_to_B = h_B_norm @ self.mesh_qk[h] @ h_B_norm / np.sqrt(self.head_dim)
            
            # Softmax
            scores = np.array([score_to_A, score_to_B])
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            attn_weights.append(attn[0])  # Attention to A
            
            # Value projection
            v_A = h_A_norm @ self.W_v_per_head[h].T
            v_B = h_B_norm @ self.W_v_per_head[h].T
            
            # Weighted sum
            v_out = attn[0] * v_A + attn[1] * v_B
            
            # Add to output
            attn_output[h * self.head_dim:(h+1) * self.head_dim] = v_out
        
        # Output projection
        attn_output = attn_output @ self.W_o.T
        
        return attn_output, np.array(attn_weights)
    
    def compute_layer3_with_mesh(self, h2_A: np.ndarray, h2_B: np.ndarray) -> np.ndarray:
        """
        Compute full layer 3 output using MESH.
        """
        # Attention
        attn_output, _ = self.compute_attention_with_mesh(h2_A, h2_B)
        
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
    
    def test_mesh_computation(self, n_samples: int = 100):
        """
        Test: Does MESH-based computation match actual layer 3?
        """
        print(f"\n--- Testing MESH Computation ({n_samples} pairs) ---")
        
        cos_sims = []
        attn_errors = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states (input to layer 3)
                h2 = self.get_hidden_at_layer([A, B], 3)
                h2_A, h2_B = h2[0], h2[1]
                
                # Get actual layer 3 output
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                
                # Compute with MESH
                h3_mesh = self.compute_layer3_with_mesh(h2_A, h2_B)
                
                # Compare
                cos = np.dot(h3_mesh, h3_actual) / (
                    np.linalg.norm(h3_mesh) * np.linalg.norm(h3_actual) + 1e-10)
                cos_sims.append(cos)
                
                # Also compare attention weights
                input_ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_attentions=True)
                    actual_attn = outputs.attentions[3][0, :, 1, 0].mean().item()
                
                _, mesh_attn = self.compute_attention_with_mesh(h2_A, h2_B)
                mesh_attn_mean = mesh_attn.mean()
                
                attn_errors.append(abs(actual_attn - mesh_attn_mean))
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Results:")
        print(f"    h3 cosine (MESH vs actual): {np.mean(cos_sims):.4f}")
        print(f"    Attention error: {np.mean(attn_errors):.4f}")
        
        return cos_sims, attn_errors
    
    def test_token_prediction_with_mesh(self, n_samples: int = 50):
        """
        Test: Can we predict the final token using MESH-computed h3?
        """
        print(f"\n--- Token Prediction with MESH ({n_samples} pairs) ---")
        
        correct = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get layer 2 hidden states
                h2 = self.get_hidden_at_layer([A, B], 3)
                
                # Compute h3 with MESH
                h3_mesh = self.compute_layer3_with_mesh(h2[0], h2[1])
                
                # Get actual final hidden state
                h_final_actual = self.get_hidden_at_layer([A, B], -1)[1]
                true_token = np.argmax(self.lm_head @ h_final_actual)
                
                # Get actual h3
                h3_actual = self.get_hidden_at_layer([A, B], 4)[1]
                
                # Compare h3 tokens
                token_from_h3_actual = np.argmax(self.lm_head @ h3_actual)
                token_from_h3_mesh = np.argmax(self.lm_head @ h3_mesh)
                
                if token_from_h3_actual == token_from_h3_mesh:
                    correct += 1
                    
            except:
                continue
        
        accuracy = correct / n_samples
        print(f"\n  h3 token match: {correct}/{n_samples} = {accuracy*100:.1f}%")
        
        return accuracy


def main():
    print("=" * 70)
    print("MESH CONTEXT: USE ACTUAL MESH TO COMPUTE h3")
    print("=" * 70)
    print("""
Using the transformer's own MESH = W_q.T @ W_k to compute layer 3.

This is the "unwinding" approach from Doc 129:
- Pre-compute MESH
- Use MESH to compute attention scores
- Apply V projection and MLP

If this works, we can store MESH in φ-format (Doc 151).
""")
    
    analyzer = MeshContextAnalyzer()
    
    # 1. Test MESH computation
    cos_sims, attn_errors = analyzer.test_mesh_computation(n_samples=100)
    
    # 2. Test token prediction
    accuracy = analyzer.test_token_prediction_with_mesh(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
