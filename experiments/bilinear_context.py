#!/usr/bin/env python3
"""
Bilinear Context: The MESH for (A,B) → h3
==========================================

Key insight from Doc 129: MESH = W_q.T @ W_k captures Q-K relationship.

For context, we need a similar structure:
    h3(A, B) = emb_A @ CONTEXT_MESH @ emb_B.T + bias

This is a bilinear form - exactly like attention scores!

But h3 is a vector (3584 dims), not a scalar. So we need:
    h3[d] = emb_A @ MESH_d @ emb_B.T + bias_d

That's 3584 MESH matrices, each (3584, 3584). Too big!

Alternative: Low-rank approximation
    MESH_d ≈ U_d @ V_d.T  (rank k)
    h3[d] = (emb_A @ U_d) @ (emb_B @ V_d).T

If k is small, this is tractable.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class BilinearContextAnalyzer:
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
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Vocab size: {self.embeddings.shape[0]}")
    
    def get_layer3_output(self, A: int, B: int) -> np.ndarray:
        """Get layer 3 output for token pair (A, B)."""
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[4][0, 1].float().cpu().numpy()
    
    def collect_data(self, n_samples: int = 500):
        """Collect (emb_A, emb_B, h3) triples."""
        print(f"\n--- Collecting {n_samples} samples ---")
        
        emb_A_list = []
        emb_B_list = []
        h3_list = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb_A_list.append(self.embeddings[A])
                emb_B_list.append(self.embeddings[B])
                h3_list.append(self.get_layer3_output(A, B))
            except:
                continue
        
        return np.array(emb_A_list), np.array(emb_B_list), np.array(h3_list)
    
    def learn_bilinear_form(self, emb_A: np.ndarray, emb_B: np.ndarray, h3: np.ndarray, rank: int = 50):
        """
        Learn low-rank bilinear form:
            h3[d] ≈ (emb_A @ U_d) * (emb_B @ V_d)
        
        Where U_d, V_d are (hidden_dim, rank) matrices.
        
        Simplified: Learn U, V such that:
            h3 ≈ (emb_A @ U) * (emb_B @ V)
        
        Where U, V are (hidden_dim, hidden_dim) and we take element-wise product.
        """
        print(f"\n--- Learning Bilinear Form (rank={rank}) ---")
        
        n_samples = len(h3)
        
        # Approach: Use alternating least squares
        # Fix V, solve for U
        # Fix U, solve for V
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(self.hidden_dim, rank) * 0.01
        V = np.random.randn(self.hidden_dim, rank) * 0.01
        
        for iteration in range(10):
            # Compute current prediction
            # pred[i] = (emb_A[i] @ U) * (emb_B[i] @ V)  # (rank,)
            # But we want h3 which is (hidden_dim,)
            
            # Actually, let's try a different formulation:
            # h3 ≈ W_out @ ((emb_A @ U) * (emb_B @ V))
            # Where W_out is (hidden_dim, rank)
            
            A_proj = emb_A @ U  # (n, rank)
            B_proj = emb_B @ V  # (n, rank)
            interaction = A_proj * B_proj  # (n, rank) - element-wise
            
            # Solve for W_out: h3 ≈ interaction @ W_out.T
            W_out, _, _, _ = np.linalg.lstsq(interaction, h3, rcond=None)
            
            pred = interaction @ W_out
            
            # Compute error
            cos_sims = [np.dot(h3[i], pred[i]) / 
                       (np.linalg.norm(h3[i]) * np.linalg.norm(pred[i]) + 1e-10)
                       for i in range(n_samples)]
            mean_cos = np.mean(cos_sims)
            
            if iteration % 2 == 0:
                print(f"    Iteration {iteration}: cosine = {mean_cos:.4f}")
            
            # Update U and V using gradient descent (simplified)
            # This is a rough approximation
            residual = h3 - pred  # (n, hidden_dim)
            
            # Gradient w.r.t. U: d(loss)/dU ≈ -emb_A.T @ (residual @ W_out.T * B_proj)
            grad_U = -emb_A.T @ (residual @ W_out.T * B_proj) / n_samples
            grad_V = -emb_B.T @ (residual @ W_out.T * A_proj) / n_samples
            
            lr = 0.1
            U -= lr * grad_U
            V -= lr * grad_V
        
        return U, V, W_out, mean_cos
    
    def test_bilinear_generalization(self, U: np.ndarray, V: np.ndarray, W_out: np.ndarray, n_test: int = 100):
        """Test if the learned bilinear form generalizes."""
        print(f"\n--- Testing Generalization ({n_test} pairs) ---")
        
        cos_sims = []
        correct = 0
        
        for i in range(n_test):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb_A = self.embeddings[A]
                emb_B = self.embeddings[B]
                h3_actual = self.get_layer3_output(A, B)
                
                # Predict
                A_proj = emb_A @ U
                B_proj = emb_B @ V
                interaction = A_proj * B_proj
                h3_pred = interaction @ W_out
                
                cos = np.dot(h3_actual, h3_pred) / (
                    np.linalg.norm(h3_actual) * np.linalg.norm(h3_pred) + 1e-10)
                cos_sims.append(cos)
                
                # Token prediction
                token_actual = np.argmax(self.lm_head @ h3_actual)
                token_pred = np.argmax(self.lm_head @ h3_pred)
                
                if token_actual == token_pred:
                    correct += 1
                    
            except:
                continue
        
        print(f"  Test cosine: {np.mean(cos_sims):.4f}")
        print(f"  Token accuracy: {correct}/{n_test} = {correct/n_test*100:.1f}%")
        
        return np.mean(cos_sims), correct / n_test
    
    def analyze_actual_mesh(self, n_samples: int = 200):
        """
        Analyze: What IS the actual MESH for layer 3?
        
        From Doc 129, MESH = W_q.T @ W_k for attention.
        
        For layer 3, the attention computes:
            attn = softmax(Q @ K.T / sqrt(d))
            output = attn @ V
        
        The MESH captures Q @ K.T. Let's extract it.
        """
        print(f"\n--- Analyzing Actual Layer 3 MESH ---")
        
        layer3 = self.model.model.layers[3]
        
        W_q = layer3.self_attn.q_proj.weight.data.float().cpu().numpy()
        W_k = layer3.self_attn.k_proj.weight.data.float().cpu().numpy()
        W_v = layer3.self_attn.v_proj.weight.data.float().cpu().numpy()
        W_o = layer3.self_attn.o_proj.weight.data.float().cpu().numpy()
        
        print(f"  W_q: {W_q.shape}")
        print(f"  W_k: {W_k.shape}")
        print(f"  W_v: {W_v.shape}")
        print(f"  W_o: {W_o.shape}")
        
        # MESH = W_q.T @ W_k
        # But W_q is (3584, 3584) and W_k is (512, 3584) due to GQA
        # So MESH = W_q.T @ W_k.T would be (3584, 512)
        
        # Actually, the attention score is:
        # score = (h @ W_q.T) @ (h @ W_k.T).T = h @ W_q.T @ W_k @ h.T
        # So MESH = W_q.T @ W_k (but dimensions don't match for GQA)
        
        # For GQA, we need to handle the head grouping
        n_heads = self.model.config.num_attention_heads
        n_kv_heads = self.model.config.num_key_value_heads
        head_dim = self.hidden_dim // n_heads
        
        print(f"  n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
        
        # Reshape W_q and W_k for per-head analysis
        W_q_heads = W_q.reshape(n_heads, head_dim, self.hidden_dim)  # (28, 128, 3584)
        W_k_heads = W_k.reshape(n_kv_heads, head_dim, self.hidden_dim)  # (4, 128, 3584)
        
        # For each head, MESH_h = W_q_h.T @ W_k_h
        # But Q heads share K heads in GQA
        heads_per_kv = n_heads // n_kv_heads
        
        print(f"\n  Computing per-head MESH...")
        
        mesh_list = []
        for h in range(n_heads):
            kv_idx = h // heads_per_kv
            # MESH_h = W_q_h.T @ W_k_kv
            # W_q_h: (128, 3584), W_k_kv: (128, 3584)
            # MESH_h = (3584, 128) @ (128, 3584) = (3584, 3584)
            mesh_h = W_q_heads[h].T @ W_k_heads[kv_idx]
            mesh_list.append(mesh_h)
        
        mesh_list = np.array(mesh_list)  # (28, 3584, 3584)
        
        print(f"  MESH shape: {mesh_list.shape}")
        
        # Analyze MESH structure
        # Average MESH across heads
        mesh_avg = mesh_list.mean(axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(mesh_avg, full_matrices=False)
        
        print(f"\n  Average MESH singular values:")
        for i in range(10):
            level = np.log(S[i]) / np.log(PHI)
            print(f"    S[{i}] = {S[i]:.4f} (φ^{level:.1f})")
        
        # Rank analysis
        total_var = (S**2).sum()
        cumvar = np.cumsum(S**2) / total_var
        
        for threshold in [0.5, 0.9, 0.99]:
            k = np.searchsorted(cumvar, threshold) + 1
            print(f"    {threshold*100:.0f}% variance: k={k}")
        
        return mesh_list, mesh_avg


def main():
    print("=" * 70)
    print("BILINEAR CONTEXT: THE MESH FOR (A,B) → h3")
    print("=" * 70)
    print("""
From Doc 129: MESH = W_q.T @ W_k captures Q-K relationship.

For context, we need:
    h3[d] = emb_A @ MESH_d @ emb_B.T

Can we learn or extract this MESH?
""")
    
    analyzer = BilinearContextAnalyzer()
    
    # 1. Collect data
    emb_A, emb_B, h3 = analyzer.collect_data(n_samples=500)
    
    # 2. Learn bilinear form
    for rank in [10, 50, 100]:
        U, V, W_out, train_cos = analyzer.learn_bilinear_form(emb_A, emb_B, h3, rank=rank)
        
        # 3. Test generalization
        test_cos, accuracy = analyzer.test_bilinear_generalization(U, V, W_out, n_test=100)
    
    # 4. Analyze actual MESH
    analyzer.analyze_actual_mesh()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
