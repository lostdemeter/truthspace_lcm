#!/usr/bin/env python3
"""
Diagnose MESH Gap: What's the Missing 30%?
============================================

MESH-based computation gets 0.70 cosine with actual h3.
Let's isolate exactly where the gap comes from.

Candidates:
1. RoPE (Rotary Position Embeddings)
2. Precision (float16 vs float32)
3. Layer norm differences
4. Attention score computation
5. V projection
6. MLP computation

We'll test each component separately.

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


class DiagnoseMeshGapAnalyzer:
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
        
        # Extract layer 3 weights
        self.extract_layer3()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def extract_layer3(self):
        """Extract layer 3 weights."""
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
        
        self.W_q_heads = W_q_heads
        self.W_k_heads = W_k_heads
        self.W_v_heads = W_v_heads
        self.heads_per_kv = heads_per_kv
        
        self.mesh_qk = []
        for h in range(self.n_heads):
            kv_idx = h // heads_per_kv
            mesh_h = W_q_heads[h].T @ W_k_heads[kv_idx]
            self.mesh_qk.append(mesh_h)
        
        self.mesh_qk = np.array(self.mesh_qk)
        print(f"  Extracted MESH: {self.mesh_qk.shape}")
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def diagnose_single_pair(self, A: int, B: int):
        """
        Diagnose the gap for a single (A, B) pair.
        
        Compare each intermediate value between MESH computation and actual.
        """
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        layer3 = self.model.model.layers[3]
        
        # Capture intermediate values with hooks
        captured = {}
        
        def capture_ln_input(module, input, output):
            captured['ln_input'] = input[0].detach().float().cpu().numpy()
            captured['ln_output'] = output.detach().float().cpu().numpy()
        
        def capture_q(module, input, output):
            captured['q_output'] = output.detach().float().cpu().numpy()
        
        def capture_k(module, input, output):
            captured['k_output'] = output.detach().float().cpu().numpy()
        
        def capture_v(module, input, output):
            captured['v_output'] = output.detach().float().cpu().numpy()
        
        def capture_attn(module, input, output):
            captured['attn_output'] = output[0].detach().float().cpu().numpy()
            if len(output) > 1 and output[1] is not None:
                captured['attn_weights'] = output[1].detach().float().cpu().numpy()
        
        def capture_mlp(module, input, output):
            captured['mlp_input'] = input[0].detach().float().cpu().numpy()
            captured['mlp_output'] = output.detach().float().cpu().numpy()
        
        # Register hooks
        hooks = [
            layer3.input_layernorm.register_forward_hook(capture_ln_input),
            layer3.self_attn.q_proj.register_forward_hook(capture_q),
            layer3.self_attn.k_proj.register_forward_hook(capture_k),
            layer3.self_attn.v_proj.register_forward_hook(capture_v),
            layer3.self_attn.register_forward_hook(capture_attn),
            layer3.mlp.register_forward_hook(capture_mlp),
        ]
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
            
            # Get hidden states
            h2 = outputs.hidden_states[3][0].float().cpu().numpy()  # Input to layer 3
            h3_actual = outputs.hidden_states[4][0].float().cpu().numpy()  # Output of layer 3
            
            # Get attention weights
            attn_weights_actual = outputs.attentions[3][0].float().cpu().numpy()  # (heads, seq, seq)
            
            h2_A, h2_B = h2[0], h2[1]
            
            results = {}
            
            # 1. Layer norm comparison
            ln_output_actual = captured['ln_output'][0]  # (2, hidden)
            ln_output_manual_A = self.rms_norm(h2_A, self.ln_weight)
            ln_output_manual_B = self.rms_norm(h2_B, self.ln_weight)
            
            cos_ln_A = np.dot(ln_output_actual[0], ln_output_manual_A) / (
                np.linalg.norm(ln_output_actual[0]) * np.linalg.norm(ln_output_manual_A) + 1e-10)
            cos_ln_B = np.dot(ln_output_actual[1], ln_output_manual_B) / (
                np.linalg.norm(ln_output_actual[1]) * np.linalg.norm(ln_output_manual_B) + 1e-10)
            
            results['layer_norm'] = (cos_ln_A, cos_ln_B)
            
            # 2. Q, K, V projection comparison
            q_actual = captured['q_output'][0]  # (2, q_dim)
            k_actual = captured['k_output'][0]  # (2, k_dim)
            v_actual = captured['v_output'][0]  # (2, v_dim)
            
            q_manual_B = ln_output_manual_B @ self.W_q.T
            k_manual_A = ln_output_manual_A @ self.W_k.T
            k_manual_B = ln_output_manual_B @ self.W_k.T
            v_manual_A = ln_output_manual_A @ self.W_v.T
            v_manual_B = ln_output_manual_B @ self.W_v.T
            
            cos_q = np.dot(q_actual[1], q_manual_B) / (
                np.linalg.norm(q_actual[1]) * np.linalg.norm(q_manual_B) + 1e-10)
            cos_k_A = np.dot(k_actual[0], k_manual_A) / (
                np.linalg.norm(k_actual[0]) * np.linalg.norm(k_manual_A) + 1e-10)
            cos_k_B = np.dot(k_actual[1], k_manual_B) / (
                np.linalg.norm(k_actual[1]) * np.linalg.norm(k_manual_B) + 1e-10)
            
            results['q_projection'] = cos_q
            results['k_projection'] = (cos_k_A, cos_k_B)
            
            # 3. Attention scores comparison (per head)
            # Actual attention weights from B to A
            attn_to_A_actual = attn_weights_actual[:, 1, 0]  # (heads,)
            
            # MESH-based attention scores (without RoPE)
            attn_to_A_mesh = []
            for h in range(self.n_heads):
                score_to_A = ln_output_manual_B @ self.mesh_qk[h] @ ln_output_manual_A / np.sqrt(self.head_dim)
                score_to_B = ln_output_manual_B @ self.mesh_qk[h] @ ln_output_manual_B / np.sqrt(self.head_dim)
                scores = np.array([score_to_A, score_to_B])
                exp_scores = np.exp(scores - scores.max())
                attn = exp_scores / exp_scores.sum()
                attn_to_A_mesh.append(attn[0])
            
            attn_to_A_mesh = np.array(attn_to_A_mesh)
            
            # Compare attention weights
            attn_corr = np.corrcoef(attn_to_A_actual, attn_to_A_mesh)[0, 1]
            attn_mae = np.abs(attn_to_A_actual - attn_to_A_mesh).mean()
            
            results['attention_correlation'] = attn_corr
            results['attention_mae'] = attn_mae
            results['attention_actual'] = attn_to_A_actual
            results['attention_mesh'] = attn_to_A_mesh
            
            # 4. Attention output comparison
            attn_output_actual = captured['attn_output'][0, 1]  # (hidden,)
            
            # Manual attention output using MESH
            attn_output_mesh = np.zeros(self.hidden_dim)
            for h in range(self.n_heads):
                kv_idx = h // self.heads_per_kv
                
                # V values
                v_A = ln_output_manual_A @ self.W_v_heads[kv_idx].T
                v_B = ln_output_manual_B @ self.W_v_heads[kv_idx].T
                
                # Weighted sum using MESH attention
                v_out = attn_to_A_mesh[h] * v_A + (1 - attn_to_A_mesh[h]) * v_B
                attn_output_mesh[h * self.head_dim:(h+1) * self.head_dim] = v_out
            
            attn_output_mesh = attn_output_mesh @ self.W_o.T
            
            cos_attn_out = np.dot(attn_output_actual, attn_output_mesh) / (
                np.linalg.norm(attn_output_actual) * np.linalg.norm(attn_output_mesh) + 1e-10)
            
            results['attention_output'] = cos_attn_out
            
            # 5. What if we use ACTUAL attention weights but manual V?
            attn_output_hybrid = np.zeros(self.hidden_dim)
            for h in range(self.n_heads):
                kv_idx = h // self.heads_per_kv
                
                v_A = ln_output_manual_A @ self.W_v_heads[kv_idx].T
                v_B = ln_output_manual_B @ self.W_v_heads[kv_idx].T
                
                # Use ACTUAL attention weights
                v_out = attn_to_A_actual[h] * v_A + (1 - attn_to_A_actual[h]) * v_B
                attn_output_hybrid[h * self.head_dim:(h+1) * self.head_dim] = v_out
            
            attn_output_hybrid = attn_output_hybrid @ self.W_o.T
            
            cos_attn_out_hybrid = np.dot(attn_output_actual, attn_output_hybrid) / (
                np.linalg.norm(attn_output_actual) * np.linalg.norm(attn_output_hybrid) + 1e-10)
            
            results['attention_output_with_actual_weights'] = cos_attn_out_hybrid
            
            # 6. MLP comparison
            mlp_input_actual = captured['mlp_input'][0, 1]
            mlp_output_actual = captured['mlp_output'][0, 1]
            
            # Manual MLP using MESH attention output
            h3_pre_mlp_mesh = h2_B + attn_output_mesh
            h3_norm_mesh = self.rms_norm(h3_pre_mlp_mesh, self.ln_mlp_weight)
            
            gate = h3_norm_mesh @ self.W_gate.T
            up = h3_norm_mesh @ self.W_up.T
            mlp_output_mesh = (gate * (1 / (1 + np.exp(-np.clip(gate, -20, 20))))) * up
            mlp_output_mesh = mlp_output_mesh @ self.W_down.T
            
            cos_mlp = np.dot(mlp_output_actual, mlp_output_mesh) / (
                np.linalg.norm(mlp_output_actual) * np.linalg.norm(mlp_output_mesh) + 1e-10)
            
            results['mlp_output'] = cos_mlp
            
            # 7. Final h3 comparison
            h3_mesh = h3_pre_mlp_mesh + mlp_output_mesh
            
            cos_h3 = np.dot(h3_actual[1], h3_mesh) / (
                np.linalg.norm(h3_actual[1]) * np.linalg.norm(h3_mesh) + 1e-10)
            
            results['h3_final'] = cos_h3
            
            return results
            
        finally:
            for hook in hooks:
                hook.remove()
    
    def run_diagnosis(self, n_samples: int = 50):
        """Run diagnosis on multiple pairs."""
        print(f"\n--- Running Diagnosis ({n_samples} pairs) ---")
        
        all_results = {
            'layer_norm_A': [],
            'layer_norm_B': [],
            'q_projection': [],
            'k_projection_A': [],
            'k_projection_B': [],
            'attention_correlation': [],
            'attention_mae': [],
            'attention_output': [],
            'attention_output_with_actual_weights': [],
            'mlp_output': [],
            'h3_final': [],
        }
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                results = self.diagnose_single_pair(A, B)
                
                all_results['layer_norm_A'].append(results['layer_norm'][0])
                all_results['layer_norm_B'].append(results['layer_norm'][1])
                all_results['q_projection'].append(results['q_projection'])
                all_results['k_projection_A'].append(results['k_projection'][0])
                all_results['k_projection_B'].append(results['k_projection'][1])
                all_results['attention_correlation'].append(results['attention_correlation'])
                all_results['attention_mae'].append(results['attention_mae'])
                all_results['attention_output'].append(results['attention_output'])
                all_results['attention_output_with_actual_weights'].append(results['attention_output_with_actual_weights'])
                all_results['mlp_output'].append(results['mlp_output'])
                all_results['h3_final'].append(results['h3_final'])
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n" + "=" * 60)
        print("DIAGNOSIS RESULTS")
        print("=" * 60)
        
        print(f"\n  Component-by-component cosine similarity:")
        print(f"    Layer norm (A):     {np.mean(all_results['layer_norm_A']):.4f}")
        print(f"    Layer norm (B):     {np.mean(all_results['layer_norm_B']):.4f}")
        print(f"    Q projection:       {np.mean(all_results['q_projection']):.4f}")
        print(f"    K projection (A):   {np.mean(all_results['k_projection_A']):.4f}")
        print(f"    K projection (B):   {np.mean(all_results['k_projection_B']):.4f}")
        
        print(f"\n  Attention comparison:")
        print(f"    Attention weight correlation: {np.mean(all_results['attention_correlation']):.4f}")
        print(f"    Attention weight MAE:         {np.mean(all_results['attention_mae']):.4f}")
        print(f"    Attention output (MESH):      {np.mean(all_results['attention_output']):.4f}")
        print(f"    Attention output (actual wts):{np.mean(all_results['attention_output_with_actual_weights']):.4f}")
        
        print(f"\n  Final outputs:")
        print(f"    MLP output:         {np.mean(all_results['mlp_output']):.4f}")
        print(f"    h3 final:           {np.mean(all_results['h3_final']):.4f}")
        
        # Identify the gap
        print(f"\n" + "=" * 60)
        print("GAP ANALYSIS")
        print("=" * 60)
        
        if np.mean(all_results['q_projection']) < 0.99:
            print(f"  ⚠️  Q projection has gap: {1 - np.mean(all_results['q_projection']):.4f}")
            print(f"      This is likely due to RoPE (position embeddings)")
        
        if np.mean(all_results['attention_correlation']) < 0.99:
            print(f"  ⚠️  Attention weights have gap: correlation = {np.mean(all_results['attention_correlation']):.4f}")
            print(f"      This is the main source of error!")
        
        if np.mean(all_results['attention_output_with_actual_weights']) > np.mean(all_results['attention_output']) + 0.1:
            print(f"  ✓  Using actual attention weights improves output significantly")
            print(f"      MESH: {np.mean(all_results['attention_output']):.4f} → Actual: {np.mean(all_results['attention_output_with_actual_weights']):.4f}")
            print(f"      The gap is in ATTENTION WEIGHT COMPUTATION (likely RoPE)")
        
        return all_results


def main():
    print("=" * 70)
    print("DIAGNOSE MESH GAP: WHAT'S THE MISSING 30%?")
    print("=" * 70)
    print("""
MESH-based computation gets 0.70 cosine with actual h3.
Let's isolate exactly where the gap comes from.

Candidates:
1. RoPE (Rotary Position Embeddings)
2. Precision (float16 vs float32)
3. Layer norm differences
4. Attention score computation
5. V projection
6. MLP computation
""")
    
    analyzer = DiagnoseMeshGapAnalyzer()
    
    results = analyzer.run_diagnosis(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
