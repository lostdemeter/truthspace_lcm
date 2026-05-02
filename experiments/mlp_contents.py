#!/usr/bin/env python3
"""
MLP Contents: What's Inside the Safe?
======================================

Key finding: At layer 3, attention explains 0.65 cosine, but residual is 77% of signal.

The MLP is the "contents of the safe" - what gets revealed after the click.

Questions:
1. What does the MLP add at layer 3?
2. Is the MLP contribution predictable from the attention output?
3. Can we precompute MLP "contents" for common patterns?

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class MLPContentsAnalyzer:
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
        self.intermediate_dim = self.model.config.intermediate_size
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Intermediate dim: {self.intermediate_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_layer_internals(self, token_ids: List[int], layer_idx: int):
        """
        Get internal states from a specific layer.
        
        Returns attention output and MLP output separately.
        """
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([token_ids]).to(device)
        
        # We need to hook into the layer to get intermediate states
        layer = self.model.model.layers[layer_idx]
        
        attn_output = None
        mlp_output = None
        
        def attn_hook(module, input, output):
            nonlocal attn_output
            attn_output = output[0].detach().float().cpu().numpy()
        
        def mlp_hook(module, input, output):
            nonlocal mlp_output
            mlp_output = output.detach().float().cpu().numpy()
        
        # Register hooks
        attn_handle = layer.self_attn.register_forward_hook(attn_hook)
        mlp_handle = layer.mlp.register_forward_hook(mlp_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                
            # Get hidden states before and after this layer
            h_before = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
            h_after = outputs.hidden_states[layer_idx + 1][0].float().cpu().numpy()
            
            return {
                'h_before': h_before,
                'h_after': h_after,
                'attn_output': attn_output[0],  # Remove batch dim
                'mlp_output': mlp_output[0],
            }
        finally:
            attn_handle.remove()
            mlp_handle.remove()
    
    def analyze_layer3_decomposition(self, n_samples: int = 100):
        """
        Decompose layer 3 into attention and MLP contributions.
        
        Layer output = input + attention_output + mlp_output (with layer norms)
        """
        print(f"\n--- Layer 3 Decomposition ({n_samples} pairs) ---")
        
        attn_norms = []
        mlp_norms = []
        total_norms = []
        
        attn_contributions = []
        mlp_contributions = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                internals = self.get_layer_internals([A, B], 3)
                
                # Position B (index 1)
                attn_out = internals['attn_output'][1]
                mlp_out = internals['mlp_output'][1]
                h_before = internals['h_before'][1]
                h_after = internals['h_after'][1]
                
                # Norms
                attn_norms.append(np.linalg.norm(attn_out))
                mlp_norms.append(np.linalg.norm(mlp_out))
                total_norms.append(np.linalg.norm(h_after - h_before))
                
                # Contribution to final output direction
                delta = h_after - h_before
                attn_contrib = np.dot(attn_out, delta) / (np.linalg.norm(delta)**2 + 1e-10)
                mlp_contrib = np.dot(mlp_out, delta) / (np.linalg.norm(delta)**2 + 1e-10)
                
                attn_contributions.append(attn_contrib)
                mlp_contributions.append(mlp_contrib)
            except Exception as e:
                continue
        
        print(f"\n  Norm analysis:")
        print(f"    Attention output norm: {np.mean(attn_norms):.1f}")
        print(f"    MLP output norm: {np.mean(mlp_norms):.1f}")
        print(f"    Total delta norm: {np.mean(total_norms):.1f}")
        
        print(f"\n  Contribution to delta:")
        print(f"    Attention: {np.mean(attn_contributions):.3f}")
        print(f"    MLP: {np.mean(mlp_contributions):.3f}")
        
        return {
            'attn_norms': attn_norms,
            'mlp_norms': mlp_norms,
            'attn_contributions': attn_contributions,
            'mlp_contributions': mlp_contributions,
        }
    
    def compare_single_vs_context_mlp(self, n_samples: int = 100):
        """
        Compare MLP output for single token vs token in context.
        
        Key question: Does the MLP output change with context?
        """
        print(f"\n--- Single vs Context MLP Comparison ---")
        
        mlp_cos_sims = []
        attn_cos_sims = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # B alone
                internals_alone = self.get_layer_internals([B], 3)
                mlp_alone = internals_alone['mlp_output'][0]
                attn_alone = internals_alone['attn_output'][0]
                
                # B in context
                internals_context = self.get_layer_internals([A, B], 3)
                mlp_context = internals_context['mlp_output'][1]
                attn_context = internals_context['attn_output'][1]
                
                # Cosine similarities
                mlp_cos = np.dot(mlp_alone, mlp_context) / (
                    np.linalg.norm(mlp_alone) * np.linalg.norm(mlp_context) + 1e-10)
                attn_cos = np.dot(attn_alone, attn_context) / (
                    np.linalg.norm(attn_alone) * np.linalg.norm(attn_context) + 1e-10)
                
                mlp_cos_sims.append(mlp_cos)
                attn_cos_sims.append(attn_cos)
            except:
                continue
        
        print(f"\n  Cosine similarity (single vs context):")
        print(f"    Attention output: {np.mean(attn_cos_sims):.4f}")
        print(f"    MLP output: {np.mean(mlp_cos_sims):.4f}")
        
        return {
            'mlp_cos_sims': mlp_cos_sims,
            'attn_cos_sims': attn_cos_sims,
        }
    
    def analyze_mlp_input_dependence(self, n_samples: int = 100):
        """
        Analyze how MLP output depends on its input.
        
        MLP input = h + attn_output (after layer norm)
        
        Key question: Is MLP output predictable from MLP input?
        """
        print(f"\n--- MLP Input Dependence Analysis ---")
        
        # Collect MLP inputs and outputs
        mlp_inputs = []
        mlp_outputs = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                internals = self.get_layer_internals([A, B], 3)
                
                # MLP input is approximately h_before + attn_output (ignoring layer norm)
                mlp_in = internals['h_before'][1] + internals['attn_output'][1]
                mlp_out = internals['mlp_output'][1]
                
                mlp_inputs.append(mlp_in)
                mlp_outputs.append(mlp_out)
            except:
                continue
        
        mlp_inputs = np.array(mlp_inputs)
        mlp_outputs = np.array(mlp_outputs)
        
        # Test: Can we predict MLP output from input?
        # MLP is: output = W_down @ SiLU(W_gate @ input) * (W_up @ input)
        # This is nonlinear, but maybe low-rank?
        
        print(f"\n  Testing linear approximation of MLP:")
        
        # SVD of input-output relationship
        # Center the data
        mlp_in_centered = mlp_inputs - mlp_inputs.mean(axis=0)
        mlp_out_centered = mlp_outputs - mlp_outputs.mean(axis=0)
        
        for k in [10, 50, 100, 200]:
            # Project input to k dimensions
            _, _, Vt_in = np.linalg.svd(mlp_in_centered, full_matrices=False)
            mlp_in_k = mlp_in_centered @ Vt_in[:k].T
            
            # Solve for W: mlp_out ≈ mlp_in_k @ W
            W, _, _, _ = np.linalg.lstsq(mlp_in_k, mlp_out_centered, rcond=None)
            
            pred_out = mlp_in_k @ W + mlp_outputs.mean(axis=0)
            
            # Cosine similarity
            cos_sims = [np.dot(pred_out[i], mlp_outputs[i]) / 
                       (np.linalg.norm(pred_out[i]) * np.linalg.norm(mlp_outputs[i]) + 1e-10)
                       for i in range(len(mlp_outputs))]
            
            print(f"    k={k}: mean cosine = {np.mean(cos_sims):.4f}")
        
        return {
            'mlp_inputs': mlp_inputs,
            'mlp_outputs': mlp_outputs,
        }


def main():
    print("=" * 70)
    print("MLP CONTENTS: WHAT'S INSIDE THE SAFE?")
    print("=" * 70)
    print("""
The attention determines WHICH safe opens (the click).
The MLP determines WHAT'S INSIDE (the contents).

Key question: Can we precompute the MLP "contents"?
""")
    
    analyzer = MLPContentsAnalyzer()
    
    # 1. Decompose layer 3
    decomp_results = analyzer.analyze_layer3_decomposition(n_samples=100)
    
    # 2. Compare single vs context MLP
    compare_results = analyzer.compare_single_vs_context_mlp(n_samples=100)
    
    # 3. Analyze MLP input dependence
    mlp_results = analyzer.analyze_mlp_input_dependence(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
