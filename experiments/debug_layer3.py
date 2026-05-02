#!/usr/bin/env python3
"""
Debug Layer 3: Why doesn't manual computation match?
=====================================================

Manual layer 3 gets 0.70 cosine even with RoPE.
Let's debug step by step to find the discrepancy.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DebugLayer3Analyzer:
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
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def debug_single_pair(self, A: int, B: int):
        """Debug layer 3 computation for a single pair."""
        print(f"\n--- Debugging pair A={A}, B={B} ---")
        
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        layer3 = self.model.model.layers[3]
        
        # Capture intermediate values
        captured = {}
        
        def capture_input_ln(module, input, output):
            captured['input_ln_output'] = output.detach().float().cpu().numpy()
        
        def capture_attn_output(module, input, output):
            captured['attn_output'] = output[0].detach().float().cpu().numpy()
            if len(output) > 1 and output[1] is not None:
                captured['attn_weights'] = output[1].detach().float().cpu().numpy()
        
        def capture_mlp_output(module, input, output):
            captured['mlp_input'] = input[0].detach().float().cpu().numpy()
            captured['mlp_output'] = output.detach().float().cpu().numpy()
        
        # Register hooks
        h1 = layer3.input_layernorm.register_forward_hook(capture_input_ln)
        h2 = layer3.self_attn.register_forward_hook(capture_attn_output)
        h3 = layer3.mlp.register_forward_hook(capture_mlp_output)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
            
            # Get hidden states
            h2_input = outputs.hidden_states[3][0].float().cpu().numpy()  # Input to layer 3
            h3_output = outputs.hidden_states[4][0].float().cpu().numpy()  # Output of layer 3
            
            # Get attention weights from layer 3
            attn_weights = outputs.attentions[3][0].float().cpu().numpy()  # (heads, seq, seq)
            
            print(f"\n  Hidden state shapes:")
            print(f"    h2_input (layer 3 input): {h2_input.shape}")
            print(f"    h3_output (layer 3 output): {h3_output.shape}")
            
            print(f"\n  Captured intermediate values:")
            for key, val in captured.items():
                print(f"    {key}: {val.shape}")
            
            print(f"\n  Attention weights at layer 3:")
            print(f"    Shape: {attn_weights.shape}")
            print(f"    From B to A (mean across heads): {attn_weights[:, 1, 0].mean():.4f}")
            print(f"    From B to B (mean across heads): {attn_weights[:, 1, 1].mean():.4f}")
            
            # Now let's manually compute and compare step by step
            print(f"\n  Step-by-step comparison:")
            
            # Step 1: Input layer norm
            ln_weight = layer3.input_layernorm.weight.data.float().cpu().numpy()
            h2_B = h2_input[1]
            
            # Manual RMS norm
            rms = np.sqrt(np.mean(h2_B**2) + 1e-6)
            h2_B_normed_manual = (h2_B / rms) * ln_weight
            
            # Actual from hook
            h2_B_normed_actual = captured['input_ln_output'][0, 1]
            
            cos_ln = np.dot(h2_B_normed_manual, h2_B_normed_actual) / (
                np.linalg.norm(h2_B_normed_manual) * np.linalg.norm(h2_B_normed_actual) + 1e-10)
            print(f"    Layer norm cosine: {cos_ln:.6f}")
            
            # Step 2: Attention output
            attn_out_actual = captured['attn_output'][0, 1]
            print(f"    Attention output norm: {np.linalg.norm(attn_out_actual):.2f}")
            
            # Step 3: MLP
            mlp_in_actual = captured['mlp_input'][0, 1]
            mlp_out_actual = captured['mlp_output'][0, 1]
            print(f"    MLP input norm: {np.linalg.norm(mlp_in_actual):.2f}")
            print(f"    MLP output norm: {np.linalg.norm(mlp_out_actual):.2f}")
            
            # Step 4: Final output
            # h3 = h2 + attn_out + mlp_out (with layer norms in between)
            h3_B_actual = h3_output[1]
            
            # Reconstruct
            h3_B_reconstructed = h2_B + attn_out_actual + mlp_out_actual
            
            cos_final = np.dot(h3_B_reconstructed, h3_B_actual) / (
                np.linalg.norm(h3_B_reconstructed) * np.linalg.norm(h3_B_actual) + 1e-10)
            print(f"    Reconstructed h3 cosine: {cos_final:.6f}")
            
            return {
                'h2_input': h2_input,
                'h3_output': h3_output,
                'attn_weights': attn_weights,
                'captured': captured,
            }
            
        finally:
            h1.remove()
            h2.remove()
            h3.remove()
    
    def test_using_hooks(self, n_samples: int = 50):
        """
        Test: If we use the actual intermediate values, can we reconstruct h3?
        
        This isolates where the discrepancy is.
        """
        print(f"\n--- Testing Reconstruction with Actual Intermediates ({n_samples} pairs) ---")
        
        cos_sims = []
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                result = self.debug_single_pair(A, B) if i == 0 else None
                
                # Quick test
                input_ids = torch.tensor([[A, B]]).to(self.device)
                
                layer3 = self.model.model.layers[3]
                captured = {}
                
                def capture_attn(module, input, output):
                    captured['attn'] = output[0].detach()
                
                def capture_mlp(module, input, output):
                    captured['mlp'] = output.detach()
                
                h1 = layer3.self_attn.register_forward_hook(capture_attn)
                h2 = layer3.mlp.register_forward_hook(capture_mlp)
                
                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids, output_hidden_states=True)
                    
                    h2_input = outputs.hidden_states[3][0, 1].float().cpu().numpy()
                    h3_actual = outputs.hidden_states[4][0, 1].float().cpu().numpy()
                    
                    attn_out = captured['attn'][0, 1].float().cpu().numpy()
                    mlp_out = captured['mlp'][0, 1].float().cpu().numpy()
                    
                    # Reconstruct
                    h3_reconstructed = h2_input + attn_out + mlp_out
                    
                    cos = np.dot(h3_reconstructed, h3_actual) / (
                        np.linalg.norm(h3_reconstructed) * np.linalg.norm(h3_actual) + 1e-10)
                    cos_sims.append(cos)
                    
                finally:
                    h1.remove()
                    h2.remove()
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Results:")
        print(f"    Mean cosine (reconstructed vs actual): {np.mean(cos_sims):.6f}")
        print(f"    Min: {np.min(cos_sims):.6f}, Max: {np.max(cos_sims):.6f}")
        
        return cos_sims


def main():
    print("=" * 70)
    print("DEBUG LAYER 3")
    print("=" * 70)
    
    analyzer = DebugLayer3Analyzer()
    
    # 1. Debug a single pair
    result = analyzer.debug_single_pair(100, 200)
    
    # 2. Test reconstruction with actual intermediates
    cos_sims = analyzer.test_using_hooks(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
