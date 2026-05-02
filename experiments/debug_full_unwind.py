#!/usr/bin/env python3
"""
Debug Full Unwind: Why 10% token accuracy despite 99.6% layer cosines?
========================================================================

Layer-by-layer cosines are 0.996-0.999, but token prediction is only 10%.
Let's investigate:
1. Is the final layer norm correct?
2. Is there a precision issue accumulating?
3. What's causing the NaN?

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DebugUnwind:
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
        self.n_layers = self.model.config.num_hidden_layers
        self.heads_per_kv = self.n_heads // self.n_kv_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        self.final_ln_weight = self.model.model.norm.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def debug_single_pair(self, A: int, B: int):
        """Debug a single pair through the full pipeline."""
        print(f"\n--- Debugging pair (A={A}, B={B}) ---")
        
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        
        # Get all hidden states
        hidden_states = [h[0].float().cpu().numpy() for h in outputs.hidden_states]
        
        print(f"\n  Number of hidden states: {len(hidden_states)}")
        print(f"  Hidden state shapes: {[h.shape for h in hidden_states[:3]]}...")
        
        # Check final hidden state
        h_final = hidden_states[-1]  # After all layers + final norm
        h_last_layer = hidden_states[-2] if len(hidden_states) > 1 else hidden_states[-1]
        
        print(f"\n  Last layer output (before final norm):")
        print(f"    Shape: {h_last_layer.shape}")
        print(f"    Norm: {np.linalg.norm(h_last_layer[1]):.4f}")
        print(f"    Has NaN: {np.isnan(h_last_layer).any()}")
        
        # Apply final layer norm manually
        h_normed_manual = self.rms_norm(h_last_layer[1], self.final_ln_weight)
        
        print(f"\n  Final layer norm (manual):")
        print(f"    Norm: {np.linalg.norm(h_normed_manual):.4f}")
        print(f"    Has NaN: {np.isnan(h_normed_manual).any()}")
        
        # Get actual logits
        logits_actual = outputs.logits[0, 1].float().cpu().numpy()
        actual_token = np.argmax(logits_actual)
        
        print(f"\n  Actual logits:")
        print(f"    Shape: {logits_actual.shape}")
        print(f"    Top token: {actual_token} = '{self.tokenizer.decode([actual_token])}'")
        
        # Compute logits from manual norm
        logits_manual = self.lm_head @ h_normed_manual
        manual_token = np.argmax(logits_manual)
        
        print(f"\n  Manual logits (from final norm):")
        print(f"    Top token: {manual_token} = '{self.tokenizer.decode([manual_token])}'")
        print(f"    Match: {actual_token == manual_token}")
        
        # Compare logits
        logits_cos = np.dot(logits_actual, logits_manual) / (
            np.linalg.norm(logits_actual) * np.linalg.norm(logits_manual) + 1e-10)
        print(f"    Logits cosine: {logits_cos:.6f}")
        
        # Check if the issue is in hidden_states indexing
        print(f"\n  Hidden states structure:")
        print(f"    hidden_states[0] = embeddings")
        print(f"    hidden_states[1] = after layer 0")
        print(f"    ...")
        print(f"    hidden_states[{self.n_layers}] = after layer {self.n_layers-1}")
        print(f"    Total: {len(hidden_states)} states for {self.n_layers} layers")
        
        # The model's hidden_states[-1] is AFTER final norm, not before
        # Let's check
        print(f"\n  Checking if hidden_states[-1] is after final norm:")
        
        # Get the output before final norm by capturing it
        captured = {}
        def capture_norm_input(module, input, output):
            captured['norm_input'] = input[0].detach().float().cpu().numpy()
            captured['norm_output'] = output.detach().float().cpu().numpy()
        
        hook = self.model.model.norm.register_forward_hook(capture_norm_input)
        
        with torch.no_grad():
            self.model(input_ids, output_hidden_states=True)
        
        hook.remove()
        
        print(f"    Norm input shape: {captured['norm_input'].shape}")
        print(f"    Norm output shape: {captured['norm_output'].shape}")
        
        # Compare hidden_states[-1] with norm output
        cos_with_norm_output = np.dot(hidden_states[-1][1], captured['norm_output'][0, 1]) / (
            np.linalg.norm(hidden_states[-1][1]) * np.linalg.norm(captured['norm_output'][0, 1]) + 1e-10)
        
        print(f"    hidden_states[-1] vs norm_output cosine: {cos_with_norm_output:.6f}")
        
        # So hidden_states[-1] IS the norm output
        # That means we should NOT apply final norm again
        
        # Test: use hidden_states[-1] directly with lm_head
        logits_direct = self.lm_head @ hidden_states[-1][1]
        direct_token = np.argmax(logits_direct)
        
        print(f"\n  Direct logits (hidden_states[-1] @ lm_head):")
        print(f"    Top token: {direct_token} = '{self.tokenizer.decode([direct_token])}'")
        print(f"    Match with actual: {actual_token == direct_token}")
        
        return {
            'actual_token': actual_token,
            'manual_token': manual_token,
            'direct_token': direct_token,
            'logits_cos': logits_cos,
        }
    
    def test_token_prediction_fixed(self, n_samples: int = 50):
        """Test token prediction with correct hidden state handling."""
        print(f"\n--- Testing Token Prediction (Fixed) ({n_samples} pairs) ---")
        
        correct_direct = 0
        correct_manual = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                
                # Actual token
                actual_token = torch.argmax(outputs.logits[0, 1]).item()
                
                # Get hidden states
                hidden_states = outputs.hidden_states
                
                # Method 1: Direct (hidden_states[-1] is after final norm)
                h_final = hidden_states[-1][0, 1].float().cpu().numpy()
                logits_direct = self.lm_head @ h_final
                direct_token = np.argmax(logits_direct)
                
                if actual_token == direct_token:
                    correct_direct += 1
                
                # Method 2: Manual norm on hidden_states[-2] (before final norm)
                # Wait, there's no hidden_states[-2] that's "before norm"
                # hidden_states has n_layers+1 entries: embeddings + each layer output
                # The final norm is applied AFTER the last layer
                
                # Actually, let me check the structure more carefully
                h_before_norm = hidden_states[self.n_layers][0, 1].float().cpu().numpy()
                h_normed = self.rms_norm(h_before_norm, self.final_ln_weight)
                logits_manual = self.lm_head @ h_normed
                manual_token = np.argmax(logits_manual)
                
                if actual_token == manual_token:
                    correct_manual += 1
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Results:")
        print(f"    Direct (hidden_states[-1]): {correct_direct}/{n_samples} = {correct_direct/n_samples*100:.1f}%")
        print(f"    Manual norm: {correct_manual}/{n_samples} = {correct_manual/n_samples*100:.1f}%")


def main():
    print("=" * 70)
    print("DEBUG FULL UNWIND")
    print("=" * 70)
    
    debugger = DebugUnwind()
    
    # Debug a single pair
    debugger.debug_single_pair(100, 200)
    
    # Test with fixed approach
    debugger.test_token_prediction_fixed(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
