#!/usr/bin/env python3
"""
Inject Layer 3: Can We Replace Layer 3 Output?
================================================

Key finding: h3 = h2 + attn_output + mlp_output (perfect reconstruction)

New approach: Instead of manually computing layer 3, test if we can:
1. Capture the layer 3 output for a pair (A, B)
2. Inject it into the model at layer 4
3. Get the correct final token

This tests whether layer 3 is truly the "click point" and whether
we can precompute and cache it.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class InjectLayer3Analyzer:
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
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_hidden_at_layer(self, token_ids: List[int], layer: int) -> np.ndarray:
        """Get hidden states at specific layer."""
        input_ids = torch.tensor([token_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[layer][0].float().cpu().numpy()
    
    def run_from_layer(self, hidden_state: torch.Tensor, start_layer: int) -> torch.Tensor:
        """
        Run the model from a specific layer using a given hidden state.
        
        This allows us to inject a modified hidden state and see the result.
        """
        hidden = hidden_state.to(self.device).half()
        
        # Run through remaining layers
        for layer_idx in range(start_layer, self.n_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Create position_ids and attention_mask
            seq_len = hidden.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            
            # Create causal mask
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            
            # Run layer
            layer_output = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden = layer_output[0]
        
        # Apply final layer norm
        hidden = self.model.model.norm(hidden)
        
        return hidden.float().cpu()
    
    def test_layer_injection(self, n_samples: int = 50):
        """
        Test: If we inject the correct layer 3 output, do we get the correct final token?
        
        This validates that layer 3 is the critical point.
        """
        print(f"\n--- Testing Layer 3 Injection ({n_samples} pairs) ---")
        
        correct = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(self.device)
                
                # Get actual outputs
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    h_final_actual = outputs.hidden_states[-1][0, 1].float().cpu().numpy()
                    h3_actual = outputs.hidden_states[4][0].float().cpu()  # After layer 3
                
                true_token = np.argmax(self.lm_head.numpy() @ h_final_actual)
                
                # Inject h3_actual and run from layer 4
                h3_tensor = h3_actual.unsqueeze(0)  # (1, 2, hidden)
                h_final_injected = self.run_from_layer(h3_tensor, start_layer=4)
                h_final_injected = h_final_injected[0, 1].numpy()
                
                pred_token = np.argmax(self.lm_head.numpy() @ h_final_injected)
                
                if true_token == pred_token:
                    correct += 1
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        accuracy = correct / n_samples
        print(f"\n  Results:")
        print(f"    Injection accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        
        return accuracy
    
    def test_single_token_injection(self, n_samples: int = 50):
        """
        Test: If we use single-token layer 3 output for B, do we get correct token?
        
        This tests whether the single-token cache can be used after layer 3.
        """
        print(f"\n--- Testing Single-Token Layer 3 Injection ({n_samples} pairs) ---")
        
        correct = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get actual final hidden state for (A, B)
                input_ids_AB = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs_AB = self.model(input_ids_AB, output_hidden_states=True)
                    h_final_actual = outputs_AB.hidden_states[-1][0, 1].float().cpu().numpy()
                
                true_token = np.argmax(self.lm_head.numpy() @ h_final_actual)
                
                # Get single-token layer 3 outputs
                input_ids_A = torch.tensor([[A]]).to(self.device)
                input_ids_B = torch.tensor([[B]]).to(self.device)
                
                with torch.no_grad():
                    outputs_A = self.model(input_ids_A, output_hidden_states=True)
                    outputs_B = self.model(input_ids_B, output_hidden_states=True)
                    
                    h3_A = outputs_A.hidden_states[4][0, 0].float().cpu()  # Single A
                    h3_B = outputs_B.hidden_states[4][0, 0].float().cpu()  # Single B
                
                # Create a fake 2-token hidden state using single-token values
                h3_fake = torch.stack([h3_A, h3_B]).unsqueeze(0)  # (1, 2, hidden)
                
                # Run from layer 4
                h_final_fake = self.run_from_layer(h3_fake, start_layer=4)
                h_final_fake = h_final_fake[0, 1].numpy()
                
                pred_token = np.argmax(self.lm_head.numpy() @ h_final_fake)
                
                if true_token == pred_token:
                    correct += 1
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        accuracy = correct / n_samples
        print(f"\n  Results:")
        print(f"    Single-token injection accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        
        return accuracy
    
    def test_attention_weighted_injection(self, n_samples: int = 50):
        """
        Test: If we use attention-weighted layer 3 output, do we get correct token?
        
        h3_B_fake = attn_to_A * h3_A + attn_to_B * h3_B
        """
        print(f"\n--- Testing Attention-Weighted Layer 3 Injection ({n_samples} pairs) ---")
        
        correct = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids_AB = torch.tensor([[A, B]]).to(self.device)
                
                # Get actual outputs including attention
                with torch.no_grad():
                    outputs_AB = self.model(input_ids_AB, output_hidden_states=True, output_attentions=True)
                    h_final_actual = outputs_AB.hidden_states[-1][0, 1].float().cpu().numpy()
                    
                    # Get layer 3 attention weights
                    attn3 = outputs_AB.attentions[3][0, :, 1, :].mean(dim=0).cpu().numpy()  # (2,)
                    attn_to_A = attn3[0]
                    attn_to_B = attn3[1]
                
                true_token = np.argmax(self.lm_head.numpy() @ h_final_actual)
                
                # Get single-token layer 3 outputs
                with torch.no_grad():
                    outputs_A = self.model(torch.tensor([[A]]).to(self.device), output_hidden_states=True)
                    outputs_B = self.model(torch.tensor([[B]]).to(self.device), output_hidden_states=True)
                    
                    h3_A = outputs_A.hidden_states[4][0, 0].float().cpu()
                    h3_B = outputs_B.hidden_states[4][0, 0].float().cpu()
                
                # Attention-weighted combination
                h3_B_weighted = attn_to_A * h3_A + attn_to_B * h3_B
                
                # Create 2-token hidden state
                h3_fake = torch.stack([h3_A, h3_B_weighted]).unsqueeze(0)
                
                # Run from layer 4
                h_final_fake = self.run_from_layer(h3_fake, start_layer=4)
                h_final_fake = h_final_fake[0, 1].numpy()
                
                pred_token = np.argmax(self.lm_head.numpy() @ h_final_fake)
                
                if true_token == pred_token:
                    correct += 1
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        accuracy = correct / n_samples
        print(f"\n  Results:")
        print(f"    Attention-weighted injection accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        
        return accuracy


def main():
    print("=" * 70)
    print("INJECT LAYER 3")
    print("=" * 70)
    print("""
Testing whether layer 3 is the critical "click point":
1. Inject actual layer 3 output -> should get 100%
2. Inject single-token layer 3 output -> measures context importance
3. Inject attention-weighted layer 3 output -> tests if attention is the key
""")
    
    analyzer = InjectLayer3Analyzer()
    
    # 1. Test actual layer 3 injection
    acc1 = analyzer.test_layer_injection(n_samples=50)
    
    # 2. Test single-token injection
    acc2 = analyzer.test_single_token_injection(n_samples=50)
    
    # 3. Test attention-weighted injection
    acc3 = analyzer.test_attention_weighted_injection(n_samples=50)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Actual layer 3 injection: {acc1*100:.1f}%")
    print(f"  Single-token injection: {acc2*100:.1f}%")
    print(f"  Attention-weighted injection: {acc3*100:.1f}%")


if __name__ == "__main__":
    main()
