#!/usr/bin/env python3
"""
Layer-wise Context Analysis: Where Does Context Matter?
=========================================================

Key finding: Attention-weighted combination of final hidden states = 0% accuracy.

The "shape change" isn't just in attention weights - it's in how V vectors
get transformed differently when there's context.

New hypothesis: Maybe context only matters at CERTAIN layers.
If we can identify which layers are "context-sensitive", we can:
1. Use cached single-token states for context-insensitive layers
2. Only compute context-sensitive layers

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class LayerwiseContextAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        # Load with eager attention
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
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_all_hidden_states(self, token_ids: List[int], position: int = -1) -> List[np.ndarray]:
        """Get hidden states at all layers for a specific position."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0, position, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def analyze_layer_context_sensitivity(self, n_samples: int = 100):
        """
        For each layer, measure how much the hidden state changes with context.
        
        Compare: h_layer(B alone) vs h_layer(B in context of A)
        """
        print(f"\n--- Layer Context Sensitivity ({n_samples} pairs) ---")
        
        # For each layer, collect cosine similarities
        layer_cos_sims = {l: [] for l in range(self.n_layers + 1)}
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # B alone
                h_B_alone = self.get_all_hidden_states([B], position=0)
                
                # B in context of A
                h_B_context = self.get_all_hidden_states([A, B], position=1)
                
                for l in range(self.n_layers + 1):
                    cos = np.dot(h_B_alone[l], h_B_context[l]) / (
                        np.linalg.norm(h_B_alone[l]) * np.linalg.norm(h_B_context[l]) + 1e-10)
                    layer_cos_sims[l].append(cos)
            except:
                continue
        
        print(f"\n  Layer-by-layer cosine similarity (B alone vs B in context):")
        print(f"  (Higher = less context-sensitive)")
        
        for l in range(self.n_layers + 1):
            mean_cos = np.mean(layer_cos_sims[l])
            std_cos = np.std(layer_cos_sims[l])
            layer_name = "embed" if l == 0 else f"layer {l-1}"
            print(f"    {layer_name}: {mean_cos:.4f} ± {std_cos:.4f}")
        
        return layer_cos_sims
    
    def find_context_injection_point(self, n_samples: int = 50):
        """
        Find where context gets "injected" into the hidden state.
        
        At some layer, the hidden state diverges from the single-token path.
        """
        print(f"\n--- Finding Context Injection Point ---")
        
        # Track where the divergence happens
        divergence_points = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B_alone = self.get_all_hidden_states([B], position=0)
                h_B_context = self.get_all_hidden_states([A, B], position=1)
                
                # Find first layer where cosine similarity drops below 0.9
                for l in range(self.n_layers + 1):
                    cos = np.dot(h_B_alone[l], h_B_context[l]) / (
                        np.linalg.norm(h_B_alone[l]) * np.linalg.norm(h_B_context[l]) + 1e-10)
                    if cos < 0.9:
                        divergence_points.append(l)
                        break
                else:
                    divergence_points.append(self.n_layers + 1)
            except:
                continue
        
        print(f"\n  Divergence point distribution:")
        for l in range(self.n_layers + 2):
            count = divergence_points.count(l)
            if count > 0:
                layer_name = "embed" if l == 0 else f"layer {l-1}" if l <= self.n_layers else "never"
                print(f"    {layer_name}: {count} samples ({count/len(divergence_points)*100:.1f}%)")
        
        return divergence_points
    
    def test_hybrid_approach(self, n_samples: int = 50):
        """
        Test: Use single-token cache up to layer X, then compute with context.
        
        If context only matters after layer X, we can skip layers 0 to X-1.
        """
        print(f"\n--- Hybrid Approach Test ---")
        
        # For each "switch point", test accuracy
        for switch_layer in [0, 1, 5, 10, 15, 20, 25]:
            correct = 0
            
            for i in range(n_samples):
                A = np.random.randint(0, self.tokenizer.vocab_size)
                B = np.random.randint(0, self.tokenizer.vocab_size)
                
                try:
                    # Get true final hidden state
                    h_true = self.get_all_hidden_states([A, B], position=1)[-1]
                    
                    # Get single-token hidden state at switch_layer
                    h_B_alone = self.get_all_hidden_states([B], position=0)
                    
                    # The hybrid approach would:
                    # 1. Use h_B_alone[switch_layer] as starting point
                    # 2. Run layers switch_layer to end with context
                    
                    # For now, just measure how close h_B_alone[switch_layer] is to h_context[switch_layer]
                    h_B_context = self.get_all_hidden_states([A, B], position=1)
                    
                    # If we could substitute h_B_alone[switch_layer] for h_B_context[switch_layer],
                    # would the final output be correct?
                    
                    # This is hard to test without modifying the model internals.
                    # Instead, let's measure the error at each layer.
                    
                    true_token = np.argmax(self.lm_head @ h_true)
                    
                    # Use single-token final hidden state as baseline
                    h_B_final_alone = h_B_alone[-1]
                    pred_token = np.argmax(self.lm_head @ h_B_final_alone)
                    
                    if true_token == pred_token:
                        correct += 1
                except:
                    continue
            
            # This just measures single-token accuracy, not hybrid
            # We need a different approach
            
        print("  (Need to implement actual hybrid computation)")
    
    def analyze_attention_contribution(self, n_samples: int = 50):
        """
        Analyze how much each layer's attention contributes to the final output.
        
        Key question: Which layers have the most "context-dependent" attention?
        """
        print(f"\n--- Attention Contribution Analysis ---")
        
        device = next(self.model.parameters()).device
        
        # For each layer, measure attention entropy and attention to first token
        layer_attn_to_first = {l: [] for l in range(self.n_layers)}
        layer_attn_entropy = {l: [] for l in range(self.n_layers)}
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            input_ids = torch.tensor([[A, B]]).to(device)
            
            try:
                with torch.no_grad():
                    outputs = self.model(input_ids, output_attentions=True)
                    
                    for l in range(self.n_layers):
                        # Attention from position 1 to all positions, averaged across heads
                        attn = outputs.attentions[l][0, :, 1, :].mean(dim=0).cpu().numpy()  # (2,)
                        
                        attn_to_first = attn[0]
                        
                        # Entropy of attention distribution
                        attn_clipped = np.clip(attn, 1e-10, 1.0)
                        entropy = -np.sum(attn_clipped * np.log(attn_clipped))
                        
                        layer_attn_to_first[l].append(attn_to_first)
                        layer_attn_entropy[l].append(entropy)
            except:
                continue
        
        print(f"\n  Per-layer attention statistics:")
        print(f"  Layer | Attn to A | Entropy")
        print(f"  ------|-----------|--------")
        
        for l in range(self.n_layers):
            mean_attn = np.mean(layer_attn_to_first[l])
            mean_entropy = np.mean(layer_attn_entropy[l])
            print(f"  {l:5d} | {mean_attn:.4f}    | {mean_entropy:.4f}")
        
        return {
            'attn_to_first': layer_attn_to_first,
            'attn_entropy': layer_attn_entropy,
        }


def main():
    print("=" * 70)
    print("LAYER-WISE CONTEXT ANALYSIS")
    print("=" * 70)
    print("""
Question: Where does context matter in the transformer?

If context only affects certain layers, we can:
1. Cache single-token states for context-insensitive layers
2. Only compute context-sensitive layers
""")
    
    analyzer = LayerwiseContextAnalyzer()
    
    # 1. Layer context sensitivity
    cos_sims = analyzer.analyze_layer_context_sensitivity(n_samples=100)
    
    # 2. Find context injection point
    divergence = analyzer.find_context_injection_point(n_samples=50)
    
    # 3. Attention contribution
    attn_results = analyzer.analyze_attention_contribution(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
