#!/usr/bin/env python3
"""
Rank-1 Debug: Why is accuracy so low?
======================================

The 91% reconstruction accuracy doesn't translate to token accuracy.
Let's understand why.

Hypothesis:
1. Small errors compound across 28 layers
2. The final hidden state is far from the true one
3. Even small deviations change the argmax token

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/thorin/truthspace-lcm/cache/rank1"


class Rank1Debugger:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        # Load precomputed data
        self.directions = np.load(os.path.join(CACHE_DIR, "universal_directions.npy"))
        with open(os.path.join(CACHE_DIR, "token_scales.pkl"), "rb") as f:
            self.token_scales = pickle.load(f)
        
        print(f"  Loaded {len(self.token_scales)} token scales")
    
    def get_layer_outputs(self, token_ids):
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def debug_single_token(self, token_id: int):
        """Debug rank-1 reconstruction for a single token."""
        print(f"\n--- Debugging token {token_id} ---")
        print(f"  Token: '{self.tokenizer.decode([token_id])}'")
        
        if token_id not in self.token_scales:
            print(f"  Token not in cache!")
            return
        
        # Get true outputs
        outputs = self.get_layer_outputs([token_id])
        
        # Reconstruct layer by layer
        hidden = outputs[0].copy()  # Start from embedding
        
        print(f"\n  Layer-by-layer analysis:")
        print(f"  {'Layer':<8} {'True Norm':<12} {'Rank1 Norm':<12} {'Error':<12} {'Cos Sim':<10}")
        print(f"  {'-'*54}")
        
        for layer_idx in range(self.n_layers):
            true_output = outputs[layer_idx + 1]
            
            # Apply rank-1 update
            scale = self.token_scales[token_id][layer_idx]
            direction = self.directions[layer_idx]
            hidden = hidden + scale * direction
            
            # Metrics
            true_norm = np.linalg.norm(true_output)
            rank1_norm = np.linalg.norm(hidden)
            error = np.linalg.norm(true_output - hidden)
            cos_sim = np.dot(true_output, hidden) / (true_norm * rank1_norm + 1e-10)
            
            print(f"  {layer_idx:<8} {true_norm:<12.1f} {rank1_norm:<12.1f} {error:<12.1f} {cos_sim:<10.4f}")
        
        # Final token prediction
        true_final = outputs[-1]
        
        true_logits = np.dot(self.lm_head, true_final)
        rank1_logits = np.dot(self.lm_head, hidden)
        
        true_token = np.argmax(true_logits)
        rank1_token = np.argmax(rank1_logits)
        
        print(f"\n  Final prediction:")
        print(f"    True token: {true_token} = '{self.tokenizer.decode([true_token])}'")
        print(f"    Rank1 token: {rank1_token} = '{self.tokenizer.decode([rank1_token])}'")
        print(f"    Match: {'✓' if true_token == rank1_token else '✗'}")
        
        # Logit analysis
        print(f"\n  Logit analysis:")
        print(f"    True logit for true token: {true_logits[true_token]:.2f}")
        print(f"    Rank1 logit for true token: {rank1_logits[true_token]:.2f}")
        print(f"    Rank1 logit for rank1 token: {rank1_logits[rank1_token]:.2f}")
        print(f"    Gap: {rank1_logits[rank1_token] - rank1_logits[true_token]:.2f}")
    
    def analyze_error_accumulation(self, n_samples: int = 20):
        """Analyze how errors accumulate across layers."""
        print(f"\n--- Error Accumulation Analysis ({n_samples} samples) ---")
        
        layer_errors = {i: [] for i in range(self.n_layers)}
        layer_cos_sims = {i: [] for i in range(self.n_layers)}
        
        sample_tokens = list(self.token_scales.keys())[:n_samples]
        
        for token_id in sample_tokens:
            outputs = self.get_layer_outputs([token_id])
            hidden = outputs[0].copy()
            
            for layer_idx in range(self.n_layers):
                true_output = outputs[layer_idx + 1]
                
                scale = self.token_scales[token_id][layer_idx]
                direction = self.directions[layer_idx]
                hidden = hidden + scale * direction
                
                error = np.linalg.norm(true_output - hidden) / np.linalg.norm(true_output)
                cos_sim = np.dot(true_output, hidden) / (np.linalg.norm(true_output) * np.linalg.norm(hidden) + 1e-10)
                
                layer_errors[layer_idx].append(error)
                layer_cos_sims[layer_idx].append(cos_sim)
        
        print(f"\n  {'Layer':<8} {'Mean Error':<12} {'Mean Cos Sim':<12}")
        print(f"  {'-'*32}")
        
        for layer_idx in range(self.n_layers):
            mean_error = np.mean(layer_errors[layer_idx])
            mean_cos = np.mean(layer_cos_sims[layer_idx])
            print(f"  {layer_idx:<8} {mean_error*100:.1f}%{'':<6} {mean_cos:.4f}")
    
    def test_hybrid_approach(self, n_tests: int = 50):
        """
        Test hybrid: use true outputs for some layers, rank-1 for others.
        
        Find the minimum number of true layers needed for good accuracy.
        """
        print(f"\n--- Hybrid Approach Test ---")
        
        sample_tokens = list(self.token_scales.keys())[:n_tests]
        
        # Test different numbers of true layers at the start
        for n_true_layers in [0, 3, 5, 10, 15, 20, 25, 27]:
            correct = 0
            
            for token_id in sample_tokens:
                outputs = self.get_layer_outputs([token_id])
                
                # Start from embedding
                hidden = outputs[0].copy()
                
                for layer_idx in range(self.n_layers):
                    if layer_idx < n_true_layers:
                        # Use true output
                        hidden = outputs[layer_idx + 1].copy()
                    else:
                        # Use rank-1
                        scale = self.token_scales[token_id][layer_idx]
                        direction = self.directions[layer_idx]
                        hidden = hidden + scale * direction
                
                # Predict
                true_logits = np.dot(self.lm_head, outputs[-1])
                rank1_logits = np.dot(self.lm_head, hidden)
                
                if np.argmax(true_logits) == np.argmax(rank1_logits):
                    correct += 1
            
            accuracy = correct / len(sample_tokens)
            print(f"  True layers 0-{n_true_layers-1 if n_true_layers > 0 else 'none'}, rank-1 rest: {accuracy*100:.1f}%")
    
    def test_per_token_scales(self, n_tests: int = 50):
        """
        What if we compute scales from the TRUE transformation (not universal direction)?
        
        This tests if the issue is the universal direction or something else.
        """
        print(f"\n--- Per-Token Scale Test ---")
        
        sample_tokens = list(self.token_scales.keys())[:n_tests]
        correct = 0
        
        for token_id in sample_tokens:
            outputs = self.get_layer_outputs([token_id])
            hidden = outputs[0].copy()
            
            for layer_idx in range(self.n_layers):
                true_transform = outputs[layer_idx + 1] - outputs[layer_idx]
                direction = self.directions[layer_idx]
                
                # Compute optimal scale for THIS token
                optimal_scale = np.dot(true_transform, direction)
                
                hidden = hidden + optimal_scale * direction
            
            # Predict
            true_logits = np.dot(self.lm_head, outputs[-1])
            rank1_logits = np.dot(self.lm_head, hidden)
            
            if np.argmax(true_logits) == np.argmax(rank1_logits):
                correct += 1
        
        accuracy = correct / len(sample_tokens)
        print(f"  Per-token optimal scales: {accuracy*100:.1f}%")
        print(f"  (This is the ceiling for rank-1 with universal directions)")


def main():
    print("=" * 70)
    print("RANK-1 DEBUG")
    print("=" * 70)
    
    debugger = Rank1Debugger()
    
    # 1. Debug a single token
    sample_token = list(debugger.token_scales.keys())[0]
    debugger.debug_single_token(sample_token)
    
    # 2. Analyze error accumulation
    debugger.analyze_error_accumulation(n_samples=20)
    
    # 3. Test hybrid approach
    debugger.test_hybrid_approach(n_tests=50)
    
    # 4. Test per-token scales (ceiling)
    debugger.test_per_token_scales(n_tests=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
