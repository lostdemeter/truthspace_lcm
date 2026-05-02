#!/usr/bin/env python3
"""
Rank-1 Transformation Analysis
===============================

Key finding: Layers 3-27 have rank-1 transformations!

This means: transform = scale × direction

Questions:
1. Is the direction SHARED across tokens? (universal direction)
2. Or does each token have its own direction? (token-specific)
3. Can we precompute all scales for the vocabulary?

If direction is universal, we only need to store:
- 28 direction vectors (shared)
- 28 scales per token (112 bytes per token)

Total for 152K vocab: 17 MB!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class Rank1Analyzer:
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
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_layer_outputs(self, token_ids: List[int]) -> List[np.ndarray]:
        """Get hidden state after each layer."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def analyze_rank1_structure(self, n_samples: int = 200) -> Dict:
        """
        Analyze if transformations share a universal direction.
        """
        print(f"\n--- Analyzing rank-1 structure ({n_samples} samples) ---")
        
        # Collect transformations per layer
        layer_transforms = {i: [] for i in range(self.n_layers)}
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                outputs = self.get_layer_outputs([token_id])
                for layer_idx in range(self.n_layers):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    layer_transforms[layer_idx].append(transform)
            except:
                continue
        
        # Analyze each layer
        results = []
        
        for layer_idx in range(self.n_layers):
            transforms = np.array(layer_transforms[layer_idx])
            
            # SVD to get principal direction
            U, S, Vt = np.linalg.svd(transforms, full_matrices=False)
            
            # The principal direction
            principal_dir = Vt[0]
            
            # Project all transforms onto principal direction
            projections = transforms @ principal_dir
            
            # Residuals (what's NOT captured by rank-1)
            rank1_approx = np.outer(projections, principal_dir)
            residuals = transforms - rank1_approx
            
            # Metrics
            transform_norms = np.linalg.norm(transforms, axis=1)
            residual_norms = np.linalg.norm(residuals, axis=1)
            
            explained_variance = 1 - (residual_norms**2).sum() / (transform_norms**2).sum()
            
            # Direction consistency: how aligned are individual transforms with principal?
            directions = transforms / (transform_norms[:, None] + 1e-10)
            alignments = np.abs(directions @ principal_dir)
            
            results.append({
                'layer': layer_idx,
                'explained_variance': explained_variance,
                'mean_alignment': alignments.mean(),
                'min_alignment': alignments.min(),
                'principal_dir': principal_dir,
                'projections': projections,
                'top_singular': S[0],
                'second_singular': S[1] if len(S) > 1 else 0,
            })
        
        # Print results
        print(f"\n  Layer-by-layer rank-1 analysis:")
        print(f"  {'Layer':<8} {'Var Expl':<12} {'Mean Align':<12} {'Min Align':<12} {'S1/S2':<10}")
        print(f"  {'-'*54}")
        
        for r in results:
            ratio = r['top_singular'] / (r['second_singular'] + 1e-10)
            print(f"  {r['layer']:<8} {r['explained_variance']*100:.1f}%{'':<6} {r['mean_alignment']:.3f}{'':<7} {r['min_alignment']:.3f}{'':<7} {ratio:.1f}")
        
        return results
    
    def test_universal_direction(self, n_train: int = 100, n_test: int = 50) -> Dict:
        """
        Test if we can use a UNIVERSAL direction learned from training tokens
        to reconstruct transformations for TEST tokens.
        """
        print(f"\n--- Testing universal direction hypothesis ---")
        
        # Collect training transforms
        train_transforms = {i: [] for i in range(self.n_layers)}
        
        print(f"  Collecting {n_train} training samples...")
        for i in range(n_train):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            try:
                outputs = self.get_layer_outputs([token_id])
                for layer_idx in range(self.n_layers):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    train_transforms[layer_idx].append(transform)
            except:
                continue
        
        # Learn universal direction per layer
        universal_dirs = {}
        for layer_idx in range(self.n_layers):
            transforms = np.array(train_transforms[layer_idx])
            _, _, Vt = np.linalg.svd(transforms, full_matrices=False)
            universal_dirs[layer_idx] = Vt[0]
        
        # Test on new tokens
        print(f"  Testing on {n_test} new tokens...")
        
        layer_accuracies = {i: [] for i in range(self.n_layers)}
        
        for i in range(n_test):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            try:
                outputs = self.get_layer_outputs([token_id])
                
                for layer_idx in range(self.n_layers):
                    true_transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    
                    # Reconstruct using universal direction
                    direction = universal_dirs[layer_idx]
                    scale = np.dot(true_transform, direction)
                    reconstructed = scale * direction
                    
                    # Accuracy
                    error = np.linalg.norm(true_transform - reconstructed)
                    true_norm = np.linalg.norm(true_transform)
                    accuracy = 1 - error / (true_norm + 1e-10)
                    
                    layer_accuracies[layer_idx].append(accuracy)
            except:
                continue
        
        # Print results
        print(f"\n  Universal direction reconstruction accuracy:")
        print(f"  {'Layer':<8} {'Mean Acc':<12} {'Min Acc':<12}")
        print(f"  {'-'*32}")
        
        for layer_idx in range(self.n_layers):
            accs = layer_accuracies[layer_idx]
            if accs:
                print(f"  {layer_idx:<8} {np.mean(accs)*100:.1f}%{'':<6} {np.min(accs)*100:.1f}%")
        
        # Overall
        all_accs = [a for accs in layer_accuracies.values() for a in accs]
        print(f"\n  Overall mean accuracy: {np.mean(all_accs)*100:.1f}%")
        
        return {
            'universal_dirs': universal_dirs,
            'layer_accuracies': layer_accuracies,
        }
    
    def precompute_vocabulary_sample(self, n_tokens: int = 1000) -> Dict:
        """
        Precompute rank-1 scales for a sample of the vocabulary.
        
        This simulates what a full precomputation would look like.
        """
        print(f"\n--- Precomputing rank-1 scales for {n_tokens} tokens ---")
        
        # First, learn universal directions from a subset
        n_train = min(200, n_tokens // 2)
        train_transforms = {i: [] for i in range(self.n_layers)}
        
        print(f"  Learning universal directions from {n_train} tokens...")
        train_token_ids = np.random.choice(self.tokenizer.vocab_size, n_train, replace=False)
        
        for token_id in train_token_ids:
            try:
                outputs = self.get_layer_outputs([int(token_id)])
                for layer_idx in range(self.n_layers):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    train_transforms[layer_idx].append(transform)
            except:
                continue
        
        # Learn directions
        universal_dirs = {}
        for layer_idx in range(self.n_layers):
            transforms = np.array(train_transforms[layer_idx])
            _, _, Vt = np.linalg.svd(transforms, full_matrices=False)
            universal_dirs[layer_idx] = Vt[0]
        
        # Now precompute scales for all tokens
        print(f"  Precomputing scales for {n_tokens} tokens...")
        
        token_scales = {}  # token_id -> [scale_layer0, scale_layer1, ...]
        
        test_token_ids = np.random.choice(self.tokenizer.vocab_size, n_tokens, replace=False)
        
        for i, token_id in enumerate(test_token_ids):
            if i % 200 == 0:
                print(f"    Token {i}/{n_tokens}...")
            
            try:
                outputs = self.get_layer_outputs([int(token_id)])
                scales = []
                
                for layer_idx in range(self.n_layers):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    direction = universal_dirs[layer_idx]
                    scale = np.dot(transform, direction)
                    scales.append(scale)
                
                token_scales[int(token_id)] = np.array(scales)
            except:
                continue
        
        # Storage analysis
        n_stored = len(token_scales)
        bytes_per_token = self.n_layers * 4  # float32 per layer
        total_bytes = n_stored * bytes_per_token
        
        # Direction storage
        direction_bytes = self.n_layers * self.hidden_dim * 4
        
        print(f"\n  Storage analysis:")
        print(f"    Tokens stored: {n_stored}")
        print(f"    Bytes per token: {bytes_per_token}")
        print(f"    Token scales total: {total_bytes / 1024:.1f} KB")
        print(f"    Direction vectors: {direction_bytes / 1024:.1f} KB")
        print(f"    Total: {(total_bytes + direction_bytes) / 1024:.1f} KB")
        
        # Extrapolate to full vocab
        full_vocab_bytes = self.tokenizer.vocab_size * bytes_per_token + direction_bytes
        print(f"\n  Extrapolated to full vocab ({self.tokenizer.vocab_size} tokens):")
        print(f"    Total: {full_vocab_bytes / 1e6:.1f} MB")
        
        return {
            'universal_dirs': universal_dirs,
            'token_scales': token_scales,
            'bytes_per_token': bytes_per_token,
        }


def main():
    print("=" * 70)
    print("RANK-1 TRANSFORMATION ANALYSIS")
    print("=" * 70)
    print("""
Key finding: Layers 3-27 have rank-1 transformations!

Questions:
1. Is the direction UNIVERSAL across tokens?
2. Can we precompute scales for the entire vocabulary?
3. What's the total storage requirement?
""")
    
    analyzer = Rank1Analyzer()
    
    # 1. Analyze rank-1 structure
    rank1_results = analyzer.analyze_rank1_structure(n_samples=200)
    
    # 2. Test universal direction hypothesis
    universal_results = analyzer.test_universal_direction(n_train=100, n_test=50)
    
    # 3. Precompute sample
    precompute_results = analyzer.precompute_vocabulary_sample(n_tokens=500)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
