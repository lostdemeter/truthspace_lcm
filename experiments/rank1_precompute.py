#!/usr/bin/env python3
"""
Rank-1 Transformer Precomputation
==================================

Precompute the entire vocabulary using rank-1 decomposition.

For layers 3-27:
    output = input + scale × direction

We precompute:
1. Universal directions (28 vectors, 392 KB)
2. Scales for all tokens (152K × 28 floats, 17 MB)

Then test end-to-end generation quality.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/thorin/truthspace-lcm/cache/rank1"


class Rank1Precomputer:
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
        self.vocab_size = self.tokenizer.vocab_size
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Vocab size: {self.vocab_size}")
        
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    def get_layer_outputs(self, token_ids: List[int]) -> List[np.ndarray]:
        """Get hidden state after each layer."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def learn_universal_directions(self, n_samples: int = 500) -> np.ndarray:
        """Learn universal direction for each layer from samples."""
        print(f"\n--- Learning universal directions from {n_samples} samples ---")
        
        # Collect transformations per layer
        layer_transforms = {i: [] for i in range(self.n_layers)}
        
        sample_tokens = np.random.choice(self.vocab_size, n_samples, replace=False)
        
        for i, token_id in enumerate(sample_tokens):
            if i % 100 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            try:
                outputs = self.get_layer_outputs([int(token_id)])
                for layer_idx in range(self.n_layers):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    layer_transforms[layer_idx].append(transform)
            except:
                continue
        
        # SVD to get principal direction per layer
        directions = np.zeros((self.n_layers, self.hidden_dim), dtype=np.float32)
        
        for layer_idx in range(self.n_layers):
            transforms = np.array(layer_transforms[layer_idx])
            _, S, Vt = np.linalg.svd(transforms, full_matrices=False)
            directions[layer_idx] = Vt[0]
            
            # Report quality
            var_explained = S[0]**2 / (S**2).sum()
            print(f"  Layer {layer_idx}: {var_explained*100:.1f}% variance explained")
        
        # Save
        np.save(os.path.join(CACHE_DIR, "universal_directions.npy"), directions)
        print(f"\n  Saved directions to {CACHE_DIR}/universal_directions.npy")
        print(f"  Size: {directions.nbytes / 1024:.1f} KB")
        
        return directions
    
    def precompute_token_scales(self, directions: np.ndarray, 
                                n_tokens: int = None,
                                batch_size: int = 100) -> Dict[int, np.ndarray]:
        """Precompute scales for tokens."""
        if n_tokens is None:
            n_tokens = self.vocab_size
        
        print(f"\n--- Precomputing scales for {n_tokens} tokens ---")
        
        token_scales = {}
        
        # Sample or use all tokens
        if n_tokens < self.vocab_size:
            token_ids = np.random.choice(self.vocab_size, n_tokens, replace=False)
        else:
            token_ids = np.arange(self.vocab_size)
        
        for batch_start in range(0, len(token_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(token_ids))
            batch = token_ids[batch_start:batch_end]
            
            if batch_start % 1000 == 0:
                print(f"  Tokens {batch_start}/{len(token_ids)}...")
            
            for token_id in batch:
                try:
                    outputs = self.get_layer_outputs([int(token_id)])
                    scales = np.zeros(self.n_layers, dtype=np.float32)
                    
                    for layer_idx in range(self.n_layers):
                        transform = outputs[layer_idx + 1] - outputs[layer_idx]
                        scales[layer_idx] = np.dot(transform, directions[layer_idx])
                    
                    token_scales[int(token_id)] = scales
                except:
                    continue
        
        # Save
        with open(os.path.join(CACHE_DIR, "token_scales.pkl"), "wb") as f:
            pickle.dump(token_scales, f)
        
        n_stored = len(token_scales)
        total_bytes = n_stored * self.n_layers * 4
        print(f"\n  Saved {n_stored} token scales to {CACHE_DIR}/token_scales.pkl")
        print(f"  Size: {total_bytes / 1024:.1f} KB")
        
        return token_scales
    
    def test_rank1_generation(self, directions: np.ndarray, 
                              token_scales: Dict[int, np.ndarray],
                              n_tests: int = 50) -> Dict:
        """Test generation quality with rank-1 replacement."""
        print(f"\n--- Testing rank-1 generation ({n_tests} tokens) ---")
        
        correct_full = 0
        correct_rank1 = 0
        
        test_tokens = np.random.choice(list(token_scales.keys()), 
                                       min(n_tests, len(token_scales)), 
                                       replace=False)
        
        for token_id in test_tokens:
            try:
                # Get true outputs
                outputs = self.get_layer_outputs([int(token_id)])
                true_final = outputs[-1]
                
                # Reconstruct using rank-1
                hidden = outputs[0].copy()  # Start from embedding
                
                for layer_idx in range(self.n_layers):
                    if layer_idx >= 3:  # Rank-1 layers
                        scale = token_scales[int(token_id)][layer_idx]
                        direction = directions[layer_idx]
                        hidden = hidden + scale * direction
                    else:  # Use true output for layers 0-2
                        hidden = outputs[layer_idx + 1].copy()
                
                # Decode both
                true_logits = np.dot(self.lm_head, true_final)
                rank1_logits = np.dot(self.lm_head, hidden)
                
                true_token = np.argmax(true_logits)
                rank1_token = np.argmax(rank1_logits)
                
                if true_token == rank1_token:
                    correct_rank1 += 1
                correct_full += 1
                
            except Exception as e:
                continue
        
        accuracy = correct_rank1 / correct_full if correct_full > 0 else 0
        print(f"\n  Rank-1 generation accuracy: {correct_rank1}/{correct_full} = {accuracy*100:.1f}%")
        
        return {
            'correct': correct_rank1,
            'total': correct_full,
            'accuracy': accuracy,
        }
    
    def test_full_rank1_generation(self, directions: np.ndarray,
                                   token_scales: Dict[int, np.ndarray],
                                   n_tests: int = 50) -> Dict:
        """Test with rank-1 for ALL layers (including 0-2)."""
        print(f"\n--- Testing FULL rank-1 generation (all layers) ---")
        
        correct = 0
        total = 0
        
        test_tokens = np.random.choice(list(token_scales.keys()),
                                       min(n_tests, len(token_scales)),
                                       replace=False)
        
        for token_id in test_tokens:
            try:
                outputs = self.get_layer_outputs([int(token_id)])
                true_final = outputs[-1]
                
                # Start from embedding
                hidden = outputs[0].copy()
                
                # Apply rank-1 for ALL layers
                for layer_idx in range(self.n_layers):
                    scale = token_scales[int(token_id)][layer_idx]
                    direction = directions[layer_idx]
                    hidden = hidden + scale * direction
                
                # Decode
                true_logits = np.dot(self.lm_head, true_final)
                rank1_logits = np.dot(self.lm_head, hidden)
                
                true_token = np.argmax(true_logits)
                rank1_token = np.argmax(rank1_logits)
                
                if true_token == rank1_token:
                    correct += 1
                total += 1
                
            except:
                continue
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n  Full rank-1 accuracy (all layers): {correct}/{total} = {accuracy*100:.1f}%")
        
        return {'correct': correct, 'total': total, 'accuracy': accuracy}
    
    def test_generation_examples(self, directions: np.ndarray,
                                 token_scales: Dict[int, np.ndarray]) -> None:
        """Show some generation examples."""
        print(f"\n--- Generation Examples ---")
        
        prompts = [
            "The capital of France is",
            "The meaning of life is",
            "Hello, how are",
        ]
        
        for prompt in prompts:
            print(f"\n  Prompt: '{prompt}'")
            
            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Get true next token
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                true_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            
            true_logits = np.dot(self.lm_head, true_hidden)
            true_token = np.argmax(true_logits)
            true_word = self.tokenizer.decode([true_token])
            
            print(f"    True next: '{true_word}'")
            
            # Try rank-1 for last token
            last_token_id = input_ids[0, -1].item()
            
            if last_token_id in token_scales:
                # Get embedding for last token
                embedding = self.model.model.embed_tokens.weight[last_token_id].float().cpu().numpy()
                
                hidden = embedding.copy()
                for layer_idx in range(self.n_layers):
                    scale = token_scales[last_token_id][layer_idx]
                    direction = directions[layer_idx]
                    hidden = hidden + scale * direction
                
                rank1_logits = np.dot(self.lm_head, hidden)
                rank1_token = np.argmax(rank1_logits)
                rank1_word = self.tokenizer.decode([rank1_token])
                
                match = "✓" if rank1_token == true_token else "✗"
                print(f"    Rank-1 next: '{rank1_word}' {match}")
            else:
                print(f"    (Token {last_token_id} not in cache)")


def main():
    print("=" * 70)
    print("RANK-1 TRANSFORMER PRECOMPUTATION")
    print("=" * 70)
    
    precomputer = Rank1Precomputer()
    
    # 1. Learn universal directions
    directions = precomputer.learn_universal_directions(n_samples=500)
    
    # 2. Precompute scales for a sample (full vocab would take too long)
    token_scales = precomputer.precompute_token_scales(directions, n_tokens=5000)
    
    # 3. Test generation with rank-1 (layers 3-27 only)
    results_partial = precomputer.test_rank1_generation(directions, token_scales, n_tests=100)
    
    # 4. Test generation with full rank-1 (all layers)
    results_full = precomputer.test_full_rank1_generation(directions, token_scales, n_tests=100)
    
    # 5. Show examples
    precomputer.test_generation_examples(directions, token_scales)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Precomputed:
  - {len(token_scales)} tokens
  - Directions: {directions.nbytes / 1024:.1f} KB
  - Scales: {len(token_scales) * 28 * 4 / 1024:.1f} KB

Generation accuracy:
  - Rank-1 (layers 3-27): {results_partial['accuracy']*100:.1f}%
  - Rank-1 (all layers): {results_full['accuracy']*100:.1f}%

Extrapolated to full vocab:
  - Total size: {(152000 * 28 * 4 + directions.nbytes) / 1e6:.1f} MB
""")


if __name__ == "__main__":
    main()
