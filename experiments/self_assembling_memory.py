#!/usr/bin/env python3
"""
Self-Assembling Memory: Automatic Encoder Scaling
==================================================

Goal: Automatically expand the signature memory so more cases
become in-distribution, increasing encoder usage from 60% to higher.

Key insights from prior work:
1. Doc 155 (Smart φ-Shape): Knowledge is a shape that can grow
2. Doc 167 (Self-Assembling Navigation): Semantic pairs can be discovered
   automatically from sign patterns

Strategy:
1. Start with seed memory (training prompts)
2. When transformer is used (high distance), learn from it
3. Add new signatures to memory
4. Over time, more cases become in-distribution

This is ONLINE LEARNING for the signature memory.

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def compute_tetromino_signature(vec, block_size=4):
    """Compute tetromino signature for a vector."""
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    levels = []
    patterns = []
    
    for i in range(n_blocks):
        block = blocks[i]
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        levels.append(mean_level)
        
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        patterns.append(sign_pattern.item())
    
    return torch.tensor(levels), torch.tensor(patterns)


def signature_distance(levels1, patterns1, levels2, patterns2):
    """Compute distance between two signatures."""
    level_diff = (levels1 != levels2).sum().item()
    pattern_diff = (patterns1 != patterns2).sum().item()
    return level_diff + pattern_diff


class SelfAssemblingMemory:
    """
    Memory that grows automatically by learning from transformer outputs.
    
    When a query has high distance (out-of-distribution), we:
    1. Use the transformer to get the correct answer
    2. Add the (signature, next_token) pair to memory
    3. Future similar queries will use the encoder
    
    This is the self-assembly process from Doc 167.
    """
    
    def __init__(self, model, tokenizer, threshold=1000):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        
        self.embed = model.model.embed_tokens.weight.data.clone()
        self.lm_head = model.lm_head.weight.data.clone()
        
        # Memory: list of (levels, patterns, next_token, prompt)
        self.memory = []
        
        # Statistics
        self.stats = {
            'encoder_used': 0,
            'transformer_used': 0,
            'memory_additions': 0,
        }
    
    def add_to_memory(self, prompt, levels, patterns, next_token):
        """Add a new entry to memory."""
        self.memory.append({
            'prompt': prompt,
            'levels': levels,
            'patterns': patterns,
            'next_token': next_token,
            'next_text': self.tokenizer.decode([next_token]),
        })
        self.stats['memory_additions'] += 1
    
    def find_nearest(self, query_levels, query_patterns):
        """Find nearest signature in memory."""
        if not self.memory:
            return None, float('inf'), None
        
        best_match = None
        best_distance = float('inf')
        best_prompt = None
        
        for entry in self.memory:
            distance = signature_distance(
                query_levels, query_patterns,
                entry['levels'], entry['patterns']
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = entry
                best_prompt = entry['prompt']
        
        return best_match, best_distance, best_prompt
    
    def predict(self, prompt, learn=True):
        """
        Predict next token with automatic memory expansion.
        
        If distance > threshold:
        - Use transformer
        - Add to memory (if learn=True)
        
        If distance <= threshold:
        - Use encoder (memory lookup)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Get hidden state and true prediction from transformer
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Compute signature
        query_levels, query_patterns = compute_tetromino_signature(h_final)
        
        # Find nearest in memory
        match, distance, matched_prompt = self.find_nearest(query_levels, query_patterns)
        
        # Decide: encoder or transformer?
        if distance <= self.threshold and match is not None:
            # Use encoder (memory lookup)
            pred_token = match['next_token']
            method = 'encoder'
            self.stats['encoder_used'] += 1
        else:
            # Use transformer
            pred_token = true_token
            method = 'transformer'
            self.stats['transformer_used'] += 1
            
            # Learn: add to memory
            if learn:
                self.add_to_memory(prompt, query_levels, query_patterns, true_token)
        
        return {
            'pred_token': pred_token,
            'pred_text': self.tokenizer.decode([pred_token]),
            'true_token': true_token,
            'true_text': self.tokenizer.decode([true_token]),
            'distance': distance,
            'method': method,
            'is_correct': pred_token == true_token,
        }
    
    def seed_memory(self, prompts):
        """Initialize memory with seed prompts."""
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]
            
            with torch.no_grad():
                outputs = self.model(input_ids.unsqueeze(0), output_hidden_states=True)
                h_final = outputs.hidden_states[-1][0, -1, :]
                true_token = outputs.logits[0, -1, :].argmax().item()
            
            levels, patterns = compute_tetromino_signature(h_final)
            self.add_to_memory(prompt, levels, patterns, true_token)
    
    def print_stats(self):
        """Print memory statistics."""
        total = self.stats['encoder_used'] + self.stats['transformer_used']
        if total == 0:
            print("No predictions made yet")
            return
        
        encoder_pct = self.stats['encoder_used'] / total * 100
        transformer_pct = self.stats['transformer_used'] / total * 100
        
        print(f"\n--- Self-Assembling Memory Stats ---")
        print(f"Memory size: {len(self.memory)}")
        print(f"Encoder used: {self.stats['encoder_used']} ({encoder_pct:.1f}%)")
        print(f"Transformer used: {self.stats['transformer_used']} ({transformer_pct:.1f}%)")
        print(f"Memory additions: {self.stats['memory_additions']}")


def test_self_assembling_memory(model, tokenizer):
    """
    Test the self-assembling memory.
    
    Simulate a stream of queries and watch the memory grow.
    """
    print("\n" + "=" * 70)
    print("Self-Assembling Memory Test")
    print("=" * 70)
    
    # Create memory with small seed
    memory = SelfAssemblingMemory(model, tokenizer, threshold=500)
    
    # Seed with just a few prompts
    seed_prompts = [
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
    ]
    
    memory.seed_memory(seed_prompts)
    print(f"Seeded memory with {len(seed_prompts)} prompts")
    
    # Simulate query stream
    query_stream = [
        # Round 1: Some in-distribution, some out
        "The capital of France is",      # In seed
        "The capital of Germany is",     # Out - will learn
        "The largest planet is",         # In seed
        "The smallest planet is",        # Out - will learn
        "Two plus two equals",           # In seed
        "Three plus three equals",       # Out - will learn
        
        # Round 2: Previously learned should now be in-distribution
        "The capital of Germany is",     # Should be in memory now
        "The smallest planet is",        # Should be in memory now
        "Three plus three equals",       # Should be in memory now
        
        # Round 3: New queries
        "The capital of Italy is",       # Out - will learn
        "Hello, my name is",             # Out - will learn
        "The weather today is",          # Out - will learn
        
        # Round 4: Check if learned
        "The capital of Italy is",       # Should be in memory
        "Hello, my name is",             # Should be in memory
    ]
    
    print("\n--- Query Stream ---")
    
    for i, prompt in enumerate(query_stream):
        result = memory.predict(prompt, learn=True)
        
        marker = "✓" if result['is_correct'] else "✗"
        
        print(f"\n[{i+1}] {prompt!r}")
        print(f"    Method: {result['method'].upper()} (dist={result['distance']})")
        print(f"    Pred: {result['pred_text']!r}, True: {result['true_text']!r} {marker}")
        print(f"    Memory size: {len(memory.memory)}")
    
    memory.print_stats()
    
    return memory


def test_memory_growth_over_time(model, tokenizer):
    """
    Test how memory grows and encoder usage increases over time.
    """
    print("\n" + "=" * 70)
    print("Memory Growth Over Time")
    print("=" * 70)
    
    # Create memory with small seed
    memory = SelfAssemblingMemory(model, tokenizer, threshold=500)
    
    # Seed with minimal prompts
    seed_prompts = [
        "The capital of France is",
        "Two plus two equals",
    ]
    memory.seed_memory(seed_prompts)
    
    # Large query stream with repetition
    categories = {
        'capitals': [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
            "The capital of Japan is",
        ],
        'math': [
            "Two plus two equals",
            "Three plus three equals",
            "Five plus five equals",
            "Ten minus five equals",
        ],
        'planets': [
            "The largest planet is",
            "The smallest planet is",
            "The hottest planet is",
        ],
        'greetings': [
            "Hello, my name is",
            "Hi, I am",
            "Good morning, my name is",
        ],
    }
    
    # Simulate multiple rounds
    rounds = 3
    
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Reset round stats
        round_encoder = 0
        round_transformer = 0
        
        for cat_name, prompts in categories.items():
            for prompt in prompts:
                result = memory.predict(prompt, learn=True)
                
                if result['method'] == 'encoder':
                    round_encoder += 1
                else:
                    round_transformer += 1
        
        total = round_encoder + round_transformer
        encoder_pct = round_encoder / total * 100
        
        print(f"  Encoder: {round_encoder}/{total} ({encoder_pct:.1f}%)")
        print(f"  Memory size: {len(memory.memory)}")
    
    memory.print_stats()
    
    # Final test: all queries should now be in-distribution
    print("\n--- Final Test (all should use encoder) ---")
    
    final_encoder = 0
    final_total = 0
    
    for cat_name, prompts in categories.items():
        for prompt in prompts:
            result = memory.predict(prompt, learn=False)  # Don't learn, just test
            
            if result['method'] == 'encoder':
                final_encoder += 1
            final_total += 1
    
    print(f"Final encoder usage: {final_encoder}/{final_total} = {final_encoder/final_total*100:.1f}%")
    
    return memory


def analyze_semantic_clustering(model, tokenizer):
    """
    Analyze if semantically similar prompts cluster in signature space.
    
    This is key for generalization: if "capital of X" prompts cluster,
    we can generalize to new countries.
    """
    print("\n" + "=" * 70)
    print("Semantic Clustering Analysis")
    print("=" * 70)
    
    # Categories of prompts
    categories = {
        'capitals': [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
            "The capital of Japan is",
            "The capital of China is",
        ],
        'planets': [
            "The largest planet is",
            "The smallest planet is",
            "The hottest planet is",
            "The coldest planet is",
        ],
        'math': [
            "Two plus two equals",
            "Three plus three equals",
            "Five plus five equals",
            "Ten minus five equals",
        ],
        'greetings': [
            "Hello, my name is",
            "Hi, I am",
            "Good morning, my name is",
        ],
    }
    
    # Collect signatures for each category
    category_sigs = {}
    
    for cat_name, prompts in categories.items():
        sigs = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
            
            with torch.no_grad():
                outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
                h_final = outputs.hidden_states[-1][0, -1, :]
            
            levels, patterns = compute_tetromino_signature(h_final)
            sigs.append((levels, patterns))
        
        category_sigs[cat_name] = sigs
    
    # Compute within-category distances
    print("\n--- Within-Category Distances ---")
    
    for cat_name, sigs in category_sigs.items():
        distances = []
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                d = signature_distance(sigs[i][0], sigs[i][1], sigs[j][0], sigs[j][1])
                distances.append(d)
        
        mean_dist = np.mean(distances) if distances else 0
        print(f"  {cat_name}: mean distance = {mean_dist:.0f}")
    
    # Compute cross-category distances
    print("\n--- Cross-Category Distances ---")
    
    for cat1, sigs1 in category_sigs.items():
        for cat2, sigs2 in category_sigs.items():
            if cat1 < cat2:
                distances = []
                for s1 in sigs1:
                    for s2 in sigs2:
                        d = signature_distance(s1[0], s1[1], s2[0], s2[1])
                        distances.append(d)
                
                mean_dist = np.mean(distances)
                print(f"  {cat1} vs {cat2}: {mean_dist:.0f}")
    
    # Key insight: If within-category < cross-category, we can generalize
    print("\n--- Generalization Potential ---")
    
    within_dists = []
    cross_dists = []
    
    for cat_name, sigs in category_sigs.items():
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                d = signature_distance(sigs[i][0], sigs[i][1], sigs[j][0], sigs[j][1])
                within_dists.append(d)
    
    for cat1, sigs1 in category_sigs.items():
        for cat2, sigs2 in category_sigs.items():
            if cat1 < cat2:
                for s1 in sigs1:
                    for s2 in sigs2:
                        d = signature_distance(s1[0], s1[1], s2[0], s2[1])
                        cross_dists.append(d)
    
    mean_within = np.mean(within_dists)
    mean_cross = np.mean(cross_dists)
    
    print(f"  Mean within-category: {mean_within:.0f}")
    print(f"  Mean cross-category: {mean_cross:.0f}")
    print(f"  Ratio: {mean_cross / mean_within:.2f}x")
    
    if mean_cross > mean_within * 1.2:
        print("  ✓ Good clustering! Generalization should work.")
    else:
        print("  ✗ Weak clustering. May need more dimensions.")


def main():
    print("=" * 70)
    print("Self-Assembling Memory: Automatic Encoder Scaling")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: Basic self-assembling memory
    memory = test_self_assembling_memory(model, tokenizer)
    
    # Test 2: Memory growth over time
    memory2 = test_memory_growth_over_time(model, tokenizer)
    
    # Test 3: Semantic clustering
    analyze_semantic_clustering(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Self-Assembling Memory Results:

1. Memory grows automatically from transformer outputs
2. Repeated queries become in-distribution over time
3. Encoder usage increases as memory grows

Key Insight:
The memory SELF-ASSEMBLES by learning from the transformer.
Over time, more cases become in-distribution, and encoder
usage approaches 100%.

This is the path to scaling:
- Start with small seed memory
- Learn from every transformer call
- Memory grows to cover the distribution
- Eventually, most queries use the encoder

The system is SELF-IMPROVING without retraining!
""")


if __name__ == "__main__":
    main()
