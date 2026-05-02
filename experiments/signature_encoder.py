#!/usr/bin/env python3
"""
Signature Encoder: Learning input_tokens → tetromino_signature
===============================================================

This is the final piece: learn to predict the hidden state's tetromino
signature directly from input tokens, WITHOUT running the transformer.

If we can do this, we eliminate the transformer entirely:
  input_tokens → signature → next_token (via lookup)

Approach:
1. Collect (input_tokens, hidden_state_signature) pairs
2. Learn a mapping from input features to signature
3. Use signature for prediction via memory lookup

The key insight from our experiments:
- Signatures cluster by semantic category
- Within-category distance: 685-791 blocks
- Cross-category distance: 851-876 blocks

If we can predict signatures within ~700 blocks of the true signature,
we'll land in the right semantic category.

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
    """
    Compute tetromino signature for a vector.
    """
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    levels = []
    patterns = []
    
    for i in range(n_blocks):
        block = blocks[i]
        
        # Mean level
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        levels.append(mean_level)
        
        # Sign pattern (4 bits)
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        patterns.append(sign_pattern.item())
    
    return torch.tensor(levels), torch.tensor(patterns)


def signature_distance(levels1, patterns1, levels2, patterns2):
    """
    Compute distance between two signatures.
    """
    level_diff = (levels1 != levels2).sum().item()
    pattern_diff = (patterns1 != patterns2).sum().item()
    return level_diff + pattern_diff


class SignatureEncoder(nn.Module):
    """
    Neural network to predict tetromino signature from input embeddings.
    
    Architecture:
    - Input: aggregated token embeddings (5 features × hidden_dim)
    - Output: (levels, patterns) for each block
    
    This is a small network - much smaller than the transformer.
    """
    
    def __init__(self, input_dim, n_blocks, n_levels=50, hidden_size=512):
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_levels = n_levels
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Level predictor (classification over n_levels)
        self.level_head = nn.Linear(hidden_size, n_blocks * n_levels)
        
        # Pattern predictor (classification over 16 patterns per block)
        self.pattern_head = nn.Linear(hidden_size, n_blocks * 16)
    
    def forward(self, x):
        """
        x: [batch, input_dim]
        Returns: level_logits [batch, n_blocks, n_levels], pattern_logits [batch, n_blocks, 16]
        """
        h = self.encoder(x)
        
        level_logits = self.level_head(h).view(-1, self.n_blocks, self.n_levels)
        pattern_logits = self.pattern_head(h).view(-1, self.n_blocks, 16)
        
        return level_logits, pattern_logits
    
    def predict(self, x, level_offset=-25):
        """
        Predict signature from input.
        Returns (levels, patterns) tensors.
        """
        level_logits, pattern_logits = self.forward(x.unsqueeze(0))
        
        levels = level_logits[0].argmax(dim=1) + level_offset
        patterns = pattern_logits[0].argmax(dim=1)
        
        return levels, patterns


def collect_training_data(model, tokenizer, n_samples=200):
    """
    Collect (input_features, signature) pairs for training.
    """
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Diverse training prompts
    prompts = [
        # Capitals (20)
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Russia is",
        "The capital of Brazil is",
        "The capital of India is",
        "The capital of Australia is",
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of Egypt is",
        "The capital of South Korea is",
        "The capital of Argentina is",
        "The capital of Poland is",
        "The capital of Sweden is",
        "The capital of Norway is",
        "The capital of Denmark is",
        "The capital of Finland is",
        # Planets (10)
        "The largest planet is",
        "The smallest planet is",
        "The hottest planet is",
        "The coldest planet is",
        "The red planet is",
        "The ringed planet is",
        "The blue planet is",
        "The closest planet to the sun is",
        "The farthest planet from the sun is",
        "The planet with the most moons is",
        # Opposites (15)
        "The opposite of hot is",
        "The opposite of big is",
        "The opposite of fast is",
        "The opposite of up is",
        "The opposite of left is",
        "The opposite of good is",
        "The opposite of happy is",
        "The opposite of light is",
        "The opposite of old is",
        "The opposite of tall is",
        "The opposite of rich is",
        "The opposite of strong is",
        "The opposite of loud is",
        "The opposite of wet is",
        "The opposite of hard is",
        # Math (15)
        "Two plus two equals",
        "Three times three equals",
        "Ten minus five equals",
        "Twenty divided by four equals",
        "Five plus five equals",
        "Six times six equals",
        "Fifteen minus ten equals",
        "Twelve divided by three equals",
        "Seven plus eight equals",
        "Nine times two equals",
        "Hundred minus fifty equals",
        "Forty divided by eight equals",
        "One plus one equals",
        "Four times four equals",
        "Eight minus three equals",
        # Facts (20)
        "Water boils at",
        "Water freezes at",
        "The speed of light is",
        "Einstein discovered",
        "Shakespeare wrote",
        "The Mona Lisa was painted by",
        "The chemical symbol for gold is",
        "The chemical symbol for silver is",
        "The chemical symbol for iron is",
        "The chemical symbol for oxygen is",
        "The tallest mountain is",
        "The longest river is",
        "The largest ocean is",
        "The smallest continent is",
        "The fastest animal is",
        "The largest mammal is",
        "The smallest bird is",
        "Diamonds are made of",
        "Glass is made of",
        "Paper is made from",
        # Scaffolding (20)
        "I went to the store and",
        "She said that she would",
        "The book is on the",
        "He walked to the",
        "They were going to the",
        "It was a very nice",
        "We need to find a",
        "The cat sat on the",
        "I think that we should",
        "Please pass me the",
        "Can you help me with",
        "I would like to",
        "Do you want to",
        "Let me show you the",
        "This is a very",
        "That was a great",
        "We should go to the",
        "They will be at the",
        "I have been to the",
        "She has always been",
        # Completions (20)
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "Once upon a time there was a",
        "In the beginning there was",
        "A journey of a thousand miles begins with a",
        "All that glitters is not",
        "Actions speak louder than",
        "The early bird catches the",
        "A penny saved is a penny",
        "Better late than",
        "Birds of a feather flock",
        "Don't count your chickens before they",
        "Every cloud has a silver",
        "Fortune favors the",
        "Good things come to those who",
        "Honesty is the best",
        "If at first you don't succeed try",
        "Knowledge is",
        "Laughter is the best",
        "Money doesn't grow on",
    ]
    
    X = []  # Input features
    Y_levels = []  # Target levels
    Y_patterns = []  # Target patterns
    
    for prompt in prompts[:n_samples]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
        
        # Input features
        token_embeds = embed[input_ids]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        # Target signature
        levels, patterns = compute_tetromino_signature(h_final)
        
        X.append(x)
        Y_levels.append(levels)
        Y_patterns.append(patterns)
    
    X = torch.stack(X)
    Y_levels = torch.stack(Y_levels)
    Y_patterns = torch.stack(Y_patterns)
    
    return X, Y_levels, Y_patterns, prompts[:n_samples]


def train_signature_encoder(X, Y_levels, Y_patterns, n_epochs=100, lr=0.001):
    """
    Train the signature encoder.
    """
    print("\n" + "=" * 70)
    print("Training Signature Encoder")
    print("=" * 70)
    
    input_dim = X.shape[1]
    n_blocks = Y_levels.shape[1]
    
    # Shift levels to be non-negative for classification
    level_min = Y_levels.min().item()
    level_max = Y_levels.max().item()
    n_levels = level_max - level_min + 1
    level_offset = level_min
    
    Y_levels_shifted = Y_levels - level_offset
    
    print(f"Input dim: {input_dim}")
    print(f"Blocks: {n_blocks}")
    print(f"Level range: [{level_min}, {level_max}] ({n_levels} classes)")
    
    # Create model
    model = SignatureEncoder(input_dim, n_blocks, n_levels, hidden_size=512)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    level_criterion = nn.CrossEntropyLoss()
    pattern_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        
        level_logits, pattern_logits = model(X)
        
        # Level loss
        level_loss = level_criterion(
            level_logits.view(-1, n_levels),
            Y_levels_shifted.view(-1)
        )
        
        # Pattern loss
        pattern_loss = pattern_criterion(
            pattern_logits.view(-1, 16),
            Y_patterns.view(-1)
        )
        
        loss = level_loss + pattern_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            # Compute accuracy
            with torch.no_grad():
                pred_levels = level_logits.argmax(dim=2)
                pred_patterns = pattern_logits.argmax(dim=2)
                
                level_acc = (pred_levels == Y_levels_shifted).float().mean().item()
                pattern_acc = (pred_patterns == Y_patterns).float().mean().item()
            
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, level_acc={level_acc:.4f}, pattern_acc={pattern_acc:.4f}")
    
    return model, level_offset


def test_signature_encoder(model, tokenizer, encoder, level_offset, X_train, prompts_train):
    """
    Test the signature encoder on new prompts.
    """
    print("\n" + "=" * 70)
    print("Testing Signature Encoder")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    # Build memory from training data
    memory = {}
    
    for i, prompt in enumerate(prompts_train):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        levels, patterns = compute_tetromino_signature(h_final)
        
        memory[i] = {
            'prompt': prompt,
            'levels': levels,
            'patterns': patterns,
            'next_token': true_token,
            'next_text': tokenizer.decode([true_token]),
        }
    
    # Test prompts
    test_prompts = [
        # In training
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "I went to the store and",
        # Similar to training
        "The capital of Thailand is",
        "The greenest planet is",
        "Three plus three equals",
        "He went to the market and",
        # Different
        "Hello, my name is",
        "The weather today is",
    ]
    
    print("\n--- Encoder Predictions ---")
    
    correct_encoder = 0
    correct_transformer = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Ground truth from transformer
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        true_levels, true_patterns = compute_tetromino_signature(h_final)
        
        # Encoder prediction
        token_embeds = embed[input_ids]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        encoder.eval()
        with torch.no_grad():
            pred_levels, pred_patterns = encoder.predict(x, level_offset)
        
        # Find nearest in memory
        best_match = None
        best_distance = float('inf')
        
        for idx, entry in memory.items():
            distance = signature_distance(
                pred_levels, pred_patterns,
                entry['levels'], entry['patterns']
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        pred_text = best_match['next_text'] if best_match else None
        pred_token = best_match['next_token'] if best_match else None
        
        # Also compute distance to true signature
        true_distance = signature_distance(pred_levels, pred_patterns, true_levels, true_patterns)
        
        is_correct = pred_token == true_token
        if is_correct:
            correct_encoder += 1
        
        marker = "✓" if is_correct else "✗"
        
        print(f"\n  {prompt!r}")
        print(f"    True: {true_text!r}")
        print(f"    Pred: {pred_text!r} (memory_dist={best_distance}, true_dist={true_distance}) {marker}")
        print(f"    Matched: {best_match['prompt']!r}")
    
    print(f"\nEncoder accuracy: {correct_encoder}/{len(test_prompts)} = {correct_encoder/len(test_prompts)*100:.1f}%")
    
    return memory


def test_end_to_end(model, tokenizer, encoder, level_offset, memory):
    """
    Test the complete encoder-only pipeline.
    
    This is the final test: can we predict next tokens WITHOUT the transformer?
    """
    print("\n" + "=" * 70)
    print("End-to-End Encoder-Only Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    test_prompts = [
        "The capital of France is",
        "The capital of Poland is",
        "The largest planet is",
        "The opposite of hot is",
        "Two plus two equals",
        "I went to the store and",
        "The quick brown fox jumps over the",
        "Hello, my name is",
    ]
    
    print("\n--- Encoder-Only Predictions (NO TRANSFORMER) ---")
    
    correct = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Ground truth (using transformer - just for comparison)
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_token = outputs.logits[0, -1, :].argmax().item()
        true_text = tokenizer.decode([true_token])
        
        # ENCODER-ONLY prediction (no transformer!)
        token_embeds = embed[input_ids]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        encoder.eval()
        with torch.no_grad():
            pred_levels, pred_patterns = encoder.predict(x, level_offset)
        
        # Find nearest in memory
        best_match = None
        best_distance = float('inf')
        
        for idx, entry in memory.items():
            distance = signature_distance(
                pred_levels, pred_patterns,
                entry['levels'], entry['patterns']
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        pred_text = best_match['next_text'] if best_match else None
        pred_token = best_match['next_token'] if best_match else None
        
        is_correct = pred_token == true_token
        if is_correct:
            correct += 1
        
        marker = "✓" if is_correct else "✗"
        
        print(f"  {prompt!r}")
        print(f"    Encoder-only: {pred_text!r}, True: {true_text!r} {marker}")
    
    print(f"\n*** ENCODER-ONLY ACCURACY: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}% ***")
    
    return correct / len(test_prompts)


def main():
    print("=" * 70)
    print("Signature Encoder: Eliminating Hidden States")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect training data
    print("\nCollecting training data...")
    X, Y_levels, Y_patterns, prompts = collect_training_data(model, tokenizer, n_samples=120)
    print(f"Collected {len(X)} samples")
    
    # Train encoder
    encoder, level_offset = train_signature_encoder(X, Y_levels, Y_patterns, n_epochs=200)
    
    # Test encoder
    memory = test_signature_encoder(model, tokenizer, encoder, level_offset, X, prompts)
    
    # End-to-end test
    accuracy = test_end_to_end(model, tokenizer, encoder, level_offset, memory)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Signature Encoder Results:

The encoder learns to predict tetromino signatures from input embeddings.
This enables prediction WITHOUT running the transformer.

Pipeline:
  input_tokens → embeddings → encoder → signature → memory lookup → next_token

Accuracy: {accuracy*100:.1f}%

If this works, we have eliminated the transformer's hidden states!
The "world knowledge" is stored in the memory lookup table.
""")
    
    return encoder, memory


if __name__ == "__main__":
    encoder, memory = main()
