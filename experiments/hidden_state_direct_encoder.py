#!/usr/bin/env python3
"""
Hidden State Direct Encoder
============================

Key finding from previous test:
- Transformer amplifies norm by 300x
- h0 and h_final are nearly orthogonal (cos ≈ 0)
- Projection to any k < 3584 gives 0% accuracy

The transformer is doing ESSENTIAL computation that can't be skipped.

BUT: What if we can learn a DIRECT mapping from tokens to the final hidden state?

The transformer is a function: f(tokens) → h_final
If we can approximate f with a simpler function, we eliminate the transformer.

Approach:
1. Collect (tokens, h_final) pairs
2. Learn a mapping: tokens → h_final
3. Use h_final → lm_head → next_token

This is essentially "distillation" but focused on the hidden state.

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

PHI = 1.6180339887498949


def test_direct_hidden_mapping():
    """
    Can we learn a direct mapping from tokens to final hidden state?
    """
    print("\n" + "=" * 70)
    print("Direct Hidden State Mapping")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    hidden_dim = model.config.hidden_size
    
    # Collect training data
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "The largest planet is",
        "The smallest country is",
        "Water is essential for",
        "The sun rises in the",
        "Music is a form of",
        "The meaning of life is",
        "Once upon a time there was a",
        "In the beginning there was",
        "The elephant is a large animal that",
        "The lion is a large animal that",
    ]
    
    X_train = []  # Token sequences (as embedding sums)
    Y_train = []  # Final hidden states
    tokens_train = []  # True next tokens
    
    print("Collecting training data...")
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Input features: various combinations of token embeddings
        token_embeds = embed[input_ids[0]]  # [seq_len, hidden_dim]
        seq_len = len(token_embeds)
        
        # Feature 1: Sum of embeddings
        feat_sum = token_embeds.sum(dim=0)
        
        # Feature 2: Mean of embeddings
        feat_mean = token_embeds.mean(dim=0)
        
        # Feature 3: Last embedding
        feat_last = token_embeds[-1]
        
        # Feature 4: Weighted sum (exponential)
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        
        # Feature 5: First embedding
        feat_first = token_embeds[0]
        
        # Concatenate all features
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        X_train.append(x)
        Y_train.append(h_final)
        tokens_train.append(true_token)
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    print(f"Training data: {X_train.shape} -> {Y_train.shape}")
    
    # Learn linear mapping with ridge regression
    # Y = X @ W + b
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    
    lambda_reg = 1.0  # Regularization
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y_train)
    
    print(f"Learned mapping: {X_with_bias.shape[1]} -> {Y_train.shape[1]}")
    
    # Test on training data
    Y_pred = X_with_bias @ W
    
    # Measure reconstruction quality
    recon_error = torch.norm(Y_pred - Y_train, dim=1).mean().item()
    recon_cos = F.cosine_similarity(Y_pred, Y_train, dim=1).mean().item()
    
    print(f"\nReconstruction quality:")
    print(f"  Mean L2 error: {recon_error:.2f}")
    print(f"  Mean cosine sim: {recon_cos:.4f}")
    
    # Test prediction accuracy
    print("\n--- Training Set Predictions ---")
    correct = 0
    for i, (prompt, true_token) in enumerate(zip(training_prompts, tokens_train)):
        # Predict using learned hidden state
        logits = Y_pred[i] @ lm_head.T
        pred_token = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        match = "✓" if pred_token == true_token else "✗"
        
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTraining accuracy: {correct}/{len(training_prompts)} = {correct/len(training_prompts)*100:.1f}%")
    
    # Test on new prompts
    test_prompts = [
        "The capital of Russia is",
        "The largest ocean is the",
        "The color of the sky is",
        "Birds can fly because they have",
    ]
    
    print("\n--- Test Set Predictions ---")
    correct = 0
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Build features
        token_embeds = embed[input_ids[0]]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first, torch.ones(1)])
        h_pred = x @ W
        
        # Predict
        logits = h_pred @ lm_head.T
        pred_token = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        match = "✓" if pred_token == true_token else "✗"
        
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTest accuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return W


def test_token_sequence_lookup():
    """
    What if we just memorize (prompt → next_token) directly?
    
    This is the simplest possible "encoder" - a lookup table.
    """
    print("\n" + "=" * 70)
    print("Token Sequence Lookup (Memorization)")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Build lookup table
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "Python is a programming language that",
        "Java is a programming language that",
    ]
    
    lookup = {}
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Key: tuple of token IDs
        key = tuple(input_ids[0].tolist())
        lookup[key] = true_token
        
        true_text = tokenizer.decode([true_token])
        print(f"  {prompt!r} → {true_text!r}")
    
    print(f"\nLookup table size: {len(lookup)} entries")
    
    # Test on similar prompts
    test_prompts = [
        "The capital of France is",  # Exact match
        "The capital of Russia is",  # Similar pattern
        "The capital of Brazil is",  # Similar pattern
    ]
    
    print("\n--- Test ---")
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        key = tuple(input_ids[0].tolist())
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        if key in lookup:
            pred_token = lookup[key]
            pred_text = tokenizer.decode([pred_token])
            true_text = tokenizer.decode([true_token])
            match = "✓" if pred_token == true_token else "✗"
            print(f"  {prompt!r} → FOUND: {pred_text!r}, true: {true_text!r} {match}")
        else:
            print(f"  {prompt!r} → NOT FOUND (would need generalization)")
    
    # The issue: lookup doesn't generalize
    # We need a way to find "similar" prompts
    
    print("\n--- Nearest Neighbor Lookup ---")
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Compute prompt embeddings (mean of token embeddings)
    prompt_embeds = {}
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        token_embeds = embed[input_ids[0]]
        prompt_embed = token_embeds.mean(dim=0)
        
        key = tuple(input_ids[0].tolist())
        prompt_embeds[key] = prompt_embed
    
    # Test nearest neighbor
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        token_embeds = embed[input_ids[0]]
        query_embed = token_embeds.mean(dim=0)
        
        # Find nearest
        best_key = None
        best_sim = -float('inf')
        for key, embed_vec in prompt_embeds.items():
            sim = F.cosine_similarity(query_embed.unsqueeze(0), embed_vec.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_key = key
        
        pred_token = lookup[best_key]
        pred_text = tokenizer.decode([pred_token])
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        match = "✓" if pred_token == true_token else "✗"
        
        nearest_prompt = tokenizer.decode(list(best_key))
        print(f"  {prompt!r}")
        print(f"    Nearest: {nearest_prompt!r} (sim={best_sim:.4f})")
        print(f"    Pred: {pred_text!r}, True: {true_text!r} {match}")


def analyze_hidden_state_components():
    """
    Decompose the hidden state to understand what it encodes.
    
    h_final = h_0 + Σ_layer delta_layer
    
    What does each layer contribute?
    """
    print("\n" + "=" * 70)
    print("Hidden State Component Analysis")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    lm_head = model.lm_head.weight.data.clone()
    
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # Get all layer hidden states
    all_h = [outputs.hidden_states[i][0, -1, :] for i in range(len(outputs.hidden_states))]
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Number of layers: {len(all_h) - 1}")
    
    # Compute deltas
    deltas = []
    for i in range(1, len(all_h)):
        delta = all_h[i] - all_h[i-1]
        deltas.append(delta)
    
    # Which layers contribute most to the final prediction?
    print("\n--- Layer Contribution to Final Prediction ---")
    
    true_token = outputs.logits[0, -1, :].argmax().item()
    true_text = tokenizer.decode([true_token])
    print(f"True next token: {true_text!r}")
    
    # Ablation: what if we remove each layer's contribution?
    print("\n--- Ablation: Remove Each Layer ---")
    
    for remove_layer in [0, 7, 14, 21, 27]:
        h_ablated = all_h[0].clone()
        for i, delta in enumerate(deltas):
            if i != remove_layer:
                h_ablated = h_ablated + delta
        
        pred = (h_ablated @ lm_head.T).argmax().item()
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        print(f"  Remove layer {remove_layer}: {pred_text!r} {match}")
    
    # What if we only use certain layers?
    print("\n--- Partial Layers ---")
    
    for n_layers in [1, 4, 8, 14, 21, 28]:
        h_partial = all_h[0].clone()
        for i in range(min(n_layers, len(deltas))):
            h_partial = h_partial + deltas[i]
        
        pred = (h_partial @ lm_head.T).argmax().item()
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        print(f"  First {n_layers} layers: {pred_text!r} {match}")
    
    # What if we scale the deltas?
    print("\n--- Scaled Deltas ---")
    
    for scale in [0.5, 1.0, 1.5, 2.0]:
        h_scaled = all_h[0].clone()
        for delta in deltas:
            h_scaled = h_scaled + scale * delta
        
        pred = (h_scaled @ lm_head.T).argmax().item()
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        print(f"  Scale {scale}: {pred_text!r} {match}")


if __name__ == "__main__":
    # Test 1: Direct hidden state mapping
    W = test_direct_hidden_mapping()
    
    # Test 2: Token sequence lookup
    test_token_sequence_lookup()
    
    # Test 3: Hidden state component analysis
    analyze_hidden_state_components()
