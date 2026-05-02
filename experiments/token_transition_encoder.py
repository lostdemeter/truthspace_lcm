#!/usr/bin/env python3
"""
Token Transition Encoder: Learning Token-to-Token Mappings
============================================================

Key insight from failed experiment:
- Fixed points encode "I just generated X" (self-predicting)
- But we need "given context, predict Y" (transition)

New approach:
1. Build a TRANSITION matrix: P(next_token | current_token, context)
2. Use the prompt's hidden state directly (it already encodes context!)
3. The prompt hidden state IS the encoder output - we just need lm_head

The radical simplification:
- The model's job is: tokens → hidden_state → next_token
- The hidden state at position -1 already contains everything needed
- Can we approximate this hidden state from tokens alone?

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


def analyze_hidden_state_structure():
    """
    Analyze: What IS the hidden state? Can we reconstruct it from tokens?
    """
    print("\n" + "=" * 70)
    print("Analyzing Hidden State Structure")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Get embedding matrix
    embed = model.model.embed_tokens.weight.data.clone()  # [vocab_size, hidden_dim]
    lm_head = model.lm_head.weight.data.clone()  # [vocab_size, hidden_dim]
    
    print(f"Embedding shape: {embed.shape}")
    print(f"LM head shape: {lm_head.shape}")
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest ocean is the",
        "Python is a programming language that",
    ]
    
    print("\n" + "=" * 70)
    print("Test 1: Can sum of embeddings predict next token?")
    print("=" * 70)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Get true hidden state and prediction
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Sum of input embeddings
        token_embeds = embed[input_ids[0]]
        h_sum = token_embeds.sum(dim=0)
        
        # Weighted sum (more recent = higher weight)
        weights = torch.exp(torch.arange(len(input_ids[0])).float() / len(input_ids[0]))
        weights = weights / weights.sum()
        h_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        
        # Predict from sum
        pred_sum = (h_sum @ lm_head.T).argmax().item()
        pred_weighted = (h_weighted @ lm_head.T).argmax().item()
        
        # Similarity between h_true and h_sum
        sim_sum = F.cosine_similarity(h_true.unsqueeze(0), h_sum.unsqueeze(0)).item()
        sim_weighted = F.cosine_similarity(h_true.unsqueeze(0), h_weighted.unsqueeze(0)).item()
        
        true_text = tokenizer.decode([true_token])
        pred_sum_text = tokenizer.decode([pred_sum])
        pred_weighted_text = tokenizer.decode([pred_weighted])
        
        print(f"\nPrompt: {prompt!r}")
        print(f"  True: {true_text!r}")
        print(f"  Sum embed: {pred_sum_text!r} (sim={sim_sum:.4f}) {'✓' if pred_sum == true_token else '✗'}")
        print(f"  Weighted:  {pred_weighted_text!r} (sim={sim_weighted:.4f}) {'✓' if pred_weighted == true_token else '✗'}")
    
    print("\n" + "=" * 70)
    print("Test 2: What's the relationship between h_true and embeddings?")
    print("=" * 70)
    
    for prompt in prompts[:2]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
        
        # Find nearest embedding to h_true
        sims = F.cosine_similarity(h_true.unsqueeze(0), embed, dim=1)
        top_k = sims.topk(5)
        
        print(f"\nPrompt: {prompt!r}")
        print(f"  Top 5 nearest embeddings to h_true:")
        for i, (sim, idx) in enumerate(zip(top_k.values, top_k.indices)):
            text = tokenizer.decode([idx.item()])
            print(f"    {i+1}. {text!r} (sim={sim.item():.4f})")
    
    print("\n" + "=" * 70)
    print("Test 3: Linear probe - can we learn h_true from embeddings?")
    print("=" * 70)
    
    # Collect training data
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "The largest planet is",
        "The smallest country is",
        "Water is essential for",
    ]
    
    X_train = []  # Sum of embeddings
    Y_train = []  # True hidden states
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
        
        token_embeds = embed[input_ids[0]]
        h_sum = token_embeds.mean(dim=0)  # Use mean instead of sum
        
        X_train.append(h_sum)
        Y_train.append(h_true)
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    print(f"Training data: {X_train.shape} -> {Y_train.shape}")
    
    # Learn linear transformation: Y = X @ W + b
    # Using least squares: W = (X^T X)^-1 X^T Y
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    W, _, _, _ = torch.linalg.lstsq(X_with_bias, Y_train)
    
    # Test on training data
    Y_pred = X_with_bias @ W
    
    # Measure accuracy
    correct = 0
    for i, prompt in enumerate(training_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        pred_token = (Y_pred[i] @ lm_head.T).argmax().item()
        
        if pred_token == true_token:
            correct += 1
    
    print(f"Linear probe accuracy (train): {correct}/{len(training_prompts)} = {correct/len(training_prompts)*100:.1f}%")
    
    # Test on new prompts
    test_prompts = [
        "The capital of China is",
        "The largest ocean is the",
        "Music is a form of",
    ]
    
    correct = 0
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        token_embeds = embed[input_ids[0]]
        h_sum = token_embeds.mean(dim=0)
        x_with_bias = torch.cat([h_sum, torch.ones(1)])
        h_pred = x_with_bias @ W
        
        pred_token = (h_pred @ lm_head.T).argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        
        match = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} → true: {true_text!r}, pred: {pred_text!r} {match}")
    
    print(f"Linear probe accuracy (test): {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return model, tokenizer, embed, lm_head


def test_direct_lm_head():
    """
    The simplest possible approach: just use lm_head on the last embedding.
    """
    print("\n" + "=" * 70)
    print("Test: Direct lm_head on last token embedding")
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
    
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest ocean is the",
        "Python is a programming language that",
        "The quick brown fox jumps over the",
    ]
    
    print("\n--- Direct embedding → lm_head ---")
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        last_token = input_ids[0, -1].item()
        
        # Get true prediction
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Direct: last embedding → lm_head
        last_embed = embed[last_token]
        pred_direct = (last_embed @ lm_head.T).argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_direct])
        last_text = tokenizer.decode([last_token])
        
        match = "✓" if pred_direct == true_token else "✗"
        print(f"  {prompt!r}")
        print(f"    Last token: {last_text!r} → Direct pred: {pred_text!r}, True: {true_text!r} {match}")


def test_bigram_baseline():
    """
    Bigram baseline: P(next | current) from co-occurrence statistics.
    """
    print("\n" + "=" * 70)
    print("Test: Bigram Baseline (Token Co-occurrence)")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect bigram statistics from generation
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The elephant is a large animal that",
        "The lion is a large animal that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "In the beginning there was",
        "Once upon a time there was a",
        "The meaning of life is",
        "Mathematics is the language of",
        "The sun rises in the",
        "Water is essential for",
        "The largest planet is",
        "The smallest country is",
    ]
    
    # Build bigram counts
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        current_ids = input_ids.clone()
        
        for i in range(15):
            with torch.no_grad():
                outputs = model(current_ids)
                next_token = outputs.logits[0, -1, :].argmax().item()
            
            current_token = current_ids[0, -1].item()
            bigram_counts[current_token][next_token] += 1
            
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]])], dim=1)
    
    print(f"Collected {len(bigram_counts)} unique current tokens")
    
    # Test
    test_prompts = [
        "The capital of France is",
        "The largest ocean is the",
        "Python is a programming language that",
        "The sun rises in the",
    ]
    
    correct = 0
    total = 0
    
    print("\n--- Bigram Predictions ---")
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        current_token = input_ids[0, -1].item()
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Bigram prediction
        if current_token in bigram_counts and bigram_counts[current_token]:
            pred_token = max(bigram_counts[current_token].items(), key=lambda x: x[1])[0]
        else:
            pred_token = -1
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token]) if pred_token >= 0 else "N/A"
        current_text = tokenizer.decode([current_token])
        
        match = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        total += 1
        
        print(f"  {prompt!r}")
        print(f"    Current: {current_text!r} → Bigram: {pred_text!r}, True: {true_text!r} {match}")
    
    print(f"\nBigram accuracy: {correct}/{total} = {correct/total*100:.1f}%")


def test_sequence_embedding_encoder():
    """
    New approach: Learn a sequence encoder that maps token sequence to hidden state.
    
    The key insight: The transformer IS the encoder. We need to approximate it.
    """
    print("\n" + "=" * 70)
    print("Test: Sequence Embedding Encoder")
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
    
    # Collect (sequence, hidden_state) pairs
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "The largest planet is",
        "The smallest country is",
        "Water is essential for",
        "The sun rises in the",
        "Music is a form of",
    ]
    
    # Build training data with position-weighted embeddings
    X_train = []
    Y_train = []
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
        
        # Position-weighted embedding sum with exponential decay
        token_embeds = embed[input_ids[0]]
        positions = torch.arange(seq_len).float()
        
        # Multiple features:
        # 1. Mean embedding
        feat_mean = token_embeds.mean(dim=0)
        
        # 2. Last embedding
        feat_last = token_embeds[-1]
        
        # 3. Exponential weighted
        weights = torch.exp(positions / seq_len)
        weights = weights / weights.sum()
        feat_exp = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        
        # 4. First embedding
        feat_first = token_embeds[0]
        
        # Concatenate features
        x = torch.cat([feat_mean, feat_last, feat_exp, feat_first])
        
        X_train.append(x)
        Y_train.append(h_true)
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    print(f"Training data: {X_train.shape} -> {Y_train.shape}")
    
    # Learn linear transformation with regularization
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    
    # Ridge regression: W = (X^T X + λI)^-1 X^T Y
    lambda_reg = 0.01
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y_train)
    
    # Test on training data
    Y_pred = X_with_bias @ W
    
    correct = 0
    print("\n--- Training Set Results ---")
    for i, prompt in enumerate(training_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        pred_token = (Y_pred[i] @ lm_head.T).argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        
        match = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTraining accuracy: {correct}/{len(training_prompts)} = {correct/len(training_prompts)*100:.1f}%")
    
    # Test on new prompts
    test_prompts = [
        "The capital of China is",
        "The largest ocean is the",
        "The meaning of life is",
    ]
    
    correct = 0
    print("\n--- Test Set Results ---")
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Build features
        token_embeds = embed[input_ids[0]]
        positions = torch.arange(seq_len).float()
        
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(positions / seq_len)
        weights = weights / weights.sum()
        feat_exp = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_mean, feat_last, feat_exp, feat_first, torch.ones(1)])
        h_pred = x @ W
        
        pred_token = (h_pred @ lm_head.T).argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        
        match = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTest accuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return W


if __name__ == "__main__":
    # Test 1: Analyze hidden state structure
    analyze_hidden_state_structure()
    
    # Test 2: Direct lm_head
    test_direct_lm_head()
    
    # Test 3: Bigram baseline
    test_bigram_baseline()
    
    # Test 4: Sequence embedding encoder
    test_sequence_embedding_encoder()
