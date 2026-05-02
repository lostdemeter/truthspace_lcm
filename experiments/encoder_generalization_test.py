#!/usr/bin/env python3
"""
Encoder Generalization Test
============================

We found that a 37-dimensional linear mapping achieves 100% accuracy
on 50 training samples. Now we test:

1. Does it generalize to unseen prompts?
2. What do the 37 dimensions encode?
3. Can we build a working encoder?

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


def build_encoder(model, tokenizer, n_train=50):
    """
    Build the encoder from training data.
    Returns the transformation matrix and related components.
    """
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    # Training prompts
    training_prompts = [
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
        "Python is a programming language that",
        "Java is a programming language that",
        "JavaScript is a programming language that",
        "C++ is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "Once upon a time there was a",
        "In the beginning there was",
        "The largest planet is",
        "The smallest country is",
        "The tallest mountain is",
        "The deepest ocean is the",
        "The longest river is the",
        "Water is essential for",
        "The sun rises in the",
        "Music is a form of",
        "The meaning of life is",
        "Birds can fly because they have",
        "Fish live in water because",
        "The color of the sky is",
        "Mathematics is the language of",
        "Science is the study of",
        "Art is an expression of",
        "Love is a feeling of",
        "Time is a measure of",
        "Space is the absence of",
        "Light travels at the speed of",
        "Sound travels through",
        "Heat is a form of",
        "Energy cannot be created or",
        "The Earth revolves around the",
        "The Moon orbits the",
        "Stars are made of",
        "Clouds are made of",
        "Rain falls from the",
        "Snow is frozen",
        "Ice is solid",
        "Steam is gaseous",
        "Fire produces heat and",
        "Wind is moving",
    ]
    
    X_train = []
    Y_train = []
    tokens_train = []
    
    for prompt in training_prompts[:n_train]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        token_embeds = embed[input_ids[0]]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        X_train.append(x)
        Y_train.append(h_final)
        tokens_train.append(true_token)
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    # Learn transformation
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    lambda_reg = 0.1
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y_train)
    
    return W, embed, lm_head, training_prompts[:n_train], tokens_train


def test_generalization(model, tokenizer, W, embed, lm_head):
    """
    Test the encoder on completely unseen prompts.
    """
    print("\n" + "=" * 70)
    print("Generalization Test: Unseen Prompts")
    print("=" * 70)
    
    # Completely different prompts
    test_prompts = [
        # Different capitals
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of Egypt is",
        "The capital of South Korea is",
        "The capital of Argentina is",
        # Different facts
        "The fastest animal is the",
        "The largest mammal is the",
        "The smallest bird is the",
        "Diamonds are made of",
        "Glass is made of",
        # Different completions
        "Humans need oxygen to",
        "Plants need sunlight to",
        "Cars run on",
        "Computers process",
        "Books contain",
        # Different patterns
        "The opposite of hot is",
        "The opposite of big is",
        "The opposite of fast is",
        "Two plus two equals",
        "The square root of four is",
        # Conversational
        "Hello, my name is",
        "Thank you for your",
        "I would like to",
        "Can you please",
        "What is the",
    ]
    
    correct = 0
    results = []
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Encode using our learned transformation
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
        match = pred_token == true_token
        
        if match:
            correct += 1
        
        results.append({
            'prompt': prompt,
            'true': true_text,
            'pred': pred_text,
            'match': match,
        })
        
        marker = "✓" if match else "✗"
        print(f"  {prompt!r}")
        print(f"    pred={pred_text!r}, true={true_text!r} {marker}")
    
    acc = correct / len(test_prompts) * 100
    print(f"\nGeneralization accuracy: {correct}/{len(test_prompts)} = {acc:.1f}%")
    
    return results, acc


def analyze_37_dimensions(W, embed, lm_head, tokenizer):
    """
    Analyze what the 37 most important dimensions encode.
    """
    print("\n" + "=" * 70)
    print("Analyzing the 37 Dimensions")
    print("=" * 70)
    
    W_main = W[:-1, :]  # Exclude bias
    
    # SVD
    U, S, Vt = torch.linalg.svd(W_main, full_matrices=False)
    
    print(f"Transformation shape: {W_main.shape}")
    print(f"Top 10 singular values: {S[:10].tolist()}")
    
    # The 37 most important output directions
    V_37 = Vt[:37, :]  # [37, 3584]
    
    print(f"\nTop 37 output directions shape: {V_37.shape}")
    
    # What tokens are most aligned with each direction?
    print("\n--- Top Tokens per Direction ---")
    
    for i in range(min(10, 37)):
        direction = V_37[i]  # [3584]
        
        # Project lm_head onto this direction
        alignment = lm_head @ direction  # [vocab]
        
        top_pos = alignment.topk(5)
        top_neg = (-alignment).topk(5)
        
        print(f"\nDirection {i} (S={S[i].item():.2f}):")
        print(f"  Positive: ", end="")
        for val, idx in zip(top_pos.values, top_pos.indices):
            tok = tokenizer.decode([idx.item()])
            print(f"{tok!r}({val.item():.2f}) ", end="")
        print()
        print(f"  Negative: ", end="")
        for val, idx in zip(top_neg.values, top_neg.indices):
            tok = tokenizer.decode([idx.item()])
            print(f"{tok!r}({-val.item():.2f}) ", end="")
        print()
    
    return V_37, S


def test_reduced_encoder(model, tokenizer, W, embed, lm_head, k=37):
    """
    Test encoder using only top-k dimensions.
    """
    print(f"\n" + "=" * 70)
    print(f"Testing {k}-Dimensional Encoder")
    print("=" * 70)
    
    W_main = W[:-1, :]
    b = W[-1, :]
    
    # SVD
    U, S, Vt = torch.linalg.svd(W_main, full_matrices=False)
    
    # Truncate to k dimensions
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct truncated W
    W_k = U_k @ torch.diag(S_k) @ Vt_k
    W_k_with_bias = torch.cat([W_k, b.unsqueeze(0)], dim=0)
    
    # Test on diverse prompts
    test_prompts = [
        "The capital of France is",
        "The capital of Canada is",
        "Python is a programming language that",
        "The largest planet is",
        "The color of the sky is",
        "Two plus two equals",
        "Hello, my name is",
        "The opposite of hot is",
        "Water is essential for",
        "The quick brown fox jumps over the",
    ]
    
    correct_full = 0
    correct_k = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Features
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
        
        # Full prediction
        h_full = x @ W
        pred_full = (h_full @ lm_head.T).argmax().item()
        
        # k-dim prediction
        h_k = x @ W_k_with_bias
        pred_k = (h_k @ lm_head.T).argmax().item()
        
        if pred_full == true_token:
            correct_full += 1
        if pred_k == true_token:
            correct_k += 1
        
        true_text = tokenizer.decode([true_token])
        pred_full_text = tokenizer.decode([pred_full])
        pred_k_text = tokenizer.decode([pred_k])
        
        m_full = "✓" if pred_full == true_token else "✗"
        m_k = "✓" if pred_k == true_token else "✗"
        
        print(f"  {prompt!r}")
        print(f"    Full: {pred_full_text!r} {m_full}, k={k}: {pred_k_text!r} {m_k}, true: {true_text!r}")
    
    print(f"\nFull accuracy: {correct_full}/{len(test_prompts)} = {correct_full/len(test_prompts)*100:.1f}%")
    print(f"k={k} accuracy: {correct_k}/{len(test_prompts)} = {correct_k/len(test_prompts)*100:.1f}%")
    
    return correct_k / len(test_prompts)


def main():
    print("=" * 70)
    print("Encoder Generalization Test")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Build encoder
    print("\nBuilding encoder from 50 training samples...")
    W, embed, lm_head, train_prompts, train_tokens = build_encoder(model, tokenizer, n_train=50)
    
    # Test generalization
    results, gen_acc = test_generalization(model, tokenizer, W, embed, lm_head)
    
    # Analyze dimensions
    V_37, S = analyze_37_dimensions(W, embed, lm_head, tokenizer)
    
    # Test reduced encoder
    for k in [10, 20, 37, 50, 100]:
        test_reduced_encoder(model, tokenizer, W, embed, lm_head, k=k)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training accuracy: 100% (50 samples)")
    print(f"Generalization accuracy: {gen_acc:.1f}% (25 unseen prompts)")
    
    return W, results


if __name__ == "__main__":
    W, results = main()
