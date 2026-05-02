#!/usr/bin/env python3
"""
Hidden State Projection Analysis
=================================

Key finding from previous test:
- Full lm_head: 100% accuracy
- σ=0.5 sign projection: 0% accuracy (random!)

The hidden state lives in a DIFFERENT space than embeddings.
We need to find the RIGHT projection for hidden states.

Hypothesis: The hidden state is transformed by 28 layers.
The SVD of embeddings doesn't capture this transformation.

New approach:
1. Compute SVD of hidden states (not embeddings)
2. Find the projection that preserves lm_head accuracy
3. Analyze what dimensions matter for prediction

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


def analyze_hidden_state_space():
    """
    Analyze the space that hidden states live in.
    """
    print("\n" + "=" * 70)
    print("Analyzing Hidden State Space")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    lm_head = model.lm_head.weight.data.clone()  # [vocab, hidden]
    embed = model.model.embed_tokens.weight.data.clone()  # [vocab, hidden]
    
    # Collect hidden states from diverse prompts
    prompts = [
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
        "The meaning of life is",
        "Once upon a time there was a",
    ]
    
    hidden_states = []
    true_tokens = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            token = outputs.logits[0, -1, :].argmax().item()
        
        hidden_states.append(h)
        true_tokens.append(token)
    
    H = torch.stack(hidden_states)  # [n_prompts, hidden_dim]
    
    print(f"Hidden states shape: {H.shape}")
    
    # SVD of hidden states
    print("\n--- SVD of Hidden States ---")
    U_h, S_h, Vt_h = torch.linalg.svd(H, full_matrices=False)
    
    print(f"Singular values (top 10):")
    for i in range(min(10, len(S_h))):
        var_explained = (S_h[:i+1]**2).sum() / (S_h**2).sum() * 100
        print(f"  S[{i}] = {S_h[i].item():.2f} ({var_explained:.1f}% cumulative)")
    
    # SVD of lm_head
    print("\n--- SVD of LM Head ---")
    U_lm, S_lm, Vt_lm = torch.linalg.svd(lm_head, full_matrices=False)
    
    print(f"Singular values (top 10):")
    for i in range(min(10, len(S_lm))):
        var_explained = (S_lm[:i+1]**2).sum() / (S_lm**2).sum() * 100
        print(f"  S[{i}] = {S_lm[i].item():.2f} ({var_explained:.1f}% cumulative)")
    
    # Test: Project hidden states using lm_head's SVD
    print("\n--- Testing Projections ---")
    
    k_values = [10, 30, 60, 100, 200, 500, 1000, 3584]
    
    for k in k_values:
        if k > min(lm_head.shape):
            continue
        
        # Project using top-k of lm_head's right singular vectors
        proj = Vt_lm[:k, :].T  # [hidden, k]
        
        # Project hidden states
        H_proj = H @ proj  # [n, k]
        
        # Project lm_head
        lm_head_proj = lm_head @ proj  # [vocab, k]
        
        # Predict
        correct = 0
        for i, (h_proj, true_token) in enumerate(zip(H_proj, true_tokens)):
            logits = h_proj @ lm_head_proj.T
            pred = logits.argmax().item()
            if pred == true_token:
                correct += 1
        
        acc = correct / len(true_tokens) * 100
        print(f"  k={k:4d}: {correct}/{len(true_tokens)} = {acc:.1f}%")
    
    # What's the minimum k for 100% accuracy?
    print("\n--- Finding Minimum k for 100% ---")
    
    for k in range(1, 100):
        proj = Vt_lm[:k, :].T
        H_proj = H @ proj
        lm_head_proj = lm_head @ proj
        
        correct = 0
        for i, (h_proj, true_token) in enumerate(zip(H_proj, true_tokens)):
            logits = h_proj @ lm_head_proj.T
            pred = logits.argmax().item()
            if pred == true_token:
                correct += 1
        
        if correct == len(true_tokens):
            print(f"  Minimum k for 100%: {k}")
            break
    
    return model, tokenizer, H, true_tokens, Vt_lm


def test_lm_head_projection_encoder():
    """
    Test: Can we use lm_head's SVD as the projection for an encoder?
    """
    print("\n" + "=" * 70)
    print("LM Head Projection Encoder")
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
    embed = model.model.embed_tokens.weight.data.clone()
    
    # SVD of lm_head
    U_lm, S_lm, Vt_lm = torch.linalg.svd(lm_head, full_matrices=False)
    
    # Use top-k projection
    k = 60  # σ=0.5 equivalent
    proj = Vt_lm[:k, :].T  # [hidden, k]
    
    print(f"Using k={k} projection from lm_head SVD")
    
    # Project lm_head
    lm_head_proj = lm_head @ proj  # [vocab, k]
    
    # Project embeddings
    embed_proj = embed @ proj  # [vocab, k]
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest ocean is the",
        "Python is a programming language that",
        "The quick brown fox jumps over the",
    ]
    
    print("\n--- Direct Hidden State Projection ---")
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Project hidden state
        h_proj = h @ proj
        
        # Predict using projected lm_head
        logits = h_proj @ lm_head_proj.T
        pred = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    # Now test: Can we approximate h from embeddings?
    print("\n--- Approximating Hidden State from Embeddings ---")
    
    # Collect training data
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "The largest planet is",
        "The smallest country is",
        "Water is essential for",
        "The sun rises in the",
    ]
    
    X_train = []  # Projected embedding features
    Y_train = []  # Projected hidden states
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
        
        # Input features: projected embeddings
        token_embeds = embed[input_ids[0]]
        token_embeds_proj = token_embeds @ proj  # [seq_len, k]
        
        # Features: mean, last, weighted
        feat_mean = token_embeds_proj.mean(dim=0)
        feat_last = token_embeds_proj[-1]
        weights = torch.exp(torch.arange(len(token_embeds_proj)).float() / len(token_embeds_proj))
        weights = weights / weights.sum()
        feat_weighted = (token_embeds_proj * weights.unsqueeze(1)).sum(dim=0)
        
        x = torch.cat([feat_mean, feat_last, feat_weighted])
        
        # Output: projected hidden state
        h_proj = h @ proj
        
        X_train.append(x)
        Y_train.append(h_proj)
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    print(f"Training: {X_train.shape} -> {Y_train.shape}")
    
    # Learn linear mapping with ridge regression
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    lambda_reg = 0.1
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y_train)
    
    # Test on training data
    Y_pred = X_with_bias @ W
    
    correct = 0
    print("\n--- Training Set ---")
    for i, prompt in enumerate(training_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Predict from learned mapping
        logits = Y_pred[i] @ lm_head_proj.T
        pred = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        if pred == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTraining accuracy: {correct}/{len(training_prompts)} = {correct/len(training_prompts)*100:.1f}%")
    
    # Test on new prompts
    test_prompts = [
        "The capital of China is",
        "The largest ocean is the",
        "Music is a form of",
    ]
    
    correct = 0
    print("\n--- Test Set ---")
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Build features
        token_embeds = embed[input_ids[0]]
        token_embeds_proj = token_embeds @ proj
        
        feat_mean = token_embeds_proj.mean(dim=0)
        feat_last = token_embeds_proj[-1]
        weights = torch.exp(torch.arange(len(token_embeds_proj)).float() / len(token_embeds_proj))
        weights = weights / weights.sum()
        feat_weighted = (token_embeds_proj * weights.unsqueeze(1)).sum(dim=0)
        
        x = torch.cat([feat_mean, feat_last, feat_weighted, torch.ones(1)])
        h_pred = x @ W
        
        # Predict
        logits = h_pred @ lm_head_proj.T
        pred = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred])
        match = "✓" if pred == true_token else "✗"
        
        if pred == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred: {pred_text!r}, true: {true_text!r} {match}")
    
    print(f"\nTest accuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return W, proj, lm_head_proj


def analyze_what_transformer_does():
    """
    Analyze: What transformation does the transformer apply?
    
    If we can characterize this transformation, we can potentially
    replace it with a simpler operation.
    """
    print("\n" + "=" * 70)
    print("Analyzing Transformer Transformation")
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
        "Python is a programming language that",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        # Get hidden states at each layer
        all_h = [outputs.hidden_states[i][0, -1, :] for i in range(len(outputs.hidden_states))]
        
        print(f"\nPrompt: {prompt!r}")
        
        # Input embedding (layer 0)
        h0 = all_h[0]
        
        # Final hidden state (layer -1)
        h_final = all_h[-1]
        
        # What's the transformation?
        delta = h_final - h0
        
        print(f"  |h0|: {torch.norm(h0).item():.2f}")
        print(f"  |h_final|: {torch.norm(h_final).item():.2f}")
        print(f"  |delta|: {torch.norm(delta).item():.2f}")
        print(f"  cos(h0, h_final): {F.cosine_similarity(h0.unsqueeze(0), h_final.unsqueeze(0)).item():.4f}")
        
        # What does each predict?
        pred_h0 = (h0 @ lm_head.T).argmax().item()
        pred_final = (h_final @ lm_head.T).argmax().item()
        
        print(f"  h0 predicts: {tokenizer.decode([pred_h0])!r}")
        print(f"  h_final predicts: {tokenizer.decode([pred_final])!r}")
        
        # Track how prediction changes through layers
        print(f"  Layer progression:")
        prev_pred = None
        for i, h in enumerate(all_h[::4]):  # Every 4th layer
            pred = (h @ lm_head.T).argmax().item()
            pred_text = tokenizer.decode([pred])
            if pred != prev_pred:
                print(f"    Layer {i*4}: {pred_text!r}")
                prev_pred = pred


if __name__ == "__main__":
    # Test 1: Analyze hidden state space
    model, tokenizer, H, true_tokens, Vt_lm = analyze_hidden_state_space()
    
    # Test 2: LM head projection encoder
    W, proj, lm_head_proj = test_lm_head_projection_encoder()
    
    # Test 3: Analyze what transformer does
    analyze_what_transformer_does()
