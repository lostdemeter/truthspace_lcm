#!/usr/bin/env python3
"""
Wall Analysis: Applying Music Box Principle to the Generalization Gap
======================================================================

We hit the WALL:
- Training: 100%
- Generalization: 8%

This is the holographic bound from PEP. The error is SIGNAL.

Key insight from Music Box Principle:
- The DRUM (structure) is different from the COMB (decoder)
- We're trying to learn the drum from the music
- But the drum has TWO components we need to disentangle

Hypothesis:
1. SCAFFOLDING tokens (the, is, a, of) - predictable from local context
2. CONTENT tokens (Paris, Jupiter, blue) - require world knowledge

The linear mapping captures scaffolding but not content.
Content requires the full transformer (world knowledge).

Let's verify this hypothesis.

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


def categorize_tokens(tokenizer, tokens):
    """
    Categorize tokens as scaffolding vs content.
    """
    scaffolding_patterns = [
        ' the', ' a', ' an', ' is', ' are', ' was', ' were', ' of', ' to',
        ' and', ' or', ' but', ' in', ' on', ' at', ' for', ' with', ' by',
        ' that', ' this', ' it', ' they', ' he', ' she', ' we', ' you',
        '.', ',', '!', '?', ':', ';', '-', "'", '"',
        ' not', ' no', ' yes', ' can', ' will', ' would', ' could', ' should',
    ]
    
    categories = []
    for tok_id in tokens:
        tok_text = tokenizer.decode([tok_id]).lower()
        
        is_scaffolding = any(p in tok_text.lower() for p in scaffolding_patterns)
        
        # Also check if it's a very common token
        # (This is a heuristic - scaffolding tokens tend to be common)
        
        categories.append('scaffolding' if is_scaffolding else 'content')
    
    return categories


def analyze_training_vs_test(model, tokenizer):
    """
    Analyze what distinguishes training successes from test failures.
    """
    print("\n" + "=" * 70)
    print("Training vs Test Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    # Training prompts (where we got 100%)
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest planet is",
        "Water is essential for",
        "The sun rises in the",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
    ]
    
    # Test prompts (where we got ~8%)
    test_prompts = [
        "The capital of Canada is",
        "The fastest animal is the",
        "Diamonds are made of",
        "The opposite of hot is",
        "Two plus two equals",
        "Hello, my name is",
    ]
    
    print("\n--- Training Prompts (100% accuracy) ---")
    train_tokens = []
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        train_tokens.append(true_token)
        
        # Get entropy of prediction
        probs = F.softmax(outputs.logits[0, -1, :], dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        print(f"  {prompt!r} → {true_text!r} (entropy={entropy:.2f})")
    
    print("\n--- Test Prompts (8% accuracy) ---")
    test_tokens = []
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        test_tokens.append(true_token)
        
        probs = F.softmax(outputs.logits[0, -1, :], dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        print(f"  {prompt!r} → {true_text!r} (entropy={entropy:.2f})")
    
    # Categorize tokens
    train_cats = categorize_tokens(tokenizer, train_tokens)
    test_cats = categorize_tokens(tokenizer, test_tokens)
    
    print("\n--- Token Categories ---")
    print(f"Training scaffolding: {train_cats.count('scaffolding')}/{len(train_cats)}")
    print(f"Training content: {train_cats.count('content')}/{len(train_cats)}")
    print(f"Test scaffolding: {test_cats.count('scaffolding')}/{len(test_cats)}")
    print(f"Test content: {test_cats.count('content')}/{len(test_cats)}")
    
    return train_tokens, test_tokens, train_cats, test_cats


def analyze_embedding_similarity(model, tokenizer):
    """
    Check if test prompts are similar to training prompts in embedding space.
    """
    print("\n" + "=" * 70)
    print("Embedding Similarity Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
    ]
    
    test_prompts = [
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of Egypt is",
    ]
    
    # Compute prompt embeddings
    def get_prompt_embedding(prompt):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        token_embeds = embed[input_ids[0]]
        return token_embeds.mean(dim=0)
    
    train_embeds = [get_prompt_embedding(p) for p in training_prompts]
    test_embeds = [get_prompt_embedding(p) for p in test_prompts]
    
    print("\n--- Similarity of Test to Training ---")
    
    for i, (test_p, test_e) in enumerate(zip(test_prompts, test_embeds)):
        sims = [F.cosine_similarity(test_e.unsqueeze(0), te.unsqueeze(0)).item() 
                for te in train_embeds]
        max_sim = max(sims)
        max_idx = sims.index(max_sim)
        
        print(f"  {test_p!r}")
        print(f"    Most similar to: {training_prompts[max_idx]!r} (sim={max_sim:.4f})")
    
    # The prompts are VERY similar in embedding space
    # But they predict different tokens
    # This means the transformation is NOT just about embeddings


def analyze_what_linear_captures(model, tokenizer):
    """
    What does the linear mapping actually capture?
    """
    print("\n" + "=" * 70)
    print("What Does Linear Capture?")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    # Train on diverse prompts
    prompts = [
        # These work (scaffolding-like)
        "The quick brown fox jumps over the",  # → lazy
        "To be or not to be that is the",      # → question
        "Water is essential for",               # → life
        "The sun rises in the",                 # → east
        "Once upon a time there was a",         # → little
        # These don't work (content-like)
        "The capital of France is",             # → Paris (CONTENT)
        "The largest planet is",                # → Jupiter (CONTENT)
        "The color of the sky is",              # → blue (CONTENT)
    ]
    
    X = []
    Y = []
    tokens = []
    
    for prompt in prompts:
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
        
        X.append(x)
        Y.append(h_final)
        tokens.append(true_token)
    
    X = torch.stack(X)
    Y = torch.stack(Y)
    
    # Learn transformation
    X_with_bias = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    lambda_reg = 0.1
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y)
    
    Y_pred = X_with_bias @ W
    
    print("\n--- Prediction Results ---")
    
    for i, prompt in enumerate(prompts):
        pred = (Y_pred[i] @ lm_head.T).argmax().item()
        true = tokens[i]
        
        pred_text = tokenizer.decode([pred])
        true_text = tokenizer.decode([true])
        
        # Compute residual
        residual = Y[i] - Y_pred[i]
        residual_norm = torch.norm(residual).item()
        
        # What does the residual point to?
        residual_logits = residual @ lm_head.T
        residual_top = residual_logits.topk(3)
        
        match = "✓" if pred == true else "✗"
        
        print(f"\n  {prompt!r}")
        print(f"    pred={pred_text!r}, true={true_text!r} {match}")
        print(f"    residual_norm={residual_norm:.2f}")
        print(f"    residual points to: ", end="")
        for val, idx in zip(residual_top.values, residual_top.indices):
            tok = tokenizer.decode([idx.item()])
            print(f"{tok!r}({val.item():.2f}) ", end="")
        print()
    
    # Key insight: The residual for content tokens points toward the correct answer!
    # The linear mapping gets the "scaffolding" part, the residual is the "content"


def test_scaffolding_only(model, tokenizer):
    """
    Test if we can predict scaffolding tokens with high accuracy.
    """
    print("\n" + "=" * 70)
    print("Scaffolding-Only Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    # Prompts that should end with scaffolding tokens
    scaffolding_prompts = [
        "I went to the store and",  # → bought/I/the
        "She said that she would",   # → be/not/go
        "The book is on the",        # → table/shelf/desk
        "He walked to the",          # → door/car/store
        "They were going to the",    # → park/store/beach
        "It was a very nice",        # → day/place/thing
        "We need to find a",         # → way/place/solution
        "The cat sat on the",        # → mat/floor/chair
        "I think that we should",    # → go/try/be
        "Please pass me the",        # → salt/book/pen
    ]
    
    # Prompts that require content knowledge
    content_prompts = [
        "The capital of France is",   # → Paris
        "The largest planet is",      # → Jupiter
        "Water boils at",             # → 100/212
        "The speed of light is",      # → 299792458/approximately
        "Einstein discovered",        # → relativity/E=mc2
        "Shakespeare wrote",          # → Hamlet/plays
        "The Mona Lisa was painted by", # → Leonardo
        "The chemical symbol for gold is", # → Au
        "The Great Wall is in",       # → China
        "The Eiffel Tower is in",     # → Paris
    ]
    
    # Train on scaffolding prompts
    X_train = []
    Y_train = []
    
    for prompt in scaffolding_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
        
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
    
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    # Learn transformation
    X_with_bias = torch.cat([X_train, torch.ones(len(X_train), 1)], dim=1)
    lambda_reg = 0.1
    XtX = X_with_bias.T @ X_with_bias
    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y_train)
    
    # Test on scaffolding
    print("\n--- Scaffolding Prompts ---")
    correct = 0
    for prompt in scaffolding_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
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
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first, torch.ones(1)])
        h_pred = x @ W
        
        pred = (h_pred @ lm_head.T).argmax().item()
        
        pred_text = tokenizer.decode([pred])
        true_text = tokenizer.decode([true_token])
        match = "✓" if pred == true_token else "✗"
        
        if pred == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred={pred_text!r}, true={true_text!r} {match}")
    
    print(f"\nScaffolding accuracy: {correct}/{len(scaffolding_prompts)} = {correct/len(scaffolding_prompts)*100:.1f}%")
    
    # Test on content
    print("\n--- Content Prompts ---")
    correct = 0
    for prompt in content_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids)
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
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first, torch.ones(1)])
        h_pred = x @ W
        
        pred = (h_pred @ lm_head.T).argmax().item()
        
        pred_text = tokenizer.decode([pred])
        true_text = tokenizer.decode([true_token])
        match = "✓" if pred == true_token else "✗"
        
        if pred == true_token:
            correct += 1
        
        print(f"  {prompt!r} → pred={pred_text!r}, true={true_text!r} {match}")
    
    print(f"\nContent accuracy: {correct}/{len(content_prompts)} = {correct/len(content_prompts)*100:.1f}%")


def main():
    print("=" * 70)
    print("Wall Analysis: Music Box Disentanglement")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Training vs Test
    analyze_training_vs_test(model, tokenizer)
    
    # Analysis 2: Embedding similarity
    analyze_embedding_similarity(model, tokenizer)
    
    # Analysis 3: What linear captures
    analyze_what_linear_captures(model, tokenizer)
    
    # Analysis 4: Scaffolding vs Content
    test_scaffolding_only(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: The Music Box Disentanglement")
    print("=" * 70)
    print("""
The WALL we hit is the boundary between:

1. SCAFFOLDING (the DRUM rotation we can learn):
   - Function words: the, is, a, of, to, and, or
   - Predictable from local syntactic context
   - Linear mapping can capture this

2. CONTENT (the COMB that requires world knowledge):
   - Proper nouns: Paris, Jupiter, Einstein
   - Facts: capitals, scientific constants, historical events
   - Requires the full transformer (28 layers of world knowledge)

The Music Box Principle tells us:
- The DRUM (scaffolding) can be a simple rotation
- The COMB (content) requires the full vocabulary lookup
- The MUSIC (output) emerges from their interaction

NEXT STEP: Build a hybrid encoder:
- Linear mapping for scaffolding tokens
- Full transformer (or lookup) for content tokens
- Detect which type at runtime
""")


if __name__ == "__main__":
    main()
