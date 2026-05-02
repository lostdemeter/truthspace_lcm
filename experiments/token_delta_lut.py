#!/usr/bin/env python3
"""
Token → Delta LUT: The Output Embedding
========================================

Key discovery: When the same token appears, the delta (hidden state transformation)
has LOW variance (1.05-2.14). This means each token has a characteristic
transformation it applies to the hidden state.

This is like an "output embedding" - the inverse of the input embedding:
- Input embedding: token → hidden state contribution
- Output embedding (delta): token → hidden state transformation

If we can build this LUT, we can predict trajectories without running the model!

Hypothesis: delta[token] ≈ f(token_embedding, lm_head_weights)

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

PHI = 1.6180339887498949


def build_token_delta_lut():
    """
    Build a comprehensive Token → Delta LUT by collecting deltas
    across many different prompts and contexts.
    """
    print("\n" + "=" * 70)
    print("Building Token → Delta LUT")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Diverse prompts to collect token deltas
    prompts = [
        # Capital cities
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        # Programming
        "Python is a programming language that",
        "Java is a programming language that",
        # Animals
        "The elephant is a large animal that",
        "The lion is a large animal that",
        # General
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "In the beginning there was",
        "Once upon a time there was a",
    ]
    
    n_tokens = 10
    
    # Collect (token, delta, context) tuples
    token_deltas = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_prev = outputs.hidden_states[-1][0, -1, :].clone()
        
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                h_curr = outputs.hidden_states[-1][0, -1, :].clone()
                
                token = outputs.logits[0, -1, :].argmax().item()
            
            # Compute delta
            if i > 0:  # Skip first position (delta from prompt)
                delta = h_curr - h_prev
                token_deltas[token].append({
                    "delta": delta,
                    "h_before": h_prev.clone(),
                    "prompt": prompt,
                    "position": i,
                })
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # Analyze the LUT
    print(f"\n--- Token → Delta LUT Statistics ---")
    print(f"Unique tokens collected: {len(token_deltas)}")
    
    # Sort by frequency
    sorted_tokens = sorted(token_deltas.keys(), key=lambda t: len(token_deltas[t]), reverse=True)
    
    print("\nTop 20 most frequent tokens:")
    for token in sorted_tokens[:20]:
        entries = token_deltas[token]
        token_text = tokenizer.decode([token])
        
        if len(entries) > 1:
            deltas = torch.stack([e["delta"] for e in entries])
            
            # Compute variance
            var = torch.var(deltas, dim=0).mean().item()
            
            # Compute mean delta
            mean_delta = deltas.mean(dim=0)
            mean_norm = torch.norm(mean_delta).item()
            
            # Compute pairwise cosine similarities
            deltas_norm = F.normalize(deltas, dim=1)
            cos_sim = (deltas_norm @ deltas_norm.T)
            # Get off-diagonal elements
            mask = ~torch.eye(len(entries), dtype=torch.bool)
            mean_cos = cos_sim[mask].mean().item()
            
            print(f"  {token_text!r:15} n={len(entries):2}, var={var:6.2f}, |Δ|={mean_norm:6.2f}, cos={mean_cos:.4f}")
        else:
            delta = entries[0]["delta"]
            norm = torch.norm(delta).item()
            print(f"  {token_text!r:15} n={len(entries):2}, |Δ|={norm:6.2f}")
    
    # Build the LUT: token → mean_delta
    lut = {}
    for token, entries in token_deltas.items():
        deltas = torch.stack([e["delta"] for e in entries])
        lut[token] = {
            "mean_delta": deltas.mean(dim=0),
            "std": deltas.std(dim=0) if len(entries) > 1 else torch.zeros_like(deltas[0]),
            "n": len(entries),
        }
    
    del model
    
    return lut, token_deltas


def analyze_delta_structure():
    """
    Analyze the structure of token deltas.
    
    Key questions:
    1. Is delta related to the token embedding?
    2. Is delta related to the lm_head weights?
    3. Is there a simple formula: delta = f(embedding, lm_head)?
    """
    print("\n" + "=" * 70)
    print("Analyzing Delta Structure")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    embed_layer = model.model.embed_tokens
    lm_head = model.lm_head
    
    # Collect deltas for common tokens
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    n_tokens = 10
    
    token_deltas = defaultdict(list)
    token_h_befores = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_prev = outputs.hidden_states[-1][0, -1, :].clone()
        
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                h_curr = outputs.hidden_states[-1][0, -1, :].clone()
                token = outputs.logits[0, -1, :].argmax().item()
            
            if i > 0:
                delta = h_curr - h_prev
                token_deltas[token].append(delta)
                token_h_befores[token].append(h_prev.clone())
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # For tokens with multiple samples, analyze the relationship
    print("\n--- Delta vs Embedding Analysis ---")
    
    for token in list(token_deltas.keys())[:10]:
        if len(token_deltas[token]) < 2:
            continue
        
        token_text = tokenizer.decode([token])
        deltas = torch.stack(token_deltas[token])
        h_befores = torch.stack(token_h_befores[token])
        
        # Get token embedding
        with torch.no_grad():
            embedding = embed_layer(torch.tensor([token]))[0]
        
        # Get lm_head row for this token
        with torch.no_grad():
            lm_head_row = lm_head.weight[token]
        
        # Compute mean delta
        mean_delta = deltas.mean(dim=0)
        
        # Cosine similarity between mean_delta and embedding
        cos_embed = F.cosine_similarity(mean_delta.unsqueeze(0), embedding.unsqueeze(0)).item()
        
        # Cosine similarity between mean_delta and lm_head_row
        cos_lm = F.cosine_similarity(mean_delta.unsqueeze(0), lm_head_row.unsqueeze(0)).item()
        
        # Cosine similarity between embedding and lm_head_row
        cos_embed_lm = F.cosine_similarity(embedding.unsqueeze(0), lm_head_row.unsqueeze(0)).item()
        
        print(f"\n  Token: {token_text!r}")
        print(f"    cos(delta, embed) = {cos_embed:.4f}")
        print(f"    cos(delta, lm_head) = {cos_lm:.4f}")
        print(f"    cos(embed, lm_head) = {cos_embed_lm:.4f}")
        
        # Is delta a linear combination of embedding and h_before?
        # delta ≈ α * embedding + β * h_before
        
        # For each sample, compute the projection
        for j in range(min(3, len(deltas))):
            d = deltas[j]
            h = h_befores[j]
            e = embedding
            
            # Project delta onto embedding
            proj_e = (d @ e) / (e @ e) * e
            residual_e = d - proj_e
            
            # Project residual onto h_before
            proj_h = (residual_e @ h) / (h @ h) * h
            residual_h = residual_e - proj_h
            
            # How much is explained?
            explained = 1 - (torch.norm(residual_h) / torch.norm(d)).item()
            
            print(f"    Sample {j}: {explained*100:.1f}% explained by (embed, h_before)")
    
    # Global analysis: Is there a universal delta formula?
    print("\n--- Universal Delta Formula Search ---")
    
    # Collect all (token, delta, h_before) tuples
    all_tokens = []
    all_deltas = []
    all_h_befores = []
    all_embeddings = []
    
    for token, deltas in token_deltas.items():
        for i, delta in enumerate(deltas):
            all_tokens.append(token)
            all_deltas.append(delta)
            all_h_befores.append(token_h_befores[token][i])
            with torch.no_grad():
                all_embeddings.append(embed_layer(torch.tensor([token]))[0])
    
    all_deltas = torch.stack(all_deltas)
    all_h_befores = torch.stack(all_h_befores)
    all_embeddings = torch.stack(all_embeddings)
    
    print(f"Total samples: {len(all_deltas)}")
    
    # Try: delta = α * embedding + β * h_before + γ
    # This is a linear regression problem
    
    # Build design matrix: [embedding, h_before, 1]
    ones = torch.ones(len(all_deltas), 1)
    X = torch.cat([all_embeddings, all_h_befores, ones], dim=1)  # [n, 2*hidden + 1]
    Y = all_deltas  # [n, hidden]
    
    print(f"Design matrix shape: {X.shape}")
    print(f"Target shape: {Y.shape}")
    
    # This is too large to solve directly. Let's try a simpler model.
    
    # Simpler model: delta = α * embedding
    # Solve for α: α = (embedding.T @ delta) / (embedding.T @ embedding)
    
    alphas = []
    for i in range(len(all_deltas)):
        e = all_embeddings[i]
        d = all_deltas[i]
        alpha = (e @ d) / (e @ e)
        alphas.append(alpha.item())
    
    alphas = np.array(alphas)
    print(f"\nSimple model: delta = α * embedding")
    print(f"  α mean: {alphas.mean():.4f}")
    print(f"  α std: {alphas.std():.4f}")
    print(f"  α range: [{alphas.min():.4f}, {alphas.max():.4f}]")
    
    # Test this model
    pred_deltas = alphas.mean() * all_embeddings
    
    # Compute correlation
    cos_sims = []
    for i in range(len(all_deltas)):
        cos = F.cosine_similarity(pred_deltas[i].unsqueeze(0), all_deltas[i].unsqueeze(0)).item()
        cos_sims.append(cos)
    
    print(f"  Mean cosine similarity: {np.mean(cos_sims):.4f}")
    
    del model
    
    return


def test_delta_lut_prediction():
    """
    Test: Can we predict trajectories using the Token → Delta LUT?
    
    Approach:
    1. Get first token from model (content)
    2. Look up delta for that token
    3. Apply delta to get next hidden state
    4. Predict next token from hidden state
    5. Repeat
    """
    print("\n" + "=" * 70)
    print("Testing Delta LUT Prediction")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    lm_head = model.lm_head
    
    # Build LUT from training prompts
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    n_tokens = 10
    
    token_deltas = defaultdict(list)
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_prev = outputs.hidden_states[-1][0, -1, :].clone()
        
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                h_curr = outputs.hidden_states[-1][0, -1, :].clone()
                token = outputs.logits[0, -1, :].argmax().item()
            
            delta = h_curr - h_prev
            token_deltas[token].append(delta)
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # Build LUT
    lut = {}
    for token, deltas in token_deltas.items():
        lut[token] = torch.stack(deltas).mean(dim=0)
    
    print(f"LUT size: {len(lut)} tokens")
    
    # Test on unseen prompts
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
    ]
    
    print("\n--- LUT Prediction Results ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        # Get initial hidden state
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :].clone()
        
        # Predict using LUT
        pred_tokens = []
        
        for i in range(n_tokens):
            # Predict token from current hidden state
            with torch.no_grad():
                logits = lm_head(h)
                token = logits.argmax().item()
            
            pred_tokens.append(token)
            
            # Look up delta for this token
            if token in lut:
                delta = lut[token]
            else:
                # Token not in LUT - use mean of all deltas
                delta = torch.stack(list(lut.values())).mean(dim=0)
            
            # Apply delta
            h = h + delta
        
        # Reference (sequential generation)
        ref_tokens = []
        with torch.no_grad():
            current_ids = input_ids.clone()
            for i in range(n_tokens):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                ref_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        matches = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
        
        ref_text = tokenizer.decode(ref_tokens)
        pred_text = tokenizer.decode(pred_tokens)
        
        print(f"\n  {test_prompt!r}")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
        
        # Analyze where it goes wrong
        print("    Token-by-token:")
        for i in range(n_tokens):
            ref_t = tokenizer.decode([ref_tokens[i]])
            pred_t = tokenizer.decode([pred_tokens[i]])
            in_lut = "✓" if pred_tokens[i] in lut else "✗"
            match = "=" if ref_tokens[i] == pred_tokens[i] else "≠"
            print(f"      {i}: ref={ref_t!r:10} {match} pred={pred_t!r:10} (in LUT: {in_lut})")
    
    del model
    
    return lut


def analyze_delta_context_dependence():
    """
    Analyze: Does the delta depend on context (h_before) or just the token?
    
    If delta depends only on token, we can use a simple LUT.
    If delta depends on context, we need a more complex model.
    """
    print("\n" + "=" * 70)
    print("Analyzing Delta Context Dependence")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect deltas for the same token in different contexts
    # Focus on common tokens that appear in many contexts
    
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The elephant is a large animal that",
        "The lion is a large animal that",
    ]
    
    n_tokens = 10
    
    token_data = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_prev = outputs.hidden_states[-1][0, -1, :].clone()
        
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True)
                h_curr = outputs.hidden_states[-1][0, -1, :].clone()
                token = outputs.logits[0, -1, :].argmax().item()
            
            delta = h_curr - h_prev
            
            token_data[token].append({
                "delta": delta,
                "h_before": h_prev.clone(),
                "h_after": h_curr.clone(),
                "prompt": prompt,
                "position": i,
            })
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # For tokens with multiple samples, analyze context dependence
    print("\n--- Context Dependence Analysis ---")
    
    for token in sorted(token_data.keys(), key=lambda t: len(token_data[t]), reverse=True)[:15]:
        entries = token_data[token]
        if len(entries) < 2:
            continue
        
        token_text = tokenizer.decode([token])
        
        deltas = torch.stack([e["delta"] for e in entries])
        h_befores = torch.stack([e["h_before"] for e in entries])
        
        # Variance of deltas
        delta_var = torch.var(deltas, dim=0).mean().item()
        
        # Variance of h_befores
        h_var = torch.var(h_befores, dim=0).mean().item()
        
        # Correlation between delta and h_before
        # For each pair, compute cos(delta_i - delta_j, h_before_i - h_before_j)
        correlations = []
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                delta_diff = deltas[i] - deltas[j]
                h_diff = h_befores[i] - h_befores[j]
                
                if torch.norm(delta_diff) > 1e-6 and torch.norm(h_diff) > 1e-6:
                    cos = F.cosine_similarity(delta_diff.unsqueeze(0), h_diff.unsqueeze(0)).item()
                    correlations.append(cos)
        
        mean_corr = np.mean(correlations) if correlations else 0
        
        # Pairwise delta similarity
        deltas_norm = F.normalize(deltas, dim=1)
        delta_sim = (deltas_norm @ deltas_norm.T)
        mask = ~torch.eye(len(entries), dtype=torch.bool)
        mean_delta_sim = delta_sim[mask].mean().item()
        
        print(f"\n  Token: {token_text!r} (n={len(entries)})")
        print(f"    Delta variance: {delta_var:.2f}")
        print(f"    h_before variance: {h_var:.2f}")
        print(f"    Delta pairwise similarity: {mean_delta_sim:.4f}")
        print(f"    Delta-context correlation: {mean_corr:.4f}")
        
        # Show contexts
        contexts = set(e["prompt"][:30] for e in entries)
        print(f"    Contexts: {contexts}")
    
    del model


if __name__ == "__main__":
    # 1. Build comprehensive Token → Delta LUT
    lut, token_deltas = build_token_delta_lut()
    
    # 2. Analyze delta structure
    analyze_delta_structure()
    
    # 3. Test LUT prediction
    lut = test_delta_lut_prediction()
    
    # 4. Analyze context dependence
    analyze_delta_context_dependence()
