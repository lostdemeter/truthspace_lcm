#!/usr/bin/env python3
"""
Content Axis LUT: The DA2 Approach Applied
===========================================

Key discovery from scaffolding_signal_lut.py:
- The content signal is LOW-RANK (3 components capture 100%)
- The projection onto the "content axis" predicts the content token
- Correlation between prompt and content similarities: 0.8578

This is exactly like DA2:
- DA2: depth = features @ weights (32 weights)
- Here: content_signal = prompt_hidden @ W_content (low-rank)

The LUT approach:
1. Find the "content axis" (principal direction of content variation)
2. Project prompt hidden onto content axis → scalar "content code"
3. LUT: content_code → content_token

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

PHI = 1.6180339887498949


def build_content_axis_lut():
    """
    Build a LUT that maps prompt hidden → content token via content axis.
    """
    print("\n" + "=" * 70)
    print("Building Content Axis LUT")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Training data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
        "The capital of Sweden is",
        "The capital of Norway is",
        "The capital of Finland is",
        "The capital of Denmark is",
        "The capital of Austria is",
        "The capital of Belgium is",
    ]
    
    # Collect prompt hiddens and first tokens
    prompt_hiddens = []
    first_tokens = []
    first_hiddens = []  # Hidden state after generating first token
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
            first_token = outputs.logits[0, -1, :].argmax().item()
        
        prompt_hiddens.append(prompt_hidden)
        first_tokens.append(first_token)
        
        # Get hidden state after first token
        with torch.no_grad():
            extended_ids = torch.cat([input_ids, torch.tensor([[first_token]])], dim=1)
            outputs = model(extended_ids, output_hidden_states=True)
            first_hidden = outputs.hidden_states[-1][0, -1, :]
        
        first_hiddens.append(first_hidden)
        
        token_text = tokenizer.decode([first_token])
        print(f"  {prompt!r} → {token_text!r}")
    
    prompt_hiddens = torch.stack(prompt_hiddens)  # [n, hidden_dim]
    first_hiddens = torch.stack(first_hiddens)    # [n, hidden_dim]
    
    # Compute the "content signal" = first_hidden - mean(first_hidden)
    # This is the deviation from the average first hidden state
    mean_first_hidden = first_hiddens.mean(dim=0)
    content_signals = first_hiddens - mean_first_hidden  # [n, hidden_dim]
    
    # SVD to find content axis
    U, S, Vh = torch.linalg.svd(content_signals, full_matrices=False)
    
    print("\n--- Content Signal Structure ---")
    print("Singular values:")
    total_var = (S**2).sum()
    for i in range(min(5, len(S))):
        var = (S[i]**2 / total_var).item() * 100
        print(f"  S[{i}] = {S[i].item():.2f} ({var:.1f}%)")
    
    # The content axis is the first principal component
    content_axis = Vh[0, :]  # [hidden_dim]
    
    # Project prompt hiddens onto content axis
    prompt_projections = prompt_hiddens @ content_axis
    
    print("\n--- Prompt Projections onto Content Axis ---")
    for i, prompt in enumerate(train_prompts):
        token_text = tokenizer.decode([first_tokens[i]])
        print(f"  proj={prompt_projections[i].item():7.2f} → {token_text!r} ({prompt!r})")
    
    # Build LUT: projection → token
    # Sort by projection to see the pattern
    sorted_indices = torch.argsort(prompt_projections)
    
    print("\n--- Sorted by Projection ---")
    for idx in sorted_indices:
        i = idx.item()
        token_text = tokenizer.decode([first_tokens[i]])
        print(f"  proj={prompt_projections[i].item():7.2f} → {token_text!r}")
    
    # The LUT is a mapping from projection value to token
    # We can use nearest neighbor or interpolation
    
    lut = {
        "content_axis": content_axis,
        "projections": prompt_projections,
        "tokens": first_tokens,
        "prompts": train_prompts,
        "mean_first_hidden": mean_first_hidden,
    }
    
    # Test on training data
    print("\n--- LUT Test on Training Data ---")
    
    correct = 0
    for i, prompt in enumerate(train_prompts):
        proj = prompt_projections[i].item()
        
        # Find nearest projection
        dists = torch.abs(prompt_projections - proj)
        dists[i] = float('inf')  # Exclude self
        nearest_idx = dists.argmin().item()
        
        pred_token = first_tokens[nearest_idx]
        true_token = first_tokens[i]
        
        pred_text = tokenizer.decode([pred_token])
        true_text = tokenizer.decode([true_token])
        
        match = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r}: pred={pred_text!r}, true={true_text!r} {match}")
    
    print(f"\nTraining accuracy (leave-one-out): {correct}/{len(train_prompts)}")
    
    del model
    
    return lut


def test_content_axis_generalization():
    """
    Test if the content axis generalizes to unseen prompts.
    """
    print("\n" + "=" * 70)
    print("Content Axis Generalization Test")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Training data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    # Test data
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
        "The capital of Canada is",
        "The capital of Australia is",
    ]
    
    # Build content axis from training data
    train_hiddens = []
    train_first_hiddens = []
    train_tokens = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
            first_token = outputs.logits[0, -1, :].argmax().item()
        
        train_hiddens.append(prompt_hidden)
        train_tokens.append(first_token)
        
        with torch.no_grad():
            extended_ids = torch.cat([input_ids, torch.tensor([[first_token]])], dim=1)
            outputs = model(extended_ids, output_hidden_states=True)
            first_hidden = outputs.hidden_states[-1][0, -1, :]
        
        train_first_hiddens.append(first_hidden)
    
    train_hiddens = torch.stack(train_hiddens)
    train_first_hiddens = torch.stack(train_first_hiddens)
    
    # Content axis
    mean_first_hidden = train_first_hiddens.mean(dim=0)
    content_signals = train_first_hiddens - mean_first_hidden
    U, S, Vh = torch.linalg.svd(content_signals, full_matrices=False)
    content_axis = Vh[0, :]
    
    # Training projections
    train_projections = train_hiddens @ content_axis
    
    print("\n--- Training Data ---")
    for i, prompt in enumerate(train_prompts):
        token_text = tokenizer.decode([train_tokens[i]])
        print(f"  proj={train_projections[i].item():7.2f} → {token_text!r}")
    
    # Test
    print("\n--- Test Results ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Project onto content axis
        test_proj = (test_hidden @ content_axis).item()
        
        # Find nearest training projection
        dists = torch.abs(train_projections - test_proj)
        nearest_idx = dists.argmin().item()
        pred_token = train_tokens[nearest_idx]
        
        pred_text = tokenizer.decode([pred_token])
        true_text = tokenizer.decode([true_token])
        nearest_prompt = train_prompts[nearest_idx]
        
        match = "✓" if pred_token == true_token else "✗"
        
        print(f"\n  {test_prompt!r}")
        print(f"    proj={test_proj:7.2f}")
        print(f"    nearest: {nearest_prompt!r} (proj={train_projections[nearest_idx].item():.2f})")
        print(f"    pred={pred_text!r}, true={true_text!r} {match}")
    
    # The content axis doesn't generalize because:
    # 1. The content (capital cities) is not in the training data
    # 2. The projection doesn't encode the SPECIFIC city, just the pattern
    
    # But what if we use the content axis to INITIALIZE, then refine?
    print("\n--- Hybrid: Content Axis Init + Model Refinement ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Get true first token from model
        with torch.no_grad():
            outputs = model(input_ids)
            true_first_token = outputs.logits[0, -1, :].argmax().item()
        
        # Use content axis to predict scaffolding structure
        # The scaffolding is: [content] . It is the ...
        
        # Get scaffolding from nearest training example
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
        
        test_proj = (test_hidden @ content_axis).item()
        dists = torch.abs(train_projections - test_proj)
        nearest_idx = dists.argmin().item()
        
        # Generate full sequence from nearest training prompt to get scaffolding
        nearest_prompt = train_prompts[nearest_idx]
        nearest_ids = tokenizer.encode(nearest_prompt, return_tensors='pt')
        
        scaffolding_tokens = []
        with torch.no_grad():
            current_ids = nearest_ids.clone()
            for i in range(10):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                scaffolding_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        # Initialize: first token from model, rest from scaffolding
        current = [true_first_token] + scaffolding_tokens[1:]
        
        # Fixed-point refinement
        for iteration in range(15):
            full_ids = torch.cat([
                input_ids,
                torch.tensor([current])
            ], dim=1)
            
            with torch.no_grad():
                outputs = model(full_ids)
                logits = outputs.logits[0]
            
            new_current = []
            for i in range(10):
                if i == 0:
                    pos_logits = logits[prompt_len - 1]
                else:
                    pos_logits = logits[prompt_len + i - 1]
                new_current.append(pos_logits.argmax().item())
            
            if new_current == current:
                break
            current = new_current
        
        # Reference
        ref_tokens = []
        with torch.no_grad():
            current_ids = input_ids.clone()
            for i in range(10):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                ref_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
        
        ref_text = tokenizer.decode(ref_tokens)
        pred_text = tokenizer.decode(current)
        
        print(f"\n  {test_prompt!r}")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/10, Iters: {iteration + 1}")
    
    del model


def analyze_token_embedding_geometry():
    """
    Analyze the geometry of token embeddings to understand the content axis.
    
    Key question: Is the content axis aligned with token embedding differences?
    """
    print("\n" + "=" * 70)
    print("Token Embedding Geometry Analysis")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Get token embeddings for capital cities
    cities = [" Paris", " Berlin", " Rome", " Madrid", " Lisbon", " Warsaw", 
              " Stockholm", " Oslo", " Tokyo", " Beijing", " Brasilia", " Ottawa"]
    
    city_ids = [tokenizer.encode(city, add_special_tokens=False)[0] for city in cities]
    
    # Get embeddings
    embed_layer = model.model.embed_tokens
    city_embeddings = embed_layer(torch.tensor(city_ids))  # [n_cities, hidden_dim]
    
    print("\n--- City Embedding Similarities ---")
    city_norm = F.normalize(city_embeddings, dim=1)
    city_sim = city_norm @ city_norm.T
    
    print("Cities:", cities)
    print("\nSimilarity matrix:")
    print(city_sim.numpy().round(3))
    
    # SVD of city embeddings
    U, S, Vh = torch.linalg.svd(city_embeddings - city_embeddings.mean(dim=0), full_matrices=False)
    
    print("\n--- City Embedding Principal Components ---")
    total_var = (S**2).sum()
    for i in range(min(5, len(S))):
        var = (S[i]**2 / total_var).item() * 100
        print(f"  S[{i}] = {S[i].item():.2f} ({var:.1f}%)")
    
    # The "city axis" is the first principal component
    city_axis = Vh[0, :]
    
    # Project cities onto city axis
    city_projections = (city_embeddings - city_embeddings.mean(dim=0)) @ city_axis
    
    print("\n--- City Projections onto City Axis ---")
    sorted_indices = torch.argsort(city_projections)
    for idx in sorted_indices:
        i = idx.item()
        print(f"  proj={city_projections[i].item():7.2f} → {cities[i]!r}")
    
    # Now compare to the content axis from prompts
    print("\n--- Comparing Content Axis to City Axis ---")
    
    # Build content axis from prompts
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    train_hiddens = []
    train_first_hiddens = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
            first_token = outputs.logits[0, -1, :].argmax().item()
        
        train_hiddens.append(prompt_hidden)
        
        with torch.no_grad():
            extended_ids = torch.cat([input_ids, torch.tensor([[first_token]])], dim=1)
            outputs = model(extended_ids, output_hidden_states=True)
            first_hidden = outputs.hidden_states[-1][0, -1, :]
        
        train_first_hiddens.append(first_hidden)
    
    train_first_hiddens = torch.stack(train_first_hiddens)
    content_signals = train_first_hiddens - train_first_hiddens.mean(dim=0)
    U_c, S_c, Vh_c = torch.linalg.svd(content_signals, full_matrices=False)
    content_axis = Vh_c[0, :]
    
    # Cosine similarity between content axis and city axis
    cos_sim = F.cosine_similarity(content_axis.unsqueeze(0), city_axis.unsqueeze(0)).item()
    
    print(f"Cosine similarity between content axis and city axis: {cos_sim:.4f}")
    
    # The content axis should be related to the city embeddings!
    # Let's check if the content axis is a linear combination of city embeddings
    
    # Project content axis onto city embedding space
    city_basis = F.normalize(city_embeddings, dim=1)  # [n_cities, hidden_dim]
    content_in_city_space = content_axis @ city_basis.T  # [n_cities]
    
    print("\n--- Content Axis Projection onto City Embeddings ---")
    for i, city in enumerate(cities):
        print(f"  {city!r}: {content_in_city_space[i].item():.4f}")
    
    del model


if __name__ == "__main__":
    # 1. Build content axis LUT
    lut = build_content_axis_lut()
    
    # 2. Test generalization
    test_content_axis_generalization()
    
    # 3. Analyze token embedding geometry
    analyze_token_embedding_geometry()
