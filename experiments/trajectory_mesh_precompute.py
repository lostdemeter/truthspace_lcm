#!/usr/bin/env python3
"""
Trajectory MESH Pre-computation
================================

Key insight: The transformer is nonlinear, but the PATTERN of transformation
might be consistent across similar prompts.

Like the φ-Unraveled Engine pre-computes MESH = W_q.T @ W_k,
we can pre-compute the "trajectory transformation" for common patterns.

Approach:
1. For a pattern (e.g., "The capital of X is"), collect many examples
2. Compute the AVERAGE transformation at each position
3. For a new prompt, apply the average transformation

This is like computing the "expected trajectory" and using it as a prior.

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


def compute_trajectory_mesh():
    """
    Compute the "trajectory MESH" - the average transformation pattern.
    
    For each position i, we compute:
        Δ[i] = h[i] - h[i-1]  (the transformation)
    
    Then we average across prompts to get the "expected" transformation.
    """
    print("\n" + "=" * 70)
    print("Computing Trajectory MESH")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Training prompts (same pattern)
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
        "The capital of Sweden is",
        "The capital of Norway is",
    ]
    
    n_tokens = 10
    
    # Collect trajectories
    all_trajectories = []
    all_tokens = []
    all_deltas = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h0 = outputs.hidden_states[-1][0, -1, :]
        
        trajectory = [h0]
        tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                token = outputs.logits[0, -1, :].argmax().item()
                tokens.append(token)
                
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        trajectory = torch.stack(trajectory)  # [n_tokens+1, hidden_dim]
        
        # Compute deltas
        deltas = trajectory[1:] - trajectory[:-1]  # [n_tokens, hidden_dim]
        
        all_trajectories.append(trajectory)
        all_tokens.append(tokens)
        all_deltas.append(deltas)
        
        print(f"  {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    # Stack
    all_trajectories = torch.stack(all_trajectories)  # [n_prompts, n_tokens+1, hidden_dim]
    all_deltas = torch.stack(all_deltas)  # [n_prompts, n_tokens, hidden_dim]
    
    # Compute average delta (the "MESH")
    mean_delta = all_deltas.mean(dim=0)  # [n_tokens, hidden_dim]
    
    print(f"\nMean delta shape: {mean_delta.shape}")
    
    # Analyze variance
    print("\n--- Delta Variance Analysis ---")
    
    for i in range(n_tokens):
        delta_at_i = all_deltas[:, i, :]  # [n_prompts, hidden_dim]
        
        # Variance
        var = torch.var(delta_at_i, dim=0).mean().item()
        
        # Mean delta norm
        mean_norm = torch.norm(mean_delta[i]).item()
        
        # What tokens appear at this position?
        tokens_at_i = [all_tokens[j][i] for j in range(len(train_prompts))]
        unique_tokens = set(tokens_at_i)
        
        if len(unique_tokens) == 1:
            token_text = tokenizer.decode([tokens_at_i[0]])
            print(f"  Pos {i}: var={var:.2f}, |Δ|={mean_norm:.2f}, SCAFFOLDING ({token_text!r})")
        else:
            token_texts = [tokenizer.decode([t]) for t in list(unique_tokens)[:3]]
            print(f"  Pos {i}: var={var:.2f}, |Δ|={mean_norm:.2f}, CONTENT ({token_texts}...)")
    
    # The MESH is the mean delta
    mesh = {
        "mean_delta": mean_delta,
        "mean_h0": all_trajectories[:, 0, :].mean(dim=0),
    }
    
    del model
    
    return mesh, all_trajectories, all_tokens, train_prompts


def test_mesh_prediction():
    """
    Test: Can we predict trajectories using the MESH?
    
    Approach:
    1. Start with prompt hidden state h0
    2. Apply mean_delta[i] to get h[i]: h[i] = h[i-1] + mean_delta[i]
    3. Extract tokens from predicted hidden states
    """
    print("\n" + "=" * 70)
    print("Testing MESH Prediction")
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
    
    # Build MESH from training data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    n_tokens = 10
    
    all_deltas = []
    all_h0s = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h0 = outputs.hidden_states[-1][0, -1, :]
        
        all_h0s.append(h0)
        
        trajectory = [h0]
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                token = outputs.logits[0, -1, :].argmax().item()
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        trajectory = torch.stack(trajectory)
        deltas = trajectory[1:] - trajectory[:-1]
        all_deltas.append(deltas)
    
    all_deltas = torch.stack(all_deltas)
    mean_delta = all_deltas.mean(dim=0)
    mean_h0 = torch.stack(all_h0s).mean(dim=0)
    
    # Test on unseen prompts
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
    ]
    
    print("\n--- MESH Prediction Results ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        # Get test h0
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_h0 = outputs.hidden_states[-1][0, -1, :]
        
        # Predict trajectory using MESH
        # Method 1: h[i] = test_h0 + cumsum(mean_delta[:i])
        pred_trajectory = [test_h0]
        h = test_h0.clone()
        for i in range(n_tokens):
            h = h + mean_delta[i]
            pred_trajectory.append(h.clone())
        
        pred_trajectory = torch.stack(pred_trajectory)
        
        # Extract tokens
        pred_tokens = []
        for i in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(pred_trajectory[i])
                token = logits.argmax().item()
            pred_tokens.append(token)
        
        # Reference
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
        
        # Method 2: Use h0 offset
        # The idea: test_h0 - mean_h0 is the "content offset"
        # Apply this offset to the mean trajectory
        
        content_offset = test_h0 - mean_h0
        
        pred_trajectory_2 = [test_h0]
        h = test_h0.clone()
        for i in range(n_tokens):
            h = h + mean_delta[i]
            pred_trajectory_2.append(h.clone())
        
        # This is the same as Method 1 because we start from test_h0
        # Let's try a different approach: scale the offset
        
        print("\n    Offset-scaled predictions:")
        
        for scale in [0.0, 0.5, 1.0]:
            pred_trajectory_s = []
            h = mean_h0 + scale * content_offset
            pred_trajectory_s.append(h.clone())
            
            for i in range(n_tokens):
                h = h + mean_delta[i]
                pred_trajectory_s.append(h.clone())
            
            pred_tokens_s = []
            for i in range(n_tokens):
                with torch.no_grad():
                    logits = lm_head(pred_trajectory_s[i])
                    token = logits.argmax().item()
                pred_tokens_s.append(token)
            
            matches_s = sum(1 for a, b in zip(ref_tokens, pred_tokens_s) if a == b)
            pred_text_s = tokenizer.decode(pred_tokens_s)
            
            print(f"      scale={scale}: {pred_text_s!r} ({matches_s}/10)")
    
    del model


def test_content_aware_mesh():
    """
    Test: Can we build a content-aware MESH?
    
    The idea: The delta depends on the CONTENT token, not just position.
    
    For scaffolding positions, delta is consistent.
    For content positions, delta depends on the token.
    
    We can build a LUT: token → delta
    """
    print("\n" + "=" * 70)
    print("Content-Aware MESH")
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
    
    # Collect data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    n_tokens = 10
    
    # For each position, collect (token, delta) pairs
    position_data = {i: [] for i in range(n_tokens)}
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h0 = outputs.hidden_states[-1][0, -1, :]
        
        trajectory = [h0]
        tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                token = outputs.logits[0, -1, :].argmax().item()
                tokens.append(token)
                
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        trajectory = torch.stack(trajectory)
        
        for i in range(n_tokens):
            delta = trajectory[i+1] - trajectory[i]
            position_data[i].append({
                "token": tokens[i],
                "delta": delta,
                "h_before": trajectory[i],
                "h_after": trajectory[i+1],
            })
    
    # Analyze: Is delta predictable from token?
    print("\n--- Token → Delta Analysis ---")
    
    for i in range(n_tokens):
        data = position_data[i]
        
        # Group by token
        token_groups = {}
        for d in data:
            t = d["token"]
            if t not in token_groups:
                token_groups[t] = []
            token_groups[t].append(d["delta"])
        
        print(f"\n  Position {i}:")
        
        for token, deltas in token_groups.items():
            token_text = tokenizer.decode([token])
            
            if len(deltas) > 1:
                # Compute variance within group
                deltas_stack = torch.stack(deltas)
                var = torch.var(deltas_stack, dim=0).mean().item()
                mean_delta = deltas_stack.mean(dim=0)
                mean_norm = torch.norm(mean_delta).item()
                
                print(f"    {token_text!r}: n={len(deltas)}, var={var:.2f}, |Δ|={mean_norm:.2f}")
            else:
                mean_norm = torch.norm(deltas[0]).item()
                print(f"    {token_text!r}: n=1, |Δ|={mean_norm:.2f}")
    
    # Build content-aware MESH
    # For each position, store: mean_delta (for scaffolding) or token→delta LUT (for content)
    
    print("\n--- Building Content-Aware MESH ---")
    
    mesh = {}
    
    for i in range(n_tokens):
        data = position_data[i]
        
        # Check if this is a scaffolding position (all same token)
        tokens_at_i = [d["token"] for d in data]
        unique_tokens = set(tokens_at_i)
        
        if len(unique_tokens) == 1:
            # Scaffolding: use mean delta
            deltas = torch.stack([d["delta"] for d in data])
            mesh[i] = {
                "type": "scaffolding",
                "mean_delta": deltas.mean(dim=0),
                "token": tokens_at_i[0],
            }
            token_text = tokenizer.decode([tokens_at_i[0]])
            print(f"  Pos {i}: SCAFFOLDING ({token_text!r})")
        else:
            # Content: build token→delta LUT
            token_lut = {}
            for t in unique_tokens:
                deltas = [d["delta"] for d in data if d["token"] == t]
                token_lut[t] = torch.stack(deltas).mean(dim=0) if len(deltas) > 1 else deltas[0]
            
            mesh[i] = {
                "type": "content",
                "token_lut": token_lut,
            }
            token_texts = [tokenizer.decode([t]) for t in list(unique_tokens)[:3]]
            print(f"  Pos {i}: CONTENT ({token_texts}...)")
    
    # Test the content-aware MESH
    print("\n--- Testing Content-Aware MESH ---")
    
    test_prompt = "The capital of Japan is"
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
    
    # Get test h0
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        test_h0 = outputs.hidden_states[-1][0, -1, :]
    
    # Get first token from model (content)
    with torch.no_grad():
        first_logits = lm_head(test_h0)
        first_token = first_logits.argmax().item()
    
    first_text = tokenizer.decode([first_token])
    print(f"\n  First token from model: {first_text!r}")
    
    # Predict trajectory
    pred_tokens = [first_token]
    h = test_h0.clone()
    
    for i in range(n_tokens):
        if i == 0:
            # Use first token's delta (if in LUT) or mean delta
            if mesh[i]["type"] == "content":
                if first_token in mesh[i]["token_lut"]:
                    delta = mesh[i]["token_lut"][first_token]
                else:
                    # Token not in LUT - use mean of all deltas
                    delta = torch.stack(list(mesh[i]["token_lut"].values())).mean(dim=0)
            else:
                delta = mesh[i]["mean_delta"]
        else:
            # Use previous predicted token
            prev_token = pred_tokens[-1]
            
            if mesh[i]["type"] == "content":
                if prev_token in mesh[i]["token_lut"]:
                    delta = mesh[i]["token_lut"][prev_token]
                else:
                    delta = torch.stack(list(mesh[i]["token_lut"].values())).mean(dim=0)
            else:
                delta = mesh[i]["mean_delta"]
        
        h = h + delta
        
        # Predict next token
        with torch.no_grad():
            logits = lm_head(h)
            next_token = logits.argmax().item()
        
        if i < n_tokens - 1:
            pred_tokens.append(next_token)
    
    # Reference
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
    
    print(f"  Ref:  {ref_text!r}")
    print(f"  Pred: {pred_text!r}")
    print(f"  Matches: {matches}/{n_tokens}")
    
    del model


if __name__ == "__main__":
    # 1. Compute trajectory MESH
    mesh, trajectories, tokens, prompts = compute_trajectory_mesh()
    
    # 2. Test MESH prediction
    test_mesh_prediction()
    
    # 3. Test content-aware MESH
    test_content_aware_mesh()
