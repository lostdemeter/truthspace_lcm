#!/usr/bin/env python3
"""
HyperMapped Autoregression
==========================

The trajectory MESH doesn't generalize because we're fitting a linear mapping.
HyperMapping solves this with NEAREST NEIGHBOR in constructed geometry.

The approach:
1. Build a database of (prompt_hidden, trajectory) pairs
2. For a new prompt, find the nearest neighbor in prompt_hidden space
3. Use that trajectory as a TEMPLATE
4. Adapt the template to the new prompt

This is like HyperMapping's forward():
    Input → Encoder → Position → Nearest Neighbor → Output

For autoregression:
    Prompt → Hidden → Nearest Prompt → Adapt Trajectory → Tokens

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


class TrajectoryDatabase:
    """
    A database of (prompt_hidden, trajectory, tokens) tuples.
    
    Like HyperMapping, we use nearest neighbor lookup.
    """
    
    def __init__(self):
        self.prompt_hiddens = []  # List of [hidden_dim] tensors
        self.trajectories = []    # List of [n_tokens, hidden_dim] tensors
        self.tokens = []          # List of token lists
        self.prompts = []         # List of prompt strings
    
    def add(self, prompt: str, prompt_hidden: torch.Tensor, 
            trajectory: torch.Tensor, tokens: List[int]):
        self.prompts.append(prompt)
        self.prompt_hiddens.append(prompt_hidden.clone())
        self.trajectories.append(trajectory.clone())
        self.tokens.append(tokens)
    
    def find_nearest(self, query_hidden: torch.Tensor, k: int = 1):
        """Find k nearest neighbors by cosine similarity."""
        if not self.prompt_hiddens:
            return []
        
        # Stack all prompt hiddens
        all_hiddens = torch.stack(self.prompt_hiddens)  # [n, hidden_dim]
        
        # Normalize
        query_norm = F.normalize(query_hidden.unsqueeze(0), dim=1)
        all_norm = F.normalize(all_hiddens, dim=1)
        
        # Cosine similarity
        sims = (query_norm @ all_norm.T).squeeze()  # [n]
        
        # Top k
        if k == 1:
            idx = sims.argmax().item()
            return [(idx, sims[idx].item())]
        else:
            topk = torch.topk(sims, min(k, len(self.prompts)))
            return list(zip(topk.indices.tolist(), topk.values.tolist()))
    
    def __len__(self):
        return len(self.prompts)


def test_nearest_neighbor_trajectory():
    """
    Test: Use nearest neighbor trajectory as prediction.
    
    This is the simplest HyperMapping approach.
    """
    print("\n" + "=" * 70)
    print("Nearest Neighbor Trajectory")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Build database
    db = TrajectoryDatabase()
    
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    n_tokens = 10
    
    print("\n--- Building Database ---")
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        trajectory = []
        tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                logits = outputs.logits[0, -1, :]
                token = logits.argmax().item()
                tokens.append(token)
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token]])
                ], dim=1)
        
        db.add(prompt, prompt_hidden, torch.stack(trajectory), tokens)
        
        print(f"  Added: {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    # Test
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
    ]
    
    print("\n--- Test: Nearest Neighbor ---")
    
    lm_head = model.lm_head
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
        
        # Find nearest
        nearest = db.find_nearest(test_hidden, k=1)
        idx, sim = nearest[0]
        
        nearest_prompt = db.prompts[idx]
        nearest_trajectory = db.trajectories[idx]
        nearest_tokens = db.tokens[idx]
        
        print(f"\n  Test: {test_prompt!r}")
        print(f"  Nearest: {nearest_prompt!r} (sim={sim:.4f})")
        
        # Use nearest trajectory directly
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(nearest_trajectory[j])
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
        
        ref_text = tokenizer.decode(ref_tokens)
        pred_text = tokenizer.decode(pred_tokens)
        nearest_text = tokenizer.decode(nearest_tokens)
        
        matches = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
        
        print(f"    Ref:     {ref_text!r}")
        print(f"    Nearest: {nearest_text!r}")
        print(f"    Pred:    {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
    
    del model
    
    return db


def test_adapted_trajectory():
    """
    Test: Adapt the nearest trajectory to the new prompt.
    
    The key insight: The STRUCTURE of the trajectory is similar,
    only the CONTENT (the specific answer) differs.
    
    Adaptation strategy:
    1. Find nearest neighbor
    2. Compute the OFFSET between prompts
    3. Apply offset to trajectory
    """
    print("\n" + "=" * 70)
    print("Adapted Trajectory")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Build database
    db = TrajectoryDatabase()
    
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    n_tokens = 10
    
    print("\n--- Building Database ---")
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        trajectory = []
        tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                logits = outputs.logits[0, -1, :]
                token = logits.argmax().item()
                tokens.append(token)
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token]])
                ], dim=1)
        
        db.add(prompt, prompt_hidden, torch.stack(trajectory), tokens)
        
        print(f"  Added: {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    # Test with adaptation
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
    ]
    
    print("\n--- Test: Adapted Trajectory ---")
    
    lm_head = model.lm_head
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
        
        # Find nearest
        nearest = db.find_nearest(test_hidden, k=1)
        idx, sim = nearest[0]
        
        nearest_prompt = db.prompts[idx]
        nearest_hidden = db.prompt_hiddens[idx]
        nearest_trajectory = db.trajectories[idx]
        
        print(f"\n  Test: {test_prompt!r}")
        print(f"  Nearest: {nearest_prompt!r} (sim={sim:.4f})")
        
        # Compute offset
        offset = test_hidden - nearest_hidden
        
        print(f"  Offset norm: {torch.norm(offset).item():.2f}")
        
        # Apply offset to trajectory
        adapted_trajectory = nearest_trajectory + offset
        
        # Extract tokens
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(adapted_trajectory[j])
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
        
        ref_text = tokenizer.decode(ref_tokens)
        pred_text = tokenizer.decode(pred_tokens)
        
        matches = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
        
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
        
        # Try different adaptation strategies
        print("\n  Adaptation strategies:")
        
        # Strategy 1: Offset only first position
        adapted_1 = nearest_trajectory.clone()
        adapted_1[0] = adapted_1[0] + offset
        
        pred_1 = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(adapted_1[j])
                token = logits.argmax().item()
            pred_1.append(token)
        
        matches_1 = sum(1 for a, b in zip(ref_tokens, pred_1) if a == b)
        print(f"    Offset first only: {tokenizer.decode(pred_1)!r} ({matches_1}/10)")
        
        # Strategy 2: Scaled offset
        for scale in [0.5, 1.0, 2.0]:
            adapted_s = nearest_trajectory + scale * offset
            
            pred_s = []
            for j in range(n_tokens):
                with torch.no_grad():
                    logits = lm_head(adapted_s[j])
                    token = logits.argmax().item()
                pred_s.append(token)
            
            matches_s = sum(1 for a, b in zip(ref_tokens, pred_s) if a == b)
            print(f"    Scale {scale}: {tokenizer.decode(pred_s)!r} ({matches_s}/10)")
        
        # Strategy 3: Use first token from model, rest from adapted
        with torch.no_grad():
            first_outputs = model(input_ids)
            first_token = first_outputs.logits[0, -1, :].argmax().item()
        
        # Get hidden state after first token
        with torch.no_grad():
            extended_ids = torch.cat([input_ids, torch.tensor([[first_token]])], dim=1)
            outputs = model(extended_ids, output_hidden_states=True)
            h1 = outputs.hidden_states[-1][0, -1, :]
        
        # Compute new offset from h1
        offset_1 = h1 - nearest_trajectory[1]
        
        adapted_3 = nearest_trajectory.clone()
        adapted_3[0] = nearest_trajectory[0] + offset  # Use original offset for first
        for j in range(1, n_tokens):
            adapted_3[j] = nearest_trajectory[j] + offset_1
        
        pred_3 = [first_token]
        for j in range(1, n_tokens):
            with torch.no_grad():
                logits = lm_head(adapted_3[j])
                token = logits.argmax().item()
            pred_3.append(token)
        
        matches_3 = sum(1 for a, b in zip(ref_tokens, pred_3) if a == b)
        print(f"    First from model: {tokenizer.decode(pred_3)!r} ({matches_3}/10)")
    
    del model
    
    return db


def test_hybrid_approach():
    """
    Hybrid: Use HyperMapping for scaffolding, model for content.
    
    The key insight from our earlier work:
    - Scaffolding tokens (., is, the) are predictable
    - Content tokens (Paris, Berlin) require the model
    
    Strategy:
    1. Use nearest neighbor to predict scaffolding positions
    2. Use the model to predict content positions
    3. Fixed-point to refine
    """
    print("\n" + "=" * 70)
    print("Hybrid Approach: HyperMapping + Model")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Build database
    db = TrajectoryDatabase()
    
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    n_tokens = 10
    
    print("\n--- Building Database ---")
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        trajectory = []
        tokens = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0, -1, :]
                trajectory.append(hidden.clone())
                
                logits = outputs.logits[0, -1, :]
                token = logits.argmax().item()
                tokens.append(token)
                
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token]])
                ], dim=1)
        
        db.add(prompt, prompt_hidden, torch.stack(trajectory), tokens)
    
    # Identify scaffolding positions by analyzing token consistency
    print("\n--- Identifying Scaffolding Positions ---")
    
    # For each position, check if the token is the same across all prompts
    scaffolding_positions = []
    
    for pos in range(n_tokens):
        tokens_at_pos = [db.tokens[i][pos] for i in range(len(db))]
        unique_tokens = set(tokens_at_pos)
        
        if len(unique_tokens) == 1:
            scaffolding_positions.append(pos)
            token_text = tokenizer.decode([tokens_at_pos[0]])
            print(f"  Position {pos}: SCAFFOLDING ({token_text!r})")
        else:
            token_texts = [tokenizer.decode([t]) for t in unique_tokens]
            print(f"  Position {pos}: CONTENT ({token_texts})")
    
    # Test
    test_prompt = "The capital of Japan is"
    
    print(f"\n--- Test: {test_prompt!r} ---")
    
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    # Find nearest
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        test_hidden = outputs.hidden_states[-1][0, -1, :]
    
    nearest = db.find_nearest(test_hidden, k=1)
    idx, sim = nearest[0]
    
    nearest_tokens = db.tokens[idx]
    
    print(f"Nearest: {db.prompts[idx]!r}")
    
    # Initialize with scaffolding from nearest, content from model
    current = []
    
    # Get first token from model
    with torch.no_grad():
        outputs = model(input_ids)
        first_token = outputs.logits[0, -1, :].argmax().item()
    
    current.append(first_token)
    
    # For remaining positions
    for pos in range(1, n_tokens):
        if pos in scaffolding_positions:
            # Use scaffolding from nearest
            current.append(nearest_tokens[pos])
        else:
            # Placeholder - will be filled by fixed-point
            current.append(nearest_tokens[pos])  # Use nearest as init
    
    print(f"Initial: {tokenizer.decode(current)!r}")
    
    # Fixed-point refinement
    for iteration in range(10):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[0]
        
        new_current = []
        for i in range(n_tokens):
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
        for i in range(n_tokens):
            outputs = model(current_ids)
            token = outputs.logits[0, -1, :].argmax().item()
            ref_tokens.append(token)
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    pred_text = tokenizer.decode(current)
    
    matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"Reference: {ref_text!r}")
    print(f"Result:    {pred_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    print(f"Iterations: {iteration + 1}")
    
    del model


if __name__ == "__main__":
    # 1. Test nearest neighbor (baseline)
    db = test_nearest_neighbor_trajectory()
    
    # 2. Test adapted trajectory
    db = test_adapted_trajectory()
    
    # 3. Test hybrid approach
    test_hybrid_approach()
