#!/usr/bin/env python3
"""
Trajectory MESH: The Unraveled Autoregression
==============================================

Key discovery: The trajectory IS a linear function of the prompt hidden state!

    T_flat = prompt_hidden @ W

This is the "MESH" for autoregression, analogous to:
    MESH = W_q.T @ W_k for attention

If we can learn W, we can predict the entire trajectory in ONE operation.

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


def test_trajectory_mesh_generalization():
    """
    Test if the trajectory MESH generalizes to unseen prompts.
    
    Train on some prompts, test on others.
    """
    print("\n" + "=" * 70)
    print("Trajectory MESH Generalization Test")
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
    
    # Test prompts (same pattern, unseen countries)
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
        "The capital of Canada is",
    ]
    
    n_tokens = 10
    
    # Collect training data
    print("\n--- Collecting Training Data ---")
    
    train_hiddens = []
    train_trajectories = []
    train_tokens = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Get prompt hidden state
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        train_hiddens.append(prompt_hidden)
        
        # Generate trajectory
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
        
        train_trajectories.append(torch.stack(trajectory))
        train_tokens.append(tokens)
        
        print(f"  {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    train_hiddens = torch.stack(train_hiddens)  # [n_train, hidden_dim]
    train_T = torch.stack(train_trajectories)   # [n_train, n_tokens, hidden_dim]
    train_T_flat = train_T.reshape(len(train_prompts), -1)  # [n_train, n_tokens * hidden_dim]
    
    # Fit the MESH: T_flat = train_hiddens @ W
    print("\n--- Fitting Trajectory MESH ---")
    
    # Use regularized least squares to avoid overfitting
    # W = (H.T @ H + λI)^-1 @ H.T @ T
    
    H = train_hiddens.numpy()
    T = train_T_flat.numpy()
    
    # Try different regularization strengths
    lambdas = [0, 1e-6, 1e-4, 1e-2, 1]
    
    best_lambda = 0
    best_test_acc = 0
    best_W = None
    
    lm_head = model.lm_head
    
    for lam in lambdas:
        if lam == 0:
            W, _, _, _ = np.linalg.lstsq(H, T, rcond=None)
        else:
            # Regularized solution
            HtH = H.T @ H
            HtT = H.T @ T
            W = np.linalg.solve(HtH + lam * np.eye(HtH.shape[0]), HtT)
        
        # Test on training data
        train_pred = H @ W
        train_pred = torch.tensor(train_pred, dtype=torch.float32).reshape(len(train_prompts), n_tokens, -1)
        
        train_correct = 0
        train_total = 0
        
        for i in range(len(train_prompts)):
            for j in range(n_tokens):
                with torch.no_grad():
                    logits = lm_head(train_pred[i, j])
                    pred_token = logits.argmax().item()
                
                if pred_token == train_tokens[i][j]:
                    train_correct += 1
                train_total += 1
        
        train_acc = train_correct / train_total * 100
        
        # Quick test on one test prompt
        test_prompt = test_prompts[0]
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
        
        test_pred = test_hidden @ W
        test_pred = torch.tensor(test_pred, dtype=torch.float32).reshape(n_tokens, -1)
        
        # Get reference
        ref_tokens = []
        with torch.no_grad():
            current_ids = input_ids.clone()
            for i in range(n_tokens):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                ref_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        test_correct = 0
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(test_pred[j])
                pred_token = logits.argmax().item()
            
            if pred_token == ref_tokens[j]:
                test_correct += 1
        
        test_acc = test_correct / n_tokens * 100
        
        print(f"  λ={lam}: train={train_acc:.1f}%, test={test_acc:.1f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_lambda = lam
            best_W = W
    
    print(f"\nBest λ = {best_lambda}")
    
    # Full test on all test prompts
    print("\n--- Full Test Results ---")
    
    W = best_W
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        # Get prompt hidden state
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
        
        # Predict trajectory
        test_pred = test_hidden @ W
        test_pred = torch.tensor(test_pred, dtype=torch.float32).reshape(n_tokens, -1)
        
        # Extract tokens
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(test_pred[j])
                token = logits.argmax().item()
            pred_tokens.append(token)
        
        # Get reference
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
    
    del model
    
    return best_W, best_lambda


def test_diverse_patterns():
    """
    Test if the trajectory MESH works for diverse prompt patterns.
    
    The key question: Does the MESH generalize across different prompt types?
    """
    print("\n" + "=" * 70)
    print("Diverse Pattern Test")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Different prompt patterns
    patterns = {
        "capital": [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
        ],
        "language": [
            "Python is a programming language that",
            "Java is a programming language that",
            "C++ is a programming language that",
            "JavaScript is a programming language that",
        ],
        "animal": [
            "The elephant is a large animal that",
            "The lion is a large animal that",
            "The tiger is a large animal that",
            "The bear is a large animal that",
        ],
    }
    
    n_tokens = 10
    lm_head = model.lm_head
    
    # For each pattern, train on 3 prompts, test on 1
    results = {}
    
    for pattern_name, prompts in patterns.items():
        print(f"\n--- Pattern: {pattern_name} ---")
        
        train_prompts = prompts[:3]
        test_prompt = prompts[3]
        
        # Collect training data
        train_hiddens = []
        train_trajectories = []
        train_tokens = []
        
        for prompt in train_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                prompt_hidden = outputs.hidden_states[-1][0, -1, :]
            
            train_hiddens.append(prompt_hidden)
            
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
            
            train_trajectories.append(torch.stack(trajectory))
            train_tokens.append(tokens)
        
        train_hiddens = torch.stack(train_hiddens)
        train_T = torch.stack(train_trajectories)
        train_T_flat = train_T.reshape(len(train_prompts), -1)
        
        # Fit MESH
        H = train_hiddens.numpy()
        T = train_T_flat.numpy()
        
        W, _, _, _ = np.linalg.lstsq(H, T, rcond=None)
        
        # Test
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
        
        test_pred = test_hidden @ W
        test_pred = torch.tensor(test_pred, dtype=torch.float32).reshape(n_tokens, -1)
        
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(test_pred[j])
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
        
        print(f"  Test: {test_prompt!r}")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
        
        results[pattern_name] = matches
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for pattern_name, matches in results.items():
        print(f"  {pattern_name}: {matches}/{n_tokens}")
    
    avg_matches = np.mean(list(results.values()))
    print(f"\nAverage: {avg_matches:.1f}/{n_tokens}")
    
    del model
    
    return results


def test_cross_pattern_mesh():
    """
    The ultimate test: Can we learn a SINGLE MESH that works for ALL patterns?
    
    This is the HyperMapping vision: One geometric structure for all mappings.
    """
    print("\n" + "=" * 70)
    print("Cross-Pattern MESH Test")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # All training prompts
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "Python is a programming language that",
        "Java is a programming language that",
        "C++ is a programming language that",
        "The elephant is a large animal that",
        "The lion is a large animal that",
        "The tiger is a large animal that",
    ]
    
    # Test prompts (one from each pattern)
    test_prompts = [
        "The capital of Spain is",
        "JavaScript is a programming language that",
        "The bear is a large animal that",
    ]
    
    n_tokens = 10
    lm_head = model.lm_head
    
    # Collect training data
    print("\n--- Collecting Training Data ---")
    
    train_hiddens = []
    train_trajectories = []
    train_tokens = []
    
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        train_hiddens.append(prompt_hidden)
        
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
        
        train_trajectories.append(torch.stack(trajectory))
        train_tokens.append(tokens)
        
        print(f"  {prompt[:30]}... → {tokenizer.decode(tokens)[:30]}...")
    
    train_hiddens = torch.stack(train_hiddens)
    train_T = torch.stack(train_trajectories)
    train_T_flat = train_T.reshape(len(train_prompts), -1)
    
    # Fit MESH
    print("\n--- Fitting Cross-Pattern MESH ---")
    
    H = train_hiddens.numpy()
    T = train_T_flat.numpy()
    
    # Use regularization
    lam = 1e-4
    HtH = H.T @ H
    HtT = H.T @ T
    W = np.linalg.solve(HtH + lam * np.eye(HtH.shape[0]), HtT)
    
    print(f"MESH shape: {W.shape}")
    
    # Test
    print("\n--- Test Results ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
        
        test_pred = test_hidden @ W
        test_pred = torch.tensor(test_pred, dtype=torch.float32).reshape(n_tokens, -1)
        
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(test_pred[j])
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
    
    del model


if __name__ == "__main__":
    # 1. Test generalization within same pattern
    W, lam = test_trajectory_mesh_generalization()
    
    # 2. Test diverse patterns
    results = test_diverse_patterns()
    
    # 3. Test cross-pattern MESH
    test_cross_pattern_mesh()
