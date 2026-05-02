#!/usr/bin/env python3
"""
Scaffolding + Signal + LUT: The DA2 Approach to Autoregression
===============================================================

Key insight from DA2 (Doc 125):
- Reference: The geometric structure (head features)
- Signal: The linear combination weights (32 parameters)
- LUT: Pre-computed φ^(e/k) values for fast lookup

For autoregression:
- Reference: Scaffolding trajectory (`. It is the...`)
- Signal: The offset/error that encodes the content (`Paris`, `Tokyo`)
- LUT: Pre-computed offset patterns for common content types

The "error" IS the signal - it encodes the content!

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


def build_scaffolding_reference():
    """
    Build the scaffolding reference structure.
    
    For a given prompt pattern, the scaffolding is the COMMON structure
    across all instances. The content is the VARIABLE part.
    """
    print("\n" + "=" * 70)
    print("Building Scaffolding Reference")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect trajectories for the same pattern
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    n_tokens = 10
    
    all_trajectories = []
    all_tokens = []
    all_hiddens = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        all_hiddens.append(prompt_hidden)
        
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
        
        all_trajectories.append(torch.stack(trajectory))
        all_tokens.append(tokens)
        
        print(f"  {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    # Stack trajectories: [n_prompts, n_tokens, hidden_dim]
    T = torch.stack(all_trajectories)
    H = torch.stack(all_hiddens)
    
    # Compute the REFERENCE (mean trajectory)
    reference_trajectory = T.mean(dim=0)  # [n_tokens, hidden_dim]
    
    print(f"\nReference trajectory shape: {reference_trajectory.shape}")
    
    # Compute the SIGNAL (offset from reference)
    signals = T - reference_trajectory  # [n_prompts, n_tokens, hidden_dim]
    
    print("\n--- Signal Analysis ---")
    
    for i, prompt in enumerate(prompts):
        signal = signals[i]
        signal_norms = torch.norm(signal, dim=1)
        
        print(f"\n  {prompt!r}")
        print(f"    Signal norms per position:")
        for j in range(n_tokens):
            token_text = tokenizer.decode([all_tokens[i][j]])
            print(f"      Pos {j} ({token_text!r}): {signal_norms[j].item():.2f}")
    
    # Key insight: The signal at position 0 encodes the CONTENT (the country name)
    # The signal at scaffolding positions should be small
    
    print("\n--- Position-wise Signal Variance ---")
    
    for j in range(n_tokens):
        # Variance of signal across prompts at this position
        signal_at_j = signals[:, j, :]  # [n_prompts, hidden_dim]
        variance = torch.var(signal_at_j, dim=0).mean().item()
        
        # What tokens appear at this position?
        tokens_at_j = [all_tokens[i][j] for i in range(len(prompts))]
        unique_tokens = set(tokens_at_j)
        
        if len(unique_tokens) == 1:
            token_text = tokenizer.decode([tokens_at_j[0]])
            print(f"  Pos {j}: variance={variance:.2f}, SCAFFOLDING ({token_text!r})")
        else:
            token_texts = [tokenizer.decode([t]) for t in unique_tokens]
            print(f"  Pos {j}: variance={variance:.2f}, CONTENT ({token_texts})")
    
    del model
    
    return reference_trajectory, signals, all_tokens, prompts


def build_signal_lut():
    """
    Build a LUT that maps prompt hidden states to content signals.
    
    Like DA2's 16K-entry LUT for φ^(e/k), we build a LUT for content offsets.
    """
    print("\n" + "=" * 70)
    print("Building Signal LUT")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect data for LUT
    prompts = [
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
    
    # Collect prompt hiddens and trajectories
    prompt_hiddens = []
    trajectories = []
    tokens_list = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        prompt_hiddens.append(prompt_hidden)
        
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
        
        trajectories.append(torch.stack(trajectory))
        tokens_list.append(tokens)
    
    prompt_hiddens = torch.stack(prompt_hiddens)  # [n, hidden_dim]
    T = torch.stack(trajectories)  # [n, n_tokens, hidden_dim]
    
    # Compute reference (mean trajectory)
    reference = T.mean(dim=0)  # [n_tokens, hidden_dim]
    
    # Compute signals (offsets from reference)
    signals = T - reference  # [n, n_tokens, hidden_dim]
    
    # The LUT maps: prompt_hidden → signal
    # But signals are high-dimensional. We need to compress them.
    
    print("\n--- Signal Compression ---")
    
    # Flatten signals: [n, n_tokens * hidden_dim]
    signals_flat = signals.reshape(len(prompts), -1)
    
    # SVD to find principal signal components
    U, S, Vh = torch.linalg.svd(signals_flat, full_matrices=False)
    
    print("Signal singular values:")
    total_var = (S**2).sum()
    for i in range(min(10, len(S))):
        var = (S[i]**2 / total_var).item() * 100
        print(f"  S[{i}] = {S[i].item():.2f} ({var:.1f}%)")
    
    # How many components do we need?
    cumsum = torch.cumsum(S**2, dim=0) / total_var * 100
    for threshold in [90, 95, 99]:
        n_components = (cumsum < threshold).sum().item() + 1
        print(f"  {threshold}% variance: {n_components} components")
    
    # Use top-k components
    k = min(5, len(S))
    signal_basis = Vh[:k, :]  # [k, n_tokens * hidden_dim]
    signal_codes = signals_flat @ signal_basis.T  # [n, k]
    
    print(f"\nSignal codes (k={k}):")
    for i, prompt in enumerate(prompts):
        print(f"  {prompt[:20]}...: {signal_codes[i].tolist()}")
    
    # Now build the LUT: prompt_hidden → signal_code
    # This is a linear mapping: signal_code = prompt_hidden @ W_lut
    
    print("\n--- Building LUT Mapping ---")
    
    # Fit: signal_codes = prompt_hiddens @ W_lut
    W_lut, _, _, _ = np.linalg.lstsq(
        prompt_hiddens.numpy(),
        signal_codes.numpy(),
        rcond=None
    )
    
    print(f"LUT weight shape: {W_lut.shape}")
    
    # Test reconstruction
    print("\n--- LUT Reconstruction Test ---")
    
    lm_head = model.lm_head
    
    for i, prompt in enumerate(prompts):
        # Predict signal code from prompt hidden
        pred_code = prompt_hiddens[i].numpy() @ W_lut
        pred_code = torch.tensor(pred_code, dtype=torch.float32)
        
        # Reconstruct signal from code
        pred_signal = pred_code @ signal_basis
        pred_signal = pred_signal.reshape(n_tokens, -1)
        
        # Reconstruct trajectory
        pred_trajectory = reference + pred_signal
        
        # Extract tokens
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(pred_trajectory[j])
                token = logits.argmax().item()
            pred_tokens.append(token)
        
        matches = sum(1 for a, b in zip(tokens_list[i], pred_tokens) if a == b)
        
        ref_text = tokenizer.decode(tokens_list[i])
        pred_text = tokenizer.decode(pred_tokens)
        
        print(f"  {prompt[:20]}...")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
    
    # The LUT is:
    # 1. reference: [n_tokens, hidden_dim] - the scaffolding
    # 2. signal_basis: [k, n_tokens * hidden_dim] - the signal components
    # 3. W_lut: [hidden_dim, k] - maps prompt hidden to signal code
    
    lut = {
        "reference": reference,
        "signal_basis": signal_basis,
        "W_lut": torch.tensor(W_lut, dtype=torch.float32),
        "k": k,
    }
    
    del model
    
    return lut, prompts, tokens_list


def test_lut_generalization():
    """
    Test if the LUT generalizes to unseen prompts.
    """
    print("\n" + "=" * 70)
    print("LUT Generalization Test")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Training prompts
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    # Test prompts
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
    ]
    
    n_tokens = 10
    
    # Build LUT from training data
    print("\n--- Building LUT from Training Data ---")
    
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
    T = torch.stack(train_trajectories)
    
    # Reference and signals
    reference = T.mean(dim=0)
    signals = T - reference
    signals_flat = signals.reshape(len(train_prompts), -1)
    
    # SVD
    U, S, Vh = torch.linalg.svd(signals_flat, full_matrices=False)
    k = min(5, len(S))
    signal_basis = Vh[:k, :]
    signal_codes = signals_flat @ signal_basis.T
    
    # LUT mapping
    W_lut, _, _, _ = np.linalg.lstsq(
        train_hiddens.numpy(),
        signal_codes.numpy(),
        rcond=None
    )
    W_lut = torch.tensor(W_lut, dtype=torch.float32)
    
    # Test on unseen prompts
    print("\n--- Test Results ---")
    
    lm_head = model.lm_head
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
        
        # Predict signal code
        pred_code = test_hidden @ W_lut
        
        # Reconstruct signal
        pred_signal = pred_code @ signal_basis
        pred_signal = pred_signal.reshape(n_tokens, -1)
        
        # Reconstruct trajectory
        pred_trajectory = reference + pred_signal
        
        # Extract tokens
        pred_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(pred_trajectory[j])
                token = logits.argmax().item()
            pred_tokens.append(token)
        
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
        
        # Analyze the error
        print(f"    Signal code: {pred_code.tolist()}")
        
        # Compare to nearest training signal code
        dists = torch.norm(signal_codes - pred_code, dim=1)
        nearest_idx = dists.argmin().item()
        print(f"    Nearest train: {train_prompts[nearest_idx]!r} (dist={dists[nearest_idx].item():.2f})")
    
    # The key insight: The LUT predicts the WRONG signal for unseen content
    # But the STRUCTURE (scaffolding) is correct!
    
    print("\n--- Hybrid: LUT Scaffolding + Model Content ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Get first token from model (the content)
        with torch.no_grad():
            outputs = model(input_ids)
            first_token = outputs.logits[0, -1, :].argmax().item()
        
        # Use LUT for scaffolding positions (1, 2, 3)
        # Use model for content positions (0, 4+)
        
        # Initialize with LUT prediction
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            test_hidden = outputs.hidden_states[-1][0, -1, :]
        
        pred_code = test_hidden @ W_lut
        pred_signal = pred_code @ signal_basis
        pred_signal = pred_signal.reshape(n_tokens, -1)
        pred_trajectory = reference + pred_signal
        
        # Extract scaffolding tokens from LUT
        scaffolding_tokens = []
        for j in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(pred_trajectory[j])
                token = logits.argmax().item()
            scaffolding_tokens.append(token)
        
        # Initialize with first token from model, scaffolding from LUT
        current = [first_token] + scaffolding_tokens[1:]
        
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
        
        matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
        
        ref_text = tokenizer.decode(ref_tokens)
        pred_text = tokenizer.decode(current)
        
        print(f"\n  {test_prompt!r}")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}, Iters: {iteration + 1}")
    
    del model


def analyze_content_signal():
    """
    Analyze the structure of the content signal.
    
    Key question: Is the content signal predictable from the prompt?
    """
    print("\n" + "=" * 70)
    print("Content Signal Analysis")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect data
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    n_tokens = 10
    
    prompt_hiddens = []
    trajectories = []
    tokens_list = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        
        prompt_hiddens.append(prompt_hidden)
        
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
        
        trajectories.append(torch.stack(trajectory))
        tokens_list.append(tokens)
    
    prompt_hiddens = torch.stack(prompt_hiddens)
    T = torch.stack(trajectories)
    
    # Reference
    reference = T.mean(dim=0)
    
    # Signals
    signals = T - reference
    
    # Focus on position 0 (the content position)
    print("\n--- Position 0 (Content) Signal ---")
    
    content_signals = signals[:, 0, :]  # [n_prompts, hidden_dim]
    
    # What's the relationship between prompt_hidden and content_signal?
    
    # Hypothesis: content_signal = f(prompt_hidden)
    # If linear: content_signal = prompt_hidden @ W_content
    
    # But we only have 4 data points and 3584 dimensions...
    # Let's look at the structure instead
    
    # Compute pairwise similarities
    print("Prompt hidden similarities:")
    prompt_norm = F.normalize(prompt_hiddens, dim=1)
    prompt_sim = prompt_norm @ prompt_norm.T
    print(prompt_sim.numpy())
    
    print("\nContent signal similarities:")
    content_norm = F.normalize(content_signals, dim=1)
    content_sim = content_norm @ content_norm.T
    print(content_sim.numpy())
    
    # Key insight: Are the similarities correlated?
    # If prompt_sim[i,j] ≈ content_sim[i,j], then the content is predictable
    
    prompt_sim_flat = prompt_sim[torch.triu(torch.ones(4, 4), diagonal=1) == 1]
    content_sim_flat = content_sim[torch.triu(torch.ones(4, 4), diagonal=1) == 1]
    
    correlation = torch.corrcoef(torch.stack([prompt_sim_flat, content_sim_flat]))[0, 1].item()
    
    print(f"\nCorrelation between prompt and content similarities: {correlation:.4f}")
    
    # The content signal direction
    print("\n--- Content Signal Direction ---")
    
    # SVD of content signals
    U_c, S_c, Vh_c = torch.linalg.svd(content_signals, full_matrices=False)
    
    print("Content signal singular values:")
    for i, s in enumerate(S_c):
        var = (s**2 / (S_c**2).sum()).item() * 100
        print(f"  S[{i}] = {s.item():.2f} ({var:.1f}%)")
    
    # The content signal is LOW-RANK!
    # This means there's a common "content direction"
    
    # The first principal component is the "content axis"
    content_axis = Vh_c[0, :]  # [hidden_dim]
    
    # Project prompt hiddens onto content axis
    prompt_projections = prompt_hiddens @ content_axis
    
    print("\nPrompt projections onto content axis:")
    for i, prompt in enumerate(prompts):
        token_text = tokenizer.decode([tokens_list[i][0]])
        print(f"  {prompt!r}: proj={prompt_projections[i].item():.2f} → {token_text!r}")
    
    # The projection onto the content axis predicts the content!
    
    del model
    
    return content_signals, content_axis


if __name__ == "__main__":
    # 1. Build scaffolding reference
    reference, signals, tokens, prompts = build_scaffolding_reference()
    
    # 2. Build signal LUT
    lut, prompts, tokens = build_signal_lut()
    
    # 3. Test LUT generalization
    test_lut_generalization()
    
    # 4. Analyze content signal structure
    content_signals, content_axis = analyze_content_signal()
