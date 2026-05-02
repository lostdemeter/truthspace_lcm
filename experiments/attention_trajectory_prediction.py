#!/usr/bin/env python3
"""
Attention-Based Trajectory Prediction
======================================

Key insight: The attention pattern encodes which tokens influence which positions.
If we can predict the attention pattern, we can predict the trajectory.

The holographic principle applied:
- Each hidden state h[i] is a weighted sum of previous hidden states
- The weights are the attention scores
- If we know the attention pattern, we know the trajectory

Connection to Additive Error Stereo:
- In stereo: E encodes ∂D/∂x (depth gradient)
- Here: Attention encodes ∂h/∂token (semantic gradient)

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


def analyze_attention_trajectory():
    """
    Analyze how attention patterns relate to hidden state trajectory.
    """
    print("\n" + "=" * 70)
    print("Attention-Based Trajectory Analysis")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    print(f"\nPrompt: {prompt!r}")
    
    # Collect trajectory with attention
    hidden_trajectory = []
    attention_patterns = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True, output_attentions=True)
            
            hidden = outputs.hidden_states[-1][0, -1, :]  # Last layer, last position
            hidden_trajectory.append(hidden.clone())
            
            # Get attention from last layer, average over heads
            attn = outputs.attentions[-1][0]  # [n_heads, seq_len, seq_len]
            attn_last_pos = attn[:, -1, :].mean(dim=0)  # [seq_len] - attention from last position
            attention_patterns.append(attn_last_pos.clone())
            
            logits = outputs.logits[0, -1, :]
            token = logits.argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    H = torch.stack(hidden_trajectory)
    lm_head = model.lm_head
    
    # Analyze attention patterns
    print("\n--- Attention Pattern Analysis ---")
    
    for i in range(n_tokens):
        attn = attention_patterns[i]
        seq_len = len(attn)
        
        # Where does attention focus?
        top_k = 3
        top_indices = torch.topk(attn, min(top_k, seq_len)).indices.tolist()
        top_values = torch.topk(attn, min(top_k, seq_len)).values.tolist()
        
        token_text = tokenizer.decode([ref_tokens[i]])
        print(f"  Token {i} ({token_text!r}): top attention at positions {top_indices} with values {[f'{v:.2f}' for v in top_values]}")
    
    # Key insight: Attention to prompt vs attention to generated tokens
    print("\n--- Prompt vs Generated Attention ---")
    
    for i in range(n_tokens):
        attn = attention_patterns[i]
        seq_len = len(attn)
        
        prompt_attn = attn[:prompt_len].sum().item()
        gen_attn = attn[prompt_len:].sum().item() if seq_len > prompt_len else 0
        
        token_text = tokenizer.decode([ref_tokens[i]])
        print(f"  Token {i} ({token_text!r}): prompt={prompt_attn:.2f}, generated={gen_attn:.2f}")
    
    # The holographic insight: Can we predict hidden states from attention?
    print("\n--- Attention-Based Hidden State Prediction ---")
    
    # Hypothesis: h[i] ≈ Σ_j attn[i,j] * h[j] (for j < i)
    # This is what attention DOES, but we're checking if we can predict it
    
    # Get all hidden states from the prompt
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1][0]  # [prompt_len, hidden_dim]
    
    print(f"Prompt hidden states shape: {prompt_hidden.shape}")
    
    # For each generated token, predict its hidden state using attention
    print("\nPredicting hidden states from attention:")
    
    for i in range(n_tokens):
        attn = attention_patterns[i]
        
        # Build the full hidden state sequence up to this point
        if i == 0:
            all_hidden = prompt_hidden
        else:
            all_hidden = torch.cat([prompt_hidden, H[:i]], dim=0)
        
        # Weighted sum
        h_pred = (attn.unsqueeze(1) * all_hidden).sum(dim=0)
        h_actual = H[i]
        
        # Compare
        cos_sim = F.cosine_similarity(h_pred.unsqueeze(0), h_actual.unsqueeze(0)).item()
        
        # Can we get the right token from h_pred?
        with torch.no_grad():
            logits_pred = lm_head(h_pred)
            token_pred = logits_pred.argmax().item()
        
        correct = token_pred == ref_tokens[i]
        marker = "✓" if correct else "✗"
        
        token_text = tokenizer.decode([ref_tokens[i]])
        pred_text = tokenizer.decode([token_pred])
        
        print(f"  Token {i}: cos_sim={cos_sim:.3f}, pred={pred_text!r}, ref={token_text!r} {marker}")
    
    # The REAL insight: Attention is COMPUTED, not given
    # We need to predict what the attention WILL BE
    print("\n--- Attention Prediction ---")
    
    # Hypothesis: Attention patterns are predictable from the query
    # The query is computed from the hidden state
    
    # For each position, the query is: q = W_q @ h
    # The keys are: k = W_k @ h (for all previous positions)
    # Attention is: softmax(q @ k.T / sqrt(d))
    
    # If we can predict the query from the prompt, we can predict attention
    
    # Let's check: How similar are the queries for consecutive positions?
    print("Query similarity analysis would require extracting W_q, W_k matrices...")
    print("Skipping for now - this requires deeper model introspection.")
    
    # Alternative: Use the PATTERN of attention
    print("\n--- Attention Pattern Regularity ---")
    
    # Check if attention patterns have structure
    # Stack attention patterns (pad to same length)
    max_len = max(len(a) for a in attention_patterns)
    padded_attns = []
    
    for attn in attention_patterns:
        padded = torch.zeros(max_len)
        padded[:len(attn)] = attn
        padded_attns.append(padded)
    
    attn_matrix = torch.stack(padded_attns)  # [n_tokens, max_len]
    
    # SVD of attention patterns
    U_a, S_a, Vh_a = torch.linalg.svd(attn_matrix, full_matrices=False)
    
    print("Attention pattern singular values:")
    for i, s in enumerate(S_a[:5]):
        var = (s**2 / (S_a**2).sum()).item() * 100
        print(f"  S[{i}] = {s.item():.3f} ({var:.1f}%)")
    
    # The attention patterns are LOW-RANK!
    # This means they're predictable
    
    del model
    
    return H, attention_patterns, ref_tokens


def test_parallel_with_attention_prediction():
    """
    Test parallel generation using predicted attention patterns.
    
    Strategy:
    1. Get prompt hidden states
    2. Predict attention patterns for all positions
    3. Compute hidden states in parallel using predicted attention
    4. Extract tokens
    5. Verify with single forward pass
    """
    print("\n" + "=" * 70)
    print("Parallel Generation with Attention Prediction")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompt = "The capital of France is"
    n_tokens = 10
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    
    print(f"\nPrompt: {prompt!r}")
    
    # Reference generation
    with torch.no_grad():
        ref_outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ref_tokens = ref_outputs[0, prompt_len:].tolist()
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    # Strategy: Use the FIXED-POINT insight combined with attention
    print("\n--- Combined Fixed-Point + Attention Strategy ---")
    
    # The key insight: Fixed-point converges in 1 iteration with good init
    # Can we use attention to get a good init?
    
    # Step 1: Get first token (this is unavoidable)
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
        first_logits = outputs.logits[0, -1, :]
        first_token = first_logits.argmax().item()
        
        # Get attention pattern for first token
        first_attn = outputs.attentions[-1][0, :, -1, :].mean(dim=0)  # [prompt_len]
        
        # Get prompt hidden states
        prompt_hidden = outputs.hidden_states[-1][0]  # [prompt_len, hidden_dim]
    
    print(f"First token: {tokenizer.decode([first_token])!r}")
    
    # Step 2: Predict attention patterns for remaining positions
    # Hypothesis: Attention patterns are similar across positions
    # Use first attention pattern as template
    
    # Actually, let's try a simpler approach:
    # Use uniform attention over prompt + exponential decay for generated tokens
    
    predicted_tokens = [first_token]
    
    # Step 3: Use greedy parallel initialization
    # Generate all tokens in parallel using first token as context
    
    with torch.no_grad():
        # Extend with first token
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[first_token]])
        ], dim=1)
        
        # Get hidden state after first token
        outputs = model(extended_ids, output_hidden_states=True)
        h1 = outputs.hidden_states[-1][0, -1, :]
        
        # Predict second token
        second_logits = outputs.logits[0, -1, :]
        second_token = second_logits.argmax().item()
    
    predicted_tokens.append(second_token)
    
    # Now we have 2 tokens - use fixed-point for the rest
    # Initialize remaining tokens with parallel prediction
    
    with torch.no_grad():
        init_ids = torch.cat([
            input_ids,
            torch.tensor([[first_token, second_token] + [0] * (n_tokens - 2)])
        ], dim=1)
        
        outputs = model(init_ids)
        logits = outputs.logits[0]
        
        for i in range(2, n_tokens):
            pos_logits = logits[prompt_len + i - 1]
            predicted_tokens.append(pos_logits.argmax().item())
    
    # Step 4: Fixed-point refinement
    current = predicted_tokens.copy()
    
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
    
    total_time = time.time() - start_time
    
    final_text = tokenizer.decode(current)
    matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"\nFinal: {final_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    print(f"Iterations: {iteration + 1}")
    print(f"Time: {total_time:.2f}s")
    
    # Compare to pure sequential
    print("\n--- Comparison ---")
    
    start_time = time.time()
    
    with torch.no_grad():
        seq_outputs = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    seq_time = time.time() - start_time
    
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Our method time: {total_time:.2f}s")
    print(f"Speedup: {seq_time / total_time:.2f}x")
    
    del model


def test_two_token_anchor():
    """
    Test: Generate first 2 tokens sequentially, then fixed-point for rest.
    
    This is the minimal anchor approach.
    """
    print("\n" + "=" * 70)
    print("Two-Token Anchor Strategy")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
        "Python is a programming language that",
        "In the year 2024, artificial intelligence",
    ]
    
    n_tokens = 10
    
    results = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Reference
        with torch.no_grad():
            ref_outputs = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        ref_tokens = ref_outputs[0, prompt_len:].tolist()
        ref_text = tokenizer.decode(ref_tokens)
        
        # Two-token anchor
        start_time = time.time()
        
        # Get first 2 tokens sequentially
        with torch.no_grad():
            current_ids = input_ids.clone()
            anchor_tokens = []
            
            for i in range(2):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                anchor_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        # Initialize rest with parallel prediction
        with torch.no_grad():
            init_ids = torch.cat([
                input_ids,
                torch.tensor([anchor_tokens + [0] * (n_tokens - 2)])
            ], dim=1)
            
            outputs = model(init_ids)
            logits = outputs.logits[0]
            
            current = anchor_tokens.copy()
            for i in range(2, n_tokens):
                pos_logits = logits[prompt_len + i - 1]
                current.append(pos_logits.argmax().item())
        
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
        
        total_time = time.time() - start_time
        
        final_text = tokenizer.decode(current)
        matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
        
        print(f"Reference: {ref_text!r}")
        print(f"Result:    {final_text!r}")
        print(f"Matches: {matches}/{n_tokens}, Iters: {iteration + 1}, Time: {total_time:.2f}s")
        
        results.append({
            "prompt": prompt,
            "matches": matches,
            "iterations": iteration + 1,
            "time": total_time,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    avg_matches = np.mean([r["matches"] for r in results])
    avg_iters = np.mean([r["iterations"] for r in results])
    avg_time = np.mean([r["time"] for r in results])
    
    print(f"Average matches: {avg_matches:.1f}/10")
    print(f"Average iterations: {avg_iters:.1f}")
    print(f"Average time: {avg_time:.2f}s")
    
    all_correct = all(r["matches"] == 10 for r in results)
    print(f"All correct: {all_correct}")
    
    del model
    
    return results


def benchmark_speedup():
    """
    Benchmark the two-token anchor strategy against sequential generation.
    """
    print("\n" + "=" * 70)
    print("Speedup Benchmark")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The quick brown fox jumps over",
    ]
    
    n_tokens = 10
    
    seq_times = []
    anchor_times = []
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Sequential generation
        start_time = time.time()
        with torch.no_grad():
            seq_outputs = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        seq_time = time.time() - start_time
        seq_tokens = seq_outputs[0, prompt_len:].tolist()
        seq_times.append(seq_time)
        
        # Two-token anchor
        start_time = time.time()
        
        # Get first 2 tokens sequentially
        with torch.no_grad():
            current_ids = input_ids.clone()
            anchor_tokens = []
            
            for i in range(2):
                outputs = model(current_ids)
                token = outputs.logits[0, -1, :].argmax().item()
                anchor_tokens.append(token)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        # Initialize rest with parallel prediction
        with torch.no_grad():
            init_ids = torch.cat([
                input_ids,
                torch.tensor([anchor_tokens + [0] * (n_tokens - 2)])
            ], dim=1)
            
            outputs = model(init_ids)
            logits = outputs.logits[0]
            
            current = anchor_tokens.copy()
            for i in range(2, n_tokens):
                pos_logits = logits[prompt_len + i - 1]
                current.append(pos_logits.argmax().item())
        
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
        
        anchor_time = time.time() - start_time
        anchor_times.append(anchor_time)
        
        matches = sum(1 for a, b in zip(seq_tokens, current) if a == b)
        
        print(f"Sequential: {seq_time:.2f}s")
        print(f"Two-anchor: {anchor_time:.2f}s (iters: {iteration + 1})")
        print(f"Speedup: {seq_time / anchor_time:.2f}x")
        print(f"Matches: {matches}/{n_tokens}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    avg_seq = np.mean(seq_times)
    avg_anchor = np.mean(anchor_times)
    
    print(f"Average sequential time: {avg_seq:.2f}s")
    print(f"Average two-anchor time: {avg_anchor:.2f}s")
    print(f"Average speedup: {avg_seq / avg_anchor:.2f}x")
    
    # Count forward passes
    # Sequential: 10 forward passes
    # Two-anchor: 2 (anchor) + 1 (init) + ~8 (fixed-point) = ~11 forward passes
    # But each forward pass processes all tokens at once!
    
    print("\n--- Forward Pass Analysis ---")
    print("Sequential: 10 forward passes (1 token each)")
    print("Two-anchor: 2 (anchor) + 1 (init) + ~8 (FP) = ~11 forward passes")
    print("But: Sequential processes 1 token, Two-anchor processes 10 tokens per pass")
    print("Effective: Sequential = 10 passes, Two-anchor = 11 passes (similar)")
    print("\nThe speedup comes from KV caching in generate(), not from fewer passes.")
    
    del model


if __name__ == "__main__":
    # Skip attention analysis (requires eager attention implementation)
    # analyze_attention_trajectory()
    
    # Test two-token anchor (the minimal approach)
    # test_two_token_anchor()
    
    # Benchmark speedup
    benchmark_speedup()
