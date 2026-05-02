#!/usr/bin/env python3
"""
Direct Content Extraction: Can We Skip the Forward Pass?
=========================================================

Key insight: The model KNOWS "France → Paris". This knowledge is in the weights.

Question: Can we extract the content token directly from the prompt hidden state
without running the full forward pass?

Hypothesis: The prompt hidden state contains a "pointer" to the content token.
The lm_head projects this pointer to vocabulary space.

If true, we can:
1. Extract the content token directly: token = argmax(lm_head(prompt_hidden))
2. Use scaffolding for the rest
3. No iteration needed!

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


def test_direct_extraction():
    """
    Test: Can we extract the content token directly from prompt hidden state?
    
    This is what the model does in the forward pass:
        logits = lm_head(hidden_state)
        token = argmax(logits)
    
    We're just checking if this works for the PROMPT hidden state.
    """
    print("\n" + "=" * 70)
    print("Direct Content Extraction Test")
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
    
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
        "The capital of Australia is",
    ]
    
    print("\n--- Direct Extraction from Prompt Hidden State ---")
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            prompt_hidden = outputs.hidden_states[-1][0, -1, :]
            
            # Direct extraction
            logits = lm_head(prompt_hidden)
            direct_token = logits.argmax().item()
            
            # Reference (from model output)
            ref_token = outputs.logits[0, -1, :].argmax().item()
        
        direct_text = tokenizer.decode([direct_token])
        ref_text = tokenizer.decode([ref_token])
        
        match = "✓" if direct_token == ref_token else "✗"
        
        print(f"  {prompt!r}")
        print(f"    Direct: {direct_text!r}, Ref: {ref_text!r} {match}")
    
    # This should work because lm_head(hidden_state) IS how the model predicts!
    # The question is: can we get the hidden state faster?
    
    print("\n--- Analysis: What's in the Prompt Hidden State? ---")
    
    # The prompt hidden state is the output of the last transformer layer
    # It encodes: prompt context + prediction of next token
    
    # Let's look at the top-k predictions
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1][0, -1, :]
        logits = lm_head(prompt_hidden)
    
    # Top-k tokens
    topk = torch.topk(logits, k=10)
    
    print(f"\n  Prompt: {prompt!r}")
    print("  Top-10 predictions:")
    for i, (idx, score) in enumerate(zip(topk.indices, topk.values)):
        token_text = tokenizer.decode([idx.item()])
        print(f"    {i+1}. {token_text!r} (score={score.item():.2f})")
    
    del model
    
    return


def test_parallel_content_extraction():
    """
    Test: Can we extract ALL content tokens in parallel?
    
    Idea: The hidden state at each position contains the prediction for the NEXT token.
    If we run the model once with the full sequence, we get all predictions.
    
    But wait - this is just normal autoregressive generation!
    
    The question is: Can we get the hidden states WITHOUT the tokens?
    """
    print("\n" + "=" * 70)
    print("Parallel Content Extraction")
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
    
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    prompt_len = input_ids.shape[1]
    n_tokens = 10
    
    # Sequential generation (reference)
    print("\n--- Sequential Generation (Reference) ---")
    
    ref_tokens = []
    ref_hiddens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            ref_hiddens.append(hidden.clone())
            
            token = outputs.logits[0, -1, :].argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    print(f"  {prompt!r} → {ref_text!r}")
    
    # Now the key question: Can we get ref_hiddens without knowing ref_tokens?
    
    # Approach 1: Use scaffolding tokens as placeholders
    print("\n--- Approach 1: Scaffolding Placeholders ---")
    
    # Get scaffolding from a similar prompt
    scaffolding_prompt = "The capital of Germany is"
    scaffolding_ids = tokenizer.encode(scaffolding_prompt, return_tensors='pt')
    
    scaffolding_tokens = []
    with torch.no_grad():
        current_ids = scaffolding_ids.clone()
        for i in range(n_tokens):
            outputs = model(current_ids)
            token = outputs.logits[0, -1, :].argmax().item()
            scaffolding_tokens.append(token)
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    scaffolding_text = tokenizer.decode(scaffolding_tokens)
    print(f"  Scaffolding: {scaffolding_text!r}")
    
    # Use scaffolding as placeholder, run model once
    placeholder_ids = torch.cat([
        input_ids,
        torch.tensor([scaffolding_tokens])
    ], dim=1)
    
    with torch.no_grad():
        outputs = model(placeholder_ids, output_hidden_states=True)
        all_hiddens = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
    
    # Extract predictions from hidden states
    pred_tokens = []
    for i in range(n_tokens):
        pos = prompt_len + i - 1 if i > 0 else prompt_len - 1
        hidden = all_hiddens[pos]
        
        with torch.no_grad():
            logits = lm_head(hidden)
            token = logits.argmax().item()
        
        pred_tokens.append(token)
    
    pred_text = tokenizer.decode(pred_tokens)
    matches = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
    
    print(f"  Predicted: {pred_text!r}")
    print(f"  Matches: {matches}/{n_tokens}")
    
    # Approach 2: Iterative refinement
    print("\n--- Approach 2: Iterative Refinement ---")
    
    current = scaffolding_tokens.copy()
    
    for iteration in range(10):
        full_ids = torch.cat([
            input_ids,
            torch.tensor([current])
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(full_ids, output_hidden_states=True)
            all_hiddens = outputs.hidden_states[-1][0]
        
        new_current = []
        for i in range(n_tokens):
            pos = prompt_len + i - 1 if i > 0 else prompt_len - 1
            hidden = all_hiddens[pos]
            
            with torch.no_grad():
                logits = lm_head(hidden)
                token = logits.argmax().item()
            
            new_current.append(token)
        
        if new_current == current:
            break
        current = new_current
    
    pred_text = tokenizer.decode(current)
    matches = sum(1 for a, b in zip(ref_tokens, current) if a == b)
    
    print(f"  After {iteration + 1} iterations: {pred_text!r}")
    print(f"  Matches: {matches}/{n_tokens}")
    
    # The key insight: We need the CORRECT tokens to get the correct hidden states
    # This is the fundamental autoregressive constraint
    
    # But what if we could predict the hidden states directly?
    print("\n--- Approach 3: Hidden State Prediction ---")
    
    # Hypothesis: h[i+1] = f(h[i], token[i])
    # If we know the function f, we can predict h[i+1] from h[i]
    
    # Let's analyze the relationship between consecutive hidden states
    print("\n  Analyzing hidden state transitions:")
    
    for i in range(n_tokens - 1):
        h_i = ref_hiddens[i]
        h_next = ref_hiddens[i + 1]
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(h_i.unsqueeze(0), h_next.unsqueeze(0)).item()
        
        # Delta
        delta = h_next - h_i
        delta_norm = torch.norm(delta).item()
        
        token_text = tokenizer.decode([ref_tokens[i]])
        
        print(f"    h[{i}] → h[{i+1}]: cos={cos_sim:.4f}, |Δ|={delta_norm:.2f} (token: {token_text!r})")
    
    del model


def test_embedding_based_prediction():
    """
    Test: Can we predict hidden states using token embeddings?
    
    Hypothesis: h[i] ≈ h[0] + Σ_j embedding[token[j]] * weight[j]
    
    This is like the DA2 approach: linear combination of features.
    """
    print("\n" + "=" * 70)
    print("Embedding-Based Hidden State Prediction")
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
    
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    n_tokens = 10
    
    # Get reference trajectory
    ref_tokens = []
    ref_hiddens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            ref_hiddens.append(hidden.clone())
            
            token = outputs.logits[0, -1, :].argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    ref_hiddens = torch.stack(ref_hiddens)  # [n_tokens, hidden_dim]
    
    # Get token embeddings
    token_embeds = embed_layer(torch.tensor(ref_tokens))  # [n_tokens, hidden_dim]
    
    # Get prompt hidden state
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h0 = outputs.hidden_states[-1][0, -1, :]
    
    # Try to fit: h[i] = h0 + Σ_j α[i,j] * embed[j]
    print("\n--- Fitting Linear Model ---")
    
    # Build basis: [h0, embed[0], embed[1], ..., embed[n-1]]
    basis = torch.cat([h0.unsqueeze(0), token_embeds], dim=0)  # [n_tokens+1, hidden_dim]
    
    # For each position i, fit coefficients
    for i in range(n_tokens):
        target = ref_hiddens[i]
        
        # Use only h0 and embeddings up to position i
        if i == 0:
            # h[0] should be close to h0 (prompt hidden)
            pred = h0
        else:
            # h[i] = α_0 * h0 + Σ_j α_j * embed[j] for j < i
            sub_basis = basis[:i+1].T.detach()  # [hidden_dim, i+1]
            
            # Solve: target = sub_basis @ coeffs
            coeffs, _, _, _ = np.linalg.lstsq(
                sub_basis.numpy(),
                target.detach().numpy(),
                rcond=None
            )
            
            pred = torch.tensor(sub_basis.numpy() @ coeffs, dtype=torch.float32)
        
        # Evaluate
        cos_sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
        
        # Can we extract the correct token?
        with torch.no_grad():
            pred_logits = lm_head(pred)
            pred_token = pred_logits.argmax().item()
        
        true_token = ref_tokens[i] if i < n_tokens else None
        
        pred_text = tokenizer.decode([pred_token])
        true_text = tokenizer.decode([ref_tokens[i]]) if i < n_tokens else "N/A"
        
        match = "✓" if pred_token == ref_tokens[i] else "✗"
        
        print(f"  h[{i}]: cos={cos_sim:.4f}, pred={pred_text!r}, true={true_text!r} {match}")
    
    # The linear model doesn't work well because the transformer is nonlinear
    # But what if we use the MESH approach?
    
    print("\n--- MESH Approach: Pre-computed Transformation ---")
    
    # The transformer computes: h[i] = Transformer(h[i-1], token[i-1])
    # Can we approximate this as: h[i] ≈ h[i-1] @ W + embed[token[i-1]] @ V
    
    # This is a bilinear model - similar to attention!
    
    # For now, let's just check if the hidden states are predictable
    # from the prompt hidden state alone (without tokens)
    
    print("\n--- Hidden State Predictability from h0 ---")
    
    # Fit: h[i] = h0 @ W[i]
    for i in range(n_tokens):
        target = ref_hiddens[i]
        
        # Linear fit: target = h0 @ w
        # w = (h0.T @ h0)^-1 @ h0.T @ target
        # But h0 is a vector, so this is just: w = target (projection)
        
        # Instead, check cosine similarity
        cos_sim = F.cosine_similarity(h0.unsqueeze(0), target.unsqueeze(0)).item()
        
        print(f"  cos(h0, h[{i}]) = {cos_sim:.4f}")
    
    del model


if __name__ == "__main__":
    # 1. Test direct extraction
    test_direct_extraction()
    
    # 2. Test parallel extraction
    test_parallel_content_extraction()
    
    # 3. Test embedding-based prediction
    test_embedding_based_prediction()
