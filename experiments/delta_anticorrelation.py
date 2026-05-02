#!/usr/bin/env python3
"""
Delta Anti-Correlation Analysis
================================

Key discovery: Delta is ANTI-CORRELATED with h_before (correlation -0.59 to -0.68).

This means: When h_before changes, delta changes in the OPPOSITE direction.

Hypothesis: The model is trying to reach a FIXED POINT for each token.
    h_after = h_before + delta
    
If delta ≈ target - h_before, then:
    h_after ≈ h_before + (target - h_before) = target
    
This would mean each token has a TARGET hidden state it's trying to reach!

Let's test this hypothesis.

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


def analyze_fixed_point_hypothesis():
    """
    Test: Does each token have a TARGET hidden state?
    
    If delta ≈ target - h_before, then:
        h_after = h_before + delta ≈ target
    
    So h_after should be MORE consistent than h_before for the same token.
    """
    print("\n" + "=" * 70)
    print("Fixed Point Hypothesis Test")
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
        "The capital of Portugal is",
        "The capital of Poland is",
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
                "h_before": h_prev.clone(),
                "h_after": h_curr.clone(),
                "delta": delta,
                "prompt": prompt,
            })
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # Analyze: Is h_after more consistent than h_before?
    print("\n--- Consistency Analysis ---")
    print("If fixed point hypothesis is true: var(h_after) < var(h_before)")
    
    for token in sorted(token_data.keys(), key=lambda t: len(token_data[t]), reverse=True)[:15]:
        entries = token_data[token]
        if len(entries) < 2:
            continue
        
        token_text = tokenizer.decode([token])
        
        h_befores = torch.stack([e["h_before"] for e in entries])
        h_afters = torch.stack([e["h_after"] for e in entries])
        deltas = torch.stack([e["delta"] for e in entries])
        
        # Variance
        var_before = torch.var(h_befores, dim=0).mean().item()
        var_after = torch.var(h_afters, dim=0).mean().item()
        var_delta = torch.var(deltas, dim=0).mean().item()
        
        # Pairwise similarity
        h_before_norm = F.normalize(h_befores, dim=1)
        h_after_norm = F.normalize(h_afters, dim=1)
        
        mask = ~torch.eye(len(entries), dtype=torch.bool)
        
        sim_before = (h_before_norm @ h_before_norm.T)[mask].mean().item()
        sim_after = (h_after_norm @ h_after_norm.T)[mask].mean().item()
        
        # Is h_after more consistent?
        more_consistent = "✓" if sim_after > sim_before else "✗"
        
        print(f"\n  Token: {token_text!r} (n={len(entries)})")
        print(f"    var(h_before) = {var_before:.2f}, sim = {sim_before:.4f}")
        print(f"    var(h_after)  = {var_after:.2f}, sim = {sim_after:.4f} {more_consistent}")
        print(f"    var(delta)    = {var_delta:.2f}")
        
        # Compute the "target" as mean(h_after)
        target = h_afters.mean(dim=0)
        
        # How close is each h_after to the target?
        distances_to_target = torch.norm(h_afters - target, dim=1)
        mean_dist = distances_to_target.mean().item()
        
        # How close is each h_before to the target?
        distances_before = torch.norm(h_befores - target, dim=1)
        mean_dist_before = distances_before.mean().item()
        
        print(f"    dist(h_before, target) = {mean_dist_before:.2f}")
        print(f"    dist(h_after, target)  = {mean_dist:.2f}")
    
    del model


def analyze_delta_as_correction():
    """
    Test: Is delta ≈ target - h_before?
    
    If so, we can predict delta from h_before alone!
    """
    print("\n" + "=" * 70)
    print("Delta as Correction Analysis")
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
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
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
                "h_before": h_prev.clone(),
                "h_after": h_curr.clone(),
                "delta": delta,
            })
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # For each token, compute the "target" and test prediction
    print("\n--- Delta Prediction Test ---")
    print("Model: delta_pred = target - h_before")
    
    for token in sorted(token_data.keys(), key=lambda t: len(token_data[t]), reverse=True)[:10]:
        entries = token_data[token]
        if len(entries) < 3:
            continue
        
        token_text = tokenizer.decode([token])
        
        h_befores = torch.stack([e["h_before"] for e in entries])
        h_afters = torch.stack([e["h_after"] for e in entries])
        deltas = torch.stack([e["delta"] for e in entries])
        
        # Compute target as mean(h_after)
        target = h_afters.mean(dim=0)
        
        # Predict delta
        delta_pred = target.unsqueeze(0) - h_befores  # [n, hidden]
        
        # Compare to actual delta
        cos_sims = F.cosine_similarity(delta_pred, deltas, dim=1)
        mean_cos = cos_sims.mean().item()
        
        # Norm comparison
        pred_norms = torch.norm(delta_pred, dim=1)
        actual_norms = torch.norm(deltas, dim=1)
        norm_ratio = (pred_norms / actual_norms).mean().item()
        
        print(f"\n  Token: {token_text!r} (n={len(entries)})")
        print(f"    cos(delta_pred, delta_actual) = {mean_cos:.4f}")
        print(f"    |delta_pred| / |delta_actual| = {norm_ratio:.4f}")
        
        # Can we use this to predict the next token?
        # h_after_pred = h_before + delta_pred = h_before + (target - h_before) = target
        # So all predictions would give the same token!
        
        # Let's check: what token does the target predict?
        with torch.no_grad():
            target_logits = lm_head(target)
            target_token = target_logits.argmax().item()
        
        target_token_text = tokenizer.decode([target_token])
        print(f"    Target predicts: {target_token_text!r}")
        
        # What about individual h_afters?
        predicted_tokens = []
        for h_after in h_afters:
            with torch.no_grad():
                logits = lm_head(h_after)
                pred_token = logits.argmax().item()
            predicted_tokens.append(pred_token)
        
        # Are they all the same?
        unique_preds = set(predicted_tokens)
        print(f"    h_after predictions: {[tokenizer.decode([t]) for t in unique_preds]}")
    
    del model


def test_target_based_prediction():
    """
    Test: Can we predict trajectories using token targets?
    
    Approach:
    1. Build a LUT: token → target (mean h_after)
    2. For prediction:
       - Get first token from model
       - Look up target for that token
       - Use target as h_after
       - Predict next token from target
       - Repeat
    """
    print("\n" + "=" * 70)
    print("Target-Based Prediction Test")
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
    
    # Build target LUT from training data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
    ]
    
    n_tokens = 10
    
    token_h_afters = defaultdict(list)
    
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
            
            token_h_afters[token].append(h_curr)
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # Build target LUT
    target_lut = {}
    for token, h_afters in token_h_afters.items():
        target_lut[token] = torch.stack(h_afters).mean(dim=0)
    
    print(f"Target LUT size: {len(target_lut)} tokens")
    
    # Analyze: What does each target predict?
    print("\n--- Target → Next Token Mapping ---")
    
    target_to_next = {}
    for token, target in target_lut.items():
        with torch.no_grad():
            logits = lm_head(target)
            next_token = logits.argmax().item()
        
        target_to_next[token] = next_token
        
        token_text = tokenizer.decode([token])
        next_text = tokenizer.decode([next_token])
        print(f"  {token_text!r:15} → {next_text!r}")
    
    # Test prediction
    test_prompts = [
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Brazil is",
    ]
    
    print("\n--- Target-Based Prediction Results ---")
    
    for test_prompt in test_prompts:
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        
        # Get initial hidden state
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :].clone()
        
        # Predict first token
        with torch.no_grad():
            logits = lm_head(h)
            first_token = logits.argmax().item()
        
        # Predict using target LUT
        pred_tokens = [first_token]
        
        for i in range(n_tokens - 1):
            current_token = pred_tokens[-1]
            
            if current_token in target_lut:
                # Use target as next hidden state
                h = target_lut[current_token]
            else:
                # Token not in LUT - use mean of all targets
                h = torch.stack(list(target_lut.values())).mean(dim=0)
            
            # Predict next token
            with torch.no_grad():
                logits = lm_head(h)
                next_token = logits.argmax().item()
            
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
        
        print(f"\n  {test_prompt!r}")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
    
    # The issue: target-based prediction creates a MARKOV CHAIN
    # Each token deterministically leads to the next
    # This loses the context dependence!
    
    print("\n--- Markov Chain Analysis ---")
    print("Following the target chain from common tokens:")
    
    for start_token in ['.', ' It', ' is', ' the']:
        start_id = tokenizer.encode(start_token, add_special_tokens=False)
        if not start_id:
            continue
        start_id = start_id[0]
        
        if start_id not in target_lut:
            continue
        
        chain = [start_id]
        h = target_lut[start_id]
        
        for _ in range(10):
            with torch.no_grad():
                logits = lm_head(h)
                next_token = logits.argmax().item()
            
            chain.append(next_token)
            
            if next_token in target_lut:
                h = target_lut[next_token]
            else:
                break
        
        chain_text = tokenizer.decode(chain)
        print(f"  {start_token!r} → {chain_text!r}")
    
    del model


def analyze_context_contribution():
    """
    Analyze: How much does context contribute to delta?
    
    We know: delta ≈ target - h_before (anti-correlation)
    
    But the target itself might depend on context!
    Let's decompose: target = base_target + context_adjustment
    """
    print("\n" + "=" * 70)
    print("Context Contribution Analysis")
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
    
    # Collect data from diverse contexts
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
            
            token_data[token].append({
                "h_before": h_prev.clone(),
                "h_after": h_curr.clone(),
                "prompt": prompt,
            })
            
            h_prev = h_curr.clone()
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    # For tokens appearing in multiple contexts, analyze h_after variation
    print("\n--- h_after Variation by Context ---")
    
    for token in sorted(token_data.keys(), key=lambda t: len(token_data[t]), reverse=True)[:10]:
        entries = token_data[token]
        if len(entries) < 3:
            continue
        
        token_text = tokenizer.decode([token])
        
        h_afters = torch.stack([e["h_after"] for e in entries])
        h_befores = torch.stack([e["h_before"] for e in entries])
        
        # Mean h_after (base target)
        base_target = h_afters.mean(dim=0)
        
        # Context adjustments
        adjustments = h_afters - base_target
        
        # How much does context contribute?
        adjustment_norms = torch.norm(adjustments, dim=1)
        base_norm = torch.norm(base_target).item()
        
        mean_adjustment = adjustment_norms.mean().item()
        adjustment_ratio = mean_adjustment / base_norm
        
        print(f"\n  Token: {token_text!r} (n={len(entries)})")
        print(f"    |base_target| = {base_norm:.2f}")
        print(f"    |adjustment| = {mean_adjustment:.2f} ({adjustment_ratio*100:.1f}% of base)")
        
        # Does the adjustment correlate with h_before?
        # adjustment ≈ α * (h_before - mean_h_before)
        mean_h_before = h_befores.mean(dim=0)
        h_before_centered = h_befores - mean_h_before
        
        # Compute correlation
        correlations = []
        for i in range(len(entries)):
            if torch.norm(adjustments[i]) > 1e-6 and torch.norm(h_before_centered[i]) > 1e-6:
                cos = F.cosine_similarity(
                    adjustments[i].unsqueeze(0),
                    h_before_centered[i].unsqueeze(0)
                ).item()
                correlations.append(cos)
        
        if correlations:
            mean_corr = np.mean(correlations)
            print(f"    corr(adjustment, h_before_centered) = {mean_corr:.4f}")
        
        # What tokens do different h_afters predict?
        next_tokens = []
        for h_after in h_afters:
            with torch.no_grad():
                logits = lm_head(h_after)
                next_token = logits.argmax().item()
            next_tokens.append(next_token)
        
        unique_next = set(next_tokens)
        next_texts = [tokenizer.decode([t]) for t in unique_next]
        print(f"    Next tokens predicted: {next_texts}")
    
    del model


if __name__ == "__main__":
    # 1. Test fixed point hypothesis
    analyze_fixed_point_hypothesis()
    
    # 2. Analyze delta as correction
    analyze_delta_as_correction()
    
    # 3. Test target-based prediction
    test_target_based_prediction()
    
    # 4. Analyze context contribution
    analyze_context_contribution()
