#!/usr/bin/env python3
"""
Same Pattern Transfer: Test with entities that share the SAME pattern
======================================================================

Key insight from previous experiment:
- Different entities produce different response FORMATS
- France: ". It is the most"
- Japan: "__.\nTokyo\n" (completely different!)

This explains why pattern transfer dropped from 83% to 26%.

Solution: Only transfer patterns between entities with the SAME format.

This experiment:
1. Identify entities with the same pattern format
2. Test transfer only within that group
3. Should achieve much higher accuracy

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def collect_trajectories(model, tokenizer, entities: List[str], n_tokens: int = 6):
    """Collect trajectories for multiple entities."""
    
    trajectories = []
    all_tokens = []
    
    for entity in entities:
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        hidden_states = []
        tokens = []
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                h = outputs.hidden_states[-1][0, -1, :]
                hidden_states.append(h)
                
                next_token = outputs.logits[0, -1, :].argmax()
                tokens.append(next_token.item())
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        trajectories.append(torch.stack(hidden_states))
        all_tokens.append(tokens)
    
    return trajectories, all_tokens


def identify_pattern_groups(all_tokens, entities, tokenizer):
    """Group entities by their response pattern."""
    
    print("\n" + "=" * 70)
    print("Identifying Pattern Groups")
    print("=" * 70)
    
    # Look at tokens 1-5 (skip content token 0)
    patterns = {}
    
    for toks, entity in zip(all_tokens, entities):
        # Pattern = tokens 1-5 (the scaffold)
        pattern_toks = tuple(toks[1:])
        pattern_text = tuple(tokenizer.decode([t]) for t in pattern_toks)
        
        if pattern_text not in patterns:
            patterns[pattern_text] = []
        patterns[pattern_text].append(entity)
    
    print(f"\nFound {len(patterns)} distinct patterns:")
    for pattern, ents in patterns.items():
        print(f"\n  Pattern: {list(pattern)}")
        print(f"  Entities: {ents}")
    
    return patterns


def test_same_pattern_transfer(model, tokenizer):
    """Test transfer only between entities with the same pattern."""
    
    print("\n" + "=" * 70)
    print("Test: Same Pattern Transfer")
    print("=" * 70)
    
    # European countries that should have similar patterns
    # (single-token capitals, ". It is..." format)
    similar_entities = [
        "France", "Germany", "Italy", "Spain", 
        "Poland", "Greece", "Austria", "Belgium",
        "Portugal", "Sweden", "Norway", "Denmark",
    ]
    
    trajectories, all_tokens = collect_trajectories(model, tokenizer, similar_entities)
    
    # Show what we got
    print("\nCollected responses:")
    for toks, entity in zip(all_tokens, similar_entities):
        text = [tokenizer.decode([t]) for t in toks]
        print(f"  {entity}: {text}")
    
    # Identify pattern groups
    patterns = identify_pattern_groups(all_tokens, similar_entities, tokenizer)
    
    # Find the largest group with ". It is..." pattern
    target_pattern = None
    target_entities = []
    
    for pattern, ents in patterns.items():
        # Look for ". It is..." pattern
        if len(ents) >= 4 and pattern[0] == '.':
            target_pattern = pattern
            target_entities = ents
            break
    
    if not target_entities:
        # Just use the largest group
        target_entities = max(patterns.values(), key=len)
        target_pattern = [p for p, e in patterns.items() if e == target_entities][0]
    
    print(f"\nUsing pattern group: {list(target_pattern)}")
    print(f"Entities in group: {target_entities}")
    
    # Filter to only these entities
    filtered_traj = []
    filtered_toks = []
    filtered_entities = []
    
    for traj, toks, entity in zip(trajectories, all_tokens, similar_entities):
        if entity in target_entities:
            filtered_traj.append(traj)
            filtered_toks.append(toks)
            filtered_entities.append(entity)
    
    if len(filtered_entities) < 4:
        print("Not enough entities in group, using all similar entities")
        filtered_traj = trajectories
        filtered_toks = all_tokens
        filtered_entities = similar_entities
    
    # Split into train/test
    n_train = len(filtered_entities) - 2
    train_traj = filtered_traj[:n_train]
    train_toks = filtered_toks[:n_train]
    train_entities = filtered_entities[:n_train]
    
    test_traj = filtered_traj[n_train:]
    test_toks = filtered_toks[n_train:]
    test_entities = filtered_entities[n_train:]
    
    print(f"\nTrain: {train_entities}")
    print(f"Test: {test_entities}")
    
    # Compute projection
    all_points = torch.cat(train_traj, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    # Extract per-position pattern
    n_steps = len(train_traj[0])
    position_patterns = []
    
    for j in range(n_steps):
        bulges = []
        for traj in train_traj:
            traj_proj = traj @ P.T
            h_start = traj_proj[0]
            h_end = traj_proj[-1]
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            bulge = traj_proj[j] - h_geo
            bulges.append(bulge)
        
        bulges = torch.stack(bulges)
        U_b, S_b, Vt_b = torch.linalg.svd(bulges, full_matrices=False)
        
        position_patterns.append({
            'basis': Vt_b[:10],
            'coeffs': bulges @ Vt_b[:10].T,
        })
    
    lm_head = model.lm_head.weight.data
    
    # Test on training data (should be 100%)
    print("\n--- Training Data ---")
    train_correct = 0
    train_total = 0
    
    for i, (traj, toks, entity) in enumerate(zip(train_traj, train_toks, train_entities)):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        recon_tokens = []
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            
            pattern = position_patterns[j]
            coeffs = pattern['coeffs'][i]
            basis = pattern['basis']
            bulge = coeffs @ basis
            
            h_j = h_geo + bulge
            h_full = h_j @ P
            logits = h_full @ lm_head.T
            recon_tokens.append(logits.argmax().item())
        
        match = sum(1 for a, r in zip(toks, recon_tokens) if a == r)
        train_correct += match
        train_total += len(toks)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in recon_tokens]
        print(f"  {entity}: {match}/{len(toks)} - {actual_text}")
    
    print(f"\nTraining accuracy: {train_correct}/{train_total} = {train_correct/train_total*100:.1f}%")
    
    # Test on held-out data with MEAN coefficients
    print("\n--- Test Data (Pattern Transfer) ---")
    test_correct = 0
    test_total = 0
    
    correct_by_position = {j: 0 for j in range(n_steps)}
    total_by_position = {j: 0 for j in range(n_steps)}
    
    for traj, toks, entity in zip(test_traj, test_toks, test_entities):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        recon_tokens = []
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            
            pattern = position_patterns[j]
            mean_coeffs = pattern['coeffs'].mean(dim=0)
            basis = pattern['basis']
            bulge = mean_coeffs @ basis
            
            h_j = h_geo + bulge
            h_full = h_j @ P
            logits = h_full @ lm_head.T
            pred_token = logits.argmax().item()
            recon_tokens.append(pred_token)
            
            total_by_position[j] += 1
            if pred_token == toks[j]:
                correct_by_position[j] += 1
        
        match = sum(1 for a, r in zip(toks, recon_tokens) if a == r)
        test_correct += match
        test_total += len(toks)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in recon_tokens]
        print(f"  {entity}: {match}/{len(toks)}")
        print(f"    Actual: {actual_text}")
        print(f"    Recon:  {recon_text}")
    
    print(f"\nTest accuracy (all positions): {test_correct}/{test_total} = {test_correct/test_total*100:.1f}%")
    
    # Accuracy by position
    print("\n--- Accuracy by Position ---")
    for j in range(n_steps):
        acc = correct_by_position[j] / total_by_position[j] if total_by_position[j] > 0 else 0
        print(f"  Position {j}: {correct_by_position[j]}/{total_by_position[j]} = {acc*100:.1f}%")
    
    # Positions 1-5 only (skip content token)
    pos_1_5_correct = sum(correct_by_position[j] for j in range(1, n_steps))
    pos_1_5_total = sum(total_by_position[j] for j in range(1, n_steps))
    pos_1_5_acc = pos_1_5_correct / pos_1_5_total if pos_1_5_total > 0 else 0
    
    print(f"\nPositions 1-5 only: {pos_1_5_correct}/{pos_1_5_total} = {pos_1_5_acc*100:.1f}%")
    
    return test_correct/test_total, pos_1_5_acc


def main():
    print("=" * 70)
    print("Same Pattern Transfer: Entities with Matching Patterns")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test same pattern transfer
    all_acc, pos_1_5_acc = test_same_pattern_transfer(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Results:
  - All positions: {all_acc*100:.1f}%
  - Positions 1-5 (pattern only): {pos_1_5_acc*100:.1f}%

Key Insight:
  Pattern transfer only works between entities with the SAME response format.
  
  Different entities produce different patterns:
    - France: ". It is the most..."
    - Japan: "__.\nTokyo\n" (multi-token, newlines)
  
  Solution for 100% accuracy:
    1. Classify entities by response pattern type
    2. Store one pattern template per type
    3. Apply matching pattern to each entity
    
  This is like Q4 (Error Quaternion) from Doc 056:
    - Detect which pattern type the entity belongs to
    - Apply the correct pattern template
""")


if __name__ == "__main__":
    main()
