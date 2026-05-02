#!/usr/bin/env python3
"""
Pattern Transfer: 83% → 100%
=============================

Current state: Pattern transfer (positions 1-4) achieves 83.3% accuracy.

Goal: Understand WHY 17% fails and fix it.

Hypotheses:
1. Not enough training data - need more entities to average
2. Entity-specific variation - need small correction per entity
3. Position-specific issues - some positions harder than others
4. Basis dimensionality - need more basis vectors

Approach:
1. Analyze which tokens fail and why
2. Test with more training data
3. Test entity-specific corrections
4. Test higher-dimensional basis

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


def analyze_failures(trajectories, all_tokens, entities, P, position_patterns, tokenizer, lm_head):
    """Analyze which tokens fail and why."""
    
    print("\n" + "=" * 70)
    print("Analyzing Pattern Transfer Failures")
    print("=" * 70)
    
    n_steps = len(trajectories[0])
    
    # Track failures by position
    failures_by_position = {j: [] for j in range(n_steps)}
    
    for i, (traj, toks, entity) in enumerate(zip(trajectories, all_tokens, entities)):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            
            # Use mean coefficients (pattern transfer)
            pattern = position_patterns[j]
            mean_coeffs = pattern['coeffs'].mean(dim=0)
            basis = pattern['basis']
            bulge = mean_coeffs @ basis
            
            h_j = h_geo + bulge
            h_full = h_j @ P
            logits = h_full @ lm_head.T
            pred_token = logits.argmax().item()
            
            actual_token = toks[j]
            
            if pred_token != actual_token:
                failures_by_position[j].append({
                    'entity': entity,
                    'actual': tokenizer.decode([actual_token]),
                    'predicted': tokenizer.decode([pred_token]),
                    'actual_id': actual_token,
                    'pred_id': pred_token,
                })
    
    # Report
    print("\nFailures by Position:")
    for j in range(n_steps):
        failures = failures_by_position[j]
        print(f"\n  Position {j}: {len(failures)} failures")
        for f in failures[:5]:  # Show first 5
            print(f"    {f['entity']}: '{f['actual']}' → '{f['predicted']}'")
    
    return failures_by_position


def test_more_training_data(model, tokenizer, n_train: int = 10, n_test: int = 5):
    """Test if more training data improves accuracy."""
    
    print("\n" + "=" * 70)
    print(f"Test: More Training Data ({n_train} train, {n_test} test)")
    print("=" * 70)
    
    all_entities = [
        # Training
        "France", "Germany", "Italy", "Spain", "Japan",
        "China", "Russia", "Brazil", "India", "Egypt",
        "Poland", "Canada", "Australia", "Mexico", "Argentina",
        # Test
        "Sweden", "Norway", "Finland", "Denmark", "Netherlands",
    ]
    
    train_entities = all_entities[:n_train]
    test_entities = all_entities[n_train:n_train+n_test]
    
    # Collect trajectories
    train_traj, train_toks = collect_trajectories(model, tokenizer, train_entities)
    test_traj, test_toks = collect_trajectories(model, tokenizer, test_entities)
    
    print(f"\nTraining entities: {train_entities}")
    print(f"Test entities: {test_entities}")
    
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
    
    # Test on training data
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
            
            # Use this entity's coefficients (should be 100%)
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
        print(f"  {entity}: {match}/{len(toks)} - {actual_text} → {recon_text}")
    
    print(f"\nTraining accuracy: {train_correct}/{train_total} = {train_correct/train_total*100:.1f}%")
    
    # Test on held-out data with MEAN coefficients (pattern transfer)
    print("\n--- Test Data (Pattern Transfer) ---")
    test_correct = 0
    test_total = 0
    
    for traj, toks, entity in zip(test_traj, test_toks, test_entities):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        recon_tokens = []
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            
            # Use MEAN coefficients (pattern transfer)
            pattern = position_patterns[j]
            mean_coeffs = pattern['coeffs'].mean(dim=0)
            basis = pattern['basis']
            bulge = mean_coeffs @ basis
            
            h_j = h_geo + bulge
            h_full = h_j @ P
            logits = h_full @ lm_head.T
            recon_tokens.append(logits.argmax().item())
        
        match = sum(1 for a, r in zip(toks, recon_tokens) if a == r)
        test_correct += match
        test_total += len(toks)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in recon_tokens]
        print(f"  {entity}: {match}/{len(toks)} - {actual_text} → {recon_text}")
    
    print(f"\nTest accuracy (pattern transfer): {test_correct}/{test_total} = {test_correct/test_total*100:.1f}%")
    
    # Analyze failures
    analyze_failures(test_traj, test_toks, test_entities, P, position_patterns, tokenizer, lm_head)
    
    return train_correct/train_total, test_correct/test_total


def test_higher_basis_dim(model, tokenizer, basis_dims: List[int] = [5, 10, 20, 50]):
    """Test if higher basis dimensionality improves accuracy."""
    
    print("\n" + "=" * 70)
    print("Test: Higher Basis Dimensionality")
    print("=" * 70)
    
    entities = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Russia", "Brazil"]
    test_entities = ["Poland", "Canada", "India", "Egypt"]
    
    train_traj, train_toks = collect_trajectories(model, tokenizer, entities)
    test_traj, test_toks = collect_trajectories(model, tokenizer, test_entities)
    
    all_points = torch.cat(train_traj, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    lm_head = model.lm_head.weight.data
    n_steps = len(train_traj[0])
    
    results = []
    
    for n_basis in basis_dims:
        print(f"\n--- Basis dim = {n_basis} ---")
        
        # Extract patterns with this basis dim
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
            
            # Use n_basis dimensions
            actual_basis = min(n_basis, Vt_b.shape[0])
            position_patterns.append({
                'basis': Vt_b[:actual_basis],
                'coeffs': bulges @ Vt_b[:actual_basis].T,
            })
        
        # Test accuracy
        test_correct = 0
        test_total = 0
        
        for traj, toks in zip(test_traj, test_toks):
            traj_proj = traj @ P.T
            h_start = traj_proj[0]
            h_end = traj_proj[-1]
            
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
                
                if pred_token == toks[j]:
                    test_correct += 1
                test_total += 1
        
        acc = test_correct / test_total
        print(f"  Accuracy: {test_correct}/{test_total} = {acc*100:.1f}%")
        results.append((n_basis, acc))
    
    print("\n--- Summary ---")
    for n_basis, acc in results:
        print(f"  Basis dim {n_basis}: {acc*100:.1f}%")
    
    return results


def test_skip_position_0(model, tokenizer):
    """
    Test accuracy when we skip position 0 (content token).
    
    Position 0 is entity-specific and should be handled by memory lookup.
    We only need pattern transfer for positions 1-5.
    """
    print("\n" + "=" * 70)
    print("Test: Skip Position 0 (Content Token)")
    print("=" * 70)
    
    entities = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Russia", "Brazil"]
    test_entities = ["Poland", "Canada", "India", "Egypt"]
    
    train_traj, train_toks = collect_trajectories(model, tokenizer, entities)
    test_traj, test_toks = collect_trajectories(model, tokenizer, test_entities)
    
    all_points = torch.cat(train_traj, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    lm_head = model.lm_head.weight.data
    n_steps = len(train_traj[0])
    
    # Extract patterns
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
    
    # Test on positions 1-5 only (skip position 0)
    print("\n--- Test Data (Positions 1-5 only) ---")
    
    correct_by_position = {j: 0 for j in range(1, n_steps)}
    total_by_position = {j: 0 for j in range(1, n_steps)}
    
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
            
            if j > 0:  # Skip position 0
                total_by_position[j] += 1
                if pred_token == toks[j]:
                    correct_by_position[j] += 1
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in recon_tokens]
        
        # Count matches for positions 1-5
        match = sum(1 for j in range(1, n_steps) if recon_tokens[j] == toks[j])
        print(f"  {entity}: {match}/{n_steps-1} (pos 1-5)")
        print(f"    Actual: {actual_text}")
        print(f"    Recon:  {recon_text}")
    
    print("\n--- Accuracy by Position (1-5) ---")
    total_correct = 0
    total_count = 0
    for j in range(1, n_steps):
        acc = correct_by_position[j] / total_by_position[j] if total_by_position[j] > 0 else 0
        print(f"  Position {j}: {correct_by_position[j]}/{total_by_position[j]} = {acc*100:.1f}%")
        total_correct += correct_by_position[j]
        total_count += total_by_position[j]
    
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\nOverall (positions 1-5): {total_correct}/{total_count} = {overall_acc*100:.1f}%")
    
    return overall_acc


def main():
    print("=" * 70)
    print("Pattern Transfer: 83% → 100%")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: More training data
    train_acc, test_acc = test_more_training_data(model, tokenizer, n_train=10, n_test=5)
    
    # Test 2: Higher basis dimensionality
    basis_results = test_higher_basis_dim(model, tokenizer, basis_dims=[5, 10, 20, 50])
    
    # Test 3: Skip position 0 (content token)
    pos_1_5_acc = test_skip_position_0(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Results:
  - Training accuracy: {train_acc*100:.1f}%
  - Test accuracy (all positions): {test_acc*100:.1f}%
  - Test accuracy (positions 1-5 only): {pos_1_5_acc*100:.1f}%

Basis dimensionality effect:
""")
    for n_basis, acc in basis_results:
        print(f"  - {n_basis} basis: {acc*100:.1f}%")
    
    print("""
Key Insight:
  Position 0 (content token) is entity-specific and handled by memory lookup.
  Positions 1-5 (pattern tokens) should transfer with high accuracy.
  
  If positions 1-5 achieve 100%, then the full system achieves 100%:
    - Position 0: Memory lookup (precached)
    - Positions 1-5: Pattern transfer
""")


if __name__ == "__main__":
    main()
