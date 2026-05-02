#!/usr/bin/env python3
"""
Constrained Pattern Transfer: 100% via Hard Endpoint Constraints
=================================================================

Previous attempt: 50% accuracy with single direction vector.

The issue: Direction alone doesn't capture per-position variation.

New approach from Doc 055-056:
- Start = KNOWN (hard constraint, W=-1 definitive)
- End = KNOWN (hard constraint, W=-1 definitive)
- Middle = INTERPOLATE with per-position bulge

Key insight: We already proved per-position coefficients give 100%.
Now we need to make them TRANSFERABLE.

The solution:
1. Decompose bulge into: magnitude (universal) × direction (per-position)
2. Learn per-position directions from training
3. Transfer: use source's per-position directions with target's geodesic

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def collect_training_data(model, tokenizer, n_tokens: int = 6):
    """Collect training trajectories."""
    
    examples = [
        ("The capital of France is", "France"),
        ("The capital of Germany is", "Germany"),
        ("The capital of Italy is", "Italy"),
        ("The capital of Spain is", "Spain"),
    ]
    
    trajectories = []
    all_tokens = []
    entities = []
    
    for prompt, entity in examples:
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
        entities.append(entity)
    
    return trajectories, all_tokens, entities


def extract_per_position_pattern(trajectories: List[torch.Tensor], P: torch.Tensor):
    """
    Extract per-position bulge pattern.
    
    For each position j:
    - Compute bulge[j] = actual[j] - geodesic[j]
    - Decompose into magnitude and direction
    - Average directions across entities (pattern is shared)
    """
    print("\n" + "=" * 70)
    print("Extracting Per-Position Pattern")
    print("=" * 70)
    
    n_steps = len(trajectories[0])
    
    # Collect bulges per position
    bulges_by_position = [[] for _ in range(n_steps)]
    
    for traj in trajectories:
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            bulge = traj_proj[j] - h_geo
            bulges_by_position[j].append(bulge)
    
    # For each position, compute:
    # 1. Average magnitude
    # 2. Average direction (normalized)
    # 3. SVD basis for that position
    
    position_patterns = []
    
    for j in range(n_steps):
        bulges_j = torch.stack(bulges_by_position[j])
        
        # Average magnitude
        mags = bulges_j.norm(dim=1)
        avg_mag = mags.mean().item()
        
        # SVD to get principal direction
        U, S, Vt = torch.linalg.svd(bulges_j, full_matrices=False)
        
        # Top direction captures most variance
        principal_dir = Vt[0]
        variance_captured = (S[0]**2 / (S**2).sum()).item()
        
        position_patterns.append({
            'magnitude': avg_mag,
            'direction': principal_dir,
            'variance_captured': variance_captured,
            'basis': Vt[:5],  # Top 5 directions
            'coeffs': bulges_j @ Vt[:5].T,  # Coefficients for each entity
        })
        
        print(f"  Position {j}: mag={avg_mag:.1f}, var_captured={variance_captured*100:.1f}%")
    
    return position_patterns


def reconstruct_with_pattern(h_start: torch.Tensor, h_end: torch.Tensor,
                             position_patterns: List[Dict], n_steps: int,
                             entity_idx: int = 0) -> List[torch.Tensor]:
    """
    Reconstruct trajectory using extracted pattern.
    
    For each position:
    - Compute geodesic point
    - Add bulge = coeffs[entity_idx] @ basis
    """
    trajectory = []
    
    for j in range(n_steps):
        t = j / (n_steps - 1) if n_steps > 1 else 0
        h_geo = (1 - t) * h_start + t * h_end
        
        # Reconstruct bulge from basis
        pattern = position_patterns[j]
        coeffs = pattern['coeffs'][entity_idx]
        basis = pattern['basis']
        bulge = coeffs @ basis
        
        h_j = h_geo + bulge
        trajectory.append(h_j)
    
    return trajectory


def test_reconstruction(model, tokenizer, trajectories, all_tokens, entities, 
                        P, position_patterns):
    """Test reconstruction accuracy."""
    print("\n" + "=" * 70)
    print("Test Reconstruction (Per-Position Pattern)")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    total_correct = 0
    total_tokens = 0
    
    for i, (traj, toks, entity) in enumerate(zip(trajectories, all_tokens, entities)):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        # Reconstruct
        recon_traj = reconstruct_with_pattern(h_start, h_end, position_patterns, 
                                               len(traj), entity_idx=i)
        
        # Decode
        recon_tokens = []
        for h_j in recon_traj:
            h_full = h_j @ P
            logits = h_full @ lm_head.T
            token_id = logits.argmax().item()
            recon_tokens.append(token_id)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in recon_tokens]
        
        match = sum(1 for a, r in zip(toks, recon_tokens) if a == r)
        total_correct += match
        total_tokens += len(toks)
        
        print(f"\n{entity}:")
        print(f"  Actual: {actual_text}")
        print(f"  Recon:  {recon_text}")
        print(f"  Match:  {match}/{len(toks)} = {match/len(toks)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_correct}/{total_tokens} = {total_correct/total_tokens*100:.1f}%")
    
    return total_correct / total_tokens


def transfer_pattern(model, tokenizer, source_idx: int, target_prompt: str,
                     trajectories, all_tokens, entities, P, position_patterns):
    """
    Transfer pattern from source entity to new target.
    
    Key: Use source's per-position coefficients with target's geodesic.
    """
    print(f"\n--- Transfer from {entities[source_idx]} ---")
    
    lm_head = model.lm_head.weight.data
    
    source_traj = trajectories[source_idx]
    source_toks = all_tokens[source_idx]
    source_proj = source_traj @ P.T
    
    # Get target's start
    input_ids = tokenizer.encode(target_prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        target_start_full = outputs.hidden_states[-1][0, -1, :]
    
    target_start = target_start_full @ P.T
    
    # Estimate target's end using source's offset
    source_offset = source_proj[-1] - source_proj[0]
    target_end = target_start + source_offset
    
    # Reconstruct using source's coefficients
    n_steps = len(source_traj)
    target_traj = []
    
    for j in range(n_steps):
        t = j / (n_steps - 1) if n_steps > 1 else 0
        h_geo = (1 - t) * target_start + t * target_end
        
        # Use source's coefficients
        pattern = position_patterns[j]
        coeffs = pattern['coeffs'][source_idx]
        basis = pattern['basis']
        bulge = coeffs @ basis
        
        h_j = h_geo + bulge
        target_traj.append(h_j)
    
    # Decode
    target_tokens = []
    for h_j in target_traj:
        h_full = h_j @ P
        logits = h_full @ lm_head.T
        token_id = logits.argmax().item()
        target_tokens.append(token_id)
    
    # Autoregressive for comparison
    input_ids = tokenizer.encode(target_prompt, return_tensors='pt')
    auto_tokens = []
    for _ in range(n_steps):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token = outputs.logits[0, -1, :].argmax()
            auto_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    source_text = [tokenizer.decode([t]) for t in source_toks]
    target_text = [tokenizer.decode([t]) for t in target_tokens]
    auto_text = [tokenizer.decode([t]) for t in auto_tokens]
    
    print(f"  Source: {source_text}")
    print(f"  Target prompt: {target_prompt!r}")
    print(f"  Transferred: {target_text}")
    print(f"  Autoregress: {auto_text}")
    
    match = sum(1 for t, a in zip(target_tokens, auto_tokens) if t == a)
    print(f"  Match: {match}/{n_steps} = {match/n_steps*100:.1f}%")
    
    return match / n_steps


def transfer_with_mean_pattern(model, tokenizer, target_prompt: str,
                               trajectories, all_tokens, entities, P, position_patterns):
    """
    Transfer using MEAN pattern (averaged across all training entities).
    
    This tests if the pattern is truly universal.
    """
    print(f"\n--- Transfer with Mean Pattern ---")
    
    lm_head = model.lm_head.weight.data
    n_steps = len(trajectories[0])
    
    # Get target's start
    input_ids = tokenizer.encode(target_prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        target_start_full = outputs.hidden_states[-1][0, -1, :]
    
    target_start = target_start_full @ P.T
    
    # Compute mean offset from training
    mean_offset = torch.zeros_like(target_start)
    for traj in trajectories:
        traj_proj = traj @ P.T
        mean_offset += (traj_proj[-1] - traj_proj[0])
    mean_offset /= len(trajectories)
    
    target_end = target_start + mean_offset
    
    # Compute mean coefficients per position
    target_traj = []
    
    for j in range(n_steps):
        t = j / (n_steps - 1) if n_steps > 1 else 0
        h_geo = (1 - t) * target_start + t * target_end
        
        # Mean coefficients
        pattern = position_patterns[j]
        mean_coeffs = pattern['coeffs'].mean(dim=0)
        basis = pattern['basis']
        bulge = mean_coeffs @ basis
        
        h_j = h_geo + bulge
        target_traj.append(h_j)
    
    # Decode
    target_tokens = []
    for h_j in target_traj:
        h_full = h_j @ P
        logits = h_full @ lm_head.T
        token_id = logits.argmax().item()
        target_tokens.append(token_id)
    
    # Autoregressive
    input_ids = tokenizer.encode(target_prompt, return_tensors='pt')
    auto_tokens = []
    for _ in range(n_steps):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token = outputs.logits[0, -1, :].argmax()
            auto_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    target_text = [tokenizer.decode([t]) for t in target_tokens]
    auto_text = [tokenizer.decode([t]) for t in auto_tokens]
    
    print(f"  Target prompt: {target_prompt!r}")
    print(f"  Mean pattern: {target_text}")
    print(f"  Autoregress:  {auto_text}")
    
    match = sum(1 for t, a in zip(target_tokens, auto_tokens) if t == a)
    print(f"  Match: {match}/{n_steps} = {match/n_steps*100:.1f}%")
    
    return match / n_steps


def main():
    print("=" * 70)
    print("Constrained Pattern Transfer: Per-Position Approach")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect training data
    trajectories, all_tokens, entities = collect_training_data(model, tokenizer, n_tokens=6)
    
    print(f"\nCollected {len(trajectories)} trajectories")
    for toks, entity in zip(all_tokens, entities):
        text = [tokenizer.decode([t]) for t in toks]
        print(f"  {entity}: {text}")
    
    # Compute projection
    all_points = torch.cat(trajectories, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    # Extract per-position pattern
    position_patterns = extract_per_position_pattern(trajectories, P)
    
    # Test reconstruction (should be 100%)
    recon_acc = test_reconstruction(model, tokenizer, trajectories, all_tokens, 
                                     entities, P, position_patterns)
    
    # Transfer to new entities
    print("\n" + "=" * 70)
    print("Transfer to New Entities")
    print("=" * 70)
    
    new_prompts = [
        "The capital of Japan is",
        "The capital of Poland is",
        "The capital of Brazil is",
    ]
    
    # Test transfer from each source
    for target_prompt in new_prompts:
        print(f"\n{'='*70}")
        print(f"Target: {target_prompt!r}")
        
        # Transfer from each source
        for source_idx in range(len(entities)):
            transfer_pattern(model, tokenizer, source_idx, target_prompt,
                           trajectories, all_tokens, entities, P, position_patterns)
        
        # Transfer with mean pattern
        transfer_with_mean_pattern(model, tokenizer, target_prompt,
                                   trajectories, all_tokens, entities, P, position_patterns)


if __name__ == "__main__":
    main()
