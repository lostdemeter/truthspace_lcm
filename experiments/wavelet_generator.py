#!/usr/bin/env python3
"""
Wavelet-Based Generator: Full Generation Without Autoregression
================================================================

This implements the complete wavelet-based generation model:

1. LEARN PHASE:
   - Collect trajectories from autoregressive generation
   - Compute bulges (deviation from geodesic)
   - SVD to get basis bulges (wavelets)
   - Store coefficients per entity

2. GENERATE PHASE:
   - Compute geodesic from start to Platonic Ideal
   - Look up coefficients for entity
   - Reconstruct bulge: Σ c_i × ψ_i(t)
   - Add bulge to geodesic
   - Decode ALL tokens at once (no autoregression!)

If this works, we've achieved fully geometric generation.

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


@dataclass
class WaveletMemory:
    """
    Memory storing bulge coefficients per entity.
    
    Instead of storing full trajectories (28,672 floats),
    we store 10 coefficients per entity.
    """
    # Basis bulges (wavelets) - learned from SVD
    basis: Optional[torch.Tensor] = None  # Shape: [n_basis, proj_dim]
    
    # Projection matrix
    P: Optional[torch.Tensor] = None  # Shape: [proj_dim, hidden_dim]
    
    # Coefficients per entity
    coefficients: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Geodesic endpoints per entity
    geodesic_start: Dict[str, torch.Tensor] = field(default_factory=dict)
    geodesic_end: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Number of steps per entity
    n_steps: Dict[str, int] = field(default_factory=dict)


class WaveletGenerator:
    """
    Generator that uses wavelet-based bulge reconstruction.
    """
    
    def __init__(self, model, tokenizer, n_basis: int = 10, proj_dim: int = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.n_basis = n_basis
        self.proj_dim = proj_dim
        
        self.memory = WaveletMemory()
        self.lm_head = model.lm_head.weight.data
        
    def learn_from_trajectories(self, prompts: List[str], entities: List[str], n_tokens: int = 8):
        """
        Learn basis bulges and store coefficients from training trajectories.
        """
        print("\n" + "=" * 70)
        print("Learning Wavelet Basis and Coefficients")
        print("=" * 70)
        
        # Collect trajectories
        trajectories = []
        all_tokens = []
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            hidden_states = []
            tokens = []
            
            for i in range(n_tokens):
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    h = outputs.hidden_states[-1][0, -1, :]
                    hidden_states.append(h)
                    
                    next_token = outputs.logits[0, -1, :].argmax()
                    tokens.append(next_token.item())
                    
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            trajectories.append(torch.stack(hidden_states))
            all_tokens.append(tokens)
        
        print(f"Collected {len(trajectories)} trajectories")
        
        # Compute projection matrix
        all_points = torch.cat(trajectories, dim=0)
        U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
        self.memory.P = Vt[:self.proj_dim, :]
        
        print(f"Projection: {all_points.shape[1]}D → {self.proj_dim}D")
        
        # Compute bulges
        all_bulges = []
        
        for i, (traj, entity) in enumerate(zip(trajectories, entities)):
            traj_proj = traj @ self.memory.P.T
            h_start = traj_proj[0]
            h_end = traj_proj[-1]
            
            # Store geodesic endpoints
            self.memory.geodesic_start[entity] = h_start
            self.memory.geodesic_end[entity] = h_end
            self.memory.n_steps[entity] = len(traj)
            
            # Compute bulges
            n_steps = len(traj)
            for j in range(n_steps):
                t = j / (n_steps - 1) if n_steps > 1 else 0
                h_geo = (1 - t) * h_start + t * h_end
                bulge = traj_proj[j] - h_geo
                all_bulges.append(bulge)
        
        all_bulges = torch.stack(all_bulges)
        print(f"Computed {len(all_bulges)} bulge vectors")
        
        # SVD to get basis bulges
        U_b, S_b, Vt_b = torch.linalg.svd(all_bulges, full_matrices=False)
        
        # Store top n_basis as wavelets
        self.memory.basis = Vt_b[:self.n_basis, :]
        
        print(f"Learned {self.n_basis} basis bulges")
        print(f"Variance explained: {(S_b[:self.n_basis]**2).sum() / (S_b**2).sum() * 100:.1f}%")
        
        # Compute and store coefficients for each entity
        idx = 0
        for i, (traj, entity) in enumerate(zip(trajectories, entities)):
            n_steps = len(traj)
            
            # Get bulges for this trajectory
            entity_bulges = all_bulges[idx:idx + n_steps]
            idx += n_steps
            
            # Project onto basis to get coefficients
            # coefficients[j] = bulge[j] @ basis.T
            # We want one set of coefficients per entity (average over positions)
            mean_bulge = entity_bulges.mean(dim=0)
            coeffs = mean_bulge @ self.memory.basis.T
            
            self.memory.coefficients[entity] = coeffs
            
            print(f"  {entity}: coeffs = {coeffs[:5].tolist()}...")
        
        return trajectories, all_tokens
    
    def generate_wavelet(self, entity: str, n_steps: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate trajectory using wavelet reconstruction.
        
        No autoregression - all at once!
        """
        if entity not in self.memory.coefficients:
            raise ValueError(f"Entity '{entity}' not in memory")
        
        # Get stored values
        h_start = self.memory.geodesic_start[entity]
        h_end = self.memory.geodesic_end[entity]
        coeffs = self.memory.coefficients[entity]
        
        if n_steps is None:
            n_steps = self.memory.n_steps[entity]
        
        # Reconstruct trajectory
        trajectory = []
        tokens = []
        
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            
            # Geodesic point
            h_geo = (1 - t) * h_start + t * h_end
            
            # Reconstruct bulge from coefficients
            # bulge = Σ c_i × ψ_i
            bulge = coeffs @ self.memory.basis
            
            # Scale bulge by position (zero at start/end, peak in middle)
            # This is the universal bulge shape
            bulge_scale = 4 * t * (1 - t)  # Parabola: 0 at t=0,1, max at t=0.5
            
            # Add scaled bulge to geodesic
            h_j = h_geo + bulge_scale * bulge
            
            trajectory.append(h_j)
            
            # Decode token
            h_full = h_j @ self.memory.P
            logits = h_full @ self.lm_head.T
            token_id = logits.argmax().item()
            tokens.append(token_id)
        
        return torch.stack(trajectory), tokens
    
    def generate_autoregressive(self, prompt: str, n_tokens: int) -> List[int]:
        """Traditional autoregressive generation for comparison."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        tokens = []
        
        for _ in range(n_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token = outputs.logits[0, -1, :].argmax()
                tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return tokens


def test_wavelet_generation(model, tokenizer):
    """
    Test wavelet-based generation vs autoregressive.
    """
    print("\n" + "=" * 70)
    print("Wavelet Generation Test")
    print("=" * 70)
    
    # Training prompts and entities
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    train_entities = ["France", "Germany", "Italy", "Spain"]
    
    # Create generator
    gen = WaveletGenerator(model, tokenizer, n_basis=10, proj_dim=100)
    
    # Learn from training data
    train_trajectories, train_tokens = gen.learn_from_trajectories(
        train_prompts, train_entities, n_tokens=6
    )
    
    # Test on training data (should work well)
    print("\n" + "=" * 70)
    print("Test on Training Data")
    print("=" * 70)
    
    for prompt, entity, actual_tokens in zip(train_prompts, train_entities, train_tokens):
        print(f"\n--- {entity} ---")
        print(f"Prompt: {prompt!r}")
        
        # Wavelet generation
        _, wavelet_tokens = gen.generate_wavelet(entity)
        
        # Autoregressive generation
        auto_tokens = gen.generate_autoregressive(prompt, len(actual_tokens))
        
        # Compare
        wavelet_text = [tokenizer.decode([t]) for t in wavelet_tokens]
        auto_text = [tokenizer.decode([t]) for t in auto_tokens]
        actual_text = [tokenizer.decode([t]) for t in actual_tokens]
        
        print(f"  Actual:     {actual_text}")
        print(f"  Wavelet:    {wavelet_text}")
        print(f"  Autoregr:   {auto_text}")
        
        # Accuracy
        wavelet_correct = sum(1 for w, a in zip(wavelet_tokens, actual_tokens) if w == a)
        auto_correct = sum(1 for w, a in zip(auto_tokens, actual_tokens) if w == a)
        
        print(f"  Wavelet accuracy: {wavelet_correct}/{len(actual_tokens)} = {wavelet_correct/len(actual_tokens)*100:.1f}%")
        print(f"  Auto accuracy:    {auto_correct}/{len(actual_tokens)} = {auto_correct/len(actual_tokens)*100:.1f}%")
    
    return gen


def test_generalization(gen, tokenizer):
    """
    Test if wavelet generation generalizes to unseen entities.
    """
    print("\n" + "=" * 70)
    print("Generalization Test (Unseen Entities)")
    print("=" * 70)
    
    # Test prompts with entities NOT in training
    test_prompts = [
        ("The capital of Japan is", "Japan"),
        ("The capital of China is", "China"),
        ("The capital of Poland is", "Poland"),
    ]
    
    # For unseen entities, we need to compute their coefficients
    # This requires one forward pass to get the starting hidden state
    
    for prompt, entity in test_prompts:
        print(f"\n--- {entity} (UNSEEN) ---")
        print(f"Prompt: {prompt!r}")
        
        # Get starting hidden state
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = gen.model(input_ids, output_hidden_states=True)
            h_start_full = outputs.hidden_states[-1][0, -1, :]
        
        # Project to manifold
        h_start = h_start_full @ gen.memory.P.T
        
        # For unseen entity, we need to estimate coefficients
        # Option 1: Use mean coefficients from training
        mean_coeffs = torch.stack(list(gen.memory.coefficients.values())).mean(dim=0)
        
        # Option 2: Find nearest training entity
        best_sim = -1
        best_entity = None
        for train_entity, train_start in gen.memory.geodesic_start.items():
            sim = F.cosine_similarity(h_start.unsqueeze(0), train_start.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_entity = train_entity
        
        print(f"  Nearest training entity: {best_entity} (sim={best_sim:.4f})")
        
        # Use nearest entity's coefficients
        coeffs = gen.memory.coefficients[best_entity]
        
        # Estimate end point (use same offset as nearest entity)
        train_start = gen.memory.geodesic_start[best_entity]
        train_end = gen.memory.geodesic_end[best_entity]
        offset = train_end - train_start
        h_end = h_start + offset
        
        # Generate
        n_steps = 6
        wavelet_tokens = []
        
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            
            h_geo = (1 - t) * h_start + t * h_end
            bulge = coeffs @ gen.memory.basis
            bulge_scale = 4 * t * (1 - t)
            h_j = h_geo + bulge_scale * bulge
            
            h_full = h_j @ gen.memory.P
            logits = h_full @ gen.lm_head.T
            token_id = logits.argmax().item()
            wavelet_tokens.append(token_id)
        
        # Autoregressive for comparison
        auto_tokens = gen.generate_autoregressive(prompt, n_steps)
        
        wavelet_text = [tokenizer.decode([t]) for t in wavelet_tokens]
        auto_text = [tokenizer.decode([t]) for t in auto_tokens]
        
        print(f"  Wavelet:    {wavelet_text}")
        print(f"  Autoregr:   {auto_text}")
        
        # Check first token (most important)
        if wavelet_text[0].strip() == auto_text[0].strip():
            print(f"  First token: ✓ MATCH")
        else:
            print(f"  First token: ✗ MISMATCH")


def test_improved_wavelet(model, tokenizer):
    """
    Test improved wavelet generation with per-position coefficients.
    """
    print("\n" + "=" * 70)
    print("Improved Wavelet Generation (Per-Position Coefficients)")
    print("=" * 70)
    
    # Training data
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    train_entities = ["France", "Germany", "Italy", "Spain"]
    n_tokens = 6
    
    # Collect trajectories
    trajectories = []
    all_tokens = []
    
    for prompt in train_prompts:
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
    
    # Compute projection
    all_points = torch.cat(trajectories, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    lm_head = model.lm_head.weight.data
    
    # For each entity, store the ACTUAL trajectory (not just coefficients)
    # This is a baseline to see what's achievable
    
    print("\n--- Baseline: Store Full Trajectory ---")
    
    for i, (traj, toks, entity) in enumerate(zip(trajectories, all_tokens, train_entities)):
        print(f"\n{entity}:")
        
        # Decode from stored trajectory
        traj_proj = traj @ P.T
        
        decoded_tokens = []
        for h_proj in traj_proj:
            h_full = h_proj @ P
            logits = h_full @ lm_head.T
            token_id = logits.argmax().item()
            decoded_tokens.append(token_id)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        decoded_text = [tokenizer.decode([t]) for t in decoded_tokens]
        
        print(f"  Actual:  {actual_text}")
        print(f"  Decoded: {decoded_text}")
        
        correct = sum(1 for d, a in zip(decoded_tokens, toks) if d == a)
        print(f"  Accuracy: {correct}/{len(toks)} = {correct/len(toks)*100:.1f}%")
    
    # Now test with bulge reconstruction
    print("\n--- Bulge Reconstruction ---")
    
    # Compute bulges for all trajectories
    all_bulges_by_position = [[] for _ in range(n_tokens)]
    
    for traj in trajectories:
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        for j in range(n_tokens):
            t = j / (n_tokens - 1) if n_tokens > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            bulge = traj_proj[j] - h_geo
            all_bulges_by_position[j].append(bulge)
    
    # SVD per position to get position-specific basis
    position_basis = []
    for j in range(n_tokens):
        bulges_j = torch.stack(all_bulges_by_position[j])
        U_j, S_j, Vt_j = torch.linalg.svd(bulges_j, full_matrices=False)
        position_basis.append(Vt_j[:5, :])  # Top 5 per position
    
    # Store coefficients per entity per position
    entity_coeffs = {}
    
    for i, (traj, entity) in enumerate(zip(trajectories, train_entities)):
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        coeffs_list = []
        for j in range(n_tokens):
            t = j / (n_tokens - 1) if n_tokens > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            bulge = traj_proj[j] - h_geo
            
            # Project onto position-specific basis
            coeffs_j = bulge @ position_basis[j].T
            coeffs_list.append(coeffs_j)
        
        entity_coeffs[entity] = coeffs_list
    
    # Reconstruct and test
    print("\nReconstruction with per-position coefficients:")
    
    for i, (traj, toks, entity) in enumerate(zip(trajectories, all_tokens, train_entities)):
        print(f"\n{entity}:")
        
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        reconstructed_tokens = []
        
        for j in range(n_tokens):
            t = j / (n_tokens - 1) if n_tokens > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            
            # Reconstruct bulge
            coeffs_j = entity_coeffs[entity][j]
            bulge_reconstructed = coeffs_j @ position_basis[j]
            
            h_reconstructed = h_geo + bulge_reconstructed
            
            # Decode
            h_full = h_reconstructed @ P
            logits = h_full @ lm_head.T
            token_id = logits.argmax().item()
            reconstructed_tokens.append(token_id)
        
        actual_text = [tokenizer.decode([t]) for t in toks]
        recon_text = [tokenizer.decode([t]) for t in reconstructed_tokens]
        
        print(f"  Actual:       {actual_text}")
        print(f"  Reconstructed: {recon_text}")
        
        correct = sum(1 for r, a in zip(reconstructed_tokens, toks) if r == a)
        print(f"  Accuracy: {correct}/{len(toks)} = {correct/len(toks)*100:.1f}%")


def synthesize_results():
    """Synthesize wavelet generation results."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Wavelet-Based Generation")
    print("=" * 70)
    print("""
Results Summary:

1. BASELINE (Store Full Trajectory)
   - Decode from stored hidden states
   - Shows what's achievable with perfect memory
   - Expected: ~100% on training data

2. WAVELET RECONSTRUCTION
   - Geodesic + bulge from coefficients
   - Tests if 10 coefficients capture enough
   - Key question: Does it match autoregressive?

3. PER-POSITION COEFFICIENTS
   - Different basis per position
   - More accurate but more storage
   - Trade-off: accuracy vs compression

KEY INSIGHTS:
=============

1. The projection (3584D → 100D) may lose information
   - Need to check if decoded tokens match

2. Bulge reconstruction quality depends on:
   - Number of basis functions
   - How well coefficients capture entity-specific info

3. Generalization requires:
   - Finding similar training entities
   - Transferring their coefficients

NEXT STEPS:
===========

If accuracy is low:
1. Increase projection dimension
2. Use more basis functions
3. Store per-position coefficients

If accuracy is high:
1. Test on more diverse prompts
2. Measure speedup vs autoregressive
3. Implement full generation pipeline
""")


def main():
    print("=" * 70)
    print("Wavelet-Based Generator: Full Generation Test")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test basic wavelet generation
    gen = test_wavelet_generation(model, tokenizer)
    
    # Test generalization
    test_generalization(gen, tokenizer)
    
    # Test improved version
    test_improved_wavelet(model, tokenizer)
    
    # Synthesis
    synthesize_results()


if __name__ == "__main__":
    main()
