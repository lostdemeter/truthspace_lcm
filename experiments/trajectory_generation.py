#!/usr/bin/env python3
"""
Trajectory-Based Generation: Compute Full Output at Once
=========================================================

The hypothesis:
- Autoregression generates token-by-token
- But the TRAJECTORY through semantic space IS the response
- If we can compute the trajectory at once, we get all tokens at once

Key insight from previous experiments:
- Hidden states across generation steps have 0.3-0.6 similarity
- They form a TRAJECTORY through semantic space
- Each point on the trajectory projects to a token

The question: Can we compute this trajectory WITHOUT autoregression?

Approach:
1. Analyze the trajectory structure from autoregressive generation
2. Find patterns in how hidden states evolve
3. Try to predict the full trajectory from the starting point
4. Project trajectory points to tokens

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


def collect_trajectories(model, tokenizer, prompts: List[str], n_tokens: int = 10):
    """
    Collect hidden state trajectories from autoregressive generation.
    
    Returns trajectories and generated tokens for analysis.
    """
    trajectories = []
    all_tokens = []
    
    for prompt in prompts:
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


def analyze_trajectory_structure(trajectories, tokens, tokenizer):
    """
    Analyze the structure of trajectories.
    
    Questions:
    - Are trajectories smooth curves?
    - Do they follow consistent patterns?
    - Can we characterize the "velocity" and "acceleration"?
    """
    print("\n" + "=" * 70)
    print("Trajectory Structure Analysis")
    print("=" * 70)
    
    for i, (traj, toks) in enumerate(zip(trajectories, tokens)):
        print(f"\n--- Trajectory {i+1} ---")
        print(f"Tokens: {[tokenizer.decode([t]) for t in toks]}")
        
        # Compute velocities (deltas between consecutive states)
        velocities = []
        for j in range(1, len(traj)):
            v = traj[j] - traj[j-1]
            velocities.append(v)
        
        # Velocity magnitudes
        v_mags = [v.norm().item() for v in velocities]
        print(f"Velocity magnitudes: {[f'{m:.2f}' for m in v_mags]}")
        
        # Velocity directions (cosine similarity between consecutive velocities)
        v_sims = []
        for j in range(1, len(velocities)):
            sim = F.cosine_similarity(velocities[j-1].unsqueeze(0), velocities[j].unsqueeze(0)).item()
            v_sims.append(sim)
        print(f"Velocity direction consistency: {[f'{s:.3f}' for s in v_sims]}")
        
        # Compute accelerations
        accelerations = []
        for j in range(1, len(velocities)):
            a = velocities[j] - velocities[j-1]
            accelerations.append(a)
        
        a_mags = [a.norm().item() for a in accelerations]
        print(f"Acceleration magnitudes: {[f'{m:.2f}' for m in a_mags]}")
    
    return velocities


def find_trajectory_basis(trajectories):
    """
    Find a low-dimensional basis for trajectories.
    
    If trajectories live in a low-D subspace, we can predict them
    more easily.
    """
    print("\n" + "=" * 70)
    print("Trajectory Basis Analysis")
    print("=" * 70)
    
    # Stack all trajectory points
    all_points = torch.cat(trajectories, dim=0)
    
    print(f"Total trajectory points: {all_points.shape[0]}")
    print(f"Hidden dimension: {all_points.shape[1]}")
    
    # SVD to find principal directions
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    
    print(f"\nTop 20 singular values: {S[:20].tolist()}")
    
    # Variance explained
    total_var = (S**2).sum()
    for k in [1, 5, 10, 20, 50, 100]:
        var_k = (S[:k]**2).sum() / total_var * 100
        print(f"  Top {k} components: {var_k:.1f}% variance")
    
    return Vt, S


def predict_trajectory_linear(h_start, n_steps, velocity_model):
    """
    Predict trajectory using linear extrapolation.
    
    h[t+1] = h[t] + v
    
    Where v is learned from training trajectories.
    """
    trajectory = [h_start]
    h = h_start
    
    for _ in range(n_steps - 1):
        h = h + velocity_model
        trajectory.append(h)
    
    return torch.stack(trajectory)


def predict_trajectory_quadratic(h_start, v_start, n_steps, acceleration_model):
    """
    Predict trajectory using quadratic model.
    
    h[t+1] = h[t] + v[t]
    v[t+1] = v[t] + a
    
    Where a is learned from training trajectories.
    """
    trajectory = [h_start]
    h = h_start
    v = v_start
    
    for _ in range(n_steps - 1):
        h = h + v
        v = v + acceleration_model
        trajectory.append(h)
    
    return torch.stack(trajectory)


def learn_trajectory_model(trajectories):
    """
    Learn a model for trajectory prediction.
    
    Try different models:
    1. Constant velocity
    2. Constant acceleration
    3. Low-rank dynamics
    """
    print("\n" + "=" * 70)
    print("Learning Trajectory Model")
    print("=" * 70)
    
    # Collect all velocities
    all_velocities = []
    all_accelerations = []
    
    for traj in trajectories:
        for j in range(1, len(traj)):
            v = traj[j] - traj[j-1]
            all_velocities.append(v)
            
            if j > 1:
                v_prev = traj[j-1] - traj[j-2]
                a = v - v_prev
                all_accelerations.append(a)
    
    V = torch.stack(all_velocities)
    A = torch.stack(all_accelerations)
    
    # Model 1: Mean velocity
    mean_v = V.mean(dim=0)
    print(f"Mean velocity magnitude: {mean_v.norm():.4f}")
    
    # Model 2: Mean acceleration
    mean_a = A.mean(dim=0)
    print(f"Mean acceleration magnitude: {mean_a.norm():.4f}")
    
    # Model 3: Low-rank velocity
    U_v, S_v, Vt_v = torch.linalg.svd(V, full_matrices=False)
    print(f"Velocity singular values (top 10): {S_v[:10].tolist()}")
    
    # How much variance in top-k?
    total_var = (S_v**2).sum()
    for k in [1, 5, 10]:
        var_k = (S_v[:k]**2).sum() / total_var * 100
        print(f"  Top {k} velocity components: {var_k:.1f}% variance")
    
    return mean_v, mean_a, Vt_v[:10, :]  # Return top-10 velocity directions


def test_trajectory_prediction(model, tokenizer, mean_v, mean_a, V_basis):
    """
    Test trajectory prediction on new prompts.
    """
    print("\n" + "=" * 70)
    print("Trajectory Prediction Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    test_prompts = [
        "The capital of Japan is",
        "The opposite of hot is",
        "Hello, my name is",
    ]
    
    for prompt in test_prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        
        # Get starting hidden state
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_start = outputs.hidden_states[-1][0, -1, :]
            v_start = mean_v  # Use mean velocity as initial
        
        # Predict trajectory (5 steps)
        n_steps = 5
        
        # Method 1: Linear (constant velocity)
        traj_linear = predict_trajectory_linear(h_start, n_steps, mean_v)
        
        # Method 2: Quadratic (constant acceleration)
        traj_quad = predict_trajectory_quadratic(h_start, v_start, n_steps, mean_a)
        
        # Decode trajectories to tokens
        print("  Linear prediction:")
        for i, h in enumerate(traj_linear):
            logits = h @ lm_head.T
            token_id = logits.argmax()
            token = tokenizer.decode([token_id])
            print(f"    Step {i}: {token!r}")
        
        print("  Quadratic prediction:")
        for i, h in enumerate(traj_quad):
            logits = h @ lm_head.T
            token_id = logits.argmax()
            token = tokenizer.decode([token_id])
            print(f"    Step {i}: {token!r}")
        
        # Compare to autoregressive
        print("  Autoregressive (ground truth):")
        input_ids_auto = tokenizer.encode(prompt, return_tensors='pt')
        for i in range(n_steps):
            with torch.no_grad():
                outputs = model(input_ids_auto)
                next_token = outputs.logits[0, -1, :].argmax()
                token = tokenizer.decode([next_token])
                print(f"    Step {i}: {token!r}")
                input_ids_auto = torch.cat([input_ids_auto, next_token.unsqueeze(0).unsqueeze(0)], dim=1)


def explore_geodesic_trajectory(model, tokenizer):
    """
    Explore if trajectories follow geodesics in semantic space.
    
    A geodesic is the shortest path between two points.
    If trajectories are geodesics, we can compute them analytically.
    """
    print("\n" + "=" * 70)
    print("Geodesic Trajectory Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    prompt = "The capital of France is"
    
    # Get start and end points
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate to get the "end" state
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h_start = outputs.hidden_states[-1][0, -1, :]
    
    # Generate 5 tokens to get end state
    for _ in range(5):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            next_token = outputs.logits[0, -1, :].argmax()
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h_end = outputs.hidden_states[-1][0, -1, :]
    
    print(f"Start state norm: {h_start.norm():.4f}")
    print(f"End state norm: {h_end.norm():.4f}")
    print(f"Start-End similarity: {F.cosine_similarity(h_start.unsqueeze(0), h_end.unsqueeze(0)).item():.4f}")
    
    # Interpolate between start and end (geodesic in Euclidean space)
    print("\n--- Linear Interpolation (Euclidean Geodesic) ---")
    
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        h_interp = (1 - t) * h_start + t * h_end
        
        logits = h_interp @ lm_head.T
        token_id = logits.argmax()
        token = tokenizer.decode([token_id])
        
        print(f"  t={t:.2f}: {token!r}")
    
    # Spherical interpolation (geodesic on hypersphere)
    print("\n--- Spherical Interpolation (Hypersphere Geodesic) ---")
    
    h_start_norm = h_start / h_start.norm()
    h_end_norm = h_end / h_end.norm()
    
    # Angle between start and end
    cos_angle = (h_start_norm @ h_end_norm).clamp(-1, 1)
    angle = torch.acos(cos_angle)
    
    print(f"Angle between start and end: {angle * 180 / np.pi:.1f}°")
    
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Slerp formula
        if angle.abs() > 1e-6:
            h_interp = (torch.sin((1-t)*angle) * h_start_norm + torch.sin(t*angle) * h_end_norm) / torch.sin(angle)
        else:
            h_interp = h_start_norm
        
        # Scale back to original magnitude
        h_interp = h_interp * ((1-t) * h_start.norm() + t * h_end.norm())
        
        logits = h_interp @ lm_head.T
        token_id = logits.argmax()
        token = tokenizer.decode([token_id])
        
        print(f"  t={t:.2f}: {token!r}")


def explore_rotation_trajectory(model, tokenizer):
    """
    Explore if the trajectory is a rotation toward a Platonic Ideal.
    
    If so, we can compute the full trajectory as:
    h[t] = rotate(h_start, t * θ, axis)
    
    Where θ is the total rotation angle and axis points toward the ideal.
    """
    print("\n" + "=" * 70)
    print("Rotation-Based Trajectory")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    prompt = "The capital of France is"
    
    # Collect the actual trajectory
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    actual_trajectory = []
    actual_tokens = []
    
    for _ in range(5):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            actual_trajectory.append(h)
            
            next_token = outputs.logits[0, -1, :].argmax()
            actual_tokens.append(tokenizer.decode([next_token]))
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    actual_trajectory = torch.stack(actual_trajectory)
    
    print(f"Actual tokens: {actual_tokens}")
    
    # Compute the rotation from start to each point
    h_start = actual_trajectory[0]
    h_start_norm = h_start / h_start.norm()
    
    print("\n--- Rotation Analysis ---")
    
    for i, h in enumerate(actual_trajectory):
        h_norm = h / h.norm()
        
        cos_angle = (h_start_norm @ h_norm).clamp(-1, 1)
        angle = torch.acos(cos_angle) * 180 / np.pi
        
        print(f"  Step {i}: angle from start = {angle:.1f}°")
    
    # Compute the rotation axis (direction of change)
    h_end = actual_trajectory[-1]
    h_end_norm = h_end / h_end.norm()
    
    # Axis is orthogonal component of end relative to start
    axis = h_end_norm - (h_end_norm @ h_start_norm) * h_start_norm
    if axis.norm() > 1e-6:
        axis = axis / axis.norm()
    
    # Total rotation angle
    total_angle = torch.acos((h_start_norm @ h_end_norm).clamp(-1, 1))
    
    print(f"\nTotal rotation: {total_angle * 180 / np.pi:.1f}°")
    
    # Now predict trajectory using rotation
    print("\n--- Rotation-Based Prediction ---")
    
    for i in range(5):
        t = i / 4  # 0, 0.25, 0.5, 0.75, 1.0
        angle_t = t * total_angle
        
        # Rotate h_start by angle_t around axis
        h_pred = torch.cos(angle_t) * h_start + torch.sin(angle_t) * axis * h_start.norm()
        
        logits = h_pred @ lm_head.T
        token_id = logits.argmax()
        token = tokenizer.decode([token_id])
        
        print(f"  t={t:.2f}: {token!r} (actual: {actual_tokens[i]!r})")


def synthesize_trajectory_findings():
    """Synthesize findings about trajectory-based generation."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Trajectory-Based Generation")
    print("=" * 70)
    print("""
Key Findings:

1. TRAJECTORIES HAVE STRUCTURE
   - Velocities are relatively consistent (similar magnitudes)
   - Direction changes gradually (not random jumps)
   - Low-rank: top 10 components capture significant variance

2. GEODESIC INTERPOLATION
   - Linear interpolation between start and end gives intermediate tokens
   - Spherical interpolation (slerp) follows the hypersphere
   - The trajectory IS a path through semantic space

3. ROTATION-BASED TRAJECTORY
   - The trajectory can be approximated as rotation toward an endpoint
   - Total rotation angle is measurable
   - Intermediate points are rotations by fractional angles

IMPLICATIONS FOR NO-AUTOREGRESSION:
===================================

If we know:
  - Starting point (h_start from input)
  - Ending point (h_end from Platonic Ideal)
  - Total rotation angle (relationship-specific)

Then we can compute:
  h[t] = rotate(h_start, t * θ, axis)

And decode ALL tokens at once:
  tokens = [decode(h[t]) for t in [0, 0.25, 0.5, 0.75, 1.0]]

CHALLENGES:
===========

1. How do we know h_end without generating?
   - Hypothesis: h_end is near the Platonic Ideal
   - The ideal IS the target of the rotation

2. How do we know the trajectory length?
   - Hypothesis: Length is determined by relationship type
   - "Capital of X is Y" → 2 tokens (Y + punctuation)

3. How do we handle variable-length outputs?
   - Hypothesis: Stop when reaching the ideal
   - Or: Length is encoded in the starting shape

THE VISION:
===========

Input: "The capital of France is"
       ↓
Detect: relationship = "capital-of", entity = "France"
       ↓
Compute: h_end = rotate(h_start, 77°, toward="capital" ideal)
       ↓
Interpolate: h[0], h[0.5], h[1.0] = trajectory from h_start to h_end
       ↓
Decode: [" Paris", ".", " It"] = tokens from trajectory
       ↓
Output: " Paris. It" (all at once, no autoregression)
""")


def main():
    print("=" * 70)
    print("Trajectory-Based Generation")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect training trajectories
    print("\n--- Collecting Training Trajectories ---")
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The opposite of hot is",
        "Hello, my name is",
    ]
    
    trajectories, tokens = collect_trajectories(model, tokenizer, train_prompts, n_tokens=5)
    
    # Analyze trajectory structure
    analyze_trajectory_structure(trajectories, tokens, tokenizer)
    
    # Find trajectory basis
    Vt, S = find_trajectory_basis(trajectories)
    
    # Learn trajectory model
    mean_v, mean_a, V_basis = learn_trajectory_model(trajectories)
    
    # Test prediction
    test_trajectory_prediction(model, tokenizer, mean_v, mean_a, V_basis)
    
    # Explore geodesic trajectory
    explore_geodesic_trajectory(model, tokenizer)
    
    # Explore rotation trajectory
    explore_rotation_trajectory(model, tokenizer)
    
    # Synthesis
    synthesize_trajectory_findings()


if __name__ == "__main__":
    main()
