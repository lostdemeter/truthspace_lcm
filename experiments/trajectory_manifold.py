#!/usr/bin/env python3
"""
Trajectory Manifold: Explicit Geometry for Autoregression
==========================================================

Key insight from HyperMapping and φ-Unraveled Engine:
- When we make geometry EXPLICIT, we don't need iteration
- MESH = W_q.T @ W_k captures the Q-K relationship
- What's the equivalent for autoregression?

The Trajectory Manifold Hypothesis:
-----------------------------------

For a given prompt, the hidden state trajectory lives on a low-dimensional
manifold. If we can characterize this manifold EXPLICITLY, we can:
1. Project the prompt onto the manifold
2. Read off all tokens at once
3. No iteration needed

This is like HyperMapping's "reproject()" - we construct the geometry we need.

The Unraveling Insight:
-----------------------

In the φ-Unraveled Engine, we pre-computed MESH = W_q.T @ W_k to avoid
error compounding from separate Q and K encoding.

For autoregression, the "self-referential structure" is:
- Token i depends on tokens 0..i-1
- This creates a chain of dependencies

Can we "unravel" this chain into a single geometric operation?

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


def analyze_trajectory_manifold():
    """
    Analyze the structure of the trajectory manifold.
    
    Key questions:
    1. What is the dimensionality of the manifold?
    2. Can we characterize it explicitly?
    3. Can we project prompts onto it directly?
    """
    print("\n" + "=" * 70)
    print("Trajectory Manifold Analysis")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect trajectories for multiple prompts
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
    ]
    
    n_tokens = 10
    
    all_trajectories = []
    all_tokens = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]
        
        # Collect trajectory
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
        
        text = tokenizer.decode(tokens)
        print(f"  Output: {text!r}")
        
        all_trajectories.append(torch.stack(trajectory))
        all_tokens.append(tokens)
    
    # Stack all trajectories: [n_prompts, n_tokens, hidden_dim]
    T = torch.stack(all_trajectories)
    print(f"\nTrajectory tensor shape: {T.shape}")
    
    # Analyze the manifold structure
    print("\n--- Manifold Structure ---")
    
    # Flatten to [n_prompts * n_tokens, hidden_dim]
    T_flat = T.reshape(-1, T.shape[-1])
    
    # SVD to find manifold dimensionality
    U, S, Vh = torch.linalg.svd(T_flat, full_matrices=False)
    
    print("Singular values (variance explained):")
    total_var = (S**2).sum()
    cumulative = 0
    for i in range(min(20, len(S))):
        var = (S[i]**2 / total_var).item() * 100
        cumulative += var
        print(f"  S[{i}] = {S[i].item():.2f} ({var:.1f}%, cumulative: {cumulative:.1f}%)")
        if cumulative > 99:
            print(f"  ... (99% variance at {i+1} dimensions)")
            break
    
    # The key insight: How many dimensions do we need?
    for threshold in [90, 95, 99, 99.9]:
        cumsum = torch.cumsum(S**2, dim=0) / total_var * 100
        n_dims = (cumsum < threshold).sum().item() + 1
        print(f"  {threshold}% variance: {n_dims} dimensions")
    
    # Analyze per-prompt structure
    print("\n--- Per-Prompt Manifold ---")
    
    for i, prompt in enumerate(prompts):
        traj = all_trajectories[i]  # [n_tokens, hidden_dim]
        
        # Center
        traj_centered = traj - traj.mean(dim=0)
        
        # SVD
        U_p, S_p, Vh_p = torch.linalg.svd(traj_centered, full_matrices=False)
        
        total_var_p = (S_p**2).sum()
        var_2d = ((S_p[:2]**2).sum() / total_var_p).item() * 100
        var_3d = ((S_p[:3]**2).sum() / total_var_p).item() * 100
        
        print(f"  {prompt[:20]}...: 2D={var_2d:.1f}%, 3D={var_3d:.1f}%")
    
    # The HyperMapping insight: Can we construct the geometry we need?
    print("\n--- Constructing Explicit Geometry ---")
    
    # Like HyperMapping's reproject(): build similarity matrix → eigendecomposition
    
    # Similarity between trajectories
    # S[i,j] = cosine_similarity(trajectory_i, trajectory_j)
    
    # But trajectories have different tokens... let's use the prompt hidden state
    
    # Get prompt hidden states
    prompt_hiddens = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]  # Last position
        
        prompt_hiddens.append(h)
    
    prompt_hiddens = torch.stack(prompt_hiddens)  # [n_prompts, hidden_dim]
    
    # Similarity matrix
    prompt_hiddens_norm = F.normalize(prompt_hiddens, dim=1)
    sim_matrix = prompt_hiddens_norm @ prompt_hiddens_norm.T
    
    print("Prompt similarity matrix:")
    print(sim_matrix.numpy())
    
    # Now the key question: Can we predict the trajectory from the prompt hidden state?
    print("\n--- Prompt → Trajectory Mapping ---")
    
    # The trajectory is a function of the prompt hidden state
    # T[i] = f(prompt_hidden[i])
    
    # If f is linear: T = prompt_hidden @ W
    # Let's fit this
    
    # Flatten trajectories: [n_prompts, n_tokens * hidden_dim]
    T_flat_per_prompt = T.reshape(len(prompts), -1)
    
    # Fit: T_flat = prompt_hiddens @ W
    # W = (prompt_hiddens.T @ prompt_hiddens)^-1 @ prompt_hiddens.T @ T_flat
    
    # This is underdetermined (5 prompts, 3584 hidden dims)
    # Use pseudo-inverse
    
    W, residuals, rank, s = np.linalg.lstsq(
        prompt_hiddens.numpy(), 
        T_flat_per_prompt.numpy(), 
        rcond=None
    )
    
    print(f"Linear fit: W shape = {W.shape}")
    
    # Test: Can we reconstruct trajectories?
    T_pred = prompt_hiddens.numpy() @ W
    T_pred = torch.tensor(T_pred).reshape(len(prompts), n_tokens, -1)
    
    # Compare
    lm_head = model.lm_head
    
    print("\nReconstruction test:")
    for i, prompt in enumerate(prompts):
        pred_tokens = []
        
        for j in range(n_tokens):
            h_pred = T_pred[i, j]
            
            with torch.no_grad():
                logits = lm_head(h_pred)
                token = logits.argmax().item()
            
            pred_tokens.append(token)
        
        matches = sum(1 for a, b in zip(all_tokens[i], pred_tokens) if a == b)
        
        ref_text = tokenizer.decode(all_tokens[i])
        pred_text = tokenizer.decode(pred_tokens)
        
        print(f"  {prompt[:20]}...")
        print(f"    Ref:  {ref_text!r}")
        print(f"    Pred: {pred_text!r}")
        print(f"    Matches: {matches}/{n_tokens}")
    
    del model
    
    return T, all_tokens, prompts


def test_unraveled_autoregression():
    """
    Apply the "unraveling" insight to autoregression.
    
    In the φ-Unraveled Engine:
    - MESH = W_q.T @ W_k captures the Q-K relationship
    - This eliminates error compounding
    
    For autoregression:
    - The "self-referential structure" is the token dependency chain
    - Can we pre-compute a "MESH" that captures this?
    
    Key insight: The hidden state at position i is a function of:
    - The prompt hidden state
    - The tokens at positions 0..i-1
    
    If we could express this as a single matrix operation...
    """
    print("\n" + "=" * 70)
    print("Unraveled Autoregression")
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
    
    # Get embedding matrix
    embeddings = model.model.embed_tokens.weight.data  # [vocab, hidden]
    lm_head = model.lm_head
    
    # Reference generation
    ref_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            ref_trajectory.append(hidden.clone())
            
            logits = outputs.logits[0, -1, :]
            token = logits.argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    H = torch.stack(ref_trajectory)
    
    # The unraveling insight:
    # h[i] = f(prompt, tokens[0:i])
    # 
    # In the transformer, this is computed through attention + MLP
    # But the RESULT is a hidden state that depends on:
    # 1. The prompt embedding
    # 2. The token embeddings for positions 0..i-1
    
    # Can we express h[i] as a linear combination?
    # h[i] = W_prompt @ prompt_emb + Σ_j W_j @ token_emb[j]
    
    print("\n--- Linear Decomposition of Hidden States ---")
    
    # Get prompt embedding (sum of token embeddings)
    prompt_emb = embeddings[input_ids[0]].sum(dim=0)  # [hidden]
    
    # Get token embeddings for generated tokens
    token_embs = embeddings[ref_tokens]  # [n_tokens, hidden]
    
    # Try to express each h[i] as a linear combination
    # h[i] = α_0 * prompt_emb + Σ_j α_j * token_emb[j]
    
    print("Fitting linear decomposition:")
    
    for i in range(n_tokens):
        # Build basis: [prompt_emb, token_emb[0], ..., token_emb[i-1]]
        if i == 0:
            basis = prompt_emb.unsqueeze(0)  # [1, hidden]
        else:
            basis = torch.cat([
                prompt_emb.unsqueeze(0),
                token_embs[:i]
            ], dim=0)  # [i+1, hidden]
        
        # Solve: h[i] = basis.T @ coeffs
        # coeffs = (basis @ basis.T)^-1 @ basis @ h[i]
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(
                basis.numpy().T,  # [hidden, i+1]
                H[i].numpy(),     # [hidden]
                rcond=None
            )
            
            # Reconstruct
            h_pred = basis.T @ torch.tensor(coeffs)
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(h_pred.unsqueeze(0), H[i].unsqueeze(0)).item()
            
            # Can we get the right token?
            with torch.no_grad():
                logits = lm_head(h_pred)
                pred_token = logits.argmax().item()
            
            correct = pred_token == ref_tokens[i]
            marker = "✓" if correct else "✗"
            
            print(f"  h[{i}]: cos_sim={cos_sim:.4f}, token={marker}")
            
        except Exception as e:
            print(f"  h[{i}]: Failed - {e}")
    
    # The key insight: The hidden state is NOT a simple linear combination
    # The transformer applies nonlinear transformations (attention, MLP)
    
    # But wait - we showed that MLP is effectively linear (Doc 132)!
    # And attention is a weighted sum (linear in the values)
    
    # The nonlinearity comes from:
    # 1. Softmax in attention
    # 2. SiLU in MLP (but this is ~linear for small inputs)
    
    print("\n--- The Unraveling Strategy ---")
    
    # Like MESH = W_q.T @ W_k, we need to find the "MESH" for autoregression
    # 
    # The key observation: The hidden state trajectory is LOW-RANK
    # We showed earlier that ~3 dimensions capture most of the variance
    
    # This means: h[i] ≈ U @ z[i]
    # where U is a fixed basis and z[i] is a low-dimensional code
    
    # If we can predict z[i] from the prompt, we can predict h[i]!
    
    # Compute the low-rank representation
    H_centered = H - H.mean(dim=0)
    U_h, S_h, Vh_h = torch.linalg.svd(H_centered, full_matrices=False)
    
    # Use top-k components
    k = 5
    basis = Vh_h[:k, :]  # [k, hidden]
    codes = H_centered @ basis.T  # [n_tokens, k]
    
    print(f"Low-rank representation: {n_tokens} hidden states → {k} codes")
    print(f"Variance explained: {((S_h[:k]**2).sum() / (S_h**2).sum()).item()*100:.1f}%")
    
    # Can we predict the codes from position?
    print("\nCode values:")
    for i in range(n_tokens):
        print(f"  z[{i}] = {codes[i].tolist()}")
    
    # The codes should follow a pattern...
    # Let's fit a polynomial to each code dimension
    
    from scipy.optimize import curve_fit
    
    def polynomial(t, a, b, c, d):
        return a + b*t + c*t**2 + d*t**3
    
    t = np.arange(n_tokens)
    
    print("\nPolynomial fit to codes:")
    fitted_codes = np.zeros((n_tokens, k))
    
    for c in range(k):
        y = codes[:, c].numpy()
        
        try:
            popt, _ = curve_fit(polynomial, t, y, maxfev=5000)
            y_pred = polynomial(t, *popt)
            fitted_codes[:, c] = y_pred
            
            mse = np.mean((y - y_pred)**2)
            print(f"  Code {c}: MSE = {mse:.4f}")
            
        except Exception as e:
            print(f"  Code {c}: Fit failed - {e}")
            fitted_codes[:, c] = y
    
    # Reconstruct hidden states from fitted codes
    H_mean = H.mean(dim=0)
    H_reconstructed = H_mean + torch.tensor(fitted_codes, dtype=torch.float32) @ basis
    
    # Test token prediction
    print("\nReconstructed token prediction:")
    
    recon_tokens = []
    for i in range(n_tokens):
        with torch.no_grad():
            logits = lm_head(H_reconstructed[i])
            token = logits.argmax().item()
        recon_tokens.append(token)
    
    matches = sum(1 for a, b in zip(ref_tokens, recon_tokens) if a == b)
    recon_text = tokenizer.decode(recon_tokens)
    
    print(f"Reference:     {ref_text!r}")
    print(f"Reconstructed: {recon_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    
    del model
    
    return H, codes, basis


def test_explicit_trajectory_geometry():
    """
    The ultimate goal: Make the trajectory geometry EXPLICIT.
    
    Like HyperMapping:
    - Input → Encoder → Position → Nearest Neighbor → Output
    
    For autoregression:
    - Prompt → Encoder → Trajectory Position → Token Extraction → Sequence
    
    The key is to make the "Trajectory Position" explicit and computable
    without iteration.
    """
    print("\n" + "=" * 70)
    print("Explicit Trajectory Geometry")
    print("=" * 70)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # The HyperMapping approach:
    # 1. Build a database of (prompt, trajectory) pairs
    # 2. For a new prompt, find the nearest neighbor
    # 3. Use that trajectory as the output
    
    # This is like "retrieval-augmented generation" but for trajectories
    
    # Let's test: If we have seen similar prompts, can we predict the trajectory?
    
    # Training prompts
    train_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]
    
    # Test prompt
    test_prompt = "The capital of Japan is"
    
    n_tokens = 10
    
    # Collect training trajectories
    train_trajectories = []
    train_tokens = []
    train_hiddens = []
    
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
        
        print(f"Train: {prompt!r} → {tokenizer.decode(tokens)!r}")
    
    train_hiddens = torch.stack(train_hiddens)
    
    # Test prompt
    print(f"\nTest: {test_prompt!r}")
    
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        test_hidden = outputs.hidden_states[-1][0, -1, :]
    
    # Find nearest neighbor
    test_hidden_norm = F.normalize(test_hidden.unsqueeze(0), dim=1)
    train_hiddens_norm = F.normalize(train_hiddens, dim=1)
    
    similarities = (test_hidden_norm @ train_hiddens_norm.T).squeeze()
    
    print("Similarities to training prompts:")
    for i, (prompt, sim) in enumerate(zip(train_prompts, similarities)):
        print(f"  {prompt!r}: {sim.item():.4f}")
    
    nearest_idx = similarities.argmax().item()
    print(f"\nNearest: {train_prompts[nearest_idx]!r}")
    
    # Use nearest trajectory
    nearest_trajectory = train_trajectories[nearest_idx]
    nearest_tokens = train_tokens[nearest_idx]
    
    # But wait - we need to ADAPT the trajectory to the new prompt
    # The trajectory for "France" won't work for "Japan"
    
    # The key insight: The STRUCTURE of the trajectory is similar
    # Only the CONTENT (the country name) changes
    
    # Let's analyze the difference between trajectories
    print("\n--- Trajectory Difference Analysis ---")
    
    # Compute pairwise trajectory differences
    for i in range(len(train_prompts)):
        for j in range(i+1, len(train_prompts)):
            diff = train_trajectories[i] - train_trajectories[j]
            diff_norm = torch.norm(diff, dim=1).mean().item()
            
            print(f"  {train_prompts[i][:15]} vs {train_prompts[j][:15]}: {diff_norm:.2f}")
    
    # The differences are large - trajectories are NOT similar
    # This is because the content (country name) dominates
    
    # But the STRUCTURE might be similar...
    # Let's look at the trajectory in a relative coordinate system
    
    print("\n--- Relative Trajectory Structure ---")
    
    # Center each trajectory
    centered_trajectories = []
    for traj in train_trajectories:
        centered = traj - traj.mean(dim=0)
        centered_trajectories.append(centered)
    
    # Now compare
    for i in range(len(train_prompts)):
        for j in range(i+1, len(train_prompts)):
            # Cosine similarity between centered trajectories
            sim = F.cosine_similarity(
                centered_trajectories[i].flatten().unsqueeze(0),
                centered_trajectories[j].flatten().unsqueeze(0)
            ).item()
            
            print(f"  {train_prompts[i][:15]} vs {train_prompts[j][:15]}: {sim:.4f}")
    
    # The centered trajectories are more similar!
    # This suggests the STRUCTURE is shared, only the CENTER differs
    
    # The center is determined by the prompt (the country name)
    # The structure is the "scaffolding" pattern
    
    print("\n--- Scaffolding Transfer ---")
    
    # Hypothesis: We can transfer the scaffolding structure
    # new_trajectory = new_center + shared_structure
    
    # Compute average structure
    avg_structure = torch.stack(centered_trajectories).mean(dim=0)
    
    # For the test prompt, we need to find the center
    # The center is related to the prompt hidden state
    
    # Fit: center = W @ prompt_hidden
    centers = torch.stack([traj.mean(dim=0) for traj in train_trajectories])
    
    # Solve: centers = train_hiddens @ W
    W_center, _, _, _ = np.linalg.lstsq(
        train_hiddens.numpy(),
        centers.numpy(),
        rcond=None
    )
    
    # Predict center for test prompt
    test_center = test_hidden.numpy() @ W_center
    test_center = torch.tensor(test_center, dtype=torch.float32)
    
    # Construct trajectory
    test_trajectory_pred = test_center + avg_structure
    
    # Extract tokens
    lm_head = model.lm_head
    
    pred_tokens = []
    for i in range(n_tokens):
        with torch.no_grad():
            logits = lm_head(test_trajectory_pred[i])
            token = logits.argmax().item()
        pred_tokens.append(token)
    
    pred_text = tokenizer.decode(pred_tokens)
    
    # Reference
    ref_tokens = []
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]
            token = logits.argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    
    matches = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
    
    print(f"Reference:  {ref_text!r}")
    print(f"Predicted:  {pred_text!r}")
    print(f"Matches: {matches}/{n_tokens}")
    
    del model
    
    return avg_structure, W_center


if __name__ == "__main__":
    # 1. Analyze trajectory manifold
    T, tokens, prompts = analyze_trajectory_manifold()
    
    # 2. Test unraveled autoregression
    H, codes, basis = test_unraveled_autoregression()
    
    # 3. Test explicit trajectory geometry
    avg_structure, W_center = test_explicit_trajectory_geometry()
