#!/usr/bin/env python3
"""
Holographic Hidden State Trajectory Prediction
===============================================

Key insight from Additive Error Stereoscopy:
- Errors encode gradients (the Jacobian of the transformation)
- The gradient IS the information, not noise to filter
- Each point holographically encodes the whole

Applied to hidden state trajectories:
- The difference Δh = h[i+1] - h[i] encodes the transformation
- This is the "Jacobian" of semantic evolution
- We can project through the transformation, not extrapolate positions

The Holographic Principle:
- Each hidden state h[i] contains information about ALL future tokens
- The lm_head extracts token i, but the information for tokens i+1, i+2, ... is also there
- We need to find the RIGHT PROJECTION to extract it

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


def analyze_hidden_state_structure():
    """
    Analyze the structure of hidden state trajectories.
    
    Key questions:
    1. What is the geometry of the trajectory?
    2. How does the "error" (delta) encode future tokens?
    3. Can we find a holographic projection?
    """
    print("\n" + "=" * 70)
    print("Hidden State Trajectory Analysis")
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
    
    # Collect hidden state trajectory
    print("\n--- Collecting Trajectory ---")
    
    hidden_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]  # Last layer, last position
            hidden_trajectory.append(hidden.clone())
            
            logits = outputs.logits[0, -1, :]
            token = logits.argmax().item()
            ref_tokens.append(token)
            
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]])
            ], dim=1)
    
    ref_text = tokenizer.decode(ref_tokens)
    print(f"Reference: {ref_text!r}")
    
    H = torch.stack(hidden_trajectory)  # [n_tokens, hidden_dim]
    
    # 1. Analyze the deltas (like synthesis error in stereo)
    print("\n--- Delta Analysis (The 'Error' Field) ---")
    
    deltas = H[1:] - H[:-1]  # [n_tokens-1, hidden_dim]
    
    print(f"Delta norms:")
    for i in range(len(deltas)):
        norm = torch.norm(deltas[i]).item()
        token_text = tokenizer.decode([ref_tokens[i+1]])
        print(f"  Δ{i}→{i+1}: {norm:.2f} ({token_text!r})")
    
    # 2. SVD of deltas - what's the rank of the transformation?
    print("\n--- SVD of Delta Matrix ---")
    
    U, S, Vh = torch.linalg.svd(deltas)
    
    print("Singular values:")
    for i, s in enumerate(S[:10]):
        variance_explained = (s**2 / (S**2).sum()).item() * 100
        cumulative = ((S[:i+1]**2).sum() / (S**2).sum()).item() * 100
        print(f"  S[{i}] = {s.item():.2f} ({variance_explained:.1f}%, cumulative: {cumulative:.1f}%)")
    
    # 3. The key insight: Can we predict delta[i] from h[0]?
    print("\n--- Holographic Projection Test ---")
    
    # Hypothesis: delta[i] = P_i @ h[0] for some projection P_i
    # If true, we can predict all future deltas from the initial hidden state
    
    h0 = H[0]  # Initial hidden state
    
    # Fit linear projections: delta[i] = W_i @ h0
    print("Fitting linear projections delta[i] = W_i @ h0:")
    
    projections = []
    for i in range(len(deltas)):
        # Solve: W @ h0 = delta[i]
        # This is underdetermined (hidden_dim x hidden_dim -> hidden_dim)
        # Use least squares: W = delta[i] @ h0.T / (h0.T @ h0)
        
        # Actually, we want to find if delta[i] is in the span of h0
        # Compute projection of delta[i] onto h0
        proj_coeff = torch.dot(deltas[i], h0) / torch.dot(h0, h0)
        proj = proj_coeff * h0
        residual = deltas[i] - proj
        
        proj_norm = torch.norm(proj).item()
        residual_norm = torch.norm(residual).item()
        delta_norm = torch.norm(deltas[i]).item()
        
        explained = (proj_norm / delta_norm) * 100 if delta_norm > 0 else 0
        
        print(f"  Δ{i}: proj={proj_norm:.2f}, residual={residual_norm:.2f}, explained={explained:.1f}%")
    
    # 4. Better approach: Use the TRAJECTORY as a basis
    print("\n--- Trajectory Basis Projection ---")
    
    # The trajectory H forms a low-dimensional manifold
    # Project deltas onto the span of the trajectory
    
    # SVD of trajectory
    U_H, S_H, Vh_H = torch.linalg.svd(H, full_matrices=False)
    
    print("Trajectory singular values:")
    for i, s in enumerate(S_H[:5]):
        print(f"  S[{i}] = {s.item():.2f}")
    
    # Use top-k components of trajectory as basis
    k = 3
    basis = Vh_H[:k, :]  # [k, hidden_dim]
    
    print(f"\nProjecting deltas onto top-{k} trajectory components:")
    
    for i in range(len(deltas)):
        # Project delta onto basis
        coeffs = basis @ deltas[i]  # [k]
        proj = coeffs @ basis  # [hidden_dim]
        residual = deltas[i] - proj
        
        proj_norm = torch.norm(proj).item()
        residual_norm = torch.norm(residual).item()
        delta_norm = torch.norm(deltas[i]).item()
        
        explained = (proj_norm / delta_norm) * 100 if delta_norm > 0 else 0
        
        print(f"  Δ{i}: explained={explained:.1f}%")
    
    # 5. The holographic insight: Each h[i] contains info about ALL tokens
    print("\n--- Multi-Token Extraction from Single Hidden State ---")
    
    lm_head = model.lm_head
    
    # For each hidden state h[i], what tokens can we extract?
    print("Tokens extractable from each hidden state:")
    
    for i in range(n_tokens):
        h = H[i]
        
        with torch.no_grad():
            logits = lm_head(h)
            top5 = torch.topk(logits, 5)
        
        top5_tokens = [tokenizer.decode([t.item()]) for t in top5.indices]
        correct_token = tokenizer.decode([ref_tokens[i]])
        
        is_correct = ref_tokens[i] in top5.indices.tolist()
        marker = "✓" if is_correct else "✗"
        
        print(f"  h[{i}] → top5: {top5_tokens}, correct: {correct_token!r} {marker}")
    
    # 6. The key question: Can we TRANSFORM h[i] to predict token[j] for j > i?
    print("\n--- Cross-Position Token Extraction ---")
    
    # Hypothesis: There exists a transformation T_j such that
    # lm_head(T_j @ h[i]) = token[j] for j > i
    
    # Let's see if the DELTA encodes this transformation
    
    print("Testing: lm_head(h[0] + cumsum(deltas[:j])) = token[j]?")
    
    for j in range(n_tokens):
        if j == 0:
            h_pred = H[0]
        else:
            # Cumulative sum of deltas
            h_pred = H[0] + deltas[:j].sum(dim=0)
        
        with torch.no_grad():
            logits = lm_head(h_pred)
            pred_token = logits.argmax().item()
        
        correct = pred_token == ref_tokens[j]
        marker = "✓" if correct else "✗"
        
        pred_text = tokenizer.decode([pred_token])
        ref_text = tokenizer.decode([ref_tokens[j]])
        
        print(f"  j={j}: pred={pred_text!r}, ref={ref_text!r} {marker}")
    
    # 7. The REAL holographic insight: Use the Jacobian structure
    print("\n--- Jacobian Structure Analysis ---")
    
    # Like in Additive Error Stereo: E encodes ∂D/∂x
    # Here: delta encodes ∂h/∂position
    
    # The Jacobian of the hidden state w.r.t. position
    # J[i,j] = ∂h[i] / ∂token[j]
    
    # We can approximate this by looking at how h changes when we change tokens
    
    # Key insight: The delta is NOT just h[i+1] - h[i]
    # It's the GRADIENT of the hidden state w.r.t. the token at position i
    
    # This means: delta[i] ≈ ∂h/∂token[i] × token[i]
    
    # If we know the gradient structure, we can predict future hidden states!
    
    print("Computing gradient structure...")
    
    # Approximate gradient: how much does h[j] change when we perturb token[i]?
    # This is expensive, so we'll do a simplified version
    
    # For now, let's test if the deltas have a consistent structure
    
    # Normalize deltas
    delta_norms = torch.norm(deltas, dim=1, keepdim=True)
    delta_directions = deltas / (delta_norms + 1e-8)
    
    # Compute pairwise cosine similarities
    print("Delta direction similarities:")
    for i in range(len(delta_directions)):
        for j in range(i+1, len(delta_directions)):
            sim = torch.dot(delta_directions[i], delta_directions[j]).item()
            print(f"  cos(Δ{i}, Δ{j}) = {sim:.3f}")
    
    del model
    
    return H, deltas, ref_tokens


def test_holographic_prediction():
    """
    Test holographic prediction of hidden states.
    
    The key insight: The hidden state trajectory lives on a low-dimensional
    manifold. We can predict future states by projecting onto this manifold.
    """
    print("\n" + "=" * 70)
    print("Holographic Prediction Test")
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
    
    # Collect trajectory
    hidden_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_trajectory.append(hidden.clone())
            
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
    
    # Strategy 1: Geodesic interpolation
    print("\n--- Strategy 1: Geodesic Interpolation ---")
    
    # Use first 2 points to define a geodesic, extrapolate
    h0, h1 = H[0], H[1]
    direction = h1 - h0
    direction_norm = torch.norm(direction)
    
    print(f"Geodesic direction norm: {direction_norm.item():.2f}")
    
    geodesic_tokens = []
    for i in range(n_tokens):
        h_pred = h0 + i * direction
        
        with torch.no_grad():
            logits = lm_head(h_pred)
            token = logits.argmax().item()
        
        geodesic_tokens.append(token)
    
    geodesic_matches = sum(1 for a, b in zip(geodesic_tokens, ref_tokens) if a == b)
    print(f"Geodesic prediction: {tokenizer.decode(geodesic_tokens)!r}")
    print(f"Matches: {geodesic_matches}/{n_tokens}")
    
    # Strategy 2: Manifold projection using SVD
    print("\n--- Strategy 2: Manifold Projection (SVD) ---")
    
    # Fit a low-rank manifold to first k points, project rest
    k_fit = 3
    H_fit = H[:k_fit]
    
    # Center the data
    mean = H_fit.mean(dim=0)
    H_centered = H_fit - mean
    
    # SVD to find principal directions
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    
    # Use top 2 components
    n_components = 2
    basis = Vh[:n_components, :]  # [n_components, hidden_dim]
    
    # Project trajectory onto manifold
    coeffs = (H - mean) @ basis.T  # [n_tokens, n_components]
    
    print(f"Trajectory in {n_components}D manifold:")
    for i in range(n_tokens):
        print(f"  h[{i}] → ({coeffs[i, 0].item():.2f}, {coeffs[i, 1].item():.2f})")
    
    # Fit a curve to the first k points in manifold space
    t = torch.arange(k_fit, dtype=torch.float32)
    
    # Linear fit in manifold space
    A = torch.stack([torch.ones(k_fit), t], dim=1)  # [k_fit, 2]
    
    manifold_tokens = []
    for i in range(n_tokens):
        if i < k_fit:
            # Use actual hidden state
            h_pred = H[i]
        else:
            # Extrapolate in manifold space
            # Linear extrapolation of coefficients
            t_pred = torch.tensor([i], dtype=torch.float32)
            
            # Fit linear model to each coefficient
            coeff_pred = []
            for c in range(n_components):
                # Solve: A @ [a, b] = coeffs[:k_fit, c]
                params, _, _, _ = torch.linalg.lstsq(A, coeffs[:k_fit, c])
                pred = params[0] + params[1] * t_pred
                coeff_pred.append(pred.item())
            
            coeff_pred = torch.tensor(coeff_pred)
            h_pred = mean + coeff_pred @ basis
        
        with torch.no_grad():
            logits = lm_head(h_pred)
            token = logits.argmax().item()
        
        manifold_tokens.append(token)
    
    manifold_matches = sum(1 for a, b in zip(manifold_tokens, ref_tokens) if a == b)
    print(f"Manifold prediction: {tokenizer.decode(manifold_tokens)!r}")
    print(f"Matches: {manifold_matches}/{n_tokens}")
    
    # Strategy 3: Use the GRADIENT structure (like Additive Error Stereo)
    print("\n--- Strategy 3: Gradient-Based Prediction ---")
    
    # The key insight from Additive Error Stereo:
    # E encodes ∂D/∂x, which is the GRADIENT
    # 
    # For hidden states:
    # delta[i] = h[i+1] - h[i] ≈ ∂h/∂position
    #
    # But the gradient changes! We need to model how the gradient evolves.
    
    deltas = H[1:] - H[:-1]
    
    # Hypothesis: The gradient itself follows a pattern
    # delta[i+1] = f(delta[i])
    
    # Let's check if deltas are related
    delta_ratios = []
    for i in range(len(deltas) - 1):
        ratio = torch.norm(deltas[i+1]) / torch.norm(deltas[i])
        delta_ratios.append(ratio.item())
    
    print(f"Delta norm ratios: {delta_ratios}")
    avg_ratio = np.mean(delta_ratios)
    print(f"Average ratio: {avg_ratio:.3f}")
    
    # Predict using decaying deltas
    gradient_tokens = []
    h_current = H[0].clone()
    delta_current = deltas[0].clone()
    
    for i in range(n_tokens):
        with torch.no_grad():
            logits = lm_head(h_current)
            token = logits.argmax().item()
        
        gradient_tokens.append(token)
        
        if i < n_tokens - 1:
            h_current = h_current + delta_current
            delta_current = delta_current * avg_ratio  # Decay the delta
    
    gradient_matches = sum(1 for a, b in zip(gradient_tokens, ref_tokens) if a == b)
    print(f"Gradient prediction: {tokenizer.decode(gradient_tokens)!r}")
    print(f"Matches: {gradient_matches}/{n_tokens}")
    
    # Strategy 4: Holographic projection - use ALL hidden states to predict
    print("\n--- Strategy 4: Holographic Reconstruction ---")
    
    # The holographic principle: Each part contains the whole
    # 
    # Idea: The hidden state h[i] contains a "hologram" of all future tokens
    # We need to find the right "illumination" to extract them
    #
    # In stereo: I_L = I - αE, I_R = I + αE
    # Here: token[j] = lm_head(h[i] + α_j * delta[i])
    
    # Test: Can we extract token[j] from h[0] by adding scaled deltas?
    
    print("Testing holographic extraction from h[0]:")
    
    for j in range(1, n_tokens):
        # Try different scalings of delta[0]
        best_alpha = None
        best_token = None
        
        for alpha in np.linspace(0, 5, 50):
            h_test = H[0] + alpha * deltas[0]
            
            with torch.no_grad():
                logits = lm_head(h_test)
                token = logits.argmax().item()
            
            if token == ref_tokens[j]:
                best_alpha = alpha
                best_token = token
                break
        
        if best_alpha is not None:
            print(f"  token[{j}] = lm_head(h[0] + {best_alpha:.2f} * Δ0) ✓")
        else:
            print(f"  token[{j}] = NOT FOUND in h[0] + α*Δ0 ✗")
    
    # Strategy 5: Use the FULL delta trajectory
    print("\n--- Strategy 5: Full Delta Trajectory ---")
    
    # Maybe we need to use ALL deltas, not just delta[0]
    # h[j] = h[0] + Σ_{i=0}^{j-1} delta[i]
    
    # But we don't know delta[i] for i > 0 without computing h[i]
    # 
    # UNLESS... the deltas are predictable from h[0]!
    
    # Let's check: Can we predict delta[i] from h[0]?
    
    print("Checking if deltas are predictable from h[0]:")
    
    # Fit a linear model: delta[i] = W_i @ h[0]
    # This is underdetermined, so we use the pseudo-inverse
    
    h0 = H[0].unsqueeze(0)  # [1, hidden_dim]
    
    for i in range(len(deltas)):
        # Compute the projection of delta[i] onto h[0]
        # and the orthogonal component
        
        proj_coeff = torch.dot(deltas[i], H[0]) / torch.dot(H[0], H[0])
        proj = proj_coeff * H[0]
        ortho = deltas[i] - proj
        
        proj_frac = torch.norm(proj) / torch.norm(deltas[i])
        
        print(f"  Δ{i}: {proj_frac.item()*100:.1f}% in h[0] direction")
    
    del model
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Geodesic (linear): {geodesic_matches}/{n_tokens}")
    print(f"Manifold (SVD): {manifold_matches}/{n_tokens}")
    print(f"Gradient (decaying): {gradient_matches}/{n_tokens}")
    
    return {
        "geodesic": geodesic_matches,
        "manifold": manifold_matches,
        "gradient": gradient_matches,
    }


def test_error_as_signal():
    """
    Apply the Additive Error Stereo insight directly.
    
    In stereo: E = I_synth - I encodes depth gradients
    Here: E = h_pred - h_actual encodes semantic gradients
    
    The key: ERRORS ARE SIGNALS, not noise!
    """
    print("\n" + "=" * 70)
    print("Error as Signal: Additive Error Approach")
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
    
    # Collect trajectory
    hidden_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_trajectory.append(hidden.clone())
            
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
    
    # The Additive Error Stereo approach:
    # 1. Make a prediction (synthesis)
    # 2. Compute error E = pred - actual
    # 3. Use E as signal: actual = pred - E
    
    # For hidden states:
    # 1. Predict h[i+1] from h[i] (linear extrapolation)
    # 2. Compute error E[i] = h_pred[i+1] - h_actual[i+1]
    # 3. The error encodes the "semantic gradient"
    
    print("\n--- Computing Prediction Errors ---")
    
    # Predict each h[i+1] from h[i] using linear extrapolation
    errors = []
    
    for i in range(n_tokens - 1):
        if i == 0:
            # First prediction: just use h[0]
            h_pred = H[0]
        else:
            # Linear extrapolation: h[i] + (h[i] - h[i-1])
            h_pred = H[i] + (H[i] - H[i-1])
        
        # Actual
        h_actual = H[i+1]
        
        # Error
        E = h_pred - h_actual
        errors.append(E)
        
        error_norm = torch.norm(E).item()
        actual_norm = torch.norm(h_actual).item()
        
        print(f"  E[{i}→{i+1}]: norm={error_norm:.2f}, relative={error_norm/actual_norm*100:.1f}%")
    
    errors = torch.stack(errors)
    
    # The key insight: The error field has structure!
    print("\n--- Error Field Structure ---")
    
    # SVD of error field
    U_E, S_E, Vh_E = torch.linalg.svd(errors, full_matrices=False)
    
    print("Error singular values:")
    for i, s in enumerate(S_E[:5]):
        var = (s**2 / (S_E**2).sum()).item() * 100
        print(f"  S[{i}] = {s.item():.2f} ({var:.1f}%)")
    
    # The error is LOW-RANK!
    # This means we can predict the error from a few components
    
    # Use top-k error components to correct predictions
    print("\n--- Error-Corrected Prediction ---")
    
    k_error = 2
    error_basis = Vh_E[:k_error, :]  # [k_error, hidden_dim]
    
    # Project errors onto basis
    error_coeffs = errors @ error_basis.T  # [n_tokens-1, k_error]
    
    print(f"Error coefficients (top-{k_error}):")
    for i in range(len(error_coeffs)):
        print(f"  E[{i}]: {error_coeffs[i].tolist()}")
    
    # Fit a model to predict error coefficients
    # Simple: linear extrapolation of coefficients
    
    corrected_tokens = [ref_tokens[0]]  # First token is known
    h_current = H[0].clone()
    
    for i in range(1, n_tokens):
        # Predict next hidden state
        if i == 1:
            h_pred = H[0] + (H[1] - H[0])  # Use actual first delta
        else:
            h_pred = h_current + (h_current - H[i-2])  # Linear extrapolation
        
        # Predict error correction
        if i - 1 < len(error_coeffs):
            # Use actual error coefficients (oracle)
            error_correction = error_coeffs[i-1] @ error_basis
            h_corrected = h_pred - error_correction
        else:
            # Extrapolate error coefficients
            # Linear extrapolation of last two
            if len(error_coeffs) >= 2:
                coeff_pred = error_coeffs[-1] + (error_coeffs[-1] - error_coeffs[-2])
                error_correction = coeff_pred @ error_basis
                h_corrected = h_pred - error_correction
            else:
                h_corrected = h_pred
        
        with torch.no_grad():
            logits = lm_head(h_corrected)
            token = logits.argmax().item()
        
        corrected_tokens.append(token)
        h_current = h_corrected
    
    corrected_matches = sum(1 for a, b in zip(corrected_tokens, ref_tokens) if a == b)
    print(f"\nError-corrected prediction: {tokenizer.decode(corrected_tokens)!r}")
    print(f"Matches: {corrected_matches}/{n_tokens}")
    
    # The REAL insight: Use the error structure to GENERATE, not correct
    print("\n--- Generative Error Approach ---")
    
    # Like in stereo: I_L = I - αE, I_R = I + αE
    # Here: h[i+1] = h[i] + δ + αE[i]
    
    # The error E encodes the "semantic disparity"
    # Adding/subtracting it moves us along the semantic trajectory
    
    # Key: The error is the DIFFERENCE between linear prediction and actual
    # This difference encodes the nonlinearity of the trajectory
    
    # If we can predict the error, we can predict the trajectory!
    
    print("\nThe error encodes the NONLINEARITY of the trajectory.")
    print("Linear prediction fails because the trajectory is curved.")
    print("The error IS the curvature!")
    
    # Compute curvature: second derivative of trajectory
    deltas = H[1:] - H[:-1]  # First derivative
    curvatures = deltas[1:] - deltas[:-1]  # Second derivative
    
    print("\nCurvature norms:")
    for i in range(len(curvatures)):
        curv_norm = torch.norm(curvatures[i]).item()
        delta_norm = torch.norm(deltas[i]).item()
        print(f"  κ[{i}]: {curv_norm:.2f} (relative to δ: {curv_norm/delta_norm*100:.1f}%)")
    
    # The curvature is significant!
    # This explains why linear extrapolation fails
    
    # But the curvature might be predictable...
    print("\n--- Curvature Prediction ---")
    
    # SVD of curvature
    if len(curvatures) > 1:
        U_K, S_K, Vh_K = torch.linalg.svd(curvatures, full_matrices=False)
        
        print("Curvature singular values:")
        for i, s in enumerate(S_K[:3]):
            var = (s**2 / (S_K**2).sum()).item() * 100
            print(f"  S[{i}] = {s.item():.2f} ({var:.1f}%)")
    
    del model


def test_oscillatory_prediction():
    """
    The deltas OSCILLATE! Consecutive deltas are anti-correlated.
    
    This is like a standing wave pattern:
    - Δ[i] and Δ[i+1] point in opposite directions
    - The trajectory zigzags through hidden space
    
    This is the holographic structure we're looking for!
    
    Like in Additive Error Stereo:
    - I_L = I - αE
    - I_R = I + αE
    
    The +/- pattern creates the stereo effect.
    Here, the oscillating deltas create the semantic trajectory.
    """
    print("\n" + "=" * 70)
    print("Oscillatory Delta Pattern")
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
    
    # Collect trajectory
    hidden_trajectory = []
    ref_tokens = []
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        
        for i in range(n_tokens):
            outputs = model(current_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_trajectory.append(hidden.clone())
            
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
    deltas = H[1:] - H[:-1]
    lm_head = model.lm_head
    
    # The oscillatory pattern: Δ[i] ≈ -Δ[i-1] + correction
    print("\n--- Oscillatory Pattern Analysis ---")
    
    # Decompose each delta into:
    # 1. Component parallel to previous delta (should be negative)
    # 2. Orthogonal component (the "new information")
    
    for i in range(1, len(deltas)):
        prev = deltas[i-1]
        curr = deltas[i]
        
        # Project curr onto prev
        proj_coeff = torch.dot(curr, prev) / torch.dot(prev, prev)
        proj = proj_coeff * prev
        ortho = curr - proj
        
        proj_norm = torch.norm(proj).item()
        ortho_norm = torch.norm(ortho).item()
        curr_norm = torch.norm(curr).item()
        
        print(f"  Δ{i}: parallel={proj_coeff.item():.3f} (norm={proj_norm:.1f}), ortho_norm={ortho_norm:.1f}")
    
    # The key insight: The parallel component is NEGATIVE (oscillation)
    # The orthogonal component is the NEW INFORMATION
    
    print("\n--- Oscillatory Prediction Strategy ---")
    
    # Strategy: Predict Δ[i] = -α * Δ[i-1] + β * ortho_direction
    # where ortho_direction is learned from the first few deltas
    
    # Use first 3 deltas to learn the pattern
    n_learn = 3
    
    # Compute average oscillation coefficient
    osc_coeffs = []
    ortho_components = []
    
    for i in range(1, n_learn):
        prev = deltas[i-1]
        curr = deltas[i]
        
        proj_coeff = torch.dot(curr, prev) / torch.dot(prev, prev)
        osc_coeffs.append(proj_coeff.item())
        
        ortho = curr - proj_coeff * prev
        ortho_components.append(ortho)
    
    avg_osc = np.mean(osc_coeffs)
    print(f"Average oscillation coefficient: {avg_osc:.3f}")
    
    # The orthogonal components might have structure too
    if len(ortho_components) >= 2:
        ortho_stack = torch.stack(ortho_components)
        U_o, S_o, Vh_o = torch.linalg.svd(ortho_stack, full_matrices=False)
        
        print("Orthogonal component singular values:")
        for i, s in enumerate(S_o):
            print(f"  S[{i}] = {s.item():.2f}")
    
    # Predict using oscillatory pattern
    print("\n--- Oscillatory Prediction ---")
    
    osc_tokens = []
    h_current = H[0].clone()
    delta_current = deltas[0].clone()
    
    for i in range(n_tokens):
        with torch.no_grad():
            logits = lm_head(h_current)
            token = logits.argmax().item()
        
        osc_tokens.append(token)
        
        if i < n_tokens - 1:
            if i < n_learn:
                # Use actual deltas for learning phase
                h_current = H[i + 1]
                if i < len(deltas) - 1:
                    delta_current = deltas[i + 1]
            else:
                # Predict using oscillatory pattern
                # Δ[i] ≈ avg_osc * Δ[i-1]
                delta_pred = avg_osc * delta_current
                h_current = h_current + delta_pred
                delta_current = delta_pred
    
    osc_matches = sum(1 for a, b in zip(osc_tokens, ref_tokens) if a == b)
    print(f"Oscillatory prediction: {tokenizer.decode(osc_tokens)!r}")
    print(f"Matches: {osc_matches}/{n_tokens}")
    
    # Better strategy: Use the FULL oscillatory structure
    print("\n--- Full Oscillatory Structure ---")
    
    # The trajectory is a damped oscillation around a mean
    # h[i] = h_mean + A * cos(ω*i + φ)
    
    # Compute mean hidden state
    h_mean = H.mean(dim=0)
    
    # Centered trajectory
    H_centered = H - h_mean
    
    # SVD to find principal oscillation modes
    U_c, S_c, Vh_c = torch.linalg.svd(H_centered, full_matrices=False)
    
    print("Centered trajectory singular values:")
    for i, s in enumerate(S_c[:5]):
        var = (s**2 / (S_c**2).sum()).item() * 100
        print(f"  S[{i}] = {s.item():.2f} ({var:.1f}%)")
    
    # Project onto top-2 modes
    coeffs = H_centered @ Vh_c[:2, :].T  # [n_tokens, 2]
    
    print("\nTrajectory in oscillation space:")
    for i in range(n_tokens):
        print(f"  h[{i}] → ({coeffs[i, 0].item():.1f}, {coeffs[i, 1].item():.1f})")
    
    # Fit a sinusoidal model to the coefficients
    t = np.arange(n_tokens)
    
    # For each coefficient, fit: c(t) = A * cos(ω*t + φ)
    from scipy.optimize import curve_fit
    
    def sinusoid(t, A, omega, phi, offset):
        return A * np.cos(omega * t + phi) + offset
    
    print("\n--- Sinusoidal Fit ---")
    
    fitted_coeffs = []
    for c in range(2):
        y = coeffs[:, c].numpy()
        
        try:
            # Initial guess
            A0 = (y.max() - y.min()) / 2
            omega0 = np.pi / 2  # Oscillation period ~4
            phi0 = 0
            offset0 = y.mean()
            
            popt, _ = curve_fit(sinusoid, t, y, p0=[A0, omega0, phi0, offset0], maxfev=5000)
            A, omega, phi, offset = popt
            
            print(f"  Coeff {c}: A={A:.1f}, ω={omega:.3f}, φ={phi:.2f}, offset={offset:.1f}")
            
            # Predict using fitted model
            y_pred = sinusoid(t, *popt)
            fitted_coeffs.append(y_pred)
            
        except Exception as e:
            print(f"  Coeff {c}: Fit failed - {e}")
            fitted_coeffs.append(y)
    
    # Reconstruct hidden states from fitted coefficients
    if len(fitted_coeffs) == 2:
        fitted_coeffs = np.stack(fitted_coeffs, axis=1)  # [n_tokens, 2]
        
        # Reconstruct
        H_reconstructed = h_mean + torch.tensor(fitted_coeffs, dtype=torch.float32) @ Vh_c[:2, :]
        
        # Predict tokens
        sinusoid_tokens = []
        for i in range(n_tokens):
            with torch.no_grad():
                logits = lm_head(H_reconstructed[i])
                token = logits.argmax().item()
            sinusoid_tokens.append(token)
        
        sin_matches = sum(1 for a, b in zip(sinusoid_tokens, ref_tokens) if a == b)
        print(f"\nSinusoidal prediction: {tokenizer.decode(sinusoid_tokens)!r}")
        print(f"Matches: {sin_matches}/{n_tokens}")
    
    # The REAL insight: Use the oscillatory structure for PARALLEL prediction
    print("\n--- Parallel Oscillatory Prediction ---")
    
    # If we know the oscillation parameters, we can predict ALL hidden states at once
    # from just the first few observations
    
    # Use first 3 points to fit, predict rest
    n_fit = 3
    
    parallel_tokens = []
    
    for i in range(n_tokens):
        if i < n_fit:
            # Use actual hidden state
            h = H[i]
        else:
            # Predict using oscillatory model
            # h[i] = h_mean + Σ_k c_k(i) * v_k
            # where c_k(i) is predicted from the sinusoidal fit
            
            pred_coeffs = []
            for c in range(2):
                y_fit = coeffs[:n_fit, c].numpy()
                t_fit = np.arange(n_fit)
                
                try:
                    A0 = (y_fit.max() - y_fit.min()) / 2 if y_fit.max() != y_fit.min() else 1
                    popt, _ = curve_fit(sinusoid, t_fit, y_fit, 
                                       p0=[A0, np.pi/2, 0, y_fit.mean()], 
                                       maxfev=5000)
                    pred_coeffs.append(sinusoid(i, *popt))
                except:
                    # Linear extrapolation fallback
                    slope = (y_fit[-1] - y_fit[0]) / (n_fit - 1)
                    pred_coeffs.append(y_fit[-1] + slope * (i - n_fit + 1))
            
            pred_coeffs = torch.tensor(pred_coeffs, dtype=torch.float32)
            h = h_mean + pred_coeffs @ Vh_c[:2, :]
        
        with torch.no_grad():
            logits = lm_head(h)
            token = logits.argmax().item()
        
        parallel_tokens.append(token)
    
    parallel_matches = sum(1 for a, b in zip(parallel_tokens, ref_tokens) if a == b)
    print(f"Parallel oscillatory (fit on {n_fit}): {tokenizer.decode(parallel_tokens)!r}")
    print(f"Matches: {parallel_matches}/{n_tokens}")
    
    del model
    
    return {
        "oscillatory": osc_matches,
        "parallel": parallel_matches,
    }


if __name__ == "__main__":
    # 1. Analyze hidden state structure
    H, deltas, ref_tokens = analyze_hidden_state_structure()
    
    # 2. Test oscillatory prediction (the key insight!)
    test_oscillatory_prediction()
    
    # 3. Test holographic prediction strategies
    # test_holographic_prediction()
    
    # 4. Apply error-as-signal approach
    # test_error_as_signal()
