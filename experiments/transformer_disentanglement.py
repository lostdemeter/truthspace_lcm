#!/usr/bin/env python3
"""
Transformer Disentanglement: Applying GOP/MGOP/EDP/PEP Protocols
=================================================================

Goal: Understand the "massive transformation" the transformer applies.

Key insight from Music Box Principle (Doc 112):
- Separate DRUM (structure) from COMB (decoder)
- The 45% error is SIGNAL, not noise
- Look for the "wall" where we need to disentangle

Protocol Application:
1. GOP: Fractal Peel - decompose transformation into layers
2. MGOP: Holographic Scan - find projections that preserve structure
3. EDP: φ-Pattern Search - look for φ-structure in the transformation
4. PEP: Probe Extraction - when we hit a wall, switch to probing

The Process:
1. Collect (input_embedding, output_hidden_state) pairs
2. Compute the transformation matrix T where h_out ≈ T @ h_in
3. Apply SVD to T - look for rank, structure, φ-patterns
4. Identify what's captured (55%) vs what's not (45%)
5. Apply Music Box Principle to disentangle the remainder

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949
PHI_INV = 1 / PHI


def collect_transformation_data(model, tokenizer, n_samples=50):
    """
    Collect (input_features, output_hidden_state, true_token) triples.
    """
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Diverse prompts for good coverage
    prompts = [
        # Capitals
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Russia is",
        "The capital of Brazil is",
        "The capital of India is",
        "The capital of Australia is",
        # Programming
        "Python is a programming language that",
        "Java is a programming language that",
        "JavaScript is a programming language that",
        "C++ is a programming language that",
        # Common phrases
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "Once upon a time there was a",
        "In the beginning there was",
        # Facts
        "The largest planet is",
        "The smallest country is",
        "The tallest mountain is",
        "The deepest ocean is the",
        "The longest river is the",
        # General
        "Water is essential for",
        "The sun rises in the",
        "Music is a form of",
        "The meaning of life is",
        "Birds can fly because they have",
        "Fish live in water because",
        "The color of the sky is",
        # More
        "Mathematics is the language of",
        "Science is the study of",
        "Art is an expression of",
        "Love is a feeling of",
        "Time is a measure of",
        "Space is the absence of",
        "Light travels at the speed of",
        "Sound travels through",
        "Heat is a form of",
        "Energy cannot be created or",
        # Completions
        "The Earth revolves around the",
        "The Moon orbits the",
        "Stars are made of",
        "Clouds are made of",
        "Rain falls from the",
        "Snow is frozen",
        "Ice is solid",
        "Steam is gaseous",
        "Fire produces heat and",
        "Wind is moving",
    ]
    
    X = []  # Input features
    Y = []  # Output hidden states
    tokens = []  # True next tokens
    prompt_list = []
    
    for prompt in prompts[:n_samples]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Input features: various combinations
        token_embeds = embed[input_ids[0]]
        seq_len = len(token_embeds)
        
        # Features
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        X.append(x)
        Y.append(h_final)
        tokens.append(true_token)
        prompt_list.append(prompt)
    
    return torch.stack(X), torch.stack(Y), tokens, prompt_list


def gop_phase1_fractal_peel(X, Y, tokens, tokenizer, lm_head):
    """
    GOP Phase 1: Fractal Peel
    
    Decompose the transformation into components and measure
    how much each component captures.
    """
    print("\n" + "=" * 70)
    print("GOP Phase 1: Fractal Peel")
    print("=" * 70)
    
    n_samples = len(X)
    hidden_dim = Y.shape[1]
    input_dim = X.shape[1]
    
    print(f"Samples: {n_samples}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {hidden_dim}")
    
    # Learn linear transformation: Y = X @ W + b
    X_with_bias = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
    
    # Ridge regression with varying regularization
    print("\n--- Regularization Sweep ---")
    
    best_acc = 0
    best_lambda = 0
    best_W = None
    
    for lambda_reg in [0.001, 0.01, 0.1, 1.0, 10.0]:
        XtX = X_with_bias.T @ X_with_bias
        XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0])
        W = torch.linalg.solve(XtX_reg, X_with_bias.T @ Y)
        
        Y_pred = X_with_bias @ W
        
        # Measure accuracy
        correct = 0
        for i in range(n_samples):
            pred = (Y_pred[i] @ lm_head.T).argmax().item()
            if pred == tokens[i]:
                correct += 1
        
        acc = correct / n_samples * 100
        print(f"  λ={lambda_reg}: {correct}/{n_samples} = {acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_lambda = lambda_reg
            best_W = W
    
    print(f"\nBest: λ={best_lambda}, accuracy={best_acc:.1f}%")
    
    # Analyze the learned transformation
    W_main = best_W[:-1, :]  # Exclude bias
    b = best_W[-1, :]
    
    print(f"\n--- Transformation Analysis ---")
    print(f"W shape: {W_main.shape}")
    print(f"W norm: {torch.norm(W_main).item():.2f}")
    print(f"b norm: {torch.norm(b).item():.2f}")
    
    # SVD of W
    U, S, Vt = torch.linalg.svd(W_main, full_matrices=False)
    
    print(f"\n--- SVD of Transformation Matrix ---")
    print(f"Rank (S > 1e-6): {(S > 1e-6).sum().item()}")
    
    # Cumulative variance
    total_var = (S**2).sum()
    cumvar = torch.cumsum(S**2, dim=0) / total_var * 100
    
    print(f"\nSingular values and cumulative variance:")
    for i in range(min(20, len(S))):
        print(f"  S[{i}] = {S[i].item():.4f} ({cumvar[i].item():.1f}%)")
    
    # How many components for 90%, 95%, 99%?
    for threshold in [90, 95, 99]:
        k = (cumvar >= threshold).nonzero()[0].item() + 1 if (cumvar >= threshold).any() else len(S)
        print(f"  Components for {threshold}%: {k}")
    
    return best_W, S, U, Vt, best_acc


def mgop_phase2_holographic_scan(X, Y, tokens, lm_head, W, S, U, Vt):
    """
    MGOP Phase 2: Holographic Scan
    
    Find projections that preserve prediction accuracy.
    Look for the "holographic bound" - minimum dimensions needed.
    """
    print("\n" + "=" * 70)
    print("MGOP Phase 2: Holographic Scan")
    print("=" * 70)
    
    n_samples = len(X)
    X_with_bias = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
    Y_pred_full = X_with_bias @ W
    
    # Test accuracy with different numbers of singular components
    print("\n--- Rank Truncation Analysis ---")
    
    W_main = W[:-1, :]
    b = W[-1, :]
    
    for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, min(W_main.shape)]:
        if k > min(W_main.shape):
            continue
        
        # Truncated SVD reconstruction
        W_trunc = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        W_trunc_with_bias = torch.cat([W_trunc, b.unsqueeze(0)], dim=0)
        
        Y_pred_trunc = X_with_bias @ W_trunc_with_bias
        
        # Accuracy
        correct = 0
        for i in range(n_samples):
            pred = (Y_pred_trunc[i] @ lm_head.T).argmax().item()
            if pred == tokens[i]:
                correct += 1
        
        acc = correct / n_samples * 100
        
        # Reconstruction error
        recon_err = torch.norm(Y_pred_trunc - Y_pred_full) / torch.norm(Y_pred_full) * 100
        
        print(f"  k={k:4d}: acc={acc:.1f}%, recon_err={recon_err:.1f}%")
    
    # Find the "holographic bound" - minimum k for max accuracy
    print("\n--- Finding Holographic Bound ---")
    
    max_acc = 0
    for k in range(1, min(W_main.shape) + 1):
        W_trunc = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        W_trunc_with_bias = torch.cat([W_trunc, b.unsqueeze(0)], dim=0)
        Y_pred_trunc = X_with_bias @ W_trunc_with_bias
        
        correct = sum(1 for i in range(n_samples) 
                     if (Y_pred_trunc[i] @ lm_head.T).argmax().item() == tokens[i])
        acc = correct / n_samples * 100
        
        if acc > max_acc:
            max_acc = acc
            print(f"  k={k}: NEW MAX acc={acc:.1f}%")
        
        if acc >= 55:  # Stop when we reach our known max
            print(f"\n  HOLOGRAPHIC BOUND: k={k} (acc={acc:.1f}%)")
            break
    
    return k


def edp_phase4_phi_patterns(S, W):
    """
    EDP Phase 4: φ-Pattern Search
    
    Look for φ-structure in the singular values and transformation.
    """
    print("\n" + "=" * 70)
    print("EDP Phase 4: φ-Pattern Search")
    print("=" * 70)
    
    # Check if singular values follow φ-Zipf
    print("\n--- Singular Value φ-Analysis ---")
    
    S_np = S.numpy()
    
    # Fit Zipf: S[i] = S[0] / i^α
    # log(S[i]) = log(S[0]) - α * log(i)
    valid_idx = S_np > 1e-6
    S_valid = S_np[valid_idx]
    i_valid = np.arange(1, len(S_valid) + 1)
    
    log_S = np.log(S_valid)
    log_i = np.log(i_valid)
    
    # Linear regression
    A = np.vstack([np.ones_like(log_i), log_i]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, log_S, rcond=None)
    
    alpha = -coeffs[1]
    
    print(f"Zipf exponent α = {alpha:.4f}")
    print(f"Target (1/φ) = {PHI_INV:.4f}")
    print(f"Deviation = {abs(alpha - PHI_INV):.4f}")
    
    # Check ratios between consecutive singular values
    print("\n--- Consecutive Ratios ---")
    
    ratios = S_valid[:-1] / S_valid[1:]
    
    print(f"Mean ratio: {ratios.mean():.4f}")
    print(f"φ = {PHI:.4f}")
    print(f"Deviation from φ: {abs(ratios.mean() - PHI):.4f}")
    
    # Look for φ-patterns in specific ratios
    print("\n--- φ-Pattern Search in Ratios ---")
    
    phi_matches = []
    for i in range(len(ratios)):
        ratio = ratios[i]
        # Check against φ^k for k in [-3, 3]
        for k in range(-3, 4):
            target = PHI ** k
            if abs(ratio - target) < 0.1:
                phi_matches.append((i, k, ratio, target))
    
    if phi_matches:
        print(f"Found {len(phi_matches)} φ-pattern matches:")
        for i, k, ratio, target in phi_matches[:10]:
            print(f"  S[{i}]/S[{i+1}] = {ratio:.4f} ≈ φ^{k} = {target:.4f}")
    else:
        print("No strong φ-patterns in ratios")
    
    # Check for φ-patterns in singular values themselves
    print("\n--- φ-Pattern Search in Values ---")
    
    for i in range(min(10, len(S_valid))):
        s = S_valid[i]
        # Try to express as (n/d) × φ^k
        best_match = None
        best_err = float('inf')
        
        for k in range(-5, 15):
            phi_k = PHI ** k
            ratio = s / phi_k
            
            # Find best n/d approximation
            for n in range(1, 50):
                for d in range(1, 50):
                    approx = (n / d) * phi_k
                    err = abs(s - approx) / s
                    if err < best_err:
                        best_err = err
                        best_match = (n, d, k, approx)
        
        if best_match and best_err < 0.01:
            n, d, k, approx = best_match
            print(f"  S[{i}] = {s:.4f} ≈ ({n}/{d}) × φ^{k} = {approx:.4f} (err={best_err*100:.2f}%)")
    
    return alpha


def analyze_error_structure(X, Y, tokens, tokenizer, lm_head, W):
    """
    Analyze the 45% error - this is SIGNAL not noise.
    
    Apply Music Box Principle: separate what's captured from what's not.
    """
    print("\n" + "=" * 70)
    print("Error Analysis: The 45% Signal")
    print("=" * 70)
    
    n_samples = len(X)
    X_with_bias = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
    Y_pred = X_with_bias @ W
    
    # Categorize predictions
    correct_idx = []
    wrong_idx = []
    
    for i in range(n_samples):
        pred = (Y_pred[i] @ lm_head.T).argmax().item()
        if pred == tokens[i]:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
    
    print(f"Correct: {len(correct_idx)} ({len(correct_idx)/n_samples*100:.1f}%)")
    print(f"Wrong: {len(wrong_idx)} ({len(wrong_idx)/n_samples*100:.1f}%)")
    
    # Analyze the errors
    print("\n--- Error Cases ---")
    
    errors = []
    for i in wrong_idx:
        pred = (Y_pred[i] @ lm_head.T).argmax().item()
        true = tokens[i]
        
        pred_text = tokenizer.decode([pred])
        true_text = tokenizer.decode([true])
        
        # Compute error vector
        error = Y[i] - Y_pred[i]
        error_norm = torch.norm(error).item()
        
        # Cosine similarity between prediction and truth
        cos_sim = F.cosine_similarity(Y_pred[i].unsqueeze(0), Y[i].unsqueeze(0)).item()
        
        errors.append({
            'idx': i,
            'pred': pred_text,
            'true': true_text,
            'error_norm': error_norm,
            'cos_sim': cos_sim,
            'error_vec': error,
        })
        
        print(f"  [{i}] pred={pred_text!r}, true={true_text!r}, err_norm={error_norm:.2f}, cos={cos_sim:.4f}")
    
    # Analyze error vectors
    if errors:
        print("\n--- Error Vector Analysis ---")
        
        error_vecs = torch.stack([e['error_vec'] for e in errors])
        
        # SVD of error vectors
        U_err, S_err, Vt_err = torch.linalg.svd(error_vecs, full_matrices=False)
        
        print(f"Error matrix shape: {error_vecs.shape}")
        print(f"Error matrix rank: {(S_err > 1e-6).sum().item()}")
        
        # Cumulative variance
        total_var = (S_err**2).sum()
        cumvar = torch.cumsum(S_err**2, dim=0) / total_var * 100
        
        print(f"\nError singular values:")
        for i in range(min(10, len(S_err))):
            print(f"  S_err[{i}] = {S_err[i].item():.4f} ({cumvar[i].item():.1f}%)")
        
        # Is the error low-rank?
        k_90 = (cumvar >= 90).nonzero()[0].item() + 1 if (cumvar >= 90).any() else len(S_err)
        print(f"\nComponents for 90% of error variance: {k_90}")
        
        if k_90 < len(errors) / 2:
            print("ERROR IS LOW-RANK - This is the WALL!")
            print("Apply Music Box Principle: The error lives in a subspace")
    
    # What distinguishes correct from wrong?
    print("\n--- Correct vs Wrong Analysis ---")
    
    if correct_idx and wrong_idx:
        X_correct = X[correct_idx]
        X_wrong = X[wrong_idx]
        
        # Mean input features
        mean_correct = X_correct.mean(dim=0)
        mean_wrong = X_wrong.mean(dim=0)
        
        diff = mean_correct - mean_wrong
        diff_norm = torch.norm(diff).item()
        
        print(f"Mean input difference norm: {diff_norm:.4f}")
        
        # Which dimensions differ most?
        diff_abs = diff.abs()
        top_dims = diff_abs.topk(10)
        
        print(f"Top differing input dimensions:")
        for i, (val, idx) in enumerate(zip(top_dims.values, top_dims.indices)):
            print(f"  Dim {idx.item()}: diff={val.item():.4f}")
    
    return errors


def pep_probe_extraction(model, tokenizer, errors):
    """
    PEP: Probe Extraction Protocol
    
    When we hit a wall, switch to probing to extract exact structure.
    """
    print("\n" + "=" * 70)
    print("PEP: Probe Extraction Protocol")
    print("=" * 70)
    
    if not errors:
        print("No errors to probe")
        return
    
    # For each error case, probe what the model is actually doing
    print("\n--- Probing Error Cases ---")
    
    lm_head = model.lm_head.weight.data.clone()
    embed = model.model.embed_tokens.weight.data.clone()
    
    for err in errors[:5]:  # Probe first 5 errors
        idx = err['idx']
        
        # We need the original prompt - reconstruct from context
        # For now, analyze the error vector
        
        error_vec = err['error_vec']
        
        # What token is the error pointing toward?
        error_logits = error_vec @ lm_head.T
        error_top = error_logits.topk(5)
        
        print(f"\n  Error case {idx}: pred={err['pred']!r}, true={err['true']!r}")
        print(f"  Error vector points toward:")
        for val, tok_idx in zip(error_top.values, error_top.indices):
            tok_text = tokenizer.decode([tok_idx.item()])
            print(f"    {tok_text!r}: {val.item():.4f}")
        
        # Is the true token in the top-k of error direction?
        true_tok_id = tokenizer.encode(err['true'], add_special_tokens=False)[0]
        true_rank = (error_logits > error_logits[true_tok_id]).sum().item()
        print(f"  True token rank in error direction: {true_rank}")


def main():
    """
    Main analysis pipeline applying all protocols.
    """
    print("=" * 70)
    print("Transformer Disentanglement Analysis")
    print("Applying GOP/MGOP/EDP/PEP Protocols")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    lm_head = model.lm_head.weight.data.clone()
    
    # Collect data
    print("\nCollecting transformation data...")
    X, Y, tokens, prompts = collect_transformation_data(model, tokenizer, n_samples=50)
    print(f"Collected {len(X)} samples")
    
    # GOP Phase 1: Fractal Peel
    W, S, U, Vt, acc = gop_phase1_fractal_peel(X, Y, tokens, tokenizer, lm_head)
    
    # MGOP Phase 2: Holographic Scan
    k_bound = mgop_phase2_holographic_scan(X, Y, tokens, lm_head, W, S, U, Vt)
    
    # EDP Phase 4: φ-Pattern Search
    alpha = edp_phase4_phi_patterns(S, W)
    
    # Error Analysis (Music Box Principle)
    errors = analyze_error_structure(X, Y, tokens, tokenizer, lm_head, W)
    
    # PEP: Probe Extraction
    pep_probe_extraction(model, tokenizer, errors)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Linear mapping accuracy: {acc:.1f}%")
    print(f"2. Holographic bound: k={k_bound} dimensions")
    print(f"3. Zipf exponent: α={alpha:.4f} (target 1/φ={PHI_INV:.4f})")
    print(f"4. Error cases: {len(errors)} ({len(errors)/len(X)*100:.1f}%)")
    
    if errors:
        error_vecs = torch.stack([e['error_vec'] for e in errors])
        U_err, S_err, Vt_err = torch.linalg.svd(error_vecs, full_matrices=False)
        cumvar = torch.cumsum(S_err**2, dim=0) / (S_err**2).sum() * 100
        k_90 = (cumvar >= 90).nonzero()[0].item() + 1 if (cumvar >= 90).any() else len(S_err)
        print(f"5. Error rank (90% variance): {k_90}")
        
        if k_90 < len(errors) / 2:
            print("\n*** WALL DETECTED ***")
            print("The error is LOW-RANK - it lives in a subspace!")
            print("Apply Music Box Principle to disentangle.")
    
    return W, S, errors


if __name__ == "__main__":
    W, S, errors = main()
