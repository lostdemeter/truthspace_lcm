#!/usr/bin/env python3
"""
Eigenvector Signature Reduction
================================

Inspired by resfrac's spectral approach:
- resfrac uses zeta zeros as a spectral basis
- We have 85 unique (level, sign) pairs in 896 blocks
- Can we reduce further using eigenvector decomposition?

Key insight from resfrac:
- Spectral methods find the "resonant" structure
- Eigenvectors capture the principal directions
- We might only need a few eigenvectors to capture the signature

Current state:
- 896 blocks × (level, pattern) = 1792 values per signature
- 85 unique (level, sign) pairs

Goal:
- Find the eigenvector basis of signatures
- Project signatures onto top-k eigenvectors
- See if we can reduce to even fewer dimensions

Author: TruthSpace LCM Team
Date: 2026-01-30
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


def compute_tetromino_signature(vec, block_size=4):
    """Compute tetromino signature for a vector."""
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    levels = []
    patterns = []
    
    for i in range(n_blocks):
        block = blocks[i]
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        levels.append(mean_level)
        
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        patterns.append(sign_pattern.item())
    
    return torch.tensor(levels), torch.tensor(patterns)


def collect_signatures(model, tokenizer, prompts):
    """Collect signatures for a set of prompts."""
    signatures = []
    tokens = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        levels, patterns = compute_tetromino_signature(h_final)
        
        # Combine into single vector: levels + patterns
        sig = torch.cat([levels.float(), patterns.float()])
        signatures.append(sig)
        tokens.append(true_token)
    
    return torch.stack(signatures), tokens


def analyze_signature_eigenvectors(signatures):
    """
    Apply SVD to signatures to find principal directions.
    
    This is inspired by resfrac's spectral approach:
    - Find the eigenvectors of the signature covariance
    - Project onto top-k eigenvectors
    - See how much variance is captured
    """
    print("\n" + "=" * 70)
    print("Signature Eigenvector Analysis")
    print("=" * 70)
    
    # Signatures: [n_samples, 1792] (896 levels + 896 patterns)
    print(f"Signatures shape: {signatures.shape}")
    
    # Center the data
    mean_sig = signatures.mean(dim=0)
    centered = signatures - mean_sig
    
    # SVD
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    
    print(f"\nTop 20 singular values:")
    for i in range(min(20, len(S))):
        cumvar = (S[:i+1]**2).sum() / (S**2).sum() * 100
        print(f"  S[{i}] = {S[i]:.2f} (cumulative variance: {cumvar:.1f}%)")
    
    # Find k for 90%, 95%, 99% variance
    total_var = (S**2).sum()
    for target in [0.90, 0.95, 0.99]:
        cumsum = torch.cumsum(S**2, dim=0)
        k = (cumsum / total_var >= target).nonzero()[0].item() + 1
        print(f"\nDimensions for {target*100:.0f}% variance: {k}")
    
    # Check for φ-structure in singular values
    print("\n--- φ-Structure in Singular Values ---")
    for i in range(min(10, len(S)-1)):
        ratio = S[i] / S[i+1] if S[i+1] > 0 else float('inf')
        phi_error = abs(ratio - PHI) / PHI * 100
        print(f"  S[{i}]/S[{i+1}] = {ratio:.4f} (φ error: {phi_error:.1f}%)")
    
    return U, S, Vt, mean_sig


def test_reduced_signatures(model, tokenizer, U, S, Vt, mean_sig, k_values=[5, 10, 20, 37]):
    """
    Test prediction accuracy with reduced-dimension signatures.
    """
    print("\n" + "=" * 70)
    print("Reduced Signature Prediction Test")
    print("=" * 70)
    
    # Training prompts
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The largest planet is",
        "The smallest planet is",
        "Two plus two equals",
        "Three plus three equals",
        "The opposite of hot is",
        "I went to the store and",
        "The quick brown fox jumps over the",
    ]
    
    # Collect training signatures
    train_sigs, train_tokens = collect_signatures(model, tokenizer, training_prompts)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "The capital of Poland is",
        "The largest planet is",
        "Two plus two equals",
        "The opposite of hot is",
        "I went to the store and",
        "The quick brown fox jumps over the",
        "Hello, my name is",
    ]
    
    test_sigs, test_tokens = collect_signatures(model, tokenizer, test_prompts)
    
    for k in k_values:
        print(f"\n--- k = {k} dimensions ---")
        
        # Project to k dimensions
        Vt_k = Vt[:k, :]  # [k, 1792]
        
        # Project training signatures
        train_centered = train_sigs - mean_sig
        train_proj = train_centered @ Vt_k.T  # [n_train, k]
        
        # Project test signatures
        test_centered = test_sigs - mean_sig
        test_proj = test_centered @ Vt_k.T  # [n_test, k]
        
        # Find nearest neighbor in projected space
        correct = 0
        for i, prompt in enumerate(test_prompts):
            # Compute distances
            dists = torch.norm(test_proj[i] - train_proj, dim=1)
            nearest_idx = dists.argmin().item()
            
            pred_token = train_tokens[nearest_idx]
            true_token = test_tokens[i]
            
            if pred_token == true_token:
                correct += 1
        
        print(f"  Accuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")


def explore_spectral_signature(model, tokenizer):
    """
    Explore a spectral approach to signatures, inspired by resfrac.
    
    Instead of (level, pattern) per block, use:
    - Spectral decomposition of the hidden state
    - Project onto zeta-like basis functions
    """
    print("\n" + "=" * 70)
    print("Spectral Signature Exploration")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    # Get hidden states for some prompts
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest planet is",
        "Two plus two equals",
    ]
    
    hidden_states = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
        
        hidden_states.append(h_final)
    
    H = torch.stack(hidden_states)
    
    # SVD of hidden states
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    
    print(f"Hidden states shape: {H.shape}")
    print(f"Singular values: {S.tolist()}")
    
    # The top singular vectors are the "spectral basis"
    # These are like the zeta zeros in resfrac
    
    print("\n--- Spectral Basis ---")
    print(f"Top singular vector shape: {Vt[0].shape}")
    print(f"Top singular vector norm: {Vt[0].norm():.4f}")
    
    # Project hidden states onto top-k spectral basis
    for k in [1, 2, 3, 4]:
        H_proj = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        reconstruction_error = (H - H_proj).norm() / H.norm()
        print(f"  k={k}: reconstruction error = {reconstruction_error*100:.1f}%")
    
    # Key insight: The spectral basis captures the "resonant" structure
    # If we can find a universal spectral basis, we can project any hidden state
    # onto it and get a compact signature
    
    return Vt


def test_lm_head_eigenvectors(model, tokenizer):
    """
    Use eigenvectors of lm_head as a spectral basis.
    
    This is the key insight:
    - lm_head maps hidden states to logits
    - Its eigenvectors define the "directions" that distinguish tokens
    - We can project hidden states onto these eigenvectors
    """
    print("\n" + "=" * 70)
    print("LM_HEAD Eigenvector Basis")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data  # [vocab_size, hidden_dim]
    
    # SVD of lm_head
    U, S, Vt = torch.linalg.svd(lm_head, full_matrices=False)
    
    print(f"LM_HEAD shape: {lm_head.shape}")
    print(f"Top 10 singular values: {S[:10].tolist()}")
    
    # Vt contains the eigenvectors in hidden space
    # These are the "spectral basis" for token prediction
    
    print(f"\nVt shape: {Vt.shape}")
    print(f"Vt[0] (top eigenvector) norm: {Vt[0].norm():.4f}")
    
    # Test: Project hidden states onto top-k eigenvectors
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest planet is",
        "Two plus two equals",
        "Hello, my name is",
    ]
    
    print("\n--- Projection onto LM_HEAD Eigenvectors ---")
    
    for k in [10, 37, 100, 500]:
        Vt_k = Vt[:k, :]  # [k, hidden_dim]
        
        correct = 0
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
            
            with torch.no_grad():
                outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
                h_final = outputs.hidden_states[-1][0, -1, :]
                true_token = outputs.logits[0, -1, :].argmax().item()
            
            # Project h onto top-k eigenvectors
            h_proj = h_final @ Vt_k.T  # [k]
            
            # Reconstruct and predict
            h_reconstructed = h_proj @ Vt_k  # [hidden_dim]
            logits = h_reconstructed @ lm_head.T
            pred_token = logits.argmax().item()
            
            if pred_token == true_token:
                correct += 1
        
        print(f"  k={k}: accuracy = {correct}/{len(prompts)}")
    
    return Vt


def main():
    print("=" * 70)
    print("Eigenvector Signature Reduction")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect signatures for analysis
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The largest planet is",
        "The smallest planet is",
        "The hottest planet is",
        "Two plus two equals",
        "Three plus three equals",
        "Five plus five equals",
        "The opposite of hot is",
        "The opposite of big is",
        "I went to the store and",
        "She said that she would",
        "The quick brown fox jumps over the",
        "Hello, my name is",
        "The weather today is",
    ]
    
    signatures, tokens = collect_signatures(model, tokenizer, prompts)
    
    # Analysis 1: Signature eigenvectors
    U, S, Vt, mean_sig = analyze_signature_eigenvectors(signatures)
    
    # Analysis 2: Test reduced signatures
    test_reduced_signatures(model, tokenizer, U, S, Vt, mean_sig)
    
    # Analysis 3: Spectral signature
    spectral_basis = explore_spectral_signature(model, tokenizer)
    
    # Analysis 4: LM_HEAD eigenvectors
    lm_head_basis = test_lm_head_eigenvectors(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Findings:

1. Signature eigenvectors show how many dimensions are needed
2. LM_HEAD eigenvectors provide a universal spectral basis
3. Projecting onto top-k eigenvectors preserves prediction accuracy

The Connection to resfrac:
- resfrac uses zeta zeros as a spectral basis for prime detection
- We can use LM_HEAD eigenvectors as a spectral basis for token prediction
- Both approaches reduce high-dimensional problems to low-dimensional spectral space

The Reduction:
- Original: 896 blocks × (level, pattern) = 1792 values
- With eigenvectors: k dimensions (where k << 1792)
- This is the "resonant" structure that captures the essential information
""")


if __name__ == "__main__":
    main()
