#!/usr/bin/env python3
"""
Qwen2.0 Embedding Space Analysis
=================================

Analyze the embedding layer for φ-patterns.

Key insight from DA2: The φ-structure was in how the model
encodes information, not necessarily in attention patterns.

For language models, the embedding space IS the encoding.
Let's look for φ-patterns there.
"""

import torch
import numpy as np
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float16,
    )
    model = model.cpu()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def analyze_embedding_geometry(model):
    """Analyze the geometric structure of the embedding space."""
    print()
    print("=" * 70)
    print("EMBEDDING SPACE GEOMETRY")
    print("=" * 70)
    print()
    
    # Get embedding weights
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    print(f"Embedding shape: {embed_weights.shape}")
    print(f"  Vocabulary size: {embed_weights.shape[0]}")
    print(f"  Embedding dimension: {embed_weights.shape[1]}")
    
    # Basic statistics
    print()
    print("Embedding statistics:")
    print(f"  Mean: {np.mean(embed_weights):.6f}")
    print(f"  Std: {np.std(embed_weights):.6f}")
    print(f"  Min: {np.min(embed_weights):.6f}")
    print(f"  Max: {np.max(embed_weights):.6f}")
    
    # Compute norms
    norms = np.linalg.norm(embed_weights, axis=1)
    print()
    print("Embedding norms:")
    print(f"  Mean norm: {np.mean(norms):.4f}")
    print(f"  Std norm: {np.std(norms):.4f}")
    print(f"  Min norm: {np.min(norms):.4f}")
    print(f"  Max norm: {np.max(norms):.4f}")
    
    # Check if norms follow φ-distribution
    print()
    print("Checking for φ-patterns in norms...")
    
    # Histogram of norms
    hist, bins = np.histogram(norms, bins=50)
    peak_idx = np.argmax(hist)
    peak_norm = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    print(f"  Peak norm: {peak_norm:.4f}")
    print(f"  φ: {PHI:.4f}")
    print(f"  Peak/φ: {peak_norm/PHI:.4f}")
    
    return embed_weights, norms


def analyze_embedding_svd(embed_weights):
    """SVD analysis of embedding matrix."""
    print()
    print("=" * 70)
    print("EMBEDDING SVD ANALYSIS")
    print("=" * 70)
    print()
    
    # SVD (use truncated for speed)
    print("Computing SVD (this may take a moment)...")
    
    # Center the embeddings
    embed_centered = embed_weights - np.mean(embed_weights, axis=0)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(embed_centered, full_matrices=False)
    
    print(f"Singular values shape: {S.shape}")
    print()
    print("Top 30 singular values:")
    print(f"  {S[:30].round(2)}")
    
    # Check for φ-ratios
    print()
    print("Consecutive singular value ratios:")
    ratios = S[:-1] / S[1:]
    
    phi_matches = []
    for i in range(min(50, len(ratios))):
        phi_diff = abs(ratios[i] - PHI)
        phi_inv_diff = abs(ratios[i] - PHI_INV)
        
        if phi_diff < 0.05:
            phi_matches.append((i, ratios[i], 'φ'))
            print(f"  S[{i}]/S[{i+1}] = {ratios[i]:.4f} ≈ φ ✓")
        elif phi_inv_diff < 0.05:
            phi_matches.append((i, ratios[i], '1/φ'))
            print(f"  S[{i}]/S[{i+1}] = {ratios[i]:.4f} ≈ 1/φ ✓")
    
    if not phi_matches:
        print("  No exact φ-ratio matches in top 50 singular values")
        print()
        print("  Sample ratios:")
        for i in range(10):
            print(f"    S[{i}]/S[{i+1}] = {ratios[i]:.4f}")
    
    # Analyze singular value decay
    print()
    print("Singular value decay analysis:")
    
    # Fit power law: S[i] ∝ i^(-α)
    log_idx = np.log(np.arange(1, len(S) + 1))
    log_S = np.log(S)
    
    # Linear fit in log-log space
    coeffs = np.polyfit(log_idx[:100], log_S[:100], 1)
    alpha = -coeffs[0]
    
    print(f"  Power law exponent α: {alpha:.4f}")
    print(f"  φ: {PHI:.4f}")
    print(f"  α/φ: {alpha/PHI:.4f}")
    
    # Check if α is related to φ
    if abs(alpha - PHI) < 0.2:
        print(f"  → α ≈ φ!")
    elif abs(alpha - 1/PHI) < 0.2:
        print(f"  → α ≈ 1/φ!")
    elif abs(alpha - 1) < 0.2:
        print(f"  → α ≈ 1 (Zipf's law)")
    
    return S, Vt, alpha


def analyze_semantic_clusters(embed_weights, tokenizer, n_clusters=20):
    """Analyze semantic clustering in embedding space."""
    print()
    print("=" * 70)
    print("SEMANTIC CLUSTER ANALYSIS")
    print("=" * 70)
    print()
    
    # Sample some interesting tokens
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
        "big", "small", "fast", "slow",
        "one", "two", "three", "four", "five",
        "red", "blue", "green", "yellow",
    ]
    
    # Get token IDs
    token_ids = {}
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            token_ids[word] = tokens[0]
    
    print(f"Found {len(token_ids)} single-token words")
    
    if len(token_ids) < 5:
        print("Not enough single-token words for analysis")
        return None
    
    # Get embeddings for these tokens
    words = list(token_ids.keys())
    ids = list(token_ids.values())
    embeddings = embed_weights[ids]
    
    print()
    print("Analyzing semantic relationships...")
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    distances = squareform(pdist(embeddings, metric='cosine'))
    
    # Find interesting pairs
    print()
    print("Semantic pairs (cosine distance):")
    
    pairs_to_check = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("good", "bad"),
        ("happy", "sad"),
        ("big", "small"),
    ]
    
    for w1, w2 in pairs_to_check:
        if w1 in words and w2 in words:
            i1, i2 = words.index(w1), words.index(w2)
            dist = distances[i1, i2]
            print(f"  {w1} <-> {w2}: {dist:.4f}")
    
    # Check for φ-patterns in semantic distances
    print()
    print("Checking for φ-patterns in semantic distances...")
    
    all_distances = distances[np.triu_indices(len(words), k=1)]
    
    # Check if distances cluster around φ-based values
    phi_distances = [PHI_INV, 0.5, PHI_INV * 2, 1.0]
    
    for phi_d in phi_distances:
        matches = np.sum(np.abs(all_distances - phi_d) < 0.05)
        if matches > 0:
            print(f"  {matches} pairs at distance ≈ {phi_d:.4f}")
    
    return distances, words


def analyze_dimension_roles(embed_weights, tokenizer):
    """Analyze what each embedding dimension encodes."""
    print()
    print("=" * 70)
    print("DIMENSION ROLE ANALYSIS")
    print("=" * 70)
    print()
    
    n_dims = embed_weights.shape[1]
    print(f"Analyzing {n_dims} dimensions...")
    
    # For each dimension, find tokens with highest/lowest values
    print()
    print("Top dimensions by variance:")
    
    dim_vars = np.var(embed_weights, axis=0)
    top_dims = np.argsort(dim_vars)[::-1][:10]
    
    for rank, dim in enumerate(top_dims):
        var = dim_vars[dim]
        
        # Get tokens with highest values in this dimension
        top_tokens = np.argsort(embed_weights[:, dim])[::-1][:5]
        bottom_tokens = np.argsort(embed_weights[:, dim])[:5]
        
        top_words = [tokenizer.decode([t]) for t in top_tokens]
        bottom_words = [tokenizer.decode([t]) for t in bottom_tokens]
        
        print(f"  Dim {dim} (var={var:.4f}):")
        print(f"    High: {top_words}")
        print(f"    Low: {bottom_words}")
    
    # Check if dimension variances follow φ-pattern
    print()
    print("Dimension variance distribution:")
    
    sorted_vars = np.sort(dim_vars)[::-1]
    ratios = sorted_vars[:-1] / sorted_vars[1:]
    
    phi_matches = []
    for i in range(min(20, len(ratios))):
        if abs(ratios[i] - PHI) < 0.1:
            phi_matches.append((i, ratios[i]))
    
    if phi_matches:
        print(f"  Found {len(phi_matches)} φ-ratio matches in variance decay")
        for i, r in phi_matches[:5]:
            print(f"    Var[{i}]/Var[{i+1}] = {r:.4f} ≈ φ")
    else:
        print("  No φ-ratio matches in variance decay")
    
    return dim_vars, top_dims


def main():
    model, tokenizer = load_model()
    
    # Analysis 1: Embedding geometry
    embed_weights, norms = analyze_embedding_geometry(model)
    
    # Analysis 2: SVD
    S, Vt, alpha = analyze_embedding_svd(embed_weights)
    
    # Analysis 3: Semantic clusters
    distances, words = analyze_semantic_clusters(embed_weights, tokenizer)
    
    # Analysis 4: Dimension roles
    dim_vars, top_dims = analyze_dimension_roles(embed_weights, tokenizer)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Embedding shape: {embed_weights.shape}")
    print(f"Power law exponent: {alpha:.4f}")
    print(f"Mean embedding norm: {np.mean(norms):.4f}")
    
    # Save results
    results = {
        'vocab_size': int(embed_weights.shape[0]),
        'embed_dim': int(embed_weights.shape[1]),
        'power_law_alpha': float(alpha),
        'mean_norm': float(np.mean(norms)),
        'top_singular_values': S[:20].tolist(),
    }
    
    with open('qwen2_embedding_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to qwen2_embedding_analysis.json")


if __name__ == "__main__":
    main()
