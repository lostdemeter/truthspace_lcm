#!/usr/bin/env python3
"""
Qwen2.0 φ-Decoder Experiment
=============================

Based on our findings:
1. Semantic axes are nearly orthogonal (82-93°)
2. Semantic distances cluster around 1/φ, 0.5, 1.0
3. Gender axis shows clear structure (man→woman ≈ king→queen)

Can we build a φ-decoder that predicts semantic relationships?

Approach:
1. Find the principal semantic directions using SVD
2. Check if these directions have φ-structure
3. Try to express embeddings in a φ-basis
4. Measure reconstruction quality
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


def create_phi_basis(dim, n_basis=64):
    """
    Create a φ-based orthogonal basis.
    
    Uses powers of φ to create quasi-random but structured vectors,
    then orthogonalizes them.
    """
    basis = np.zeros((n_basis, dim))
    
    for i in range(n_basis):
        # Use φ powers to create quasi-random vectors
        indices = np.arange(dim)
        basis[i] = np.cos(2 * np.pi * (i * PHI + indices * PHI**2))
    
    # Orthogonalize using Gram-Schmidt
    for i in range(n_basis):
        for j in range(i):
            basis[i] -= np.dot(basis[i], basis[j]) * basis[j]
        norm = np.linalg.norm(basis[i])
        if norm > 1e-10:
            basis[i] /= norm
        else:
            # If vector is zero, create a new random one
            basis[i] = np.random.randn(dim)
            basis[i] /= np.linalg.norm(basis[i])
    
    return basis


def analyze_embedding_in_phi_basis(embed_weights, n_basis=64):
    """
    Project embeddings onto φ-basis and analyze structure.
    """
    print()
    print("=" * 70)
    print("φ-BASIS EMBEDDING ANALYSIS")
    print("=" * 70)
    print()
    
    vocab_size, embed_dim = embed_weights.shape
    print(f"Embedding shape: {embed_weights.shape}")
    
    # Create φ-basis
    print(f"Creating {n_basis}-dimensional φ-basis...")
    phi_basis = create_phi_basis(embed_dim, n_basis)
    
    # Project embeddings onto φ-basis
    print("Projecting embeddings...")
    phi_coords = embed_weights @ phi_basis.T  # [vocab_size, n_basis]
    
    print(f"φ-coordinates shape: {phi_coords.shape}")
    
    # Reconstruct embeddings from φ-basis
    reconstructed = phi_coords @ phi_basis  # [vocab_size, embed_dim]
    
    # Measure reconstruction error
    error = embed_weights - reconstructed
    rel_error = np.linalg.norm(error) / np.linalg.norm(embed_weights)
    
    print()
    print(f"Reconstruction with {n_basis} φ-basis vectors:")
    print(f"  Relative error: {rel_error:.4f}")
    print(f"  Variance explained: {(1 - rel_error**2) * 100:.2f}%")
    
    # Compare with PCA
    print()
    print("Comparing with PCA...")
    
    # Center embeddings
    embed_centered = embed_weights - np.mean(embed_weights, axis=0)
    
    # SVD for PCA
    U, S, Vt = np.linalg.svd(embed_centered, full_matrices=False)
    
    # Reconstruct with top n_basis PCA components
    pca_coords = embed_centered @ Vt[:n_basis].T
    pca_reconstructed = pca_coords @ Vt[:n_basis]
    
    pca_error = embed_centered - pca_reconstructed
    pca_rel_error = np.linalg.norm(pca_error) / np.linalg.norm(embed_centered)
    
    print(f"PCA reconstruction with {n_basis} components:")
    print(f"  Relative error: {pca_rel_error:.4f}")
    print(f"  Variance explained: {(1 - pca_rel_error**2) * 100:.2f}%")
    
    return phi_coords, phi_basis, rel_error, pca_rel_error


def find_optimal_phi_dimensions(embed_weights, max_dims=200):
    """
    Find how many φ-basis dimensions we need for good reconstruction.
    """
    print()
    print("=" * 70)
    print("OPTIMAL φ-DIMENSION SEARCH")
    print("=" * 70)
    print()
    
    vocab_size, embed_dim = embed_weights.shape
    
    # Test different numbers of dimensions
    dims_to_test = [8, 16, 32, 64, 128, 256, 512, min(896, embed_dim)]
    
    results = []
    
    for n_dims in dims_to_test:
        if n_dims > embed_dim:
            continue
            
        phi_basis = create_phi_basis(embed_dim, n_dims)
        phi_coords = embed_weights @ phi_basis.T
        reconstructed = phi_coords @ phi_basis
        
        error = embed_weights - reconstructed
        rel_error = np.linalg.norm(error) / np.linalg.norm(embed_weights)
        var_explained = (1 - rel_error**2) * 100
        
        results.append({
            'n_dims': n_dims,
            'rel_error': rel_error,
            'var_explained': var_explained,
        })
        
        print(f"  {n_dims:4d} dims: error={rel_error:.4f}, var_explained={var_explained:.2f}%")
    
    return results


def analyze_phi_coordinate_structure(phi_coords, tokenizer, model):
    """
    Analyze the structure of φ-coordinates for semantic patterns.
    """
    print()
    print("=" * 70)
    print("φ-COORDINATE STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    
    # Get some interesting tokens
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
        "one", "two", "three", "four", "five",
    ]
    
    word_coords = {}
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            word_coords[word] = phi_coords[tokens[0]]
    
    print(f"Analyzing {len(word_coords)} words in φ-space")
    print()
    
    # Check if semantic relationships are preserved
    print("Semantic relationships in φ-space:")
    
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("good", "bad"),
    ]
    
    for w1, w2 in pairs:
        if w1 in word_coords and w2 in word_coords:
            c1 = word_coords[w1]
            c2 = word_coords[w2]
            
            # Cosine distance in φ-space
            cos_dist = 1 - np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            
            # Euclidean distance
            euc_dist = np.linalg.norm(c1 - c2)
            
            print(f"  {w1} <-> {w2}:")
            print(f"    Cosine distance: {cos_dist:.4f}")
            print(f"    Euclidean distance: {euc_dist:.4f}")
            
            # Check for φ-patterns
            if abs(cos_dist - PHI_INV) < 0.1:
                print(f"    → Cosine ≈ 1/φ!")
    
    # Check if coordinate values cluster around φ-based points
    print()
    print("Coordinate value distribution:")
    
    all_coords = phi_coords.flatten()
    
    # Check for clustering around φ-based values
    phi_values = [0, PHI_INV, -PHI_INV, 1, -1, PHI, -PHI]
    
    for pv in phi_values:
        matches = np.sum(np.abs(all_coords - pv) < 0.1)
        pct = matches / len(all_coords) * 100
        if pct > 0.1:
            print(f"  Values near {pv:+.3f}: {pct:.2f}%")
    
    return word_coords


def test_phi_analogy(phi_coords, phi_basis, tokenizer, model):
    """
    Test if analogies work in φ-space.
    """
    print()
    print("=" * 70)
    print("φ-SPACE ANALOGY TEST")
    print("=" * 70)
    print()
    
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    def get_token_id(word):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if len(tokens) == 1 else None
    
    # Test: king - man + woman = ?
    analogies = [
        ("king", "man", "woman", "queen"),
        ("man", "boy", "girl", "woman"),
    ]
    
    for a, b, c, expected in analogies:
        ids = [get_token_id(w) for w in [a, b, c, expected]]
        
        if all(id is not None for id in ids):
            # In φ-space
            phi_a = phi_coords[ids[0]]
            phi_b = phi_coords[ids[1]]
            phi_c = phi_coords[ids[2]]
            phi_expected = phi_coords[ids[3]]
            
            # Analogy in φ-space
            phi_result = phi_a - phi_b + phi_c
            
            # Find nearest neighbor in φ-space
            distances = np.linalg.norm(phi_coords - phi_result, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_word = tokenizer.decode([nearest_idx])
            
            # Distance to expected
            dist_to_expected = np.linalg.norm(phi_result - phi_expected)
            
            print(f"{a} - {b} + {c} = ?")
            print(f"  Expected: {expected}")
            print(f"  Nearest in φ-space: {nearest_word} (idx={nearest_idx})")
            print(f"  Distance to expected: {dist_to_expected:.4f}")
            
            # Also check in original space
            orig_result = embed_weights[ids[0]] - embed_weights[ids[1]] + embed_weights[ids[2]]
            orig_distances = np.linalg.norm(embed_weights - orig_result, axis=1)
            orig_nearest_idx = np.argmin(orig_distances)
            orig_nearest_word = tokenizer.decode([orig_nearest_idx])
            
            print(f"  Nearest in original space: {orig_nearest_word}")
            print()


def main():
    model, tokenizer = load_model()
    
    # Get embedding weights
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    # Analysis 1: φ-basis projection
    phi_coords, phi_basis, phi_error, pca_error = analyze_embedding_in_phi_basis(embed_weights, n_basis=64)
    
    # Analysis 2: Find optimal dimensions
    dim_results = find_optimal_phi_dimensions(embed_weights)
    
    # Analysis 3: Coordinate structure
    word_coords = analyze_phi_coordinate_structure(phi_coords, tokenizer, model)
    
    # Analysis 4: Analogy test
    test_phi_analogy(phi_coords, phi_basis, tokenizer, model)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"φ-basis reconstruction error (64 dims): {phi_error:.4f}")
    print(f"PCA reconstruction error (64 dims): {pca_error:.4f}")
    print()
    
    if phi_error < pca_error * 1.5:
        print("φ-basis is competitive with PCA!")
        print("This suggests the embedding space has φ-structure.")
    else:
        print("PCA significantly outperforms φ-basis.")
        print("The embedding space may not have strong φ-structure.")


if __name__ == "__main__":
    main()
