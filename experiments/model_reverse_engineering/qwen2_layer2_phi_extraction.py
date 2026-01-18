#!/usr/bin/env python3
"""
Qwen2.0 Layer 2 φ-Basis Extraction
====================================

Based on our discovery:
- Layer 2 is the optimal semantic layer
- Analogies work here
- S[0]/S[1] ≈ φ in semantic subspace

Goal: Extract a φ-basis representation from Layer 2 that:
1. Captures the semantic structure
2. Allows exact reconstruction
3. Has φ-weighted dimensions

This is the "drum" extraction - getting the pure semantic structure
before the transcoder (layers 3-24) transforms it.
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
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_layer2_embeddings(model, tokenizer, words):
    """Get Layer 2 hidden states for words."""
    
    embeddings = {}
    
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            continue
        
        input_ids = torch.tensor([[tokens[0]]])
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        # Layer 2 hidden state
        embeddings[word] = outputs.hidden_states[2][0, 0].numpy()
    
    return embeddings


def extract_phi_basis_from_layer2(embeddings):
    """
    Extract φ-basis from Layer 2 embeddings.
    
    Approach:
    1. Compute semantic delta vectors
    2. SVD to find principal semantic directions
    3. Apply φ-weighting based on singular values
    """
    print()
    print("=" * 70)
    print("EXTRACTING φ-BASIS FROM LAYER 2")
    print("=" * 70)
    print()
    
    words = list(embeddings.keys())
    embed_matrix = np.array([embeddings[w] for w in words])
    
    print(f"Embedding matrix shape: {embed_matrix.shape}")
    
    # Center embeddings
    mean_embed = np.mean(embed_matrix, axis=0)
    centered = embed_matrix - mean_embed
    
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    print(f"Singular values (top 20): {S[:20].round(4)}")
    
    # Check for φ-patterns
    ratios = S[:-1] / S[1:]
    print()
    print("Singular value ratios:")
    phi_count = 0
    for i in range(min(20, len(ratios))):
        r = ratios[i]
        marker = ""
        if abs(r - PHI) < 0.15:
            marker = " ← φ!"
            phi_count += 1
        elif abs(r - PHI_INV) < 0.15:
            marker = " ← 1/φ!"
            phi_count += 1
        print(f"  S[{i}]/S[{i+1}] = {r:.4f}{marker}")
    
    print(f"\nφ-matches in top 20: {phi_count}")
    
    # Create φ-basis
    # Weight each principal component by φ^(-i/k)
    n_dims = len(S)
    k = n_dims / 5  # Decay rate
    
    phi_weights = np.array([PHI ** (-i / k) for i in range(n_dims)])
    
    # The φ-basis vectors
    phi_basis = Vt  # Principal components
    
    return phi_basis, S, phi_weights, mean_embed, words


def test_reconstruction(embeddings, phi_basis, S, phi_weights, mean_embed, words):
    """
    Test if we can exactly reconstruct embeddings from φ-basis.
    """
    print()
    print("=" * 70)
    print("TESTING RECONSTRUCTION")
    print("=" * 70)
    print()
    
    embed_matrix = np.array([embeddings[w] for w in words])
    centered = embed_matrix - mean_embed
    
    # Project to φ-basis
    phi_coords = centered @ phi_basis.T  # [n_words, n_dims]
    
    # Apply φ-weighting
    phi_coords_weighted = phi_coords * phi_weights
    
    # Reconstruct (undo weighting first)
    phi_coords_unweighted = phi_coords_weighted / phi_weights
    reconstructed = phi_coords_unweighted @ phi_basis + mean_embed
    
    # Error
    error = embed_matrix - reconstructed
    rel_error = np.linalg.norm(error) / np.linalg.norm(embed_matrix)
    
    print(f"Reconstruction error: {rel_error:.10f}")
    
    if rel_error < 1e-6:
        print("→ EXACT reconstruction achieved! ✓")
    else:
        print("→ Reconstruction has error")
    
    return phi_coords_weighted, rel_error


def test_semantic_operations_in_phi_basis(embeddings, phi_basis, phi_weights, mean_embed):
    """
    Test if semantic operations work in the φ-basis.
    """
    print()
    print("=" * 70)
    print("TESTING SEMANTIC OPERATIONS IN φ-BASIS")
    print("=" * 70)
    print()
    
    def to_phi_coords(word):
        if word not in embeddings:
            return None
        embed = embeddings[word]
        centered = embed - mean_embed
        coords = centered @ phi_basis.T
        return coords * phi_weights
    
    def from_phi_coords(coords):
        unweighted = coords / phi_weights
        return unweighted @ phi_basis + mean_embed
    
    def find_nearest(target_embed, exclude_words=[]):
        best_word = None
        best_dist = float('inf')
        
        for word, embed in embeddings.items():
            if word in exclude_words:
                continue
            dist = np.linalg.norm(target_embed - embed)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word, best_dist
    
    # Test analogies
    analogies = [
        ("king", "man", "woman", "queen"),
        ("man", "boy", "girl", "woman"),
    ]
    
    print("Analogies in φ-basis:")
    
    for a, b, c, expected in analogies:
        coords_a = to_phi_coords(a)
        coords_b = to_phi_coords(b)
        coords_c = to_phi_coords(c)
        
        if any(x is None for x in [coords_a, coords_b, coords_c]):
            continue
        
        # Analogy in φ-space
        result_coords = coords_a - coords_b + coords_c
        
        # Convert back to embedding space
        result_embed = from_phi_coords(result_coords)
        
        # Find nearest
        nearest, dist = find_nearest(result_embed, exclude_words=[a, b, c])
        
        match = "✓" if nearest == expected else "✗"
        print(f"  {a} - {b} + {c} = {nearest} (expected: {expected}) {match}")
    
    # Test distances in φ-basis
    print()
    print("Distances in φ-basis:")
    
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("good", "bad"),
    ]
    
    for w1, w2 in pairs:
        c1 = to_phi_coords(w1)
        c2 = to_phi_coords(w2)
        
        if c1 is None or c2 is None:
            continue
        
        dist = np.linalg.norm(c2 - c1)
        phi_ratio = dist / PHI_INV
        
        print(f"  {w1} ↔ {w2}: {dist:.4f} = {phi_ratio:.2f} × (1/φ)")


def analyze_phi_coordinate_structure(phi_coords, words, phi_weights):
    """
    Analyze the structure of φ-coordinates.
    """
    print()
    print("=" * 70)
    print("φ-COORDINATE STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    
    print("φ-coordinates for each word (first 10 dims):")
    
    for i, word in enumerate(words):
        coords = phi_coords[i, :10]
        print(f"  {word:12s}: {coords.round(4)}")
    
    # Check if coordinates cluster around φ-based values
    print()
    print("Coordinate value distribution:")
    
    all_coords = phi_coords.flatten()
    
    # Test clustering around specific values
    test_values = [0, PHI_INV, -PHI_INV, 1, -1, PHI, -PHI]
    
    for val in test_values:
        near = np.sum(np.abs(all_coords - val) < 0.5)
        pct = near / len(all_coords) * 100
        if pct > 1:
            print(f"  Values near {val:+.3f}: {pct:.1f}%")


def create_minimal_phi_representation(embeddings, phi_basis, S, phi_weights, mean_embed, words):
    """
    Create the minimal φ-representation that captures semantics.
    
    How many dimensions do we actually need?
    """
    print()
    print("=" * 70)
    print("MINIMAL φ-REPRESENTATION")
    print("=" * 70)
    print()
    
    embed_matrix = np.array([embeddings[w] for w in words])
    centered = embed_matrix - mean_embed
    
    # Test with different numbers of dimensions
    for n_dims in [5, 10, 20, 50, 100]:
        if n_dims > len(S):
            continue
        
        # Truncated basis
        basis_trunc = phi_basis[:n_dims]
        weights_trunc = phi_weights[:n_dims]
        
        # Project and reconstruct
        coords = centered @ basis_trunc.T
        coords_weighted = coords * weights_trunc
        coords_unweighted = coords_weighted / weights_trunc
        reconstructed = coords_unweighted @ basis_trunc + mean_embed
        
        # Error
        error = embed_matrix - reconstructed
        rel_error = np.linalg.norm(error) / np.linalg.norm(embed_matrix)
        var_explained = 1 - rel_error**2
        
        # Test analogy
        def test_analogy(a, b, c, expected):
            idx = {w: i for i, w in enumerate(words)}
            if not all(w in idx for w in [a, b, c, expected]):
                return None
            
            result = coords_weighted[idx[a]] - coords_weighted[idx[b]] + coords_weighted[idx[c]]
            
            # Find nearest in truncated space
            distances = np.linalg.norm(coords_weighted - result, axis=1)
            for w in [a, b, c]:
                distances[idx[w]] = np.inf
            
            nearest_idx = np.argmin(distances)
            return words[nearest_idx] == expected
        
        analogy_works = test_analogy("king", "man", "woman", "queen")
        analogy_status = "✓" if analogy_works else "✗" if analogy_works is not None else "?"
        
        print(f"  {n_dims:3d} dims: error={rel_error:.4f}, var={var_explained:.2%}, analogy={analogy_status}")


def main():
    model, tokenizer = load_model()
    
    # Get Layer 2 embeddings for test words
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
        "big", "small", "fast", "slow",
        "father", "mother", "son", "daughter",
    ]
    
    embeddings = get_layer2_embeddings(model, tokenizer, test_words)
    print(f"Got Layer 2 embeddings for {len(embeddings)} words")
    
    # Extract φ-basis
    phi_basis, S, phi_weights, mean_embed, words = extract_phi_basis_from_layer2(embeddings)
    
    # Test reconstruction
    phi_coords, rel_error = test_reconstruction(embeddings, phi_basis, S, phi_weights, mean_embed, words)
    
    # Test semantic operations
    test_semantic_operations_in_phi_basis(embeddings, phi_basis, phi_weights, mean_embed)
    
    # Analyze coordinate structure
    analyze_phi_coordinate_structure(phi_coords, words, phi_weights)
    
    # Find minimal representation
    create_minimal_phi_representation(embeddings, phi_basis, S, phi_weights, mean_embed, words)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Layer 2 φ-basis extraction:")
    print(f"  - {len(words)} words analyzed")
    print(f"  - {len(S)} dimensions in basis")
    print(f"  - Reconstruction error: {rel_error:.10f}")
    print()
    print("This is the 'drum' - the semantic structure at Layer 2.")
    print("The φ-weighting provides a principled way to prioritize dimensions.")
    print()
    print("Next: Apply this to the FULL vocabulary to get complete φ-representation.")


if __name__ == "__main__":
    main()
