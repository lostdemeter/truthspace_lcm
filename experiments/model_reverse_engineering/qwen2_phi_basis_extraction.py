#!/usr/bin/env python3
"""
Qwen2.0 φ-Basis Extraction
===========================

Goal: Convert Qwen2's embedding space to a φ-basis representation
where operations become trivial (like DA2 where decoding = summation).

Music Box Principle:
- DRUM = Word positions in φ-space (what we're extracting)
- COMB = find_nearest decoder (trivial once we have the drum)
- MUSIC = Emergent output

Universal Dimension Principle:
- Content dimensions (gender, age, size)
- Pattern dimensions (tense, register, tone)
- Style dimensions (spacing, case)

Approach:
1. Extract semantic axes from transformation pairs
2. Orthogonalize to get independent dimensions
3. Express each embedding as φ-weighted coordinates
4. Verify reconstruction is exact
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

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


def extract_semantic_axes(embed_weights, tokenizer):
    """
    Extract semantic axes from transformation pairs.
    
    Each axis represents a dimension like gender, tense, size, etc.
    """
    print()
    print("=" * 70)
    print("EXTRACTING SEMANTIC AXES (THE DRUM)")
    print("=" * 70)
    print()
    
    def get_embed(word):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            return embed_weights[tokens[0]]
        return None
    
    # Define transformation pairs for each dimension
    # These are the "bumps on the drum" that define the structure
    dimension_pairs = {
        # Content dimensions
        'gender': [
            ('king', 'queen'), ('man', 'woman'), ('boy', 'girl'),
            ('father', 'mother'), ('son', 'daughter'), ('brother', 'sister'),
            ('he', 'she'), ('him', 'her'),
        ],
        'age': [
            ('boy', 'man'), ('girl', 'woman'),
            ('young', 'old'), ('child', 'adult'),
        ],
        'size': [
            ('big', 'small'), ('large', 'tiny'), ('huge', 'little'),
            ('giant', 'dwarf'),
        ],
        'sentiment': [
            ('good', 'bad'), ('happy', 'sad'), ('love', 'hate'),
            ('beautiful', 'ugly'), ('nice', 'mean'),
        ],
        # Pattern dimensions (from Universal Dimension Principle)
        'tense_past_present': [
            ('went', 'go'), ('was', 'is'), ('had', 'have'),
            ('did', 'do'), ('said', 'say'),
        ],
        'tense_present_future': [
            ('go', 'going'), ('is', 'will'),
        ],
        'formality': [
            ('hello', 'hi'), ('yes', 'yeah'), ('no', 'nah'),
            ('please', 'pls'),
        ],
        'intensity': [
            ('good', 'great'), ('bad', 'terrible'),
            ('like', 'love'), ('dislike', 'hate'),
        ],
    }
    
    axes = {}
    axis_pairs_used = {}
    
    for dim_name, pairs in dimension_pairs.items():
        deltas = []
        valid_pairs = []
        
        for w1, w2 in pairs:
            e1, e2 = get_embed(w1), get_embed(w2)
            if e1 is not None and e2 is not None:
                delta = e2 - e1
                deltas.append(delta)
                valid_pairs.append((w1, w2))
        
        if len(deltas) >= 2:
            # Average the deltas to get the axis
            axis = np.mean(deltas, axis=0)
            axis = axis / np.linalg.norm(axis)  # Normalize
            
            axes[dim_name] = axis
            axis_pairs_used[dim_name] = valid_pairs
            
            print(f"{dim_name}: {len(valid_pairs)} pairs")
        else:
            print(f"{dim_name}: Not enough pairs (need 2+)")
    
    print(f"\nExtracted {len(axes)} semantic axes")
    
    return axes, axis_pairs_used


def orthogonalize_axes(axes):
    """
    Orthogonalize axes using Gram-Schmidt.
    
    This ensures dimensions are independent.
    """
    print()
    print("=" * 70)
    print("ORTHOGONALIZING AXES")
    print("=" * 70)
    print()
    
    axis_names = list(axes.keys())
    axis_vectors = np.array([axes[name] for name in axis_names])
    
    # Check initial angles between axes
    print("Initial angles between axes:")
    for i in range(len(axis_names)):
        for j in range(i+1, len(axis_names)):
            cos = np.dot(axis_vectors[i], axis_vectors[j])
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if angle < 80 or angle > 100:
                print(f"  {axis_names[i]} ↔ {axis_names[j]}: {angle:.1f}°")
    
    # Gram-Schmidt orthogonalization
    ortho_vectors = np.zeros_like(axis_vectors)
    
    for i in range(len(axis_vectors)):
        v = axis_vectors[i].copy()
        
        # Subtract projections onto previous vectors
        for j in range(i):
            v = v - np.dot(v, ortho_vectors[j]) * ortho_vectors[j]
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            ortho_vectors[i] = v / norm
        else:
            print(f"  Warning: {axis_names[i]} is linearly dependent, skipping")
            ortho_vectors[i] = np.zeros_like(v)
    
    # Create orthogonalized axes dict
    ortho_axes = {name: ortho_vectors[i] for i, name in enumerate(axis_names)}
    
    # Verify orthogonality
    print()
    print("Verifying orthogonality:")
    max_dot = 0
    for i in range(len(axis_names)):
        for j in range(i+1, len(axis_names)):
            dot = abs(np.dot(ortho_vectors[i], ortho_vectors[j]))
            max_dot = max(max_dot, dot)
    
    print(f"  Max dot product: {max_dot:.6f} (should be ~0)")
    
    return ortho_axes


def project_to_phi_basis(embed_weights, ortho_axes, tokenizer):
    """
    Project all embeddings onto the orthogonal semantic axes.
    
    This gives us the φ-basis coordinates for each word.
    """
    print()
    print("=" * 70)
    print("PROJECTING TO φ-BASIS")
    print("=" * 70)
    print()
    
    axis_names = list(ortho_axes.keys())
    axis_matrix = np.array([ortho_axes[name] for name in axis_names])  # [n_axes, embed_dim]
    
    # Project all embeddings
    # phi_coords[i, j] = projection of embedding i onto axis j
    phi_coords = embed_weights @ axis_matrix.T  # [vocab_size, n_axes]
    
    print(f"φ-coordinates shape: {phi_coords.shape}")
    print(f"  {phi_coords.shape[0]} tokens × {phi_coords.shape[1]} dimensions")
    
    # Analyze coordinate distribution
    print()
    print("Coordinate statistics per dimension:")
    
    for i, name in enumerate(axis_names):
        coords = phi_coords[:, i]
        print(f"  {name:20s}: mean={coords.mean():+.4f}, std={coords.std():.4f}, "
              f"range=[{coords.min():.2f}, {coords.max():.2f}]")
    
    return phi_coords, axis_names


def test_reconstruction(embed_weights, phi_coords, ortho_axes, axis_names):
    """
    Test if we can reconstruct embeddings from φ-coordinates.
    
    If reconstruction is exact, we've captured the structure.
    """
    print()
    print("=" * 70)
    print("TESTING RECONSTRUCTION")
    print("=" * 70)
    print()
    
    axis_matrix = np.array([ortho_axes[name] for name in axis_names])  # [n_axes, embed_dim]
    
    # Reconstruct embeddings from φ-coordinates
    reconstructed = phi_coords @ axis_matrix  # [vocab_size, embed_dim]
    
    # Compute reconstruction error
    error = embed_weights - reconstructed
    
    # Per-embedding error
    embed_norms = np.linalg.norm(embed_weights, axis=1)
    error_norms = np.linalg.norm(error, axis=1)
    rel_errors = error_norms / (embed_norms + 1e-10)
    
    print(f"Reconstruction with {len(axis_names)} semantic axes:")
    print(f"  Mean relative error: {rel_errors.mean():.4f}")
    print(f"  Max relative error: {rel_errors.max():.4f}")
    print(f"  Variance explained: {(1 - (error_norms**2).sum() / (embed_norms**2).sum()) * 100:.2f}%")
    
    # The remaining error is the "residual" - dimensions not captured by our axes
    residual = error
    residual_norms = np.linalg.norm(residual, axis=1)
    
    print()
    print(f"Residual (unexplained) per embedding:")
    print(f"  Mean: {residual_norms.mean():.4f}")
    print(f"  This is the 'comb' - the decoder needs to handle this")
    
    return reconstructed, residual


def analyze_residual_structure(residual, embed_weights, tokenizer):
    """
    Analyze the structure of the residual.
    
    The residual contains dimensions we haven't captured yet.
    Can we find more semantic axes in it?
    """
    print()
    print("=" * 70)
    print("ANALYZING RESIDUAL STRUCTURE")
    print("=" * 70)
    print()
    
    # SVD of residual to find hidden dimensions
    print("SVD of residual...")
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    
    print(f"Top 10 singular values: {S[:10].round(2)}")
    
    # Check for φ-patterns in singular values
    ratios = S[:-1] / S[1:]
    phi_matches = []
    for i, r in enumerate(ratios[:20]):
        if abs(r - PHI) < 0.1:
            phi_matches.append((i, r, 'φ'))
        elif abs(r - PHI_INV) < 0.1:
            phi_matches.append((i, r, '1/φ'))
    
    print(f"\nφ-ratios in residual singular values: {len(phi_matches)}")
    for i, r, label in phi_matches[:5]:
        print(f"  S[{i}]/S[{i+1}] = {r:.4f} ≈ {label}")
    
    # How many dimensions to capture 90% of residual variance?
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    n_90 = np.searchsorted(cumvar, 0.9) + 1
    n_99 = np.searchsorted(cumvar, 0.99) + 1
    
    print(f"\nResidual dimensionality:")
    print(f"  90% variance: {n_90} dimensions")
    print(f"  99% variance: {n_99} dimensions")
    
    return S, Vt


def create_phi_decoder(phi_coords, axis_names, embed_weights, tokenizer):
    """
    Create a φ-decoder that maps coordinates back to words.
    
    This is the "comb" in the music box.
    """
    print()
    print("=" * 70)
    print("CREATING φ-DECODER (THE COMB)")
    print("=" * 70)
    print()
    
    def get_token_id(word):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if len(tokens) == 1 else None
    
    # Test the decoder on semantic transformations
    print("Testing φ-decoder on transformations:")
    print()
    
    test_transforms = [
        # (word, dimension, direction, expected)
        ("king", "gender", +1, "queen"),
        ("man", "gender", +1, "woman"),
        ("boy", "age", +1, "man"),
        ("good", "sentiment", -1, "bad"),
        ("big", "size", -1, "small"),
    ]
    
    for word, dim, direction, expected in test_transforms:
        word_id = get_token_id(word)
        expected_id = get_token_id(expected)
        
        if word_id is None or expected_id is None:
            continue
        
        # Get word's φ-coordinates
        word_coords = phi_coords[word_id].copy()
        
        # Apply transformation (move along dimension)
        dim_idx = axis_names.index(dim) if dim in axis_names else -1
        if dim_idx < 0:
            continue
        
        # How much to move? Use the average delta for this dimension
        # For now, use a fixed step
        step = 0.5 * direction  # Adjust based on coordinate scale
        word_coords[dim_idx] += step
        
        # Find nearest word in φ-space
        distances = np.linalg.norm(phi_coords - word_coords, axis=1)
        nearest_id = np.argmin(distances)
        nearest_word = tokenizer.decode([nearest_id])
        
        # Check if we got the expected word
        match = "✓" if nearest_id == expected_id else "✗"
        
        print(f"  {word} + {dim}({direction:+d}) → {nearest_word} (expected: {expected}) {match}")
    
    return phi_coords


def main():
    model, tokenizer = load_model()
    
    # Get embedding weights
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    print(f"Embedding shape: {embed_weights.shape}")
    
    # Step 1: Extract semantic axes (the drum's bumps)
    axes, pairs_used = extract_semantic_axes(embed_weights, tokenizer)
    
    # Step 2: Orthogonalize axes
    ortho_axes = orthogonalize_axes(axes)
    
    # Step 3: Project to φ-basis
    phi_coords, axis_names = project_to_phi_basis(embed_weights, ortho_axes, tokenizer)
    
    # Step 4: Test reconstruction
    reconstructed, residual = test_reconstruction(embed_weights, phi_coords, ortho_axes, axis_names)
    
    # Step 5: Analyze residual
    S, Vt = analyze_residual_structure(residual, embed_weights, tokenizer)
    
    # Step 6: Create and test φ-decoder
    create_phi_decoder(phi_coords, axis_names, embed_weights, tokenizer)
    
    print()
    print("=" * 70)
    print("SUMMARY: φ-BASIS EXTRACTION")
    print("=" * 70)
    print()
    print(f"Extracted {len(axis_names)} semantic dimensions:")
    for name in axis_names:
        print(f"  - {name}")
    print()
    print("These form the 'drum' - the structure of the space.")
    print("The residual contains additional dimensions not yet captured.")
    print()
    print("Next steps:")
    print("1. Add more semantic axes to capture more variance")
    print("2. Find the optimal φ-weighting for each axis")
    print("3. Test if transformations work in φ-space")


if __name__ == "__main__":
    main()
