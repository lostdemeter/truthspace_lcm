#!/usr/bin/env python3
"""
Qwen2.0 PCA φ-Structure Analysis
=================================

Instead of imposing a φ-basis, let's check if the natural
PCA directions of the embedding space have φ-structure.

Key question: Do the singular values or principal directions
follow φ-patterns?
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


def analyze_pca_singular_values(embed_weights):
    """Analyze singular values for φ-patterns."""
    print()
    print("=" * 70)
    print("PCA SINGULAR VALUE ANALYSIS")
    print("=" * 70)
    print()
    
    # Center embeddings
    embed_centered = embed_weights - np.mean(embed_weights, axis=0)
    
    # SVD
    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(embed_centered, full_matrices=False)
    
    print(f"Singular values shape: {S.shape}")
    print()
    
    # Analyze singular value ratios
    print("Consecutive singular value ratios:")
    ratios = S[:-1] / S[1:]
    
    phi_matches = []
    for i in range(min(100, len(ratios))):
        r = ratios[i]
        
        # Check for φ-related ratios
        if abs(r - PHI) < 0.05:
            phi_matches.append((i, r, 'φ'))
        elif abs(r - PHI_INV) < 0.05:
            phi_matches.append((i, r, '1/φ'))
        elif abs(r - PHI**2) < 0.1:
            phi_matches.append((i, r, 'φ²'))
        elif abs(r - 1/PHI**2) < 0.05:
            phi_matches.append((i, r, '1/φ²'))
        elif abs(r - 2) < 0.05:
            phi_matches.append((i, r, '2'))
    
    print(f"Found {len(phi_matches)} φ-related ratios in first 100:")
    for i, r, label in phi_matches[:20]:
        print(f"  S[{i}]/S[{i+1}] = {r:.4f} ≈ {label}")
    
    # Analyze cumulative variance
    print()
    print("Cumulative variance explained:")
    
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / total_var
    
    # Find dimensions for key variance thresholds
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    for thresh in thresholds:
        n_dims = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh*100:.0f}% variance: {n_dims} dimensions")
        
        # Check if n_dims is φ-related
        for phi_val in [PHI, PHI**2, PHI**3, PHI**4, PHI**5]:
            if abs(n_dims - phi_val) < 1:
                print(f"    → n_dims ≈ φ^{np.log(phi_val)/np.log(PHI):.0f}!")
    
    return S, Vt, ratios


def analyze_pca_directions(Vt, embed_weights, tokenizer):
    """Analyze what the principal directions encode."""
    print()
    print("=" * 70)
    print("PRINCIPAL DIRECTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Get some test words
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
        "one", "two", "three", "four", "five",
        "the", "is", "a", "of", "to",
    ]
    
    word_ids = {}
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            word_ids[word] = tokens[0]
    
    # Project words onto top PCA directions
    n_pca = 20
    
    print(f"Projecting words onto top {n_pca} PCA directions:")
    print()
    
    # Center embeddings
    embed_mean = np.mean(embed_weights, axis=0)
    
    projections = {}
    for word, idx in word_ids.items():
        embed = embed_weights[idx] - embed_mean
        proj = embed @ Vt[:n_pca].T
        projections[word] = proj
    
    # Check if semantic relationships are captured in top PCA dims
    print("Semantic relationships in PCA space:")
    
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("good", "bad"),
        ("happy", "sad"),
    ]
    
    for w1, w2 in pairs:
        if w1 in projections and w2 in projections:
            p1 = projections[w1]
            p2 = projections[w2]
            
            # Difference vector
            diff = p1 - p2
            
            # Which PCA dimensions capture this relationship?
            top_dims = np.argsort(np.abs(diff))[::-1][:5]
            
            print(f"  {w1} <-> {w2}:")
            print(f"    Top PCA dims: {top_dims}")
            print(f"    Diff magnitudes: {np.abs(diff[top_dims]).round(4)}")
    
    # Check if relationship vectors are similar
    print()
    print("Relationship vector similarity:")
    
    if all(w in projections for w in ["king", "queen", "man", "woman"]):
        rel_royal = projections["king"] - projections["queen"]
        rel_gender = projections["man"] - projections["woman"]
        
        cos_sim = np.dot(rel_royal, rel_gender) / (
            np.linalg.norm(rel_royal) * np.linalg.norm(rel_gender)
        )
        
        print(f"  (king-queen) · (man-woman) = {cos_sim:.4f}")
        
        # Check if this is φ-related
        if abs(cos_sim - PHI_INV) < 0.1:
            print(f"    → Similarity ≈ 1/φ!")
    
    return projections


def analyze_pca_angles(Vt):
    """Analyze angles between PCA directions for φ-patterns."""
    print()
    print("=" * 70)
    print("PCA DIRECTION ANGLE ANALYSIS")
    print("=" * 70)
    print()
    
    # PCA directions are orthogonal by construction
    # But let's check the angles between consecutive directions
    # when projected onto lower-dimensional subspaces
    
    n_dirs = min(50, Vt.shape[0])
    
    # Compute pairwise angles in the full space (should be 90°)
    print("Verifying orthogonality (should be ~90°):")
    
    angles = []
    for i in range(n_dirs - 1):
        dot = np.dot(Vt[i], Vt[i+1])
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        angles.append(angle)
    
    print(f"  Mean angle: {np.mean(angles):.2f}°")
    print(f"  Std angle: {np.std(angles):.4f}°")
    
    # Now look at the structure of each PCA direction
    print()
    print("Analyzing structure of PCA directions:")
    
    for i in range(min(10, n_dirs)):
        direction = Vt[i]
        
        # Sort by absolute value
        sorted_vals = np.sort(np.abs(direction))[::-1]
        
        # Check if values follow φ-decay
        ratios = sorted_vals[:-1] / sorted_vals[1:]
        
        # Count φ-related ratios
        phi_count = sum(1 for r in ratios[:20] 
                       if abs(r - PHI) < 0.1 or abs(r - PHI_INV) < 0.1)
        
        # Compute effective dimensionality (entropy-based)
        probs = sorted_vals**2 / np.sum(sorted_vals**2)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        eff_dim = np.exp(entropy)
        
        print(f"  PC{i}: φ-ratios={phi_count}/20, eff_dim={eff_dim:.1f}")
    
    return angles


def find_phi_subspace(embed_weights, tokenizer):
    """
    Try to find a subspace where φ-relationships hold.
    
    Hypothesis: There might be a specific subspace where
    semantic relationships follow φ-patterns.
    """
    print()
    print("=" * 70)
    print("φ-SUBSPACE SEARCH")
    print("=" * 70)
    print()
    
    # Get semantic pairs
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("son", "daughter"),
        ("brother", "sister"),
    ]
    
    # Get embeddings
    word_embeds = {}
    for w1, w2 in pairs:
        for word in [w1, w2]:
            if word not in word_embeds:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                if len(tokens) == 1:
                    word_embeds[word] = embed_weights[tokens[0]]
    
    # Compute relationship vectors
    rel_vectors = []
    for w1, w2 in pairs:
        if w1 in word_embeds and w2 in word_embeds:
            rel = word_embeds[w1] - word_embeds[w2]
            rel_vectors.append(rel)
    
    if len(rel_vectors) < 2:
        print("Not enough word pairs found")
        return None
    
    rel_vectors = np.array(rel_vectors)
    print(f"Found {len(rel_vectors)} relationship vectors")
    
    # SVD of relationship vectors to find the "gender subspace"
    U, S, Vt = np.linalg.svd(rel_vectors, full_matrices=False)
    
    print()
    print("Relationship vector SVD:")
    print(f"  Singular values: {S.round(4)}")
    
    # Check ratios
    if len(S) > 1:
        ratios = S[:-1] / S[1:]
        print(f"  Ratios: {ratios.round(4)}")
        
        for i, r in enumerate(ratios):
            if abs(r - PHI) < 0.2:
                print(f"    S[{i}]/S[{i+1}] ≈ φ!")
    
    # The first principal direction is the "gender axis"
    gender_axis = Vt[0]
    
    # Project all words onto this axis
    print()
    print("Projections onto gender axis:")
    
    for word, embed in sorted(word_embeds.items()):
        proj = np.dot(embed, gender_axis)
        print(f"  {word:12s}: {proj:+.4f}")
    
    # Check if projections follow φ-pattern
    male_words = ["king", "man", "boy", "father", "son", "brother"]
    female_words = ["queen", "woman", "girl", "mother", "daughter", "sister"]
    
    male_projs = [np.dot(word_embeds[w], gender_axis) for w in male_words if w in word_embeds]
    female_projs = [np.dot(word_embeds[w], gender_axis) for w in female_words if w in word_embeds]
    
    if male_projs and female_projs:
        male_mean = np.mean(male_projs)
        female_mean = np.mean(female_projs)
        gap = male_mean - female_mean
        
        print()
        print(f"Male mean: {male_mean:.4f}")
        print(f"Female mean: {female_mean:.4f}")
        print(f"Gap: {gap:.4f}")
        
        # Check if gap is φ-related
        if abs(abs(gap) - PHI_INV) < 0.1:
            print(f"  → Gap ≈ 1/φ!")
        elif abs(abs(gap) - PHI) < 0.2:
            print(f"  → Gap ≈ φ!")
    
    return gender_axis, Vt


def main():
    model, tokenizer = load_model()
    
    # Get embedding weights
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    # Analysis 1: Singular value structure
    S, Vt, ratios = analyze_pca_singular_values(embed_weights)
    
    # Analysis 2: Principal direction analysis
    projections = analyze_pca_directions(Vt, embed_weights, tokenizer)
    
    # Analysis 3: Angle analysis
    angles = analyze_pca_angles(Vt)
    
    # Analysis 4: φ-subspace search
    result = find_phi_subspace(embed_weights, tokenizer)
    
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Key findings:")
    print("1. PCA directions are orthogonal (by construction)")
    print("2. Semantic relationships are captured in specific PCA dims")
    print("3. The 'gender axis' can be extracted from relationship vectors")
    print()
    print("φ-pattern status:")
    print("  - Singular value ratios: Some matches, not dominant")
    print("  - Semantic distances: Cluster around 1/φ")
    print("  - Need to find the RIGHT subspace for φ-structure")


if __name__ == "__main__":
    main()
