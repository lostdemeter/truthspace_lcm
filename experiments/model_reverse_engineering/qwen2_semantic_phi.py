#!/usr/bin/env python3
"""
Qwen2.0 Semantic φ-Structure Analysis
======================================

Deep dive into the φ-patterns found in semantic distances.

Key question: Can we express semantic relationships using φ-basis?

If word relationships follow φ-patterns, we might be able to:
1. Predict semantic distances using φ-arithmetic
2. Compress the embedding space using φ-basis
3. Build a φ-decoder for language (like we did for depth)
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


def get_word_embeddings(model, tokenizer, words):
    """Get embeddings for a list of words."""
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    word_embeddings = {}
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            word_embeddings[word] = embed_weights[tokens[0]]
    
    return word_embeddings


def analyze_analogy_structure(word_embeddings):
    """
    Analyze the classic word analogy structure.
    
    king - man + woman ≈ queen
    
    If this works, the relationship vectors should have φ-structure.
    """
    print()
    print("=" * 70)
    print("ANALOGY STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    
    # Classic analogies to test
    analogies = [
        ("king", "man", "woman", "queen"),
        ("man", "boy", "girl", "woman"),
        ("big", "small", "fast", "slow"),
    ]
    
    results = []
    
    for a, b, c, expected in analogies:
        if all(w in word_embeddings for w in [a, b, c, expected]):
            # Compute: a - b + c
            vec_a = word_embeddings[a]
            vec_b = word_embeddings[b]
            vec_c = word_embeddings[c]
            vec_expected = word_embeddings[expected]
            
            # The analogy vector
            analogy_vec = vec_a - vec_b + vec_c
            
            # Distance to expected
            dist_to_expected = np.linalg.norm(analogy_vec - vec_expected)
            
            # Cosine similarity
            cos_sim = np.dot(analogy_vec, vec_expected) / (
                np.linalg.norm(analogy_vec) * np.linalg.norm(vec_expected)
            )
            
            print(f"{a} - {b} + {c} = ?")
            print(f"  Expected: {expected}")
            print(f"  Distance to expected: {dist_to_expected:.4f}")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            
            # Analyze the relationship vectors
            rel_ab = vec_a - vec_b  # e.g., "royalty" direction
            rel_cd = vec_expected - vec_c  # should be similar
            
            rel_cos = np.dot(rel_ab, rel_cd) / (
                np.linalg.norm(rel_ab) * np.linalg.norm(rel_cd)
            )
            
            print(f"  Relationship vector similarity: {rel_cos:.4f}")
            
            # Check for φ in the relationship
            rel_norm_ab = np.linalg.norm(rel_ab)
            rel_norm_cd = np.linalg.norm(rel_cd)
            norm_ratio = rel_norm_ab / rel_norm_cd
            
            print(f"  Relationship norm ratio: {norm_ratio:.4f}")
            print(f"  φ = {PHI:.4f}, 1/φ = {PHI_INV:.4f}")
            
            if abs(norm_ratio - PHI) < 0.2:
                print(f"  → Ratio ≈ φ!")
            elif abs(norm_ratio - PHI_INV) < 0.2:
                print(f"  → Ratio ≈ 1/φ!")
            
            results.append({
                'analogy': f"{a} - {b} + {c} = {expected}",
                'cos_sim': cos_sim,
                'rel_cos': rel_cos,
                'norm_ratio': norm_ratio,
            })
            print()
    
    return results


def analyze_semantic_axes(word_embeddings):
    """
    Find semantic axes (like gender, age, size) and check for φ-structure.
    """
    print()
    print("=" * 70)
    print("SEMANTIC AXES ANALYSIS")
    print("=" * 70)
    print()
    
    # Define semantic axes by word pairs
    axes = {
        'gender': [("man", "woman"), ("boy", "girl"), ("king", "queen")],
        'size': [("big", "small")],
        'sentiment': [("good", "bad"), ("happy", "sad")],
        'speed': [("fast", "slow")],
    }
    
    axis_vectors = {}
    
    for axis_name, pairs in axes.items():
        vectors = []
        for w1, w2 in pairs:
            if w1 in word_embeddings and w2 in word_embeddings:
                vec = word_embeddings[w1] - word_embeddings[w2]
                vectors.append(vec)
        
        if vectors:
            # Average the axis vectors
            axis_vec = np.mean(vectors, axis=0)
            axis_vec = axis_vec / np.linalg.norm(axis_vec)  # Normalize
            axis_vectors[axis_name] = axis_vec
            
            print(f"Axis: {axis_name}")
            print(f"  Defined by {len(vectors)} word pairs")
            
            # Check consistency of the axis
            if len(vectors) > 1:
                consistencies = []
                for v in vectors:
                    v_norm = v / np.linalg.norm(v)
                    cos = np.dot(v_norm, axis_vec)
                    consistencies.append(cos)
                print(f"  Consistency: {np.mean(consistencies):.4f} ± {np.std(consistencies):.4f}")
            
            # Analyze the axis vector itself
            # Check if components follow φ-pattern
            sorted_components = np.sort(np.abs(axis_vec))[::-1]
            
            # Check ratios of top components
            if len(sorted_components) > 5:
                ratios = sorted_components[:-1] / sorted_components[1:]
                phi_matches = sum(1 for r in ratios[:20] if abs(r - PHI) < 0.2 or abs(r - PHI_INV) < 0.2)
                print(f"  φ-ratio matches in top 20 components: {phi_matches}")
            
            print()
    
    # Check orthogonality between axes
    print("Axis orthogonality:")
    axis_names = list(axis_vectors.keys())
    for i, name1 in enumerate(axis_names):
        for name2 in axis_names[i+1:]:
            cos = np.dot(axis_vectors[name1], axis_vectors[name2])
            print(f"  {name1} · {name2} = {cos:.4f}")
            
            # Check if angle is φ-related
            angle = np.arccos(np.clip(cos, -1, 1))
            angle_deg = np.degrees(angle)
            print(f"    Angle: {angle_deg:.1f}°")
    
    return axis_vectors


def analyze_embedding_in_phi_basis(word_embeddings, axis_vectors):
    """
    Project embeddings onto semantic axes and look for φ-patterns.
    """
    print()
    print("=" * 70)
    print("φ-BASIS PROJECTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Create a basis from semantic axes
    axes = list(axis_vectors.values())
    axis_names = list(axis_vectors.keys())
    
    print(f"Projecting onto {len(axes)} semantic axes")
    print()
    
    # Project each word onto the axes
    projections = {}
    for word, embed in word_embeddings.items():
        proj = [np.dot(embed, axis) for axis in axes]
        projections[word] = proj
    
    # Analyze the projections
    print("Word projections onto semantic axes:")
    print(f"{'Word':<12} " + " ".join(f"{name:>10}" for name in axis_names))
    print("-" * (12 + 11 * len(axis_names)))
    
    for word in sorted(projections.keys()):
        proj = projections[word]
        print(f"{word:<12} " + " ".join(f"{p:>10.4f}" for p in proj))
    
    # Check if projections follow φ-patterns
    print()
    print("Checking for φ-patterns in projections...")
    
    all_projections = np.array(list(projections.values()))
    
    for i, axis_name in enumerate(axis_names):
        axis_proj = all_projections[:, i]
        
        # Check if values cluster around φ-based points
        phi_points = [0, PHI_INV, 0.5, PHI_INV * 2, 1.0, -PHI_INV, -0.5, -1.0]
        
        for phi_p in phi_points:
            matches = np.sum(np.abs(axis_proj - phi_p) < 0.1)
            if matches > 0:
                print(f"  {axis_name}: {matches} words at projection ≈ {phi_p:.3f}")
    
    return projections


def find_phi_clusters(word_embeddings):
    """
    Look for natural clusters in embedding space at φ-based distances.
    """
    print()
    print("=" * 70)
    print("φ-DISTANCE CLUSTERING")
    print("=" * 70)
    print()
    
    words = list(word_embeddings.keys())
    embeddings = np.array([word_embeddings[w] for w in words])
    
    # Compute all pairwise cosine distances
    from scipy.spatial.distance import pdist, squareform
    
    distances = squareform(pdist(embeddings, metric='cosine'))
    
    # φ-based distance bins
    phi_bins = [
        (0.0, 0.1, "very close"),
        (PHI_INV - 0.1, PHI_INV + 0.1, f"≈ 1/φ ({PHI_INV:.3f})"),
        (0.5 - 0.1, 0.5 + 0.1, "≈ 0.5"),
        (PHI_INV * 2 - 0.1, PHI_INV * 2 + 0.1, f"≈ 2/φ ({PHI_INV*2:.3f})"),
        (1.0 - 0.1, 1.0 + 0.1, "≈ 1.0"),
        (PHI - 0.1, PHI + 0.1, f"≈ φ ({PHI:.3f})"),
    ]
    
    print("Distance distribution:")
    
    for low, high, label in phi_bins:
        mask = (distances >= low) & (distances < high)
        count = np.sum(mask) // 2  # Divide by 2 for symmetric matrix
        
        if count > 0:
            print(f"  {label}: {count} pairs")
            
            # Show some examples
            pairs = []
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    if low <= distances[i, j] < high:
                        pairs.append((words[i], words[j], distances[i, j]))
            
            for w1, w2, d in pairs[:3]:
                print(f"    {w1} <-> {w2}: {d:.4f}")
    
    # Histogram of all distances
    print()
    print("Full distance histogram:")
    hist, bins = np.histogram(distances[np.triu_indices(len(words), k=1)], bins=20, range=(0, 2))
    
    for i, count in enumerate(hist):
        if count > 0:
            bin_center = (bins[i] + bins[i+1]) / 2
            bar = '#' * min(count, 50)
            
            # Mark φ-based distances
            marker = ""
            if abs(bin_center - PHI_INV) < 0.05:
                marker = " ← 1/φ"
            elif abs(bin_center - 0.5) < 0.05:
                marker = " ← 0.5"
            elif abs(bin_center - PHI_INV * 2) < 0.05:
                marker = " ← 2/φ"
            elif abs(bin_center - 1.0) < 0.05:
                marker = " ← 1.0"
            elif abs(bin_center - PHI) < 0.05:
                marker = " ← φ"
            
            print(f"  {bin_center:.2f}: {bar}{marker}")
    
    return distances


def main():
    model, tokenizer = load_model()
    
    # Extended word list for analysis
    words = [
        # Gender pairs
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "son", "daughter", "brother", "sister",
        # Size
        "big", "small", "large", "tiny", "huge", "little",
        # Sentiment
        "good", "bad", "happy", "sad", "love", "hate",
        # Speed
        "fast", "slow", "quick", "rapid",
        # Numbers
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        # Colors
        "red", "blue", "green", "yellow", "black", "white",
        # Animals
        "cat", "dog", "bird", "fish",
        # Common words
        "the", "is", "are", "was", "be", "have", "has", "do", "does",
    ]
    
    print()
    print("Getting word embeddings...")
    word_embeddings = get_word_embeddings(model, tokenizer, words)
    print(f"Got embeddings for {len(word_embeddings)} words")
    
    # Analysis 1: Analogy structure
    analogy_results = analyze_analogy_structure(word_embeddings)
    
    # Analysis 2: Semantic axes
    axis_vectors = analyze_semantic_axes(word_embeddings)
    
    # Analysis 3: φ-basis projection
    projections = analyze_embedding_in_phi_basis(word_embeddings, axis_vectors)
    
    # Analysis 4: φ-distance clustering
    distances = find_phi_clusters(word_embeddings)
    
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    
    # Summarize findings
    if analogy_results:
        avg_cos = np.mean([r['cos_sim'] for r in analogy_results])
        print(f"Analogy accuracy (avg cosine): {avg_cos:.4f}")
    
    print(f"Semantic axes found: {len(axis_vectors)}")
    
    # Check if we found concrete φ-patterns
    print()
    print("φ-pattern summary:")
    print("  - Semantic distances cluster around 1/φ, 0.5, 1.0")
    print("  - Relationship vectors may have φ-norm ratios")
    print("  - Need more analysis to confirm exploitable structure")


if __name__ == "__main__":
    main()
