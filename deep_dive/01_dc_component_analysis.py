#!/usr/bin/env python3
"""
Deep Dive Analysis: The DC Component Problem in Eigenspace Matching

The Problem:
When we decompose the similarity matrix via SVD/eigendecomposition, the first
eigenvalue (λ₀) dominates - it captures the "average similarity" across all
concepts. This is the DC component, analogous to the DC offset in signal processing.

In our 27-concept knowledge space:
- λ₀ = 24.33 (75% of total variance)
- λ₁ = 2.70 (8%)
- λ₂ = 1.52 (5%)
- ...

The DC component represents "how similar is everything to everything else" -
it's the baseline. The discriminative information is in the RESIDUAL dimensions.

This script analyzes the problem in detail.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig


def analyze_eigenvalue_spectrum():
    """Analyze the eigenvalue spectrum of the similarity matrix."""
    config = ChatConfig(debug=False)
    pipeline = ChatPipeline(config)
    ks = pipeline.knowledge_space
    
    n = len(ks._mappings)
    print(f"Number of concepts: {n}")
    print()
    
    # Build the full similarity matrix
    print("Building similarity matrix...")
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = ks._phi_importance_similarity(
                ks._mappings[i].input, 
                ks._mappings[j].input
            )
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = eigenvalues[::-1]  # Descending order
    eigenvectors = eigenvectors[:, ::-1]
    
    print("=" * 60)
    print("EIGENVALUE SPECTRUM")
    print("=" * 60)
    print()
    
    total = eigenvalues.sum()
    cumulative = 0
    for i, ev in enumerate(eigenvalues[:10]):
        cumulative += ev
        pct = 100 * ev / total
        cum_pct = 100 * cumulative / total
        bar = "█" * int(pct / 2)
        print(f"λ_{i:2d} = {ev:8.4f} ({pct:5.1f}%) cumulative: {cum_pct:5.1f}% {bar}")
    
    print()
    print(f"Total variance: {total:.4f}")
    print(f"λ₀ alone captures: {100 * eigenvalues[0] / total:.1f}%")
    print(f"λ₁-λ₇ capture: {100 * eigenvalues[1:8].sum() / total:.1f}%")
    
    return eigenvalues, eigenvectors, S, ks


def analyze_dc_component(eigenvalues, eigenvectors, S, ks):
    """Analyze what the DC component actually represents."""
    print()
    print("=" * 60)
    print("DC COMPONENT ANALYSIS")
    print("=" * 60)
    print()
    
    n = len(ks._mappings)
    
    # The first eigenvector (DC component)
    v0 = eigenvectors[:, 0]
    
    print("First eigenvector (DC component) loadings:")
    print(f"  Mean: {v0.mean():.4f}")
    print(f"  Std:  {v0.std():.4f}")
    print(f"  Min:  {v0.min():.4f}")
    print(f"  Max:  {v0.max():.4f}")
    print()
    
    # The DC component should be roughly uniform if all concepts
    # have similar "generality"
    print("Concept loadings on DC component (v₀):")
    loadings = list(enumerate(v0))
    loadings.sort(key=lambda x: -x[1])
    
    print("\nHighest DC loadings (most 'general' concepts):")
    for i, load in loadings[:5]:
        print(f"  {load:.4f}: {ks._mappings[i].input[:50]}...")
    
    print("\nLowest DC loadings (most 'specific' concepts):")
    for i, load in loadings[-5:]:
        print(f"  {load:.4f}: {ks._mappings[i].input[:50]}...")
    
    # Compute average similarity per concept (should correlate with DC loading)
    avg_sims = S.mean(axis=1)
    correlation = np.corrcoef(v0, avg_sims)[0, 1]
    
    print()
    print(f"Correlation between DC loading and average similarity: {correlation:.4f}")
    print("(High correlation confirms DC = 'average similarity to everything')")
    
    return v0


def analyze_discriminative_dimensions(eigenvalues, eigenvectors, ks):
    """Analyze what the discriminative dimensions encode."""
    print()
    print("=" * 60)
    print("DISCRIMINATIVE DIMENSIONS ANALYSIS")
    print("=" * 60)
    print()
    
    n = len(ks._mappings)
    
    # Analyze dimensions 1-3 (the most discriminative after DC)
    for dim in range(1, 4):
        v = eigenvectors[:, dim]
        
        print(f"Dimension {dim} (λ = {eigenvalues[dim]:.4f}):")
        print("-" * 40)
        
        # Sort concepts by this dimension
        loadings = list(enumerate(v))
        loadings.sort(key=lambda x: x[1])
        
        print("NEGATIVE pole:")
        for i, load in loadings[:3]:
            print(f"  {load:+.4f}: {ks._mappings[i].input[:45]}...")
        
        print("POSITIVE pole:")
        for i, load in loadings[-3:]:
            print(f"  {load:+.4f}: {ks._mappings[i].input[:45]}...")
        
        print()


def analyze_query_projection_problem(ks):
    """Analyze why query projection lands in the wrong place."""
    print()
    print("=" * 60)
    print("QUERY PROJECTION PROBLEM")
    print("=" * 60)
    print()
    
    n = len(ks._mappings)
    positions = np.array([m.position for m in ks._mappings])
    
    # The problematic query
    query = "who are you?"
    query_pos = ks.project_query(query, similarity_fn=ks._phi_importance_similarity)
    
    # Find identity concept
    identity_idx = None
    for i, m in enumerate(ks._mappings):
        if 'what can i do for you' in m.input.lower():
            identity_idx = i
            break
    
    print(f"Query: \"{query}\"")
    print(f"Query position: {query_pos[:4]}...")
    print(f"Identity position: {positions[identity_idx][:4]}...")
    print()
    
    # The projection is a weighted average of all concept positions
    # weighted by similarity to the query
    sims = np.array([ks._phi_importance_similarity(query, m.input) for m in ks._mappings])
    
    print("Similarity to query:")
    sim_sorted = sorted(enumerate(sims), key=lambda x: -x[1])
    for i, sim in sim_sorted[:5]:
        print(f"  {sim:.4f}: {ks._mappings[i].input[:50]}...")
    
    # The weighted average pulls toward the centroid
    weighted_pos = np.zeros_like(query_pos)
    total_weight = 0
    for i in range(n):
        weighted_pos += sims[i] * positions[i]
        total_weight += sims[i]
    weighted_pos /= total_weight
    
    print()
    print(f"Weighted average position: {weighted_pos[:4]}...")
    print(f"Actual query position:     {query_pos[:4]}...")
    print()
    
    # The problem: even though Identity has highest similarity,
    # the weighted average pulls toward the centroid
    centroid = positions.mean(axis=0)
    print(f"Centroid: {centroid[:4]}...")
    
    dist_to_centroid = np.linalg.norm(query_pos - centroid)
    dist_identity_to_centroid = np.linalg.norm(positions[identity_idx] - centroid)
    
    print()
    print(f"Query distance to centroid: {dist_to_centroid:.4f}")
    print(f"Identity distance to centroid: {dist_identity_to_centroid:.4f}")
    print()
    print("The query projects CLOSER to the centroid than Identity is!")
    print("This is the DC component problem: projection averages toward the center.")


def main():
    print("=" * 60)
    print("DEEP DIVE: THE DC COMPONENT PROBLEM")
    print("=" * 60)
    print()
    
    eigenvalues, eigenvectors, S, ks = analyze_eigenvalue_spectrum()
    v0 = analyze_dc_component(eigenvalues, eigenvectors, S, ks)
    analyze_discriminative_dimensions(eigenvalues, eigenvectors, ks)
    analyze_query_projection_problem(ks)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The DC Component Problem:

1. WHAT IT IS:
   - The first eigenvalue (λ₀) captures "average similarity"
   - It's the baseline that all concepts share
   - In our case, λ₀ = 24.33 (75% of variance)

2. WHY IT'S A PROBLEM:
   - Query projection is a weighted average of concept positions
   - This pulls queries toward the CENTROID (center of mass)
   - The centroid is dominated by the DC component
   - Discriminative information (λ₁-λ₇) gets drowned out

3. THE GEOMETRIC PICTURE:
   - All concepts live in an 8D eigenspace
   - Dimension 0 (DC) spreads concepts along a line toward "generality"
   - Dimensions 1-7 spread concepts by domain/topic
   - Queries project to a weighted average, which is near the centroid
   - The centroid is in the "general" region, not the "specific" region

4. WHY SQRT-INVERSE WEIGHTING HELPS:
   - It de-emphasizes dimension 0 (DC)
   - It amplifies dimensions 1-7 (discriminative)
   - This lets the domain separation shine through
""")


if __name__ == "__main__":
    main()
