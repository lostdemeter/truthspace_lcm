#!/usr/bin/env python3
"""
Deep Dive Analysis: Possible Solutions to the DC Component Problem

We've identified the problem: the DC component (λ₀) dominates and pulls
queries toward the centroid. Now let's systematically analyze solutions.

Solutions to explore:
1. Sqrt-inverse eigenvalue weighting (current fix)
2. DC removal (subtract the DC component entirely)
3. Whitening (normalize by eigenvalue)
4. Projection refinement (iterative projection)
5. Attractor-based snapping
"""

import numpy as np
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig


def get_test_data():
    """Get the knowledge space and test queries."""
    config = ChatConfig(debug=False)
    pipeline = ChatPipeline(config)
    ks = pipeline.knowledge_space
    
    n = len(ks._mappings)
    positions = np.array([m.position for m in ks._mappings])
    
    # Build similarity matrix and get eigenvalues
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = ks._phi_importance_similarity(
                ks._mappings[i].input, 
                ks._mappings[j].input
            )
    
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    queries = [
        ('what is physics?', 'Physics is the natural'),
        ('what is science?', 'Science is the systematic'),
        ('who are you?', 'HyperChat'),
        ('what can you do?', 'HyperChat'),
        ('thank you', 'welcome'),
        ('hello', 'Hello'),
        ('tell me about machine learning', 'Machine learning'),
        ('what is python?', 'Python'),
    ]
    
    return ks, positions, eigenvalues, eigenvectors, queries


def test_method(name, distance_fn, ks, positions, queries):
    """Test a distance method and return accuracy."""
    correct = 0
    for query, expected in queries:
        query_pos = ks.project_query(query, similarity_fn=ks._phi_importance_similarity)
        
        dists = [distance_fn(query_pos, positions[i]) for i in range(len(positions))]
        best_idx = np.argmin(dists)
        response = ks._mappings[best_idx].input
        
        if expected.lower() in response[:50].lower():
            correct += 1
    
    return correct, len(queries)


def analyze_solutions():
    """Analyze different solutions to the DC component problem."""
    print("Loading data...")
    ks, positions, eigenvalues, eigenvectors, queries = get_test_data()
    n = len(positions)
    ndim = positions.shape[1]
    
    print(f"Concepts: {n}, Dimensions: {ndim}")
    print(f"Eigenvalues: {eigenvalues[:ndim]}")
    print()
    
    print("=" * 70)
    print("SOLUTION COMPARISON")
    print("=" * 70)
    print()
    
    results = []
    
    # Solution 0: Baseline (unweighted Euclidean)
    def euclidean(q, p):
        return np.linalg.norm(q - p)
    
    correct, total = test_method("Euclidean", euclidean, ks, positions, queries)
    results.append(("Baseline (Euclidean)", correct, total))
    print(f"0. Baseline (Euclidean):        {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 1: Sqrt-inverse eigenvalue weighting (current fix)
    weights_sqrt_inv = 1.0 / np.sqrt(eigenvalues[:ndim] + 1e-10)
    weights_sqrt_inv = weights_sqrt_inv / weights_sqrt_inv.sum()
    
    def sqrt_inv_weighted(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_sqrt_inv * diff**2))
    
    correct, total = test_method("Sqrt-Inv", sqrt_inv_weighted, ks, positions, queries)
    results.append(("Sqrt-Inverse Weighting", correct, total))
    print(f"1. Sqrt-Inverse Weighting:      {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 2: DC Removal (set dimension 0 weight to 0)
    weights_no_dc = np.ones(ndim)
    weights_no_dc[0] = 0
    weights_no_dc = weights_no_dc / weights_no_dc.sum()
    
    def no_dc(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_no_dc * diff**2))
    
    correct, total = test_method("No DC", no_dc, ks, positions, queries)
    results.append(("DC Removal", correct, total))
    print(f"2. DC Removal:                  {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 3: Whitening (divide by sqrt of eigenvalue)
    # This normalizes each dimension to unit variance
    weights_whitening = 1.0 / eigenvalues[:ndim]
    weights_whitening = weights_whitening / weights_whitening.sum()
    
    def whitened(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_whitening * diff**2))
    
    correct, total = test_method("Whitening", whitened, ks, positions, queries)
    results.append(("Whitening (1/λ)", correct, total))
    print(f"3. Whitening (1/λ):             {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 4: Log-inverse weighting
    weights_log_inv = 1.0 / np.log(eigenvalues[:ndim] + np.e)
    weights_log_inv = weights_log_inv / weights_log_inv.sum()
    
    def log_inv_weighted(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_log_inv * diff**2))
    
    correct, total = test_method("Log-Inv", log_inv_weighted, ks, positions, queries)
    results.append(("Log-Inverse Weighting", correct, total))
    print(f"4. Log-Inverse Weighting:       {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 5: Exponential decay from DC
    # Weight dimension i by exp(-i/τ) where τ controls decay rate
    tau = 2.0
    weights_exp = np.exp(-np.arange(ndim) / tau)
    weights_exp[0] = 0.1  # Reduce DC but don't eliminate
    weights_exp = weights_exp / weights_exp.sum()
    
    def exp_decay(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_exp * diff**2))
    
    correct, total = test_method("Exp Decay", exp_decay, ks, positions, queries)
    results.append(("Exponential Decay", correct, total))
    print(f"5. Exponential Decay:           {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 6: Only middle dimensions (1-4)
    weights_mid = np.zeros(ndim)
    weights_mid[1:5] = 1.0
    weights_mid = weights_mid / weights_mid.sum()
    
    def mid_only(q, p):
        diff = q - p
        return np.sqrt(np.sum(weights_mid * diff**2))
    
    correct, total = test_method("Mid Only", mid_only, ks, positions, queries)
    results.append(("Middle Dims Only (1-4)", correct, total))
    print(f"6. Middle Dims Only (1-4):      {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 7: Cosine similarity (angle-based, ignores magnitude)
    def cosine_dist(q, p):
        dot = np.dot(q, p)
        norm_q = np.linalg.norm(q)
        norm_p = np.linalg.norm(p)
        if norm_q == 0 or norm_p == 0:
            return 1.0
        cos_sim = dot / (norm_q * norm_p)
        return 1.0 - cos_sim  # Convert to distance
    
    correct, total = test_method("Cosine", cosine_dist, ks, positions, queries)
    results.append(("Cosine Distance", correct, total))
    print(f"7. Cosine Distance:             {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Solution 8: Mahalanobis-like (full inverse covariance)
    # This is the "proper" statistical distance
    cov = np.cov(positions.T)
    try:
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(ndim))
        
        def mahalanobis(q, p):
            diff = q - p
            return np.sqrt(diff @ cov_inv @ diff)
        
        correct, total = test_method("Mahalanobis", mahalanobis, ks, positions, queries)
        results.append(("Mahalanobis", correct, total))
        print(f"8. Mahalanobis:                 {correct}/{total} ({100*correct/total:.0f}%)")
    except:
        print(f"8. Mahalanobis:                 FAILED (singular covariance)")
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    # Sort by accuracy
    results.sort(key=lambda x: -x[1])
    
    print("Ranked by accuracy:")
    for i, (name, correct, total) in enumerate(results):
        print(f"  {i+1}. {name}: {correct}/{total} ({100*correct/total:.0f}%)")
    
    return results


def analyze_why_solutions_work():
    """Deeper analysis of why certain solutions work."""
    print()
    print("=" * 70)
    print("WHY DO THESE SOLUTIONS WORK?")
    print("=" * 70)
    print()
    
    ks, positions, eigenvalues, eigenvectors, queries = get_test_data()
    ndim = positions.shape[1]
    
    # The key insight: DC component is the "average similarity" direction
    # All concepts have SOME similarity to each other (they're all text)
    # The DC component captures this baseline
    
    print("Eigenvalue distribution:")
    total = eigenvalues[:ndim].sum()
    for i in range(ndim):
        pct = 100 * eigenvalues[i] / total
        print(f"  λ_{i}: {eigenvalues[i]:.4f} ({pct:.1f}%)")
    
    print()
    print("The DC component (λ₀) captures 'how similar is everything'")
    print("Discriminative dimensions (λ₁-λ₇) capture 'what makes things different'")
    print()
    
    # Compute the "discriminative power" of each dimension
    # = how much does this dimension separate the concepts?
    print("Discriminative power per dimension:")
    for i in range(ndim):
        spread = positions[:, i].std()
        print(f"  Dim {i}: std = {spread:.4f}")
    
    print()
    print("Key insight: The problem isn't the eigenspace itself,")
    print("it's that QUERY PROJECTION pulls toward the centroid.")
    print()
    print("Solutions that work all do one thing:")
    print("  → De-emphasize the DC component in the DISTANCE metric")
    print()
    print("This lets the discriminative dimensions determine the match,")
    print("even though the query is projected near the centroid.")


def main():
    print("=" * 70)
    print("DEEP DIVE: SOLUTION ANALYSIS")
    print("=" * 70)
    print()
    
    results = analyze_solutions()
    analyze_why_solutions_work()
    
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on this analysis:

1. SQRT-INVERSE WEIGHTING is a good balance:
   - Simple to implement
   - Doesn't completely remove DC (which has some signal)
   - Emphasizes discriminative dimensions proportionally

2. DC REMOVAL is more aggressive:
   - Completely ignores the "generality" dimension
   - May lose some signal in edge cases
   - But simpler conceptually

3. WHITENING (1/λ) is the "statistically correct" approach:
   - Normalizes each dimension to unit variance
   - But may over-amplify noisy low-eigenvalue dimensions

4. COSINE DISTANCE is angle-based:
   - Ignores magnitude entirely
   - Good if direction is more important than distance
   - But loses information about "how far" in eigenspace

RECOMMENDATION: Stick with sqrt-inverse weighting as the default,
but consider DC removal for cases where generality is truly irrelevant.
""")


if __name__ == "__main__":
    main()
