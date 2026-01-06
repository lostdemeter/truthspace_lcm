# Deep Dive: The DC Component Problem in Eigenspace Matching

## Executive Summary

When matching queries to concepts in eigenspace, the **DC component** (first eigenvalue) dominates and pulls queries toward the centroid, reducing discriminative power. **Sqrt-inverse eigenvalue weighting** solves this by emphasizing discriminative dimensions, improving accuracy from **75% → 88%**.

## The Problem

### What is the DC Component?

The DC component (λ₀) is the first eigenvalue of the similarity matrix. It captures:
- **"Average similarity"** - how similar everything is to everything else
- **Centrality** - concepts with high DC loading are "hubs" (similar to many concepts)
- **Baseline structure** - the common language patterns all concepts share

In our 27-concept knowledge space:
```
λ₀ = 10.81 (58% of variance)  ← DC component
λ₁ = 2.83  (15%)              ← Discriminative
λ₂ = 1.57  (8%)               ← Discriminative
...
```

### Why is it a Problem?

Query projection is a **weighted average** of concept positions:
```
query_position = Σ (similarity[i] × position[i]) / Σ similarity[i]
```

This pulls queries toward the **centroid** (center of mass), which is dominated by the DC component. The discriminative information in dimensions 1-7 gets drowned out.

**Evidence:**
- Query "who are you?" projects 0.117 from centroid
- Identity concept is 0.204 from centroid
- Query is CLOSER to centroid than the correct answer!

## The Solution

### Sqrt-Inverse Eigenvalue Weighting

Weight each dimension by `1/sqrt(λ)` when computing distance:

```python
weights = 1.0 / np.sqrt(eigenvalues + 1e-10)
weights = weights / weights.sum()

distance = sqrt(Σ weights[i] × (query[i] - concept[i])²)
```

**Results:**
| Method | Accuracy |
|--------|----------|
| Baseline (Euclidean) | 75% |
| **Sqrt-Inverse Weighting** | **88%** |
| DC Removal | 75% |
| Whitening (1/λ) | 50% |
| Cosine Distance | 75% |

### Why Does It Work?

1. **De-emphasizes DC**: The DC dimension gets weight 1/sqrt(10.81) ≈ 0.30
2. **Amplifies discriminative dims**: Dimension 7 gets weight 1/sqrt(0.46) ≈ 1.47
3. **Balanced**: Unlike full whitening (1/λ), doesn't over-amplify noise

## Theoretical Foundation

### 1. Perron-Frobenius Theorem
The DC component is the **Perron-Frobenius eigenvector** - it represents centrality in the similarity graph. Correlation with degree centrality: **0.9984**.

### 2. Spectral Clustering Connection
In spectral clustering, the first eigenvector is **ignored** because it's constant for connected graphs. The Fiedler vector (second eigenvector) captures cluster structure. Our discriminative dimensions (λ₁-λ₇) are analogous to Fiedler vectors.

### 3. Whitening Theory
Positions are computed as `P = V @ sqrt(Λ)`. Weighting by `1/sqrt(λ)` **undoes** this scaling, normalizing each dimension's contribution. This is a form of "whitening" that doesn't over-amplify noise.

### 4. Connection to φ-Encoding
The eigenvalue decay follows a power law: `λ_k ∝ k^(-1.48)`. This reflects the self-similar structure induced by φ-importance weighting. The DC captures baseline language structure; discriminative dimensions capture **meaning**.

## Scaling Behavior

**Good news: The problem doesn't get worse at scale.**

| N concepts | DC Dominance (λ₀/Σλ) |
|------------|---------------------|
| 10 | 42% |
| 27 | 41% |
| 100 | 32% |
| 500 | 22% |
| 1000 | 21% |

DC dominance **decreases** at scale because more concepts = more discriminative structure to capture.

### Recommendations for Scale

1. **Keep sqrt-inverse weighting** as the default
2. **Monitor DC dominance** - if it exceeds 70%, consider DC removal
3. **At 1000+ concepts**: May need more than 8 dimensions; consider randomized SVD

## Files in This Directory

| File | Description |
|------|-------------|
| `01_dc_component_analysis.py` | Detailed analysis of what the DC component is |
| `02_solution_analysis.py` | Comparison of different weighting methods |
| `03_scaling_analysis.py` | How the problem behaves at larger scales |
| `04_theoretical_foundation.py` | Mathematical foundations (Perron-Frobenius, spectral clustering, whitening) |

## Key Insights

1. **The DC component is not a bug** - it's a fundamental feature of similarity matrices
2. **Query projection naturally pulls toward the centroid** - this is unavoidable
3. **The solution is in the distance metric**, not the projection
4. **Sqrt-inverse weighting is theoretically grounded** in spectral graph theory
5. **The problem gets better at scale**, not worse

## Implementation

The fix is implemented in `truthspace_lcm/core/knowledge_space.py`:

```python
# Get eigenvalues for weighting
weights = 1.0 / np.sqrt(eigenvalues + 1e-10)
weights = weights / weights.sum()

# Weighted distance
diff = query_position - concept_position
distance = np.sqrt(np.sum(weights * diff**2))
```

---

## Part 2: Absolute vs Relative Coordinates

### The Deeper Insight

The DC component problem exists because we're using **RELATIVE coordinates** (eigenspace from similarity matrix) instead of **ABSOLUTE coordinates** (φ-lattice with semantic dimensions).

### The Old TruthSpace Approach

The original TruthSpace (`temp/old_core/truthspace.py`) used:

```python
# 12 dimensions with semantic meaning
PHI_BLOCK_WEIGHTS = [
    φ², φ², φ², φ²,     # Actions: dims 0-3
    1.0, 1.0, 1.0, 1.0,  # Domains: dims 4-7
    φ⁻², φ⁻², φ⁻², φ⁻²  # Relations: dims 8-11
]

# Positions at φ^level
position[dim] = φ^level
```

**Key properties:**
1. Positions are **ABSOLUTE** - defined by φ^level on fixed dimensions
2. Positions are **VERIFIABLE** - you can check if they're valid
3. **No DC component** - positions aren't derived from similarity
4. **Semantic dimensions** - each axis has meaning

### The Problem with Eigenspace

| Property | Eigenspace | φ-Lattice |
|----------|------------|-----------|
| Coordinate type | Relative | Absolute |
| Position range | [0.1, 0.5] (compressed) | [φ⁻¹⁰, φ¹⁰] (full range) |
| Verifiable | No | Yes |
| DC component | Yes (58%) | No |
| Dimension meaning | Emergent | Semantic |

The eigenspace compresses all positions into a narrow range (φ⁻² to φ⁻¹), giving very little dynamic range for discrimination.

### Connection to Zeta Line Method

From `docs/zeta_line_method.md`:
- Neural network weights cluster at φ^(-k) levels
- This is the "zeta line" through truth space
- Values naturally align to this absolute coordinate system

From `docs/kerr_truth_space_discovery.md`:
- Event horizon at σ = 1/(2φ) separates regimes
- The "natural curve" for navigation exists
- Zeta zeros are waypoints on this curve

### Proposed Hybrid Approach

1. **φ-Lattice for Bootstrap**: Place core concepts at absolute φ^k positions
2. **Similarity for Navigation**: Use similarity to find direction, not position
3. **Snap to Lattice**: New concepts snap to nearest φ-lattice point
4. **Zeta Waypoints**: Navigate via zeta zeros between φ-levels

### Files in This Directory

| File | Description |
|------|-------------|
| `01_dc_component_analysis.py` | What the DC component is |
| `02_solution_analysis.py` | Comparison of weighting methods |
| `03_scaling_analysis.py` | Behavior at larger scales |
| `04_theoretical_foundation.py` | Mathematical foundations |
| `05_absolute_vs_relative_coordinates.py` | Comparison of coordinate systems |
| `06_phi_lattice_implementation.py` | Exploration of φ-lattice approach |

---

## Summary

The DC component problem is a symptom of using **relative coordinates**. The solution may be to return to **absolute φ-based coordinates** as in the original TruthSpace, using:

- **φ^k levels** for mathematically verifiable positions
- **Semantic dimensions** for meaningful axes
- **Zeta zeros** as navigation waypoints
- **Similarity** for direction, not position

*"Similarity matrices are pointing at absolute positions, but we're having trouble finding them because we're using the wrong coordinate system."*
