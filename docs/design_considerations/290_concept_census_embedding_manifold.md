# DC 290: Concept Census — The Embedding Space is a Continuous Manifold

**Status**: Finding (F160)
**Date**: 2026-03-05
**Frontier**: 16
**Depends on**: DC 289, DC 288

## 1. Question

How many "concepts" does Qwen2-7B know? Can the 152064×3584 embedding matrix
be compressed into a smaller set of concept prototypes without losing the
model's predictive behavior?

## 2. Background

F158 showed that concept composition works in embedding space (rank 8-18 for
compound words like dragon+shrimp→lobster). F159 showed the output space is
full-rank (55.8% of SVD dims needed for top-1 prediction). This experiment
directly attacks the question: is the embedding space compressible?

## 3. Experiment Design

Four-part analysis of the φ-decoded embedding matrix:

1. **SVD Energy Profile**: Eigendecomposition of the covariance matrix to
   measure effective rank and variance distribution
2. **K-Means Clustering**: Sweep k ∈ {100, 500, 1000, 2000, 5000, 10000}
   on SVD-projected embeddings to find natural clusters
3. **Reconstruction Test**: Three compression strategies tested against
   lm_head predictions:
   - SVD-only (truncated projection)
   - Cluster+SVD (replace tokens with cluster centers in SVD space)
   - Cluster-only (replace tokens with cluster centers in full space)
4. **Concept Labeling**: Inspect representative tokens per cluster

## 4. Results

### 4.1 SVD Energy Profile — Slow Decay, No Knee

```
Variance %   Dims needed   % of 3584
    50.0%          361        10.1%
    75.0%          861        24.0%
    90.0%         1435        40.0%
    95.0%         2072        57.8%
    99.0%         3224        90.0%
```

Top eigenvalue captures only 1.20% of total variance. The eigenvalue decay
follows Zipf with α ≈ 0.12, matching the MLP weight matrices (F147). There
is no spectral gap that would indicate a low-dimensional structure.

### 4.2 K-Means Clustering — No Elbow

```
k        Inertia     Reduction
100      64873.9     —
500      64437.4     0.7%
1000     63564.1     1.4%
2000     62720.4     1.3%
5000     61201.3     2.4%
10000    57413.8     6.2%
```

Total inertia reduction from k=100 to k=10000: **11.5%**. No elbow exists.
The embedding space is not naturally clustered — it is a continuous manifold
where points are roughly uniformly distributed across dimensions.

### 4.3 Reconstruction Test — Every Dimension Matters

**SVD-only** (best strategy):

| Dims | Top-1 | Cosine | Compression |
|------|-------|--------|-------------|
| 500  | 10.8% | 0.765  | 7×          |
| 1000 | 25.6% | 0.875  | 3.5×        |
| 2000 | 53.1% | 0.954  | 1.8×        |
| 3000 | 70.9% | 0.972  | 1.2×        |
| 3584 | 98.2% | 0.982  | 1.0×        |

**Cluster+SVD** (k=20000, best clustering result):

| Dims | Top-1 | Cosine | Compression |
|------|-------|--------|-------------|
| 200  | 0.3%  | 0.364  | 108.5×      |
| 500  | 2.3%  | 0.398  | 45.0×       |

Clustering is catastrophically worse than SVD at every compression level.
The per-token residual (the difference between a token's embedding and its
cluster center) carries almost ALL discriminative information.

### 4.4 Cluster Labels — Surface, Not Semantic

Clusters organize by writing system, morphology, and syntax:
- Hebrew/Arabic/Korean/Thai tokens form script-based clusters
- English clusters group suffixes (-tion, -ness) and prefixes
- Small clusters (2-5 tokens) are surface variant pairs: "Carlos"/"ĠCarlos"

The clusters capture **orthographic** similarity, not **semantic** similarity.

## 5. Interpretation

### 5.1 Two Regimes in Embedding Space

The embedding space serves two distinct purposes:

1. **Composition** (low-dimensional): Vector addition of embeddings produces
   meaningful compound concepts at rank 8-18 out of 152K vocabulary. This
   operates in the top ~300 SVD dimensions (the "shape").

2. **Discrimination** (full-dimensional): Identifying WHICH token an
   embedding represents requires ~3200 dimensions. This is the per-token
   "address" in the manifold (the "position").

These are complementary, not contradictory. A token's embedding = shape + position.
Shape encodes what-kind-of-thing. Position encodes which-specific-thing.

### 5.2 Why Clustering Fails

K-means replaces each token with its cluster center, destroying the position.
Two tokens in the same cluster become indistinguishable. Since lm_head needs
position for discrimination, cluster-based reconstruction is useless.

SVD preserves all tokens' relative positions in the retained dimensions.
The information loss is spread across all tokens equally rather than
catastrophically for within-cluster tokens.

### 5.3 The φ-Encoding IS the Compression

The φ-encoding from F147 achieves 5.27× compression of the raw weight matrices
while preserving structure. At dims=3584 (no dimensionality reduction), the
φ-decoded embeddings achieve 98.2% top-1 accuracy. The 1.8% gap is the
φ-encoding reconstruction error — but this means φ-encoding preserves
98.2% of the model's discriminative ability.

This suggests φ-encoding is already the right representation: it compresses
the VALUES while preserving the GEOMETRY. Dimensionality reduction compresses
the GEOMETRY and destroys discriminative power.

## 6. Implications for TruthSpace

1. **No concept codebook.** TruthSpace cannot replace 152064 embeddings with
   N << 152064 concept prototypes. There are no discrete concepts.

2. **The manifold must be stored, not summarized.** All 152064 positions are
   needed for token-level discrimination.

3. **Composition is a subspace operation.** Concept composition (addition,
   potentially rotation/reflection) operates in the top ~300 dimensions.
   This is a small fraction of the space but carries the "meaning."

4. **φ-encoding may be sufficient.** Rather than seeking dimensionality
   reduction, φ-encoding compresses the representation into a geometric
   basis (sign × φ^(exp/128)) that preserves both shape and position.

5. **The next question is structure, not compression.** Instead of asking
   "how many concepts?", ask "what is the geometry of the manifold?" The
   152064 points live on a specific surface in 3584-d space. What surface?
   What are its symmetries? Can we characterize it without enumerating
   every point?

## 7. Open Questions

1. **Is the manifold surface characterizable?** 152064 points in 3584-d
   space. What is the intrinsic dimensionality? (UMAP, local PCA, etc.)

2. **Can composition be formalized in the shape subspace?** We know addition
   works (F158). What about rotation, reflection, scaling? Is there a
   group structure in the top-300 dimensions?

3. **What does the 98.2% ceiling mean?** The 1.8% of tokens that fail even
   at full dimensionality — are they noise, or structurally important edge
   cases? Are they the tokens where φ-encoding loses precision?

4. **Is there a better basis than SVD?** SVD finds the directions of maximum
   variance. But token discrimination might benefit from a basis that
   maximizes SEPARATION rather than variance (e.g., LDA, or a φ-aligned basis).

## 8. Files

- `experiments/model_reverse_engineering_v2/frontier16_concept_census.py`
- `experiments/model_reverse_engineering_v2/frontier16_retest_part3.py`
- `experiments/model_reverse_engineering_v2/frontier16_output.log`
- `experiments/model_reverse_engineering_v2/frontier16_retest_output.log`
