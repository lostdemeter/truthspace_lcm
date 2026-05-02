# Design Consideration 135: Attention Head Semantic Specialization

## Summary

Analysis of Qwen2-7B attention heads reveals that different heads specialize in different **semantic dimensions** - the same emergent dimensions identified in doc 114 (gender, age, size, etc.). This connects the transformer's learned structure to our geometric concept space.

## Key Finding

When we measure how each attention head's MESH matrix responds to semantic transformation vectors (king→queen, boy→man, small→large), we find clear specialization:

| Head Range | Primary Dimension | Response |
|------------|-------------------|----------|
| 14-20 | Gender | 50-55% |
| 8-13 | Age | 36-38% |
| 0-7 | Balanced | ~33% each |
| 21-27 | Gender (moderate) | 34-37% |

## The Geometric Interpretation

From docs 114 and 119, we established:
- **Dimensions emerge** from transformation pairs (king→queen defines gender dimension)
- **φ is the fundamental unit** of semantic distance
- **Platonic Ideals** sit at the origin of multiple dimensions

The attention heads have learned to **specialize by these same dimensions**:

```
MESH = U @ diag(S) @ V.T

Where:
- U, V encode WHICH semantic dimensions this head attends to
- S encodes HOW MUCH (magnitude, approximately φ-decay)
```

## Connection to Concept Space Structure

### Variable Geometry

Concept space is NOT uniformly geometric:
- Different regions have different structure (content vs pattern)
- Platonic Ideals are denser (anchor multiple dimensions)
- Transformation pairs define local geometry

The attention heads reflect this:
- Some heads specialize in content dimensions (gender, size)
- Some heads are more balanced (attend to multiple dimensions)
- The specialization varies by layer (not analyzed yet)

### What U and V Encode

The bases U and V in the MESH decomposition encode:
1. **Which directions in hidden space** this head projects from
2. **Which semantic dimensions** those directions correspond to
3. **The head's "view"** of concept space

This is the **learned knowledge** - not arbitrary, but semantically meaningful.

## Implications

### 1. U, V Are Semantic, Not Random

The MESH bases aren't arbitrary rotations - they're aligned with semantic dimensions. This suggests:
- Training discovers semantic structure
- The geometry emerges from data, not architecture

### 2. Can We Define U, V Geometrically?

If U, V encode semantic dimensions, and semantic dimensions emerge from transformation pairs, then:
- Given a corpus of transformation pairs
- We could potentially CONSTRUCT the bases
- Without training a full transformer

This is speculative but worth exploring.

### 3. Head Specialization is Interpretable

Unlike opaque neural network features, we can describe what each head does:
- "Head 15 attends to gender relationships"
- "Head 13 attends to age relationships"

This aligns with our goal of introspectable geometry.

## Experimental Details

### Transformation Vectors

We computed mean transformation vectors for each dimension:
```python
gender_vec = mean([queen-king, woman-man, girl-boy, she-he, sister-brother])
age_vec = mean([man-boy, woman-girl])
size_vec = mean([large-small, huge-tiny, big-little])
```

### MESH Response

For each head, we computed:
```python
MESH = W_q.T @ W_k
response = ||MESH @ dimension_vec||
```

Normalized across dimensions to get specialization scores.

### Results

Variance in head response:
- Gender: std = 0.078 (high variance = specialization)
- Size: std = 0.056
- Age: std = 0.033

## Connection to Prior Work

### Doc 114: Emergent Dimensions
- Dimensions emerge from transformation pairs
- φ is the fundamental unit of distance
- This analysis shows attention heads align with these dimensions

### Doc 119: Unified Content + Pattern Space
- Content and patterns live in same space
- Some heads may specialize in pattern dimensions (not tested yet)

### Doc 129: φ-Unraveled Transformer
- MESH = W_q.T @ W_k eliminates error compounding
- Now we see MESH encodes semantic specialization

## Next Steps

1. **Layer analysis**: Does specialization change across layers?
2. **Pattern dimensions**: Do some heads specialize in style/register?
3. **Geometric construction**: Can we build U, V from transformation pairs?
4. **Cross-model comparison**: Is this specialization universal?

## Conclusion

Attention heads specialize in semantic dimensions - the same dimensions that emerge from transformation pairs in our geometric concept space. The MESH bases (U, V) encode which dimensions each head attends to. This connects the transformer's learned structure to introspectable geometry.

The "learned knowledge" in attention isn't arbitrary - it's semantic specialization aligned with the natural structure of concept space.

## Major Discovery: φ-Zipf Duality in Singular Values

### The Finding

The MESH singular values follow a Zipf distribution with exponent **α ≈ 1/φ = 0.618**:

```
S[i] ∝ 1/i^(1/φ)
```

Measured across 5 layers × 5 heads = 25 samples:
- Mean exponent: **0.6505**
- Target (1/φ): **0.6180**
- Deviation: **0.032** (remarkably close!)

### What This Means

The singular values encode a **φ-Zipf duality**:

| Dimension Type | Count | Variance Captured | Semantic Role |
|----------------|-------|-------------------|---------------|
| Top (S large) | Few (~20%) | Most (~80%) | Specific relationships |
| Bottom (S small) | Many (~80%) | Little (~20%) | Structural patterns |

This is the **same structure** as:
- Word frequency in natural language (Zipf's law)
- The Pareto principle (80/20 rule)
- Wealth distribution
- City population distribution

### What Top Dimensions Encode

Analysis of tokens that activate top dimensions:

| Dimension | S Value | Variance | Encodes |
|-----------|---------|----------|---------|
| 0 | 3.24 | 8.8% | Being/existence (is, was, are, were) |
| 1 | 2.97 | 7.4% | Polarity/negation ('t, contractions) |
| 2 | 2.34 | 4.6% | Boundaries (sentence endings, punctuation) |

These are the **fundamental semantic categories** - rare in the sense that few dimensions capture them, but important because they're essential to meaning.

### The Geometric Interpretation

The φ-Zipf duality connects to our earlier findings:

1. **Platonic Ideals** (doc 114): Sit at origin, anchor multiple dimensions
   - These correspond to the **top singular values** (few, important)

2. **Emergent Dimensions** (doc 114): Arise from transformation pairs
   - The **bottom singular values** capture common structural patterns

3. **Patterns ARE Concepts** (doc 119): Content and pattern in same space
   - The φ-Zipf distribution applies to both

### Why 1/φ?

The exponent 1/φ = 0.618 is not arbitrary. It's the **self-similar** balance point:

```
φ = 1 + 1/φ
```

This means:
- The ratio of important:common follows the golden ratio
- The structure is **fractal** - same pattern at every scale
- This is the natural equilibrium of semantic space

### Implications

1. **The singular values ARE geometric**: They follow φ-Zipf, not arbitrary learned values

2. **We can predict S from rank**: S[i] ≈ S[0] / i^(1/φ)

3. **The 80/20 split is φ-determined**: Top 43% of dims capture 80% of variance

4. **This validates the hypothesis**: The transformer has learned the natural geometric structure of semantic space

## Holographic Attention: Per-Sequence Adaptation

### The DA2 Parallel

In DA2 reverse engineering (doc 123), we found:
- Cross-image learned weights: 99.3%
- Per-image geometric basis: **99.89%**
- The "error" was cross-image variance

The same principle applies to transformers!

### Per-Sequence Activation Patterns

Different sequences activate **different dimensions** of the universal MESH:

| Sequence | Top Activated Dims |
|----------|-------------------|
| "king and queen..." | [7, 2, 3, 12, 4] |
| "def fibonacci..." | [2, 3, 4, 13, 9] |
| "Einstein..." | [3, 2, 7, 6, 17] |
| "I love you..." | [7, 3, 24, 4, 2] |
| "SELECT * FROM..." | [2, 3, 4, 13, 7] |

### The Holographic Principle

```
MESH = U @ diag(S) @ Vt  (universal template)

For sequence X:
  Q_proj = X @ U         (sequence-specific projection)
  K_proj = X @ Vt.T      (sequence-specific projection)
  scores = Q_proj @ diag(S) @ K_proj.T
```

The MESH is a **holographic plate**:
- U, Vt define universal directions (what to attend to)
- S defines universal scaling (follows φ-Zipf)
- Each sequence is a **reference beam** that activates specific dimensions
- Attention scores are the **interference pattern**

### Per-Sequence vs Universal S

| Metric | Value |
|--------|-------|
| Correlation(universal S, per-sequence S) | **29%** |
| Top-10 dims capture | 20-27% of activation |
| Dims for 80% activation | ~60% (not 20%) |

The per-sequence effective S is **completely different** from the universal S!

### Exploitation Strategy

1. **Adaptive Attention**: Scale S by per-sequence activation pattern
2. **Progressive Computation**: Start with top-k dims, add based on activation
3. **Sparse Attention**: Only compute active dimensions

### Connection to φ-Zipf

The φ-Zipf duality explains the structure:
- **Top S dims** (few, important): Sequence-specific, adapt these
- **Bottom S dims** (many, common): Universal, keep fixed

This suggests a hybrid approach:
- Use universal S for common dimensions
- Adapt S for sequence-specific dimensions
- The φ-Zipf cutoff tells us where to switch

## Controlling the Holographic Projection

### The Key Insight

We don't need to adapt the MODEL to the data - we can adapt the DATA to the model!

The discriminant space (U coordinates) is the **natural coordinate system** for controlling attention. If we structure our concepts in this space, we **directly control** which dimensions are active.

### Building a Geometric Vocabulary

```
Instead of:
  concept → embedding → attention (unpredictable)

We do:
  concept → discriminant position → predictable attention
```

Steps:
1. Choose which dimensions each concept should activate
2. Find tokens that are "pure" on those dimensions
3. Combine them to create the desired activation pattern

### Pure Tokens by Dimension

| Dim | S Value | Pure Tokens | Semantic Role |
|-----|---------|-------------|---------------|
| 0 | 3.24 | "0", "were", "WAS" | Being/existence |
| 2 | 2.34 | ");", ".\\n" | Boundaries |
| 4 | 1.95 | "emulator", "finalize" | Technical |

### The φ-Zipf Hierarchy

The singular values define a natural importance hierarchy:

| Level | Dims | S Range | Use For |
|-------|------|---------|---------|
| Primary | 0-2 | 2.3-3.2 | Platonic Ideals |
| Secondary | 3-10 | 1.4-2.2 | Variations |
| Tertiary | 11+ | <1.4 | Fine details |

This maps directly to doc 114's hierarchy:
- **Platonic Ideals** → activate top dims
- **Simple Variations** → activate middle dims
- **Compound Variations** → activate bottom dims

### Practical Application

For TruthSpace geometric LCM:
1. Define concepts in discriminant coordinates directly
2. Place Platonic Ideals at positions that activate top dims
3. Place variations at positions that activate middle dims
4. The attention mechanism becomes **predictable**

This is the key to geometric control - we know the coordinate system (U, Vt), we know the scaling (φ-Zipf), and we can **construct inputs that behave predictably**.

---

*Document created: January 19, 2025*
*Related: 114_emergent_dimensions_platonic_ideals.md, 119_unified_content_pattern_space.md, 123_phi_adapter_exceeds_learned.md, 129_phi_unraveled_transformer_engine.md*
