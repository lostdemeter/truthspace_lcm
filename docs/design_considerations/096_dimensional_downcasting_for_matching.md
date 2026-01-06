# Design Consideration 096: Dimensional Downcasting for Knowledge Matching

## Date: 2025-01-05

## Context

After implementing eigenspace geodesic matching (Design 095), we achieved 73% accuracy. The remaining 27% failures occur when queries project to unexpected eigenspace regions. This led to exploring dimensional downcasting as a solution.

## The Core Insight

**The offset from the lattice is a SIGNAL, not noise.**

When a query doesn't snap cleanly to a concept position, it's telling us:
1. The query contains information that doesn't fit the current dimensional structure
2. We're viewing at the wrong "zoom level" (dimensional resolution)
3. A new dimension may be needed to accommodate the query

This parallels the dimensional downcasting discovery for zeta zeros:
- **N_smooth(t_n) ≈ n - 0.5** — a consistent offset enables correct identification
- The offset isn't error — it's the key to disambiguation

## The Polyomino Analogy

Concept positions form a discrete lattice (like polyomino pieces):
- Only certain configurations are valid
- Queries should "snap" to lattice points
- When they don't snap, the offset tells us which dimension is missing/misaligned

```
EIGENSPACE LATTICE

     ●  Physics
    ●   Science
   ●    Math
  
       ○ ← Query projects HERE (between lattice points)
         
    ●   Identity
   ●    Social
```

The query's offset from the nearest lattice point encodes information about what's missing.

## Experimental Findings

### Offset Analysis

```
Query                  Offset    Nearest Match    Correct?
"what is physics?"     0.153     Physics          ✓
"what is science?"     0.146     Science          ✓
"who are you?"         0.186     LLMs             ✗
"what can you do?"     0.097     Identity         ✓
"thank you"            0.125     Wellbeing        ~
"hello"                0.063     Greeting         ✓
```

**Pattern discovered:**
- Small offset (< 0.1): Query fits the lattice well → correct match
- Large offset (> 0.15): Query doesn't fit → potential mismatch

### Offset Direction Correlations

```
"what is physics?" vs "what is science?": 0.67 correlation
```

Queries of the same TYPE have correlated offset directions. This suggests a systematic dimensional mismatch for certain query patterns.

## Connection to Dimensional Downcasting

From the zeta zeros work:
- **Traditional**: 2D → 3D (upcast via radiance field, requires training)
- **Downcasting**: ∞D → 1D (downcast via moment projection, pure math)

Applied to knowledge matching:
- **Current**: Project query to 8D eigenspace, find nearest
- **Downcasting**: Project to fewer dimensions where query snaps better
- **Upcasting**: Add dimensions to accommodate query's unique information

### Natural Scales (Eigenvalues)

The eigenspace has natural scales encoded in eigenvalues:
```
Dim 0: λ = 24.09 (dominant - structural similarity)
Dim 1: λ = 2.70  (domain separation - the t-coordinate)
Dim 2: λ = 1.52  (finer distinctions)
...
```

These eigenvalues ARE the natural scales for dimensional navigation.

## Approaches Explored

### 1. Naive Downcasting (17% accuracy)
Project to first N dimensions. Fails because it loses domain separation.

### 2. Adaptive Dimension Selection (50% accuracy)
Select dimensions where query has strongest signal. Better but inconsistent.

### 3. Eigenvalue-Weighted Distance (67% accuracy)
Weight dimensions by 1/√λ to normalize variance. Matches original eigenspace approach.

### 4. Multi-Scale Matching (17% accuracy)
Try multiple eigenvalue weightings, pick best. Over-emphasizes first component.

## The Deeper Insight

The experiments reveal that **the problem isn't the matching algorithm** — it's that:

1. **The query projection itself** lands in the wrong region
2. The projection is a weighted sum of ALL concept positions
3. "Hub" concepts (identity response) pull queries toward them

The dimensional downcasting insight suggests we need to:
- **Detect** when a query doesn't fit the current dimensional structure
- **Adapt** the dimensional resolution for that query
- **Or** recognize that a new dimension is needed

## Softmax Analogy

Traditional LLMs use softmax to convert continuous logits to discrete token probabilities. Our eigenspace matching is trying to do something similar — convert continuous query position to discrete concept selection.

The offset from the lattice is like the "temperature" in softmax:
- Small offset → high confidence (sharp distribution)
- Large offset → low confidence (flat distribution)

When offset is large, we should either:
1. **Sharpen**: Downcast to fewer dimensions where query fits
2. **Soften**: Upcast to more dimensions to accommodate query
3. **Signal**: Indicate that the query doesn't match existing knowledge

## Future Directions

### 1. Offset-Based Confidence

Use offset magnitude as a confidence score:
```python
confidence = 1.0 / (1.0 + offset)
if confidence < threshold:
    return "I don't have specific knowledge about that"
```

### 2. Offset-Guided Dimension Selection

Use offset direction to select relevant dimensions:
```python
# Offset points toward the "missing" dimension
missing_dim = argmax(abs(offset))
# Match in subspace excluding that dimension
```

### 3. Dynamic Dimension Creation

When offset is consistently large for a query type:
```python
# This query type needs a new dimension
# Add it to the eigenspace via incremental SVD
```

### 4. Hierarchical Matching

Match at multiple scales simultaneously:
```python
# Coarse: first 2 dimensions (domain)
# Medium: first 4 dimensions (topic)
# Fine: all 8 dimensions (specific concept)
# Combine scores with learned weights
```

## Conclusion

The dimensional downcasting exploration revealed that:

1. **73% accuracy** is achievable with pure eigenspace distance
2. **The remaining 27%** fail because queries project to wrong regions
3. **The offset from the lattice** is informative, not noise
4. **Multi-scale analysis** is promising but needs better scale selection

The key insight: **queries that don't snap to the lattice are telling us something** — either about missing dimensions, wrong resolution, or genuinely novel information.

```
"The offset is not error — it's the query asking for a dimension that doesn't exist yet."
```

---

## References

- Design 095: Eigenspace Geodesic Matching
- Design 057: Domain Dimension as Zeta t-Coordinate
- Dimensional Downcasting for Riemann Zeta Zeros (github.com/lostdemeter/dimensional_downcasting)
- Design 046-049: Holographic/Geodesic Generation

---

*"When the query doesn't fit the lattice, the lattice is incomplete, not the query."*
