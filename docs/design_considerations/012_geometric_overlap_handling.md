# Design Consideration 012: Geometric Overlap Handling

## The Problem

When encoding natural language queries, **synonyms cause over-counting**. For example:

```
Query: "show disk space"
  - "show" → READ (dim 1)
  - "disk" → STORAGE (dim 5)
  - "space" → STORAGE (dim 5)

With SUM: dim5 = STORAGE + STORAGE = 4.65
Rule "read storage": dim5 = STORAGE = 2.32

Distance = 2.33 (query is "further out" than rule)
```

This causes the query to match the wrong rule (e.g., "read time" instead of "read storage") because the double-counted STORAGE pushes the position away from the intended target.

## The Insight

Traditional LLMs handle this naturally through:
- High dimensionality (billions of parameters)
- Learned representations where synonyms cluster together
- Attention mechanisms that focus on relevant subspaces

We need a **geometric operation** that achieves the same effect without learning.

The key insight came from considering **fractal geometry**: in a Sierpinski gasket, overlapping copies don't accumulate—they occupy the same space. The structure is defined by what's removed, not what's added.

## Three Approaches

### 1. MAX-per-Dimension

The simplest solution: take the maximum activation per dimension instead of summing.

```python
def encode_max(text):
    position = np.zeros(dim)
    for word in words:
        if word maps to primitive:
            value = φ^level × position_decay
            position[dim] = max(position[dim], value)  # MAX, not +=
    return position
```

**Properties:**
- Simple to implement
- Prevents synonym stacking
- Preserves level information (φ^level)
- Winner-take-all within each dimension

**Results:** 100% accuracy on core test suite

### 2. Fractal/Set-Based Encoding

Treat active primitives as a **set**, not a list. Each unique (dimension, level) pair contributes once.

```python
def encode_fractal(text):
    active = {}  # (dim, level) → first_position
    for i, word in enumerate(words):
        if word maps to primitive:
            key = (dimension, level)
            if key not in active:
                active[key] = i  # Only first occurrence
    
    position = np.zeros(dim)
    for (d, level), first_pos in active.items():
        position[d] = max(position[d], φ^level × φ^(-first_pos/2))
    return position
```

**Properties:**
- Set-theoretic foundation
- Idempotent: applying same primitive twice has no effect
- Preserves word order through position decay
- Mirrors Sierpinski property: overlapping regions don't accumulate

**Results:** 100% accuracy on core test suite

### 3. IFS (Iterated Function System) Encoding

Model primitives as **contractive maps** that converge to an attractor.

```python
def encode_ifs(text, iterations=10):
    # Collect unique primitives as contractive maps
    maps = {}
    for word in words:
        if word maps to primitive:
            key = (dimension, level)
            if key not in maps:
                anchor = zeros(dim)
                anchor[dimension] = φ^level
                maps[key] = anchor
    
    # Contraction ratio = 1/φ (golden ratio inverse)
    c = 1/φ ≈ 0.618
    
    # Iterate to find attractor
    position = zeros(dim)
    for _ in range(iterations):
        for anchor in maps.values():
            position = c × position + (1-c) × anchor
    
    return position
```

**Properties:**
- Mathematically elegant: meaning as fixed point
- Self-similar: contraction ratio is 1/φ
- Converges regardless of iteration order
- Connects to dynamical systems theory

**Results:** 93-100% accuracy (slight edge cases)

## Mathematical Comparison

### Synonym Collapse Test

| Query | SUM (dim5) | MAX (dim5) | FRACTAL (dim5) | IFS (dim5) |
|-------|------------|------------|----------------|------------|
| "storage" | 2.32 | 2.32 | 4.24 | 4.20 |
| "disk storage" | 4.65 | 2.32 | 4.24 | 4.20 |
| "disk space storage" | 6.97 | 2.32 | 4.24 | 4.20 |

All three approaches (MAX, FRACTAL, IFS) collapse synonyms to the same value.

### Distance to Rule "read storage"

| Query | SUM | MAX | FRACTAL | IFS |
|-------|-----|-----|---------|-----|
| "show disk space" | 2.62 | 0.00 | 0.00 | 0.00 |

All three achieve **exact match** (distance = 0) for synonym-heavy queries.

## The Deep Connection

All three approaches express the same underlying principle:

> **Overlapping activations occupy the same region; they don't accumulate.**

This is the fractal insight: in a Sierpinski gasket, overlapping copies of the structure don't create "more" structure—they confirm the same pattern.

### Why φ Appears

The golden ratio φ appears naturally in all three approaches:

1. **MAX**: Uses φ^level for encoding (stronger level separation than ρ)
2. **FRACTAL**: Uses φ for position decay
3. **IFS**: Uses 1/φ as contraction ratio (self-similar scaling)

This connects to the φ-dimensional navigation work (Design Consideration 010): φ provides natural hierarchical structure.

## Connection to LLMs

Traditional LLMs achieve synonym handling through:
- **Embedding spaces** where synonyms cluster
- **Attention** that focuses on relevant dimensions
- **Learned saturation** (sigmoid/tanh activations)

Our geometric approach achieves similar results through:
- **Set operations** (unique primitives)
- **Fixed structure** (φ-lattice)
- **Contractive dynamics** (IFS attractor)

The key difference: we use **explicit geometry** instead of learned representations.

## Recommendations

### For Production Use

**Use MAX-per-dimension** with φ-encoding:
- Simplest implementation
- 100% accuracy on test suite
- Easy to understand and debug

```python
def encode(text):
    position = np.zeros(12)
    for i, word in enumerate(words):
        if word in primitives:
            value = φ^level × φ^(-i/2)
            position[dim] = max(position[dim], value)
    return position
```

### For Theoretical Development

**Explore IFS further**:
- Most mathematically elegant
- Connects to dynamical systems, group theory
- May reveal deeper structure

### For Future Research

1. **Attention as dimension**: Can we add a 13th dimension that encodes "where to look"?
2. **Fractal resolution**: Can we use Sierpinski-like subdivision for hierarchical matching?
3. **IFS composition**: What happens when we compose multiple IFS encodings?

## Implementation Status

- [x] MAX-per-dimension encoder (tested, 100% accuracy)
- [x] Fractal encoder (tested, 100% accuracy)
- [x] IFS encoder (tested, 93% accuracy - needs investigation)
- [ ] Integration with ComposableResolver
- [ ] Integration with ingestion pipeline

## IFS Failure Analysis

### The Problem

The original IFS implementation failed on queries like "search text in files" → `mv` instead of `grep`.

### Root Cause

When IFS iterates with **multiple anchors across different dimensions**, the contraction dynamics cause all dimensions to blend toward a common value:

```
Query "search text in files":
  - SEARCH (dim2, level=2) → anchor = 2.618
  - DATA (dim6, level=1) → anchor = 1.618
  - FILE (dim5, level=0) → anchor = 1.0

After 10 iterations with c = 1/φ:
  All active dims converge to ~0.5
```

This loses the **level information** (φ^level) that distinguishes SEARCH from RELOCATE.

### The Fix

**IFS v3: Instant Convergence** - Don't iterate across dimensions. Apply contraction per-dimension only:

```python
def encode_ifs_v3(text):
    position = zeros(dim)
    for word in words:
        if word maps to primitive:
            value = φ^level
            position[dim] = max(position[dim], value)  # Instant convergence
    return position
```

This is mathematically equivalent to MAX, but framed as IFS with instant convergence to the attractor.

### The Insight

The IFS attractor is well-defined for a **single dimension** with multiple same-level activations (synonyms collapse). But when multiple dimensions are active, the cross-dimensional iteration causes unwanted blending.

The correct formulation: **IFS per dimension, MAX across dimensions**.

### IFS Level-Grouped (Alternative Fix)

Another approach that preserves the iterative IFS flavor:

```python
def encode_ifs_level_grouped(text):
    # Group by (dimension, level)
    level_groups = {}
    for word in words:
        key = (dimension, level)
        level_groups[key].append(position_decay)
    
    # Contract within each group, MAX across groups
    for (d, level), strengths in level_groups.items():
        # Synonyms at same level contract together
        contracted = contract(strengths)
        position[d] = max(position[d], φ^level × contracted)
```

This achieves 100% accuracy and maintains the IFS conceptual framework.

### The Elegant Solution: Pure MAX

After exploring various IFS formulations, the simplest and most elegant is **Pure MAX**:

```python
def encode_pure_max(text):
    position = zeros(dim)
    for word in words:
        value = φ^level × position_decay
        position[dim] = max(position[dim], value)
    return position
```

This achieves:
- **Perfect synonym collapse**: "storage" = "disk storage" = "disk space storage"
- **100% accuracy** on test suite
- **Sierpinski property**: overlapping activations occupy the same space

## Explored Questions

### 1. Multi-dimensional IFS with Block-Diagonal Contraction

Tested block-diagonal contraction where each block (actions/domains/relations) contracts independently. Result: Works but offers no advantage over pure MAX. The key insight is that contraction should happen **per (dimension, level)**, not per block.

### 2. Attention as Hyperdimensional Axis

Tested dynamic attention weighting based on query structure (first primitive defines focus). Result: **Performed worse** (4/5 vs 5/5). The static φ-block weights are already optimal—dynamic attention based on word order doesn't help and can hurt by de-emphasizing important primitives.

**Conclusion**: The φ-block weights (φ², 1, φ⁻²) already encode the optimal "attention" structure. Actions matter most, then domains, then relations. This is fixed, not query-dependent.

### 3. Unified Formulation

All three approaches (MAX, FRACTAL, IFS) express the same principle:

> **Treat primitives as regions in a lattice. Overlapping activations confirm the same region; they don't stack.**

The unified formulation is **set-based encoding with MAX aggregation**:
- Collect unique (dimension, level) pairs
- Take MAX activation per dimension
- This is equivalent to instant IFS convergence

## Open Questions (Remaining)

1. **Can we extend this to learned primitives?** The current primitives are hand-crafted. Could we learn new primitives from data while preserving the geometric structure?

2. **How does this scale to larger vocabularies?** With more commands and more overlapping signatures, will MAX still provide sufficient discrimination?

3. **Can we use this for generation, not just resolution?** Given a target position in truth space, can we generate text that encodes to that position?

## Conclusion

The Sierpinski/fractal insight provides an elegant solution to synonym overlap:

> **Treat primitives as regions in fractal space. Overlapping activations confirm the same region; they don't stack.**

This is achieved through:
- **MAX**: Hard saturation per dimension
- **FRACTAL**: Set-based unique primitives
- **IFS**: Contractive maps converging to attractor

All three achieve 100% (or near-100%) accuracy on our test suite, compared to 86% with naive SUM encoding.

The φ-based scaling provides natural level separation, and the set/max operations provide natural overlap handling. Together, they enable purely geometric semantic resolution.
