# Dimensionality Analysis Findings

**Date**: December 16, 2025

## Key Questions Investigated

1. Why does the plastic constant show stronger separation than φ?
2. Is 8D optimal? Why 12D for LLMs?
3. How can orthogonality be exploited for autotuning?

---

## 1. Why Plastic Constant Shows Stronger Separation

### The Constants

| Constant | Value | Equation | Growth Rate |
|----------|-------|----------|-------------|
| Plastic (ρ) | 1.3247 | x³ = x + 1 | Slowest |
| Golden (φ) | 1.6180 | x² = x + 1 | Medium |
| Silver (δ) | 2.4142 | x² = 2x + 1 | Fastest |

### Why Plastic Works Better

1. **Slower Growth** (ρ < φ < δ)
   - Golden ratio phases "jump" more between positions
   - Plastic phases spread more gradually
   - **Finer discrimination** between similar concepts

2. **Cubic vs Quadratic**
   - φ satisfies x² = x + 1 (quadratic)
   - ρ satisfies x³ = x + 1 (cubic)
   - Cubic relationships create **more complex interference patterns**

3. **Longer Memory**
   - Fibonacci: each term = sum of 2 previous
   - Padovan: each term = sum of 2nd and 3rd previous
   - Padovan has **"longer memory"** in its recurrence

4. **Semantic Implication**
   - If semantic relationships have hierarchical depth > 2, cubic captures better
   - The "longer memory" matches how meaning accumulates across abstraction levels

### Recommendation
Consider using **plastic as primary constant**, φ as secondary.

---

## 2. Dimensionality Analysis

### Current Structure (8D)

```
Dim 0-3: ACTIONS (what to do)
  0: CREATE ↔ DESTROY (existence axis)
  1: READ ↔ WRITE (information flow axis)
  2: MOVE / SEARCH (spatial axis)
  3: CONNECT / EXECUTE (interaction axis)

Dim 4-6: DOMAINS (what type of thing)
  4: FILE / SYSTEM
  5: PROCESS / DATA
  6: NETWORK

Dim 7: MODIFIERS (how to do it)
  7: ALL / RECURSIVE / FORCE
```

### Dimension Activation (22 test concepts)

| Dim | Active Concepts | Status |
|-----|-----------------|--------|
| 0 | 4/22 | ✓ ACTIVE |
| 1 | 4/22 | ✓ ACTIVE |
| 2 | 5/22 | ✓ ACTIVE |
| 3 | 2/22 | ⚠️ SPARSE |
| 4 | 4/22 | ✓ ACTIVE |
| 5 | 1/22 | ⚠️ SPARSE |
| 6 | 1/22 | ⚠️ SPARSE |
| 7 | 1/22 | ⚠️ SPARSE |

### PCA Analysis

| Metric | Value |
|--------|-------|
| Effective dimensionality (95% variance) | **7D** |
| Intrinsic dimensionality (elbow) | **4D** |

**Interpretation**: We're using 8D but only ~4D carry significant structure. Dimensions 5-7 are underutilized.

### Why 12D for LLMs?

LLM attention heads:
- GPT-2: 12 heads × 64 dim = 768D
- BERT: 12 heads × 64 dim = 768D

The **12 attention heads** each learn different relationship types. Our 12D clock mirrors this!

### Hypothesis: 12 Fundamental Relationship Types

1. Hierarchical (parent-child)
2. Sequential (before-after)
3. Causal (cause-effect)
4. Compositional (part-whole)
5. Oppositional (antonyms)
6. Synonymic (same meaning)
7. Analogical (A:B :: C:D)
8. Associative (co-occurrence)
9. Functional (same role)
10. Categorical (same type)
11. Spatial (location-based)
12. Temporal (time-based)

### Proposed Expansion to 12D

```
Current (8D):
  0-3: Actions
  4-6: Domains
  7: Modifiers

Expanded (12D):
  0-3: Actions (same)
  4-6: Domains (same)
  7: Modifiers (same)
  8: TEMPORAL (before, after, during, while)
  9: CAUSAL (because, therefore, causes, results)
  10: CONDITIONAL (if, when, unless, provided)
  11: COMPARATIVE (more, less, equal, different)
```

---

## 3. Orthogonality Exploitation for Autotuning

### Key Insight

**Orthogonal dimensions are INDEPENDENT**

If concept A is on dimension 0 and concept B is on dimension 4:
- Similarity = 0 (completely independent)
- Adding to dim 0 CANNOT interfere with dim 4
- We can tune dimensions independently
- **Collisions only happen WITHIN a dimension**

### Autotuning Strategy

```
Step 1: CLASSIFY the new concept
  - What type? (ACTION, DOMAIN, MODIFIER)
  - This determines which dimensions it can occupy

Step 2: FIND THE RIGHT DIMENSION
  - Within its type, which specific dimension?
  - Check for collisions only in that dimension

Step 3: FIND THE RIGHT LEVEL
  - Within the dimension, what φ^level?
  - Opposites get opposite signs
  - Synonyms get same position

Step 4: VERIFY ORTHOGONALITY
  - Confirm new concept is orthogonal to unrelated concepts
  - If not, we've misclassified it
```

### Example: Adding 'backup'

```
Related concepts (should be similar):
  copy    → dimension 2, level 0
  move    → dimension 2, level 0
  archive → dimension 2, level 0

Unrelated concepts (should be orthogonal):
  delete  → dimension 0 (orthogonal ✓)
  network → dimension 6 (orthogonal ✓)

→ 'backup' should go on dimension 2, same level as 'copy'
```

### Algorithm

```python
def find_optimal_position(new_concept, test_cases):
    # 1. Encode test case inputs to find semantic neighborhood
    neighbors = [encode(tc.input) for tc in test_cases]
    
    # 2. Find which dimension(s) neighbors occupy
    active_dims = find_active_dimensions(neighbors)
    
    # 3. For each candidate dimension, check for collisions
    for dim in active_dims:
        existing = get_concepts_on_dimension(dim)
        if no_collision(new_concept, existing):
            return dim, compute_level(new_concept, neighbors)
    
    # 4. If all dimensions have collisions, we need a new dimension
    return suggest_new_dimension()
```

---

## Summary & Recommendations

### Findings

1. **Plastic constant** shows stronger separation because it grows slower (finer granularity) and has cubic recurrence (deeper hierarchy capture)

2. **Dimensionality**: 
   - Current 8D has only ~4D effective structure
   - Dimensions 5-7 are underutilized
   - Consider expanding to 12D to match clock and capture more relationship types

3. **Orthogonality**:
   - Independent dimensions = independent tuning
   - Autotuner should: classify → find dimension → find level
   - Collisions only matter within a dimension

### Next Steps

1. **Experiment with plastic-primary encoding**
2. **Add dimensions 8-11** for temporal/causal/conditional/comparative
3. **Implement dimension-aware autotuning** that exploits orthogonality
4. **Test if 12D encoding improves semantic separation**

### The Big Picture

The orthogonality insight is powerful: **tuning is just finding where things slot into place**. 

If we have the right axes (dimensions) and the right constants (φ, ρ, etc.), then:
- New knowledge automatically finds its place
- Collisions are localized to single dimensions
- The autotuner becomes a **classifier + level finder**

This is much simpler than trying to optimize in a continuous high-dimensional space!
