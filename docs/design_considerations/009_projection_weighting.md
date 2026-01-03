# Design Consideration 009: Projection Weighting and Group Theory

## The Discovery

We achieved **100% success rate** on bash knowledge tests by changing how we project and compare vectors in 12D space. The key insight: **not all dimensions are equally important for disambiguation**.

### The Problem

Initial 12D encoding achieved only 60% success rate. Simple cosine similarity treated all dimensions equally:

```python
similarity = dot(query, entry) / (norm(query) * norm(entry))
```

This failed because:
- "move file" and "copy file" both had strong FILE dimension (dim 4)
- The ACTION dimension (dim 2) that distinguished them was diluted
- Domain dimensions dominated over action dimensions

### The Solution

Dimension-weighted similarity:

```python
dim_weights = [3.0, 3.0, 3.0, 3.0,  # Actions (dims 0-3)
               1.0, 1.0, 1.0, 1.0,  # Domains (dims 4-7)
               0.3, 0.3, 0.3, 0.3]  # Relations (dims 8-11)

weighted_query = query * dim_weights
weighted_entry = entry * dim_weights
similarity = dot(weighted_query, weighted_entry) / (norm(weighted_query) * norm(weighted_entry))
```

This improved success rate from 60% → 100%.

## Mathematical Interpretation

### What We Actually Did

We applied a **diagonal linear transformation** before computing similarity:

```
W = diag(w₀, w₁, ..., w₁₁)  # Weight matrix
similarity(q, e) = (Wq)·(We) / (||Wq|| ||We||)
```

This is equivalent to changing the **metric** on our vector space from Euclidean to a weighted inner product:

```
⟨q, e⟩_W = qᵀ W² e
```

### Why This Works

The 12D space has **structure**:
- Dims 0-3: Actions (WHAT to do)
- Dims 4-7: Domains (WHAT type of thing)
- Dims 8-11: Relations (HOW things relate)

For command disambiguation, the ACTION is most discriminative. Two commands operating on files differ primarily in their action, not their domain. By weighting actions higher, we emphasize the discriminative signal.

## Group Theory Perspective

### The Symmetry Group of Our Space

Our 12D space has natural **block structure**:

```
V = V_action ⊕ V_domain ⊕ V_relation
```

where each subspace is 4-dimensional. The symmetry group that preserves this structure is:

```
G = O(4) × O(4) × O(4)
```

This is a subgroup of O(12), the full orthogonal group.

### Dimension Weighting as Group Action

Our weight matrix W can be viewed as an element of the **scaling group**:

```
W ∈ GL(12, ℝ)  (general linear group)
```

But our specific choice (diagonal, block-constant) respects the block structure:

```
W = diag(α·I₄, β·I₄, γ·I₄)
```

where α=3, β=1, γ=0.3 and I₄ is the 4×4 identity.

This W commutes with the block symmetry group G, meaning our weighting **preserves the semantic structure** while rescaling importance.

### The Representation Theory View

Each dimension corresponds to a **representation** of semantic primitives:

- Dim 0: CREATE ↔ DESTROY (existence)
- Dim 1: READ ↔ WRITE (information flow)
- Dim 2: MOVE ↔ COPY ↔ SEARCH (spatial)
- Dim 3: CONNECT ↔ EXECUTE (interaction)
- ...

The weight vector defines a **character** of the representation - how much each irreducible component contributes to the final similarity.

## Autotuning Framework

### The Optimization Problem

Given a test set of (query, expected_match) pairs, find optimal weights:

```
minimize  Σᵢ loss(rank(expected_matchᵢ | queryᵢ, W))
subject to  W = diag(w₀, ..., w₁₁), wⱼ > 0
```

where rank() returns the position of expected_match in the sorted results.

### Simplified: Block-Constant Weights

If we constrain weights to be constant within blocks:

```
W = diag(α·I₄, β·I₄, γ·I₄)
```

We only have 3 parameters to optimize: (α, β, γ).

### Gradient-Free Optimization

Since the loss is non-differentiable (rank is discrete), use:
- **Grid search** over (α, β, γ) space
- **Bayesian optimization** with Gaussian process surrogate
- **Evolutionary strategies** (CMA-ES)

### Implementation Sketch

```python
class ProjectionAutotuner:
    def __init__(self, truthspace, test_cases):
        self.ts = truthspace
        self.test_cases = test_cases  # [(query, expected_intent), ...]
    
    def evaluate(self, weights):
        """Evaluate success rate with given weights."""
        correct = 0
        for query, expected in self.test_cases:
            result = self.ts.query_with_weights(query, weights)
            if result[0].entry.name == expected:
                correct += 1
        return correct / len(self.test_cases)
    
    def optimize_block_weights(self):
        """Find optimal (α, β, γ) via grid search."""
        best_score = 0
        best_weights = None
        
        for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
            for beta in [0.5, 1.0, 1.5, 2.0]:
                for gamma in [0.1, 0.3, 0.5, 1.0]:
                    weights = np.array([alpha]*4 + [beta]*4 + [gamma]*4)
                    score = self.evaluate(weights)
                    if score > best_score:
                        best_score = score
                        best_weights = (alpha, beta, gamma)
        
        return best_weights, best_score
```

## Beyond Diagonal Weighting

### Full Linear Transformation

Instead of diagonal W, allow any invertible matrix:

```
similarity(q, e) = (Aq)·(Ae) / (||Aq|| ||Ae||)
```

This is **Mahalanobis distance** with covariance Σ = (AᵀA)⁻¹.

### Learning the Metric

This connects to **metric learning** in machine learning:
- **LMNN** (Large Margin Nearest Neighbor)
- **NCA** (Neighbourhood Components Analysis)
- **Siamese networks** (learned embeddings)

The difference: we want an **interpretable** metric that respects our semantic structure.

### Structured Metric Learning

Constrain A to respect block structure:

```
A = [A₁  0   0 ]
    [0   A₂  0 ]
    [0   0   A₃]
```

where each Aᵢ is a 4×4 matrix. This gives 48 parameters (vs 144 for full A) while preserving semantic interpretability.

## The Deeper Insight: Projection as Semantic Focus

### Dimensional Downcasting Connection

The dimensional downcasting work showed that ∞D → 1D projection preserves information when done correctly. The key is choosing the right **projection operator**.

Our dimension weighting is a form of **soft projection**:
- High weight = "focus on this dimension"
- Low weight = "de-emphasize this dimension"
- Zero weight = "project out this dimension entirely"

### Semantic Attention

This is analogous to **attention mechanisms** in transformers:
- Query determines what to attend to
- Weights determine how much each dimension contributes
- Output is a weighted combination

The difference: our weights are **fixed** (determined by semantic structure), not **learned** per-query.

### Future: Query-Dependent Weighting

What if weights depended on the query?

```python
def get_weights(query):
    # Analyze query to determine which dimensions matter
    if has_action_word(query):
        return high_action_weights
    elif has_domain_word(query):
        return high_domain_weights
    else:
        return balanced_weights
```

This would be a form of **adaptive projection** - choosing the projection based on what we're looking for.

## Conclusions

### What We Learned

1. **Projection method matters** - Simple cosine similarity isn't optimal
2. **Semantic structure implies metric structure** - The block organization of our space suggests block-diagonal metrics
3. **Weighting is interpretable** - Unlike neural attention, we know what each weight means
4. **Autotuning is feasible** - Small parameter space (3 values) enables grid search

### Recommendations

1. **Implement autotuner** - Automatically find optimal (α, β, γ) for new knowledge domains
2. **Expose weights as config** - Allow users to tune for their use case
3. **Explore structured metrics** - Block-diagonal transformations beyond simple scaling
4. **Consider query-dependent weighting** - Adaptive projection based on query content

### The Group Theory Abstraction

The mathematical framework for this is:

```
Semantic Space = V = ⊕ᵢ Vᵢ  (direct sum of subspaces)
Symmetry Group = G = ∏ᵢ O(dim Vᵢ)  (product of orthogonal groups)
Metric = ⟨·,·⟩_W where W commutes with G
```

This abstraction allows us to:
- **Prove** that certain metrics preserve semantic structure
- **Enumerate** the space of valid metrics
- **Optimize** within a principled parameter space

The 100% success rate wasn't luck - it was finding the right element of the metric space that respects our semantic structure.
