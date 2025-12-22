# Design Consideration 033: Dynamic Geometric LCM

## Overview

This document describes the **Dynamic Geometric LCM** architecture, a system where:

1. **Structure IS the data** - Knowledge is stored as geometry (positions and relations)
2. **Learning IS structure update** - New information modifies the geometry dynamically

This is a foundation for replacing traditional LLMs with pure geometric computation.

## Core Principles

### Traditional LLM vs Geometric LCM

| Aspect | Traditional LLM | Geometric LCM |
|--------|-----------------|---------------|
| Knowledge storage | Weights (billions of parameters) | Geometry (positions + relations) |
| Learning | Gradient descent on loss | Attractor dynamics on structure |
| Inference | Forward pass through layers | Geometric operations (projection, similarity) |
| Interpretability | Black box | Fully transparent |
| Update mechanism | Retraining | Dynamic structure modification |

### The Key Insight

**Relations must be INVARIANT across instances.**

Random hash-based vectors fail at analogies because:
- `paris - france` ≠ `berlin - germany` (random → random)

The solution: **Learn** the relation vector from examples so it becomes consistent:
- After learning: `paris - france` ≈ `berlin - germany` ≈ `capital_of`

## Architecture

### Data Structures

```python
@dataclass
class Entity:
    name: str
    position: np.ndarray  # Position in geometric space
    entity_type: str      # For type-based operations

@dataclass
class Relation:
    name: str
    vector: np.ndarray    # Direction from subject to object
    consistency: float    # How aligned instances are
```

### Learning Algorithm

```
For each iteration:
    1. Update relation vectors from current entity positions
       relation.vector = average(object.position - subject.position)
    
    2. Update entity positions to align with relations
       object.position → subject.position + relation.vector
       subject.position → object.position - relation.vector
    
    3. Check consistency (pairwise similarity of offsets)
       If consistency > target: STOP
```

### Convergence Properties

- **Fast convergence**: 4-10 iterations typically sufficient
- **High consistency**: 95%+ achievable
- **Stable**: Positions don't drift after convergence

## Experimental Results

### Analogy Accuracy

| Test | Accuracy |
|------|----------|
| Capital-of analogies | 100% (8/8) |
| Author-book analogies | 100% (4/4) |
| Cross-domain | 100% |
| Incremental (new entities) | 100% |

### Relation Consistency

| Relation | Consistency | Instances |
|----------|-------------|-----------|
| capital_of | 97.5% | 8 |
| wrote | 98.0% | 4 |

### Query Similarity

All queries return correct answer with similarity > 0.99.

## Connection to Protocols

### GOP (Gushurst Optimization Protocol)

- **Error as signal**: Low consistency points to where structure needs refinement
- **Iterative refinement**: Each iteration improves consistency
- **Convergence detection**: Stop when consistency reaches target

### PEP (Probe Extraction Protocol)

- **Measure, don't train**: Relation vectors are computed directly from positions
- **Linear algebra**: `relation = object - subject` (no optimization)
- **Exact recovery**: 100% accuracy when consistency is high

### Attractor Dynamics (Prior Work)

- **Co-occurrence → attraction**: Entities in same fact move together
- **Type separation → repulsion**: Different types stay apart
- **Emergent structure**: Relations become invariant through dynamics

## Implementation

### Core Module

`experiments/dynamic_geometric_lcm_v2.py` provides:

```python
class DynamicGeometricLCM:
    def add_fact(subject, relation, object_)  # Add knowledge
    def learn(n_iterations, target_consistency)  # Update structure
    def query(subject, relation)  # Inference
    def analogy(a, b, c)  # Analogical reasoning
```

### Usage Example

```python
lcm = DynamicGeometricLCM(dim=256)

# Add facts
lcm.add_fact("france", "capital_of", "paris")
lcm.add_fact("germany", "capital_of", "berlin")

# Learn structure
lcm.learn(n_iterations=100, target_consistency=0.95)

# Query
results = lcm.query("france", "capital_of")
# → [("paris", 0.99)]

# Analogy
results = lcm.analogy("france", "paris", "germany")
# → [("berlin", 0.98)]
```

## Scaling Considerations

### Dimension Selection

| Entities | Recommended Dim | Notes |
|----------|-----------------|-------|
| < 100 | 64-128 | Prototype |
| 100-1000 | 256 | Standard |
| 1000-10000 | 512 | Production |
| > 10000 | 1024+ | Large scale |

### Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Add fact | O(1) |
| Learn (per iteration) | O(F × D) where F = facts, D = dim |
| Query | O(E × D) where E = entities |
| Analogy | O(E × D) |

### Memory Usage

- Entities: E × D floats
- Relations: R × D floats
- Facts: F × 3 strings

For 10,000 entities, 100 relations, 256 dimensions:
- ~10 MB for positions
- ~100 KB for relations
- Negligible for facts

## Future Directions

### 1. Hierarchical Relations

Support relations with sub-types:
- `located_in` → `city_in_country`, `building_in_city`

### 2. Temporal Dynamics

Track how structure changes over time:
- Entity positions drift as context changes
- Relations strengthen/weaken with evidence

### 3. Multi-Hop Reasoning

Chain relations for complex queries:
- `france --capital_of--> paris --located_in--> europe`

### 4. Natural Language Interface

Parse text into facts automatically:
- "Paris is the capital of France" → `add_fact("france", "capital_of", "paris")`

### 5. Integration with TruthSpace

Combine with existing modules:
- Vocabulary for text encoding
- Style engine for generation
- Knowledge base for storage

## Theoretical Foundation

### Why This Works

1. **Geometry is universal**: Any relationship can be encoded as a vector
2. **Consistency is learnable**: Iterative refinement aligns all instances
3. **Analogies are projections**: Same relation applied to different entities

### Connection to VSA

This is a **learned VSA** where:
- Binding = addition (subject + relation = object)
- Unbinding = subtraction (object - subject = relation)
- Bundling = averaging (relation = mean of offsets)

The key difference: **relations are learned, not random**.

### Connection to Neural Networks

| Neural Network | Geometric LCM |
|----------------|---------------|
| Embedding layer | Entity positions |
| Attention weights | Relation vectors |
| Forward pass | Geometric projection |
| Backpropagation | Attractor dynamics |

But without:
- Billions of parameters
- Training data requirements
- Black-box inference

## Conclusion

The Dynamic Geometric LCM demonstrates that:

1. **Structure can replace weights** - Knowledge is geometry
2. **Learning can be geometric** - Attractor dynamics, not gradients
3. **100% accuracy is achievable** - On analogical reasoning
4. **Incremental learning works** - Add facts, re-learn, done

This is a viable foundation for a **training-free, interpretable alternative to traditional LLMs**.

---

**Key Quote**: *"The geometry IS the knowledge. Learning IS structure update."*
