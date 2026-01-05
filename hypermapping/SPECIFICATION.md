# HyperSpace: A Hyperdimensional Data Structure

## Formal Specification

### 1. Definition

A **HyperSpace** is a data structure that maps keys to values through an intermediate N-dimensional geometric space.

```
HyperSpace<K, V> = {
    dims: ℕ                           # Number of dimensions
    nodes: Map<NodeId, HyperNode<V>>  # Storage
    codec: HyperCodec<K>              # Key encoder/decoder
}

HyperNode<V> = {
    id: String                        # Unique identifier
    position: ℝⁿ                      # N-dimensional coordinates
    value: V                          # Stored value
}

HyperCodec<K> = {
    encode: K → ℝⁿ                    # Key to position
    decode: ℝⁿ × List<HyperNode> → K  # Position to key
}
```

### 2. Core Operations

| Operation | Signature | Complexity | Description |
|-----------|-----------|------------|-------------|
| `add` | `(K, V) → HyperNode` | O(1) | Add key-value pair |
| `get` | `K → V` | O(1) | Get value by exact key |
| `query` | `K × ℕ → List<(K, V, ℝ)>` | O(n) / O(log n)* | Find k nearest neighbors |
| `remove` | `K → ()` | O(1) | Remove by key |
| `feedback` | `K × Bool → ()` | O(1) | Learning signal |
| `attract` | `K × K → ()` | O(1) | Move keys closer |
| `repel` | `K × K → ()` | O(1) | Move keys apart |

*O(log n) with spatial indexing (k-d tree, ball tree)

### 3. Similarity Metric

The default similarity metric is **cosine similarity**:

```
similarity(p₁, p₂) = (p₁ · p₂) / (‖p₁‖ × ‖p₂‖)
```

Range: [-1, 1] where 1 = identical direction, -1 = opposite, 0 = orthogonal

### 4. Learning Dynamics

Learning is implemented via position movement:

```
# Success: Attract toward query
position ← position + α × (query_position - position)

# Failure: Repel from query
position ← position + β × (position - query_position)

# Where α > β (attraction stronger than repulsion)
```

This implements **attractor/repeller dynamics** where:
- Similar concepts converge to same position
- Dissimilar concepts diverge to different positions

### 5. Critical Line

The **critical line** (σ = 0.5) from the Riemann zeta function serves as a stability boundary:

```
magnitude(position) = ‖position‖

persists = magnitude ≥ CRITICAL_LINE (0.5)
```

Nodes below the critical line are "temporary" and may be pruned.

### 6. Codecs

A **HyperCodec** defines how domain values map to/from positions:

| Codec | Domain | Encoding | Use Case |
|-------|--------|----------|----------|
| `IdentityCodec` | ℝⁿ | Identity | Direct position manipulation |
| `HashCodec` | String | Hash → random | Key-value storage |
| `TextCodec` | String | Word co-occurrence | Semantic similarity |
| `NumericCodec` | ℝᵐ | Linear projection | Numeric data |
| `CategoricalCodec` | Enum | One-hot → position | Classification |

### 7. Chaining (Pipeline)

Multiple HyperSpaces can be chained:

```
pipeline = space₁ >> space₂ >> space₃

process(input):
    result₁ = space₁.query(input)
    result₂ = space₂.query(result₁.value)
    result₃ = space₃.query(result₂.value)
    return result₃
```

With routing:
```
router(current_index, result) → next_index
```

### 8. Serialization

HyperSpace serializes to JSON:

```json
{
    "type": "HyperSpace",
    "version": "1.0",
    "dims": 8,
    "nodes": {
        "node_id": {
            "id": "node_id",
            "position": [0.1, 0.2, ...],
            "value": "stored_value"
        }
    }
}
```

### 9. Comparison to Traditional Data Structures

| Property | Dict | List | Tree | HyperSpace |
|----------|------|------|------|------------|
| Access | O(1) by key | O(1) by index | O(log n) | O(1) exact, O(n) similar |
| Search | O(n) | O(n) | O(log n) | O(n) / O(log n)* |
| Insert | O(1) | O(1) / O(n) | O(log n) | O(1) |
| Delete | O(1) | O(n) | O(log n) | O(1) |
| Similarity | ❌ | ❌ | ❌ | ✓ |
| Learning | ❌ | ❌ | ❌ | ✓ |
| Continuous | ❌ | ❌ | ❌ | ✓ |

### 10. Mathematical Foundation

HyperSpace is based on:

1. **Holographic Projection** - Eigendecomposition of similarity matrix
2. **Attractor Dynamics** - Self-organizing position movement
3. **Critical Line** - Stability boundary from zeta function
4. **Cosine Similarity** - Angular distance in high-dimensional space

### 11. Use Cases

| Problem | Traditional | HyperSpace |
|---------|-------------|------------|
| Key-value storage | Dict | HyperSpace + HashCodec |
| Semantic search | Embeddings + Vector DB | HyperSpace + TextCodec |
| Classification | Neural network | HyperSpace + CategoricalCodec |
| Recommendation | Collaborative filtering | HyperSpace + feedback |
| Translation | Seq2seq model | HyperPipeline |

### 12. API Summary

```python
# Creation
space = HyperSpace(dims=8, codec=TextCodec(8))

# Dict-like operations
space[key] = value          # Add
value = space[key]          # Get
del space[key]              # Remove
key in space                # Contains
len(space)                  # Size
for k, v in space.items()   # Iterate

# Geometric operations
matches = space.query(key, k=5)     # Find similar
nearest = space.nearest(key)         # Find closest

# Learning
space.feedback(key, success=True)    # Reinforce
space.attract(key1, key2)            # Move closer
space.repel(key1, key2)              # Move apart

# Chaining
pipeline = space1 >> space2 >> space3
result = pipeline.process(input)

# Persistence
space.save("path.json")
space = HyperSpace.load("path.json")
```

---

## Design Philosophy

1. **Structure IS Information** - Positions encode relationships
2. **Geometry IS Computation** - Similarity queries are the computation
3. **Learning IS Movement** - Feedback moves positions
4. **Codecs ARE Interfaces** - Domain-specific encoding/decoding

## Relationship to Neural Networks

HyperSpace can solve problems typically requiring neural networks:

| Neural Network | HyperSpace Equivalent |
|----------------|----------------------|
| Embedding layer | Codec.encode() |
| Attention | query() with similarity |
| Feedforward | Pipeline processing |
| Backpropagation | feedback() |
| Weights | Node positions |

The key difference: **Positions are explicit and interpretable**, not opaque weights.
