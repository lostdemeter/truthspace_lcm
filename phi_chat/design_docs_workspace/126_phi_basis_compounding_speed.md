# Design Consideration 126: φ-Basis Compounding for Speed

## The Core Insight

From reverse engineering Qwen2's attention mechanism, we discovered:

```
actual_attention = φ_attention + sparse_E
```

Where:
- `φ_attention` = geometric basis (computable, no storage needed)
- `sparse_E` = error signal (45% can be zeroed with 99.9967% accuracy)

**Less data = faster compute.**

This principle applies universally: if we can represent information geometrically, we can derive it instead of storing it.

## The Memory-Bandwidth Problem

### What We Learned from Qwen2-7B

| Metric | Value |
|--------|-------|
| Model size | 15.2 GB |
| Tokens/sec | 30-45 |
| Bandwidth utilization | 69% |
| Bottleneck | Memory bandwidth |

The model is **memory-bound**: each token requires reading all 15.2 GB of weights. We're using 690 GB/s of the RTX 3090 Ti's 1000 GB/s peak bandwidth.

### The Solution: Store Less, Derive More

```
Traditional:     Store everything → Read everything → Slow
φ-Basis:         Store primitives → Derive compounds → Fast
```

## The Additive Error Paradigm

### In Attention

```python
# What we discovered:
actual_attention = phi_attention + E

# Where E is sparse:
E_sparse = E.copy()
E_sparse[|E| < 0.001] = 0  # 45% zeroed

# Reconstruction: 99.9971% accuracy
reconstructed = phi_attention + E_sparse
```

### In Knowledge Representation

```python
# The same principle:
actual_concept = phi_compound + correction

# Store only:
# 1. Platonic Ideals (origin points)
# 2. Transformation pairs (sparse corrections)
# 3. Dimension registry (emergent)

# Derive on-demand:
# - Compound positions (geometric)
# - Variations (φ-transform)
# - Relationships (distance in space)
```

## Storage Comparison

### Traditional Corpus

```
STORAGE: O(n²)
├── Every concept position
├── Every pairwise relationship
├── Every variation
└── Every compound

Example: 10,000 concepts
→ 10,000 positions
→ ~50,000,000 relationships
→ Massive storage, slow traversal
```

### φ-Basis Corpus

```
STORAGE: O(n)
├── Platonic Ideals (~100)
├── Transformation Pairs (~1,000)
└── Dimension Registry (~10)

Example: 10,000 concepts
→ 100 ideals stored
→ 1,000 pairs stored
→ 9,900 concepts DERIVED geometrically
→ 99% reduction in storage
```

## The Speed Gain

### Why Less Data = Faster

1. **Memory bandwidth**: Reading 100 ideals vs 10,000 concepts
2. **Cache efficiency**: Small data fits in L2/L3 cache
3. **Parallelism**: Geometric operations are SIMD-friendly
4. **Derivation is cheap**: φ-transforms are just multiply-add

### Benchmark Prediction

| Operation | Traditional | φ-Basis | Speedup |
|-----------|-------------|---------|---------|
| Load corpus | 100 ms | 1 ms | 100× |
| Find concept | 10 ms | 0.1 ms | 100× |
| Traverse relationship | 1 ms | 0.01 ms | 100× |
| Compound derivation | N/A | 0.001 ms | ∞ |

## Implementation Strategy

### Phase 1: Identify Primitives

```python
def identify_primitives(corpus):
    """Find the minimal set of concepts that span the space."""
    
    # 1. Find Platonic Ideals (multi-dimension anchors)
    ideals = find_ideals(corpus)
    
    # 2. Find transformation pairs (dimension definitions)
    pairs = find_pairs(corpus)
    
    # 3. Everything else is derivable
    derived = corpus - ideals - pairs
    
    return ideals, pairs, derived
```

### Phase 2: Lazy Derivation

```python
class PhiBasisCorpus:
    def __init__(self, ideals, pairs):
        self.ideals = ideals  # Small: ~100
        self.pairs = pairs    # Small: ~1,000
        self.cache = {}       # LRU cache for derived concepts
    
    def get_concept(self, word):
        # Check if it's a primitive
        if word in self.ideals:
            return self.ideals[word]
        
        # Check cache
        if word in self.cache:
            return self.cache[word]
        
        # Derive geometrically
        position = self.derive_position(word)
        self.cache[word] = position
        return position
    
    def derive_position(self, word):
        """Derive position from nearest ideal + transformations."""
        nearest_ideal = self.find_nearest_ideal(word)
        transformations = self.find_transformations(word, nearest_ideal)
        
        position = nearest_ideal.position.copy()
        for transform in transformations:
            position = apply_phi_transform(position, transform)
        
        return position
```

### Phase 3: Sparse Error Correction

```python
def add_sparse_correction(self, word, actual_position):
    """Store correction only if derivation is wrong."""
    
    derived = self.derive_position(word)
    error = actual_position - derived
    
    # Only store if error is significant
    if np.linalg.norm(error) > THRESHOLD:
        self.corrections[word] = error
    # Otherwise, derivation is good enough (like 45% zeroed in attention)
```

## Connection to Qwen2 Findings

| Qwen2 Attention | φ-Basis Corpus |
|-----------------|----------------|
| φ_attention (basis) | Platonic Ideals |
| sparse_E (error) | Transformation Pairs |
| 45% zeroed | 99% derived |
| 99.9971% accuracy | Self-similar transforms |
| Memory-bound → Compute-bound | Storage-bound → Derivation-bound |

## The Key Equation

```
SPEED = 1 / DATA_SIZE

Less data to read = Faster computation
Geometric derivation = Infinite concepts from finite primitives
```

## Success Criteria

1. **Storage reduction**: 99% less data than traditional corpus
2. **Speed improvement**: 100× faster concept lookup
3. **Accuracy preservation**: 99.99% match to full corpus
4. **Self-similarity**: All derived concepts follow φ-transforms

## Conclusion

The φ-basis compounding insight from Qwen2 attention applies universally:

1. **Store primitives** (the φ-basis)
2. **Store sparse corrections** (the error signal)
3. **Derive everything else** (geometric compounding)

This transforms a memory-bound system into a compute-bound system, which is always faster because:
- Memory bandwidth: ~1000 GB/s
- Compute throughput: ~100 TFLOPS
- Ratio: Compute is 100× cheaper than memory

**Less data, faster compute, same accuracy.**

## References

- Design 115: Self-Assembling Corpus Roadmap
- Design 124: φ-Transformer Replacement
- Experiment: `experiments/model_reverse_engineering/qwen2_additive_error_attention.py`
- Experiment: `experiments/model_reverse_engineering/qwen2_gpu_phi_attention.py`
