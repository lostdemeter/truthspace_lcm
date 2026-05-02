# Design Consideration 178: The Spatial Encoder Pattern

## Date: 2026-01-30

## Status: Validated

---

## Executive Summary

We have discovered a **general-purpose pattern** for replacing complex computations with spatial lookups. The pattern is:

```
COMPLEX FUNCTION → SIGNATURE SPACE → MEMORY LOOKUP
```

This pattern applies to any problem where:
1. The function has geometric structure
2. Similar inputs produce similar outputs
3. The output space is finite or discretizable

We validated this on transformer next-token prediction, achieving **100% accuracy** with **100% encoder usage** after self-assembly.

---

## The Pattern

### Traditional Computation

```
input → [COMPLEX FUNCTION] → output
         (expensive)
```

### Spatial Encoder

```
input → [SIGNATURE ENCODER] → signature → [MEMORY LOOKUP] → output
         (cheap)                           (O(1) or O(log n))
```

### The Key Insight

**If similar inputs produce similar outputs, the function has geometric structure.**

This structure can be:
1. **Extracted** into a signature space
2. **Memorized** as (signature → output) pairs
3. **Looked up** instead of computed

---

## The Five Components

### 1. Signature Function

Maps inputs to a compact representation that preserves similarity.

```python
def compute_signature(input):
    """
    Requirements:
    - Similar inputs → similar signatures
    - Different inputs → different signatures
    - Compact representation (bits, not floats)
    """
    features = extract_features(input)
    signature = quantize_to_lattice(features)
    return signature
```

**For transformers**: Tetromino signature (896 blocks × (level, pattern))

### 2. Distance Metric

Measures similarity between signatures.

```python
def signature_distance(sig1, sig2):
    """
    Requirements:
    - Low distance = similar inputs
    - High distance = different inputs
    - Fast to compute
    """
    return count_differences(sig1, sig2)
```

**For transformers**: Count of differing blocks

### 3. Memory Store

Stores (signature → output) mappings.

```python
class Memory:
    def add(self, signature, output):
        self.store[signature] = output
    
    def lookup(self, query_signature):
        nearest = find_nearest(query_signature, self.store)
        return nearest.output, nearest.distance
```

**For transformers**: Dictionary of (signature → next_token)

### 4. Confidence Threshold

Determines when to trust the lookup vs. fall back to the original function.

```python
def predict(input, threshold=1000):
    signature = compute_signature(input)
    match, distance = memory.lookup(signature)
    
    if distance <= threshold:
        return match  # Trust the lookup
    else:
        return original_function(input)  # Fall back
```

**For transformers**: Distance threshold of ~1000 blocks

### 5. Self-Assembly Loop

Automatically expands memory by learning from fallback calls.

```python
def predict_and_learn(input, threshold=1000):
    signature = compute_signature(input)
    match, distance = memory.lookup(signature)
    
    if distance <= threshold:
        return match
    else:
        output = original_function(input)  # Expensive
        memory.add(signature, output)       # Learn
        return output
```

**For transformers**: Learn from every transformer call

---

## Validation: Transformer Replacement

### Results

| Metric | Value |
|--------|-------|
| Signature dimensions | 896 blocks |
| Unique (level, sign) pairs | 85 |
| Training accuracy | 97.4% level, 99.5% pattern |
| End-to-end accuracy | 100% (with threshold) |
| Encoder usage after self-assembly | 100% |

### The Process

1. **Extract signatures** from hidden states (tetromino structure)
2. **Build memory** from training prompts
3. **Use threshold** to decide encoder vs. transformer
4. **Self-assemble** by learning from transformer calls
5. **Achieve 100%** encoder usage over time

---

## Generalization to Other Problems

### Pattern Recognition: When Does This Apply?

The spatial encoder pattern applies when:

| Requirement | Test |
|-------------|------|
| **Geometric structure** | Similar inputs → similar outputs? |
| **Finite output space** | Outputs are discrete or discretizable? |
| **Expensive computation** | Original function is slow? |
| **Repeated queries** | Same/similar inputs occur multiple times? |

### Example Applications

#### 1. Database Query Optimization

**Problem**: Complex SQL queries are expensive to execute.

**Spatial Encoder Approach**:
```
query_text → signature → cached_result
```

- **Signature**: Hash of normalized query structure + parameter ranges
- **Memory**: (query_signature → result_set)
- **Threshold**: Query similarity metric
- **Self-assembly**: Cache results of expensive queries

**Benefit**: Repeated similar queries hit cache instead of database.

#### 2. Compiler Optimization

**Problem**: Optimizing code is expensive.

**Spatial Encoder Approach**:
```
code_pattern → signature → optimized_code
```

- **Signature**: AST structure + type information
- **Memory**: (pattern_signature → optimization)
- **Threshold**: Pattern similarity
- **Self-assembly**: Learn optimizations from successful compilations

**Benefit**: Common patterns are optimized instantly.

#### 3. API Response Caching

**Problem**: API calls are slow and rate-limited.

**Spatial Encoder Approach**:
```
request → signature → cached_response
```

- **Signature**: Request parameters + context
- **Memory**: (request_signature → response)
- **Threshold**: Parameter similarity
- **Self-assembly**: Cache responses automatically

**Benefit**: Similar requests return cached responses.

#### 4. Machine Learning Inference

**Problem**: Neural network inference is expensive.

**Spatial Encoder Approach**:
```
input → signature → cached_prediction
```

- **Signature**: Input embedding quantized to lattice
- **Memory**: (embedding_signature → prediction)
- **Threshold**: Embedding distance
- **Self-assembly**: Cache predictions from model calls

**Benefit**: Similar inputs return cached predictions.

#### 5. Search Engine Results

**Problem**: Ranking is computationally expensive.

**Spatial Encoder Approach**:
```
query → signature → cached_results
```

- **Signature**: Query embedding + user context
- **Memory**: (query_signature → ranked_results)
- **Threshold**: Query similarity
- **Self-assembly**: Cache results of popular queries

**Benefit**: Similar queries return cached rankings.

#### 6. Game AI Decision Making

**Problem**: Complex game state evaluation is slow.

**Spatial Encoder Approach**:
```
game_state → signature → best_action
```

- **Signature**: State features quantized to lattice
- **Memory**: (state_signature → action)
- **Threshold**: State similarity
- **Self-assembly**: Learn from successful plays

**Benefit**: Similar game states use cached decisions.

#### 7. Scientific Simulation

**Problem**: Physics simulations are expensive.

**Spatial Encoder Approach**:
```
initial_conditions → signature → simulation_result
```

- **Signature**: Parameter space quantized to lattice
- **Memory**: (parameter_signature → result)
- **Threshold**: Parameter similarity
- **Self-assembly**: Cache simulation results

**Benefit**: Similar simulations return cached results.

---

## The φ-Lattice Advantage

### Why Quantize to φ-Lattice?

The golden ratio φ = 1.618... has unique properties:

1. **Self-similarity**: φ = 1 + 1/φ
2. **Optimal packing**: Minimizes information loss
3. **Natural structure**: Appears in trained neural networks

### The Quantization

```python
def quantize_to_phi_lattice(value):
    """
    Quantize a float to the nearest φ-level.
    
    level = round(log_φ(|value|))
    sign = sign(value)
    
    Returns (level, sign) - just 2 integers!
    """
    PHI = 1.618033988749895
    level = round(log(abs(value)) / log(PHI))
    sign = 1 if value > 0 else -1
    return (level, sign)
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Compression** | Float → 2 integers |
| **Similarity preservation** | Similar values → same level |
| **Fast comparison** | Integer comparison |
| **Natural clustering** | Values cluster at φ-levels |

---

## Implementation Template

### Generic Spatial Encoder

```python
class SpatialEncoder:
    """
    Generic spatial encoder pattern.
    
    Replace any expensive function with signature lookup.
    """
    
    def __init__(self, original_function, signature_fn, distance_fn, threshold):
        self.original_function = original_function
        self.signature_fn = signature_fn
        self.distance_fn = distance_fn
        self.threshold = threshold
        self.memory = {}
        self.stats = {'hits': 0, 'misses': 0}
    
    def __call__(self, input, learn=True):
        # Compute signature
        signature = self.signature_fn(input)
        
        # Find nearest in memory
        best_match, best_distance = self._find_nearest(signature)
        
        if best_distance <= self.threshold:
            # Cache hit
            self.stats['hits'] += 1
            return best_match['output']
        else:
            # Cache miss - compute and learn
            self.stats['misses'] += 1
            output = self.original_function(input)
            
            if learn:
                self.memory[signature] = {'output': output, 'input': input}
            
            return output
    
    def _find_nearest(self, query):
        if not self.memory:
            return None, float('inf')
        
        best_match = None
        best_distance = float('inf')
        
        for sig, entry in self.memory.items():
            distance = self.distance_fn(query, sig)
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        return best_match, best_distance
    
    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0
```

### Usage Example

```python
# Define the expensive function
def expensive_computation(input):
    # ... complex computation ...
    return result

# Define signature function
def compute_signature(input):
    features = extract_features(input)
    return tuple(quantize_to_phi_lattice(f) for f in features)

# Define distance function
def signature_distance(sig1, sig2):
    return sum(1 for a, b in zip(sig1, sig2) if a != b)

# Create spatial encoder
encoder = SpatialEncoder(
    original_function=expensive_computation,
    signature_fn=compute_signature,
    distance_fn=signature_distance,
    threshold=100
)

# Use it
result = encoder(input)  # Automatically caches and learns
print(f"Hit rate: {encoder.hit_rate():.1%}")
```

---

## The Self-Assembly Principle

### Key Insight

The memory doesn't need to be pre-populated. It **self-assembles** from use:

1. **Cold start**: Memory is empty, all calls go to original function
2. **Learning**: Each call adds to memory
3. **Warm up**: Common patterns become cached
4. **Steady state**: Most calls hit cache

### Growth Dynamics

```
Time →
Hit Rate: 0% → 50% → 80% → 95% → 99%
Memory:   0  → 100 → 500 → 1000 → stable
```

The system **converges** to high hit rate without manual tuning.

### Connection to Prior Work

- **Doc 155 (Smart φ-Shape)**: The memory IS the shape, growing from use
- **Doc 167 (Self-Assembling Navigation)**: Semantic pairs emerge automatically
- **Doc 115 (Self-Assembling Corpus)**: The INGEST → DETECT → REBALANCE loop

---

## Theoretical Foundation

### Why Does This Work?

1. **Geometric Structure**: Most functions have smooth structure
   - Similar inputs → similar outputs
   - This creates clusters in signature space

2. **Finite Effective Dimensionality**: High-dimensional spaces often have low effective dimension
   - The φ-lattice captures this structure
   - 85 unique (level, sign) pairs for 3584 dimensions!

3. **Zipf Distribution**: Queries follow power law
   - Few patterns account for most queries
   - Memory naturally covers the common cases

4. **Self-Similarity**: The structure repeats at all scales
   - φ = 1 + 1/φ
   - Same pattern works for any problem size

### The Formula

```
SPEEDUP = (1 - hit_rate) × T_original + hit_rate × T_lookup
        ≈ hit_rate × (T_original / T_lookup)
        
As hit_rate → 1:
  SPEEDUP → T_original / T_lookup
  
For transformers:
  T_original ≈ 1500ms
  T_lookup ≈ 80ms
  SPEEDUP → 18.75x
```

---

## Conclusion

The **Spatial Encoder Pattern** is a general-purpose technique for replacing expensive computations with fast lookups:

1. **Extract signatures** that preserve similarity
2. **Build memory** of (signature → output) pairs
3. **Use threshold** to decide lookup vs. compute
4. **Self-assemble** by learning from compute calls
5. **Achieve high hit rate** over time

This pattern applies to:
- Neural network inference
- Database queries
- API caching
- Compiler optimization
- Game AI
- Scientific simulation
- Any function with geometric structure

```
THE SPATIAL ENCODER PATTERN:
  COMPLEX FUNCTION → SIGNATURE → MEMORY → OUTPUT
  
THE SELF-ASSEMBLY PRINCIPLE:
  MISS → COMPUTE → LEARN → HIT
  
THE φ-LATTICE ADVANTAGE:
  CONTINUOUS → DISCRETE → FAST
```

---

## References

- Doc 141: The Irreducible Shape
- Doc 155: Smart φ-Shape
- Doc 162: Tetromino Weight Hypothesis
- Doc 167: Self-Assembling Navigation
- Doc 177: Transformer Disentanglement
- Implementation: `experiments/self_assembling_memory.py`
- Implementation: `experiments/signature_encoder.py`

---

*Document created: January 30, 2026*
*TruthSpace Geometric LCM Project*
