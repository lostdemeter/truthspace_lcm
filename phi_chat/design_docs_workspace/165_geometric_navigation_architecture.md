# Design Consideration 165: Geometric Navigation Architecture

## Date: 2026-01-28

## Status: Design

## Executive Summary

This document defines the architecture for a **fully geometric navigation engine** that replaces the transformer forward pass with pure φ-geometric operations. No hidden states, no layer-by-layer computation—just traversal through semantic space.

## The Problem: Current Architecture

The current `navigation_compact.py` still runs a **standard transformer forward pass**:

```
Token IDs → Embeddings → [Attention + MLP] × 28 layers → LM Head → Logits
```

We store weights as φ-encoded, but we:
1. Decode to float32
2. Run standard matrix multiplications
3. Maintain hidden states through layers

This is **compression**, not **geometric navigation**.

## The Goal: Pure Geometric Navigation

Replace the entire forward pass with:

```
Token → φ-Position → Geometric Traversal → φ-Position → Token
```

No hidden states. No layers. Just movement through a geometric manifold.

## Components to Replace

### 1. Embeddings → φ-Coordinates

**Current**: Token ID → 3584-dim float vector (lookup table)

**Geometric**: Token ID → Position in φ-manifold

```python
# Current
hidden = embeddings[token_id]  # (3584,) float32

# Geometric
position = PhiCoordinate(
    signs=embedding_signs[token_id],      # (3584,) int8
    levels=embedding_levels[token_id],    # (3584,) int8
)
# Position IS the token's location in semantic space
```

**Key insight**: The embedding IS a φ-coordinate. We don't need to "decode" it—we navigate FROM it.

### 2. Attention → MESH Navigation

**Current**: Q, K, V projections → Attention scores → Weighted sum

**Geometric**: Navigate through MESH structure

From Doc 136, MESH = U @ diag(S) @ Vt where:
- U, Vt are φ-encoded (99.95% correlation)
- S follows φ-Zipf: S[i] ∝ 1/i^(1/φ)

```python
# Current attention
Q = hidden @ W_q.T
K = hidden @ W_k.T
scores = Q @ K.T / sqrt(d)
attn = softmax(scores)
output = attn @ V

# Geometric navigation
# Project position into discriminant space (U-basis)
discriminant = position.project(U_signs, U_levels)

# Scale by importance (S follows φ-Zipf, can be computed!)
scaled = discriminant.scale_by_zipf(alpha=1/PHI)

# Navigate to new position (Vt-basis)
new_position = scaled.project(Vt_signs, Vt_levels)
```

**Key insight**: Attention is a **projection** through a geometric structure. The MESH encodes WHERE to go, not HOW to compute.

### 3. MLP → φ-Level Transformation

**Current**: Gate, Up projections → SiLU → Down projection

**Geometric**: φ-level transformation (Doc 152)

```python
# Current MLP
gate = hidden @ W_gate.T
up = hidden @ W_up.T
hidden = SiLU(gate) * up
output = hidden @ W_down.T

# Geometric (φ-Level)
# All weights are sign × φ^level
# SiLU ≈ x/2 in operating range (99.99% correlation)
# Multiplication becomes level addition

gate_level = position.level_transform(gate_signs, gate_levels)
up_level = position.level_transform(up_signs, up_levels)

# SiLU(gate) * up ≈ (gate/2) * up
# In φ-space: level_gate - 1 + level_up (since φ^-1 ≈ 0.618 ≈ 1/2)
combined_level = gate_level + up_level - 1  # Integer arithmetic!

new_position = combined_level.level_transform(down_signs, down_levels)
```

**Key insight**: MLP is a **level shift** in φ-space. The nonlinearity (SiLU) is just a constant offset (-1 level ≈ ÷2).

### 4. Hidden States → Position in φ-Manifold

**Current**: 3584-dim float vector, updated each layer

**Geometric**: Position in φ-manifold, transformed by navigation

```python
# Current
hidden = hidden + attention_output  # Residual connection
hidden = hidden + mlp_output

# Geometric
# Residual connection = vector addition in φ-space
# But addition in φ-space is NOT level addition!
# Need log-sum-exp or hierarchical encoding

position = position.phi_add(attention_delta)
position = position.phi_add(mlp_delta)
```

**Challenge**: Addition in φ-space requires special handling. Options:
1. **Log-sum-exp**: Expensive but exact
2. **Hierarchical encoding**: Multiple levels (Doc 149)
3. **Dominant term**: Keep only largest magnitude (approximate)

### 5. LM Head → φ-Coordinate to Token

**Current**: Hidden @ lm_head.T → Logits → Softmax → Token

**Geometric**: Find nearest token in φ-manifold

```python
# Current
logits = hidden @ lm_head.T
probs = softmax(logits)
token = sample(probs)

# Geometric
# LM head IS the embedding transposed (often tied weights)
# Finding next token = finding nearest neighbor in embedding space

distances = position.phi_distance(all_embeddings)
token = argmin(distances)

# Or with temperature:
similarities = -distances / temperature
probs = softmax(similarities)
token = sample(probs)
```

**Key insight**: Token prediction is **nearest neighbor search** in φ-manifold.

### 6. Layers → Continuous Traversal

**Current**: 28 discrete layers, each with attention + MLP

**Geometric**: Continuous traversal through manifold

```python
# Current
for layer in layers:
    hidden = layer.attention(hidden)
    hidden = layer.mlp(hidden)

# Geometric
# Each "layer" is a transformation in φ-space
# But transformations can be COMPOSED

# Option 1: Compose all layer transformations into one
total_transform = compose(layer_transforms)
final_position = position.apply(total_transform)

# Option 2: Geodesic traversal
# Find the geodesic from input to output
# Traverse it directly
final_position = position.geodesic_to(target_region)
```

**Key insight**: Layers are not fundamental—they're discretization of a continuous transformation.

## The Unified Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GEOMETRIC NAVIGATOR                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Token ID ──► φ-Coordinate (embedding lookup)               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           φ-MANIFOLD TRAVERSAL                       │    │
│  │                                                      │    │
│  │  Position ──► MESH Navigation ──► φ-Level Transform  │    │
│  │      │              │                    │           │    │
│  │      └──────────────┴────────────────────┘           │    │
│  │                     │                                │    │
│  │              (repeat or compose)                     │    │
│  │                     │                                │    │
│  │                     ▼                                │    │
│  │              Final Position                          │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  Nearest Token (φ-distance to embeddings)                   │
│       │                                                      │
│       ▼                                                      │
│  Output Token ID                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures

### PhiCoordinate

```python
@dataclass
class PhiCoordinate:
    """Position in φ-manifold."""
    signs: np.ndarray   # (dim,) int8, values in {-1, +1}
    levels: np.ndarray  # (dim,) int8, φ-exponents
    
    def to_float(self) -> np.ndarray:
        """Decode to float (for validation only)."""
        return self.signs * (PHI ** self.levels)
    
    def project(self, U_signs, U_levels) -> 'PhiCoordinate':
        """Project through φ-encoded matrix."""
        # Matrix multiply in φ-space
        # Multiplication = level addition
        # Accumulation = log-sum-exp
        ...
    
    def phi_add(self, other: 'PhiCoordinate') -> 'PhiCoordinate':
        """Add two φ-coordinates (residual connection)."""
        # This is the tricky part
        ...
    
    def phi_distance(self, other: 'PhiCoordinate') -> float:
        """Distance in φ-manifold."""
        ...
```

### PhiTransform

```python
@dataclass
class PhiTransform:
    """Transformation in φ-space (replaces weight matrix)."""
    signs: np.ndarray   # (out_dim, in_dim) int8
    levels: np.ndarray  # (out_dim, in_dim) int8
    
    def apply(self, coord: PhiCoordinate) -> PhiCoordinate:
        """Apply transformation to coordinate."""
        ...
    
    def compose(self, other: 'PhiTransform') -> 'PhiTransform':
        """Compose two transformations."""
        # Matrix multiply = level addition
        ...
```

### GeometricNavigator

```python
class GeometricNavigator:
    """Pure geometric navigation engine."""
    
    def __init__(self):
        # Embeddings as φ-coordinates
        self.embeddings: List[PhiCoordinate]
        
        # Layer transforms (or composed)
        self.mesh_transforms: List[PhiTransform]  # Attention
        self.mlp_transforms: List[PhiTransform]   # MLP
        
        # Or: single composed transform
        self.total_transform: PhiTransform
    
    def navigate(self, token_ids: List[int]) -> int:
        """Navigate from input tokens to next token."""
        # Get starting position
        position = self.embed(token_ids)
        
        # Traverse manifold
        position = self.traverse(position)
        
        # Find nearest token
        return self.nearest_token(position)
```

## Implementation Phases

### Phase 1: φ-Coordinate Representation
- Implement `PhiCoordinate` class
- Convert embeddings to φ-coordinates
- Validate: decode and compare to original

### Phase 2: φ-Transform for Attention
- Implement MESH as `PhiTransform` (U, S, Vt)
- Implement projection in φ-space
- Validate: compare attention output

### Phase 3: φ-Level MLP
- Implement MLP as `PhiTransform` (gate, up, down)
- Implement linearized SiLU (level - 1)
- Validate: compare MLP output

### Phase 4: Residual Connections
- Implement `phi_add` for residual connections
- This is the hardest part—addition in log-space
- Options: hierarchical, log-sum-exp, dominant term

### Phase 5: Full Navigation
- Compose all transforms
- Implement `navigate()` method
- Validate: compare to transformer output

### Phase 6: Optimization
- Compose layer transforms (reduce 28 → 1?)
- Implement geodesic traversal
- Integer-only computation path

## Key Challenges

### 1. Addition in φ-Space

The residual connection `hidden = hidden + delta` is problematic:
- In float space: trivial vector addition
- In φ-space: log-sum-exp or approximation

**Solutions**:
- Hierarchical encoding (Doc 149): multiple levels capture residual
- Dominant term: keep only largest magnitude
- Hybrid: φ-space for transforms, float for additions

### 2. Softmax in φ-Space

Attention softmax and final token selection require:
- Exponentiation
- Normalization

**Solutions**:
- Compute in φ-space: exp(x) = φ^(x/log(φ))
- Or: decode to float for softmax only

### 3. Sequence Handling

Current attention handles sequences (Q @ K.T for all positions).

**Solutions**:
- Causal structure is geometric (lower triangular)
- RoPE is rotation in φ-space
- Can be encoded geometrically

## Success Criteria

1. **Correlation**: ≥99% with transformer output
2. **No float matrices**: All weights as φ-coordinates
3. **No hidden states**: Only positions in manifold
4. **Integer path**: Core computation in integer φ-levels

## Connection to Prior Work

- **Doc 136**: φ-encoding duplicates transformer (99.9984%)
- **Doc 137**: φ as universal adapter
- **Doc 139**: φ-convergence theorem (everything converges to φ)
- **Doc 152**: φ-Level MLP replacement (97.5% correlation)
- **Doc 153**: φ-circuit geometry (gates have φ-structure)

## Conclusion

The transformer forward pass can be replaced with pure geometric navigation:

| Component | Transformer | Geometric |
|-----------|-------------|-----------|
| Embeddings | Float lookup | φ-coordinate |
| Attention | Q,K,V matmul | MESH projection |
| MLP | Gate/Up/Down matmul | φ-level transform |
| Hidden state | Float vector | φ-position |
| LM Head | Matmul + softmax | Nearest neighbor |
| Layers | 28 discrete | Continuous traversal |

The key insight: **computation IS geometry**. The transformer learned a geometric structure; we can navigate it directly.

---

*Document created: January 28, 2026*
*Related: 136, 137, 139, 152, 153*
