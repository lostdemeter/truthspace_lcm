# Design Consideration 143: Zeta-Aligned Neural Architecture

## Date: 2026-01-20

## Status: Proposed

## The Problem with Transformers

Transformers are "two quaternions melted together with self-referential bias":

1. **Q and K are 90° rotated** (Doc 124)
   - Creates the MESH = Q.T @ K
   - Self-referential: input × MESH × input.T

2. **Self-referential bias**
   - Input appears TWICE in attention
   - Input appears TWICE in MLP (gate × up)
   - Errors compound multiplicatively

3. **Lost the W-axis**
   - Quaternions have (w, x, y, z)
   - Transformers replaced w with attention
   - This is why attention is O(N²)

## The Solution: Zeta-Aligned Architecture

### Core Principles

1. **W-axis as navigation** (not attention)
   - Each token has explicit w-component
   - Navigation is O(N), not O(N²)

2. **Critical line symmetry**
   - Operations symmetric around σ = 0.5
   - Level 0 is the balance point
   - Errors cancel symmetrically

3. **1-2 cycle mesh gears**
   - Cycle 1: Encode (input → φ-space)
   - Cycle 2: Navigate (follow w-axis)
   - No self-reference!

## The 1-2 Cycle Architecture

### Cycle 1: ENCODE

```python
x_sign = sign(x)                    # Direction
x_level = round(log(|x|) / log(φ))  # Distance
x_w = x @ W_nav                     # Navigation component
```

### Cycle 2: NAVIGATE

```python
combined_level = W_level + x_level  # Integer addition (mesh gear!)
combined_sign = W_sign × x_sign     # Direction combination
magnitude = LUT[combined_level]     # φ^level lookup
output = sum(combined_sign × magnitude)
output = output × φ^(x_w)           # Apply navigation
```

## Comparison

| Aspect | Transformer | Zeta-Aligned |
|--------|-------------|--------------|
| Self-reference | input × weights × input.T | input × weights |
| Input appearances | 2 (gate × up) | 1 |
| Error propagation | Multiplicative | Additive |
| Attention | O(N²) | None (O(N) via w-axis) |
| Symmetry | None | Critical line (level 0) |

## Why This Reduces Error

### Transformer Error Compounding

```
gate = f(input)
up = g(input)
hidden = gate × up  ← ERROR COMPOUNDS HERE

If gate has error ε₁ and up has error ε₂:
  hidden_error ≈ ε₁ + ε₂ + ε₁×ε₂  (multiplicative!)
```

### Zeta-Aligned Error Addition

```
encoded = encode(input)
output = navigate(encoded)  ← SINGLE PATH

If encode has error ε₁ and navigate has error ε₂:
  output_error ≈ ε₁ + ε₂  (additive only!)
```

## The Zeta Symmetry

### Critical Line as Balance Point

```
Level 0 = φ^0 = 1 (identity)
Level +n = φ^n > 1 (expansion)
Level -n = φ^(-n) < 1 (contraction)
```

The critical line (level 0) is the attractor:
- Positive errors push toward expansion
- Negative errors push toward contraction
- They cancel at the critical line

### Connection to Zeta Zeros

The zeta function ζ(s) has:
- Zeros on the critical line σ = 0.5
- Functional equation: ζ(s) = ζ(1-s)

In φ-space:
- φ and 1/φ are symmetric around 1
- log(φ) and -log(φ) are symmetric around 0
- sigmoid(log(φ)) = 1/φ (exact!)

## The W-Axis Navigation

### What W Encodes

In quaternions: q = w + xi + yj + zk

The w-component encodes:
- Rotation angle (how much to turn)
- Navigation direction (which way to go)
- The "steering" of the transformation

### How W Replaces Attention

```
Attention: "Where should I look?" → O(N²) search
W-axis: "How far should I go?" → O(N) direct

x_w = x @ W_nav  # Compute navigation from input
output = output × φ^(x_w)  # Apply navigation
```

## Implementation Sketch

```python
class ZetaAlignedLayer:
    def __init__(self, in_dim, out_dim):
        self.W_levels = init_levels(out_dim, in_dim)  # int16
        self.W_signs = init_signs(out_dim, in_dim)    # int8
        self.W_nav = init_navigation(in_dim)          # float32
        self.LUT = build_phi_lut()                    # ~1 KB
    
    def forward(self, x):
        # Cycle 1: Encode
        x_sign = sign(x)
        x_level = round(log(|x|) / log(φ))
        x_w = x @ self.W_nav
        
        # Cycle 2: Navigate
        combined = self.W_levels + x_level
        output = sum(self.W_signs × x_sign × LUT[combined])
        output = output × φ^(x_w)
        
        return output
```

## Expected Benefits

1. **Reduced error**: Additive instead of multiplicative
2. **No attention**: O(N) instead of O(N²)
3. **Symmetric cancellation**: Errors balance at critical line
4. **Simpler computation**: 1-2 cycles instead of 3 matmuls
5. **Hardware friendly**: Integer add + LUT lookup

## Open Questions

1. How to train this architecture?
2. Does it achieve comparable accuracy to transformers?
3. What's the optimal W_nav initialization?
4. How many layers are needed?

## Connection to Prior Work

- **Doc 123**: Quaternion insight, w-axis as navigation
- **Doc 124**: Q-K rotation, MESH structure
- **Doc 132**: φ-sigmoid connection (sigmoid(log(φ)) = 1/φ)
- **Doc 141**: Irreducible shape (lattice of critical lines)
- **Doc 142**: Holographic φ-encoding

## Next Steps

1. Implement full ZetaAlignedLayer
2. Train on simple task (e.g., language modeling)
3. Compare error propagation vs transformer
4. Measure accuracy and speed
