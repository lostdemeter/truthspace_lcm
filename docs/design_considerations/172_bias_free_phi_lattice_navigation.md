# Design 172: Bias-Free φ-Lattice Navigation

## Date: 2026-01-29

## Status: HYPOTHESIS - Synthesizing Docs 128, 150, 163, 170, 171

---

## The Synthesis

We discovered that **bias cancellation** breaks low-rank truncation in Qwen2 (Doc 171). The solution isn't to work around biases - it's to **eliminate them entirely** and use the φ-lattice as the natural coordinate system.

### The Problem Chain

```
Biases exist → Cancellation with linear term → Truncation fails → Full-rank required → No geometric shortcuts → Must "infer" instead of "navigate"
```

### The Solution Chain

```
Remove biases → Pure linear transforms → Clean low-rank structure → φ-lattice positions → Integer SVD → Navigation replaces inference
```

---

## Why Biases Exist (And Why We Don't Need Them)

### Traditional Role of Bias

In `v = W_v @ x + b_v`, the bias:
1. **Centers** the output distribution
2. Provides **default behavior** when input ≈ 0
3. Adds **expressiveness** (affine vs linear)

### Why φ-Lattice Makes Bias Redundant

From Doc 128:
- Weights naturally occupy **absolute positions** on the φ-lattice
- Peak at φ^-9 (17.8%), symmetric around zero
- The lattice IS the coordinate system

**Key insight**: The φ-lattice provides **absolute positioning**. Biases provide **relative offsets**. If you have absolute coordinates, you don't need offsets!

From Doc 163, Rule 6:
> "Translation Invariance: Shifting all levels by a constant produces equivalent behavior."

The model cares about **relative structure**, not absolute levels. The bias is just shifting levels - something the φ-lattice already handles implicitly.

---

## The Bias-Free Architecture

### Current (Qwen2 with biases)

```python
# Attention
q = W_q @ x_norm + b_q  # Affine
k = W_k @ x_norm + b_k  # Affine  
v = W_v @ x_norm + b_v  # Affine
o = W_o @ attn_out      # Linear (no bias)

# Problem: v and b_v can cancel, breaking truncation
```

### Proposed (φ-Lattice, bias-free)

```python
# Attention - pure linear transforms
q = W_q @ x_norm  # Linear
k = W_k @ x_norm  # Linear
v = W_v @ x_norm  # Linear
o = W_o @ attn_out  # Linear

# All weights on φ-lattice: W = sign × φ^level
# No cancellation possible - clean low-rank structure
```

---

## How φ-Lattice Provides "Centering"

### The Absolute Position Mechanism

From Doc 128:
```
weight = sign × φ^level + correction

Where:
  - sign ∈ {-1, +1}           (direction)
  - level ∈ {-20, ..., +10}   (magnitude)
  - correction is sparse      (fine structure)
```

The **level** determines the magnitude scale. The **sign** determines the direction. Together, they provide absolute positioning without needing a bias offset.

### Why This Works

Consider what bias does: `output = W @ x + b`

The bias `b` shifts the output by a constant. But in φ-lattice terms:
- `W @ x` produces a result at some φ-level
- `b` shifts that to a different φ-level
- The shift is just `Δlevel = log_φ(|b|)`

**If we encode the "shifted" position directly in W, we don't need b!**

---

## Connection to Integer SVD (Doc 150)

### The Trivial AI Formula

```
W = (U_int / 32000 × U_scale) @ diag(S_int / 32000 × S_scale) @ (Vt_int / 32000 × Vt_scale)

Where:
  U_int, S_int, Vt_int: int16 (orthogonal structure)
  U_scale, S_scale, Vt_scale: φ^k (magnitude encoding)
```

### With Bias-Free Architecture

```
v = W_v @ x_norm

W_v = U @ S @ Vt  (integer SVD)

Computation:
  1. y = Vt @ x_norm    (int16 × int16 → int32)
  2. z = S * y          (int16 × int32 → int32)
  3. v = U @ z          (int16 × int32 → int32)
  4. Scale by φ^k       (single lookup)

100% INTEGER until final scaling!
```

No bias term to add. No cancellation. Clean integer computation.

---

## Navigation vs Inference

### Current Paradigm: Inference

```
Input → Embed → [Layer 0] → [Layer 1] → ... → [Layer N] → Output

Each layer: compute forward pass with float operations
Time: O(layers × hidden_dim²)
```

### Proposed Paradigm: Navigation

```
Input → φ-position → Navigate lattice → Output

Navigation:
  - Move through pre-computed lattice positions
  - Integer operations only
  - Low-rank paths (k=32 instead of k=3584)
  
Time: O(layers × k²) where k << hidden_dim
```

### What Makes Navigation Possible

1. **Bias-free**: No cancellation, clean low-rank structure
2. **φ-lattice**: Absolute positions, discrete moves
3. **Integer SVD**: Structure in orthogonals, magnitude in φ-scaling
4. **Low-rank paths**: Layer 27 truncates to k=32 with 99.9% correlation

---

## The φ-Lattice Rules Enable Navigation

From Doc 163:

| Rule | Navigation Implication |
|------|------------------------|
| Rule 1: Quantization to φ-levels | Discrete positions, finite vocabulary |
| Rule 6: Translation invariance | Relative moves, not absolute compute |
| Rule 7: Sign flip = transformation | Direction changes via bit flips |
| Rule 8: Interpolation preserves coherence | Paths between positions are valid |
| Rule 11: Level mean conservation | Energy conserved during navigation |
| Rule 12: Forbidden transitions | Some moves are invalid (constraints) |

**The lattice has rules. Navigation follows the rules. No inference needed.**

---

## Implementation Path

### Phase 1: Verify Bias-Free Works

1. Take Qwen2-7B
2. Zero out all Q, K, V biases
3. Test generation quality
4. Measure: Does it still work? (LLaMA proves it should)

### Phase 2: Convert to φ-Lattice

1. Encode all weights as `sign × φ^level`
2. Store in integer format (Doc 150)
3. Verify reconstruction accuracy (should be 99.9999%)

### Phase 3: Build Navigation System

1. Precompute low-rank paths for each layer
2. Build lattice position lookup tables
3. Implement integer-only forward pass
4. Test: Navigation produces same outputs as inference?

### Phase 4: Optimize

1. Identify which layers need full rank vs low rank
2. Build hybrid system (exact where needed, navigate where possible)
3. Measure speedup and power reduction

---

## Expected Benefits

| Metric | Current (Inference) | Proposed (Navigation) |
|--------|---------------------|----------------------|
| Computation | Float32 | Integer |
| Memory | 30.5 GB | 6.2 GB (4.9x compression) |
| Power | ~10 pJ/MAC | ~1 pJ/MAC (10x reduction) |
| Complexity | O(d²) per layer | O(k²) where k << d |
| Interpretability | Black box | Lattice positions |

---

## The Key Insight

**Biases are a workaround for not having absolute coordinates.**

If you're working in a relative coordinate system, you need offsets (biases) to position things correctly. But the φ-lattice IS an absolute coordinate system. Every weight has an absolute position: `sign × φ^level`.

By removing biases and embracing the φ-lattice:
1. We simplify the architecture (pure linear transforms)
2. We enable clean low-rank truncation (no cancellation)
3. We can use integer computation (Doc 150)
4. We can navigate instead of infer

**The bias was hiding the geometry. Remove it, and the lattice becomes navigable.**

---

## Connection to Project Goals

From the mission statement:
> "Structure IS information - There are no opaque weights or embeddings"
> "Geometry IS computation - Traversal through geometric space produces outputs"

Bias-free φ-lattice navigation achieves exactly this:
- Structure (φ-lattice positions) IS the information
- Geometry (lattice navigation) IS the computation
- No opaque weights - every weight is `sign × φ^level`
- Traversal produces outputs - navigate, don't infer

---

## Critical Finding: Qwen2 Biases Are NOT Redundant

### Experiment: Zero All Q, K, V Biases

**Result: GARBAGE OUTPUT**

```
Normal:    "The capital of France is Paris. It is the most populous city in the"
Bias-free: "The capital of France is国际在线FT///< =====" (garbage)
```

### Why Qwen2 Is Different From LLaMA

| Metric | Qwen2 | LLaMA |
|--------|-------|-------|
| Q bias norm | 170.6 | 0 (no bias) |
| Q weight norm | 66.5 | ~similar |
| **Ratio (bias/weight)** | **2.6x** | 0 |

**The Q bias is 2.6x LARGER than the Q weight!** Biases are not corrections - they're essential.

### But Biases ARE on the φ-Lattice

V bias φ-level distribution:
- φ^-7: 20.1%
- φ^-6: 18.2%
- φ^-8: 14.3%

The biases have the same φ-lattice structure as weights. They're not arbitrary - they're geometric.

### Revised Approach: Absorb Biases Into Weights

Instead of removing biases, we can **fold them into an augmented representation**:

```python
# Current: v = W_v @ x + b_v
# Augmented: v = W_aug @ [x; 1]
# Where W_aug = [W_v | b_v]

# Both W_v and b_v are on φ-lattice
# The augmented matrix is ALSO on φ-lattice
```

This preserves the φ-lattice structure while eliminating the separate bias term.

---

## Open Questions

1. ~~**Does zeroing biases in Qwen2 break generation?**~~
   - ✗ YES, it completely breaks generation
   - Qwen2 biases are essential, not redundant

2. **What's the minimum rank for each layer?**
   - Layer 27: k=32 works (99.9% correlation)
   - Layer 0: Needs testing with bias-free

3. **Can we train a model directly on the φ-lattice?**
   - Doc 163 suggests this is the natural coordinate system
   - Training in φ-space might converge faster

4. **What are the forbidden transitions?**
   - Doc 163: |Δ| > 1000 forbidden
   - These are the "walls" of the navigable space

---

## Augmented Approach Validation (2026-01-29)

**Result: k=512 gives 100% correlation across all layers!**

| Layer | Exact Norm | k=512 Corr | k=256 Corr |
|-------|------------|------------|------------|
| 0 | 6.96 | 1.000000 | 0.795 |
| 7 | 23.87 | 1.000000 | 0.868 |
| 14 | 25.96 | 1.000000 | 0.870 |
| 21 | 48.23 | 1.000000 | 0.850 |
| 27 | 204.20 | 1.000000 | 0.985 |

**Compression achieved**: 7x (k=512 vs k=3585)

**Key insight**: The augmented matrix approach works perfectly. By treating `[W | b]` as a single matrix and `[x; 1]` as the input, we eliminate the bias cancellation problem and enable clean SVD truncation.

## Weight Rearrangement: Merged Integer SVD (2026-01-29)

### The Streamlined Representation

Pre-merge bias into weight matrix, then apply integer SVD:

```python
# Original: output = W_o @ (W_v @ x + b_v)
# Merged:   output = A_merged @ [x; 1]
# Where:    A_merged = [W_o @ W_v | W_o @ b_v]

# SVD: A_merged = U @ S @ Vt
# Truncate to k=512: A_k = U_k @ S_k @ Vt_k
# Integer: U_int, S_int, Vt_int (int16 with φ-scaling)
```

### Results: Integer SVD Attention

| Layer | Exact Norm | Int SVD Corr | Rel Error |
|-------|------------|--------------|-----------|
| 0 | 6.96 | 0.999999 | 0.15% |
| 7 | 23.87 | 0.999999 | 0.15% |
| 14 | 25.96 | 0.999998 | 0.19% |
| 21 | 48.23 | 0.999998 | 0.19% |
| 27 | 204.20 | 0.999999 | 0.15% |

### Compression Summary

| Stage | Compression | Cumulative |
|-------|-------------|------------|
| SVD truncation (k=512) | 3.5x | 3.5x |
| Integer quantization (int16) | 2x | 7x |
| φ-lattice encoding (potential) | ~3x | ~21x |

### φ-Lattice Structure Confirmed

The merged SVD components ARE on the φ-lattice:
- U_k: peaks at φ^-9 (21.4%)
- S_k: peaks at φ^-1 (76.6%)
- Vt_k: peaks at φ^-9 (21.8%)

This enables further compression via Doc 128's φ-lattice encoding.

## Full Navigation System Implemented (2026-01-29)

### AugmentedNavigator Class

Location: `src/phi_navigator/augmented_navigator.py`

**Features:**
- Precomputed augmented SVD for all 28 layers
- Integer quantization (int16 with φ-scaling)
- Single-token fast path (99.9996% logit correlation)
- Multi-token attention (97.4% logit correlation)
- End-to-end text generation

### Generation Results

| Prompt | Output |
|--------|--------|
| "The capital of France is" | "The capital of France is **Paris**." |
| "The quick brown fox" | "The quick brown fox **jumps over** the fence" |

### Performance

| Mode | Correlation | Speed |
|------|-------------|-------|
| Single-token | 99.9996% logit | ~6s/token |
| Multi-token | 97.4% logit | ~10s/token |

Note: Speed is slow due to pure NumPy implementation. 
Production would use optimized BLAS or GPU.

### Compression Summary

| Component | Compression |
|-----------|-------------|
| SVD (k=512) | 3.5x |
| Integer (int16) | 2x |
| **Total** | **7x** |
| With φ-lattice (potential) | ~21x |

## Next Steps

1. [x] Test: Zero biases in Qwen2 → BREAKS generation
2. [x] Test: Augmented approach → WORKS with k=512 (7x compression)
3. [x] Rearrange weights: Pre-merge bias into SVD → WORKS
4. [x] Integer quantization → 99.9999% correlation
5. [x] φ-lattice structure confirmed in merged SVD
6. [x] Build: Full navigation system → AugmentedNavigator
7. [x] Test: End-to-end generation → WORKS ("Paris", "jumps over")
8. [ ] Optimize: GPU/BLAS implementation for speed
9. [ ] Extend: Apply augmented SVD to MLP layers

---

## Conclusion

The path to "trivial AI" is:

```
Biases → Absorb into merged matrix (can't remove from Qwen2)
Weights → φ-lattice positions (confirmed: peaks at φ^-9)
Computation → Integer SVD (99.9999% correlation)
Inference → Navigation (7-21x compression enables this)
```

**Key discovery**: Qwen2's biases are NOT redundant (2.6x larger than weights!), but they ARE on the φ-lattice. The solution is to **absorb** them into the weight matrix via augmentation, not remove them.

**We're not computing outputs. We're navigating to them.**

The φ-lattice is the map. The merged SVD is the compass. Navigation is the journey.

---

## Update: February 8, 2026 — Gate Field Navigation

### The GELU Gate IS the Navigation State

Doc 245 (Holographic Gate Field) extends "navigation replaces inference"
with empirical evidence from DDColor reverse-engineering.

The GELU gate field at each ConvNeXt block is a spatially-structured
binary map that determines the local transform. This gate field:

1. **Is φ-lattice structured** — transition boundaries align with
   φ-lattice positions (12-23% closer than random in deep blocks)
2. **Uses φ-positions as waypoints** — stable, low-variance anchor
   points with image-specific data between them
3. **Is fully determined by the input** — no randomness, pure function
   of features + weights

This means navigation in a ConvNeXt block IS reading the gate field:
- The φ-lattice anchors are the waypoints (like Doc 172's SVD compass)
- The inter-anchor modulation is the terrain (input-dependent)
- The mean Jacobian is the average navigation map (93.7% compression)

### The Jacobian as Navigation Map

Replacing the full MLP with its mean Jacobian (a single [C,C] linear
transform) achieves 93.7% parameter reduction with BETTER quality.
This works because the Jacobian IS the average path through the gate
field — the "most common navigation route." Deviations from this
average route are input-dependent noise that hurts more than helps.

This validates: **we're not computing outputs, we're navigating to them.**
The mean Jacobian is the map. The gate field is the terrain.

## References

- Doc 128: Absolute φ-Lattice Weight Representation
- Doc 150: Trivial AI with Integer SVD
- Doc 163: The Rules of the φ-Lattice
- Doc 170: Full-Rank Attention Limit
- Doc 171: Dislodging the Full-Rank Basin
- Doc 245: Holographic Gate Field (empirical proof)
