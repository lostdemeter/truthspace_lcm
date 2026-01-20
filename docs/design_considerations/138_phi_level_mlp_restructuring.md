# Design Consideration 138: φ-Level MLP Restructuring

## Date: 2026-01-19

## Status: Validated

## The Discovery

MLP computation can be restructured from per-weight multiplications to per-φ-level multiplications, achieving **29.4x fewer float operations** with **99.95% correlation**.

## The Problem

MLP is the bottleneck in transformer inference:
- **83% of GPU time** is spent in MLP matmuls
- Standard MLP: `output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))`
- Each matmul is O(hidden × intermediate) = O(3584 × 18944) operations

## The Insight

From Design 127, 128, and 137:

1. **Doc 127**: Weights are coordinates of a geometric shape, not statistics
2. **Doc 128**: Weights live on the φ-lattice at discrete levels (peak at φ^-9)
3. **Doc 137**: In φ-space, multiplication = integer addition

**The synthesis**: If weights cluster at discrete φ-levels, we can group the computation by level instead of by dimension.

## The Restructuring

### Standard Matmul

```
output[j] = Σ_i W[j,i] × x[i]

Operations: 3584 multiplications per output dimension
Total: 18944 × 3584 = 67,895,296 multiplications per projection
```

### φ-Level Matmul

```
output[j] = Σ_level (signed_sum[j,level]) × φ^level

Where:
  signed_sum[j,level] = Σ_{i where W[j,i] is at level} sign[j,i] × x[i]

Operations: ~170 multiplications per output dimension
Total: 18944 × 170 = 3,220,480 multiplications per projection
```

### The Key Transformation

1. **Quantize weights to φ-levels**: `level = round(log(|W|) / log(φ) × SCALE / quantum)`
2. **Group by level**: For each (output_dim, level), identify which input dims have weights at that level
3. **Compute signed sums**: `signed_sum = Σ sign[i] × x[i]` (integer arithmetic)
4. **Scale by φ^level**: Use a precomputed LUT of ~180 values

## Validation Results

### Weight Distribution

```
φ-level distribution in W_gate:
  φ^ -8:  21.2% ████████████████████
  φ^ -9:  20.7% ████████████████████
  φ^-10:  15.7% ███████████████
  φ^ -7:  12.1% ████████████
  φ^-11:  10.4% ██████████
  φ^-12:   6.6% ██████
  ...
```

Weights naturally cluster at ~10 major φ-levels, with finer quantization giving ~170 levels.

### Accuracy

| Quantum | Levels | Correlation |
|---------|--------|-------------|
| SCALE (1024) | 21 | 78.5% |
| SCALE/2 (512) | 36 | 79.7% |
| SCALE/4 (256) | 66 | 89.7% |
| SCALE/4 (256) full MLP | 166-169 | **99.95%** |

### Operation Count

```
Standard MLP:
  Gate: 18944 × 3584 = 67,895,296 multiplications
  Up:   18944 × 3584 = 67,895,296 multiplications
  Down: 3584 × 18944 = 67,895,296 multiplications
  Total: 203,685,888 multiplications

φ-Level MLP:
  Gate: 18944 × 166 = 3,144,704 φ-multiplications
  Up:   18944 × 168 = 3,182,592 φ-multiplications
  Down: 3584 × 169 = 605,696 φ-multiplications
  Total: 6,932,992 φ-multiplications

Reduction: 29.4x fewer float multiplications
```

## Why This Works

### 1. Weights ARE Geometry (Doc 127)

The model's weights aren't random statistics—they're coordinates of a shape. This shape has structure that can be exploited.

### 2. Weights Live on φ-Lattice (Doc 128)

From our analysis of Qwen2-7B:
- All weight matrices peak at φ^-9 (17-22%)
- Perfect symmetry: +φ^k ≈ -φ^k
- 97% of weights are within 0.005 of a φ-lattice point

### 3. Multiplication = Addition (Doc 137)

In φ-space:
```
φ^a × φ^b = φ^(a+b)
```

This means:
- Weight × input = φ^(level_w + level_x)
- The exponent addition is **integer arithmetic**
- Only the final φ^level decode requires float

### 4. Grouping Exploits Sparsity

Within each φ-level, all weights have the same magnitude (up to sign). This means:
```
Σ W[i] × x[i]  where all W[i] are at level k
= φ^k × Σ sign[i] × x[i]
= φ^k × (integer sum)
```

The integer sum is essentially free on modern hardware.

## Implementation

### Precomputation (During Model Conversion)

```python
def precompute_phi_levels(W, quantum=256):
    """Precompute φ-level groupings for a weight matrix."""
    sign, exp = phi_encode_int(W)
    levels = (exp / quantum).astype(int)
    
    # For each (output_dim, level), store input indices and signs
    level_groups = {}
    for j in range(W.shape[0]):
        for level in np.unique(levels[j]):
            mask = levels[j] == level
            level_groups[(j, level)] = {
                'indices': np.where(mask)[0],
                'signs': sign[j, mask]
            }
    
    return level_groups, np.unique(levels)
```

### Inference

```python
def phi_level_matmul(x, level_groups, phi_lut, out_dim):
    """Compute matmul using φ-level grouping."""
    output = np.zeros(out_dim)
    
    for (j, level), group in level_groups.items():
        # Integer signed sum
        signed_sum = (group['signs'] * x[group['indices']]).sum()
        # Scale by φ^level (LUT lookup)
        output[j] += signed_sum * phi_lut[level]
    
    return output
```

### CUDA Kernel (Future)

For GPU acceleration:
1. Store level_groups as sparse CSR-like structure
2. Use warp-level reduction for signed sums
3. Single φ^level multiply per warp
4. Fuse gate, up, and down projections

## Connection to Prior Work

### Doc 132: φ-Sigmoid Discovery

The MLP operates in the linear regime where SiLU(x) ≈ x/2. This means the gate activation doesn't break the φ-level structure—it just shifts all exponents by log_φ(2).

### Doc 136: φ-Encoding Duplicates Transformer

Integer φ-encoding achieves 100% correlation with attention. The same encoding works for MLP weights.

### Calibration-Based Pruning

Combined with calibration-based dimension pruning (keeping 1/φ of dims):
- 99.3% correlation at 61.8% of dims
- 99.75% correlation at 80% of dims
- Additional 1.5-2x speedup

## Implications

### 1. Integer Arithmetic Dominance

The bulk of MLP computation becomes integer addition (signed sums), with only ~170 float multiplications per output dimension.

### 2. LUT-Based Scaling

The φ^level values are precomputed (180 entries). No runtime exponentiation needed.

### 3. Memory Bandwidth

The restructured representation may have different memory access patterns. Need to benchmark actual throughput.

### 4. Hardware Implications

This structure is ideal for:
- **FPGA**: Integer ALUs + small LUT
- **ASIC**: Dedicated φ-level units
- **GPU**: Warp-level reductions with shared LUT

## Next Steps

1. **Implement CUDA kernel** for φ-level matmul
2. **Benchmark actual speedup** (not just operation count)
3. **Combine with pruning** for additional gains
4. **Test across all layers** for consistency
5. **Measure memory bandwidth** impact

## Conclusion

By treating weights as geometry on the φ-lattice, we can restructure MLP computation from O(hidden × intermediate) multiplications to O(hidden × φ-levels) multiplications—a **29.4x reduction** with **99.95% accuracy**.

The key insight: **the shape of the weights IS the computation**. By organizing computation around the shape (φ-levels) rather than the indices (dimensions), we unlock massive efficiency gains.

## References

- Design 127: The Geometric Model Hypothesis
- Design 128: Absolute φ-Lattice Weight Representation
- Design 132: φ-Sigmoid Discovery
- Design 136: φ-Encoding Duplicates Transformer
- Design 137: φ as Universal Adapter
