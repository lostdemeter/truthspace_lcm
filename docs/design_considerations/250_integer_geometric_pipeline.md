# Doc 250: The Integer Geometric Pipeline — 23/23 Operations ACHIEVED

**Date:** February 17-18, 2026  
**Status:** COMPLETE — Finding 50  
**Prerequisites:** Doc 249 (Geometric Computing Vocabulary), Finding 49 (Purity Audit)

## Result

| Metric | Before (Finding 49) | After (Finding 50) |
|--------|---------------------|---------------------|
| Parameters on φ-lattice | 99.9956% | **100%** |
| Operations geometric | 19/23 (82.6%) | **23/23 (100%)** |
| Float32 in forward pass | accumulation + norms + SiLU | **none** |
| Next-token prediction match | — | **6/6 (100%)** |

All 4 non-geometric operations now have pure integer implementations:
1. **RMS norm** → exp double + accumulate + halve + subtract + scale
2. **SiLU** → 2D integer LUT: (sign, exp) → (sign, exp)
3. **Accumulation** → block-scaled fixed-point with 2^30 scale
4. **Biases/norms** → φ-encoded on the fly

## Original State (pre-implementation)

Finding 49 audited the Qwen2-7B φ-inference pipeline:

The 4 non-geometric operations were:
1. **RMS norm** (pre-attention) × 28 layers
2. **RMS norm** (pre-MLP) × 28 layers  
3. **RMS norm** (final) × 1
4. **SiLU activation** × 28 layers

Plus two cross-cutting concerns:
- **Float32 parameters**: 333,312 biases + norm weights not yet φ-encoded
- **Accumulation**: even pure matmul uses float32 for Σ products

---

## The Accumulation Problem

Every operation that sums multiple values faces the same core challenge:
addition has no simple form in exponent space. `φ^a + φ^b ≠ φ^(something simple)`.

Current pure matmul: sign XOR + exp ADD → LUT decode → **float32 sum**.

### Solution: Block-Scaled Fixed-Point Accumulation

For each output element, accumulate relative to the dominant term:

```
1. max_e = max_k(exp_a[k] + exp_b[k])           ← int32 max
2. shifted[k] = (exp_a[k] + exp_b[k]) - max_e   ← int32 sub (all ≤ 0)
3. val_int[k] = sign[k] × LUT_fixed[shifted[k]]  ← LUT: int16 → int32
4. sum_int = Σ_k val_int[k]                       ← int64 accumulation
5. result_sign = sign(sum_int)                     ← int comparison
6. result_exp = max_e + REVERSE_LUT[|sum_int|]     ← LUT: int64 → int16
```

**LUT_fixed**: shifted exponent → scaled integer. SCALE = 2^16.
- LUT_fixed[0] = 65536 (φ⁰ = 1.0)
- LUT_fixed[-128] = 40503 (φ^(-1))
- Terms shifted < -2560 (φ^(-20)) → negligible, truncate to 0

**Overflow**: 3584 terms × 65536 max = 235M → fits int32. Use int64 for safety.

**REVERSE_LUT**: |sum_int| → exponent offset. Pre-computed log table on high bits.

**Result**: Entire accumulation = integer ops + LUT. No IEEE float anywhere.

---

## Task 1: φ-Encode Remaining Float32 Parameters

**What**: Encode 333,312 values (biases + norm weights) as PhiEncoded.

**Why safe**: Finding 45 proved φ-quant bias → 6/6 with better margins.
Norm weights: φ-encode correlation = 0.999988.

**Steps**:
1. Re-encode biases in `.npz` files as (signs, exponents)
2. Update `phi_engine.py` loading to use PhiEncoded for biases/norms
3. Update `PhiAttention` and `rms_norm()` to accept PhiEncoded
4. Validate: 35-prompt test, expect identical results

**Effort**: Small.

---

## Task 2: φ-SiLU via Integer LUT

**What**: SiLU is a 1D scalar function → pre-compute for every (sign, exp) pair.

```python
# Build at init:
for sign in [-1, +1]:
    for exp in range(EXP_MIN, EXP_MAX+1):
        x = sign * PHI ** (exp / PHI_GRID)
        y = silu(x)
        SILU_LUT_SIGN[sign_idx, exp_idx] = np.sign(y)    # int8
        SILU_LUT_EXP[sign_idx, exp_idx] = phi_encode(y)   # int16

# At inference: pure integer lookup
out_sign = SILU_LUT_SIGN[x_sign_idx, x_exp - EXP_MIN]
out_exp  = SILU_LUT_EXP[x_sign_idx, x_exp - EXP_MIN]
```

**LUT size**: 2 × 30,001 × 3 bytes = ~180 KB (fits L2 cache).

**Gate × Up multiply** after SiLU: sign XOR + exp ADD. Pure integer.

**Effort**: Small-Medium.

---

## Task 3: φ-RMS Norm via Integer Ops

**What**: `y = x / sqrt(mean(x²)) * weight` decomposed:

| Step | Operation | φ-integer form |
|------|-----------|---------------|
| x² | square | sign=+1, exp=2×exp_x |
| Σx² | sum | Block-scaled accumulation (Task 0) |
| /D | mean | Integer divide |
| sqrt | square root | Halve the exponent |
| x/rms | normalize | Subtract exponents |
| ×weight | scale | XOR signs + add exponents |

Every step is integer except the accumulation, which uses the
block-scaled fixed-point solution from above.

**Key**: RMS is a single scalar per position. One accumulation of D=3584
terms, then D subtractions. Very efficient.

**Effort**: Medium (depends on accumulation infrastructure).

---

## Task 4: End-to-End Pure Integer Pipeline

Wire everything together. Activations flow as (int8 sign, int16 exp) pairs.

### Additional sub-problems:

**Residual Add** (`h + layer_output`): Two-term block-scaled sum per element.
Find max of two exponents, scale smaller, add as int, re-encode.

**RoPE** (`x*cos + x_rot*sin`): φ-encode cos/sin tables (constants).
Two multiplies (sign XOR + exp ADD) + two-term residual add per element.

**φ-Softmax**: `φ^(x/ln(φ))` where x is φ-encoded. The exponentiation is
a 1D function → same LUT pattern as SiLU. Normalization uses accumulation.

**Embedding**: Store as (signs, exps) table. Lookup returns φ-encoded vectors.

**Effort**: Large (integration of all tasks).

---

## Implementation Order

```
Phase A: Foundation
  A1. Build block-scaled fixed-point accumulator + REVERSE_LUT
  A2. φ-encode biases + norm weights (Task 1)
  A3. Validate: hybrid mode still 35/35

Phase B: Activations  
  B1. SiLU LUT (Task 2) — easiest, 1D function
  B2. φ-RMS norm (Task 3) — uses accumulator from A1
  B3. Validate each independently against float baseline

Phase C: Integration
  C1. φ-encoded activation format throughout pipeline
  C2. Residual add (two-term block-scaled)
  C3. RoPE in φ-integer
  C4. Softmax LUT + normalization
  C5. Embedding as φ-encoded lookup

Phase D: Validation
  D1. End-to-end pure integer forward pass
  D2. 35-prompt validation — target: 35/35
  D3. Measure correlation vs float baseline
  D4. Profile: integer ops only, no IEEE float multiply/divide
```

---

## What "Integer Math" Means

After this roadmap, the entire forward pass uses only:

| Operation | Type | Where |
|-----------|------|-------|
| Sign XOR | int8 × int8 | Multiply signs |
| Exponent ADD | int16 + int16 → int32 | Multiply magnitudes |
| LUT lookup | int → int | Decode/encode, SiLU, softmax exp |
| int64 ADD | int64 + int64 | Accumulation (sums) |
| int32 MAX | max(int32, int32) | Block scaling |
| int32 SUB | int32 - int32 | Shift exponents |
| int64 DIV | int64 / int32 | Mean (RMS norm) |
| REVERSE LUT | int64 → int16 | Re-encode accumulated sum |
| Array index | int → (int8, int16) | Embedding, SiLU LUT |

**Zero IEEE float operations.** No float multiply, no float divide,
no float sqrt, no float exp. Everything is integer arithmetic + pre-computed
lookup tables.

The LUTs are computed once at initialization from the φ-lattice definition.
They ARE the geometry — pre-computed traversals of the φ-lattice structure.

---

## Precision Budget

| Component | Precision source | Bits |
|-----------|-----------------|------|
| Weights | φ-grid K=128 | ~17 bits equivalent |
| Activations | same φ-grid | ~17 bits |
| Accumulation | block-scaled 2^16 | 16 bits relative |
| SiLU LUT | exact to grid | ~17 bits |
| Total pipeline | limited by grid | ~16-17 bits effective |

float32 has ~24 bits of mantissa. We lose ~7 bits. For next-token prediction
(argmax of logits), this is more than sufficient — the margin between top-1
and top-2 is typically 0.1-5.0 in logit space.

---

## Files

- **This roadmap**: `docs/design_considerations/250_integer_geometric_pipeline.md`
- **Purity audit**: `experiments/model_reverse_engineering_v2/phase5_geometric_purity_audit.py`
- **Current pure matmul**: `phi_geometric/inference/phi_matmul.py` (phi_matmul_pure)
- **Current SiLU**: `phi_geometric/inference/phi_components.py` (phi_silu)
- **Current RMS norm**: `phi_geometric/inference/phi_components.py` (rms_norm)
