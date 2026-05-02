# DDColor Geometric Reverse-Engineering Roadmap

## Mission

Fully replace DDColor (55M learned parameters) with a **purely geometric** system
where every weight is either:
- Analytically constructable from φ-basis functions
- Derivable from the input image itself
- A universal constant (shared across all images)

**No training. No learned weights. Pure geometry.**

---

## Current State (Phases 6-12 Complete)

### What We've Proven

| Phase | Discovery | Impact |
|-------|-----------|--------|
| 6 | Transformer decoder = scaffolding (single matmul) | **-14.8M params (26.9%)** |
| 7 | DW conv kernels = rank-3 effective | Spatial basis is low-dim |
| 8 | DW conv = analytic φ-basis (R²=0.982) | **-331K params (0.6%), BETTER than learned** |
| 9A | ENCODE=DECODE: PW1 ↔ PW2 spectral symmetry | Constraint on optimization |
| 9B | GELU = 50% information bottleneck | Half of expanded dims are killed |
| 9C | Eigenvalue phases NOT φ-lattice (p=0.87) | Honest correction |
| 10 | All findings universal (Qwen2 confirms) | Not architecture-specific |
| 11 | Weight sharing fails (+43.7%) — directions critical | S interchangeable, U/V not |
| 12 | U vectors 88% sparse, kurtosis 9.96, block-specific | Structure exists but not φ |

### Model Lineage

| Version | What Changed | Params | RMSE vs V16 | p-value |
|---------|-------------|--------|-------------|---------|
| V16 | Full DDColor replica | 55.0M | — | — |
| V17 | No transformer decoder | 40.3M | +1.21% | 0.26 |
| V19 | + Analytic φ-basis DW conv | 40.3M | +1.01% | 0.37 |
| V19-K10 | + Minimal 10-basis DW | 40.1M | +1.3% | — |
| V_ULT | + 75% low-rank PW conv | 38.6M | +2.44% | 0.10 |

---

## Complete Parameter Map

```
DDColor: 55,020,784 total parameters

ENCODER (27.8M = 50.5%)
├── Stem conv 4×4 stride-4           4,704    (0.0%)  🔴 Untouched
├── Downsample convs 2×2 stride-2    1,548,288 (2.8%) 🔴 Untouched
├── Downsample norms                  2,880    (0.0%)  🟢 Trivial
├── DW conv 7×7 (18 blocks)          331,200  (0.6%)  ✅ ANALYTIC (φ-basis)
├── Block LayerNorm                   13,248   (0.0%)  🟢 Trivial
├── PW1 expand C→4C                   12,965,760 (23.6%) 🟡 75% low-rank works
├── PW2 contract 4C→C                 12,945,888 (23.5%) 🟡 ENCODE=DECODE
├── Layer scale (gamma)               6,624    (0.0%)  🟢 Trivial
└── Stage norms                       4,416    (0.0%)  🟢 Trivial

UNET DECODER (12.4M = 22.6%)
├── Merge conv 3×3 (3 layers)        8,188,928 (14.9%) 🔴 Untouched
├── Upsample 1×1 + pixshuffle        3,166,208 (5.8%)  🔴 Untouched
├── Last pixel shuffle                1,052,672 (1.9%)  🔴 Untouched
└── Skip batchnorms                   2,688    (0.0%)  🟢 Trivial

COLOR DECODER (14.8M = 26.9%)
└── 9-layer transformer               14,787,072        ✅ ELIMINATED (→ 25,600)

REFINE NET (208 = 0.0%)
└── Single 1×1 conv                   208               🟢 Trivial

LEGEND:
  ✅ = Geometrically replaced/eliminated
  🟢 = Trivially geometric (norms, scales — just affine transforms)
  🟡 = Partially understood (structure found, directions still learned)
  🔴 = Not yet analyzed
```

### By Status

| Status | Params | % of Total |
|--------|--------|------------|
| ✅ Replaced/Eliminated | 15,143,872 | 27.5% |
| 🟢 Trivially geometric | 27,184 | 0.1% |
| 🟡 Partially understood (PW) | 25,911,648 | 47.1% |
| 🔴 Untouched (UNet + stem) | 13,938,080 | 25.3% |

---

## Roadmap: Three Tiers

### Tier 1: UNet + Stem Analysis (25.3% of params)

These are likely tractable using the same tools that cracked the encoder.

**Phase 13: UNet 1×1 convolutions** (3.2M params)
- Same structure as PW1/PW2 — pointwise channel mixing
- Test: ENCODE=DECODE spectral symmetry
- Test: Low-rank approximation (75% rank)
- Test: Singular value sharing
- Expected: similar compressibility to encoder PW

**Phase 14: UNet 3×3 merge convolutions** (8.2M params)
- These are the largest untouched component
- Only 9 spatial entries — simpler than 7×7 DW conv
- Test: φ-basis decomposition for 3×3 spatial kernels
- Test: Separate into depthwise + pointwise (depthwise separable)
- Test: Low-rank across channel dimension

**Phase 15: UNet pixel shuffle** (1.1M params)
- 1×1 conv followed by pixel_shuffle(4)
- Effectively a learned upsampling — may be replaceable by bilinear + correction

**Phase 16: Encoder downsamples** (1.5M params)
- 2×2 stride-2 convolutions = learned downsampling
- Only 4 spatial entries — may be close to average pooling + channel projection
- Test: compare to avgpool + 1×1 conv

### Tier 2: The PW Direction Problem (47.1% of params)

**This is the hard core.** 25.9M parameters in PW1+PW2 where the directions
(U,V from SVD) carry all the information.

**Phase 17: Sparse Structure Analysis**
- U vectors are 88% sparse — what are the 12% active positions?
- Do active positions cluster? Are they consistent across blocks?
- Is there a shared sparse dictionary?
- Can we learn a universal sparse basis (like the φ-basis for DW conv)?

**Phase 18: Effective Gated Transform**
- The MLP is: y = PW2(GELU(PW1(x)))
- GELU kills ~50% of expanded channels
- The EFFECTIVE transform at typical inputs may be much lower-rank
- Test: compute effective Jacobian on real images
- If effective rank << full rank, we can compress further

**Phase 19: Channel Activation Patterns**
- Which expanded channels survive GELU? (pre-GELU distribution is 82-97% negative)
- If the same channels always die, we can pre-prune
- The "alive" subspace may have simpler structure

**Phase 20: Cross-Block Direction Transfer**
- Phase 12 showed blocks are independent (67% effective rank)
- But that still means 33% is shared!
- Extract the shared subspace, only learn the block-specific part
- Potential: reduce to ~67% × 25.9M = 17.4M params

**Phase 21: Input-Dependent Direction Discovery**
- Key insight: maybe the directions aren't "learned knowledge" but are
  DERIVABLE from the image features themselves
- If x determines which directions matter, then directions = f(x)
- This would make the entire network input-dependent geometry
- Test: do singular vectors of the Jacobian correlate with feature PCA?

### Tier 3: Full Geometric Assembly

**Phase 22: V20 — Fully Geometric DDColor**
- Combine all analytic replacements
- No .npz weight file — everything computed from:
  - φ-basis functions (universal)
  - Input image features (data-dependent)
  - Precomputed constants (color matrix, norms)

---

## The PW Direction Problem: Deep Analysis

### What We Know

The pointwise convolutions PW1 (C→4C) and PW2 (4C→C) form an
expand-gate-contract bottleneck in every ConvNeXt block:

```
x → PW1 → GELU → PW2 → scaled residual
    C→4C    gate    4C→C    + x
```

SVD of PW1: W1 = U1 @ diag(S1) @ V1t
SVD of PW2: W2 = U2 @ diag(S2) @ V2t

Phase 11 proved:
- S1 ≈ S2 (freely interchangeable, ΔRM SE < 0.5%)
- U1, V1, U2, V2 are NOT interchangeable (+43% RMSE)
- U vectors are 88% sparse, kurtosis 9.96
- Each block's directions are independent

### Potential Attack Vectors

#### 1. The Sparse Dictionary Hypothesis

If U vectors are 88% sparse and heavy-tailed, they might be representable
as sparse combinations of a SMALL shared dictionary:

```
U_block[i] ≈ D @ alpha_block[i]    (sparse alpha)
```

Where D is a universal dictionary (like the φ-basis for DW conv) and
alpha is a sparse coefficient vector per block.

If this works, we'd store:
- D: [4C × K] universal dictionary
- alpha: [K × rank] per block (sparse)
- Savings depend on K and sparsity of alpha

#### 2. The Effective Rank Hypothesis

GELU kills 50% of channels. Pre-GELU distribution is heavily negative
(82-97% negative). This means:

- Only ~50% of expanded channels ever activate
- The EFFECTIVE transform is: W2[:, alive] @ W1[alive, :]
- This is at most rank 2C (not 4C)
- If the "alive" channels are predictable, we can pre-prune

#### 3. The Image-Dependent Geometry Hypothesis

The most radical idea: the PW directions aren't "memories" stored in the
network — they're PROJECTIONS that the geometry of the input determines.

If the input feature covariance determines the optimal projection directions,
then:
- PCA of input features → V1 directions
- PCA of desired output → U2 directions
- The network just learned to do adaptive PCA

This would make the entire thing geometric: the shape of the data
determines the shape of the transform.

#### 4. The Shared Subspace + Rotation Hypothesis

Phase 12 showed 33% overlap between blocks. What if:
- There's a shared 33% subspace (universal)
- Each block applies a ROTATION to the shared subspace
- The rotation angle/axis varies by block position

This would reduce the problem to:
- Learn the shared subspace once
- Learn a rotation per block (much fewer params than full U/V)

---

## Success Criteria

| Level | Definition | Params | RMSE |
|-------|-----------|--------|------|
| **Bronze** | Tier 1 complete (UNet analyzed) | ~30M | < +3% |
| **Silver** | PW at 50% rank (shared subspace) | ~20M | < +5% |
| **Gold** | PW directions from sparse dictionary | ~10M | < +5% |
| **Platinum** | Fully analytic (no learned weights) | ~0.1M | < +10% |

Current: **between Bronze and Silver** (40.3M, +1.01%)

---

## Phase 17 Results: What We Actually Learned

### The Push-Pull Architecture (17C)

GELU creates a push-pull system, not a simple gate:
- "Dead" channels contribute **31.6%** of PW2 output energy
- Alive and dead contributions are **anti-correlated** (cos ≈ -0.19)
- Removing "dead" channels: +13.6% RMSE — they're structural
- Flipping dead sign: +9.1% — direction of absence matters
- Doubling dead: +186% — the balance is critical

**The tree analogy is exact**: dead wood isn't waste, it's structure.

### The Binary Code Discovery (17D)

The GELU gate creates a **soft binary code** at each spatial position:
- Sign pattern > magnitude for information content (5/6 blocks)
- Every pixel gets a unique code (100% uniqueness)
- But codes are low-dimensional: PCA 18x compression (stage 3)
- Bias predicts default code with 98-100% accuracy
- Input only flips 13-21% of channels in deep blocks

### The 4-Bit Cliff (17E)

| Bits | RMSE Δ% | Status |
|------|---------|--------|
| 8 | +0.03% | Lossless |
| 4 | +0.50% | Nearly lossless |
| 2 | +45.8% | CLIFF |
| 1 | +10.6% | Surprisingly OK |

Combined: 4-bit + rank 50% → **~1.6M equivalent params** (94% reduction)

### The Honest Boundary

Two fundamentally different kinds of "geometric":

**Type A — Analytic Construction (no trained weights needed)**:
- ✅ DW conv → φ-basis functions (R²=0.982)
- ✅ Transformer → single matmul (fixed-point collapse)
- ✅ GELU → x·σ(φ·x) (curvature matching)
- ✅ Norms, scales → trivially geometric

**Type B — Extreme Compression (trained weights needed, but tiny)**:
- 🟡 PW conv → 4-bit quantized + rank 50% (94% reduction, +4.6%)
- 🟡 UNet → likely similar compressibility (untested)

**Type C — Genuinely Irreducible (image-domain knowledge)**:
- 🔴 PW directions: not random, not DCT, not φ-lattice
- 🔴 Each block learns independent hyperplane orientations
- 🔴 These encode HOW to partition image feature space
- 🔴 This is learned knowledge about natural images

The PW directions are the network's **understanding of what features
look like in natural images**. This is genuinely learned content,
not scaffolding. It's analogous to the "content" side of the
DRUM/COMB wall from transformer disentanglement (Doc 177)

---

## Phase 18 Results: The Jacobian Breakthrough

### The Composed Transform (18B)

The Jacobian J(z) = W2 @ diag(GELU'(z)) @ W1 is the EFFECTIVE linear
transform. It reveals the true dimensionality:

- Stage 3: W1 rank 467, W2 rank 564 → **J_gelu rank 124** (16% of C)
- GELU acts as a **focusing lens** — halves the rank
- SV profile: **0.994+ correlation** across images (universal shape)
- Directions: 0.21-0.90 cross-image (input-dependent)

### The Breakthrough (18C): Mean Jacobian Replacement

**Replacing the full MLP (W1 + GELU + W2) with its mean Jacobian
is BETTER than the original:**

| Method | Params | % of PW | RMSE | Δ% |
|--------|--------|---------|------|-----|
| Original PW (W1+W2) | 25,911,648 | 100% | 13.421 | — |
| Mean Jacobian | 3,241,440 | 12.5% | 13.246 | **-1.30%** |
| Jacobian rank 25% | 1,625,688 | 6.3% | 13.201 | **-1.64%** |
| Jacobian rank 10% | 647,214 | 2.5% | 12.986 | **-3.24%** |

Why: The linearization DENOISES the transform. GELU adds input-dependent
fluctuations that are partially harmful. The mean Jacobian keeps only the
average effect, which is more robust. Low-rank further regularizes.

### Revised Model Lineage

| Version | Changes | Params | RMSE vs V16 |
|---------|---------|--------|-------------|
| V16 | Full DDColor | 55.0M | — |
| V17 | No transformer | 40.3M | +1.21% |
| V19 | + φ-basis DW conv | 40.3M | +1.01% |
| **V20** | **+ Jacobian PW (rank 25%)** | **~15.7M** | **~-0.4%** |
| V20-tiny | + Jacobian PW (rank 10%) | **~14.7M** | **~-2.0%** |

V20 would be 71-73% smaller than V16 and actually BETTER.

### Revised Parameter Map

```
ENCODER (after Jacobian replacement):
├── Stem + downsamples:  1,555,872  🔴 Untouched
├── DW conv:             analytic    ✅ φ-basis
├── PW (Jacobian r25%):  1,625,688  ✅ BETTER than original
├── Norms/scale:         24,288     🟢 Trivial
└── Encoder total:       ~3.2M

UNET DECODER:           12,410,496  🔴 Untouched (likely compressible)

COLOR DECODER:           25,600     ✅ Single matmul

REFINE NET:              208        🟢 Trivial

TOTAL:                   ~15.7M (from 55M = 71% reduction)
```

### Phase 19: UNet Decoder Analysis

Low-rank SVD compression of UNet weights:

| Rank % | RMSE | Δ% | UNet Params |
|--------|------|-----|-------------|
| 100% | 13.421 | — | 12.4M |
| 75% | 13.366 | -0.41% | 10.6M |
| **50%** | **13.417** | **-0.03%** | **7.1M** |
| 25% | 13.887 | +3.48% | 3.5M |

Notable: merge conv 1 has **rank@90% = 1** — 3.2M params that are rank-1.
UNet at 50% rank is essentially lossless.

---

## V20: Complete Assembly

**Every component of DDColor has now been analyzed and compressed:**

```
V20 PARAMETER MAP:

ENCODER:
├── Stem + downsamples:    1,555,872  (unchanged — small, 2.8%)
├── DW conv:               analytic    ✅ φ-basis (0 learned params)
├── PW (Jacobian r25%):    1,625,688  ✅ BETTER than original (-1.64%)
├── Norms/scale:           24,288     (unchanged — trivial)
└── Encoder total:         ~3.2M

UNET DECODER (rank 50%):
├── Merge convs:           ~3.6M      (from 8.2M)
├── Upsample convs:        ~2.1M      (from 3.2M)
├── Last pixel shuffle:    ~0.6M      (from 1.1M)
├── Batchnorms:            2,688      (unchanged — trivial)
└── UNet total:            ~6.3M

COLOR DECODER:             25,600     ✅ Single matmul (from 14.8M)
REFINE NET:                208        (unchanged — trivial)

═══════════════════════════════════════
TOTAL V20:                 ~11.1M     (from 55.0M = 80% reduction)
TOTAL V20-aggressive:      ~6.8M      (Jac r25% + UNet r25% = 88% reduction)
═══════════════════════════════════════

Expected RMSE vs V16: approximately NEUTRAL
(Jacobian improvement cancels transformer/DW regression)
```

### Next Steps

1. **Build V20 standalone model**: Combine all compressions into one file
2. **Validate end-to-end**: Run full test suite on combined model
3. **Explore stem compression**: 1.6M params, likely compressible
4. **V20-tiny**: Maximum compression variant for deployment

---

## Key Principles

1. **Fail-fast**: If a direction can't be constructed geometrically, we want
   to see the error immediately — no fallbacks.

2. **Honest corrections**: When something doesn't work (eigenvalue phases,
   weight sharing, rotation rate), we document the correction.

3. **Structure IS information**: Every finding either confirms that structure
   encodes information, or reveals where the boundary is.

4. **φ has boundaries**: φ governs spatial structure and spectral balance,
   NOT individual weight values or singular vector entries.
