# Doc 239: The Two Frontiers — Where Geometry Ends and Learning Begins

**Date:** February 7, 2026  
**Status:** Definitive Finding  
**Prerequisites:** Doc 238 (Reverse Navigation Encoder)

## The Two Questions

Having proven the rank-99% geometric encoder works (49% rank compression, beats DDColor 14/19), two frontier questions remained:

1. **Are the pointwise convolutions (channel mixing) also φ-structured?**
2. **Can we derive the spatial basis filters from first principles?**

Both questions now have clear answers.

---

## Q1: Pointwise Convolutions Are NOT φ-Structured

### The Data

Each ConvNeXt block has two pointwise convolutions:
- `pwconv1`: dim → 4×dim (expand channels)
- `pwconv2`: 4×dim → dim (compress channels)

| Property | pwconv1 | pwconv2 | DW conv (comparison) |
|----------|---------|---------|---------------------|
| **Mean S[0]/S[1]** | **1.237** | **1.160** | **1.619–1.807** |
| **φ error** | **23.6%** | **28.3%** | **0.06%–11.7%** |
| **Mean rank90** | **240** | **249** | **3–7** |
| Low-rank? | No | No | **Yes** |
| φ-structured? | No | No | **Yes** |

### Interpretation

The contrast is stark:
- **Depthwise (spatial) convolutions**: rank 3-7, φ-structured → **geometric**
- **Pointwise (channel) convolutions**: rank 240-249, NOT φ-structured → **learned content**

The pointwise convolutions are the **irreducible learned component** of the encoder. They perform channel mixing — combining geometric features (edges, blobs, gradients) into semantic features (sky-detector, grass-detector, skin-detector). This mixing requires knowledge of the visual world that cannot be derived from first-principles geometry.

### Connection to Scaffolding/Content (Doc 177)

This cleanly resolves the scaffolding/content boundary:

| Component | Scaffolding or Content? | Evidence |
|-----------|------------------------|---------|
| Spatial filters (DW conv) | **Scaffolding** | Low-rank, φ-structured, canonical matches |
| Importance weighting (singular values) | **Scaffolding** | φ-ratios in top modes |
| Channel mixing (PW conv) | **Content** | Full-rank, NOT φ-structured |
| Layer scale (γ) | **Mixed** | Moderate φ alignment (22-34% error) |
| Downsample convolutions | **Content** | High-rank, NOT φ-structured |

The encoder's computation decomposes into:
1. **Geometric scaffolding** (spatial filtering): What spatial patterns to look for → derivable
2. **Learned content** (channel mixing): What those patterns mean semantically → requires training data

---

## Q2: First-Principles Basis — 60% of Performance From Geometry Alone

### The Canonical Basis

We constructed 31 handcrafted 7×7 filters from classical computer vision:
- **Gaussian blobs** (σ = 1.0, 2.0, 3.0)
- **Difference of Gaussians** (DoG — blob detectors)
- **Laplacian of Gaussian** (LoG — edge/blob hybrid)
- **Oriented edge detectors** (0°, 45°, 90°, 135°)
- **Oriented second derivatives** (4 orientations)
- **Gabor filters** (4 orientations × 2 frequencies)
- **Center-surround** (2 scales)
- **Quadrant filters** (4 quadrants)
- **DC** (mean)

No learning. No training data. Pure geometry.

### How Well Do They Match?

**Kernel variance captured by canonical reconstruction:**

| Stage | Channels | Variance Captured |
|-------|----------|------------------|
| Stage 0 | 96 | 86.2% |
| Stage 2 | 384 | 86.9% |
| Stage 3 | 768 | **92.8%** |

**Top SVD modes vs canonical filters (Stage 3 Block 0):**

| Mode | Variance | Best Match | Correlation |
|------|----------|-----------|-------------|
| 0 | 49.9% | Gaussian σ=1 | r=0.811 ★★ |
| 1 | 19.0% | DC (mean) | r=0.698 ★ |
| 2 | 14.6% | Edge 90° | r=0.748 ★★ |
| 3 | 5.2% | Center-surround r=2 | r=0.699 ★ |
| 4 | 3.7% | Second derivative 90° | r=0.487 |
| 5+ | <2% each | Various | <0.5 |

The dominant modes (88.5% of variance) match canonical geometric filters. The critical band modes (5+) don't have clean canonical equivalents.

### End-to-End Performance

| Model | Mean Error | Gap Closed |
|-------|-----------|-----------|
| Zero (gray) | 16.31 | 0% |
| **Canonical basis (31 filters)** | **14.33** | **12.1%** |
| Full encoder 2D | 13.02 | 20.1% |

**31 handcrafted geometric filters, with zero training, achieve 60% of the full encoder's performance.** This is the purely geometric contribution to colorization.

### What the Remaining 40% Requires

The gap between canonical (12.1%) and full (20.1%) comes from:
1. The **critical band** (modes 8-23): spatial patterns that don't match textbook filters
2. The **pointwise mixing**: combining features semantically (requires learned weights)
3. The **residual connections**: accumulated refinement over 18 blocks

The critical band filters are not random — they have structure — but they're not any of the 31 canonical types we tested. They may represent image-statistics-specific patterns (natural image priors learned from millions of photos) that are geometric in nature but not in our current vocabulary.

---

## The Complete Picture

### What the Encoder Actually Does

```
GRAYSCALE INPUT
     ↓
STEM: 13 geometric bases (edges, blobs, DC)           ← SCAFFOLDING
     ↓  [φ-structured: S[0]/S[1] = 1.56]
×18 BLOCKS:
  ├─ Depthwise 7×7: ~23 basis filters per block       ← SCAFFOLDING (rank-99%)
  │    Top 7: canonical (Gauss, edges, DoG)              φ-structured
  │    Modes 8-23: critical band (learned geometry?)     NOT φ-structured  
  │    Modes 24+: noise (safely removed)
  ├─ LayerNorm                                         ← SCAFFOLDING
  ├─ Pointwise expand (dim → 4×dim)                    ← CONTENT (full rank)
  ├─ GELU                                              ← SCAFFOLDING
  ├─ Pointwise compress (4×dim → dim)                  ← CONTENT (full rank)
  └─ Layer scale + residual                            ← MIXED
     ↓
UNET DECODER → 256 features
     ↓
LINEAR PROJECTION (2D) → ab color
```

### The Boundary

| Layer Type | Total in Encoder | Geometric? | Compressible? |
|-----------|-----------------|-----------|--------------|
| Spatial filters (DW) | 18 blocks | ✓ φ-structured, low-rank | ✓ 49% rank → 99% quality |
| Channel mixing (PW) | 36 layers | ✗ NOT φ, full-rank | ✗ This IS the learned content |
| Norms, scales, GELU | Throughout | ✓ Deterministic operations | N/A (no learned params) |
| Downsample | 3 layers | ✗ NOT φ, high-rank | ✗ Learned transitions |

### Parameter Budget

| Component | Original Params | After Geometric Compression |
|-----------|---------------|---------------------------|
| DW spatial convs | 325K | 160K (49% rank budget) |
| Pointwise convs | ~50M | ~50M (irreducible) |
| Norms, scales | ~200K | ~200K (unchanged) |
| Downsample | ~1.5M | ~1.5M (unchanged) |
| **Total encoder** | **~52M** | **~52M (0.3% saved)** |

The spatial convolutions — which ARE geometric — are only 0.6% of the encoder's parameters. The 99.4% that matters (pointwise mixing) is the irreducible learned content.

---

## What This Means for the Hypothesis

### The Hypothesis: "Structure IS Information"

**Partially confirmed, with a precise boundary.**

- ✓ The **spatial operations** are geometric (edges, blobs, Gabor, DoG) with φ-structure
- ✓ The **importance hierarchy** follows φ-ratios (S[0]/S[1] → φ at every stage)  
- ✓ 60% of colorization performance comes from first-principles geometry alone
- ✗ The **channel mixing** is irreducibly learned — it carries world knowledge
- ✗ The spatial operations are only 0.6% of total parameters

### The Reframing

The encoder is a **geometric scaffolding filled with learned content**:

> The SHAPE of the computation is geometric (low-rank spatial filters, φ-structured importance, hierarchical multi-scale processing). But the SUBSTANCE of the computation — what each channel means semantically — is learned from data.

This is like a building: the architecture (beams, columns, load-bearing structure) follows physical law and geometry. But the contents of each room (furniture, purpose, meaning) are chosen by the inhabitants. You can't derive the building's contents from its structure, but the structure constrains what contents are possible.

### The Unexpected Finding

The deepest insight: **φ structures the importance hierarchy, not the content.** The top singular values (which patterns matter most) follow φ-ratios. The actual patterns themselves are canonical geometry. But which patterns to combine and how — the semantic content — is learned and does not follow φ.

This suggests φ is a **structural organizer**, not a **content generator**. It determines how information is weighted and arranged, not what the information means.

---

## Experimental Files

| File | Purpose |
|------|---------|
| `pointwise_and_basis_analysis.py` | Both frontier analyses in one script |
| `geometric_encoder_rank99.py` | Rank-99% encoder (beats DDColor) |
| `rank_sweep.py` | Phase transition discovery |
| `reverse_kernel_analysis.py` | Kernel SVD and φ-analysis |

## Summary

The two frontiers are resolved:

1. **Pointwise convolutions are NOT geometric** — full-rank, not φ-structured, irreducible learned content (99.4% of encoder params)
2. **31 canonical filters achieve 60% of full performance** — the geometric foundation IS real, but it's the scaffolding, not the substance

The encoder is a hierarchy of geometric primitives (scaffolding) combined through learned channel mixing (content). φ structures the importance of the spatial primitives but does not appear in the semantic mixing. The hypothesis holds for structure; it does not hold for content. The boundary between them is precisely measurable.
