# Doc 233: The Rank-2 Color Plane Discovery

**Date:** February 7, 2026  
**Status:** Active Discovery  
**Prerequisites:** Doc 230 (Computational Primitives), Doc 231 (Gap Analysis), Doc 232 (Synthetic Knowledge)

## The Discovery

DDColor's entire color decoder — 100 learned queries, 9 transformer layers, 256-dimensional embeddings — produces output that lives on a **2-dimensional plane**. 99.3% of variance is captured by rank 2. This plane is **universal**: alignment across 8 different images is 0.965 (nearly identical).

The 2D plane is aligned within 2° of the native Lab color axes (a, b). The 256-dim computation reduces to 2 numbers per pixel.

## The Complete Decomposition

```
ab(pixel) = Σ_q  dot(color_vector_q, img_feature(pixel)) × [w_a(q), w_b(q)]
```

Where:
- `color_vector_q` = 256-dim query vector (image-dependent, from transformer)
- `img_feature(pixel)` = 256-dim per-pixel feature (from encoder + UNet)
- `[w_a(q), w_b(q)]` = 2D color direction (FIXED, from refine net weights)

### The Three Components

| Component | Parameters | What It Computes | Nature |
|-----------|-----------|-----------------|--------|
| Encoder (ConvNeXt) | 27.8M | WHERE in feature space each pixel lives | Learned geometry |
| Color Decoder (Transformer) | ~25M | WHICH query claims which territory | Negotiated assignment |
| Refine Net | 200 + 6 | HOW queries map to ab color | Fixed color wheel |

### The 200 Numbers

The refine net weights `[2, 100]` assign each of the 100 queries a position on the 2D color wheel:
- Each query has `(w_a, w_b)` = magnitude and angle in ab-space
- Magnitudes range from 0.008 to 0.330
- Angles are nearly uniformly distributed across the full 360° color wheel
- The queries tile color space like compass directions

Input grayscale channels contribute error of only 0.01 to the final output — they are negligible. The colorization is **entirely** determined by query activations × the 200 numbers.

## Key Measurements

### Rank-2 Variance (per image)
| Image | Rank-1 | Rank-2 | Rank-3 |
|-------|--------|--------|--------|
| 5193 | 56.1% | **99.6%** | 99.6% |
| 5503 | 60.3% | **99.3%** | 99.3% |
| 5586 | 58.8% | **99.3%** | 99.3% |
| 5992 | 58.4% | **99.3%** | 99.3% |
| 6040 | 55.3% | **99.5%** | 99.5% |
| 6460 | 66.1% | **99.1%** | 99.1% |
| 6614 | 60.8% | **98.6%** | 98.6% |
| 6763 | 61.5% | **99.4%** | 99.4% |

### Plane Alignment Across Images
Mean alignment: **0.965** (1.0 = identical planes)  
Minimum alignment: 0.922  
The plane is a universal structure, not image-dependent.

### Query Activity
- Only **33-36 unique dominant queries** per image (out of 100)
- 82-84 queries needed for 90% total activation energy
- Queries that dominate vary by image (territory is content-dependent)

### Attention Sparsification (The Commitment Moment)
| Transformer Layer | Active Queries | Attention Entropy | State |
|-------------------|---------------|-------------------|-------|
| 0 | 10/100 | 5.4/5.5 | Exploring |
| 3 | 89/100 | 5.1/5.5 | Still exploring |
| **6** | **100/100** | **1.8/5.5** | **Committed (sparse)** |
| 8 | 82/100 | 5.6/8.3 | Refining |

Layer 6 is the phase transition where queries lock onto spatial territories.

## The Four Macro Operations

DDColor implements: **ENCODE → SELECT → SORT → ASSIGN**

### 1. ENCODE (ConvNeXt, 27.8M params)
Image → per-pixel feature vectors in 768-dim space.  
Each ConvNeXt block = depthwise conv → layernorm → pointwise → GELU → residual.  
This is: rotation + projection + nonlinear scaling.  
Feature clusters form naturally — "grass-like" features end up near each other.

### 2. SELECT (Cross-Attention, layers 0-8)
100 queries attend to image features via dot product.  
Early layers: diffuse attention (exploring).  
Layer 6: sparse attention (committing to territories).  
This is geometric selection — proximity in feature space determines assignment.

### 3. SORT (Self-Attention, layers 0-8)
Queries attend to each other. This prevents overlapping territories.  
Queries negotiate: "I'll take this region, you take that one."  
This is a competitive exclusion process in geometric space.

### 4. ASSIGN (einsum + refine net)
`dot(color_vector_q, pixel_feature)` → activation strength per query per pixel.  
Activation × `[w_a, w_b]` → ab contribution.  
Sum over queries → final ab color.  
This is a weighted projection onto the universal color plane.

## Connection to Prior Work

### The Scaffolding/Content Wall (Doc 177)
The 200 refine weights are **scaffolding** — fixed, universal, derivable.  
The territory assignment is **content** — image-dependent, requires recognition.  
But "recognition" here means "feature proximity," not semantic labeling.

### The Bulge Discovery (Doc 180)
Trajectories = geodesic + bulge, where bulge has 10 basis functions.  
Color queries = universal plane + per-image coordinates, where coordinates have ~35 active dimensions.  
Both show: structure is low-rank. The "content" is a small perturbation on a fixed frame.

### Correction SVD (this session)
The lattice→DDColor correction is rank 5 for 90% variance.  
Edge-bounded regions explain 93% of correction.  
KS v2 damping closes 11-32% of the gap.  
These are all manifestations of the same low-rank structure.

## Implications

### 1. The Color Wheel Is Free
The 200 numbers — each query's position on the color wheel — are fixed weights we already have. They tile color space uniformly. No learning needed for the color assignments themselves.

### 2. The Expensive Part Is Territory Assignment
55M parameters exist to compute which query claims which pixel. This requires:
- Feature extraction that clusters similar textures/objects together
- Competitive assignment that gives each region a single query
- Multi-scale processing (the transformer cycles through 3 scales)

### 3. Our Edge Regions Are Geometric Queries
The KS v2 edge-bounded regions (67 per image) are a purely geometric version of DDColor's 100 query territories (33-36 active per image). The numbers are close. The mechanism is analogous. What's missing is the feature quality.

### 4. The Path Forward
Map our edge-bounded regions to the universal color wheel. Each region needs:
- A feature vector (from brightness, texture, position, edges)
- A color wheel position (the closest of the 200 fixed directions)
- An activation strength (how confidently to apply that color)

If this mapping works, we replace 55M parameters with:
- Edge segmentation (~0 params, geometric)
- Feature extraction (~0 params, geometric)
- Region → color wheel mapping (~few hundred params)

## Experimental Files

| File | Purpose |
|------|---------|
| `lattice_navigator/reverse_engineer_macro.py` | Identified the 4 macro operations |
| `lattice_navigator/rank2_deep_dive.py` | Analyzed the rank-2 plane structure |
| `lattice_navigator/correction_analysis.py` | SVD of lattice→DDColor correction (5 modes) |
| `lattice_navigator/ks_v2_damping.py` | Edge-bounded region damping (11-32% gap) |
| `colorizer_comparisons/rank2_analysis/` | Territory maps and query visualizations |

## Open Questions

1. **Can we predict territory assignment from geometric features alone?** Our edge regions approximate DDColor's territories. Can brightness/texture/position features select the right color wheel direction?

2. **What determines the sparsification at layer 6?** Is this a phase transition with geometric structure (phi patterns)? Or is it an emergent property of the competitive dynamics?

3. **Is the universal color plane derivable from first principles?** The plane aligns with Lab axes. The query angles tile uniformly. Is this the optimal structure, or one of many possible structures?

4. **Can the Karplus-Strong damping operate on the color wheel directly?** Instead of damping ab values, damp the color wheel assignments per region. This would combine the territory mechanism with the resonance approach.
