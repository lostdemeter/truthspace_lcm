# Doc 234: The 208-Number Decomposition

**Date:** February 7, 2026  
**Status:** Proven  
**Prerequisites:** Doc 233 (Rank-2 Color Plane Discovery)

## The Result

DDColor's entire 55-million-parameter colorization model reduces to:

```
ab(pixel) = Σ_q  activation(q, pixel) × [w_a(q), w_b(q)]  +  W_input × grayscale(pixel)  +  bias
```

Where:
- `activation(q, pixel)` = 100 query activation maps (the ONLY thing 55M params compute)
- `[w_a(q), w_b(q)]` = 200 fixed numbers (the universal color wheel)
- `W_input` = 6 numbers (input channel weights, 2×3)
- `bias` = 2 numbers (output bias)

**Total fixed parameters: 208.**  
**Reconstruction error: 0.000.**

This is not an approximation. It is an exact decomposition.

## Proof

Three oracle versions tested on 8 held-out images:

| Version | Mean Error | What It Does |
|---------|-----------|-------------|
| v1: per-region averaging | 5.48 | Average query activations within edge regions |
| **v2: per-pixel queries** | **0.002** | Apply color wheel per-pixel from query maps |
| **v3: per-pixel + input** | **0.000** | Add input channel contribution |

v1's error came from activation cancellation at territory boundaries — when a region straddles two query territories, averaging their activations before weighting cancels them. Per-pixel application (v2) eliminated this entirely.

The remaining 0.002 error (v2 → v3) is the input channel contribution: 6 weights mapping the grayscale input to ab. This is negligible — the input channels contribute almost nothing to the final color. The color is entirely determined by the 100 query activation maps.

## What This Means

### The 55M Parameters Have One Job

The encoder (27.8M), UNet decoder (~2M), and transformer color decoder (~25M) exist solely to compute 100 activation maps. Each map is a spatial heatmap: "how strongly does query q claim pixel (h,w)?"

Everything else — the color assignment, the ab output, the final image — follows deterministically from 208 fixed numbers applied to these maps.

### The Color Wheel Is Universal

The 200 numbers (100 queries × 2 ab channels) are:
- Fixed across all images
- Nearly uniformly distributed across 360° of color space
- Magnitudes range from 0.008 to 0.330
- Aligned within 2° of the native Lab color axes

These numbers tile color space like compass directions. Each query "owns" a direction. The model doesn't learn colors per image — it activates pre-existing color directions.

### Only 33-36 Queries Are Active Per Image

Of the 100 available queries, only 33-36 are dominant for any given image. The rest contribute weakly. This means each image uses roughly one-third of the color wheel — a selective activation of the universal palette.

### The Territory Assignment IS the Intelligence

The hard part — the part that requires 55M parameters — is deciding which query claims which pixel. This is a geometric operation in 256-dimensional feature space:

1. The encoder maps each pixel to a 256-dim feature vector
2. The transformer shapes each query into a 256-dim color vector
3. The dot product `color_vector · pixel_feature` determines activation
4. Winner-take-most dynamics assign territories

The "knowledge" of what color things should be is encoded in the SHAPE of the feature space — pixels with similar features end up near each other, and queries position themselves to claim coherent clusters.

## The Navigation Insight

Ground truth colorization is a point in the same space as DDColor's output. Both are configurations of the 100 activation maps. The difference between DDColor's colorization and ground truth is a specific perturbation of these maps.

This means:
1. We can **measure** the exact difference between DDColor and ground truth in activation-map space
2. We can **interpolate** between any two colorizations by blending activation maps
3. We can **steer** the model by modifying activations at specific layers
4. Ground truth is **reachable** — it's just another point we can navigate to

### Points in Color Space

| Point | Description | How to Reach |
|-------|-------------|-------------|
| DDColor output | Model's best guess | Forward pass |
| Ground truth | Actual colors | Oracle with GT activation maps |
| Lattice output | Geometric estimation | Synthetic knowledge lattice |
| Any interpolation | Blend between any two | α × maps_A + (1-α) × maps_B |

Every colorization is a set of 100 activation maps. The 208 fixed numbers convert any set of maps to ab colors. Navigation between colorizations is navigation between activation map configurations.

### What We Can Learn By Navigating

The path from DDColor's output to ground truth reveals:
- **Which queries need to change** — tells us which color assignments DDColor gets wrong
- **How activations shift** — tells us what geometric operations correct errors
- **Where corrections concentrate** — tells us which spatial regions are hardest
- **Whether corrections are low-rank** — tells us if simple steering suffices

If the DDColor→GT correction is low-rank (as we expect from the correction SVD finding of 5 modes for 90% variance), then navigation is cheap: a few parameters steer the entire output.

## The Decomposition Stack

```
IMAGE (H × W × 3)
  ↓
FEATURES (H × W × 256)         ← 27.8M params (encoder)
  ↓
ACTIVATION MAPS (100 × H × W)  ← 25M params (transformer)
  ↓
AB OUTPUT (H × W × 2)          ← 208 fixed numbers
```

Each layer compresses:
- Image → Features: 3 channels → 256 channels (expansion, then compression via pooling)
- Features → Activations: 256 channels → 100 maps (selection via dot product)
- Activations → AB: 100 maps → 2 channels (projection via color wheel)

The final step is a linear projection from 100 dimensions to 2. This is the rank-2 structure we discovered in Doc 233. The color wheel IS this projection matrix.

## Connection to the Hypothesis

> "Structure IS information — There are no opaque weights or embeddings"

The 208 numbers are fully transparent:
- Each number has a clear meaning (query q's contribution to channel a or b)
- The structure is geometric (angles on a color wheel)
- The mechanism is a dot product (geometric proximity)

The 55M "opaque" parameters compute activation maps — but these maps are interpretable as territory assignments. Each query has a spatial territory, and the assignment is determined by feature similarity (another geometric operation).

> "Geometry IS computation — Traversal through geometric space produces outputs"

The entire colorization is traversal:
- Encode: project image into feature space
- Select: queries traverse feature space to find their territory
- Assign: project territories onto the color wheel

> "The shape IS the knowledge"

The encoder's shape (its weight geometry) determines how pixels cluster in feature space. The transformer's shape determines how queries partition this space. The color wheel's shape determines what colors emerge. Change any shape, change the output.

## Experimental Files

| File | Purpose |
|------|---------|
| `territory_mapper.py` | Region→query mapping, oracle v1 (per-region) |
| `oracle_fix.py` | Oracle v2 (per-pixel) and v3 (exact), proving 0.000 error |
| `rank2_deep_dive.py` | Discovery of the rank-2 structure and 200 numbers |
| `reverse_engineer_macro.py` | Identification of ENCODE→SELECT→SORT→ASSIGN |
| `colorizer_comparisons/oracle_fixed/` | Visual proof of exact reconstruction |

## Next Steps

1. **Navigate to ground truth:** Compute activation maps that produce GT colors and study the DDColor→GT path
2. **Steer at intermediate layers:** Can we modify transformer attention to redirect query territories?
3. **Geometric control:** Can we achieve desired colorizations by geometrically constructing activation maps?
4. **Learn from the journey:** What does the shape of the DDColor→GT path tell us about how information is stored?
