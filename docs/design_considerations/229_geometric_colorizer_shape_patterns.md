# Doc 229: Geometric Colorizer Shape Patterns

## Purpose

Catalog how geometric design choices affect colorization behavior.
Each pattern is a reusable building block for designing geometric AI systems.

---

## The Problem

Given a grayscale image, produce plausible color. Three known solutions exist:
- **Ground Truth**: The actual colors (unknowable from grayscale alone)
- **DDColor**: A trained neural network's guess (plausible, ~0.4 correlation with GT)
- **From Scratch**: Our geometric approach (Gabor + k-NN, desaturated but structurally correct)

All three are valid paths through φ-space. The question: which geometric shapes control which behaviors?

---

## Pattern Library

### Pattern 1: Averaging → Desaturation

**Shape**: Mean / weighted mean of k neighbors
**Effect**: Colors converge toward gray. The more neighbors averaged, the more muted.
**Why**: Color is a 2D space (a, b). Averaging points scattered around a circle converges to the center (gray).

```
Saturated colors live on the PERIMETER of ab-space
Averaging pulls toward the CENTER
More averaging = more gray
```

**Observed**: v1 (cluster mean) sat=77, v3 (k-NN weighted) sat=83, GT sat=66-115

**Fix options tried**:
| Approach | Effect |
|----------|--------|
| Median instead of mean | Slightly better, still pulls toward center |
| Trimmed mean (remove 20% outliers) | Removes splotches but still desaturates |
| 1.3x saturation boost | Helps but uniform - boosts wrong colors too |
| 1-NN (no averaging) | Most saturated but very noisy/splotchy |

**Key insight**: The fix isn't to boost AFTER averaging. It's to NOT AVERAGE in the first place. Need a selection mechanism, not an aggregation mechanism.

**Geometric interpretation**: Averaging is a CONTRACTION toward origin in ab-space. We need a ROTATION (preserving distance from origin) or SELECTION (picking one point, not blending).

---

### Pattern 2: Gabor Filters at φ-Scales → Texture Discrimination

**Shape**: Gabor wavelets at frequencies f₀ × φⁿ, 8 orientations each
**Effect**: Captures texture at self-similar scales. Distinguishes grass from concrete from glass.

```
Scale 0: f = 0.03        (coarse texture)
Scale 1: f = 0.03 × φ    (medium-coarse)
Scale 2: f = 0.03 × φ²   (medium)
Scale 3: f = 0.03 × φ³   (medium-fine)
Scale 4: f = 0.03 × φ⁴   (fine texture)
```

**Observed**: 40 Gabor features (5 scales × 8 orientations) are the primary discriminators.
They correctly separate sky/trees/buildings/signs in feature space.

**Failure mode**: Materials that look IDENTICAL in grayscale but have different colors (blue glass vs tan concrete at same brightness/texture) cannot be distinguished.

**Geometric interpretation**: Gabor features live on a CYLINDER (angle × frequency). φ-spacing ensures self-similar coverage - same pattern at every scale.

---

### Pattern 3: Position Encoding → Spatial Priors

**Shape**: (y/h, x/w) normalized coordinates, weighted at 0.3×
**Effect**: Provides "sky is usually at the top" type priors.

**Observed at 1.0× weight**: Too strong. Entire top of image goes blue/yellow regardless of content.
**Observed at 0.3× weight**: Gentle prior. Helps with sky but doesn't dominate.
**Observed at 0.0× weight**: No spatial bias at all. Same texture → same color everywhere.

**Key insight**: Position weight controls the PRIOR STRENGTH. Too high = rigid template. Too low = texture-only (misses spatial context). 0.3 is a reasonable balance.

**Geometric interpretation**: Position is a LINEAR GRADIENT across the image. It adds a gentle slope to the feature landscape. Weight controls the slope angle.

---

### Pattern 4: Guided Filter → Edge-Aware Smoothing

**Shape**: Uses grayscale as a "guide" to smooth color predictions while preserving edges.
**Effect**: Colors stay consistent within regions bounded by grayscale edges.

**Parameters**:
| Parameter | Effect |
|-----------|--------|
| radius=5 | Small smoothing, preserves detail |
| radius=15 | Large smoothing, more uniform regions |
| eps=0.01 | Tight edge following |
| eps=0.1 | Looser edge following |

**Observed**: Guided filter is the single most impactful post-processing step. Without it, colors are noisy/splotchy. With it, colors respect object boundaries.

**Geometric interpretation**: The guided filter performs LOCAL LINEAR REGRESSION of color on grayscale intensity. It assumes color = a × intensity + b within each patch. This is a PROJECTION onto the subspace spanned by the grayscale image.

---

### Pattern 5: Bilateral Filter → Outlier Removal

**Shape**: Smooths based on both spatial distance AND color distance.
**Effect**: Removes isolated color splotches without blurring edges.

**Observed**: Applied BEFORE guided filter, removes the green/orange outlier pixels that the k-NN occasionally produces.

**Geometric interpretation**: Bilateral filter is a MANIFOLD SMOOTHER. It assumes the color manifold is locally smooth and removes points that deviate from the local manifold.

---

### Pattern 6: Cluster Count → Color Vocabulary Size

**Shape**: k-means with N clusters on feature space
**Effect**: Controls how many distinct "color concepts" the system can express.

| Clusters | Effect |
|----------|--------|
| 50 | Coarse, few colors, blocky |
| 100 | Moderate variety |
| 200 | Fine, more nuance |

**Observed**: DDColor uses 100 color queries. This appears to be a natural vocabulary size for color concepts.

**Geometric interpretation**: Clusters are VORONOI CELLS in feature space. Each cell gets one color. More cells = finer color resolution but more risk of inconsistency between adjacent cells.

---

### Pattern 7: k-NN k-Value → Confidence vs Diversity

**Shape**: Number of nearest neighbors consulted
**Effect**: Controls the trade-off between confidence (agreeing neighbors) and diversity (varied colors).

| k | Effect |
|---|--------|
| 1 | Most saturated, most noisy |
| 5 | Good color, some noise |
| 15 | Smoother, starts to desaturate |
| 25 | Very smooth, noticeably desaturated |

**Key insight**: k is not a hyperparameter to tune - it's a DESIGN CHOICE about how much certainty you want. Low k = bold guesses. High k = safe averages.

**Geometric interpretation**: k controls the RADIUS of the neighborhood in feature space. Larger radius = more points = more averaging = more contraction toward center.

---

### Pattern 8: Flat Spectrum → Democratic Basis (DDColor query_feat)

**Shape**: 100 query vectors in 256-dim space with nearly uniform singular values
**Effect**: All queries contribute equally. No query is "more important" than another.

```
S[0]=22.5, S[99]=12.8
Zipf alpha = 0.235 (much flatter than 1/phi = 0.618)
Effective rank: 90% at 71 dims, 95% at 82 dims
```

**Meaning**: DDColor treats its 100 color concepts as an EQUAL-WEIGHT basis.
This is fundamentally different from language models where some tokens matter more (scaffolding vs content).
In color space, blue is not "more important" than green.

**Geometric interpretation**: A nearly-orthogonal set of directions in high-dimensional space. Like a democratic vote where each query gets equal say.

---

### Pattern 9: Emerging φ-Structure in Deep Layers (DDColor color_embed MLP)

**Shape**: 3-layer MLP (256→256→256), each with Zipf-like singular value spectrum
**Effect**: Information gets progressively compressed through layers, approaching φ-structure.

```
Layer 0: Zipf alpha = 0.476, condition = 2,360
Layer 1: Zipf alpha = 0.456, condition = 1,481
Layer 2: Zipf alpha = 0.574, condition = 3,542
```

**Key observation**: Layer 2 (deepest) is CLOSEST to 1/φ = 0.618. The model didn't start with φ-structure - it EMERGED through optimization. Deeper = more φ.

**Geometric interpretation**: Each MLP layer is a curved mapping through 256-dim space. The curvature increases with depth, and the optimal curvature approaches φ-scaling. This is consistent with Doc 135 (φ-Zipf duality in attention heads).

---

### Pattern 10: Direction/Magnitude Separation (DDColor refine_net)

**Shape**: Final 2×103 projection where queries map to ab-space at ±0.3 range, but natural colors span ±50.
**Effect**: Queries encode DIRECTION (what hue), attention encodes MAGNITUDE (how saturated).

```
DDColor query color range:  ±0.3
Natural image color range:  ±50
Ratio: ~167x

Color = direction × magnitude
Query says WHAT color → attention says HOW MUCH
```

**This is why our k-NN approach desaturates**: We store actual colors (±50 range) and average them. DDColor stores tiny direction vectors and multiplies by attention-derived magnitude. The magnitude is set PER PIXEL by the transformer - it can be bold for a saturated region and subtle for a muted one.

**Geometric interpretation**: Polar decomposition of color. Direction = angle on the ab circle. Magnitude = distance from origin. DDColor factors these apart. We mush them together.

**Design principle**: When building a system that needs both "what" and "how much", SEPARATE THEM. Store one in the basis (directions), compute the other dynamically (magnitudes).

---

### Pattern 11: Non-Uniform Color Density (Natural Images)

**Shape**: Natural image colors cluster heavily near origin (gray), with sparse but important saturated outliers.
**Effect**: Need MORE query coverage near gray, FEWER at saturated extremes.

```
66.6% of pixels have saturation > 5
Colors cluster on green-yellow axis (foliage, earth tones)
Blue-red axis is sparser but perceptually important
```

**Geometric interpretation**: The natural color manifold is NOT a disk - it's more like a thick-centered star with arms pointing toward common saturated colors (sky blue, foliage green, skin tone, earth brown).

---

## The Scaffolding/Content Wall (Revisited)

Our colorizer hits the SAME wall as the transformer disentanglement (Doc 177):

| Component | Scaffolding | Content |
|-----------|-------------|---------|
| **Colorizer** | Where color boundaries go | Which color goes where |
| **Transformer** | Function word positions | Proper noun identities |
| **Geometric** | Solvable from texture/edges | Requires world knowledge |

The Gabor features perfectly capture SCAFFOLDING (edges, textures, regions).
The k-NN color assignment attempts to capture CONTENT (this texture = blue glass).
The desaturation happens because averaging over uncertain CONTENT converges to gray.

**DDColor solves this** with 55M parameters of learned world knowledge.
**Our scratch approach** has ~95K feature-color pairs. Not enough content.

---

## Open Questions

1. **Can we select instead of average?** Pick the MOST CONFIDENT neighbor instead of blending?
2. **Can φ-structure help with content?** Do natural colors follow φ-harmonic patterns?
3. **Is there a geometric prior for saturation?** Something that says "colors should be bold"?
4. **Can we use the STRUCTURE of the k-NN distances to detect uncertainty?** (High variance among neighbors = uncertain = be bold rather than average?)

---

## Experiment Log

| Version | Approach | Key Change | Result |
|---------|----------|------------|--------|
| v1 | Cluster mean | Baseline | Desaturated, sat=77 |
| v2 | Cluster median + boost | Preserve outliers | Slightly better |
| v3 | k-NN weighted | More variation | Splotchy but saturated |
| v4 | + bilateral/guided | Smooth splotches | Yellow sky splotch |
| v5 | + sky detection | Fix sky | Bled into buildings |
| v6 | Better features (52) | More discriminative | Less confusion |
| v7 | Dominant color selection | Mode instead of mean | Some outlier splotches |
| v8 | Trimmed mean + dual filter | Remove outliers | Smoothest, still desaturated |

---

## Next Steps

- Try CONFIDENCE-WEIGHTED selection (bold when sure, cautious when unsure)
- Analyze whether DDColor's color queries follow φ-structure
- Test if natural image colors follow φ-harmonic distributions
- Build pattern for "saturation preservation" that doesn't require post-hoc boost
