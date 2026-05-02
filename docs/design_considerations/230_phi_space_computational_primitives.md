# Doc 230: φ-Space Computational Primitives

## The Question

> "If knowledge is a shape, can we create that shape ourselves?"

We have proven:
1. DDColor's learned weights ARE a geometric shape (V16 extracts them perfectly)
2. That shape has measurable properties (flat spectrum, emerging φ, direction/magnitude separation)
3. The shape was NOT designed - it EMERGED from optimization on a recognition task

The question is NOT "can we copy the shape" (we already did - V16).
The question is: **can we DERIVE the shape from the constraints alone?**

If the shape is determined by the task, we shouldn't need to train a model.
We should be able to CONSTRUCT the shape from first principles.

---

## The Five Geometric Primitives

Every operation DDColor performs is one of these five transformations:

### 1. PROJECTION (→ subspace)
**What it does**: Selects which dimensions matter. Discards the rest.
**Computation**: Like attention - "look at THIS, ignore THAT"
**φ-property**: Projected subspaces follow Zipf-1/φ importance weighting

```
Full space → Subspace
[256 dims] → [rank 90% at 71 dims]
Keep 28% of dimensions, capture 90% of information
```

**Where DDColor uses it**:
- Encoder: pixel space → 256-dim feature space
- refine_net: 103-dim → 2-dim ab space

### 2. ROTATION (preserve norms)
**What it does**: Changes direction without changing magnitude. Transforms meaning.
**Computation**: Like translation in language - "king" rotated = "queen"
**φ-property**: Optimal rotations approach golden angle (137.5°)

```
Input direction → Output direction
Magnitude unchanged
Information PRESERVED, not created or destroyed
```

**Where DDColor uses it**:
- Color embed MLP: rotates through 256-dim space 3 times
- Each rotation has Zipf alpha approaching 1/φ

### 3. DILATION (scale magnitude)
**What it does**: Makes things bigger or smaller. Amplifies or suppresses.
**Computation**: Like confidence - "THIS region is very blue" vs "this region is slightly blue"
**φ-property**: Natural scaling by powers of φ

```
direction × magnitude → direction × (new magnitude)
Only the "how much" changes, not the "what"
```

**Where DDColor uses it**:
- Attention weights: scale each query's contribution per pixel
- This is the DYNAMIC part - it changes per image

### 4. CONTRACTION (→ center)
**What it does**: Pulls toward the average. Reduces variance.
**Computation**: Like consensus - everyone compromises toward the middle
**φ-property**: Destroys information. Converges to fixed point.

```
Multiple points → Their mean
Variance DECREASES
Information LOST
```

**Where OUR colorizer uses it** (the bug):
- k-NN averaging: blue + yellow → gray
- This is why we desaturate!

**Key insight**: DDColor does NOT use contraction. It uses PROJECTION (selection).
We've been using the wrong primitive.

### 5. REFLECTION (flip sign)
**What it does**: Inverts across an axis. Negation.
**Computation**: Like "not-blue" or "the opposite of warm"
**φ-property**: φ and 1/φ are reflections (φ × 1/φ = 1)

```
+a → -a (red becomes green)
+b → -b (yellow becomes blue)
```

**Where DDColor uses it**:
- W_input has negative weights (G→a is -0.094, G→b is -0.150)
- Green channel OPPOSES both a and b axes

---

## DDColor as a Shape Program

DDColor's entire pipeline, expressed as geometric primitives:

```
Step 1: PROJECT   pixel space → feature space (encoder)
Step 2: ROTATE    features through attention (Q@K.T)
Step 3: PROJECT   attention scores → weights (softmax = projection onto simplex)
Step 4: DILATE    query contributions by attention (per-pixel scaling)
Step 5: ROTATE    query activations through MLP (3 layers, approaching φ-Zipf)
Step 6: PROJECT   256-dim → 2-dim ab space (refine_net)
```

The PROGRAM is: **Project → Rotate → Project → Dilate → Rotate → Project**

This is 3 projections, 2 rotations, 1 dilation.
No contractions. No averaging. No information destruction.

---

## Can We Build This Without Training?

For each primitive, we need to determine its parameters from first principles:

### Projection 1 (pixel → features)
**What we need**: A way to extract meaningful features from pixels
**From first principles**: Gabor filters at φ-scaled frequencies ARE this projection.
They project pixel space onto texture space. We already have this.
**Status**: ✓ SOLVED

### Rotation 1 (feature transformation)
**What we need**: A rotation that maps texture features to color-relevant features
**From first principles**: This is where world knowledge lives.
"This texture at this position = sky = blue" is a ROTATION in feature space.
**Status**: ✗ THIS IS THE WALL

### Projection 2 (attention/selection)
**What we need**: A way to select which "color concept" applies to each pixel
**From first principles**: If we had the right features, nearest-neighbor IS a projection.
But it's a projection in the WRONG space (feature space, not semantic space).
**Status**: ~ PARTIALLY SOLVED (k-NN does selection, but in wrong space)

### Dilation (magnitude)
**What we need**: Per-pixel magnitude (how saturated should this pixel be?)
**From first principles**: Local contrast, edge proximity, and position
provide some magnitude information. But the semantic component is missing.
**Status**: ~ PARTIALLY SOLVED

### Rotation 2 (color embedding)
**What we need**: Map selected color concept to actual ab direction
**From first principles**: If our selection were correct, this would just be a lookup.
The rotation is only needed because the selection space ≠ color space.
**Status**: ✓ TRIVIALLY SOLVED (once selection works)

### Projection 3 (to ab)
**What we need**: Final 2D projection
**From first principles**: This IS the ab coordinate system. Nothing to learn.
**Status**: ✓ SOLVED

---

## The Shape of the Wall

The wall is Rotation 1: the mapping from texture to semantics.

But look at what we know about this rotation's SHAPE:
- It approaches Zipf alpha = 1/φ in its singular values
- Its effective rank is ~100 dimensions (90% variance)
- It's a composition of 3 sub-rotations (MLP layers)
- Each sub-rotation has condition number ~1500-3500

**Hypothesis**: We don't need to know WHAT the rotation does.
We need to know its SHAPE. And its shape is:

```
R = R₂ ∘ R₁ ∘ R₀

Where each Rᵢ has:
- Singular values following S[j] = S[0] / j^αᵢ
- α₀ ≈ 0.48, α₁ ≈ 0.46, α₂ ≈ 0.57
- Approaching 1/φ with depth
```

Can we construct a random rotation with THESE PROPERTIES and get plausible colorization?

If the SHAPE determines the behavior, then any rotation with the right shape
should produce plausible (though different) colors.

If the CONTENT of the rotation matters, then only the trained rotation works,
and shape alone is insufficient.

This is a TESTABLE HYPOTHESIS.

---

## The Experiment

Build a colorizer where:
1. Features: Gabor at φ-scales (we have this)
2. Rotation: RANDOM matrix with Zipf-1/φ singular values
3. Selection: Softmax over 100 queries (project onto simplex)
4. Dilation: From local image statistics
5. Color lookup: 100 color directions from natural image KMeans

If this produces plausible colors → shape IS sufficient
If this produces garbage → content matters, shape is necessary but not sufficient

---

## Connection to Prior Work

| Discovery | Implication for Colorizer |
|-----------|--------------------------|
| Doc 135: φ-Zipf in attention | Rotation shape is universal |
| Doc 177: Scaffolding/Content wall | Features = scaffolding (solvable), color = content (wall) |
| Doc 180: Bulge = geodesic deviation | Color assignment = bulge (10 coefficients per concept?) |
| Doc 160: Intelligence is geometric | The shape IS the computation |

The bulge discovery is particularly relevant: if world knowledge is encoded
in only 10 coefficients per concept (2,867x compression), then the "world knowledge"
for colorization might be similarly compressible.

100 color concepts × 10 coefficients = 1,000 numbers.
That's the entire "world knowledge" for colorization?

---

## Experimental Result: v10

### Setup
Random rotation matrices with three spectral shapes:
- Flat (α=0): all singular values equal
- φ-Zipf (α=1/φ): DDColor's measured shape
- Steep (α=2): sharply decaying spectrum

Pipeline: Gabor features → shaped rotation → softmax → color lookup → smooth

### Result: ALL THREE PRODUCE GRAY

The saturation metric lied (measuring noise). Visual inspection: all gray.

**Why**: Softmax over random scores → near-uniform weights → weighted average
of 100 color centers → mean of all colors → gray.

This is contraction through the back door. We thought we replaced averaging
with selection, but softmax(random) IS averaging.

### What This Proves

**Shape (spectral envelope) is NECESSARY but NOT SUFFICIENT.**

The specific ORIENTATION of the rotation encodes world knowledge.
A random rotation with perfect φ-Zipf spectrum is as useless as a flat one.

This is analogous to: having the right SIZE box doesn't help if the box is empty.
The spectrum defines the container. The orientation defines the content.

---

## Revised Understanding: Three Layers of Shape

The v10 result reveals that "shape" has three distinct layers:

### Layer 1: Operation Type (SOLVED)
Which geometric primitive to use.
- Contraction = averaging = gray (BAD)
- Projection = selection = preserves saturation (GOOD)
- Our v8→v9→v10 journey proved this matters enormously

### Layer 2: Spectral Envelope (NECESSARY)
The singular value distribution of the transformation.
- Flat = democratic (all features equal)
- φ-Zipf = hierarchical (some features more important)
- Steep = dominated by few features
- DDColor learned φ-Zipf. This is NOT arbitrary.
- But having the right envelope with wrong orientation = gray

### Layer 3: Orientation (THE WALL)
Which specific directions in feature space map to which colors.
- "This texture = sky = blue" is an orientation choice
- "This edge pattern = foliage = green" is an orientation choice
- DDColor learned these from millions of images
- Random orientations = garbage regardless of envelope

**The question becomes**: Is Layer 3 (orientation) DERIVABLE from constraints?

---

## Is Orientation Derivable?

Multiple valid orientations exist (DDColor ≠ Ground Truth, both plausible).
So orientation is NOT unique. There's a FAMILY of valid orientations.

What constrains the family?

### Structural Constraints (from the input)
1. **Spatial coherence**: connected similar-texture regions → same color
2. **Edge alignment**: color boundaries align with intensity boundaries
3. **Scale consistency**: same texture at different scales → same color

### Statistical Constraints (from natural images)
4. **Distribution matching**: output colors should follow natural image statistics
5. **Correlation structure**: sky pixels are correlated with each other, etc.

### Semantic Constraints (world knowledge)
6. **Object consistency**: a single object has consistent color
7. **Physical plausibility**: sky=blue, grass=green, skin=skin-tone
8. **Scene coherence**: colors within a scene should "make sense" together

Constraints 1-3 are pure GEOMETRY → derivable from shape alone.
Constraints 4-5 are STATISTICS → derivable from data distribution.
Constraints 6-8 are SEMANTICS → require world knowledge.

The wall is constraints 6-8. But notice:
- v8 (k-NN) partially captures 6-8 through example-based lookup
- The desaturation comes from the OPERATION (contraction), not the knowledge

### Hypothesis (revised)
**The correct approach is k-NN content + softmax selection.**
Not random rotation + softmax, and not k-NN + averaging.
We need the right CONTENT (from data) delivered through the right OPERATION (selection).

---

## Connection to Bulge Theory (Doc 180)

The bulge discovery showed:
- World knowledge compresses to 10 coefficients per concept
- 100 color concepts × 10 coefficients = 1,000 numbers
- That's potentially ALL the "world knowledge" for colorization

The rotation's ORIENTATION might be encodable as:
- 100 color directions (from natural image clustering) = 200 numbers
- 100 sets of feature-space coordinates (which features activate each color) = ???

If we can identify WHICH FEATURES activate each color concept,
we have the orientation. This is not random - it's derived from data.
But it might be derivable from very LITTLE data (the bulge is only 10 coefficients).

**Next experiment**: Learn the rotation from minimal data (5-10 images)
and test generalization. If it generalizes → the orientation has low complexity
and is essentially derivable. If not → full training is required.

---

## Pattern Summary (Updated)

| # | Pattern | Type | Status |
|---|---------|------|--------|
| 1 | Averaging → Desaturation | Operation | Proven |
| 2 | Gabor at φ-scales → Texture | Projection | Working |
| 3 | Position weight → Spatial prior | Parameter | Tuned |
| 4 | Guided filter → Edge smoothing | Post-process | Working |
| 5 | Bilateral filter → Outlier removal | Post-process | Working |
| 6 | Cluster count → Color vocabulary | Parameter | Working |
| 7 | k-NN k-value → Confidence/diversity | Parameter | Tuned |
| 8 | Flat spectrum → Democratic basis | Envelope | Measured |
| 9 | Emerging φ in MLP → Natural compression | Envelope | Measured |
| 10 | Direction/magnitude separation | Architecture | Understood |
| 11 | Non-uniform color density | Distribution | Measured |
| 12 | **Shape envelope ≠ knowledge** | **NEGATIVE** | **v10 proved** |

The most important finding is #12: spectral shape is necessary but not sufficient.
The content (orientation) of transformations encodes irreducible world knowledge.
But that knowledge may be highly compressible (bulge theory suggests ~1000 numbers).
