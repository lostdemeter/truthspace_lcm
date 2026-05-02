# Design Consideration 214: Geometric Colorization Roadmap

## Current State Assessment

### What We've Built

| Component | Status | MAE | Notes |
|-----------|--------|-----|-------|
| Baseline (nearest-neighbor) | ✓ | 27.78 | Blocky, slow |
| Sharp (YUV separation) | ✓ | 11.85 | Preserves detail |
| Path clusters | ✓ | 11.32 | Semantic grouping |
| Focused (6D SVD) | ✓ | 14.45 gen | Best generalization |
| Geometric manipulation | ✓ | 6.59 | 0.5x scale helps |

### Key Discoveries

1. **Color is NOT linearly encoded** (R² ≈ 0.05)
   - Unlike DA2's depth, color requires non-linear relationships
   - Same texture can be many colors (grass vs rust vs concrete)

2. **Fewer dimensions generalize better**
   - 4-6D beats 16-24D on new images
   - More dims = overfitting to training data

3. **We're oversaturating**
   - 0.5x saturation scale improves MAE by 19%
   - The true path has less saturation than we predict

4. **Structure emerges in clusters**
   - Clusters correspond to semantic categories
   - But clusters are discrete, not continuous

### The Core Problem

We're treating this as a **lookup problem** when it's actually a **transformation problem**.

```
CURRENT APPROACH:
  features → find nearest cluster → return cluster's color
  
WHAT WE NEED:
  features → apply geometric transformation → get color
```

The lookup approach fails because:
- Clusters are discrete islands, not a continuous manifold
- New images fall between clusters
- We're memorizing, not learning the transformation

## The Hypothesis

**The grayscale→color transformation is a GEOMETRIC OPERATION on a manifold.**

Like how:
- Gender flip is Δx = -2.0 (king→queen)
- Tense change is a rotation
- Negation is a reflection

There should be a **single transformation** (or small set) that maps grayscale features to color, and it should be expressible geometrically.

## The Plan

### Phase 1: Understand the Manifold

**Goal**: Find the structure of the grayscale→color mapping.

1. **Collect paired data**: (grayscale_features, true_color) for many patches
2. **Analyze the mapping**: 
   - Is it a rotation? 
   - A projection?
   - A φ-scaled transformation?
3. **Find invariants**: What's preserved under the transformation?

### Phase 2: Learn the Transformation

**Goal**: Express the mapping as a geometric operation.

Instead of:
```python
color = nearest_neighbor_lookup(features)
```

We want:
```python
color = geometric_transform(features, learned_parameters)
```

Where `geometric_transform` could be:
- A rotation matrix in joint (feature, color) space
- A projection onto a color subspace
- A φ-scaled warping

The key: **few parameters** that capture the universal transformation.

### Phase 3: Validate Generalization

**Goal**: Prove the transformation works on unseen images.

- Train on images 0-100
- Validate on images 200-210
- Test on images 300-310

If the transformation is truly geometric, it should generalize perfectly (or near-perfectly) because geometry is universal.

### Phase 4: Refine with Ground Truth

**Goal**: Use the inverse laser copier principle.

Given ground truth, we can:
1. Measure error in predicted color
2. Adjust transformation parameters to reduce error
3. The adjustment should be geometric (not gradient descent)

## Specific Experiments

### Experiment 1: Rotation Analysis

**Question**: Is the grayscale→color mapping a rotation in joint space?

```python
# Joint space: [features, U, V]
# If it's a rotation, we should find:
#   color_point = R @ grayscale_point
# Where R is a rotation matrix
```

### Experiment 2: Projection Analysis

**Question**: Is color a projection of features onto a subspace?

```python
# If it's a projection:
#   [U, V] = P @ features
# Where P is a projection matrix (low rank)
```

### Experiment 3: φ-Transformation Analysis

**Question**: Does the transformation involve φ-scaling?

```python
# If it's φ-scaled:
#   color = Σ φ^(level_i) × feature_i
# Where levels are learned
```

### Experiment 4: Manifold Learning

**Question**: What's the intrinsic dimensionality of the mapping?

```python
# Use techniques like:
# - Isomap
# - t-SNE
# - UMAP
# To find the manifold structure
```

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Test MAE | 11.32 | 8.0 | 5.0 |
| Generalization MAE | 14.45 | 10.0 | 8.0 |
| Parameters | ~25 clusters × 8D | < 50 | < 20 |
| Interpretability | Clusters | Transform | Single operation |

## The Vision

If we succeed, we'll have:

1. **A geometric transformation** that maps grayscale→color
2. **Few parameters** (maybe just a rotation matrix or φ-levels)
3. **Perfect generalization** (because geometry is universal)
4. **Interpretable structure** (we can explain WHY it works)

This would validate the core hypothesis:
> Structure IS information. The transformation IS the knowledge.

## Progress Update

### Transformation Analysis Results

| Analysis | R² for U | R² for V | Notes |
|----------|----------|----------|-------|
| Linear | 0.008 | 0.031 | Too low |
| Quadratic | 0.063 | 0.066 | 8x better |
| Manifold (6D) | 0.006 | 0.016 | Still low |

**Key finding**: Quadratic texture interactions (con×tex_h, tex_h×tex_v) are the most predictive features.

### Principled Colorizer Results

| Metric | Value |
|--------|-------|
| Test MAE | 12.15 |
| Generalization MAE | 15.23 |
| Gap | 3.08 |

### The Core Problem

Even with quadratic features and manifold projection, R² is only ~0.06. This means:

**Color is NOT deterministically encoded in grayscale features.**

The same grayscale texture CAN be multiple colors. This is fundamentally different from DA2's depth, which IS deterministically encoded.

### Revised Understanding

The colorization problem has TWO components:

1. **Deterministic component**: Some color information IS encoded
   - Sky tends to be blue (position + smoothness)
   - Grass tends to be green (position + texture)
   - This is what our R² of 0.06 captures

2. **Ambiguous component**: Most color information is NOT encoded
   - A gray texture could be concrete, metal, fabric, etc.
   - This requires CONTEXT or SEMANTICS we don't have

### The Path Forward

**Option A: Accept the limitation**
- Use nearest-neighbor for the ambiguous component
- Use manifold projection for the deterministic component
- Combine them

**Option B: Add context**
- Include neighboring patches
- Global scene features
- This might resolve ambiguity

**Option C: Semantic features**
- Use a pretrained model (CLIP, etc.) for semantic understanding
- But this violates our "pure geometry" principle

### The Real Insight

We've been treating this as "find the transformation" when we should be treating it as "find the CONSTRAINTS."

In the Music Box Principle:
- The **drum** contains the data (examples)
- The **comb** contains the constraints (geometric rules)
- The **music** emerges from their interaction

For colorization:
- The drum = our training examples (grayscale, color) pairs
- The comb = the geometric constraints we've discovered
- The music = the colorized output

**The constraints we've found:**
1. Saturation should be scaled by ~0.5x
2. Texture interactions (con×tex_h) predict color direction
3. 5-6 dimensions capture the essential structure
4. Position (y_pos) correlates with warmth (sky vs ground)

**What we're missing:**
The constraints should SHAPE the drum, not replace it. We need:
1. A drum populated with examples
2. Constraints that GUIDE how we traverse the drum
3. The constraints reduce the search space, they don't eliminate it

### The Principled Architecture

```
INPUT: grayscale patch

STEP 1: Extract constrained features
  - Use the 6D manifold projection
  - This reduces 22 features to 6 essential dimensions
  - The manifold IS a geometric constraint

STEP 2: Query the drum in constrained space
  - Find nearest neighbors in 6D manifold space (not raw features)
  - The constraint focuses the search

STEP 3: Apply geometric corrections
  - Scale saturation by 0.5x
  - This is a learned constraint from ground truth

OUTPUT: predicted color
```

The key insight: **Constraints don't replace the drum, they focus it.**

### Implementation Plan

1. **Build the constrained drum**
   - Store examples in manifold-projected space
   - This is geometrically principled (SVD finds the natural basis)

2. **Query with geometric focus**
   - Use the manifold distance, not raw feature distance
   - Weight by the singular values (importance)

3. **Apply learned corrections**
   - Saturation scaling
   - Any other geometric adjustments from manipulation experiments

4. **Refine constraints with ground truth**
   - When we have the answer, adjust the constraints
   - Not the drum contents, but the geometric rules

This is the "inverse laser copier" you described:
- The drum is the "charge pattern"
- The constraints are the "optics"
- We adjust the optics (constraints) to get the right output

## φ-Native Architecture (Latest)

### The Breakthrough Insight

> "Since everything needs to pass through our phi focal point, we can reduce the number of searches that we need to do because invalid paths simply wouldn't exist"

Instead of filtering through φ, **make the model BE φ**:
- Features quantized to φ-grid positions
- Only φ-valid positions exist in the drum
- Invalid paths are impossible by construction

### Results

| Approach | Test MAE | Gen MAE | Drum Size | Gap |
|----------|----------|---------|-----------|-----|
| Baseline (NN) | 27.78 | - | ~18K | - |
| Sharp (YUV) | 11.85 | - | ~18K | - |
| Focused (6D) | 11.41 | 14.45 | ~18K | 3.04 |
| Constrained Drum | 11.74 | 16.17 | ~18K | 4.43 |
| φ-Focal | 12.29 | 16.62 | ~12K | 4.32 |
| **φ-Native (8 levels, 4D)** | **12.49** | **15.81** | **8,194** | **3.32** |

### Key Discoveries

1. **Fewer dimensions generalize better**: 4D beats 6D
2. **More φ-levels = finer resolution**: 8 levels optimal
3. **Compression is natural**: 18K → 8K (44%)
4. **The φ-grid constrains the search space**

### The φ-Native Model

```python
# Quantize to φ-grid
level = log_φ(value)  # Find φ-level
level_int = round(level)  # Quantize
position = φ^level_int  # Only valid positions

# The drum contains only φ-valid entries
# Invalid paths literally don't exist
```

### Next Steps

1. **Optimize the φ-grid**: Find the ideal balance of levels and dimensions
2. **Learn the grid structure**: Let the data determine optimal φ-levels
3. **Hierarchical φ-structure**: Coarse-to-fine refinement
4. **Validate on diverse images**: Test generalization across domains

## IK (Inverse Kinematics) Approach

### The Analogy

| IK Concept | Colorization Equivalent |
|------------|------------------------|
| End effector | Target color (U, V) |
| Joint angles | φ-dimension values |
| Joint limits | φ-grid quantization |
| Jacobian | How joints affect color |
| Redundant DOF | Multiple paths to same color |

### Results

| Joints | R²_U | R²_V | Test MAE | Gen MAE |
|--------|------|------|----------|---------|
| 2 | 0.006 | 0.001 | 11.93 | 15.85 |
| 6 | 0.025 | 0.023 | 11.86 | **15.57** |

**Best generalization: 15.57** with 6 joints.

### The Jacobian

```
J_U = [ 0.0005, -0.0032, -0.0042,  0.0014,  0.0023,  0.0261]
J_V = [ 0.0005, -0.0017,  0.0105, -0.0111, -0.0059, -0.0166]
```

This tells us:
- Joint 6 strongly affects U (blue-yellow)
- Joints 3-4 strongly affect V (red-green)
- The relationship is LINEAR in joint space

### Key Insight

The IK framing gives us:
1. **Interpretability**: We know how each dimension affects color
2. **Solvability**: Given target, we can solve for joints
3. **Constraints**: φ-grid acts as joint limits
4. **Redundancy**: Multiple solutions possible (style freedom)

## Phase Transition / Modal Selection

### The Problem: Dull Colors

Linear averaging of colors gives dull, washed-out results. The centroid of multiple color modes is always less saturated than the modes themselves.

### The Solution: Choose Modes, Don't Average

Color exists in discrete MODES:
- Mode 5: sat=0.003 (neutral/gray) - 46% of data
- Mode 6: sat=0.299 (saturated) - rare but vivid
- Other modes: moderate saturation

### Results

| Mode Selection | Test MAE | Gen MAE |
|----------------|----------|---------|
| weighted | 12.41 | 17.15 |
| **max** | 12.85 | **15.14** |

**Max mode selection achieves best generalization: 15.14**

### The Phase Transition

The transition is about COMMITMENT:
- Low confidence → average → dull (superposition)
- High confidence → choose mode → saturated (collapse)

Like quantum measurement:
- Superposition of color modes
- Measurement collapses to eigenstate
- The "measurement" is mode selection

### Best Results Summary

| Approach | Gen MAE | Key Insight |
|----------|---------|-------------|
| Baseline NN | ~28 | Just lookup |
| φ-Native (4D) | 15.81 | Quantize to φ-grid |
| IK (6 joints) | 15.57 | Learn Jacobian |
| IK Null Space | 15.53 | 4/6 DOF free |
| **Modal (max)** | **15.14** | Choose modes, don't average |

## Self-Assembly Experiments

### Results

| Approach | Structure | Gen MAE |
|----------|-----------|---------|
| v1 (basic attractor/repeller) | 66 anchors | 16.23 |
| v2 (diversity preservation) | 1,382 anchors | 15.71 |
| v3 (two-level clustering) | 129 phases, 5,881 clusters | 16.18 |

### Key Finding

Self-assembly finds what's IN the data, but not necessarily what's USEFUL for prediction.

The pre-defined 8-mode structure (Gen MAE = 15.14) outperforms all self-assembled structures because:
1. It forces **commitment** to a mode
2. It has the right **granularity** (not too many, not too few)
3. The phase transition (choose vs average) is explicit

### The Lesson

For colorization:
- **Descriptive structure** (what colors exist) ≠ **Predictive structure** (what colors to use)
- Self-assembly is good for discovery, but needs task-specific refinement
- The optimal number of modes (~8) may be a fundamental property of color perception

---

*This document will be updated as we progress through the plan.*
