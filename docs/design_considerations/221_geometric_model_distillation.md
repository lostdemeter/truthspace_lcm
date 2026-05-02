# Design Consideration 221: Geometric Model Distillation

## The Problem

We have two endpoints:
1. **Minimal geometric model** (19 atoms, 21 parameters) - produces semantic overlays
2. **DDColor** (2.4M parameters) - produces realistic colorization

The gap between them is enormous. Our minimal model understands "sky is blue" but doesn't know how to apply that knowledge to real images.

**Question**: Can we bridge this gap without training on real data?

---

## The Insight

DDColor already works. It's been trained. Its weights are on the φ-lattice.

We don't need to train from scratch. We can **distill** DDColor's knowledge into a geometric form.

```
Distillation = Using a complex model to teach a simpler one
```

The "simulated data" is DDColor's own output. We use DDColor to generate (input, output) pairs, then fit a geometric model to reproduce them.

---

## The Approach

### Traditional Distillation
```
Teacher (DDColor) → Student (smaller model)
    |                    |
    v                    v
  2.4M params         fewer params
```

### Geometric Distillation
```
Teacher (DDColor) → Geometric Student
    |                    |
    v                    v
  2.4M params         φ-lattice structure
  (opaque)            (interpretable)
```

The difference: we're not just compressing parameters. We're **extracting the geometric structure** that DDColor learned.

---

## The Method

### Step 1: Generate Distillation Data

Run DDColor on diverse images to create (grayscale, color) pairs.

```python
for image in dataset:
    gray = to_grayscale(image)
    color = ddcolor(gray)
    pairs.append((gray, color))
```

We don't need real ground truth. DDColor's output IS the target.

### Step 2: Extract Feature-Color Mappings

For each pixel, extract:
- **Local features**: luminance, gradient, texture
- **Global context**: position, semantic region
- **Output color**: ab values from DDColor

```python
for gray, color in pairs:
    for pixel in image:
        features = extract_features(gray, pixel)
        ab = color[pixel]
        mappings.append((features, ab))
```

### Step 3: Fit Geometric Model

Find the φ-lattice structure that best predicts color from features.

```python
# Option A: Expand our atoms
atoms = fit_atoms(mappings)  # Learn atom positions from data

# Option B: Learn attention weights
attention = fit_attention(mappings)  # Learn which atoms to use when

# Option C: Learn the full mapping
model = fit_geometric_model(mappings)  # End-to-end geometric fit
```

### Step 4: Validate

Compare geometric model output to DDColor output.

```python
for gray in test_images:
    color_ddcolor = ddcolor(gray)
    color_geometric = geometric_model(gray)
    error = compare(color_ddcolor, color_geometric)
```

---

## What We're Really Doing

### The φ-Lattice Hypothesis

DDColor's weights live on the φ-lattice. We've proven this:
- 100% Fibonacci structure in weight differences
- Peak at φ^-9 across all layers
- Self-similar across scales

If the weights are geometric, the **function** they compute is geometric.

Distillation extracts this function in explicit form.

### The Compression Question

How much can we compress?

| Representation | Parameters | Notes |
|----------------|------------|-------|
| DDColor full | 2,384,896 | Opaque, trained |
| DDColor queries only | 25,600 | 100 queries × 256 dims |
| Rank-20 queries | 7,120 | Low-rank approximation |
| Our 19 atoms | 21 | Too simple, doesn't work |
| Distilled geometric | ? | What we're finding |

The answer depends on the **intrinsic dimensionality** of the colorization task.

---

## The Key Questions

### Q1: What features matter?

DDColor uses:
- ConvNeXt encoder (texture, edges, objects)
- Multi-scale features (local + global)
- Cross-attention (feature → query matching)

Which of these are essential? Which are redundant?

### Q2: How many atoms do we need?

Our 19 atoms are semantic categories. DDColor has 100 queries.

But the queries have effective rank ~94. Are all 94 dimensions needed?

Hypothesis: A smaller set of **geometric atoms** (not semantic) might suffice.

### Q3: What is the attention doing?

DDColor's cross-attention matches image features to color queries.

This is essentially: "For this texture/context, which color applies?"

Can we express this as a geometric operation?

---

## The Distillation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    GEOMETRIC DISTILLATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────────────┐     │
│  │  Images  │────▶│ DDColor  │────▶│ (gray, color)    │     │
│  │  (any)   │     │ (teacher)│     │  pairs           │     │
│  └──────────┘     └──────────┘     └────────┬─────────┘     │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │ Feature          │     │
│                                    │ Extraction       │     │
│                                    └────────┬─────────┘     │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │ Geometric        │     │
│                                    │ Model Fitting    │     │
│                                    └────────┬─────────┘     │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │ φ-Lattice        │     │
│                                    │ Colorizer        │     │
│                                    └──────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Works

### 1. DDColor Already Solved the Problem

We're not learning colorization from scratch. We're learning to **approximate** a working solution.

### 2. The Approximation is Geometric

Because DDColor's weights are on the φ-lattice, the approximation will also be geometric.

### 3. Simpler Models Can Approximate Complex Ones

This is the core insight of distillation. A student model can often match a teacher with far fewer parameters.

### 4. We Control the Representation

Unlike neural distillation, we choose the geometric form. This gives us interpretability.

---

## The Experiments

### Experiment 1: Feature Analysis

What features predict DDColor's output?

```python
features = [luminance, gradient_x, gradient_y, 
            position_x, position_y, local_variance]
            
for feature in features:
    correlation = correlate(feature, ddcolor_output)
    print(f"{feature}: {correlation}")
```

### Experiment 2: Atom Expansion

Can we learn better atoms from DDColor's output?

```python
# Cluster DDColor's color outputs
colors = ddcolor_outputs.reshape(-1, 2)  # All ab values
centroids = kmeans(colors, n_clusters=100)

# These centroids are our new atoms
atoms = centroids
```

### Experiment 3: Attention Distillation

Can we learn when to use which atom?

```python
# For each pixel, which atom does DDColor effectively choose?
for pixel in pixels:
    features = extract_features(pixel)
    color = ddcolor_output[pixel]
    
    # Find closest atom
    closest_atom = argmin(distance(color, atoms))
    
    # Learn: features → atom selection
    attention_data.append((features, closest_atom))

# Fit a simple model
attention_model = fit(attention_data)
```

### Experiment 4: End-to-End Geometric Fit

Can we fit a full geometric model?

```python
# The model: features → φ-lattice → color
model = GeometricColorizer(n_atoms=100, n_features=10)

# Train on DDColor outputs
for gray, color in pairs:
    pred = model(gray)
    loss = mse(pred, color)
    model.update(loss)
```

---

## Connection to Knowledge Chemistry

The Knowledge Chemistry framework (Doc 219-220) provides the structure:

| Chemistry | Distillation |
|-----------|--------------|
| Atoms | Learned color centroids |
| Molecules | Feature-atom associations |
| Reactions | Context-dependent transformations |

Distillation **populates** the chemistry with learned knowledge.

---

## The Path Forward

### Phase 1: Data Generation
- Run DDColor on 10,000 images
- Extract (gray, color) pairs
- Store feature-color mappings

### Phase 2: Analysis
- What features predict color?
- How many atoms are needed?
- What is the attention structure?

### Phase 3: Model Fitting
- Fit geometric atoms
- Fit attention/selection model
- Combine into geometric colorizer

### Phase 4: Validation
- Compare to DDColor output
- Measure compression ratio
- Test on new images

### Phase 5: Generalization
- Apply same approach to other models
- Depth estimation, segmentation, etc.
- Build library of geometric distillation tools

---

## The Ultimate Goal

```
Any trained model → Geometric distillation → φ-lattice representation
```

If we can distill any model into geometric form:
1. We understand what it learned
2. We can compress it
3. We can combine knowledge from multiple models
4. We can build new models from geometric principles

This is the bridge between:
- **Trained models** (opaque, effective)
- **Geometric models** (interpretable, principled)

---

## Files

| File | Purpose |
|------|---------|
| `phi_geometric/evaluations/minimal_colorizer.py` | Our 19-atom baseline |
| `phi_geometric/evaluations/extract_ddcolor_atoms.py` | Extract DDColor's queries |
| `phi_geometric/evaluations/run_ddcolor_correct.py` | Run DDColor correctly |
| `docs/design_considerations/221_geometric_model_distillation.md` | This document |

---

## Conclusion

Geometric Model Distillation is the bridge between trained models and geometric understanding.

We don't need real training data. We use the trained model itself to generate the data we need.

The result is a geometric model that:
1. Reproduces the trained model's behavior
2. Is expressed on the φ-lattice
3. Is interpretable and compressible
4. Can be combined with other geometric knowledge

This is how we "train without training" - by distilling existing knowledge into geometric form.
