# Design Consideration 120: Geometric Skeleton and Phase Corrections

## Context

In the φ-holographic depth estimation experiments, we achieved MAE ~0.19 using magnitude-only dimensions (vertical gradient, edges, frequency, saliency). However, residual analysis showed no new dimensions could be discovered - the remaining error seemed structureless.

The user's insight: **"If what we're producing is just geometric data then maybe filling things in past this point is phase shifts."**

## The Insight

In holographic/wave systems:
- **Magnitude** = how much of a signal
- **Phase** = how signals combine (constructive vs destructive interference)

Applied to our geometric depth model:
- **Magnitude dimensions** = how much of each depth cue (edges, vertical gradient, etc.)
- **Phase** = signed correction that determines whether to ADD or SUBTRACT each cue

## Experiment Results

### Baseline (Magnitude Only)
```
Dimensions:
  vertical_gradient: weight=2.618
  edges: weight=3.236  
  frequency: weight=1.000
  saliency: weight=0.309

Train MAE: 0.1927
Test MAE: 0.2131
```

### With Phase Correction
Analyzed residuals to find which features predict the SIGN of the error:
```
Correlation of features with residual:
  luminance: r = 0.0527
  edges: r = -0.2488  ← NEGATIVE correlation
  vertical: r = -0.1273
  frequency: r = -0.2028
```

The **edges** feature has r = -0.25 with residuals, meaning:
- Where edges are strong, we OVER-predict depth (residual is negative)
- We need **destructive interference** for edges in some regions

Learned correction: `α = -0.4` for edges

```
Train MAE: 0.1889 (improvement: 0.0038)
Test MAE: 0.2036 (improvement: 0.0095, 4.5%)
```

## Interpretation

### Phase = Sign of Contribution

In wave physics, phase determines whether waves reinforce or cancel:
- Phase 0° → constructive (add)
- Phase 180° → destructive (subtract)

In our model, the signed correction α serves the same role:
- α > 0 → add more of this feature (constructive)
- α < 0 → subtract this feature (destructive)

### Why Edges Need Destructive Correction

The base model weights edges heavily (weight=3.236). But edges don't always indicate depth boundaries - sometimes they're texture within objects at constant depth. The negative α correction reduces edge contribution where edges are strong but don't correspond to depth changes.

## Connection to Holographic Principle

This validates a key aspect of the holographic model:

1. **Magnitude encodes WHAT** - the strength of each depth cue
2. **Phase encodes HOW** - whether cues reinforce or cancel at each location

The remaining error after phase correction is likely:
- Semantic content (object identity, scene understanding)
- Information that requires learning from data, not geometric extraction

## Implementation

```python
# Base prediction (magnitude only)
depth = Σ weight_i × magnitude_i

# With phase correction
depth = Σ weight_i × magnitude_i + Σ α_j × correction_feature_j

# Where α can be positive (constructive) or negative (destructive)
```

## The Skeleton Hypothesis

User insight: **"What if what we're generating is the skeleton of a subsequent model that could then be used to make adequate depth estimation?"**

### Experiment: Skeleton as Structural Prior

Tested four models to understand what the geometric skeleton provides:

| Model | Test MAE | Description |
|-------|----------|-------------|
| Skeleton Only | 0.2132 | φ-weighted composition, no learning |
| Skeleton + Corrections | **0.1990** | Skeleton + learned signed corrections |
| No Skeleton (equal init) | 0.2100 | Learn weights from scratch |
| No Skeleton (φ-init) | 0.2094 | Start with φ-weights, then learn |

### Key Finding: The Skeleton's True Value

The skeleton is NOT valuable because of:
- Its **φ-weights** (learner discovers similar weights: only 0.0007 difference)
- Its **composition structure** (no significant advantage)

The skeleton IS valuable as a **foundation for refinement**:
- Skeleton + corrections achieves best MAE (0.1990)
- 5% better than learning from scratch (0.2100)

### Interpretation

The geometric skeleton provides:
1. **Relevant dimensions** - which features matter for depth (edges, vertical gradient, etc.)
2. **Reasonable starting point** - a prior that's "close enough" to refine
3. **Stable foundation** - corrections can be learned on top without destabilizing

This is analogous to how biological systems work:
- DNA provides the scaffold/skeleton
- Epigenetics provides the corrections/refinements
- The scaffold constrains the solution space, making learning tractable

### Implications for TruthSpace

Our φ-geometry doesn't need to be the final answer. It can be:
1. **The skeleton** that defines the structure
2. **Phase corrections** that refine the structure
3. **Learned components** that fill in semantic details

The geometry IS the scaffold. What we're discovering is the structural prior that makes subsequent learning possible.

## φ-Lattice and φ-Zipf Experiments

### Question: Would absolute φ-lattice coordinates help?

From Design 099, the insight is that similarity matrices give RELATIVE positions, but we want ABSOLUTE positions on a φ-lattice where positions are at φ^k for integer k.

### Experiment: φ-Lattice Coordinates for Depth

Tested using absolute φ-lattice coordinates with semantic dimensions:
- vertical_position: φ^k where k encodes height in frame
- edge_strength: φ^k where k encodes boundary importance
- texture_frequency: φ^k where k encodes detail level
- saliency: φ^k where k encodes attention importance

| Approach | Test MAE |
|----------|----------|
| φ-Lattice (default φ^k) | 0.4033 |
| φ-Lattice (learned mapping) | 0.2093 |
| Previous best (phase holographic) | 0.1990 |

**Finding**: Quantizing to discrete φ-levels loses information. The learned mapping helps but doesn't beat continuous features with phase corrections.

### Experiment: φ-Zipf Duality

Tested two interpretations of φ-Zipf:

**1. Rarity-based (feature presence)**
- Weight by inverse of mean feature value (rarer = higher weight)
- Result: Inverted importance of vertical (0.081 vs 0.447)
- Test MAE: 0.303 (worse than baseline)

**2. Correlation-based (predictive power)**
- Weight by correlation with depth
- Discovered actual correlation order: `vertical (0.66) > saliency (0.20) > edges (0.14) > frequency (0.14)`
- Our assumed order was: `vertical > edges > frequency > saliency`

**Key Finding**: φ-Zipf duality applies to INFORMATION content (correlation with target), not feature rarity. Saliency is more predictive than edges or frequency, but our original φ-weights had it lowest.

### Why φ-Lattice Alone Doesn't Match Phase Holographic

The phase holographic model (MAE 0.199) uses:
1. **Complex interference**: magnitude × e^(iφ) with learned phases
2. **Coordinate descent**: optimizes both weights AND phases
3. **Spatially-varying phase**: edges use orientation, frequency uses local Fourier phase

The φ-lattice approach (MAE 0.312) uses:
1. **Simple weighted sum**: no phase/interference
2. **Fixed weights**: correlation-ranked but not optimized
3. **Single correction term**: less expressive than full phase model

### Conclusion

The φ-lattice provides a valid coordinate system, but:
- **Discrete quantization loses information** compared to continuous features
- **φ-Zipf should weight by correlation** (predictive power), not rarity
- **Phase/interference is essential** for combining features optimally
- **The skeleton + phase corrections** remains the best approach

The insight from Design 099 is correct: absolute positions are better than relative. But for depth estimation, the "absolute position" is the continuous feature value, and the "navigation" is the phase correction that determines how features combine.

## Physical Mechanisms Experiment

### User Insight

> "If our φ-Zipf (correlation-based) is correct, but our interpretation isn't correct, maybe we're missing out on multiple sources of light, polarization, and the bend of the lens. It seems like we're doing things correctly, but we're operating in the output space, not the internal physical mechanisms."

### Experiment: Physical Mechanism Model

Implemented depth estimation based on physical mechanisms:
1. **Shading** (Lambertian: I ∝ cos(θ))
2. **Shadows** (occlusion from light source)
3. **Defocus blur** (depth of field)
4. **Chromatic aberration** (R/B focus difference)
5. **Texture gradient** (perspective)
6. **Vanishing point geometry**
7. **Polarization proxy** (specular vs diffuse)

| Model | Test MAE | Notes |
|-------|----------|-------|
| Physical Mechanisms | 0.2053 | Close but not better |
| Multi-Light Model | 0.2561 | Worse - estimation too simplistic |
| Output Space (best) | 0.1990 | Still best |

### What the Optimizer Found

```
shadow: weight=0.354, phase=135° (highest!)
vanishing: weight=0.309, phase=0°
shading: weight=0.286, phase=90°
defocus: weight=0.219, phase=0°
chromatic: weight=0.068, phase=0°
texture: weight=0.000 (disabled)
polarization: weight=0.000 (disabled)
```

Key insight: **Shadow** is the most informative physical mechanism, with phase=135° (partially destructive interference).

### Why Physical Model Doesn't Beat Output Space

The problem is **ill-posedness**. The rendering equation:

```
Image = ∫ Light(ω) × BRDF(ω, ω') × Geometry(ω') dω
```

Has many (Light, BRDF, Geometry) solutions for the same image. We're trying to invert this without:
1. **Full light field** (we estimate direction, not distribution)
2. **Material properties** (BRDF varies per pixel)
3. **Semantic priors** ("sky" is always far, regardless of pixels)

Depth Anything V2 learns these priors from millions of images. Our geometric model captures the **structure** but lacks the **semantic disambiguation**.

### The Deeper Insight

**Output space**: Features → Depth (statistical correlation)
**Physical space**: Light × Material × Geometry → Image (generative model)

The φ-geometric interpretation:
- **φ-lattice** = coordinate system of physical world (light, material, geometry)
- **Phase** = how physical quantities combine (interference)
- **Zipf** = prior distribution over scenes (common vs rare configurations)

Our output-space model works because it learns **statistical regularities** of how physical mechanisms manifest. The physical model tries to derive these from first principles but lacks semantic understanding.

### Implication for TruthSpace

This validates the **skeleton + corrections** approach:
1. **Skeleton** = geometric structure (physical mechanisms)
2. **Corrections** = learned refinements (semantic priors)
3. **Neither alone is sufficient** - need both

The geometry provides the **coordinate system**, but navigation requires **learned priors** about what configurations are likely.

## Files

- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_phase_holographic.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_dynamic_dimensions.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_skeleton_model.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_phi_lattice.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_svd_positional.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_physical_model.py`
