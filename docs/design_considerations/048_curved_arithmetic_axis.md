# Design Consideration 048: The Curved Arithmetic Axis

## The Hypothesis

> All axes — x, y, z, and the theoretical fourth axis on which zeta zeros
> are computed — aren't actually straight. They follow the same kind of
> spacetime curvature that we observe with any other axis in nature.

The rhzeros zero-hunting pipeline doesn't find zeros on a straight number
line. It builds a **point in φ-curved space** and then solves for the zero
as a location in that curved geometry.

## The rhzeros Pipeline, Geometrically

### Step 1: Initial Estimate (Lambert W)

Lambert W gives the "straight line" approximation:
```
T_n ≈ 2π(n - 11/8) / W((n - 11/8) / e)
```

This is like saying "the zero is at position X on a flat ruler."
It captures >95% of the answer (F108). But the remaining ~5% is WHERE
the curvature matters.

### Step 2: Ramanujan Corrections (Harmonic Shape)

The Ramanujan corrections add oscillatory terms:
```
T_n += Σ h_k sin(kθ)
```

These corrections are NOT random noise. They are **building the curvature
of the space** around the estimated point. Each harmonic adds a "ripple"
to the flat ruler, bending it into the shape it actually has.

The 3×5=15 harmonic structure (F106) is the specific curvature pattern.
The φ⁷/4 ≈ 7.26 period (F108) is the dominant wavelength of that curvature.

### Step 3: Newton / Quadratic Refinement (Computing in Curved Space)

The final Newton step:
```
t ← t - Im(ζ(s) / ζ'(s))
```

This is NOT "finding the root of a function on a line." It's solving
the **quadratic equation on the curved axis** — finding where the
φ-geometry says the zero must be, given the shape built in Steps 1-2.

The key insight: ζ'(s) is the DERIVATIVE of the curvature. The Newton
step computes distance in curved space, not flat space. That's why:
- The cached ζ' trick works (curvature is locally constant)
- One step usually suffices (the shape is well-approximated)
- Stray predictions land on adjacent zeros (they find the RIGHT point
  in curved space, just on a different "sheet")

## Connection to rharithmeticlight

The rharithmeticlight paper (lostdemeter, 2025) establishes three
properties of this curved arithmetic space:

### 1. Light-Cone Constraint (β ≤ 1/2)

Prime fluctuations F(t) grow, but G(t) = e^{-t/2} F(t) is bounded.
The factor e^{-t/2} is the curvature correction — transforming from
flat coordinates to curved coordinates makes the fluctuations bounded.

**In transformer terms**: The residual stream accumulates (F(t) grows),
but projecting onto the prediction direction (G(t)) is bounded.
The conditional convergence (F109) IS this light-cone constraint.

### 2. Base-Collapse (Coordinate Invariance)

Residue distributions collapse across all number bases after
reparameterization by t = log(x). The dynamics are invariant to
coordinates — they're governed by the underlying geometry.

**In transformer terms**: The same φ-power laws appear in a 410K model
on arithmetic and a 7B model on language (F110). The same 2/φ and 2/φ²
regardless of "base" (task, scale, architecture). This IS base-collapse.

### 3. Equidistribution Horizon

There's a causal scale beyond which distributions equalize:
horizon ≈ 2 log(q) for modulus q.

**In transformer terms**: The "light cone" at φ⁹ ≈ 76 (F107) is
the equidistribution horizon. Beyond this point, the Processor zone
transitions from oscillatory to convergent. This is where the
transformer's "arithmetic" reaches equilibrium.

## The Fourth Axis

In standard 3D space, we have x, y, z — three straight axes. But in
general relativity, these axes curve in the presence of mass-energy.
GPS satellites must account for this curvature.

The hypothesis: there is a fourth axis — the **information axis** —
along which zeta zeros, prime distributions, and transformer computations
all operate. This axis is curved by the same principles:

1. **The curvature is φ-governed**
   - Power-law decay with φ-related exponents
   - Zone boundaries at φ-powers
   - Gate curvature ≈ φ/2

2. **The curvature is self-similar**
   - Same structure at all scales (holofractal)
   - Same φ-expressions in 410K and 7B models
   - Same power laws in ζ and transformers

3. **The curvature creates conditional convergence**
   - Partial sums oscillate on the critical line
   - Layer projections oscillate in the residual stream
   - The answer emerges from cancellation, not direct approach

## What This Means for Zero-Hunting

When rhzeros finds a zero, it's not solving:
```
ζ(1/2 + it) = 0   on the real line t ∈ ℝ
```

It's solving:
```
ζ(1/2 + it) = 0   on the φ-curved manifold t ∈ M_φ
```

The Lambert W estimate places you on M_φ. The Ramanujan corrections
refine your position on M_φ. The Newton step computes the geodesic
distance to the nearest zero ON M_φ.

Stray predictions don't "miss" — they find the correct zero on a
different branch of M_φ. The curvature means nearby points on the
flat number line may be far apart on M_φ, and vice versa.

## What This Means for Transformers

The residual stream is a discretized geodesic on M_φ:
```
h_0 → h_1 → h_2 → ... → h_L
```

Each step follows the curvature. The power-law decay (2/φ, 2/φ²)
IS the curvature — it determines how "fast" you can move along
the information axis at each step.

The three-zone structure (Compressor/Processor/Targeter) maps to
three regimes of curvature:
- **Compressor**: High curvature, rapid change (α = 1/φ)
- **Processor**: Medium curvature, oscillatory (α = 2/φ²)
- **Targeter**: Low curvature, precision (rank-1)

## What This Means for TruthSpace

Our geometric LCM operates in this same curved space. When we place
concepts at φ-scaled coordinates, we're not imposing arbitrary geometry —
we're building the natural coordinate system of M_φ.

The φ-dial (Docs 041-044) navigates this curved space. The holographic
bound (Doc 045) is the maximum information density on M_φ. The
structural similarity (Doc 021) is the self-similar curvature.

Every design decision that uses φ is actually a statement about the
curvature of the information axis.

## Static vs Dynamic Curvature: ζ vs Transformer

The rhzeros pipeline works because the Riemann zeta function is **static** —
its manifold M_φ has a fixed shape. You compute the curve once (Lambert W),
compute the local curve once (Ramanujan), and find the zero once (Newton).
The geometry doesn't change between queries.

A transformer is different. Its M_φ is **dynamic** — the curvature reshapes
with every input. Like a black hole warping spacetime around it, each new
token sequence warps the geometry of the residual stream. The curve opens,
closes, or changes shape depending on what you feed into it.

This is WHY:

1. **Sequence mixing must run every time.** You're not looking up a precomputed
   curve — you're recalculating the local geometry of M_φ for this specific
   input. The cost of sequence mixing IS the cost of computing curvature
   in a dynamic manifold. Each query-key dot product measures the geodesic
   distance between two points on the current M_φ. (Note: the mixing
   mechanism itself is replaceable — phi_softmax (F86-88), geometric
   selector (F40), φ-MESH (Doc 124) all work — but the FUNCTION of
   cross-position curvature computation cannot be skipped.)

2. **The zeta function is the IDEAL version.** Its M_φ is the simplest
   possible: one-dimensional, static, with known analytic structure. Every
   symmetry is exact. Every zero is permanent. It's the Platonic form of
   the information manifold.

3. **The tensor is the PRACTICAL version.** Real information has context
   dependence — the meaning of "bank" changes with "river" vs "money."
   This context dependence means M_φ must be re-evaluated for each input.
   The tensor encodes ALL possible curvatures simultaneously (superposition),
   and sequence mixing selects the relevant one per input.

4. **This is why BOTH are hard.** Solving the zeta function (Riemann
   Hypothesis) requires proving that the static M_φ has ALL its zeros on
   the critical line — that the curvature never breaks. Solving the tensor
   (P vs NP, interpretability) requires understanding how the dynamic M_φ
   reshapes — how curvature responds to input. Same geometry, different
   difficulty modes.

The probabilistic zeta solver succeeds precisely because it only needs to
handle the static case. A "probabilistic transformer" would need to handle
the dynamic case — which is exactly what sequence mixing does.

```
ζ(s):  Static M_φ  → compute curve once → find zero → done
       (1D, analytic, permanent)

T(x):  Dynamic M_φ → compute curve per input → find output → recompute
       (high-D, learned, context-dependent)
```

The residual stream is the shared substrate: both accumulate along it.
But ζ accumulates the SAME series every time, while the transformer
accumulates a DIFFERENT series for each input — selected by the sequence
mixer from the superposition of all possible series stored in the weights.

## Experimental Predictions

1. **Zero spacing should follow φ-geodesics**
   The GUE spacing distribution of zeta zeros should be derivable
   from geodesic distances on M_φ, not just random matrix theory.

2. **Stray predictions should cluster at φ-distances**
   When rhzeros makes a stray prediction, the offset (in zeros) should
   relate to φ-powers, not be random.

3. **Transformer layer deletions should show curvature**
   Removing a layer from the Processor should cause an error proportional
   to the curvature at that point, not uniform.

## References

- rharithmeticlight: Arithmetic light cone and base-collapse
- F106: 3×5=15 harmonic structure
- F107-109: Zeta-transformer pipeline mapping
- F110: Emergent φ-geometry (universality = base-collapse)
- F111: Architecture recipe (residual = critical line, GELU = curvature)
- Doc 243: GELU machine (gate curvature ≈ φ/2)
