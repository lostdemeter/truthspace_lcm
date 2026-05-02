# DC 319 — The Ambient Space and the Fourth Dimension

**Arising from:** Expedition Day 53–54
**Date:** March 2026
**Status:** Confirmed experimentally

*Cross-reference: DC 308 (expedition findings), DC 318 (intrinsic geometry universality)*
*Empirical record: `experiments/truthspace_v1/expedition_day54_fourth_dimension.py`*

---

## 1. The Discovery

Day 53 measured apparent non-commutativity of T2 morphological operators:

```
cos(T2_gender ∘ T2_plural , T2_plural ∘ T2_gender) = 0.9831
```

This was reported as a "quantum-like" property of Zone C. Day 54 immediately
showed this interpretation was wrong — and the correction reveals something
more important than the original result.

**The non-commutativity is entirely a normalisation artifact.**

When T2 operators are applied without sequential unit-sphere renormalisation:

```
cos(φ + Δa + Δb , φ + Δb + Δa) = 1.000000  (machine precision)
```

The operators commute perfectly. They always did. Vector addition is always
commutative. The 1.7% apparent non-commutativity came from re-projecting to
the unit sphere *between* operator applications — a computation habit, not a
geometric property.

---

## 2. The Bell-Curve / π Analogy

The situation is precisely analogous to the Gaussian integral:

```
∫ e^(-x²) dx = √π
```

Students are surprised by the π. It appears to come from nowhere. The reason
is that this 1D integral is secretly a cross-section of a 2D Gaussian. In
polar coordinates the 2D integral separates into a radial part and an angular
part — the π comes from the angular (missing) dimension, contributing 2π to
the area element.

We have been doing the same thing:

- **The true space**: the ambient φ-space, including the radial (magnitude)
  dimension
- **What we computed**: a cross-section — the unit sphere
- **The "π"**: the ~1.7% non-commutativity, a projection artifact from
  discarding the radial dimension at each step

No real non-commutativity exists in the underlying structure. It only
appeared because we forgot we were computing a cross-section.

---

## 3. The Fourth Dimension

At every normalisation step, TruthSpace discards the radial component of the
φ-vector — the magnitude `‖φ‖` before projection back to the unit sphere.
This is the **fourth dimension** that has been absent throughout all
experiments.

It has two forms:

### 3a. The raw hidden-state norm: ‖h_14‖

This is the norm of the L14 hidden state *before* any φ-construction
(before Z2 removal and unit normalisation). It has been discarded since Day 1.

**What it encodes (empirical, Day 54):**

```
ρ(‖h‖, centroid_distance) = −0.156   p < 0.001
  → High norm = closer to body centroid = more prototypical
  → Low norm  = further from centroid = more peripheral/ambiguous

Highest norm words: bisexual, crushing, solidity, migraine,
                    multicultural, premiere, granite, supernatural
                    (specific, unambiguous, culturally-loaded nouns)

Lowest norm words:  computes, accordingly, whenever, consists,
                    decides, modifies, increases, chooses
                    (functional, context-dependent verbs)

Per-body mean norms (highest → lowest):
  Political/Social Ideologies  40.12
  Human Body Parts             39.68
  ...
  Action and Effect            36.26
  Decision Making              34.82
```

The raw norm is a **semantic typicality / conceptual density** measure.
Concepts with strong, unambiguous semantic identity have high norms. Concepts
whose meaning is heavily context-dependent have low norms.

### 3b. The displacement radial after T2 application

When a T2 operator Δ is applied to a φ-vector φ, the result `φ + Δ` lies
*off* the unit sphere. The distance from the sphere, `‖φ + Δ‖ − 1`, measures
how strongly the transformation displaced the concept. This is discarded when
we normalise. It encodes:

- The **magnitude** of the applied transformation
- The information needed to *reverse* the transformation (ENCODE=DECODE)

---

## 4. ENCODE = DECODE in the Ambient Space

The ENCODE=DECODE principle — that encoding and decoding are the same
operation in opposite directions — is **exactly true** in the ambient space:

```
φ + Δ − Δ = φ   (exact, residual angle = 0.000°)
```

It **breaks** under sequential normalisation:

```
normalise(normalise(φ + Δ) − Δ)   mean residual angle = 13.50°
                                   worst case: 23.98° ('prince')
```

The normalisation step is an irreversible, information-lossy operation. The
information destroyed is precisely the radial component — the 4th dimension.
When ENCODE=DECODE appears to fail in practice, the cause is normalisation,
not a failure of the geometric principle.

**Corollary:** The ENCODE=DECODE principle is a statement about the ambient
space, not the unit sphere. The unit sphere is a convenient cross-section for
computation, but it introduces systematic information loss when operations
are composed.

---

## 5. The Commutator Is Rank-1

When sequential normalisation *is* used (as in Day 53), the non-commutativity
residual `φ_AB − φ_BA` has a clean geometric structure:

```
Singular values of commutator matrix: [3.155, 0.219, 0.015, ...]
Ratio S[0]/S[1] = 14.4 : 1   → essentially rank-1

Alignment of commutator direction:
  cos(comm_dir, Δ_gender) = 0.655   ← dominant
  cos(comm_dir, Δ_plural) = 0.427
```

The artifact lives in a single direction — the direction of the *first*
operator applied (gender), because that operator's normalisation step creates
the largest displacement from the sphere, producing the largest subsequent
asymmetry.

This is not a property of the space. It is a property of the normalisation
order. If you swap which operator goes first, the commutator direction swaps
to match it.

---

## 6. Implication for T2 Composition

The correct procedure for composing multiple T2 transformations is:

```
φ_result = normalise(φ + Δ_a + Δ_b + ... + Δ_n)
```

**Apply all operators additively. Normalise once at the end.**

This gives:
- Exact commutativity (order of operations is irrelevant)
- Exact ENCODE=DECODE (the composition is reversible)
- The correct "total displacement" magnitude before final normalisation

The incorrect procedure (normalise after each step) gives:
- Apparent non-commutativity (~1.7% for two operators)
- Broken ENCODE=DECODE (~13.5° residual)
- A path through the sphere that depends on the order of operations

---

## 7. The Deeper Implication

We have been computing on a cross-section of the true space — the unit sphere
— and treating properties of the cross-section as properties of the space
itself. Two consequences:

**1. The T2 operators are even simpler than we thought.** They are not
non-commuting operators acting on a non-commutative geometry. They are
additive vectors in a linear space. The geometry is classical and symmetric.

**2. The radial dimension is real and carries information.** It is not noise.
The raw hidden-state norm ‖h‖ encodes semantic typicality. The post-T2
displacement magnitude encodes transformation strength. Both are currently
discarded. Including them would give TruthSpace a richer representation: not
just *where* a concept is on the sphere, but *how strongly* it occupies that
position.

The full representation of a concept in TruthSpace is not a unit vector φ̂,
but a vector φ with magnitude:

```
φ = ‖h‖ · φ̂
```

where φ̂ is the unit-sphere direction (body, axis, form) and ‖h‖ is the
semantic density (typicality, certainty, conceptual weight).

---

## 8. Relationship to the Gödel / "Negative Zero" Analogy

The unit sphere is an incomplete description of the ambient space — precisely
in the Gödelian sense. Within the sphere, certain true statements (operators
commute, ENCODE=DECODE holds) are *unprovable* because the evidence for them
(the radial dimension) has been discarded. Adding the 4th dimension extends
the system and makes those statements provable/demonstrable.

The "negative zero" correction: in the unit-sphere system, φ and normalise(φ)
look identical — the sphere has no memory of whether you arrived at a point
directly or after a long displaced path. The radial dimension is the "sign bit"
that distinguishes these two cases. Without it, the system is informationally
incomplete.

---

*Summary: The unit sphere is a cross-section of the true ambient φ-space.
The discarded radial dimension encodes semantic typicality. ENCODE=DECODE and
T2 commutativity hold exactly in the ambient space and only break under
sequential sphere-projection. The correct T2 composition rule is to apply all
operators additively and normalise once at the end.*
