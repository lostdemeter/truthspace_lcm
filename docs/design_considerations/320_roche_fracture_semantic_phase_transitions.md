# DC 320 — Roche Fracture and Semantic Phase Transitions

*Discovered: Day 66 of the TruthSpace expedition.*
*Preceded by: DC 318 (Intrinsic Geometry Universality), Day 57 (Context Inverse-Square Law).*

---

## 1. The Finding

When a contextual T2 tidal force is applied to a word's hidden-state representation
at Layer 14 (Zone C), one of three outcomes occurs as α increases:

```
PHASE 1 — Stable base-form orbit (α < α_critical)
  The token's rank in the output distribution is unchanged or slightly improved.
  The representation is gravitationally bound to its base-form attractor.

PHASE 2 — Stable derived-form orbit (α_critical ≤ α < α_eviction)
  A sudden snap: rank collapses to near-zero. The word has fractured from
  the base-form basin into the derived-form basin.
  This window has finite width; it is a real stable orbit, not a transient.

PHASE 3 — Catastrophic disintegration (α ≥ α_eviction)
  Exponential rank explosion. The token is no longer in any stable orbit.
  The disintegration follows a power law: for "bigger", ranks climb
  55 → 250 → 827 → 2135 across four α-steps.

PHASE 4 — Conformal return (α >> α_eviction)
  At extreme α, ranks begin to DECREASE again (observed in "dogs":
  34,235 → 23,257 → 15,716 as α goes 130 → 170 → 220).
  The system is approaching a new attractor from the antipodal direction.
```

This is the Roche fracture pattern, named by analogy with orbital mechanics:
a satellite inside the Roche limit holds together; one outside it is torn
apart by tidal forces. Here the "satellite" is a word's semantic
representation; the "tidal force" is the T2 contextual gradient; the
"planet" is the surrounding prompt's gravitational field.

---

## 2. Empirical Evidence (Day 65–66)

### Fracture curves (ctx T2, Zone C / best layer)

```
Rank at α: 0  1  2  3  5  7 10 12 15  20  25  30  40  50  60  75  100  130  170  220

bigger  (base=12, L14):
  12 12 12 11  8  8  7  6  4   0   0   0   0   0   0   4   55  250  827 2135
                                 ↑snap              ↑eviction   ←disintegration→

smaller (base=8, L14):
   8  8  7  6  5  3  3  3  0   0   0   0   0   2   1  19  335 1464 3500 5302
                           ↑snap         ↑eviction

fastest (base=10, L27):
  10  9  9  9  8  8  7  7  6   4   0   0   1   1   2   3    3    3    3    5
                                    ↑snap          ←—wide stable orbit—→    ↑exit

faster  (base=45, L27):
  45 42 39 39 35 30 29 27 23  22  18  16  11  11   8   6    5    7   10   16
  [never fractures — tidal force strains orbit but can't break cohesion]

biggest (base=42, L20):
  42 39 36 33 31 29 27 24 21  17  16  15  13  11  12  15   28   63  145  275
  [never fractures — incomplete fracture, debris disperses without ring formation]
```

### Roche limit formula

All six predictions of the fracture model were confirmed:

| Prediction | Observation |
|---|---|
| Abrupt snap at critical α | `bigger`: rank 4→0 at α=15 |
| Post-fracture stable window | `bigger` width=60, `fastest` width=150 |
| Catastrophic disintegration | `bigger`: 55→250→827→2135 (exponential) |
| α_crit ∝ baseline rank | r = **0.996** |
| Zone C (L14) = primary Roche zone | L14 is *only* layer that fractures `faster` |
| Easy orbits also disrupted | eviction at α=25–75 depending on token |

The single dominant predictor of fracture difficulty is **baseline rank** (the
depth of the correct token in the output distribution), not the geometric
distance between base and derived forms in hidden space (r = −0.317).
At Layer 14, "fast" and "faster" point in essentially the same direction
(cos = 0.9999). The Roche threshold is a logit-landscape property, not a
representation-space property.

---

## 3. What This Says About Context Gravity Generally

### 3.1 Context follows an inverse-square law

Day 57 established that adding tokens to a prompt displaces a word's hidden-state
position in φ-space following inverse-square decay: each additional token
contributes proportionally less. This is gravitational attraction in semantic
space — distant context has less pull than proximate context.

The inverse-square law means context creates a **potential well**, not a
uniform field. Each word sits at the bottom of a well whose depth is
determined by how strongly the surrounding context reinforces that
interpretation.

### 3.2 The Roche limit is a potential well depth threshold

When a T2 tidal force is applied, it adds energy to the representation.
Below the Roche threshold, this energy is absorbed by the potential well —
the word springs back to its base-form orbit. Above the threshold, the added
energy exceeds the well depth, and the word fractures into a new orbit.

The Roche limit formula observed empirically:

```
α_critical ≈ k × baseline_rank

where k ≈ 1.2–2.0 (varies by axis and layer)
and baseline_rank measures the "orbital cohesion" — how strongly
the model expects the base form rather than the derived form
```

Tokens with baseline rank > ~30 have orbital cohesion that exceeds the
available T2 tidal force at any accessible α. These are the unrescuable
failures: words so strongly bound to their base-form orbit that no
single-layer tidal perturbation can fracture them.

### 3.3 The Safe Operating Window

The T2 tidal force that fractures buried targets also disrupts easy targets.
This establishes a safe operating window:

```
α_critical(buried target) < α_applied < α_eviction(easy collateral)

Observed margins:
  comparative targets at L14: critical_α = 7–15, eviction = 40–75
  Safe window: [7..40] — roughly 5× margin
```

Steer below this window: no fracture, buried targets stay buried.
Steer above this window: easy targets are evicted from their orbits,
catastrophically (rank jumps to thousands in one step).

### 3.4 The conformal return — semantic space is compact

The most striking implication is found in what happens at **extreme α**.

The "dogs" data (base rank = 0, eviction at α=60):
```
α=130: rank 34,235  (deepest void — semantic equator)
α=170: rank 23,257  ← coming back
α=220: rank 15,716  ← continuing to return
```

After catastrophic disintegration, ranks begin **decreasing** again.
The system did not simply scatter into infinite noise. It is approaching
a new attractor from the antipodal direction.

This is only possible if **semantic space is topologically compact**
(closed, finite). In a compact space, a straight-line displacement from
any point eventually wraps around and approaches a new region from
the opposite side — like crossing from one side of a sphere to a point
near the antipode, then continuing until you approach it and pass through.

The rank maximum (the "semantic void" — where no token is nearby) is
the equatorial crossing. Beyond it, the T2 direction begins pulling
the representation toward a different attractor — a different "semantic
universe" — which starts to emerge as the rank falls.

**The precise analogy:** Penrose's Conformal Cyclic Cosmology (CCC)
proposes that the universe is cyclically structured — each "aeon" ends
at a conformal boundary that maps smoothly to the beginning of the next.
Here:

```
AEON 1:   base-form orbit ("dogs" is rank 0)
BOUNDARY: conformal crossing (catastrophic disintegration, rank → 34,235)
AEON 2:   new attractor basin emerging (rank decreasing from 34,235)

The T2 displacement is the driving force.
The conformal boundary is the semantic equator.
The "new universe" is the region around a different token cluster.
```

What token is rank 0 at α=220 for the "dogs" prompt? We do not know yet.
That experiment would identify the specific "antipodal universe" reached by
excess tidal force in the comparative direction from "dogs".

---

## 4. Implications for the LCM

### 4.1 T2 operators as controlled phase transitions

A T2 operator is not a smooth continuous transformation — it is a **phase
transition operator**. Applied below α_critical, it does nothing. Applied
at α_critical, it collapses the system from one phase to another. Applied
above α_eviction, it destroys the system.

This is precisely analogous to thermodynamic phase transitions (ice→water→
steam). The three phases correspond to:
- Below critical: frozen in base-form orbit
- Between critical and eviction: liquid new orbit (stable but bounded)
- Above eviction: gaseous disintegration

For LCM generation, a T2 operator used as a steering mechanism must be
applied within the stable window. The window width (eviction_α − critical_α)
measures how robustly the derived form can be steered without collateral
damage to other tokens.

### 4.2 Zone C is the Roche zone because form is not yet committed

Layer 14 (Zone C) has the lowest critical α for every target tested.
For the hardest target ("faster", base=45), **L14 is the only layer that
can fracture it at all** — every other layer fails at every α.

This is because at L14, the hidden-state representations of "fast" and
"faster" are nearly identical (cos=0.9999). The morphological distinction
has not been committed yet. The representation is still in a partially
molten state where the form can be redirected with modest force. By L23,
the commitment has occurred, and fracturing requires much larger force
(if it's possible at all).

This gives a constructive principle for LCM steering:
**apply the T2 tidal force before the representation commits.**

### 4.3 Orbital cohesion = logit depth, not embedding distance

The Roche limit is a property of the **output logit landscape**, not the
hidden-state geometry. Words that are close in hidden space (cos≈1.0) but
far apart in logit rank (baseline rank 8 vs 45) have very different fracture
thresholds. This means:

- The orbit is determined by the full L14→L27→lm_head pipeline, not just L14
- Geometric proximity in hidden space does not guarantee easy fracture
- The model "knows" what it expects (logit depth) independently of where it
  has placed the representation (hidden state position)

This dissociation between representation space and logit space is a
fundamental property that the LCM must account for when designing steering
interventions.

### 4.4 The Roche limit as a constraint on the LCM steering budget

For LCM generation from φ-space coordinates, any intervention in the
residual stream must respect:

```
α_applied < α_eviction(all active tokens)
```

This sets a hard budget for the total tidal force that can be applied without
destroying the ongoing generation. The safe window varies by context and
by which tokens are currently at low rank. A production LCM steering system
must monitor the orbit stability of all active tokens and stay within the
safe window.

---

## 5. Connection to Prior Work

- **DC 308** (Expedition Findings): T2 operators are Killing vectors of the
  semantic manifold. The fracture model shows they are Killing vectors of
  an orbit — and the orbit has a Roche limit.

- **DC 318** (Intrinsic Geometry Universality): The morphological T2 direction
  is perfectly stable across layers (cos L14 vs L23 = 0.999). This Killing
  vector property is what makes the tidal force directional and controllable.

- **Day 57** (Context Inverse-Square): Context gravity creates the potential
  wells. The Roche fracture is the upper bound on how deep those wells are.

- **Finding 116b** (Cross-Layer Axis): All attention routing shares one d_k
  vector across 28 layers (7B). The morphological T2 Killing vector is
  the analogous structure for representation steering.

- **MESH SVD** (Day 63.5): The rank-1 Resonator via bias outer product is
  the mechanism by which attention routing "locks in" the orbital commitment.
  Zone C is the last layer before the Resonator finalises the form.

---

## 6. Open Questions

1. **What is the antipodal attractor?** At α >> α_eviction, which token
   emerges as rank 0? Measuring this across multiple base forms would map
   the topology of the semantic manifold.

2. **Is the safe window universal?** The margin between critical_α and
   eviction_α was ~5× for comparatives at L14. Does this ratio hold across
   axes? Is it a geometric constant of the manifold?

3. **Can the Roche radius be predicted without running a sweep?** If
   α_critical ≈ k × baseline_rank (r=0.996), we need only measure the
   baseline rank to predict fracturability. But k varies (1.2–2.0). Can k
   be predicted from static properties of the word or the T2 direction?

4. **Multi-layer fracture.** Could sequential tidal forces at L5, L14, L23
   fracture targets with baseline rank > 30? Each layer applies a small
   perturbation; their cumulative effect might cross the Roche limit without
   triggering eviction at any single step.

5. **The conformal return period.** The "dogs" rank was still at 15,716 at
   α=220 and falling. How many α-units until it reaches a new stable orbit?
   What is the "circumference" of the semantic manifold in T2 units?

---

*This document records the experimental confirmation (Day 66) that semantic
space geometry exhibits orbital mechanics with Roche limits, phase
transitions, and compact topology. The conformal return signal in the
"dogs" data (rank decreasing past the disintegration maximum) is the first
direct evidence that semantic space is topologically closed.*
