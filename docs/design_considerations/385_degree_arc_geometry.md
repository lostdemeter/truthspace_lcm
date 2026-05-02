# DC 385: Degree Arc Geometry in W_E

**Geometric Audit + Degree Arc + Degree Plane experiments**

**Central finding: adjective degree gradation in W_E is encoded as a
consistent circular arc in a word-specific 2D subspace of the embedding
space. The arc parameters are universal across all tested adjectives.**

---

## Why This Investigation Was Necessary

The preceding work (Days 237–243) drifted into statistical methods:
- LOO retrieval accuracy comparisons
- Centroid neighbourhood analysis
- Nearest-neighbour voting strategies
- Mean direction computation

These are not wrong, but they are not geometric. The TruthSpace hypothesis
says geometry IS computation. The correct question is: **what mathematical
transformation maps emb(pos) → emb(comp) → emb(sup)?**

---

## Geometric Audit Results

### A. Norm Audit

No morphological transformation is norm-preserving (isometric):

| Paradigm | norm_b/norm_a | std | Type |
|---|---|---|---|
| adj_pos2sup | 1.053 | 0.036 | Scaling (+5.3%) |
| adj_pos2comp | 1.013 | 0.035 | Scaling (+1.3%) |
| past_tense | 0.931 | 0.026 | Scaling (−6.9%) |
| gender | 0.975 | 0.056 | Variable |
| antonym_size | 1.025 | 0.075 | Variable |

A pure rotation would give ratio = 1.000 with std ≈ 0. None qualify.
All morphological transformations are **scaled rotations** with
word-specific scaling factors.

### B. Linear Map

The best-fit linear map M = B @ pinv(A) has SVD singular values
ranging 0.63–1.80. A pure rotation would have all singular values = 1.
The transformation is an anisotropic scaled rotation: it expands in
some dimensions of the paradigm subspace and contracts in others.

### C. Translation vs Rotation

Translation residual (mean direction model):
- antonym_size: 0.672 (worst — no consistent direction)
- past_tense: 0.417 (best — most consistent direction)

The rotation model (M = B @ pinv(A)) fits the training data with
zero residual by construction (n << H degrees of freedom). But this
comparison reveals: the mean direction model has substantial residual
even for the highest-accuracy paradigm (past_tense: 0.417). The
transformation has word-specific components that the mean vector
cannot capture.

### D. Path Curvature on the Unit Sphere

For adj_degree triples (pos, comp, sup) measured on the unit sphere:

```
θ(pos→comp): mean = 54.7°  std = 4.1°
θ(comp→sup): mean = 57.2°  std = 2.8°
θ(pos→sup):  mean = 60.6°  std = 3.9°
Sum of steps: 111.9°
Direct arc:    60.6°
Excess:        51.3°  =  85% of the direct distance
```

**pos→comp→sup is NOT collinear.** The path on the unit sphere subtends
nearly double the arc of the direct geodesic. The comparative form is
49.4° off the great circle connecting pos and sup.

This rules out the simple translation model geometrically: if the
transformation were a simple vector addition in the same direction,
the three points would be collinear. They are not.

---

## Degree Arc Geometry

### The Circumscribed Circle

Three points in R^H define a unique 2D affine plane and a unique
circumscribed circle. For each adj_degree triple (pos, comp, sup),
this circle has consistent parameters across all 24 tested adjectives:

| Parameter | Mean | Std | Range |
|---|---|---|---|
| R (radius) | 0.342 | 0.019 | 0.304–0.379 |
| Ω_total (arc angle) | 229.6° | 7.8° | 217.8–248.6° |
| Ω_pc (pos→comp arc) | 109.5° | 7.4° | 97.4–131.2° |
| Ω_cs (comp→sup arc) | 120.1° | 6.5° | 110.4–138.4° |
| t_comp (arc fraction) | 0.477 | 0.025 | 0.440–0.540 |
| d_origin/R | 0.128 | — | — |

### φ-Relationships

```
Ω_pc ≈ π/φ = 111.25°         measured: 109.5°  diff = 1.75°
Ω_total ≈ 2π/φ = 222.49°     measured: 229.6°  diff = 7.1° (within 1σ)
```

The partial arc from pos to comp subtends π/φ on the circumscribed
circle. The total arc subtends approximately 2π/φ. These are
φ-relationships at the angle level, not at the magnitude level.

The comparative position on the arc: t_comp ≈ 0.477 ≈ 1/2.
The comparative is approximately the **arc midpoint**, not the golden
section (1/φ ≈ 0.618).

### Circle Center and Origin

d_origin/R = 0.128: the circumscribed circle center is 12.8% of the
radius away from the zero-vector projection into the 2D plane. The
origin approximately coincides with the circle center in the 2D subspace.

This means: in the 2D plane of each triple, pos, comp, and sup are all
approximately equidistant from the embedding-space origin. The three
morphological forms lie on a circle centered at the origin (in their
local 2D subspace).

### Arc Orientation

All 24 tested adjectives produce CCW arcs (before fixing SVD sign).
After SVD-corrected bases, the orientation depends on the sign convention
of the SVD basis vectors, but the arc shape is invariant.

---

## Degree Plane Analysis

### Do All Adjectives Share a Common 2D Plane?

No. The SVD-corrected principal angles between plane pairs:

```
Mean θ₁ = 62.9°  (would be 0° if planes were identical)
Mean θ₂ = 69.8°
cos(θ₁) mean = 0.455  std = 0.071
```

Each adjective has its own **private 2D degree plane**. The planes
are approximately 63° apart on average — close to orthogonal.

Most aligned plane pairs share semantic structure:
- high ↔ low (cos=0.719) — opposites, share scale dimension
- small ↔ short (cos=0.645) — related size concepts

Least aligned pairs:
- old ↔ hot (cos=0.307), old ↔ safe (cos=0.316) — unrelated concepts

### Shared Degree Plane

SVD of all 46 difference vectors [comp-pos, sup-pos] stacked gives a
best-fit shared degree plane. Singular value spectrum:

```
k=0: S=2.200, var=30.1%
k=1: S=1.329, var=11.0%
k=2: S=0.846, var=4.4%
---
S₀/S₁ = 2.200/1.329 = 1.655 ≈ φ = 1.618  (within 2.3%)
```

The φ-ratio appears at the level of the variance structure of the
degree paradigm. The two principal directions of the degree
transformation (superlative direction and comparative direction) have
a φ-ratio of variance.

Per-word fit quality: ALL POOR (27–58%, mean=41.4%). No single 2D
plane adequately describes all 24 triples.

### Shared Plane Axes = Degree Discriminators

The two axes of the shared degree plane perfectly discriminate degree:

```
e1 (axis 1):  + end = ALL SUPERLATIVES  — end = base adjectives
e2 (axis 2):  + end = ALL COMPARATIVES  — end = SUPERLATIVES
```

Degree word coordinates in (e1, e2):

| Word | e1 | e2 | Form |
|---|---|---|---|
| big | −0.126 | +0.005 | BASE |
| bigger | +0.176 | +0.250 | COMP |
| biggest | +0.211 | −0.178 | SUP |
| fast | −0.122 | −0.026 | BASE |
| faster | +0.152 | +0.201 | COMP |
| fastest | +0.266 | −0.200 | SUP |
| high | −0.178 | −0.025 | BASE |
| higher | +0.119 | +0.223 | COMP |
| highest | +0.239 | −0.208 | SUP |

The three degree forms form a **consistent triangle** in the shared
2D projection across all 24 adjectives:

```
          e2
          ↑
          |    COMP (+e1, +e2)
          |   /
  BASE ───|──── ────────── SUP (+e1, −e2)
(-e1, ~0) |
```

---

## The Bundle of Private Arcs

The complete geometric picture of adjective degree in W_E:

**Each adjective (pos, comp, sup) traces a consistent circular arc in
its own private 2D subspace of R^1536.**

Properties of each arc:
- **Radius**: R ≈ 0.342 (universal, std=0.019)
- **Arc angle**: Ω ≈ 229.6° ≈ 2π/φ (universal, std=7.8°)
- **Comparative position**: t ≈ 0.477 ≈ arc midpoint (universal, std=0.025)
- **Circle center**: approximately at the origin projection into the 2D plane
- **Partial arc pos→comp**: Ω_pc ≈ 109.5° ≈ π/φ = 111.25°

Properties of the bundle:
- Each arc lives in a different 2D subspace (principal angle θ₁ ≈ 63° between pairs)
- The shared plane's variance ratio S₀/S₁ ≈ φ
- The shared plane axes discriminate degree (e1=superlative, e2=comparative)
- Semantically related adjectives have more aligned planes

---

## Implications for TruthSpace

### 1. The Transformation IS a Rotation

The path pos→comp→sup is a rotation of ~109° (then ~120°) in a private
2D subspace. This is the actual geometric operation, not a translation.
The "mean direction" retrieval works as an approximation because:
- All 24 arcs have the same angle (~109° or ~120°)
- The arcs are in different planes, so the mean direction averages across planes
- For held-out words, the mean direction points in the average rotation direction

The correct computation is: **rotate emb(pos) by π/φ in the private
degree plane to get emb(comp); rotate by another ~120° to get emb(sup).**
But we do not know the private degree plane for a novel word without
seeing its comparative or superlative form.

### 2. The φ-Ratio Appears at Multiple Scales

- Single step angle: Ω_pc ≈ π/φ ≈ 111.25°
- Total arc angle: Ω ≈ 2π/φ ≈ 222.5°
- SVD variance ratio: S₀/S₁ ≈ φ
- Single-step sphere angle (geometric audit): θ_char ≈ 90/φ ≈ 55.6°

The φ-ratio is present at the arc angle, the sphere angle, and the
variance structure levels. This is consistent with a system where φ
emerges naturally from the relationship between the two morphological
steps (pos→comp and comp→sup being related by φ in some measure).

### 3. Private Planes ≠ Failure of Universality

The fact that each adjective has its own private plane does NOT contradict
the universality of the arc shape. The universal law is:

> For any adjective A with forms (A_pos, A_comp, A_sup), the three
> embeddings trace an arc of radius ≈0.342 and angle ≈229.6° in their
> shared 2D subspace.

The law is universal. The location (which plane in R^1536) is private
to each adjective, determined by its semantic content.

### 4. What Remains Unknown

- Why is the arc angle ≈ 2π/φ? Is there a derivation from the training
  objective (next-token prediction) that produces this?
- Why is the circle center near the origin? Is this a general property
  of how W_E encodes semantic concepts?
- Do other paradigms (gender, past_tense, plural) also have consistent
  arc geometries? If yes, what are their R and Ω?
- Is there a way to discover the private degree plane for a novel adjective
  WITHOUT knowing its comparative/superlative forms?

---

---

## Multi-Paradigm Extension

Extending the arc framework to all paradigms by using the embedding-space
zero vector O as the third point for pair-based paradigms.

### Universal Radius

All paradigms produce HIGH-consistency circumscribed circles:

| Paradigm | n | R | R_std | Ω_ab | Consistency |
|---|---|---|---|---|---|
| adj_pos2comp | 24 | 0.341 | 0.013 | 110.52° | HIGH |
| adj_comp2sup | 20 | 0.354 | 0.012 | 114.43° | HIGH |
| gender | 15 | 0.334 | 0.018 | 114.71° | HIGH |
| plural | 18 | 0.326 | 0.018 | 95.73° | HIGH |
| past_tense | 18 | 0.324 | 0.014 | 95.47° | HIGH |
| capital | 12 | 0.319 | 0.010 | 126.94° | HIGH |
| antonym_size | 9 | 0.380 | 0.026 | 152.67° | HIGH |

All R values lie in [0.319, 0.380]. The radius is approximately universal
across all morphological paradigms in W_E.

### φ-Quantized Arc Angles

The characteristic arc angle Ω_ab maps to specific φ-related values:

```
adj_degree (single step):  Ω ≈ π/φ   = 111.25°   [within 0.73°]
gender transformation:     Ω ≈ π/φ   = 111.25°   [within 3.46°]  ← SAME CLASS
plural:                    Ω ≈ π/2   =  90.00°   [within 5.73°]
past_tense:                Ω ≈ π/2   =  90.00°   [within 5.47°]  ← SAME CLASS
capital:                   Ω ≈ 2π/3  = 120.00°   [within 6.94°]
antonym_size:              Ω ≈ 152.67° [no clean φ-match]         ← IRREGULAR
```

The arc angle partitions paradigms into geometric classes:
- **π/2 class** (90°): inflectional suffixation (plural -s, past -ed)
- **π/φ class** (111.25°): single-step morphology, gender
- **2π/3 class** (120°): encyclopedic association (country→capital)
- **5π/6 class** (150°): semantic opposition (antonym_size)

The arc angle is a **semantic distance measure**: larger angle = more
semanticchange. Inflection < derivation < encyclopedic < contrast.

### The Zero Vector Lies On Each Morphological Circle

Verification: (O, pos, comp) gives the same circumscribed circle as
(pos, comp, sup) for 8/10 tested adjectives. The four points
{O, pos, comp, sup} are approximately co-circular.

Consequence: the embedding-space zero vector is not arbitrary — it is a
geometrically distinguished point that lies on every word's morphological
arc. This may reflect a normalization property of the training objective
(softmax requires the pre-softmax activations to be centered).

### Verb Triples (base, past, -ing) Match adj_degree

| Metric | adj_degree | verb triples |
|---|---|---|
| R | 0.342 | 0.281 |
| Ω_total | 229.6° | 233.3° |
| t (middle form) | 0.477 | 0.480 |
| φ-match | 2π/φ | 2π/φ |

Both adj_degree and verb paradigms trace arcs of total angle ≈ 2π/φ with
the middle form at approximately the arc midpoint. The only difference is
the radius: verbs have smaller R (0.281 vs 0.342). This is consistent
with the norm audit finding that past_tense forms have smaller L2 norms
(ratio 0.931) than base adjectives.

### The Geometric Law

**In W_E, every morphological transformation maps source to target via
a circular arc of consistent radius R ≈ 0.33, where the arc angle Ω
identifies the paradigm type and is quantized to {π/2, π/φ, 2π/3, 5π/6}.**

Formally: for each morphological pair (A, B), the circumscribed circle
of the triangle (O, emb(A), emb(B)) has:
- R ≈ 0.33 (universal, paradigm-independent)
- Ω = arc(emb(A)→emb(B)) ∈ {π/2, π/φ, 2π/3, 5π/6} (paradigm-specific)

The complete morphological arc for three-form paradigms (pos/comp/sup
or base/past/-ing) subtends Ω_total ≈ 2π/φ with the middle form at
t ≈ 0.5 (arc midpoint).

---

## Corrected Oracle and Arc Rotation Verification

### Verification: Arc Rotation Achieves 100% When Correctly Applied

The arc rotation with TRUE circumscribed circle center and CORRECT sign:

| Method | Accuracy |
|---|---|
| Exact arc rotation (actual arc_pc, true C) | **23/23 = 100.0%** |
| Canonical π/φ rotation (correct sign, true C) | **23/23 = 100.0%** |
| mean_dir (baseline) | 22/23 = 95.7% |

Corollary: the arc model predicts chord lengths 2R·sin(Ω/2) to < 0.5%
error across all paradigms. The arc IS the exact geometric description.

### Complete Paradigm Chord Table

| Paradigm | R | Ω | chord_arc | chord_meas | φ-class |
|---|---|---|---|---|---|
| adj_pos2comp | 0.341 | 110.52° | 0.5609 | 0.5589 | π/φ |
| adj_comp2sup | 0.354 | 114.43° | 0.5945 | 0.5938 | π/φ |
| gender | 0.334 | 114.71° | 0.5625 | 0.5577 | π/φ |
| plural | 0.326 | 95.73° | 0.4830 | 0.4828 | π/2 |
| past_tense | 0.324 | 95.47° | 0.4794 | 0.4778 | π/2 |
| capital | 0.319 | 126.94° | 0.5709 | 0.5706 | 2π/3 |
| antonym_size | 0.380 | 152.67° | 0.7385 | 0.7335 | 5π/6 |

All |diff| < 0.005. The arc model fully characterizes morphological distances.

### The Rotation Sign Problem

The arc rotation direction (CW vs CCW) is word-specific:
- 7/23 adj_degree arcs go CCW, 16/23 go CW
- Best predictor: `sign(e2_proj(pos))` → 73.9% sign accuracy
- But: even 73.9% sign accuracy + local plane estimate → only 43.5% retrieval
  (mean_dir achieves 95.7%)

The sign depends on the **private plane orientation** relative to shared
degree axes — a word-specific property not recoverable from the base form alone.

### Why mean_dir Works

mean_dir (add mean of comp−pos vectors) achieves 95.7% because:
1. All adj_degree arcs have the same chord length ≈ 0.56
2. All chord directions approximately parallel in shared degree space
3. Mean chord ≈ any individual chord within 5% Euclidean error

mean_dir is the **average chord of the universal arc** — a practical
proxy that requires no knowledge of private plane or rotation direction.

### Two Levels of Geometric Description

```
DESCRIPTION level (exact, not computable from base alone):
  - Private 2D plane per word (e1, e2)
  - Circle center C in that plane (d_origin ≈ 0.04)
  - Rotation angle Ω ≈ π/φ with word-specific sign
  - Accuracy: 100% when all parameters known

COMPUTATION level (approximate, universally applicable):
  - Add mean chord vector in shared degree plane
  - Chord length = 2R·sin(Ω/2) ≈ 0.56, direction ≈ e1+e2 direction
  - Accuracy: 95.7% (one word fails due to tokenization)
```

---

## Summary and Open Questions

### What Is Established

1. **The arc is real**: every morphological paradigm encodes transformations
   as circular arcs with R ≈ 0.33 and φ-quantized arc angles.

2. **The paradigm table is complete**: four arc angle classes
   {π/2, π/φ, 2π/3, 5π/6} correspond to {inflectional, derivational,
   encyclopedic, contrastive} transformation types.

3. **The arc is exact**: rotating by Ω around the true circle center
   in the private 2D plane achieves 100% retrieval.

4. **Chord lengths are predicted to <0.5%** by the arc model.

5. **mean_dir = mean chord**: the practical method works because all
   chords are approximately parallel (same arc, same direction).

---

## Geometric Unification (Corrective)

### R and Ω Are Not Independent Properties

Subsequent analysis revealed that both R and Ω are DERIVED from the
pair's basic geometric relationship, not independent geometric facts.

**R** (circumscribed circle radius for triangle O, A, B):
By the law of sines: R = d_AB / (2·sin(θ))  where d_AB = ||A-B||, θ = acos(cos(A,B))

Empirical verification: R_theory matches R_measured to < 0.002 for ALL paradigms.
R is fully determined by (||A||, ||B||, cos(A,B)) — not an independent property.

**Ω** (arc angle for the O,A,B circumscribed circle):
By the inscribed angle theorem (O is a vertex of the triangle and thus
**on** the circumscribed circle): Ω_OAB = 2·acos(cos(A,B))

Verification (Ω_OAB = 2·acos(cos) vs Ω_PCS from corrected oracle):

| Paradigm | cos(A,B) | Ω_OAB | Ω_PCS | |diff| |
|---|---|---|---|---|
| adj_pos2comp | 0.5676 | 110.83° | 110.52° | 0.31° |
| gender | 0.5281 | 116.25° | 114.71° | 1.54° |
| plural | 0.6695 | 95.94° | 95.73° | 0.21° |
| past_tense | 0.6729 | 95.42° | 95.47° | 0.05° |
| capital | 0.4462 | 127.00° | 126.94° | 0.06° |
| antonym_size | 0.2338 | 152.96° | 152.67° | 0.29° |

All |diff| < 1.6°. The OAB and PCS circles are approximately identical,
confirming: **O, pos, comp, sup are approximately co-circular** (<1.6°).

### The True Geometric Discovery

The entire arc geometry reduces to ONE fundamental observation:
**The cosine similarity between morphological forms is paradigm-specific
and consistent.** R and Ω are both derived from this single fact.

```
Paradigm         mean_cos   Ω ≈ 2·acos(cos)   φ-connection
adj_degree:      0.567      111.0°            cos = cos(π/(2φ)) within 0.1%
gender:          0.528      116.2°            cos ≈ cos(π/(2φ)) within 1.5%
plural:          0.670       95.9°            no clean φ-match
past_tense:      0.673       95.4°            no clean φ-match
capital:         0.446      127.0°            no clean φ-match
antonym_size:    0.234      153.0°            no clean φ-match
```

Only adj_degree has a firmly φ-related cosine value:
- cos(π/(2φ)) = cos(55.625°) = 0.5671
- Measured adj_pos2comp, 24 words: 0.5676 (diff = 0.003)
- Extended English, 16 more pairs: mean = 0.598 (diff = 0.034, weaker)
- Chinese adj_degree (5 pairs): 0.329 — NOT φ-related

Gender (cos=0.528) best matches cos(π/3)=0.500. Plural/past_tense
(cos≈0.67) best match cos(π/4)=0.707. Capital (cos=0.446) best matches
cos(π/3). **These are standard fractions of π, not φ-related values.**
The φ-quantization applies specifically to English adj_degree.

### What Remains Genuinely Independent

Despite R and Ω being derived quantities:

1. **{O, pos, comp, sup} co-circularity**: <1.6° deviation across all paradigms.
   This is NOT a tautology — the four points satisfying a common circle
   equation is a constraint. It links the embedding origin to the
   morphological arc.

2. **The private 2D plane (pos, comp, sup)**: word-specific and required
   for 100% oracle accuracy. This plane is NOT just the span of any two
   of the three forms — it requires the full triple.

3. **cos(pos,comp) = cos(π/(2φ)) for adj_degree**: a specific φ-related
   value with no obvious architectural explanation.

4. **The corrected oracle achieves 100%**: rotating by arc_pc around the
   true circle center in the private plane perfectly reconstructs comp.
   This geometric operation is exact, regardless of whether R and Ω are
   derived quantities.

### What Remains Open

1. **Is cos(pos,comp) ≈ 0.57 an exact φ-value for adj_degree?**
   The 24-word sample gives 0.5676 (diff=0.003 from cos(π/(2φ)));
   16 additional words give 0.598. The claim requires a larger,
   unbiased sample across all English adj_degree pairs (thousands of words).
   Current evidence: suggestive but not conclusive.

2. **Sign from base form**: 73.9% sign accuracy from e2_proj, insufficient
   for reliable retrieval (43.5%). Is there a clean geometric predictor?

3. **Private plane from base form**: can we predict the PCS plane from the
   word's semantic neighbourhood without seeing its inflected forms?

4. **The antonym case**: cos ≈ 0.234 (most separated). Is this an
   approximate 5π/6 arc, or is the rough φ-match coincidental?

---

## Files

- `expedition_geometric_audit.py` — norm audit, linear map, path curvature, φ-audit
- `expedition_degree_arc.py` — circumscribed circle, arc parameters, plane comparison
- `expedition_degree_plane.py` — SVD-corrected plane comparison, shared plane axes
- `expedition_multi_paradigm_arc.py` — cross-paradigm arc, verb triples, chord analysis
- `expedition_arc_rotation.py` — LOO arc rotation vs mean_dir (first oracle attempt)
- `expedition_corrected_oracle.py` — corrected oracle (true center+sign), chord table
- `expedition_sign_predict.py` — sign prediction from base form, fundamental limit
- `expedition_universal_R.py` — R is determined by formula; Ω by inscribed angle theorem
- `expedition_phi_cosine.py` — φ-cosine cross-language and extended English test
- `384_subspace_structure.md` — prior DC on paradigm subspace dimensionality
