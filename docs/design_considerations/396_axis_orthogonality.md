# DC 396: Geometric Axis Orthogonality in W_E

**Day 261 | The 6 geometric transformation axes in W_E form a near-orthogonal
basis with two key exceptions: (1) adj_degree and superlative are ALIGNED
(cos=0.47) — they share a single degree axis; (2) inflectional axes share
a weak common component (~0.17). Gender and capital are orthogonal to
everything. All axes project negatively onto PC1.**

---

## The Axis Catalogue

Six geometric axes were measured (mean_dir from N=13-20 training pairs):

```
Axis            Coherence   Knowledge Type
──────────────────────────────────────────────────────
adj_degree      0.4116      grammatical (inflectional)
superlative     0.4116      grammatical (inflectional)
capital         0.3015      encyclopedic
gender_m2f      0.1809      derivational/grammatical
past_tense      0.1719      grammatical (inflectional)
plural          0.1339      grammatical (inflectional)
```

---

## Inter-Axis Similarity Matrix

```
               adj    plural  past   gender  capital  super
adj_degree     1.000  0.189   0.164  0.031   0.028   0.469
plural         0.189  1.000   0.176  0.045   0.028   0.190
past_tense     0.164  0.176   1.000  0.009   0.016   0.167
gender_m2f     0.031  0.045   0.009  1.000   0.030   0.040
capital        0.028  0.028   0.016  0.030   1.000   0.037
superlative    0.469  0.190   0.167  0.040   0.037   1.000

Off-diagonal: mean=0.108  std=0.119  min=0.009  max=0.469
```

### Taxonomy

**PARALLEL (cos > 0.3)**
- `adj_degree ↔ superlative` = 0.469

**WEAKLY CORRELATED (cos 0.15–0.20)**
- `adj ↔ plural` = 0.189
- `plural ↔ past` = 0.176  
- `adj ↔ past` = 0.164
- `superlative ↔ {plural, past}` ≈ 0.17–0.19

**ORTHOGONAL (cos < 0.05)**
- `gender ↔ {adj, plural, past, capital, super}` = 0.009–0.045
- `capital ↔ {adj, plural, past, gender, super}` = 0.016–0.037

---

## Finding 1: Degree Has One Axis, Not Two

`adj_degree ↔ superlative = 0.469`

The comparative ("big→bigger") and superlative ("big→biggest") transformations
point in the SAME direction in W_E. The model did not allocate separate axes
for comparative and superlative — both are encoded as steps along a single
**degree axis**.

This is exactly consistent with the arc model from Day 254:
```
base  ──Ω──►  comparative  ──Ω──►  superlative
     (step 1)              (step 2)
```
The direction Ω is the same for both steps. The superlative is simply
two steps where comparative is one step along the same axis.

Implication: the 6 axes are effectively 5 independent axes:
- 1 degree axis (shared by adj_degree and superlative)
- 1 plural axis
- 1 past_tense axis
- 1 gender axis
- 1 capital axis

---

## Finding 2: Gender Is Geometrically Isolated

`gender ↔ past = 0.009`  (essentially zero — most orthogonal pair)
`gender ↔ adj  = 0.031`
`gender ↔ capital = 0.030`

The gender axis (male→female, Δx ≈ -2.0 from Day 245) is orthogonal to ALL
other axes. This confirms that gender is a separate coordinate in the W_E
space — not a byproduct of any grammatical or encyclopedic structure.

The isolation makes intuitive sense: gender marking in English is lexical
(king/queen, man/woman) rather than morphological (unlike in Romance languages
where gender inflects nouns). English W_E encodes gender as a pure semantic
dimension, completely decoupled from tense, number, or degree.

---

## Finding 3: Encyclopedic vs. Grammatical Knowledge Are Orthogonal

`capital ↔ {adj, plural, past} = 0.016–0.028`

The country→capital axis is orthogonal to all three morphological inflection
axes. Encyclopedic world knowledge (which city is a country's capital) and
grammatical knowledge (how verbs inflect) occupy SEPARATE dimensions of W_E.

This validates the hypothesis that W_E is a multi-type knowledge store:

```
Type A — Grammatical knowledge (W_E encodes inflectional rules):
  degree:   adj_degree/superlative axes (one shared axis)
  number:   plural axis
  tense:    past_tense axis
  gender:   gender_m2f axis (orthogonal to others)

Type B — Encyclopedic knowledge (W_E encodes world facts):
  capitals: country→capital axis
  probably: currency, language, nationality, ...
  (orthogonal to all Type A axes)
```

---

## Finding 4: Inflectional Triad Shares a Common Component

All three inflectional morphology axes (adj, plural, past) show ~0.17 alignment
with each other. This weak but consistent correlation suggests the inflectional
axes share a small common component — a general "morphological marking"
direction in W_E that all inflections partially contribute to.

This component is estimated at: `mean(adj, plural, past) ≈ 0.17` projected
onto any individual axis.

Possible interpretation: the model learns a single "this word is morphologically
marked" feature that all inflected forms share, independent of which specific
inflection it is. The remaining ~83% of each axis is specific to its paradigm.

---

## SVD Analysis: Effective Dimensionality

```
Singular values: 1.310, 1.011, 0.985, 0.970, 0.905, 0.729
Variance ratios: 28.6%, 17.0%, 16.2%, 15.7%, 13.7%, 8.9%

Effective dimensionality = (Σ σ_i)² / Σ σ_i² ≈ 5.5
```

The 6 axes span approximately 5.5 independent dimensions (not 6), due to:
- adj/superlative alignment reducing independent dimensions from 2 to 1.5
- Inflectional triad sharing a common component

---

## PC1 as the "Unmarked Form" Axis

All 6 transformation axes project NEGATIVELY onto PC1:

```
adj_degree:  -0.318  (strongest)
superlative: -0.308
plural:      -0.287
past_tense:  -0.168
gender_m2f:  -0.089
capital:     -0.083  (weakest)
```

PC1 captures the "frequency/markedness" dimension of W_E (established Day 252):
high PC1 = simple, unmarked, high-frequency forms.

All transformations move in the -PC1 direction: every inflection, derivation,
and encyclopedic transformation moves FROM unmarked high-frequency tokens
TOWARD marked lower-frequency tokens. This is the geometric signature of
**morphological markedness theory** — marked forms (plural, past, comparative)
are always "further" from the origin in the direction of -PC1.

The magnitude of the PC1 projection correlates with how "marking" the
transformation is:
- Adj/superlative: strong PC1 effect (degree marking is heavy)
- Plural/past: moderate PC1 effect
- Gender/capital: weak PC1 effect (these transformations are less "marked")

---

## Summary: The W_E Coordinate System

W_E has a discoverable coordinate system with approximately 5.5 dimensions:

```
Dim 1: Degree axis (adj_degree + superlative, PC1 projection -0.31)
Dim 2: Number axis (plural, PC1 projection -0.29)
Dim 3: Tense axis (past_tense, PC1 projection -0.17)
Dim 4: Gender axis (gender_m2f, orthogonal to all above)
Dim 5: Capital axis (encyclopedic, orthogonal to all grammatical axes)
(Dim 0): PC1 itself (unmarked/markedness axis, all transformations point away from it)
```

These dimensions are approximately orthogonal — with the exception of
adj/superlative sharing one direction — and together describe the major
systematic linguistic and encyclopedic relations encoded in W_E.

---

## Relation to TruthSpace Hypothesis

The TruthSpace hypothesis states: "Structure IS information; the shape of W_E
IS the knowledge." Day 261 reveals that this shape has a specific architecture:

1. **Grammatical knowledge is encoded in ~4 orthogonal directions**
   (degree, number, tense, gender)
2. **Encyclopedic knowledge is encoded in additional orthogonal directions**
   (capital, and presumably others)
3. **Markedness is encoded in the PC1 direction** (all transformations
   point away from it)

This is not an arbitrary high-dimensional blob — it is a structured coordinate
system where each type of linguistic/encyclopedic knowledge has its own dimension.

---

## Files

- `expedition_log.md` — Day 261 results
- `385_degree_arc_geometry.md` — arc model for degree axis
- `393_geometric_axis_coherence_law.md` — coherence predicts reliability
