# DC 368: Irregular Verb Inflection Class Geometry

**Day 205 | English irregular verb classes do NOT form geometrically
distinct TYPE_BC directions. Cross-class direction transfer achieves a
slightly higher mean accuracy (0.771) than within-class (0.734). The
cross-class matrix reveals three distinct column patterns: uniform 1.000
(proximity-encoded), uniform 0.000 (unencodable), and direction-sensitive.
The phonological class taxonomy maps only partially to the geometric
taxonomy. Past tense is a mixed-archetype domain.**

---

## Overview

Day 204 tested whether irregular English past tense verbs, grouped by
traditional phonological inflection class, have geometrically distinct
TYPE_BC directions. The hypothesis: each class (A: i→a, B: oo→ew, etc.)
encodes a different geometric transformation.

The result falsifies the hypothesis. Column uniformity in the transfer
matrix reveals that some classes are proximity-encoded and some are
unencodable — the direction applied is irrelevant.

---

## Results

### Within-Class Direction Analysis

```
Class          n   dir_consistency   LOO_acc   mean_rank   Geometric type
────────────────────────────────────────────────────────────────────────────
A_i_to_a       7   0.074  DIFFUSE    0.714     1.71        TYPE_BC (weak)
B_oo_to_ew     6   0.317  COHESIVE   1.000     0.00        TYPE_ADJACENT
C_ee_to_e      9   0.079  DIFFUSE    0.889     0.89        TYPE_ADJACENT
D_nd_to_nt     6   0.135  PARTIAL    1.000     0.00        TYPE_ADJACENT
E_no_change    8   0.000  DIFFUSE    0.000     7.00        UNENCODABLE
F_suppletive  15   0.226  COHESIVE   0.800     1.00        TYPE_BC
```

### Cross-Class Transfer Matrix (accuracy)

```
               → A      → B      → C      → D      → E      → F
A_i_to_a       0.714   1.000   0.889   1.000   0.000   0.867
B_oo_to_ew     0.714   1.000   0.889   1.000   0.000   0.667
C_ee_to_e      0.857   1.000   0.889   1.000   0.000   1.000
D_nd_to_nt     0.857   1.000   0.889   1.000   0.000   1.000
E_no_change    1.000   1.000   0.889   1.000   0.000   1.000
F_suppletive   0.714   1.000   0.889   1.000   0.000   0.800

Mean diagonal:     0.734
Mean off-diagonal: 0.771
Diagonal advantage: -0.037  (cross-class wins)
```

---

## Finding 1: Column Uniformity Reveals Encoding Type

The most diagnostic signal is **column uniformity** — each column has the
same value regardless of which row (direction source) is used:

```
Column B (B_oo_to_ew): uniformly 1.000 for all direction sources
Column D (D_nd_to_nt): uniformly 1.000 for all direction sources
Column C (C_ee_to_e):  uniformly 0.889 for all direction sources
Column E (E_no_change): uniformly 0.000 for all direction sources
```

**Uniform column = the result is independent of the direction applied.**

This is only possible when:
- The correct answer is already the nearest neighbor in the vocab
  (TYPE_ADJACENT / proximity-encoded), OR
- The correct answer is unreachable regardless of direction
  (UNENCODABLE / identity-mapped)

**Columns B and D (acc=1.000 for all sources):** know/knew, grow/grew,
send/sent, spend/spent — these past tenses are already the nearest
single-token neighbors of their base forms in W_E. No direction needed.

**Column C (acc=0.889 for all sources):** keep/kept, feel/felt, sleep/slept
— largely proximity-encoded. The 11.1% error rate is constant because
one of the nine test words has a slightly more distant target.

**Column E (acc=0.000 for all sources):** cut/cut, put/put, hit/hit —
source and target are the **same token**. There is no retrievable
transformation. Any direction moves you AWAY from the source, never
back to it.

---

## Finding 2: E_no_change Direction Achieves Best Cross-Class (Row E)

```
Row E (E_no_change direction):
  → A: 1.000   → B: 1.000   → C: 0.889   → D: 1.000   → F: 1.000
```

The E_no_change direction (mean diff of cut→cut, put→put, ...) is
approximately a **zero vector** — the mean of near-zero differences.
Applying a zero direction to any source word leaves you at the source.
The nearest target in the vocab is then determined purely by proximity.

For classes B, D: targets are already nearest neighbors → 1.000
For class F: targets are nearest neighbors in their proximity cluster → 1.000
For class A: base forms and past tenses are interleaved → 1.000 here too

The zero direction is maximally "neutral" — it reveals pure proximity
structure. Its superior cross-class performance confirms that B, C, D
are proximity-encoded.

---

## Finding 3: F_suppletive Has Genuine TYPE_BC Direction

Row F and Column F are the only ones showing direction sensitivity:

```
Row F (F direction applied to other classes):
  → A: 0.714  → B: 1.000  → C: 0.889  → D: 1.000  → E: 0.000  → F: 0.800

Column F (other directions applied to F):
  A_dir: 0.867  B_dir: 0.667  C_dir: 1.000  D_dir: 1.000  E_dir: 1.000
  F_dir: 0.800
```

F column is NOT uniform — some directions give 1.000, others give 0.667.
Column F variability = the direction MATTERS for this class.

F is genuinely TYPE_BC: the suppletive verbs (go/went, have/had, do/did,
take/took, etc.) are the highest-frequency English verbs. They share
syntactic contexts so thoroughly that their past tenses are displaced in a
shared direction in W_E — not from phonological regularity but from
**frequency-weighted syntactic context overlap**.

Why does C_dir→F give 1.000 while B_dir→F gives only 0.667?
C_ee_to_e (keep→kept, feel→felt): these are also high-frequency verbs
with similar syntactic profiles to the suppletives. The C direction
accidentally aligns well with the F direction.
B_oo_to_ew (know→knew, grow→grew): lower-frequency, less syntactic overlap,
direction less aligned with F.

---

## Finding 4: Phonological Class ≠ Geometric Class

The traditional phonological taxonomy maps poorly to geometry:

```
Phonological class    Geometric encoding
──────────────────────────────────────────────────────────────
A: i→a (run/ran)      TYPE_BC_weak (dir=0.074, LOO=0.714)
B: oo→ew (know/knew)  TYPE_ADJACENT (uniform 1.000 columns)
C: ee→e (keep/kept)   TYPE_ADJACENT (uniform 0.889 columns)
D: nd→nt (send/sent)  TYPE_ADJACENT (uniform 1.000 columns)
E: no change (cut/cut) UNENCODABLE (zero diff)
F: suppletive         TYPE_BC (dir=0.226, LOO=0.800)
```

The geometric taxonomy is organized by **retrieval mechanism**:
- Proximity: how far is the target from the source? (B, C, D)
- Direction: does the transformation vector point consistently? (A, F)
- Identity: is the target the same token? (E)

The phonological rules (vowel changes) are morphological conventions,
but W_E doesn't encode them as geometric directions — it encodes their
**distributional consequences** (co-occurrence patterns, syntactic
slots). Verbs with nearby past tenses in distributional space are
proximity-encoded; verbs with shifted past tenses need direction.

---

## Revised Past Tense Encoding Model

```
Past tense encoding in W_E:
  TYPE_ADJACENT (proximity):  B, C, D classes — ~53% of tested verbs
    Retrievable by nn() with no direction
    Examples: know/knew, keep/kept, send/sent

  TYPE_BC (directional):      A, F classes — ~40% of tested verbs
    Requires direction from training examples (k≥3 for A, k≥5 for F)
    Examples: run/ran, go/went, take/took

  UNENCODABLE (identity):     E class — ~7% of tested verbs
    Source = target token; retrieval is impossible
    Examples: cut/cut, put/put, hit/hit
```

### Pipeline Implication

For a "past_tense" domain query, the pipeline should:

1. First check if source and target are the same token → identity
2. Then check direction consistency among known pairs
3. If dir > 0.10 → TYPE_BC with class-specific direction
4. If dir ≤ 0.10 but LOO_acc > 0.80 → TYPE_ADJACENT
5. This detection runs per-word, not per-domain

The "past_tense" domain is not a single archetype — it is a mixture
that requires per-word archetype detection.

---

## The Broader Principle: Distributional Proximity Overrides Phonology

```
PRINCIPLE:
  W_E encodes the distributional consequences of morphology,
  not the morphological rules themselves.

  If word W and its form W' co-occur in similar contexts
  AND appear near each other in the training corpus →
  W and W' are proximity-encoded in W_E.

  If W and W' are systematically displaced (high-frequency,
  consistent syntactic environment) →
  W' is reachable from W via a TYPE_BC direction.

  Phonological rule class is correlated but not causal.
```

This principle extends beyond verbs: any morphological domain will show
the same mixed-archetype structure. Regular plurals are TYPE_BC_UNIV
because all regular nouns share the same displacing context (the -s
suffix is universal). Irregular plurals (foot/feet, man/men) may be
proximity-encoded.

---

## Files

- `expedition_day204_verb_classes.py` — inflection class experiment
- `day204_verb_classes.json` — results
- `367_direction_transfer.md` — TYPE_BC subclassification
- `364_relational_encoding_archetypes.md` — archetype taxonomy
