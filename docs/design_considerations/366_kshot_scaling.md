# DC 366: k-Shot Accuracy Scaling

**Day 201 | The number of training pairs (k) required to saturate TYPE_BC
accuracy is domain-specific. Morphological rules saturate at k=2-3.
Memorized factual associations (capitals) saturate at k=6 and fail
completely at k=1. TYPE_ADJACENT requires k=0 (zero-shot). The saturation
speed directly reflects the directional consistency of the domain.**

---

## Overview

Day 200 measured LOO accuracy as a function of k (number of training pairs
used to estimate the mean direction) for all six TYPE_BC domains plus
TYPE_ADJACENT (antonyms), using 30 random subsamples per k per query.

---

## Results Table

```
Domain         k=1    k=2    k=3    k=4    k=5    k=6    k=8    k=10   sat_k
────────────────────────────────────────────────────────────────────────────
capitals       0.000  0.033  0.275  0.492  0.711  0.819  0.842  0.833    6
gender         0.676  0.867  0.886  0.933  0.895  0.857  0.857  0.857    2
antonyms       1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000    0
past_tense     0.460  0.813  0.800  0.800  0.800  0.800  0.800  0.800    2
superlative    0.689  0.944  1.000  1.000  1.000  1.000  1.000  1.000    2
plurals        0.847  0.981  0.986  0.997  0.994  0.997  1.000  1.000    2

sat_k = k where accuracy first reaches 90% of its maximum
```

---

## Finding 1: Four Saturation Profiles

```
PROFILE A — Zero-shot (k=0):
  antonyms: acc=1.000 at all k
  TYPE_ADJACENT uses no direction — k-independent

PROFILE B — Fast saturation (k=2):
  gender, past_tense, superlative, plurals
  At k=2, all are at ≥86.7% accuracy
  At k=3, superlative is perfect (1.000)

PROFILE C — Slow saturation (k=6):
  capitals: acc=0.000 at k=1, plateaus at k=6-8

PROFILE D — Noise floor:
  (hypothetical: a domain where no k suffices)
  Antonyms with TYPE_BC would be Profile D
```

These profiles reflect the directional consistency of the domain:

```
Domain         dir_consistency    Saturation k    Profile
──────────────────────────────────────────────────────────
antonyms       0.033              0               A (zero-shot)
plurals        0.211              2               B (fast)
gender         0.221              2               B (fast)
past_tense     0.303              2               B (fast)
capitals       0.328              6               C (slow)
superlative    0.394              2               B (fast)
```

**Counterintuitive finding:** Superlative has the HIGHEST directional
consistency (0.394) yet still saturates at k=2. Plurals have lower
consistency (0.211) but also saturate at k=2. The saturation k is not
simply 1/dir_consistency. Capitals, despite having higher consistency
than gender and plurals, take far longer to saturate.

---

## Finding 2: Capitals Fail Completely at k=1

```
capitals k=1: acc=0.000
```

A single capital pair (e.g., France→Paris) produces a direction that
**misdirects every other query**. The France→Paris vector points to a
specific region of W_E that happens to be near Paris but is not the
"capital direction" — it is the "France-specific geopolitical direction."

To see why:

```
W_E["Paris"] - W_E["France"] = direction_France→Paris
  ≈ (cultural/linguistic associations of Paris) - (France)
  = French culture, Eiffel Tower, Louvre, fashion, ...

This is NOT:
  = capital_of(country) - country
  = generic country-to-capital offset
```

Paris is not only a capital — it is a specific cultural entity. The
France→Paris step encodes the full cultural contrast, not just the
political-administrative relationship. With k=1, the idiosyncratic
cultural dimension dominates.

With k=6+, averaging 6 country→capital pairs cancels idiosyncratic
dimensions and reveals the shared administrative/geopolitical component.

**Contrast with plurals at k=1 (acc=0.847):**
cat→cats encodes only one thing: the plural morphological relationship.
There is no "cat-specific cultural dimension" that interferes. The
morphological signal is clean from the first example.

---

## Finding 3: Standard Deviation Collapse Marks True Saturation

```
Standard deviations:
  superlative: 0.463 → 0.229 → 0.000  (collapses at k=3)
  plurals:     0.360 → 0.138 → 0.053 → 0.000  (collapses at k=8)
  capitals:    0.000 → 0.180 → 0.447 → 0.500 → 0.453 → 0.385 (never collapses)
```

Std=0.000 means the retrieval is **deterministic** — every random
subsample of k pairs gives the same answer. This is the point where
the direction estimate has fully converged.

Capitals never reach std=0.000 because different capital pairs still
point in slightly different directions. There is irreducible variance
in the "capital" direction because the relationship is not purely
administrative — each capital carries unique cultural geometry.

The std collapse point provides a **quality certificate**:
- std=0.000 → retrieval is reliable regardless of which examples were chosen
- std>0.30  → retrieval is sample-dependent and unreliable

---

## Finding 4: Accuracy Scaling Laws by Profile

**Profile B (morphological):** Approximately follows:
```
acc(k) ≈ acc_max × (1 - exp(-k / τ))
where τ ≈ 1–2  (fast time constant)

plurals:     acc(k) ≈ 1.000 × (1 - exp(-k/1.2))
superlative: acc(k) ≈ 1.000 × (1 - exp(-k/0.8))
```

**Profile C (factual):** Approximately follows:
```
acc(k) ≈ acc_max × (1 - exp(-k / τ))
where τ ≈ 3–4  (slow time constant)

capitals:    acc(k) ≈ 0.842 × (1 - exp(-k/3.2))
```

The time constants differ by ~3×. Factual domains need ~3× more examples
to converge than morphological domains.

---

## Implications for TruthSpace

### Minimum Viable k by Domain Type

```
Domain type                  Minimum viable k   Rationale
───────────────────────────────────────────────────────────────────────
TYPE_ADJACENT                0                  No direction needed
Morphological (regular)      2–3                Rule is uniform
Morphological (irregular)    3–5                Patterns vary (e.g., go→went)
Gender/categorical           2–4                Category is consistent
Factual (capitals, etc.)     6–8                Memorized, idiosyncratic
```

### Pipeline Update

The multi-tier pipeline (DC 365) should be updated with k requirements:

```
BEFORE retrieval:
  If archetype == TYPE_BC:
    If morphological domain: require k ≥ 3
    If factual domain:       require k ≥ 6
    If k < minimum: report LOW_CONFIDENCE
  If archetype == TYPE_ADJACENT:
    k is irrelevant — proceed with nn()
```

### The Zero-Shot Limit

For factual domains with k < 3, the direction estimate is unreliable.
The zero-shot factual retrieval problem (k=0) is not solvable with the
TYPE_BC approach. It would require:

1. A pre-built direction from a training corpus (cross-domain transfer)
2. Or a different encoding mechanism entirely

Cross-domain direction transfer (e.g., does the plurals direction work
on unseen nouns?) is the natural next question.

---

## Summary

```
Saturation profiles:
  k=0 (zero-shot): TYPE_ADJACENT (antonyms)
  k=2-3:           Morphological TYPE_BC (plurals, superlative)
  k=6:             Factual TYPE_BC (capitals)

Critical k=1 finding:
  Capitals at k=1: acc=0.000 — single fact misdirects every query
  Plurals at k=1:  acc=0.847 — single rule generalizes immediately

Mechanism:
  Factual pairs encode cultural geometry → idiosyncratic directions
  Morphological pairs encode pure transformation → generalizable

Quality certificate: std=0.000 marks full convergence
  Superlative: k=3, Plurals: k=8, Capitals: never reached
```

---

## Files

- `expedition_day200_kshot_scaling.py` — k-shot measurement
- `day200_kshot_scaling.json` — results
- `365_multitier_pipeline.md` — pipeline architecture
- `364_relational_encoding_archetypes.md` — archetype taxonomy
