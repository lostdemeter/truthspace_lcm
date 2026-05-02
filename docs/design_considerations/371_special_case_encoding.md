# DC 371: Special-Case Encoding — Antonym Axes and Numbers

**Day 211 | Numbers word→digit is TYPE_BC with dir_consistency=0.850 —
the strongest directional domain found, stronger than capitals (0.368).
Antonym attribute axes are geometrically orthogonal (mean cross-axis
cosine=0.033). Per-attribute axis retrieval beats proximity for 5/10
attributes (those with no proximity-accessible antonym). Ensemble axis
selection via max-projection fails (-0.083 vs baseline). The numbers
domain fix: reclassify from TYPE_ORDINAL to TYPE_BC via standard
dir_consistency check.**

---

## Overview

Day 210 investigated two failure cases from Day 208:
1. Numbers (`TYPE_ORDINAL`) — ordinal Spearman detection failed
2. Antonyms (`TYPE_ADJACENT`) — proximity gives only 0.500 accuracy

Three experiments:
- A: Characterize per-attribute antonym axes
- B: Measure word→digit direction consistency
- C: Test ensemble axis selection for antonym retrieval

---

## Experiment A: Antonym Semantic Axes

### Axis Orthogonality

```
Cross-attribute cosine matrix (showing representative subset):
              temp   size   speed  vol    age    brite  sharp  text   wealth emot
temperature  1.000  0.053  0.002  0.047  0.017 -0.003  0.003  0.031  0.007  0.060
size         0.053  1.000  0.008  0.037 -0.006  0.044  0.011 -0.014  0.008  0.009
speed        0.002  0.008  1.000 -0.026 -0.017  0.064  0.003  0.042 -0.002  0.042
volume       0.047  0.037 -0.026  1.000  0.021 -0.069  0.040 -0.009  0.060 -0.026
age          0.017 -0.006 -0.017  0.021  1.000  0.055  0.026  0.039  0.061  0.068

Mean off-diagonal cosine: 0.033
```

Ten attribute axes are nearly orthogonal in W_E. Each semantic attribute
occupies a dedicated subspace direction with effectively zero overlap.
This is geometrically remarkable: W_E encodes each antonym attribute as
an independent axis in its 1536-dimensional space.

### Per-Attribute Retrieval

```
Attribute     n   axis_acc   nn_acc   Winner
──────────────────────────────────────────────────────────────
temperature   3   0.667      0.667    tie
size          6   0.333      0.333    tie
speed         3   0.333      0.000    AXIS
volume        3   0.333      0.000    AXIS
age           3   0.333      0.000    AXIS
brightness    3   0.667      0.667    tie
sharpness     2   0.000      0.000    neither
texture       3   1.000      1.000    tie
wealth        3   0.667      0.000    AXIS
emotion       3   0.667      0.000    AXIS
```

**Split into two categories:**

- **Tie (temperature, size, brightness, texture):** The antonym is
  already the nearest neighbor in the antonym vocabulary. Adding the
  axis direction gives the same result. These are dual-archetype:
  TYPE_ADJACENT AND TYPE_BC simultaneously.

- **Axis wins (speed, volume, age, wealth, emotion):** The antonym is
  NOT the nearest neighbor without direction. Example: "fast" → "quick"
  is nearer than "slow" without axis; the speed axis points away from
  "quick" toward "slow". The axis is the only retrieval mechanism that
  works for these pairs.

- **Neither (sharpness):** Neither proximity nor axis retrieves
  "dull" from "sharp" correctly in the constrained vocabulary.
  Sharpness axis has low self-consistency within the three-word training set.

### Cross-Attribute Axis Transfer

```
All cross-attribute axis transfers: 0.000 accuracy
```

Axes do not transfer across attributes at all. Using the temperature
axis on a volume pair gives 0.000. This is a direct consequence of
axis orthogonality: flipping along the temperature axis does not
move you toward the volume antonym.

---

## Experiment B: Numbers Word→Digit — TYPE_BC, Not Ordinal

### Tokenization

Single-token coverage:

```
one→1   ✓    two→2   ✓    three→3 ✓    four→4  ✓    five→5  ✓
six→6   ✓    seven→7 ✓    eight→8 ✓    nine→9  ✓
ten→10  ✗ (10 is multi-token)
```

9/20 number pairs are fully single-token. The digit "10" tokenizes as
two tokens in Qwen ("1"+"0"), so ten→10 is not testable.

### Geometric Structure

```
Word  Digit   cosine   digit_rank
one   1       0.169    0 ✓
two   2       0.156    0 ✓
three 3       0.219    0 ✓
four  4       0.261    0 ✓
five  5       0.223    0 ✓
six   6       0.269    0 ✓
seven 7       0.228    0 ✓
eight 8       0.239    0 ✓
nine  9       0.217    0 ✓

Mean cosine(word, digit): 0.220
Mean rank among digit tokens: 0.00 (9/9 correct)
dir_consistency: 0.850
```

Three simultaneous properties:
1. `cosine(word, digit) = 0.220` — moderately low (not close neighbors overall)
2. `digit_rank = 0` for all pairs — each word's nearest digit is its correct digit
3. `dir_consistency = 0.850` — the word→digit displacement is extremely consistent

This combination means: word tokens are FAR from digit tokens in absolute
W_E distance, but within the digit subspace, each word points to exactly
its correct digit. The word→digit displacement is TYPE_BC with the
highest dir_consistency measured across all domains.

### Interpretation

The consistent word→digit direction is the **numeral-script axis** in
W_E — a single direction that encodes "written as digits" vs "written
as words". Every number word (one, two, ..., nine) is displaced from
its digit counterpart along approximately the same axis.

This parallels the cross-lingual structure found in Day 194 (ice/冰):
the English word and its CJK equivalent are cross-script neighbors, with
a consistent per-token displacement that encodes the script difference.

Here: "one"/"1", "two"/"2", etc. are **cross-representation** neighbors —
the same concept expressed in two different writing conventions (words vs
numerals), with a consistent geometric displacement.

### Pipeline Fix

```
OLD STEP 1: TYPE_ORDINAL check (Spearman ρ on displacement magnitudes)
            → FAILS for numbers (Spearman=0 because no magnitude order)

NEW STEP 1: Standard TYPE_BC check (dir_consistency > 0.15)
            → CATCHES numbers (dc=0.850, well above threshold)
            → TYPE_ORDINAL check becomes unnecessary for numbers

Result: numbers correctly classified as TYPE_BC.
Retrieval: use mean-direction + nn_digit vocabulary.
```

---

## Experiment C: Antonym Ensemble — Fails Without Attribute Label

```
Ensemble axis selection: 5/12 = 0.417
Proximity (nn):          6/12 = 0.500
Difference: -0.083 (axis HURTS overall)
```

The max-projection heuristic (select axis with largest |src · axis|)
is unreliable:

```
hot   → cold:    correct attr=temperature  ✓  (retrieved: cold)
big   → small:   correct attr=size         ✗  (retrieved: tiny, not small)
hard  → soft:    correct attr=texture      ✓  (retrieved: soft)
light → dark:    correct attr=brightness   ✗  (retrieved: dim, not dark)
rich  → poor:    correct attr=wealth       ✓  (retrieved: poor)
```

When the selected attribute matches the query's actual attribute,
retrieval succeeds. When it mismatches, the wrong axis pushes the
query to a wrong pole. The net effect is worse than pure proximity.

**Implication:** Antonym retrieval improvement requires knowing the
attribute category. Without supervision (attribute label), the axes
cannot be used reliably.

---

## Updated Archetype Taxonomy

```
Archetype       dir_consistency   nn_rank   Detection
──────────────────────────────────────────────────────────────────────────
IDENTITY        —                 —         norm(tgt-src) < 0.05
TYPE_BC_DIGIT   0.850             0         digit target check → TYPE_BC
TYPE_BC_UNIV    0.27–0.41         0         dir_consistency > 0.15
TYPE_BC_CLASS   0.25–0.38         0         dir_consistency > 0.15
DUAL_ARCHETYPE  0.30–0.32         0         TYPE_BC AND TYPE_ADJACENT
TYPE_ADJACENT   0.00–0.14         0         fallback
TYPE_ANTONYM    0.00 per attr     varied    supervised (attribute label)
UNENCODABLE     0.00              high      diff_norm ≈ 0
```

The `TYPE_ANTONYM` is a NEW archetype subclass:
- dir_consistency ≈ 0 across all antonym pairs (no global antonym direction)
- But per-attribute dir_consistency is HIGH within each attribute axis
- Requires attribute label for correct retrieval
- Superficially appears as TYPE_ADJACENT (dc=0.020) but is not

---

## Summary of Open Problems

### Solved by Day 210

- Numbers: now correctly classified as TYPE_BC (dc=0.850). Fix: remove
  ordinal Spearman check, use standard dir_consistency threshold.

### Partially Solved

- Antonyms with axis label: 5/10 attribute axes improve over proximity
  when the correct attribute is known. If a pipeline receives a domain
  name like "temperature_antonym", it can apply the axis.

### Unsolved

- Antonyms without label: no unsupervised method to identify attribute
  axis from the word pair alone. Proximity remains the best fallback.
- Sharpness axis: low self-consistency within training set; neither
  method works. May require more training examples.
- Multi-token numbers (10+): not testable in the single-token W_E framework.

---

## Files

- `expedition_day210_special_cases.py` — special case experiments
- `day210_special_cases.json` — results
- `370_domain_classifier.md` — two-stage classifier (Day 208)
- `369_archetype_detection.md` — per-pair vs per-domain analysis
