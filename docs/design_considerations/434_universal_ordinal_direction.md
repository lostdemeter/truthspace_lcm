# DC 434: The Universal Ordinal Direction in W_E

**Day 299 | All "category-name → ordinal-number" axes converge on a
SINGLE DIRECTION in W_E: cos(letter→pos, month→num)=0.665,
cos(planet→orbital, month→num)=0.766, cos(season→quarter, month→num)=
0.769, cos(card→number, month→num)=0.745. All are anti-parallel to
digit→word (cos −0.67 to −0.88). The card→number axis (pc=0.789,
9/9 100%) confirms the pattern even for face cards (Ace→1). Planet
names have a CAPITALISATION ATTRACTOR PROBLEM: 'Mercury'→'mercury',
giving 2/7 accuracy despite pc=0.609. Arithmetic increment axes
(n→n+1, n→n+2, n→n+3) share direction (cos 0.76–0.84) but have
negative pc due to non-uniform step sizes on the number line.**

---

## Evidence for the Universal Ordinal Direction

### All Name→Number Axes Align

```
Axis              pc      vs month→num  vs digit→word
digit→word        0.851   −0.731        —  (it IS digit→word)
weekday→number    0.842   +0.779        −0.712
month→number      0.803   —  (reference) −0.731
card→number       0.789   +0.745        −0.882  ***
season→quarter    0.691   +0.769        −0.720
planet→orbital    0.609   +0.766        −0.760
ordinal→cardinal  0.582   +0.107        +0.169  (different axis!)
letter→pos (UC)   0.504   +0.665        −0.675
lc-letter→pos     0.447   +0.696        −0.728
```

Every axis that maps from a **named category member** to its **ordinal
number** points in the same direction (cos ≥ 0.66 with month→num).
Every axis that maps in the reverse direction (digit→word) is
anti-parallel (cos ≤ −0.67).

The ordinal→cardinal axis (first→one, second→two) is the exception:
it maps from ordinal WORDS to cardinal WORDS, both in the word domain,
and is orthogonal to the name→number direction (cos ≈ 0.11).

### The Ordinal Direction is Not One of the Named Axes

The cosines are high but not 1.0. This means the "universal ordinal
direction" is a specific vector in W_E that each individual axis
APPROXIMATES but does not perfectly recover. Different source domains
(months vs. weekdays vs. card names) introduce slight deviations
because the sources cluster in different semantic regions.

The universal ordinal direction is approximately:
```
v_ord ≈ mean(ax_month→num, ax_weekday→num, ax_card→num, ax_season→quarter, ...)
```
with all axes normalised. Computing this mean would give the best
estimate of the true ordinal direction.

---

## Card→Number: Perfect Accuracy (pc=0.789, 9/9)

### Why Cards Work

Playing card names Two through Nine ARE number names. Their embeddings
are in the cardinal-number cluster. The axis from `Two` to `2`, `Three`
to `3`, etc. is essentially the digit→word axis in REVERSE — which
has pc=0.851.

The Ace→1 mapping is remarkable: `Ace` is not a number word, but it
maps to `1` correctly. This is because:
- `Ace` is strongly associated with "1" in card contexts
  ("Ace of Spades = 1", "ace in the hole", "ace = best = first")
- The `Ace` embedding is displaced from the number cluster by a
  consistent offset that matches the card→number axis direction

### Why Card→number Beats Month→number in Anti-Parallelism

```
cos(card→num, digit→word) = −0.882  (most anti-parallel)
cos(month→num, digit→word) = −0.731
```

Card names (Two, Three, ..., Nine) are LITERALLY in the number-word
cluster. The axis from them to their digit symbols (2, 3, ..., 9) is
almost the exact reverse of digit→word (1→one, 2→two, ...). The
−0.882 cosine reflects that card names and cardinal words are in
nearly the same position in W_E (both are number names), making the
reverse nearly perfect.

---

## Planet Names: The Capitalisation Attractor Problem

### Result
```
planet→orbital: pc=0.609, coh=0.815, 2/7 accuracy
Mercury  → 1: got mercury  (lowercase attractor)
Venus    → 2: got 5        (MISS, no attractor — Venus is a planet but no common lower)
Earth    → 3: got earth    (lowercase attractor)
Mars     → 4: got mars     (lowercase attractor)
Jupiter  → 5: got 5        (HIT — Jupiter has no common lowercase)
Saturn   → 6: got saturn   (lowercase attractor)
Neptune  → 8: got 8        (HIT — Neptune has no common lowercase)
```

### The Pattern

Only `Jupiter` and `Neptune` work. The common factor: these are not
everyday English words. `Mercury` (element), `Earth` (the ground),
`Mars` (the planet/god/candy), `Saturn` (the god/ring system), `Venus`
(the goddess/planet — wait, Venus DID fail, getting 5 not 2) all have
LOWERCASE forms that are closer embeddings than the number targets.

Actually: `Venus` failed differently — it got `5` (Jupiter's number),
not `venus`. This is because `Venus` doesn't have a strong everyday
lowercase sense, but its embedding is close to Jupiter's number (5)
rather than its own number (2). This is a **scale calibration** issue:
the scale (0.26) is calibrated for Jupiter/Neptune and overshoots for
Venus (which needs less travel to reach a number).

### Diagnosis: Three Failure Modes for Proper Nouns

1. **Capitalisation attractor**: The lowercase form of the proper noun
   is embedded close to the proper noun and intercepts the axis
   (Mercury→mercury, Earth→earth, Mars→mars, Saturn→saturn).

2. **Scale miscalibration**: Different category members need different
   scales (Venus is at scale~0.5, Jupiter at scale~0.26). The mean
   scale fails for outliers.

3. **High pc despite low accuracy**: pc=0.609 because the CHORD
   VECTORS point in consistent directions — the axis direction is
   correct. But the SCALE is wrong for most members.

This is the **proper noun labelling problem**: capitalised words have
lowercase twins that function as dominant attractors at small step
sizes, requiring careful scale selection per-word.

---

## Arithmetic Increment Axes

### All Increments Share Direction

```
cos(n→n+2, n→n+1) = 0.839
cos(n→n+3, n→n+1) = 0.758
```

The "add 2" and "add 3" operations point in nearly the SAME direction
as "add 1" in W_E. This means:
- There is ONE "increasing" direction on the number line
- Moving by different amounts in the same direction is geometrically
  consistent
- The number line is (approximately) linear in the sense that all
  increment operations navigate in the same geometric direction

### Why All Increments Have Negative pc

Despite sharing direction, all increment axes have negative pc:
```
n→n+1: pc=−0.115   n→n+2: pc=−0.076   n→n+3: pc=−0.006
```

The pc INCREASES toward zero as the increment INCREASES. This makes
sense:

For large increments (n→n+3), the chord vectors all have a consistent
"large step right" direction, minimising the rotational noise from
non-uniform step sizes. For small increments (n→n+1), the small steps
amplify the rotational noise (the tiny 8→9 step points in nearly the
same direction as the large 2→3 step, but with very different
magnitudes). As increment k→∞, the chord vectors all converge on the
"increasing" direction → pc→1.0.

The negative pc is a **finite step artifact**: the number line has
non-uniform spacing, so small consecutive steps have INCONSISTENT
directions. Large steps average out the spacing non-uniformity.

### Implication for "Arithmetic Reasoning"

A single axis cannot perform "add 1" reliably because the step sizes
are non-uniform. But there IS a consistent "increasing" direction.
A full arithmetic operation would require KNOWING WHERE ON THE LINE
you currently are (which position) to calibrate the correct step size.

---

## Letters: Partial Alphabet Line

### Results
```
Uppercase r(alpha_idx, PC1) = 0.762  p < 0.0001
Lowercase r(alpha_idx, PC2) = 0.620  p = 0.0007
```

Letters form a PARTIAL alphabet line — weaker than digits (r=0.977)
but still significant. The alphabet line is:
- In PC1 for uppercase (A...Z arranged roughly linearly)
- In PC2 for lowercase (different geometric orientation)

This asymmetry makes sense: uppercase and lowercase letters have
different tokenization contexts and different embedding neighborhoods.

### src_pc Comparison

```
digits (1-9):    src_pc = 0.757  (tightest cluster)
weekdays:        src_pc = 0.650
months:          src_pc = 0.559
UC letters:      src_pc = 0.332  (much looser)
LC letters:      src_pc = 0.235  (loosest)
```

Letters have the loosest source cluster of all ordinal categories.
This is because letters appear in every context — A appears in "apple",
"aardvark", "alpha", "ampere" — making letter embeddings more
semantically dispersed than calendar words.

The lower src_pc explains why letter→alpha-pos has lower pc (0.504)
despite the alphabet line being geometrically real: the source cluster
noise degrades the chord vector consistency.

---

## The Linearity Hierarchy (Day 299 Final)

```
Axis                          pc      Category
digit→word                    0.851   LABELLING (number symbol → spoken name)
weekday→number                0.842   LABELLING (temporal entity → ordinal)
month→number                  0.803   LABELLING (temporal entity → ordinal)
card→number                   0.789   LABELLING (number name → symbol)
season→quarter                0.691   LABELLING (proper noun → ordinal)  *low n=4
planet→orbital                0.609   LABELLING (proper noun → ordinal)  **attractor
ordinal→cardinal              0.582   LABELLING (ordinal word → cardinal word)
country→demonym               0.563   SEMANTIC  (highest non-labelling)
letter→alpha-pos              0.504   LABELLING (letter → ordinal)
+est (superlative)            0.436   INFLECTIONAL
+er (comparative)             0.393   INFLECTIONAL
elem:single-letter            0.390   SEMANTIC
...
word→antonym                  0.020   SEMANTIC  (lowest positive)
month (consecutive)          −0.090   TEMPORAL  (circular)
digit n→n+1                  −0.115   NUMERIC   (non-uniform)
weekday (consecutive)        −0.153   TEMPORAL  (circular)
```

**The top 7 axes are ALL labelling axes.** The first non-labelling axis
(country→demonym, 0.563) ranks 8th. Labelling axes dominate the upper
tier of the linearity spectrum.

The labelling axis class is defined by:
1. Source = tight semantic cluster (all members of one category)
2. Target = canonical ordinal numbers (the number line in W_E)
3. Mapping = bijective, unambiguous, culturally fixed

---

## Day 300 Plan

Day 300 is a milestone. Rather than introducing new axes, we will:

1. **Grand synthesis**: compute the universal ordinal direction (mean
   of all labelling axes) and measure its coherence.

2. **Ordinal direction as a coordinate axis**: how much of the
   embedding variance is explained by the ordinal direction? Is it
   one of the principal components of W_E?

3. **Generalisation test for card→number**: train on Two-Six,
   hold out Seven-Nine and Ace. Can the axis generalise across
   the full card range?

4. **The full linearity spectrum plot**: all 30+ axes mapped on one
   dimension, colour-coded by type (LABELLING, INFLECTIONAL,
   DERIVATIONAL, SEMANTIC, TEMPORAL).

---

## Files

- `expedition_log.md` — Day 299 results
- `433_labelling_axes.md` — DC 433: labelling axes class
- `day299_universal_ordinal.py` — experiment script
