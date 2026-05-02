# DC 433: Labelling Axes — The Highest-Linearity Class in W_E

**Day 298 | Four "labelling axes" (mappings from named-category members
to their ordinal numbers) form a new highest-linearity class: digit→word
pc=0.851, weekday→number pc=0.842, month→number pc=0.803,
ordinal→cardinal pc=0.582. These exceed all morphological and semantic
axes previously measured. The common factor: SOURCE = tight cluster of
named entities; TARGET = the W_E number line (digits 1–9, linearly
arranged with Pearson r=0.977). Consecutive number sequences have
NEGATIVE pc (digit consec pc=−0.115) because step sizes are non-uniform.
Cross-domain: month→num and weekday→num share the same axis direction
(cos=0.779); digit→word is anti-parallel to both (cos≈−0.72).**

---

## The Labelling Axis Class

A **labelling axis** is a displacement vector that maps from a named
category member to its canonical ordinal number:

```
January  →  1    Monday    →  1    first   →  one    1  →  one
February →  2    Tuesday   →  2    second  →  two    2  →  two
...           ...                  ...            ...
September → 9    Sunday    →  7    tenth   →  ten    9  →  nine
```

These are the highest-pc axes measured in 298 days of experiment:

```
Axis              pc      coh     train     holdout
digit→word        0.851   0.931   8/9  89%  (all except 1→one)
weekday→number    0.842   0.930   7/7 100%  (perfect)
month→number      0.803   0.908   9/9 100%  3/3 holdout (Jul/Aug/Sep)
ordinal→cardinal  0.582   0.789   9/10 90%  (first→one fails)
```

For comparison, the previous highest:
```
country→demonym   0.563
+est (superlative) 0.436
+er (comparative)  0.393
```

The labelling axes beat every inflectional, derivational, and semantic
axis by a large margin.

---

## Why Labelling Axes Have Maximum pc

Three factors combine to produce exceptional linearity:

### Factor 1: Source Cluster Homogeneity

```
src_pc (digit strings)  = 0.757  (tightest source cluster ever measured)
src_pc (months)         = 0.559
src_pc (weekdays)       = 0.650
```

All months co-occur in calendar contexts. All weekdays co-occur in
schedule contexts. All digit symbols appear in numeric contexts. The
source words are maximally homogeneous — they form the TIGHTEST clusters
in W_E for their respective categories.

### Factor 2: Target Linearity (The Number Line)

Digits 1–9 form a near-perfect LINEAR arrangement in W_E:

```
PC1 projections of digit embeddings:
  1: −0.227   2: −0.251   3: −0.142   4: −0.055   5: +0.005
  6: +0.107   7: +0.171   8: +0.194   9: +0.198
Pearson r(digit_idx, PC1) = 0.977   p < 0.0001
```

The number line is one of the most linear structures in W_E. The digits
were learned from their ORDINAL properties — they always appear in the
same relative sequence — so they are arranged as a line.

Cardinal words (one, two, three, ...) are equally linear:
```
Pearson r(word_idx, PC1) = 0.983   p < 0.0001
```

When we step from a month name toward a number, we're stepping toward
a LINEARLY ARRANGED target cluster. All training displacement vectors
point to different positions on this line, but they all point in the
SAME DIRECTION (toward the line), hence high pairwise cosine.

### Factor 3: Bijective, Unambiguous Mapping

Each month has EXACTLY ONE canonical number. Each weekday has EXACTLY
ONE canonical number. There is no:
- Dialectal variation (Monday is ALWAYS 1 in ISO week numbering)
- Grammatical context dependence (January is ALWAYS the 1st month)
- Polysemy at the target (number 1 is unambiguous in this context)

This eliminates all sources of chord-vector noise.

---

## The Number Line vs. Circular Structures

### Numbers Form a Line

Digit strings (1–9) have Pearson r=0.977 between digit index and PC1.
This is a LINE: digits are arranged in numeric order along the first
principal component of their embedding subspace.

```
1←→2←→3←→4←→5←→6←→7←→8←→9
```

Why a line rather than a circle? Because the number sequence is
BOUNDED and OPEN (not cyclic). There is no "9 wraps to 1" in standard
mathematical context.

### Months Form a Circle (DC 432)

Months form a RING in the PC1-PC2 plane. This is because the month
sequence is cyclic: December is followed by January.

### The Paradox Resolved

**Months form a circle, yet month→number has pc=0.803.**

This is not a contradiction:
- The RING structure encodes temporal cyclicity (December→January wrap)
- The LABELLING structure encodes calendar position (January = 1st month)
- These two encodings coexist in orthogonal dimensions

The chord vectors for month→number all point toward the number line
(numbers 1–9), not along the month ring. The ring structure is in a
completely different geometric direction.

---

## Why Consecutive Digit Axis Has Negative pc

Despite digits forming a LINE (r=0.977), the consecutive axis (1→2,
2→3, ...) has pc=−0.115. This seems contradictory.

### Non-Uniform Step Sizes

The PC1 spacings between consecutive digits:
```
1→2: −0.227 to −0.251  Δ = −0.024  (TINY — 1 and 2 are close)
2→3: −0.251 to −0.142  Δ = +0.109  (LARGE — 2 and 3 jump)
3→4: −0.142 to −0.055  Δ = +0.087
4→5: −0.055 to +0.005  Δ = +0.060
5→6: +0.005 to +0.107  Δ = +0.102
6→7: +0.107 to +0.171  Δ = +0.064
7→8: +0.171 to +0.194  Δ = +0.023  (TINY — 7,8,9 cluster together)
8→9: +0.194 to +0.198  Δ = +0.004  (NEAR-ZERO)
```

The steps are HIGHLY NON-UNIFORM. The 1→2 step is tiny (PC1 barely
moves). The 2→3 step is 4× larger. At the high end, 8→9 is near-zero.

The number line in W_E is NOT a ruler with equal tick marks. It is a
**compressed representation** where:
- Small numbers (1, 2) are tightly clustered together
- Middle numbers (3–7) spread more uniformly
- Large numbers (8, 9) compress again (within the single-digit range)

This compression means each consecutive step vector has a DIFFERENT
direction in the full 3072-dimensional embedding space, producing
negative pairwise cosine.

### The "2 Anomaly"

Both digit "2" and cardinal word "two" are MORE negative in PC1 than
"1"/"one":
```
digit: 1→ −0.227, 2→ −0.251
word:  one→ −0.319, two→ −0.338
```

"Two" is embedded slightly FURTHER LEFT than "one" on PC1. This violates
the expected monotonic ordering. The likely cause: "one" is extremely
polysemous (indefinite pronoun: "one does not simply...") and is embedded
slightly further from the pure ordinal cluster, placing it near "two"
in PC1 rather than cleanly preceding it.

---

## Cross-Domain Axis Alignment

The labelling axes reveal important geometric structure:

```
cos(month→num, weekday→num) = +0.779  *** ALIGNED ***
cos(month→num, digit→word)  = −0.731  *** ANTI-PARALLEL ***
cos(weekday→num, digit→word)= −0.712  *** ANTI-PARALLEL ***
cos(ordinal→card, others)   ≈  0.11   (ORTHOGONAL)
```

### Month→num ≈ Weekday→num (cos=0.779)

Both operations map from a NAMED TEMPORAL ENTITY (month, weekday) to
its ORDINAL NUMBER. The geometric direction is nearly identical. The
difference (cos=0.779 rather than 1.0) arises because:
- Months are labelled 1–12 (using 1–9 single-token range)
- Weekdays are labelled 1–7
- The source clusters are in slightly different positions (calendar vs
  schedule semantic regions)

But the DIRECTION OF MOVEMENT (from named entity toward number line)
is the same. This confirms that the number line has a single consistent
direction in W_E that ALL labelling operations navigate toward.

### Digit→word ≈ −(Month→num) (cos=−0.73)

`digit→word` maps in the REVERSE direction: from digit symbols (1, 2,
...) to their spoken names (one, two, ...). This reversal produces
anti-parallelism with month→num.

The reversal is not perfect (cos=−0.73, not −1.0) because:
- Month→num: source = CALENDAR words, target = NUMBER SYMBOLS
- Digit→word: source = NUMBER SYMBOLS, target = NUMBER WORDS
The source domains differ, so the reversal is approximate.

### Ordinal→cardinal is Orthogonal (cos≈0.11)

`ordinal→cardinal` (first→one, second→two) maps from ordinal WORDS
to cardinal WORDS. Both source and target are in the "word/name"
region of W_E, so this is a within-word-domain operation, geometrically
orthogonal to the "word→symbol" labelling axes.

---

## Practical Implication: Universal Ordinal Retrieval

Any category with a canonical ordinal labelling can be retrieved via
the labelling axis. The axis direction is consistent across domains
(cos=0.779 between months and weekdays), suggesting a UNIVERSAL
ORDINAL DIRECTION in W_E.

Prediction: Other ordinal systems will produce similar pc values:
- Planet→orbital order (Mercury=1, Venus=2, ...) → predicted pc ≈ 0.70
- Playing card rank→number (Ace=1, Two=2, ...) → predicted pc ≈ 0.75
- Letter→alphabetical position (A=1, B=2, ...) → predicted pc ≈ 0.65

The universal ordinal direction is one of the most consistently encoded
semantic relationships in the embedding space — because ordinal position
is one of the most frequently used pieces of information in human text.

---

## Day 299 Plan

Test the universal ordinal direction hypothesis with:

1. **Letter→alphabet position** (A=1, B=2, ..., Z=26): do letters
   1–9 achieve same high pc? Are letters arranged on a line in W_E?

2. **Planet→orbital position** (Mercury=1, ..., Neptune=8): small
   n=8, but strong categorical identity.

3. **Season→quarter** (Spring=1, Summer=2, Autumn=3, Winter=4):
   n=4, expected lower pc due to small set. But cyclic like months?

4. **Number arithmetic axis**: 1+1=2, 2+1=3, ... — is the "add one"
   operation geometrically captured? Test: n→n+1 vs n→n+2 vs n→n+5.

---

## Files

- `expedition_log.md` — Day 298 results
- `432_circular_temporal_encoding.md` — DC 432: month ring structure
- `day298_number_line.py` — experiment script
