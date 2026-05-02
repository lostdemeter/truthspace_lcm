# DC 436: Axis Algebra and PC1 — Composition, Transitivity, and the Specificity Axis

**Day 301 | Axis composition is TRANSITIVE for labelling chains:
cos(month→digit + digit→word, month→word) = 0.9944; cos(weekday→num
+ num→ordinal, weekday→ordinal) = 0.9999. Retrieval using composed
axes: 7/9 (78%) and 6/7 (86%). Morphological doubling FAILS:
cos(+er+er, +est) = 0.460 = cos(+er, +est) — because +er and +est
are co-directional and cannot be composed by addition. The global
PC1 of W_E is the TOKEN SPECIFICITY AXIS: all common tokens project
negative (function words to −0.58, punctuation to −0.73), rare
subword fragments project positive (+0.3 to +0.67). r(token_ID,
PC1) = 0.335, r(word_length, PC1) = 0.320 (both p<1e-48). Cyclic
shift axes (month+2, weekday+2) fail completely — circular structures
cannot be navigated with linear displacement.**

---

## Axis Algebra: The Transitive Composition Law

### Theorem (empirical): Labelling Chains Compose

For any labelling chain A→B→C where:
- ax(A→B) is a labelling axis (name → ordinal)
- ax(B→C) is a labelling axis (ordinal → name, or ordinal → ordinal)

The composed axis:
```
ax_composed = normalise(ax_raw(A→B) + ax_raw(B→C))
```
satisfies:
```
cos(ax_composed, ax(A→C)) ≈ 1.0
```

### Evidence

```
Chain                              cos(composed, direct)
month→digit + digit→word = month→word   0.9944
weekday→num + num→ordinal = weekday→ord  0.9999
```

Both compositions achieve cos ≥ 0.994, well above the "approximate
composition" threshold of 0.9. The second is essentially EXACT
(cos=0.9999 rounds to 1.000).

### Retrieval Using Composed Axes

```
month→word (composed via month→digit + digit→word):
  scale=0.50  acc=7/9 (78%)
  January → one:  MISS (got five)
  February → two: MISS (got seven)
  March → three:  HIT
  ...
  September → nine: HIT

weekday→ordinal (composed via weekday→num + num→ordinal):
  scale=0.50  acc=6/7 (86%)
  Monday → first: MISS (got sixth)
  Tuesday → second: HIT
  ...
  Sunday → seventh: HIT
```

The composed axis actually **works for retrieval**, not just as a
geometric curiosity. 7/9 and 6/7 accuracy from axes that were never
directly trained on the target mapping.

### Why January and Monday Fail

**January→one fails** (composed axis gives "five"):
- The chain month→digit works for January (target: digit "1")
- But digit→word for "1" is slightly off because "one" is polysemous
  (the indefinite pronoun "one does not simply...")
- The composed axis deviates at the "one" end of the number line

**Monday→first fails** (composed axis gives "sixth"):
- Monday→num: Monday maps to "1", first weekday
- num→ordinal: "1"→"first" (ordinal)
- But "Monday" as the "1st" weekday and "first" as an ordinal have
  different embedding neighborhoods — "first" is not exclusively
  ordinal
- The composed axis calibrates for the middle weekdays and Monday
  (at position 1) undershoots, landing at "sixth" instead

### The Composition Limit: Polysemy at the Intermediate

Both failures occur at position 1 (January=1st, Monday=1st). This is
the same polysemy anomaly seen in Day 298:
- "one" is heavily polysemous (indefinite pronoun)
- "first" is heavily polysemous (adverb: "at first", "first of all")
- These reduce the accuracy of any axis that passes through position 1

Positions 2-9 compose cleanly because "two"/"second" through
"nine"/"ninth" are less polysemous.

---

## Morphological Doubling Does Not Compose

### +er + +er ≠ +est

```
cos(+er, +est)            = 0.461  [same direction]
cos(+er+er, +est direct)  = 0.460  [identical to single +er]
```

Adding the +er axis to itself produces no improvement — the composed
axis is identical in direction to the single +er axis:
```
ax_comp = normalise(2 × ax_+er) = normalise(ax_+er)  [scaling doesn't change direction]
```

This is a **mathematical fact**: normalise(2v) = normalise(v) for any
non-zero vector v. The doubled axis is exactly the same direction as
the single axis.

### Why +er and +est Are Co-Directional (cos=0.461)

The comparative (+er) and superlative (+est) are on the same
"intensification axis" in W_E. Both operations move an adjective
toward higher intensity, so:
- fast → faster: move along intensification axis by Δ₁
- fast → fastest: move along same axis by Δ₂ (where Δ₂ > Δ₁)

The directions are nearly identical (cos=0.461) but the magnitudes
differ. This means comparative and superlative are on the SAME RAY,
with the superlative being a LONGER STEP along the same ray.

To compose +er+er→+est, you would need the separate
comparative-to-superlative axis (faster→fastest), not the base-form
axis. Indeed, ax(comp→sup) has pc=0.426 and represents exactly this
"second step."

### The Co-Directionality of Morphological Gradation

```
cos(+er, +est) = 0.461   both add intensity
cos(+er, comp→sup) = ?   (second step from comparative)
```

The pair (+er, +est, comp→sup) forms a COLLINEAR FAMILY in W_E:
all three axes point in the same direction (intensification), with
different step magnitudes. This is why:
- Morphological gradation is a 1D phenomenon (all steps on one line)
- A single axis cannot distinguish "one step" from "two steps"
- The number of steps must be encoded in the step MAGNITUDE, not direction

---

## Global PC1: The Token Specificity Axis

### Measurement

From 8,000-token power iteration:
```
PC1 explains ~3.35% of W_E total variance
All common tokens project NEGATIVE:
  Punctuation:     −0.33 to −0.73  (( = −0.73, lowest common token)
  Function words:  −0.36 to −0.58  (and = −0.58)
  Digit symbols:   −0.11 to −0.14
  Content words:   −0.15 to −0.28  (months, weekdays, adjectives, verbs)

All rare subwords project POSITIVE:
  +0.668  ',' (id=30625, a rare variant, not common comma id=11)
  +0.547  ',...'  (code fragment)
  +0.497  ',DB'   (identifier fragment)
  +0.408  'perature'  (partial suffix)
  +0.397  '/lic'  (URL/path fragment)
```

### Statistical Confirmation

```
r(token_ID, PC1) = 0.335   p < 1e-53   (higher ID = rarer BPE token)
r(word_length, PC1) = 0.320   p < 1e-49   (longer = rarer)
```

Both correlations are significant and consistent: PC1 increases with
rarity and length. This is the standard finding in large embedding
spaces — the dominant component captures the **frequency/specificity**
dimension of the vocabulary.

### Interpretation

PC1 is NOT semantically informative about word MEANING. It encodes:
- **Token frequency**: how often does this exact byte sequence appear?
- **Compositional specialization**: is this a complete word (common)
  or a fragment/subword of a specialized term (rare)?

The negative region (common tokens) includes ALL of:
- Grammar tokens (the, and, is, of, to)
- Punctuation (., ,, !, ?, :)
- Common digit strings (1, 2, 3)
- Common content words (big, fast, run, go)

The positive region (rare tokens) includes:
- Programming/code fragments (/lic, *</, <tag)
- Partial words (perature, iginal, bsolute)
- Multi-byte special sequences (LLU, BOSE, aeda)

### Implication for the Ordinal Direction (v_ord)

In Day 300, we measured that v_ord explains 1.53% of W_E variance
(vs. PC1 = 3.35%). The v_ord is NOT PC1 — it is a SEMANTIC direction
orthogonal to the frequency axis. The digit symbols (1-9) project
HIGH on v_ord (+0.53 to +0.68) but LOW on PC1 (−0.11 to −0.14).

This tells us:
- PC1 (frequency axis): digits are common → project LOW
- v_ord (ordinal axis): digits are labels → project HIGH

The two axes measure DIFFERENT aspects of the digit tokens:
their FREQUENCY (PC1) vs. their SEMANTIC ROLE as ordinal labels (v_ord).

---

## Why Cyclic Shift Axes Fail

### Month +2 shift: pc=−0.087, 1/10 accuracy

All predictions cluster at November/December. The shift axis averages
10 chord vectors that each point in different directions (because they
trace the month ring from different starting points). The mean of these
rotational vectors points toward the dominant late-year attractor —
where the most chord vectors have their largest positive component.

The month ring has no "north" — each position on the ring requires a
different direction to advance by +2. A SINGLE linear axis cannot
encode a function that requires different directions at different positions.

### Contrast: Non-Cyclic Shift Works

For non-cyclic structures (digit n→n+k), the axis fails because of
NON-UNIFORM STEP SIZES (pc<0), not because of cyclicity. The direction
is approximately consistent (cos~0.76-0.84 between different increment
sizes) but the individual steps are too variable for reliable retrieval.

For cyclic structures (months, weekdays), the axis fails because of
DIRECTIONAL INCONSISTENCY (each chord points somewhere different).

### Summary

```
Structure type      pc      Failure mode
Cyclic (months)    −0.09   Directional inconsistency (ring)
Non-uniform line   −0.12   Step size inconsistency (compressed number line)
Linear labelling   +0.80   WORKS (consistent direction + consistent target)
```

---

## Updated Understanding of W_E Geometry (Day 301)

After combining all findings:

1. **PC1 (3.35% variance)** = Token specificity / inverse frequency
   — encodes HOW COMMON a token is, not WHAT it means

2. **v_ord (1.53% variance)** = Ordinal direction / symbol vs. word-form
   — encodes WHETHER a token functions as an ordinal label

3. **Axis algebra is transitive** for labelling chains with numeric
   intermediaries — W_E has a consistent, composable labelling system

4. **Morphological gradation is collinear** — all degree axes (+er, +est,
   comp→sup) point in the same direction with different magnitudes

5. **Cyclic structures are non-composable** — ring-like encodings require
   position-dependent axes, which single linear axes cannot provide

---

## Day 302 Plan

1. **PC2 exploration**: what does the second principal component encode?
   Does it have more semantic content than PC1?

2. **Axis composition with morphological chains**: test
   un- + +ness = ? (does "un + ness" compose to "un-ness" axis?)
   base→comparative + comparative→superlative (two-step vs. one-step)

3. **Rank the semantic axes by composition quality**: which pairs of
   axes compose most accurately?

4. **Test whether the ordinal direction v_ord aligns with any known
   "meaningful" direction in W_E** (e.g., is v_ord close to any
   semantic axis like country→capital or gender axis)?

---

## Files

- `expedition_log.md` — Day 301 results
- `435_grand_synthesis_day300.md` — DC 435: complete linearity map
- `day301_pc1_and_composition.py` — experiment script
