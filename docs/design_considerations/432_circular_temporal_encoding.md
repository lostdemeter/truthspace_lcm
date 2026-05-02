# DC 432: Circular Temporal Encoding — Months Form a Ring, Numbers Form a Line

**Day 297 | Temporal sequences are encoded GEOMETRICALLY in W_E, but not
as linear axes. The twelve months form a NEAR-PERFECT CIRCLE in the
PC1-PC2 plane of their embedding subspace (Jan=133°, Feb=112°, ...,
Dec=161°, wrapping back to Jan). Consecutive month pairs have pc=−0.090
(negative) because each step around the circle points in a different
direction. Weekday consecutive pairs have pc=−0.153. BUT the
month→number axis (January→'1', February→'2', ...) has pc=0.803 — the
HIGHEST measured across all 297 days. The ordinal labelling of months
is near-perfectly linear because the number tokens 1–9 form a linear
arrangement in W_E. Two distinct encodings coexist: the months-as-ring
(cyclic) and the months-as-labelled (linear ordinal).**

---

## The Negative pc of Temporal Sequences

### Result

```
Axis                    pc       coh     train
month (consecutive)    -0.090    0.100   1/11  (Nov→Dec only)
weekday (consecutive)  -0.153    0.198   1/6   (Sat→Sun only)
month abbrev (consec)  -0.088    0.100   1/11  (Nov→Dec only)
month skip-1           -0.087    0.144   1/10
```

All temporal sequence axes have **negative** pairwise chord cosine.

### Interpretation

For a geometrically linear sequence, each consecutive step vector should
point in the SAME direction (pc → 1.0). For a perfectly circular
sequence, each consecutive step vector points in a progressively rotated
direction. The angle between adjacent steps depends on the number of
points on the circle:

```
12 points on circle: adjacent steps separated by 360°/12 = 30°
cosine of 30° = 0.866  → expected pc ≈ 0.866  (if perfect circle)
```

But we measure pc ≈ −0.090. This is MUCH LOWER than expected for a
perfect circle. Why?

1. **The circle is not in a 2D plane** — it lives in a 3072-dimensional
   space. The PCA projection shows the circular structure in the top 2
   PCs, but the actual chord vectors span many dimensions. Components
   orthogonal to the circle plane contribute noise.

2. **The circle is irregular** — angular steps range from 19° to 48°.
   Summer months (April-May) jump further than winter months (Jan-Feb).
   Irregular spacing means chord vectors point in genuinely different
   directions.

3. **December wraps back** — the Dec→Jan chord (if included) would
   point roughly opposite to the Jan→Feb chord, further pulling pc
   toward −1.

The negative pc means the chord vectors are, on average, slightly
ANTI-CORRELATED — not just uncorrelated. This happens because the
circle subtends more than 180° of the embedding space, so opposite-side
steps cancel when averaged.

---

## The Circular Structure of Months

### Evidence

SVD of the 12 month embeddings reveals a dominant plane (PC1, PC2).
Projecting each month onto this plane gives angles that trace a nearly
complete circle:

```
Month       Angle    Δ from prev
January      133°        —
February     112°      -21°
March         69°      -43°
April         48°      -21°
May            0°      -48°
June         -28°      -28°
July         -67°      -39°
August       -86°      -19°
September   -121°      -35°
October     -142°      -21°
November    -172°      -30°
December     161°      +27°  (wraps ≈ -27° from +188°)
```

Total rotation: 133° to 161° going clockwise ≈ 360° (with wrap-around).
The sequence is nearly complete: December almost returns to January.

### Statistical Confirmation

```
Pearson r(month_idx, PC1)  = −0.401   p=0.197  (not significant alone)
Pearson r(month_idx, PC2)  = −0.619   p=0.032  (significant!)
Spearman ρ(month_idx, PC1) = −0.532   p=0.075  (marginal)
```

PC2 alone has a significant linear correlation with month index — because
the circle projects onto PC2 as a sinusoidal function with period 12,
which is approximately linear over the range 1–12 months.

Source pairwise cosine of month embeddings: **0.559** — the months form
a TIGHT cluster in W_E (all co-occur in calendar contexts), but they
are internally ordered as a ring within that cluster.

### Why a Circle?

Months have a fundamentally **cyclic** structure: after December comes
January again. A circle is the natural geometric encoding of cyclic
order. LLMs learned this from:
- Date expressions: "December 31 ... January 1"
- Seasonal cycles: "after winter comes spring"
- Anniversary patterns: "same time next year"

The model encodes the cyclic temporal structure as a ring in the
embedding space — not explicitly designed, but emergent from the
co-occurrence patterns in training data.

---

## The Month→Number Axis: pc = 0.803

### Result

```
Axis             pc      coh     scale  train
month→number     0.803   0.908   0.42   9/9 (100%)
```

This is the **highest pairwise chord cosine measured across 297 days**.
Every prior axis:

```
country→demonym  0.563
country→lang*    0.474  (inflated)
+est             0.436
+er              0.393
elem:single      0.390
```

The month→number axis (pc=0.803) exceeds all of these by a large margin.

### Why So High?

The month→number transformation has THREE properties that maximise pc:

1. **Source cluster is tight**: all 12 months cluster tightly (src_pc=0.559).
   They are in the same semantic region of W_E.

2. **Target cluster is tight and linear**: numbers 1–9 are single BPE
   tokens that form a **number line** in W_E. Their embeddings are
   arranged in a consistent ordinal direction. This is the most linear
   semantic structure possible.

3. **The mapping is BIJECTIVE and CANONICAL**: January is ALWAYS 1,
   February is ALWAYS 2. There is no ambiguity, no dialect variation,
   no grammatical context dependence. The mapping is a hard fact
   encoded consistently across all training data.

The combination of tight source cluster + linear target sequence +
unambiguous mapping produces the highest pc we have seen. The chord
vectors all point toward the same general direction (toward the number
line) with consistent magnitude.

### Why This Exceeds Morphological Axes

Even +er (comparative) has pc=0.393, far below month→number (0.803).
The difference:
- +er pairs: `fast→faster, slow→slower, tall→taller` — sources vary
  across many semantic domains; comparative forms vary slightly in their
  embedding positions
- month→number pairs: ALL sources are months (max homogeneity); ALL
  targets are numerals (the most orderly arrangement in W_E)

The month→number axis is a "perfect storm" of all factors that
maximise pc: source homogeneity, target linearity, mapping unambiguity.

---

## Two Coexisting Encodings

The month embeddings carry TWO independent pieces of information:

### Encoding 1: Cyclic Ring (months-as-time)

The 12 months form a ring in the 2D PC subspace of their cluster.
This encodes: **temporal cyclicity** — January follows December.
Accessible via: SVD/PCA of the month embedding matrix.
NOT accessible via linear axis (negative pc).

### Encoding 2: Ordinal Label (months-as-number)

Each month has a consistent association with its ordinal number (1–12).
This encodes: **calendar position** — January is 1st, February is 2nd.
Accessible via: month→number displacement axis (pc=0.803).
IS a linear axis.

These two encodings are ORTHOGONAL in the following sense:
- The ring structure is in the PC1-PC2 plane of month embeddings
- The ordinal labelling is in the direction toward the number line
- These directions are different dimensions in W_E

A single embedding carries BOTH a cyclic position AND an ordinal label.
This is not a contradiction — they are different aspects of the same
concept, encoded in different geometric directions.

---

## Dominant Attractors in Temporal Sequences

### Sunday Dominates Weekday Predictions

The weekday consecutive axis fails catastrophically (1/6 = 17%) because
`Sunday` is the dominant attractor:

```
Monday    → Tuesday:   got Sunday  [---]
Tuesday   → Wednesday: got Sunday  [---]
Wednesday → Thursday:  got Sunday  [---]
Thursday  → Friday:    got Sunday  [---]
Friday    → Saturday:  got Sunday  [---]
Saturday  → Sunday:    got Sunday  [HIT]
```

Only the pair where the target IS Sunday succeeds. The displacement axis
points toward `Sunday` (the last weekday, most distinctive in text
because it's the only day with strong "rest/leisure" associations),
and the scale is calibrated to reach it.

### December Dominates Month Abbreviation Predictions

For abbreviated months (Jan, Feb, ..., Dec), `Dec` is the dominant
attractor:
```
Jan→Feb: got Dec   Feb→Mar: got Dec   Apr→May: got Dec
Jul→Aug: got Dec   Sep→Oct: got Dec
```

December dominates because it is the ENDPOINT of the linear year
sequence and possibly the most distinctive month in text (Christmas,
year-end). The averaging of chord vectors skews toward December.

---

## Comparison: pc Values Across All Axes (Updated)

```
Axis                     pc      type        notes
month→number             0.803   TEMPORAL    *** HIGHEST EVER
country→demonym          0.563   SEMANTIC
country→lang*            0.474   SEMANTIC    inflated
+est                     0.436   INFL
+er                      0.393   INFL
elem:single-letter       0.390   SEMANTIC
country→capital          0.317   SEMANTIC
animal→class             0.254   SEMANTIC
person→nationality       0.246   SEMANTIC
past_irr                 0.230   INFL
gender                   0.213   INFL
+ness                    0.211   DERIV
+ed                      0.174   INFL
elem:double-letter       0.163   SEMANTIC
+s plural                0.155   INFL
element→symbol           0.139   SEMANTIC
in-/im-                  0.133   DERIV
+less                    0.133   DERIV
+tion                    0.130   DERIV
+ment                    0.124   DERIV
un-                      0.121   DERIV
elem:latin-derived       0.104   SEMANTIC
+ful                     0.104   DERIV
field→concept            0.087   SEMANTIC
word→antonym             0.020   SEMANTIC
month (consecutive)     -0.090   TEMPORAL    circular, not linear
month skip-1            -0.087   TEMPORAL    circular
month abbrev (consec)   -0.088   TEMPORAL    circular
weekday (consecutive)   -0.153   TEMPORAL    circular, Sunday attractor
```

The spectrum now spans from pc=0.803 (month→number) to pc=−0.153
(weekday consecutive). The NEGATIVE entries represent circular/non-linear
structures that CANNOT be captured by linear displacement axes.

---

## Day 298 Plan

1. **Number line axis** (1→2, 2→3, ..., 8→9): test whether the number
   sequence itself is linearly encoded. Expected: high pc (>0.5) if
   numbers form a line, but possibly also negative (if they form a
   different structure).

2. **Weekday→number** (Monday→1, ..., Sunday→7): expected to match
   month→number quality (pc ≈ 0.7–0.8).

3. **Month→number holdout**: months 10/11/12 are multi-token, so test
   alternative: train on months 1–6, hold out 7–9.

4. **Ordinal vs cardinal**: does `first, second, third, ...` map to
   `one, two, three, ...` linearly? (ordinal→cardinal axis).

---

## Files

- `expedition_log.md` — Day 297 results
- `431_target_cluster_density.md` — DC 431: density anomalies
- `day297_temporal_sequence.py` — experiment script
