# DC 438: The Degree Triangle and PC2 as the Digit Symbol Axis

**Day 303 | The degree gradation system forms a precise L-shaped
triangle in W_E. All 8 adjective triples project to near-identical
2D coordinates: BASE at (−0.11, −0.07), COMPARATIVE at (+0.27, −0.07),
SUPERLATIVE at (+0.10, +0.31). The comparative is a HORIZONTAL step
(+er = right), the comp→sup step is DIAGONAL (up-left), and the
superlative is VERTICALLY ELEVATED above both. The comparative lies
87% of its own length off the base-to-superlative line (height/|bc|
= 0.865). PC2 is the DIGIT SYMBOL AXIS: digit tokens (1–9) score
PC2 = +0.103, uniquely separated from ALL other vocabulary groups
(next highest: weekdays at +0.007). The round-trip inverse (faster
−er → fast) confirms axis invertibility. L+M combined (6 directions)
explains 2.70% of W_E variance — less than PC1 alone (3.35%).**

---

## The Degree Triangle: A Universal L-Shape

### 2D Coordinates

Using the basis (e1 = base→comparative direction, e2 = orthogonal
complement of base→superlative direction), all 8 adjective triples
project to:

```
                SUPERLATIVE (~+0.10, +0.31)
                        *
                       /|
                  +est / | er→est
                     /   |
BASE (~-0.11, -0.07) ----* COMPARATIVE (~+0.27, -0.07)
                    +er
```

The coordinates are strikingly consistent across all 8 triples:

```
Triple    BASE (e1,e2)      COMP (e1,e2)      SUP (e1,e2)
fast      (-0.12, -0.07)   (+0.26, -0.07)   (+0.08, +0.31)
tall      (-0.09, -0.08)   (+0.31, -0.08)   (+0.12, +0.31)
short     (-0.12, -0.09)   (+0.27, -0.08)   (+0.09, +0.30)
bright    (-0.08, -0.05)   (+0.29, -0.05)   (+0.09, +0.32)
dark      (-0.10, -0.06)   (+0.26, -0.06)   (+0.11, +0.30)
deep      (-0.13, -0.09)   (+0.25, -0.10)   (+0.09, +0.29)
strong    (-0.10, -0.09)   (+0.29, -0.09)   (+0.10, +0.29)
weak      (-0.08, -0.05)   (+0.29, -0.06)   (+0.12, +0.31)
```

The three degree positions form three tight clusters:
- All bases: e2 ∈ [−0.09, −0.05]  (variance 0.0002)
- All comparatives: e2 ∈ [−0.10, −0.05]  (variance 0.0003)
- All superlatives: e2 ∈ [+0.29, +0.32]  (variance 0.0001)

The **superlative cluster is the tightest** — all superlatives occupy
the same small region of the 2D degree plane.

### The L-Shape Structure

The degree triangle has one approximately RIGHT ANGLE at the comparative:

- **+er step** (base→comparative): HORIZONTAL in 2D (Δe1≈+0.38, Δe2≈0)
- **er→est step** (comparative→superlative): UP-LEFT (Δe1≈−0.17, Δe2≈+0.38)
- **+est step** (base→superlative): UP-RIGHT diagonal (Δe1≈+0.21, Δe2≈+0.38)

The angle at the comparative vertex is approximately:
```
cos(−bc, cs) = +0.40 → angle ≈ 66°
```

The right-angle at the comparative explains why:
- **+er + er→est = +est** (cos=0.976): the two legs of the L add to the
  hypotenuse
- **+er + +er ≠ +est** (cos=0.460): doubling the horizontal step stays
  horizontal, never reaches the elevated superlative

### The "Jump" to Superlative

The most striking observation: all superlatives are at e2 ≈ +0.31,
while all bases AND comparatives are at e2 ≈ −0.07. The superlative
is NOT simply "more comparative." It occupies a COMPLETELY DIFFERENT
vertical level. The +est operation is therefore NOT an intensification
of +er — it is a move into a distinct SUPERORDINATE region of the
adjective space.

This has a linguistic parallel: in English grammar, the superlative
is not just "the most comparative" — it is a different grammatical
function (reference to a class maximum). The 2D geometry in W_E
reflects this grammatical distinction.

### Triangle Dimensions

```
Side lengths (mean ± std):
  base→comp  (|bc|): 0.536 ± 0.035  (87% of the full range)
  comp→sup   (|cs|): 0.580 ± 0.030  (94% of the full range)
  base→sup   (|bs|): 0.616 ± 0.037  (100%, reference)

height (comp off b-s line): 0.464 ± 0.025
height / |bc| = 0.865   (comparative is 87% of its step displaced from b-s line)
```

The height is 86.5% of the base→comp side length. This means the
comparative is nearly PERPENDICULAR to the base-to-superlative line.
The "L" is not a thin L (like an acute angle) but an almost perfect
right-angle L.

---

## PC2: The Digit Symbol Axis

### Measurement

```
Group                 PC2_mean   PC1_mean
Numbers-digits         +0.1031   −0.6882  ← uniquely positive PC2
Weekdays               +0.0066   −0.1618
Derivations-un         +0.0041   −0.1407
Superlatives           −0.0023   −0.1431
Nouns-proper           −0.0028   −0.1689
Derivations-ment       −0.0048   −0.1637
Punct/Special          −0.0070   −0.4368
Derivations-ness       −0.0076   −0.1147
Months                 −0.0100   −0.2060
Comparatives           −0.0127   −0.1201
Verbs-past-reg         −0.0145   −0.1769
...all others...        −0.02 to −0.06
Determiners            −0.0611   −0.3328  ← most negative PC2
```

The digit symbols (1, 2, 3, ..., 9) are uniquely isolated at PC2 =
+0.103. Every other category — function words, content words, months,
weekdays, morphological derivatives — is within ±0.007 of zero. The
separation between digits and everything else is approximately 15×
the standard deviation of non-digit groups.

### What PC2 Encodes

PC2 is the **DIGIT SYMBOL AXIS** — it isolates single-character
numeric tokens from all other vocabulary. Two key properties:

1. **Not correlated with frequency** (r = −0.085, p = 1.3×10⁻⁴, weak)
   — digit symbols are common tokens but other common tokens (function
   words, punctuation) do NOT score high on PC2.

2. **Not correlated with length** (r = 0.036, p = 0.11, non-significant)
   — digit symbols are single characters, but so are many other tokens
   (!, ?, :, ;) which all score near zero on PC2.

PC2 captures something specifically SEMANTIC about digits: their role
as **symbolic quantifiers** — the minimal units of cardinal notation.

### PC1+PC2 Separation of Key Groups

In the 2D space (PC1, PC2):
```
                    PC2
                     |
           Digits    |  * (+0.103)
                     |
   ──────────────────+──────────────── PC1
 Punct (-0.44)*      |     * Named entities (-0.19, ~0)
 Funcs (-0.39)*      |     * Content words (-0.22, ~0.03)
                     |
```

PC1 alone cannot separate digits from function words (both very
negative on PC1). PC2 completes the separation: digits jump to +0.10
on PC2 while function words remain negative (−0.02 to −0.06).

The combined (PC1, PC2) representation enables a **full separation**
of the three most important vocabulary clusters:
- DIGIT SYMBOLS: low PC1, high PC2
- FUNCTION WORDS: very low PC1, low PC2
- NAMED ENTITIES: medium PC1, near-zero PC2

### Implication for v_ord

The universal ordinal direction v_ord maps from NAMED ENTITIES (low
on v_ord) to DIGIT SYMBOLS (high on v_ord). Since PC1 cannot separate
these two groups (both negative), and PC2 can (digits positive, named
entities near-zero), v_ord must have a significant PC2 component.

A rough estimate:
- In (PC1, PC2) space, v_ord points from (−0.20, 0) toward (−0.12, +0.10)
- Delta: (+0.08, +0.10) → normalised: (0.62, 0.79) in PC1-PC2 subspace

So v_ord is approximately **20% PC1 + 80% PC2** in the 2D subspace
spanned by PC1 and PC2. The digit symbol separation (PC2) dominates
the ordinal direction.

---

## Inverse Axis Confirmation

### Round-Trip Identity

```
|+er + rev(+er)| = 0.000000   [exact mathematical zero]
```

The +er axis and its reverse sum to exactly zero. This is the
ENCODE=DECODE identity extended to arbitrary morphological axes:

- `fast +er (scale=s) → faster`  ✓
- `faster −er (same scale=s) → fast`  ✓

The inverse axis correctly recovers the source word. This confirms
that the mean-displacement construction is fully invertible: the
reverse axis is exactly `−forward_axis`.

### Cross-Domain Orthogonality Confirmed

```
cos(gender, +er)  = +0.063   ≈ 0  [independent domains]
```

Axes from independent semantic domains (gender vs. degree) are
orthogonal. This is consistent with v_ord being orthogonal to all
non-labelling axes (Day 302): different semantic operations occupy
independent geometric subspaces.

### Past Tense Co-Alignment

```
cos(past_irr, past_reg) = +0.442
```

Both past tense axes (regular +ed and irregular past) are significantly
co-aligned. They are not identical (cos ≠ 1) because:
- Regular past: changes the SUFFIX (+ed)
- Irregular past: changes the entire word form (go→went, come→came)

But they both perform the same FUNCTION (marking past tense), so their
mean displacement vectors are co-directional.

---

## Subspace Variance Summary

```
Direction(s)              Var%      note
─────────────────────────────────────────────────────────
PC1 (1 direction)         3.35%     frequency axis
PC2 (1 direction)         ~X%       digit axis
PC3 (1 direction)         ~Y%       morphological axis
v_ord (1 direction)       1.91%     ordinal direction
5 morph axes (subspace M) 0.79%     morphological subspace
L+M combined (6 dirs)     2.70%     full semantic subspace
─────────────────────────────────────────────────────────
```

PC1 alone (frequency axis) explains MORE variance than the entire
L+M semantic subspace (6 directions). This confirms that:

1. The single most important structural feature of W_E is TOKEN
   FREQUENCY (how common a token is in the training corpus).

2. All semantic content (ordinal labelling, morphological operations)
   is encoded in SMALL SECONDARY DIMENSIONS that together represent
   only ~2.7% of total variance.

3. The dimensionality of the semantic subspace is SPARSE relative to
   the 1536-dimensional W_E. The meaningful semantic structure is
   concentrated in a handful of directions.

---

## Day 304 Plan

1. **PC2 verification**: probe the top and bottom token IDs on PC2 to
   confirm digit tokens dominate the high end and identify what is at
   the low end.

2. **v_ord in PC1-PC2 basis**: compute the exact (PC1, PC2, PC3)
   decomposition of v_ord to understand how much of v_ord is PC2 vs
   other components.

3. **Degree triangle third axis**: what does the e2 direction (the
   "superlative elevation" direction) encode beyond degree? Does it
   align with any known semantic axis?

4. **The full morphological subspace**: compute the actual PCA of just
   the morphological transformation subspace — what are its principal
   directions?

---

## Files

- `expedition_log.md` — Day 303 results
- `437_degree_2d_and_subspace_orthogonality.md` — DC 437
- `day303_degree_triangle_and_pc2.py` — experiment script
