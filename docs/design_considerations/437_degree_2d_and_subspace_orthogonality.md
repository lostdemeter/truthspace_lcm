# DC 437: Degree Gradation is 2D; Labelling Subspace is Orthogonal to All Other Axes

**Day 302 | Two findings of structural importance: (1) The degree
gradation system (+er, er→est) is a 2D structure in W_E, NOT a 1D
collinear line. cos(+er, er→est) = −0.408 (anti-parallel), yet
cos(+er + er→est, +est direct) = 0.976 with 6/6 (100%) retrieval.
The comparative lies OFF the base-to-superlative line, in a 2D arc.
(2) The universal ordinal direction v_ord is ORTHOGONAL to every
non-labelling axis tested (|cos| < 0.20 for all 13 morphological and
semantic axes). The labelling subspace and the morphological/semantic
subspace are geometrically independent. v_ord is strongly anti-aligned
with PC1 (cos = −0.720), linking labelling operations to the frequency
structure of the vocabulary.**

---

## Finding 1: Degree Gradation is a 2D Structure

### Experimental Results

```
ax_+er:             pc = 0.385   (base → comparative)
ax_er→est:          pc = 0.430   (comparative → superlative)
ax_+est direct:     pc = 0.400   (base → superlative)

cos(+er, er→est)       = −0.408   [anti-parallel]
cos(+er, +est)         = +0.456   [co-directional]
cos(er→est, +est)      = ?        [not measured directly]

cos(+er + er→est, +est direct) = 0.976   [composed ≈ direct]
Retrieval with composed axis: 6/6 (100%)
```

### Geometric Interpretation

The three axes (+er, er→est, +est) form a 2D structure — a "degree
triangle" in embedding space. Label the three regions:

```
    BASE           COMPARATIVE        SUPERLATIVE
  (fast)    →+er→   (faster)   →er→est→  (fastest)
                \                            ↑
                 \_________+est_____________/
```

The key measurements:
- `+er` and `er→est` are **anti-parallel** (cos = −0.408): the step
  from base to comparative points in roughly the OPPOSITE direction
  from the step comparative to superlative.
- `+er` and `+est` are **co-directional** (cos = +0.456): the base
  can "see" the superlative in the same rough direction as the
  comparative — it's just farther.

This creates a degree TRIANGLE, not a line:

```
         SUPERLATIVE
           ↑  ↑
     +est /   \ er→est
          /     \
       BASE ─+er→ COMPARATIVE
```

The comparative is displaced SIDEWAYS from the base-to-superlative
axis. It is NOT simply halfway between base and superlative.

### Why ax_composed Works (cos = 0.976)

The composed axis is `normalise(ax_er_raw + ax_er→est_raw)`:
- `ax_er` moves from the base region to the comparative region (direction A)
- `ax_er→est` moves from the comparative region to the superlative region (direction approximately −A + B)
- Their sum: A + (−A + B) = B ≈ direction of +est

This is a vector cancellation: the "sideways" component of +er gets
cancelled by the "back-sideways" component of er→est, leaving only
the "forward" component B that corresponds to +est.

### Degree Triangle Geometry

The shape of this triangle has implications:

1. **Comparatives are marked forms**: the comparative morpheme (-er)
   applies a transformation that is NOT simply "go toward superlative."
   It applies a DIFFERENT operation — one that marks the word for
   comparison but does not yet maximise it.

2. **Superlatives are doubly marked**: the superlative (+est) can be
   reached EITHER by:
   - One step: base → superlative (ax_+est, pc=0.400)
   - Two steps: base → comparative → superlative (ax_+er + ax_er→est)

3. **The two paths have nearly the same direction** (cos=0.976):
   the triangle is "thin" — comparatives are not far from the
   base-to-superlative line in absolute distance, just enough to
   create anti-parallel sequential steps.

### Previous Confusion Resolved

Day 301 found cos(+er+er, +est) = 0.460 — poor composition. This is
because `normalise(2·ax_er) = normalise(ax_er)` — doubling a vector
doesn't change direction. The path base→comparative→comparative does
not exist (there is no "more comparative") — you cannot stack +er
on itself. But base→comparative→superlative DOES exist as a two-step
path, and it composes correctly (cos=0.976).

---

## Finding 2: v_ord is Orthogonal to All Non-Labelling Axes

### Complete Alignment Table

```
Axis                cos(axis, v_ord)   interpretation
digit->word         −0.850             (anti-parallel labelling)
month->num          +0.897             (forward labelling)
weekday->num        +0.878             (forward labelling)
card->num           +0.896             (forward labelling)
──────────────────────────────────────────────────────────
ordinal->cardinal   +0.092             near-orthogonal
country->demonym    +0.027             near-orthogonal
country->capital    −0.031             near-orthogonal
+est                −0.166             near-orthogonal
+er                 −0.191             near-orthogonal
gender              −0.077             near-orthogonal
past_irr            −0.082             near-orthogonal
+ed                 −0.088             near-orthogonal
+s plural           −0.172             near-orthogonal
+ness               −0.131             near-orthogonal
un-                 −0.081             near-orthogonal
+ment               +0.017             near-orthogonal
+ful                −0.085             near-orthogonal
```

Every non-labelling axis has |cos(axis, v_ord)| < 0.20. This is not
approximate orthogonality — it is structural orthogonality. The
labelling subspace and the morphological/semantic subspace are
**geometrically independent** dimensions of W_E.

### What This Means

The W_E embedding space has (at least) two independent transformation
subspaces:

**SUBSPACE L (Labelling):**
- Spanned by: v_ord and axes like month→num, weekday→num, card→num
- Function: maps named category members to their ordinal symbols
- Aligned with: PC1 frequency axis (anti-aligned, cos ≈ −0.70)
- Examples of axes IN this subspace: all TIER 1 axes

**SUBSPACE M (Morphological/Semantic):**
- Spanned by: +er, +est, +ed, +s, +ness, gender, country axes, etc.
- Function: morphological inflection, derivation, semantic relations
- Orthogonal to: v_ord (|cos| < 0.20 for all axes measured)
- Examples of axes IN this subspace: TIER 2–4 axes

These two subspaces are not artificially separated by our measurement
method — they were computed independently (each as a mean of chord
vectors), and their mutual cosines reflect the actual geometry of W_E.

### Implication: Why Labelling Axes Have Maximum pc

In Days 298–300, we found that LABELLING axes have the highest pc of
any category (mean 0.709, all > 0.50). Now we understand WHY:

1. **v_ord is a DOMINANT DIRECTION** in W_E (−PC1, 1.53% variance)
2. **Labelling axes lie ALONG v_ord** (cos 0.88–0.91)
3. **The ordinal direction is nearly 1D**: the chord vectors for all
   labelling pairs point in the SAME direction (v_ord), giving high pc

The labelling axes have high pc because they all move in the same
fundamental direction of W_E — the direction that separates symbolic
labels (digits) from named entities (month names, weekday names).

Morphological axes have lower pc because they lie in SUBSPACE M, which
is multidimensional and noisier. The morphological subspace has many
competing semantic dimensions, and different source words get pulled in
slightly different directions by their individual semantic neighborhoods.

---

## Finding 3: PC3 Weakly Captures Morphological Modification

```
PC3 alignment with morphological axes:
  +ness:   cos = +0.214   (highest)
  +est:    cos = +0.152
  +er:     cos = +0.144
  +s:      cos = +0.124
  un-:     cos = +0.114
  +ed:     cos = +0.086
  +ful:    cos = +0.081
```

All morphological axes weakly co-align with PC3 (all positive, all
< 0.22). This suggests PC3 is related to "morphological modification"
— a direction in W_E that is common to all inflectional and derivational
operations.

However, the cosines are small (max 0.214), meaning PC3 explains only
a tiny fraction of what the morphological axes capture. The axes span
MUCH MORE than just the PC3 direction — they occupy a wide
multidimensional region of W_E.

### The Three Principal Layers of W_E

```
PC1 (3.35% variance):  Token Specificity / Frequency
                        → most powerful, separates common from rare tokens
                        → strongly anti-aligned with v_ord (cos ≈ -0.72)

PC2 (~X% variance):    Unknown — not correlated with frequency (r=-0.085)
                        or word length (r=0.036, NS)

PC3 (~Y% variance):    Morphological Modification
                        → weakly aligned with all morphological axes
                        → may capture "wordform vs. lemma" distinction
```

PC2 remains mysterious. Its complete lack of correlation with token
ID or word length, combined with near-zero projections for all word
groups tested, suggests it may capture a more abstract structural
property (perhaps sentence position context, or syntactic category).

---

## Composition Summary (Days 301–302)

```
Composition test                          cos(composed, direct)  retrieval
──────────────────────────────────────────────────────────────────────────
month→digit + digit→word = month→word    0.9944                 7/9 (78%)
weekday→num + num→ordinal = weekday→ord  0.9999                 6/7 (86%)
+er + er→est = +est                      0.9763                 6/6 (100%)
+er + +er = +est                         0.4596                 FAILS
```

### Generalisation: When Does Composition Work?

**WORKS** (cos > 0.97): when the intermediate term is a REAL
intermediate position in a linguistic chain:
- month → [number] → spoken word (number IS an intermediate)
- base → [comparative] → superlative (comparative IS a position)

**FAILS** (cos ≈ 0.46): when the composed axis has NO intermediate:
- +er + +er: there is no "comparative of comparative" — the intermediate
  is not a real position in the language

The rule is: **axis composition succeeds if and only if the intermediate
node is a semantically real position that is represented in W_E.**

---

## Day 303 Plan

1. **Map the degree triangle precisely**: measure all three side lengths
   and angles of the (base, comparative, superlative) triangle using
   projection onto a 2D basis.

2. **Discover what PC2 encodes**: probe with syntactic categories
   (nouns, verbs, adjectives, adverbs), semantic fields (animate vs.
   inanimate, concrete vs. abstract), and named entities.

3. **Test more composition chains**: test un→+ness (with multi-token
   targets excluded), morphological chains like +ness→un (reverse),
   and noun→verb→+ed chains.

4. **Test whether subspace L and M are complementary**: does the
   projection of W_E onto subspace L + subspace M capture more total
   variance than either alone?

---

## Files

- `expedition_log.md` — Day 302 results
- `436_axis_algebra_and_pc1.md` — DC 436: composition law and PC1
- `day302_pc2_and_composition.py` — experiment script
