# DC 423: Multi-Axis Composition — Orthogonality and Simultaneous Transformation

**Day 288 | The five morphological axes {gender, plural, comp, sup,
past} form a near-orthogonal basis in W_E (all pairwise cos < 0.17
except comp↔sup=0.453). Simultaneous application of two orthogonal
axes (gender + plural) achieves the same accuracy as sequential
application (5/6 both ways), demonstrating that orthogonal axes
compose without interference. Axis-based retrieval outperforms classic
word2vec analogy (100% vs 67% on gender pairs). Cross-domain axis
application is POS-selective: d_past−d_comp applied to verbs returns
past tense; applied to adjectives returns the same word. This confirms
that each word's embedding position in W_E selects the geometrically
relevant axis component.**

---

## Inter-Axis Orthogonality

```
          gender  plural  comp    sup     past
gender    1.000   0.110   0.058   0.087   0.020
plural    0.110   1.000   0.154   0.166   0.096
comp      0.058   0.154   1.000   0.453   0.088
sup       0.087   0.166   0.453   1.000   0.111
past      0.020   0.096   0.088   0.111   1.000
```

Most axes are nearly orthogonal (cos < 0.17). The exception is
comp↔sup = 0.453, which is expected: both axes start from the same
base adjective cluster and both target degree-form clusters, so their
mean directions must be correlated.

### Why Near-Orthogonal?

Each morphological transformation affects a different aspect of word
form:
- **gender**: lexical swap (king↔queen) — affects semantic gender
- **plural**: morphological suffix (+s) — affects count
- **comp**: adjectival degree (+er) — affects comparison
- **past**: temporal form (+ed/irregular) — affects tense

These are linguistically independent dimensions. W_E encodes them
as independent subspace directions because they are independent
features of word meaning. The near-orthogonality of the axes is a
**geometric reflection of linguistic independence**.

The comp↔sup correlation (0.453) arises because both are degree forms
derived from the same base adjectives. They are not fully independent:
superlative implies comparative (if X is fastest, then X is faster
than all). The partial correlation in W_E captures this implication.

---

## Simultaneous vs Sequential Transformation

### The Test

Apply gender axis AND plural axis to produce feminine plural forms:
king → queens, man → women, boy → girls, etc.

```
Sequential:    king → queen (gender) → queens (plural)  [5/6]
Simultaneous:  king → queens (gender + plural in one step) [5/6]
```

Best simultaneous scales: scale_g=0.61, scale_pl=1.10

**Same accuracy. Same pairs succeed and fail.**

### Why Simultaneous Works

The gender and plural axes are nearly orthogonal (cos=0.110). When
added:

```
d_combined = 0.61 * d_gender + 1.10 * d_plural
```

The two components act independently in their respective subspaces.
The combined displacement moves orthogonally in both the gender and
plural dimensions simultaneously, landing near the feminine-plural
token without passing through the intermediate feminine-singular
token.

### The Failure Case: 'god'

`god → goddesses` fails both sequential and simultaneous methods.
Cause: 'goddesses' is a multi-token word in the BPE vocabulary (it
splits into multiple subword tokens). This is not a failure of the
multi-axis architecture — it is the same invisible ceiling from
DC 419 applying to simultaneous composition as well as single-hop.

### Implication for TruthSpace

**Multiple orthogonal morphological transformations can be composed
into a single geometric step.** This extends the TruthSpace principle:
not just "encoding IS decoding", but "two independent encodings can
be applied simultaneously without interference".

This is geometrically equivalent to computing a vector sum in an
orthogonal basis: the result is exact if the basis vectors are
exactly orthogonal, and approximately exact if they are nearly
orthogonal (cos < 0.15).

---

## Axis-Based Retrieval vs Classic Word2Vec Analogy

### Comparison

```
Method                          Gender test    Mixed test
Axis-based (trained d_gender)   6/6  (100%)   ---
Classic analogy (a-b+c)         4/6  (67%)    7/14 (50%)
```

### Why Axis-Based is Superior

The classic word2vec analogy `king - man + woman = ?` uses a SINGLE
training pair (man, woman) as the axis direction. The axis-based
method uses ALL training pairs and takes the mean direction.

For the single-pair analogy:
```
d_analogy = woman - man  (one chord, not normalized)
```

For the axis-based method:
```
d_axis = mean(normed(woman-man), normed(daughter-son), ..., normed(12 pairs))
```

The mean-normalised direction is more robust. Artefacts in any single
pair (e.g., 'man' being loaded with masculine-connotation words other
than gender, or 'woman' having feminist movement associations) are
averaged out over many pairs.

### Failure Modes of Classic Analogy

1. **Capitalisation mismatch**: `queens - men + women = Kings` (capital
   K kills the match). The analogy arithmetic doesn't control for
   capitalisation.

2. **Multilingual interference**: `father - man + woman = 父亲` (Chinese
   character for "father" is in top-3). The analogy vector points into
   a neighbourhood occupied by multilingual synonyms.

3. **Low-frequency targets**: `god - man + woman = god` (goddess is
   less frequent than god in training data, so the analogy vector
   overshoots or undershoots the goddess neighbourhood).

The axis-based method mitigates all three by using the mean direction
over many pairs, which suppresses the influence of any single
problematic word.

---

## POS-Selective Axis Application

When d_past − d_comp is applied to various words:

```
Adjectives:  fast → Fast (same word)   slow → Slow
Verbs:       walk → walked             run → ran    go → went
```

**The axis is POS-selective**: verbs respond to d_past, adjectives
respond to neither (both axes cancel on non-participating words).

### The Mechanism

Each word's embedding `e_w` has a projection onto each axis:

```
proj_past(e_w) = e_w · d_past
proj_comp(e_w) = e_w · d_comp
```

For a verb like 'walk':
- `proj_past(walk)` is large (verbs are close to the past-tense axis direction)
- `proj_comp(walk)` is small (verbs are NOT close to the comparative axis)

So `e_walk + d_past − d_comp ≈ e_walk + d_past` → walked.

For an adjective like 'fast':
- `proj_comp(fast)` is large
- `proj_past(fast)` is small

So `e_fast + d_past − d_comp ≈ e_fast − d_comp` → pushes slightly
away from comparative (back toward the base, which is 'fast' itself).
The displacement is insufficient to escape the 'fast' neighbourhood.

### Implication

**The W_E space is "smart" about which axes are relevant to each word.**
A word's embedding position encodes its morphological participation:
verbs are positioned in W_E to respond to the past-tense axis; nouns
to the plural axis; adjectives to the comparative/superlative axes.
This is not programmed — it emerges from the distributional geometry
of W_E training.

This is the TruthSpace "structure IS information" principle: the
word's geometric position encodes which morphological transformations
it participates in.

---

## The Morphological Basis: Summary

The five morphological axes form a near-orthogonal basis:

```
B_morph = {d_gender, d_plural, d_comp, d_sup, d_past}
```

Properties:
1. **Near-orthogonal**: all pairwise cos < 0.17 (except comp↔sup=0.453)
2. **Composable**: subtraction (DC 422) and addition both valid
3. **Reversible**: all axes 100% reversible in accuracy (DC 421)
4. **Symmetric or asymmetric**: gender and comp↔sup symmetric (ratio=1.00);
   plural, comp, sup, past asymmetric (ratio=0.51-0.89)
5. **POS-selective**: each word responds only to its relevant axes
6. **Superior to analogy**: axis-based retrieval > classic a-b+c (100% vs 67%)

### Open Questions for Day 289+

1. **Generalisation**: do axes transfer to unseen words? (predicted YES)
2. **Larger basis**: are there more orthogonal morphological axes?
   (adverb, gerund, infinitive, past-participle, etc.)
3. **Semantic axes**: do the findings extend to semantic relations?
   (nat→lang, person→nat) — partial evidence from Days 281-285.
4. **Basis completeness**: can any morphological transformation be
   expressed as a linear combination of the five primitive axes?

---

## Files

- `expedition_log.md` — Day 288 results
- `422_axis_algebra.md` — axis subtraction (Day 287)
- `421_morphological_reversibility.md` — scale ratios (Day 286)
- `420_encode_decode_symmetry.md` — ENCODE=DECODE (Day 285)
