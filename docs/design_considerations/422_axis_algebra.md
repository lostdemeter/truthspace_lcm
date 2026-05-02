# DC 422: Axis Algebra — The Morphological Vector Space

**Day 287 | Testing whether axes can be composed algebraically.
d_sup - d_comp ≈ comp→sup axis (cos=0.9791), same scale 0.26, 16/16
(100%) both ways. Axis subtraction is valid. The comparative→superlative
axis is perfectly symmetric (ratio=1.00). Additive axis combination
d_pl + d_past gives POS-sensitive output: verbs return past tense
(walk→walked, run→ran), nouns return plural (dog→dogs). The morphological
axis space is a LINEAR VECTOR SPACE where subtraction produces new valid
transformation axes and addition mixes competing transformations
weighted by word-type embedding.**

---

## The Subtraction Principle

### Setup

Two primitive axes from a common origin (base form):
- d_comp: base → comparative (fast → faster), scale=0.51
- d_sup: base → superlative (fast → fastest), scale=0.59

Both share the same source cluster. Their target clusters are distinct
but related (both are adjectival degree forms).

### Prediction

The transformation comparative → superlative should be encoded by:
```
d_comp→sup ≈ normed(d_sup - d_comp)
```

Because:
```
superlative = base + 0.59 * d_sup
comparative = base + 0.51 * d_comp
superlative - comparative ≈ 0.59 * d_sup - 0.51 * d_comp
```

After normalisation, this gives an approximate unit direction from
the comparative cluster toward the superlative cluster.

### Result

```
Direct comp→sup axis:   cos=0.9791 with composed axis
Direct scale:           0.26       Composed scale:      0.26
Direct accuracy:        16/16      Composed accuracy:   16/16
Direct reverse:         16/16      Composed reverse:    16/16
```

**97.9% alignment between direct and composed. Same scale. Same
accuracy. Axis subtraction is confirmed valid.**

### Why cos=0.9791 and not 1.0?

The 2.1% deviation arises from:

1. The d_comp and d_sup axes are NOT computed from exactly the same
   set of base words (some words have single-token comparative but
   not superlative, or vice versa). Different training sets introduce
   small deviations.

2. The axis normalisation: `normed(d_sup - d_comp)` is not the same
   as `normed(d_sup) - normed(d_comp)` in general. The magnitude
   information is lost during individual normalisations.

3. The mean chord direction for the direct comp→sup axis is optimised
   specifically for that transformation; the composed axis inherits
   small misalignments from both primitive axes.

Despite these deviations, the alignment is sufficient for 100%
accuracy at the same scale (0.26).

---

## The New Symmetric Axis: Comparative→Superlative

The comp→sup axis has ratio=1.00: the same scale (0.26) retrieves
the target correctly in both directions.

This is consistent with Day 286's rule:
> A semantic axis has ratio=1.00 if source and target word clusters
> have approximately equal neighbourhood density.

Comparative forms (faster, taller) and superlative forms (fastest,
tallest) have approximately equal frequency in English text (both
are derived adjective forms, moderately common, sparser than base
forms). The axis is symmetric for the same reason as the gender axis.

This gives us a **complete symmetry taxonomy** for morphological axes:

| Axis pair | Density relationship | ratio | Symmetric |
|---|---|---|---|
| masculine ↔ feminine | equal | 1.00 | YES |
| comparative ↔ superlative | equal | 1.00 | YES |
| singular ↔ plural | base slightly denser | 0.89 | NEARLY |
| base ↔ comparative | base much denser | 0.76 | NO |
| base ↔ superlative | base much denser | 0.52 | NO |
| base ↔ past | base much denser | 0.51 | NO |

**The pattern**: symmetry holds when BOTH endpoints are derived forms
(similar derivational status and frequency). Asymmetry appears when
one endpoint is a high-frequency base form.

---

## Additive Axis Composition: POS Sensitivity

Combining d_pl (singular→plural) and d_past (base→past) additively
at their optimal scales (sf_pl=1.24, sf_past=1.32):

```
ax_combined = normed(1.24 * d_pl + 1.32 * d_past)
```

Applied to various words:

```
dog   → ['dogs', 'Dog', 'Dogs']         plural wins  (noun)
walk  → ['walked', 'walks', 'Walk']     past wins    (verb)
cat   → ['cats', '(cat', 'Cat']         plural wins  (noun)
run   → ['ran', 'runs', '-run']         past wins    (irregular verb!)
go    → ['went', 'goes', 'went']        past wins    (irregular verb!)
```

**Nouns return plural; verbs return past tense.** The combined axis
retrieves whichever morphological form is closer in W_E to the
target of each specific word.

This is NOT confusion or noise — it is a meaningful geometric property:
- The W_E embedding of 'dog' is closer to 'dogs' (via d_pl direction)
  than to any past-tense form (there is no past tense of 'dog')
- The W_E embedding of 'walk' is closer to 'walked' (via d_past direction)
  than to 'walks' in the combined displacement

### What the POS Sensitivity Reveals

The additive composition `d_pl + d_past` acts as a POS DETECTOR:
- If the word is a noun, the plural axis wins → plural form
- If the word is a verb, the past axis wins → past form
- Ambiguous words (like 'walk', 'call') go to whichever is more
  strongly encoded in W_E

This is NOT programmed in — it emerges from the geometry of the
combined axis applied to each word's embedding position.

For irregular forms: run→ran, go→went. The past axis encodes irregular
morphology through the statistical pattern of irregular verbs in W_E.
The combined axis still retrieves the correct irregular past tense,
not the pluralised noun form.

### Implications

1. **Axis addition is a soft-selection operator**: adding two axes
   creates a combined axis that selects the transformation most
   geometrically accessible from each source word.

2. **POS information is implicit in word embeddings**: the W_E
   positions of 'walk' and 'dog' encode their primary morphological
   paradigm. Nouns are closer to their plural forms; verbs to their
   past forms, in the direction defined by each axis.

3. **Irregular morphology is encoded geometrically**: the past axis
   correctly retrieves 'ran' from 'run' and 'went' from 'go' even
   though these are irregular. The axis encodes the statistical
   geometry of the past-tense relationship, not a regular suffix rule.

---

## The Morphological Vector Space

Combining findings from Days 284–287:

```
Morphological axes form a vector space M where:
  - Each axis d_i is a unit vector in W_E
  - d_sup - d_comp ≈ d_comp→sup    (subtraction is valid)
  - d_pl + d_past = f(POS(word))   (addition is POS-gated)
  - -d_i = inverse transformation  (at same scale if symmetric)
  - scale(d_i) = f(neighbourhood density ratio)
```

### The Dimension Count

We have the following distinct primitive axes:
1. d_plural (singular→plural)
2. d_comparative (base→comparative)
3. d_superlative (base→superlative)
4. d_gender (masc→fem)
5. d_past (base→past)

From these 5 primitive axes, we can compose:
- d_comp→sup = d_sup - d_comp
- d_past→plural = ? (untested — would require verb-noun overlap)
- d_past→comp = d_comp - d_past ? (untested)
- etc.

In principle, any pairwise difference of two axes that share source
cluster should produce the target-to-target transformation axis.

### Known Limitations

1. Composition works for axes SHARING THE SAME SOURCE CLUSTER.
   d_comp - d_past is unlikely to work because comparative and past
   tense forms do not share source words (adjectives vs verbs).

2. The composed axis is always approximate (cos ≈ 0.96-0.98 with
   the direct axis), never exact. For dense domains with thousands
   of training pairs, the approximation might be better.

3. The POS-gated addition only gives a consistent output when the
   two competing axes pull in clearly different directions. If two
   axes are similar (cos > 0.9), the sum axis is dominated by both
   equally and would give inconsistent outputs.

---

## Files

- `expedition_log.md` — Day 287 results
- `421_morphological_reversibility.md` — scale ratio analysis (Day 286)
- `420_encode_decode_symmetry.md` — ENCODE=DECODE (Day 285)
- `415_axis_type_taxonomy.md` — axis types (Day 280)
