# DC 390: W_E Morphological Axis Architecture (Days 251–253)

**Days 251–253 | W_E encodes morphological paradigms as dedicated near-orthogonal
transformation axes. Each paradigm (adj_degree, plural, past_tense) occupies a
distinct geometric direction, all anti-correlated with the dominant PC1 (function/
structure) axis of the vocabulary.**

---

## The Architecture

```
W_E = 151,936 × 1536 matrix

PC1 (dominant variance axis):
  HIGH end: (, -, and, in, ., [, =, ,, to, for, a, or, &
            → punctuation, function words, operators, single letters
  LOW end:  emoji+whitespace, multilingual garbage, code artifacts
  Interpretation: 'versatility/co-occurrence-frequency' axis

Three morphological transformation axes (all anti-correlated with PC1):

  ADJ_DEGREE axis (mean_dir for pos→comp):
    PC1 alignment: -0.294
    TOP:  taller, hotter, louder, brighter, richer, thicker, bigger, nicer...
          → 100% genuine adj comparatives
    BOT:  base adj forms (long, fast, short, deep) + function words
    Δ(comp - base):  mean=+0.371, n=20/20 positive

  PAST_TENSE axis (mean_dir for base→past):
    PC1 alignment: -0.168
    TOP:  lasted, waited, belonged, existed, worked, wished, listened, wanted...
          → 100% genuine past tense verb forms
    BOT:  base verb forms (use, call, start, play, turn, work, go)
    Δ(past - base):  mean=+0.240, n=10/10 positive

  PLURAL axis (mean_dir for singular→plural):
    PC1 alignment: -0.247
    TOP:  noisy (multilingual tokens, code artifacts, some YEARS/DAYS)
    BOT:  base singular forms (man, day, hand, child, word, dog)
    Δ(plural - base): mean=+0.216, n=10/10 positive
    Note: weaker and noisier than adj and past_tense axes
```

---

## Axis Relationships

### Cross-cosines between paradigm axes
```
              ADJ    PLURAL  PAST_TENSE
ADJ           1.000  0.169   0.161
PLURAL        0.169  1.000   0.150
PAST_TENSE    0.161  0.150   1.000
```

Cross-cosines ≈ 0.15–0.17, well above random (0.026) but much smaller than 1.
The axes are **substantially distinct** but not perfectly orthogonal — they
share a common component of "moving away from function words" (all anti-PC1).

### Decomposition
Each paradigm axis can be decomposed as:
```
paradigm_axis ≈ α × (-PC1) + β × paradigm_specific
```
where:
- `-PC1` is shared across all paradigms (move away from function words)
- `paradigm_specific` is unique to each paradigm

The shared component (anti-PC1) is the reason all cross-cosines are ~0.15–0.17
rather than ~0.026 (random). The paradigm-specific components are near-orthogonal
to each other.

---

## The Semantic Content of Each Axis

### ADJ_DEGREE: The Comparative Axis (Strongest, Δ=0.37)

All adj_degree comparatives occupy the positive half of this axis. The axis
encodes the semantic concept "morphological degree augmentation."

- Moving along the adj axis: base → comparative → superlative
- Arc angle Ω = π/φ = 111.25° per step (from DC 385/388)
- The arc passes through the origin (co-circularity with O confirmed)

Why ADJ is the cleanest axis:
- Gradable scalar adjectives are semantically uniform (all encode intensity on a scale)
- The "-er" comparative suffix is nearly unambiguous for genuine adj comparatives
  (cos ≈ 0.5646 ± 0.006, the φ-cosine discriminator from Day 250)
- Single consistent transformation: "more of the property"

### PAST_TENSE: The Aspect Axis (Intermediate, Δ=0.24)

Past tense verbs cluster cleanly on the positive side. The axis encodes
"completed action" vs "ongoing/potential action":

- Top: verbs denoting completed actions: lasted, waited, existed, belonged
- Bottom: base verbs with diverse tense usage: use, call, start, play

### PLURAL: The Number Axis (Weakest, Δ=0.22)

The weakest and noisiest paradigm axis. The positive end is contaminated
by multilingual tokens sharing the "-s" suffix (German "frauen", "männer",
Polish "skóry", etc.). The base forms do correctly appear at the bottom.

Why PLURAL is noisier:
- The "-s" suffix is used in MANY languages and word types (not just English plurals)
- Plural forms overlap with other word classes (possessives, 3rd-person verbs)
- Plural transformation is less semantically uniform (count nouns have very varied contexts)

This noisiness explains the lower LOO accuracy for plural paradigm (39–72% vs 75–88% for adj).

---

## Axis Cleanness Predicts Retrieval Accuracy

```
Paradigm    Axis Δ   Top-20 purity   LOO accuracy (mean_dir)
adj_degree  0.371    100% comparatives    75-88%
past_tense  0.240    100% past tense      ~70-80%
plural      0.216    ~20% actual plurals  39-72%
```

The relationship is direct: the cleaner the transformation axis (higher Δ,
more homogeneous top-20), the better the mean_dir retrieval accuracy.
The plural axis is weak because the plural paradigm has more cross-lingual
and cross-class contamination in the vocabulary.

---

## Connection to Day 251: Global vs. Private

Day 251 showed that the adj_degree transformation direction is approximately
**global** — all semantic subclasses of adjectives (SIZE, TEMPERATURE, SPEED,
INTENSITY, TEMPORAL, QUALITY) share the same transformation axis with intra/cross
coherence ratio of only 1.20×.

Day 253 extends this: the axis is not just global within adj_degree, it's a
**dedicated semantic axis** of W_E that exclusively places adj comparatives
at the positive end. It's one of approximately 3 morphological axes (plus
possibly gender, superlative, and other paradigms) embedded in the 1536D
vocabulary space.

The "private plane" model from earlier (DC 385) needs this revision:
- The arc plane is approximately `span{emb(word), adj_axis}` for all adj
- The adj_axis is a single global direction in R^1536
- Individual variation is secondary noise (~20% of variance in the direction)

---

## The PC1 / Morphological Axes Relationship

```
PC1 (function/structure):   high for simple function words, low for complex tokens
adj axis (-0.29 from PC1):  high for morphologically complex adj comparatives
past axis (-0.17 from PC1): high for morphologically complex past tense forms
plur axis (-0.25 from PC1): high for morphologically complex plurals

Pattern: morphological complexity = moving away from PC1
The more morphologically complex the form, the lower its PC1 projection.
```

This suggests a universal principle: **morphological transformation axes
are all "anti-syntactic" in W_E** — they move tokens from the high-frequency,
syntactically-versatile region (high PC1) toward the morphologically-specific,
lower-frequency region (low PC1).

---

## Practical Implications

1. **Paradigm identification**: given a word pair (A, B), the paradigm can be
   identified by projecting (B-A) onto each morphological axis and finding
   the highest alignment. adj_degree, plural, past_tense have distinct axes.

2. **Axis-based retrieval**: rather than computing a mean_dir from training pairs,
   the transformation direction can be directly identified as the corresponding
   morphological axis (no training pairs needed, just the axis vector).

3. **Paradigm disambiguation**: when a word has multiple valid morphological
   forms (e.g., "fast" can become "faster" via adj axis OR "fasted" via past
   axis), project onto each axis to determine which transformation is requested.

4. **Axis cleanness as quality metric**: the Δ score (mean projection difference
   between source and target forms) serves as a quality metric for paradigm axes.
   Higher Δ → cleaner retrieval.

---

## Files

- `expedition_private_plane.py` — Day 251 semantic subclass analysis
- `expedition_fullvocab_adj.py` — Day 250 full vocabulary mining
- `private_plane.json` — Day 251 coherence matrix results
- `389_arc_direction_is_global.md` — arc direction is global (corrected)
- `388_phi_quantization_confirmed.md` — φ-quantization of adj_degree
- `387_we_arc_geometry_synthesis.md` — complete arc synthesis
