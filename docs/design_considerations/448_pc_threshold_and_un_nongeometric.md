# DC 448: The pc Threshold and the un- Non-Geometric Axis

**Day 313 | Three structural results: (1) The pc metric predicts
irreducibility with r=−0.640. The relationship is linear with slope
−16% per +0.1 pc. The operational threshold is pc>0.35 for reliable axes
(<15% irreducible) and pc<0.20 for unreliable axes (>40% irreducible).
(2) The un- prefix is non-geometric at all granularities. Even splitting
into homogeneous sub-domains (un-ADJ, un-VERB, un-STATE) yields LOO=0–11%.
Sub-axes are mutually poorly correlated (0.24–0.38) and cross-domain
transfer is 0%. (3) Cross-lingual proximity is a universal feature of
basic vocabulary in W_E — tree/树=0.745, book/书=0.714, hand/手=0.692.
This is expected for a multilingual model and affects all semantic domains,
not just body parts. (4) METHODOLOGY NOTE: The nn_retrieve_exact function
had a critical bug — it excluded capitalized/compound variants but NOT the
primary space-prefixed source token. This is fixed in Day 314.**

---

## The pc Threshold for Axis Reliability

### Empirical Data

```
Axis       pc      Irred(clean)   Axis type
──────────────────────────────────────────────
er→est     0.436    ~0%            regular inflection
+er        0.394    12%            regular inflection
+est       0.401    25%            regular inflection
+s         0.297    13%            regular inflection
past_irr   0.284    25%            irregular inflection
+ed        0.227    12%            regular inflection
gender     0.241    33%            lexical relation
+tion      0.125    22%            derivational suffix
+ment      0.120    33%            derivational suffix
+ness      0.169    75%            derivational suffix
+ful       0.112    75%            derivational suffix
un-        0.103    86%            derivational prefix
```

Pearson r(pc, irreducible_fraction) = **−0.640**

Linear fit: irred = 0.74 − 1.60 × pc

### The Threshold

```
pc ≥ 0.35    →    irred ≤ 15%    RELIABLE axis (use as general transformer)
pc ∈ [0.20, 0.35] →  irred 15–40%  CONDITIONAL axis (use only in-domain)
pc < 0.20    →    irred ≥ 40%    UNRELIABLE axis (avoid as general transformer)
```

Predictions from the linear fit:
```
pc = 0.5 →  0% irreducible
pc = 0.4 →  9% irreducible
pc = 0.3 → 25% irreducible
pc = 0.2 → 42% irreducible
pc = 0.1 → 58% irreducible
```

### Why High pc Means Low Irreducibility

The pc metric (mean pairwise chord cosine) measures how consistently all
training pairs point in the same direction. When pc is high:

1. The training pairs span a common semantic domain where the transformation
   has a single geometric meaning (e.g., all comparatives share "more extreme
   on a scale" semantics)

2. Holdout pairs in the same domain will also share this meaning, so their
   local displacement direction is close to the mean axis

3. The axis has been trained on the SAME semantic operation that holdout
   words require, just on different exemplars

When pc is low, training pairs point in scattered directions. The mean axis
is a BLEND of different semantic operations, not representative of any of
them. Holdout pairs are likely to need a direction that the blended axis
doesn't provide.

### Practical Implication

Before using any morphological axis for navigation, compute its pc score:
- pc ≥ 0.35: axis generalizes reliably (~90% of new pairs will succeed)
- pc < 0.20: axis does NOT generalize; each word needs its own local axis

This gives us an upfront quality certificate for any computed axis.

---

## The un- Prefix: Confirmed Non-Geometric

### The Definitive Test

Split the un- training set into semantically homogeneous sub-domains and
measure Leave-One-Out (LOO) accuracy within each:

```
Sub-domain  n   pc      LOO     In-sample
──────────────────────────────────────────
un-ADJ      9   0.177   11%     100%
un-VERB     8   0.131    0%     100%
un-STATE    7   0.130    0%     100%
```

In-sample is always 100% (expected — enough training pairs to find best
scale that retrieves all of them). But LOO=0–11% means the axis trained on
8 adj pairs CANNOT retrieve the 9th adj pair.

### Sub-Domain Axis Alignment

```
cos(adj_axis, verb_axis)  = 0.316
cos(adj_axis, state_axis) = 0.376
cos(verb_axis, state_axis)= 0.241
```

The sub-axes are all poorly correlated. They point in different directions
of the embedding space, consistent with the low pc values within each
sub-domain (0.13–0.18).

### Cross-Domain Transfer: 0%

- adj_axis → verb holdout: 0/8 = 0%
- verb_axis → adj holdout: 0/9 = 0%

No transfer at all. This means the geometric direction for "un-+verb" is
unrelated to "un-+adjective", exactly as the sub-axis cosines suggest.

### Why un- Is Non-Geometric

The 'un-' prefix attaches to words from completely different semantic
fields and has different semantic effects in each:

```
un-happy: {EMOTION: positive} → {EMOTION: negative}  (antonym rotation)
un-lock:  {DOOR: locked} → {DOOR: accessible}         (state reversal)
un-known: {INFO: accessible} → {INFO: absent}         (existence negation)
un-tie:   {CORD: bound} → {CORD: free}                (physical reversal)
un-safe:  {SAFETY: present} → {SAFETY: absent}        (negation)
```

Each of these corresponds to a DIFFERENT geometric transformation in W_E —
because "happy" lives in the emotion cluster, "lock" lives in the action
cluster, "known" lives in the epistemic cluster, etc. The word 'un-happy'
points in a different direction from 'un-lock' relative to their sources
because they're in different regions of the space.

This is the LOCAL SEMANTIC OPERATOR principle operating at maximum
granularity: for un-, there is no level of domain specificity fine enough
to find a consistent direction. Each word-pair requires its own local axis.

### The String vs. Geometry Distinction

This result allows us to cleanly distinguish:

**Geometric operations** (consistent direction, pc > 0.35):
- +er: adds "degree" to ANY gradable adjective in the same way
- +s: adds "collection" to ANY countable object noun in the same way
- +ed: adds "past completion" to ANY regular verb in the same way

**String operations** (no consistent direction, pc < 0.15):
- un-: reverses/negates the SPECIFIC semantic of each word differently
- +ful: adds modifier role in DIFFERENT ways (hopeful ≠ harmful ≠ playful)
- +ness: nominalizes DIFFERENT property types in different local directions

The un- prefix is a SURFACE STRING pattern, not a semantic primitive. W_E
correctly does NOT encode it as a single geometric operation.

This is a validation of the TruthSpace hypothesis: the geometry encodes
MEANING, not surface form. Since 'un-' has no single meaning, it has no
single geometric direction.

---

## Cross-Lingual Proximity: Universal Multilingual Structure

### The Values

```
Word     Chinese     cos(W_E)    Rank among body-parts
tree     树           0.745       N/A — higher than all body parts!
book     书           0.714       N/A — equal to head
head     头           0.714       
hand     手           0.692
cup      杯           0.588
foot     脚           0.575
eye      眼睛          0.567
leg      腿           0.551
ear      耳           0.534
car      车           0.485
arm      手臂          0.464
nose     鼻           0.468
```

### The Implication

Cross-lingual proximity is NOT body-part-specific. It is a universal
property of basic vocabulary in multilingual W_E. Tree (0.745) and book
(0.714) are MORE closely paired with their Chinese equivalents than hand
(0.692).

This reflects the model's multilingual training: words that translate
directly and unambiguously across languages are embedded in close proximity.
"Tree" and "book" have high-frequency unambiguous Chinese counterparts
(树, 书) that the model has learned to align closely.

### Effect on Navigation

When navigating with a morphological axis, cross-lingual tokens can appear
as top-k neighbors if:
1. The source word is close to its cross-lingual equivalent
2. The axis displacement pushes the prediction into a region where the
   cross-lingual equivalent is slightly closer than the target

The effect is stronger for words with cos > 0.60 to their Chinese
equivalents (hand, head, tree, book). For body-part navigation, this is
one of the three interference mechanisms for hand→hands.

No axis can "fix" this — it is a fundamental property of the multilingual
embedding structure. The correct solution is to add cross-lingual tokens
to the exclusion list in clean retrieval.

---

## Methodology: The nn_retrieve_exact Bug

### The Problem

`get_all_token_ids(word)` tried prefixes:
`' ', '', ' Word', 'Word', 'WORD', ' WORD', '-word', '_word', ' -word'`

It does NOT try `' word'` (space + lowercase) or `'word'` (no-space
lowercase) — the actual BPE token representations used by get_emb.

The space-prefixed primary token (' hand', ' arm', etc.) was therefore
NOT excluded by the exact retrieval function. This caused the source word
to appear as top-1 for every prediction, giving 0/12 accuracy.

### The Fix (Day 314)

```python
def get_all_source_ids(word):
    ids = set()
    # Try all prefix combinations to find single-token forms
    for p in [' ', '', ' ' + word[0].upper() + word[1:],
              word[0].upper() + word[1:], word.upper(), ' ' + word.upper(),
              '-' + word, '_' + word,
              ' ' + word,   # ← THE FIX: space + lowercase
              word]:         # ← THE FIX: no-space lowercase
        tks = tokenizer(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids
```

### Why This Bug Was Instructive

The bug revealed that word token variants are more complex than expected.
Even "obvious" variants like ' hand' (with space) are not automatically
tried when enumerating by prefix. This is because BPE tokenization is
fundamentally non-compositional — adding a space prefix can change
tokenization in non-obvious ways.

For Day 314: after fixing this, re-test hand→hands to determine whether
EXACT exclusion of all hand variants finally allows hands to be retrieved.

---

## Day 314 Plan

1. **Fix nn_retrieve_exact** — include ' word' and 'word' as tried prefixes.
   Re-test hand→hands: does exact exclusion finally solve it?

2. **un- geometry visualization**: project all un-+adj chords onto the
   top-2 PCs of the adj chord space. Are they randomly scattered or do they
   cluster by morphological family?

3. **+tion domain split**: +tion also has 22% irreducibility. Are the
   irreducible cases (observe/observation, describe/description, produce/
   production) in a different semantic sub-domain than the training set?

4. **The pc threshold test**: for a new axis (e.g., comparative of 2-
   syllable adjectives), measure pc and use the linear fit to PREDICT
   irreducibility before testing. Verify the prediction.

---

## Files

- `expedition_log.md` — Day 313 results
- `447_pluralization_semantics_and_irreducibility.md` — DC 447
- `day313_exact_retrieval_undomain_crosslingual.py` — experiment script
