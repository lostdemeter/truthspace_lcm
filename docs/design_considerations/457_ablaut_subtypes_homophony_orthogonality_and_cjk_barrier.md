# DC 457: Ablaut Sub-Types, Homophony Orthogonality, and the CJK Density Barrier

**Day 322 | Four discoveries: (1) All five English ablaut verb classes (umlaut,
see/saw, take/took, sing/sang, break/broke) are POSITIVELY CORRELATED in
embedding space (cosines 0.27-0.66), sharing one global 'strong past' axis
direction. The take/took class is the most coherent (pc=0.320) and the ALL_ablaut
axis achieves 7/10=70% wild-card accuracy on unseen irregular verbs. (2) The -al
suffix creates TWO ORTHOGONAL axes: relational (nation→national) vs nominal
(arrive→arrival), with cos=-0.020. This is the first confirmed ORTHOGONAL
homophony split. (3) EN→ZH and EN→JA translation axes have pc≈0.09 (the axis
DIRECTION exists) but in=0% — the Chinese/Japanese character space is too dense
to isolate individual targets via displacement. cos(EN→ZH, EN→JA)=0.794 but
both CJK axes are factual_local in behavior. (4) The predictor reaches 83%
(25/30); the remaining 5 errors all cluster at the pc=0.05-0.22 boundary and
require a 4th irred=100% flag to separate factual_local from semantic_diverse.**

---

## Ablaut Sub-Types: One Global Axis, Five Sub-Directions

### Per-Class Results

```
Class            n   pc      LOO%   description
umlaut(-ght)     6   0.166   50%    go/went, think/thought, bring/brought
see/saw type     6   0.204   67%    see/saw, say/said, come/came
take/took        8   0.320   88%    take/took, give/gave, get/got, know/knew
sing/sang        4   0.110   25%    sing/sang, ring/rang, drink/drank, swim/swam
break/broke      7   0.170   57%    break/broke, ride/rode, write/wrote, drive/drove
ALL_ablaut      31   0.185   71%    all five classes combined
```

### The Sub-Type Cosine Matrix

```
              umlaut   see/saw  take/took  sing/sang  break/broke
umlaut           —     +0.471    +0.559     +0.275     +0.364
see/saw                  —      +0.658     +0.428     +0.516
take/took                           —      +0.428     +0.591
sing/sang                                    —       +0.408
break/broke                                             —
```

All 10 pairwise cosines are **positive**. This is the key finding: despite
the phonological diversity of the vowel changes, every ablaut class moves
verbs in the SAME GENERAL DIRECTION in embedding space — from present-tense
action semantics toward past-tense completion semantics.

### The Ablaut Hierarchy

**take/took class** is the most coherent (pc=0.320) and most correlated with
other classes (mean inter-class cos=0.559). Why? These verbs are the core
English strong verbs, the most frequent irregular forms. They're learned
earliest and embedded most stably. The vowel change (a→oo/ave/ot) is diverse
but the semantic shift is maximally consistent.

**sing/sang class** is least coherent (pc=0.110) and least correlated with
others. The /i/→/a/ ablaut is phonologically minimal (one vowel step), and
the verb cluster (sing, ring, drink, swim) is semantically diverse (music,
sound, liquid, motion).

The ALL_ablaut axis (pc=0.185) captures the common "strong past" direction
by averaging across classes. Its LOO=71% demonstrates that this averaged axis
still generalizes — the global direction is learnable despite sub-class diversity.

### Wild-Card Paradigm Completion: 70%

```
✓ steal  → stole    (break/broke class)
✓ freeze → froze    (break/broke class)
✓ speak  → spoke    (break/broke class)
✗ find   → found    (finds instead)
✗ bind   → bound    (binds instead)
✗ wind   → wound    (winds instead)
✓ fly    → flew     (take/took cluster)
✓ draw   → drew     (take/took cluster)
✓ fall   → fell     (see/saw cluster)
✓ feel   → felt     (see/saw cluster)
```

The three failures (find/found, bind/bound, wind/wound) belong to the
**-nd→-ound class** — a sub-group defined by the -nd suffix pattern:
- find/found, bind/bound, wind/wound, grind/ground

This class is NOT represented in any training set. The axis can't generalize
to it because the "strong past" direction from 'find' leads toward 'broke'-type
targets (stole, drove), not toward 'found'.

This is a **missing class** problem: the ablaut axis handles 5 of 6 major
strong verb classes, but not the -nd/-ound class.

### Implication: Morphological Paradigms Are Explicitly Geometric

W_E has encoded all major English ablaut paradigms as geometric structure.
The strong verb paradigm (as a whole) is as coherent as the regular plural.
This confirms that **morphological paradigms ARE geometric directions** — not
patterns in surface form, but directions in semantic space that correspond
to systematic meaning shifts.

---

## Homophony Orthogonality: The -al Split

### The Result

```
+al_relational: pc=0.117, LOO=50%, irred=0%   → phonol_scatter-allomorph
+al_nominal:    pc=0.182, LOO=67%, irred=50%  → phonol_scatter

cos(+al_relational, +al_nominal) = -0.0195 ≈ 0
```

**Near-perfect orthogonality**. The two -al axes occupy perpendicular
directions in the 1536-d embedding space.

### Why Orthogonal?

The two -al derivations operate on DIFFERENT source and target clusters:

**+al_relational** (nation→national, region→regional):
- Sources: abstract nouns from the GEOGRAPHY/POLITICS/SOCIETY cluster
- Targets: attributive adjectives near their source nouns
- Direction: within the abstract-noun cluster, toward its attributive subspace

**+al_nominal** (arrive→arrival, propose→proposal):
- Sources: verbs from the ACTION/EVENT cluster
- Targets: nominalized events near the PROCESS/RESULT cluster
- Direction: from verb space toward nominalized-event space

These two directions are orthogonal because the source clusters are
orthogonal: the GEOGRAPHY noun cluster and the ACTION verb cluster
are in completely different regions of the 1536-d semantic space.

### Comparison with -er and -ly Splits

```
Homophonous pair        cos      split type
+al_rel vs +al_nom    -0.020    orthogonal (complete split)
+ly_adv vs +ly_adj    +0.156    partial split (mild correlation)
+er_comp vs +er_noun   ?        not measured yet
```

The -al split is the STRONGEST homophony separation found. The -ly split
is partial: adverbial -ly (quickly, slowly) and adjectival -ly (friendly,
lovely) share a mild directional correlation, likely because both produce
words in the "manner/quality" semantic neighborhood.

### The Homophony Principle

**Suffix homophony creates orthogonal axes when the source categories are
orthogonal.** The -al suffix disambiguates geometrically because:
- relational adjectives → source = noun category, perpendicular to verb category
- nominal derivation → source = verb category

The orthogonality of the GRAMMATICAL CATEGORIES (noun vs verb) propagates
to orthogonality of the GEOMETRIC AXES. This is further evidence that W_E
encodes grammatical structure as geometric structure.

---

## The CJK Density Barrier

### The Phenomenon

```
EN→ZH: 40/40 pairs single-token, pc=0.099, in=0%, irred=100%
EN→JA: 23/23 pairs single-token, pc=0.088, in=0%, irred=100%
EN→ES: 22/40 pairs single-token, pc=0.073, in=73%, irred=91%
```

Chinese and Japanese translations have **pc>0** (the axis direction exists),
but **in=0%** at every scale. The displacement moves into the CJK cluster but
cannot isolate any individual character.

Why does EN→ES work (73%) but EN→ZH fails (0%)?

### The Density Hypothesis

The key difference is **target cluster density**:

**Spanish words**: scattered throughout clean token space. 'Casa' is an
isolated token near 'house'/'home' but not particularly close to other
Spanish words (its nearest neighbors include English words with shared roots:
'casa' ↔ 'house', 'mansion', 'casita'). Each Spanish token is
semi-isolated, making displacement navigation possible.

**Chinese characters**: all single Hanzi characters (水,火,日,月,山...)
cluster TOGETHER in a dense sub-region of the embedding space. They're
all "CJK characters" to the tokenizer and all appear in similar positions
(beginning of sentences, in proper nouns, in compound words). This shared
positional pattern creates a dense cluster where no single character can
be isolated by a uniform displacement.

### CJK Subspace Structure

```
cos(EN→ZH, EN→JA) = +0.794   [nearly identical directions]
cos(EN→ZH, EN→ES) = +0.205   [different: CJK vs European]
cos(EN→JA, EN→ES) = +0.246   [different: CJK vs European]
```

ZH and JA share almost the same translation axis because Japanese kanji
and Chinese hanzi share the SAME tokenizer tokens. When Qwen2 encounters
日, 月, 水, it uses the same token IDs regardless of context. The embedding
of 日 is determined by its combined frequency in Chinese AND Japanese text.

The CJK-European boundary (cos≈0.22) marks a fundamental division in
Qwen2's embedding space between:
- **European cluster**: isolated tokens with unique Latin/Roman roots
- **CJK cluster**: dense Hanzi/Kanji subspace with shared character encodings

### Implication for Translation Geometry

Translation axis type depends on target language density:
- **European languages** (ES, FR, DE): factual_local but IN can be >70%
  for very common words — these are sparsely distributed in token space
- **CJK languages** (ZH, JA): factual_local with in=0% — dense character
  space blocks navigation

Both types have irred≈100% (no generalization to holdout), but CJK
additionally fails in-sample. This suggests the geometry of the CJK
sub-embedding is qualitatively different from European vocabulary.

---

## Predictor v3: The 83% Fix

### Current Errors

```
Axis         pc      LOO%   irred%   predicted               true
+able        0.220    0%     60%     borderline              semantic_diverse
pres         0.165    0%    100%     semantic_diverse        factual_local
EN→DE        0.101    0%    100%     semantic_diverse        translation
adj_ant      0.055   30%     90%     translation/fact_local  polar_local
country→cur  0.173    0%     33%     borderline              semantic_diverse
```

### Fix: irred=100% as Decisive Flag

For the pres and EN→DE errors:
```
pc=0.10-0.20, LOO<50%, irred=100% → factual_local or translation
                       irred=60-80% → semantic_diverse
                       irred<60% → borderline or morph_moderate
```

The hard boundary between factual_local and semantic_diverse is **irred≈90%**:
- semantic_diverse: irred=60-90% (can sometimes be retrieved)
- factual_local: irred=90-100% (never retrievable by generalization)

For the adj_ant error: adj_ant has LOO=30% which falls into the
"translation-partial" zone by the current rule (pc=0.05-0.10, LOO>15%).
Fix: add a secondary classifier — if LOO>15% AND irred>80% → polar_local-partial.

### Updated Decision Tree

```
IF pc > 0.35:
  → morph_uniform OR relational_geom

ELIF pc > 0.20:
  IF LOO > 50%: → morph_moderate (irred<30%) OR phonol_scatter (irred<30%)
  ELIF irred < 30%: → morph_moderate
  ELIF irred > 60%: → semantic_diverse
  ELSE: → borderline

ELIF pc > 0.10:
  IF LOO > 50%: → phonol_scatter
  ELIF irred >= 90%: → factual_local (irred=100%) OR translation (cross-lingual?)
  ELIF irred > 60%: → semantic_diverse
  ELIF irred < 20%: → phonol_scatter-allomorph
  ELSE: → borderline

ELIF pc > 0.05:
  IF irred >= 90% AND LOO < 15%: → translation OR factual_local
  ELIF LOO > 15% AND irred > 80%: → polar_local-partial
  ELIF LOO > 15%: → borderline antonym/phonol_scatter
  ELSE: → polar_local

ELSE:
  IF LOO > 15%: → polar_local-partial
  ELSE: → polar_local
```

Expected accuracy with this fix: 28-29/30 ≈ 93-97%.

---

## Day 323 Plan

1. **Predictor v3 benchmark**: apply the fixed decision tree to all 30 axes.
   Verify accuracy improves to ≥90%.

2. **-er homophony cosine**: measure cos(+er_comparative, +er_noun) to confirm
   the split is also near-orthogonal (like -al) or merely divergent.

3. **-nd/-ound ablaut class**: add wind/found/bound pairs to training. Does
   the wild-card accuracy improve to 9/10 when -nd class is represented?

4. **CJK density test**: measure the mean pairwise cosine within the CJK
   character cluster vs the European word cluster. Quantify the density
   difference that causes the navigation failure.

5. **Language subspace map**: build a complete picture of the translation axes
   by measuring all pairs: cos(ZH,JA), cos(ZH,ES), cos(ZH,FR), cos(ZH,DE),
   cos(ES,FR), cos(ES,DE), cos(FR,DE). Visualize the language subspace.

---

## Files

- `expedition_log.md` — Day 322 results
- `456_irregular_morphology_homophony_split_and_predictor_validation.md` — DC 456
- `day322_ablaut_homophony_benchmark_zh_ja.py` — experiment script
