# DC 427: Semantic Axis Generalisation — Zero-Shot 100% and the Inflated-pc Problem

**Day 292 | Semantic axis holdout tests: country→demonym (pc=0.598)
generalises to 12/15 standard holdout AND 10/10 zero-shot rare countries
(100%: Finland→Finnish, Belgium→Belgian, Hungary→Hungarian,
Croatia→Croatian, Serbia→Serbian, Iceland→Icelandic, Wales→Welsh,
Scotland→Scottish). country→capital (pc=0.335) generalises to 11/14
(79%) but fails on 'surprise capitals' (Toronto instead of Ottawa,
Melbourne instead of Canberra, Zurich instead of Bern). country→lang
fails holdout (42%) despite high pc=0.529 — the pc is INFLATED by
training data conflation: all training countries use the same word for
demonym and language. Antonym cluster (speed) has pc=0.387 but 0/2
holdout — also pc-inflation via shared target. Mini-train curves: even
2 pairs for country→demonym give 80% generalisation.**

---

## country→demonym: The Most Generalisable Semantic Axis

### Results

```
Training: 10 countries (France, Germany, Spain, Italy, Japan,
          China, Russia, Egypt, Brazil, Portugal)
Standard holdout: 12/15 (80%)
Zero-shot rare:   10/10 (100%)
```

### Zero-Shot 100% — The Key Result

Axis trained on 10 common Western European + Asian countries correctly
predicts demonyms for countries NOT in the training set:

```
Finland  → Finnish     [HIT]   (not in training)
Belgium  → Belgian     [HIT]   (not in training)
Hungary  → Hungarian   [HIT]   (not in training)
Romania  → Romanian    [HIT]   (not in training)
Croatia  → Croatian    [HIT]   (not in training)
Serbia   → Serbian     [HIT]   (not in training)
Iceland  → Icelandic   [HIT]   (not in training)
Wales    → Welsh       [HIT]   (not in training)
Scotland → Scottish    [HIT]   (not in training)
Ireland  → Irish       [HIT]   (not in training)
```

This is the strongest generalisation result across all experiments.
The country→demonym axis has learned the RULE for forming demonyms,
not the specific training-pair mappings.

### Why It Works

The demonym transformation is near-morphological:
- Most European demonyms follow suffix patterns (-an, -ish, -ian, -ese)
- Country names form a tight semantic cluster in W_E
- The axis is effectively the mean direction from "country name cluster"
  to "demonym cluster", and every new country maps along this direction

The high pairwise cosine (0.598) reflects genuine geometric consistency:
the displacement from France→French, Germany→German, Spain→Spanish
all point in approximately the same direction in W_E because they all
apply the same "add demonym suffix" operation.

### Mini-Train Efficiency

```
n=2 pairs  →  80% holdout generalisation
n=5 pairs  →  88% holdout generalisation
n=10 pairs →  80% holdout generalisation
```

**2 training pairs give 80% generalisation.** This is the highest
data-efficiency result in the entire study, matching the comparative
(+er) axis at 91%/2 pairs.

---

## country→capital: Factual Knowledge Limits

### Results

```
Training:  10 countries
Holdout:   11/14 (79%)
```

### The Hits

```
Greece    → Athens      [HIT]
Poland    → Warsaw      [HIT]
Sweden    → Stockholm   [HIT]
Norway    → Oslo        [HIT]
Argentina → Buenos      [HIT]   (Buenos Aires truncated to Buenos)
Mexico    → Mexico      [HIT]   (Mexico City == country name)
Netherlands → Amsterdam [HIT]
Portugal  → Lisbon      [HIT]
Ireland   → Dublin      [HIT]
Austria   → Vienna      [HIT]
Denmark   → Copenhagen  [HIT]
```

### The "Surprise Capital" Failures

```
Canada    → Ottawa     got=Toronto    [---]
Australia → Canberra   got=Melbourne  [---]
Switzerland → Bern     got=Zurich     [---]
```

These three countries share a characteristic: their LARGEST or most
PROMINENT city is NOT the capital:
- Canada: Toronto (largest) vs Ottawa (capital)
- Australia: Melbourne/Sydney (largest) vs Canberra (capital)
- Switzerland: Zurich (largest, financial hub) vs Bern (capital)

In text corpora, Toronto, Melbourne, and Zurich appear far more
frequently in proximity to "Canada", "Australia", and "Switzerland"
than Ottawa, Canberra, and Bern. W_E therefore encodes the
"most prominent city" association, not the "capital city" association.

The geometric axis retrieves the strongest city-country association
in W_E, which for most countries IS the capital (since capital cities
are usually prominent), but fails for countries where this is not true.

### Implication

The country→capital axis encodes "MOST PROMINENT ASSOCIATED CITY",
which is usually the capital but fails for countries with non-prominent
capitals. This is a direct reflection of training data statistics in W_E:
the model learned from text where Toronto is discussed in the context
of Canada far more than Ottawa.

This is NOT a failure of the geometric approach — it is a CORRECT
retrieval from the geometric structure. The structure faithfully
encodes what W_E learned. If we want "true" capital cities, we need
training data that specifically emphasises the capital relationship,
or a lookup table for the exceptions.

---

## The Inflated-pc Problem

### country→lang: High pc, Low Holdout

```
pc=0.529  (RANK 2 overall, above +est and +er)
Holdout:  5/12 (42%)  [FAILS]
```

Expected from pc=0.529: >85% generalisation.
Actual: 42%. This is a **MISMATCH** — the linearity principle breaks.

### Root Cause: Training Data Contamination

The training pairs were:
```
France→French, Germany→German, Spain→Spanish, Italy→Italian,
Portugal→Portuguese, Japan→Japanese, China→Chinese, Russia→Russian,
Egypt→Arabic, Netherlands→Dutch
```

For 9 of 10 training pairs, the language name == the demonym. French,
German, Spanish, Italian, Portuguese, Japanese, Chinese, Russian, Dutch
are ALL BOTH the demonym AND the language name.

The axis therefore learned: "France→French" in the DEMONYM sense, not
the LANGUAGE sense. When tested on countries where language ≠ demonym:

```
Britain  → English    got=Britain   (got the country, not language)
Brazil   → Portuguese got=Brazilian (got demonym, not language)
Mexico   → Spanish    got=Mexican   (got demonym, not language)
India    → Hindi      got=Indian    (got demonym, not language)
Iran     → Persian    got=Iranian   (got demonym, not language)
Israel   → Hebrew     got=Israeli   (got demonym, not language)
```

The axis is correctly applying the demonym rule but the EVALUATION
tests the language rule. The two rules diverge for non-European
countries.

The pc was high (0.529) not because the LANGUAGE transformation is
linear, but because the training data made the DEMONYM transformation
appear to be the language transformation. This is training data
contamination: the training set does not represent the full distribution
of the language relationship.

### The Antonym Cluster Speed-axis: Shared-Target Inflation

The speed antonym cluster (fast→slow, quick→slow, rapid→gradual,
swift→sluggish) showed pc=0.387 from full training data. From 2
training pairs (fast→slow, quick→slow):

```
pc_mini = 0.640   (very high!)
holdout = 0/2     (fails completely)
```

The pc=0.640 from 2 training pairs is INFLATED by the SHARED TARGET:

Both "fast" and "quick" map to "slow". If fast and quick are nearby
in W_E (they are synonyms), and slow is far from both in roughly the
same direction, then:

```
normed(slow - fast) ≈ normed(slow - quick)
```

These two chord vectors point in nearly the same direction NOT because
there is a linear transformation, but because the two SOURCE words
are near each other AND they share the same TARGET. This creates
artificial chord consistency.

When tested on "rapid→gradual" and "swift→sluggish", the axis fails
because:
1. "gradual" and "sluggish" are NOT near "slow" in W_E
2. The axis is pointing toward "slow", not toward "the antonym of the
   source word"

---

## Detecting Inflated pc: Source-Target Correlation

The two inflation mechanisms share a common structure:

**Mechanism 1: Shared-target inflation**
- Multiple sources map to the same target
- pc is inflated by source-cluster homogeneity, not by transformation
  regularity
- Detection: check if many pairs share the same target word

**Mechanism 2: Training-scope conflation**
- Training pairs only cover a subset of the relation
- pc is high within the subset but the learned axis is wrong for the
  full relation
- Detection: check that training pairs represent the full distribution
  of source-target type combinations

### A Diagnostic for pc Validity

Before trusting a high pc:
1. Check pairwise source cosines: `mean_cos(s_i, s_j)` — if high, pc
   may be inflated by source clustering
2. Check target diversity: if many training pairs share the same target,
   pc is inflated by shared-target effect
3. Check training scope: do training pairs represent the full distribution
   of the relation?

For the canonical high-quality axes (comparative, superlative, country
demonym):
- Sources are diverse (many different adjectives / countries)
- Targets are diverse (thicker, taller, sharper... French, German, Spanish...)
- Training scope covers the full distribution

For inflated axes (speed antonyms, country→lang with European training):
- Sources may be clustered (fast≈quick), or
- Targets may be non-diverse (many sources → same target), or
- Training scope is limited to a subset

---

## Revised Unified Linearity Principle

The original principle: `pc > 0.35 → high generalisation`

Revised: `pc > 0.35 AND training_scope_adequate AND target_diverse → high generalisation`

The additional conditions:
- **Training scope adequate**: training pairs represent the full distribution
  of the relation across all relevant source subtypes
- **Target diverse**: most training pairs have distinct targets

When these conditions hold AND pc > 0.35, generalisation is confirmed.
When pc is inflated by shared targets or limited scope, the 0.35 threshold
is misleading.

---

## Complete Holdout Summary: Days 289–292

```
Axis               pc     domain   holdout         generalises?
+est (sup)         0.436  MORPH    1/1  (100%)     YES
+er (comp)         0.393  MORPH    6/6  (100%)     YES
country->demonym   0.598  SEM      12/15 (80%)     YES (100% zero-shot)
country->lang*     0.529  SEM      5/12  (42%)     NO  (pc inflated)
country->cap       0.335  SEM      11/14 (79%)     YES (surprise-cap fails)
animal->class      0.202  SEM      4/5   (80%)     YES
+s (plural)        0.155  MORPH    27/29 (93%)     YES (20 pairs)
person->nat        0.246  SEM      ~56%            MARGINAL
gender             0.213  MORPH    2/5   (40%)     NO (suppletive)
past_reg           0.174  MORPH    7/12  (58%)     NO (verb diversity)
past_irr           0.230  MORPH    5/12  (42%)     NO (irregular)
field->concept     0.087  SEM      ~25%            NO (attractor)
word->antonym      0.020  SEM      0/12  (0%)      NO
```

*pc inflated by training data scope limitation

---

## Files

- `expedition_log.md` — Day 292 results
- `426_unified_linearity.md` — Day 291 correlation analysis
- `425_linearity_principle.md` — Day 290 source class analysis
- `419_attractor_universality.md` — attractor phenomenon
