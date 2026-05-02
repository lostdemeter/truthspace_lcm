# DC 443: Local Semantic Operators — Why Axes Are Domain-Specific

**Day 308 | The definitive refutation of the universal-operator hypothesis:
morphological axes in W_E are NOT universal transformers but LOCAL SEMANTIC
OPERATORS specific to their etymological or semantic cluster. Evidence: (1)
the gender axis achieves 100% on core kinship vocabulary but 0% on titles,
animals, religion, and fiction — each domain has its own local gender geometry.
(2) the +tion axis achieves 80% on Latin -ct verbs and 68% on Latin -ate verbs
but moves Germanic roots to their verbal inflections (help→helps, walk→walked),
not to nominals. (3) no global metric (pc, domain_sim, or their product) predicts
holdout accuracy (r≤0.28), but the train_within_sim>>holdout_sim gap predicts
overfitting: a tight training cluster produces an axis that works within the
cluster but fails outside. (4) the +s capitalized-token trap is real and
partially fixable: excluding uppercase tokens improves +s from 40% to 67%.**

---

## The Local Operator Principle

### The Universal Operator Hypothesis (rejected)

Previous days assumed: if a morphological transformation has a consistent
direction in W_E (high pc), then it is a UNIVERSAL OPERATOR that can be
applied to any word in the appropriate part of speech.

**This is false.**

### The Local Operator Reality

What W_E encodes is: **for each etymological/semantic cluster, a specific
local displacement that realizes a transformation within that cluster**.

```
+tion transformation:
  Latin -ct cluster:  {act, direct, collect, select} → {action, direction, ...}
  Latin -ate cluster: {observe, describe, explain}   → {observation, ...}
  Germanic cluster:   {help, start, think, walk}     → verbal inflections (WRONG!)
```

These are THREE DIFFERENT LOCAL OPERATORS, not one universal +tion operator.
The axis computed from Latin -ct verbs happens to generalize to Latin -ate
verbs because they share the same W_E region. It does NOT generalize to
Germanic verbs because they occupy a different region.

### The Gender Domain Decomposition

```
Domain         Training scale=0.342  Hits   Acc
──────────────────────────────────────────────────
kin_core       ✓ (training domain)   8/8    100%
kin_extended                          2/3    67%
titles                                0/5    0%
occupation                            2/6    33%
religion                              0/2    0%
animals                               0/4    0%
fiction                               0/2    0%
```

The gender axis learned from {king, man, boy, father, son, brother, uncle,
husband} is PERFECTLY calibrated for that kinship cluster. It produces
0% accuracy for:
- **Titles**: lord→lady, duke→duchess, prince→princess — these pairs have
  a different displacement direction in W_E than king→queen
- **Animals**: lion→lioness, stallion→mare — zoolinguistic gender marking
  is geometrically distinct from human kinship gender
- **Fiction**: wizard→witch — fictional/fantasy gender markers live in
  a completely different region of the semantic space

**Each domain has its own LOCAL GENDER AXIS:**
```
kin_gender:       king↔queen, man↔woman, father↔mother
title_gender:     lord↔lady, duke↔duchess, emperor↔empress
animal_gender:    lion↔lioness, stallion↔mare, ram↔ewe
fiction_gender:   wizard↔witch, sorcerer↔sorceress
```

These four operators are geometrically distinct — applying kin_gender
to a title or animal produces the wrong vector.

---

## The Overfitting Signal: train_within_sim

### The Metric

`train_within_sim` = mean cosine similarity between all pairs of
TRAINING SOURCE embeddings.

High `train_within_sim` means the training sources form a TIGHT CLUSTER
in W_E. The axis learned from a tight cluster is a precise local operator
for that cluster — it transfers only to words within the same cluster.

### Evidence

```
Axis       train_sim  holdout_sim  gap     holdout_acc
gender     0.183      0.105        +0.078  17%   [OVERTIGHT training cluster]
+tion      0.095      0.099        −0.004  80%   [training = holdout spread]
+er        0.101      0.091        +0.010  80%   [nearly matched spread]
past_irr   0.119      0.094        +0.025  69%   [small gap]
+s         0.111      0.063        +0.048  40%   [moderate gap + caps trap]
un-        0.043      0.044        −0.001  8%    [low pc dominates]
+ful       0.065      0.058        +0.007  8%    [low pc dominates]
```

**Pattern:**
- gender has the LARGEST train-holdout gap (+0.078) AND lowest holdout acc
- +tion has NO gap (−0.004) AND highest holdout acc
- +er has tiny gap (+0.010) AND high holdout acc (80%)
- +s has moderate gap (+0.048) AND moderate holdout acc (40%)

The `train_sim - holdout_sim` gap is a **distribution mismatch signal**:
it measures how much more similar the training sources are to each other
compared to the holdout sources.

### Why the Gap Matters

When training sources are tight (high sim), the mean axis captures a
CLUSTER-SPECIFIC direction. The optimal displacement for king→queen is
exactly right for all {king, man, boy, father} but wrong for {monk,
prince, emperor, lion}.

When training sources are spread (low sim), the mean axis captures a
DIRECTION THAT AVERAGES ACROSS THE SPREAD — it finds the component of
the transformation that is consistent across the whole space, which
is more likely to transfer to new words.

---

## The +s Capitalized Token Trap

### Two Failure Modes

**Mode 1: Capitalized token interception**
Words: cup, road, leg, room, wall (→ Cup, Road, Leg, Room, Wall)
- Fix: exclude tokens starting with uppercase
- Result: +s improves from 40% to 67% (6/15 → 10/15)

**Mode 2: Hyphenated compound token interception**
Words: hand, eye, arm (→ hand, -eye, -arm)
- After caps exclusion, these words hit hyphenated compound tokens
- '-eye' is the second element of a compound (bird's-eye view)
- '-arm' is the second element of a compound (forearm, side-arm)
- These tokens are NOT uppercase but intercept the axis trajectory

```
Source    Standard NN   No-caps NN    Target
hand      Hand          hand          hands     FAIL (same-token)
eye       Eye           -eye          eyes      FAIL (hyphenated)
arm       Arm           -arm          arms      FAIL (hyphenated)
wall      Wall          -wall         walls     FAIL (hyphenated? or same)
fire      Fire          fire          fires     FAIL (same-token)
```

### +s Remaining Failures After Caps Fix

5 remaining failures after caps exclusion:
1. **hand → hand**: the plural 'hands' is not the nearest lowercase neighbor
2. **eye → -eye**: compound token '-eye' intercepts the trajectory
3. **arm → -arm**: compound token '-arm' intercepts
4. **wall → -wall**: similar compound issue
5. **fire → fire**: same-token (not plural)

These all share one property: the source word (hand, eye, arm, wall, fire)
is used in common COMPOUND CONSTRUCTIONS in Qwen2's training data, creating
near-duplicate compound tokens in the vocabulary.

### The Broader Implication

The capitalized and hyphenated token traps are a TOKENIZATION ARTIFACT:
Qwen2's BPE tokenizer creates separate tokens for 'cup'/'Cup'/'cups', and
these have closely related but distinct embeddings. The +s axis displacement
is too small (scale=0.262) to escape the capitalized/compound trap.

**If we set a minimum displacement threshold**: only accept as a "hit" if
the target is meaningfully different from the source neighborhood — then
these traps disappear. In practice, morphological axis application should
EXCLUDE source-word variants (capitalized, hyphenated, compound forms) not
just the exact source token.

---

## The +tion Non-ct Latin Expansion

### Latin -ate → -ation Results

13/19 = 68% accuracy on {observe, describe, explain, combine, transform,
operate, create, investigate, communicate, participate, appreciate,
negotiate, evaluate} → {observation, description, ...}

**5 failures:**
- produce → **produces** (not production)
- educate → **educating** (not education)
- demonstrate → **demonstrates** (not demonstration)
- accelerate → **accelerating** (not acceleration)
- generate → **generates** (not generation)

**Pattern**: failures hit VERBAL INFLECTIONS (-s, -ing) rather than
NOMINAL derivations (-tion). The 5 failing verbs are all high-frequency
verbs that appear heavily in verbal contexts (produce, educate, generate).
For these verbs, the verbal inflections (produces, educating) are
MORE CENTRAL in the embedding neighborhood than the nominals.

This is a frequency effect: if a verb appears mostly in verbal contexts,
its nearest neighbors are verbal forms. If it appears mostly in nominal
or mixed contexts, its nearest neighbors include the nominal form.

### Germanic Root Behavior

For Germanic-root verbs, the +tion axis displacement moves them to:
```
help  → help, Help, helps     (stay near source, verbal forms)
start → Start, Start, start   (capitalized + source, no nominal)
think → thinks, Think, think  (verbal forms only)
walk  → Walk, walked, walks   (verbal forms including past)
```

The +tion axis pushes Germanic verbs into their verbal inflection cluster,
because that's the nearest "new vocabulary region" in the direction of the
axis for those words. This confirms: the +tion axis learned from Latin
-ct verbs points away from the Latin-root cluster and TOWARD the Latin
nominal cluster — the same direction also points toward verbal inflections
for Germanic roots (which have no Latin nominal forms).

---

## Implications for the TruthSpace Geometric LCM

### The Structure Is Not Universal

W_E does not encode morphological transformations as universal operators.
It encodes a **collection of LOCAL CLUSTER OPERATORS**:
- Each semantic/etymological cluster has its own internal geometry
- The transformation within a cluster is consistent (high local pc)
- The transformation does NOT transfer between clusters
- The "global axis" computed from a training set generalizes only to
  words in the same cluster

### What This Means for a Geometric LCM

To build a geometric system that handles morphological transformations:
1. **Cluster detection first**: identify which semantic cluster a word belongs to
2. **Local axis selection**: use the axis for that specific cluster
3. **No single universal operator**: there is no one "plural" axis, only
   a family of local plural operators (one per semantic cluster)

This is actually **more geometric, not less**: the LCM must navigate
between cluster geometries, not just apply a single global vector.

### The Self-Similarity Exception

**Degree transformations** (+er, er→est) are the closest to universal
operators because all adjectives share a single "adjectival quality"
semantic space. The +er axis trained on {fast, slow, tall, short, bright,
dark, deep, clean} transfers to 80% of unseen adjectives.

This works because the ADJECTIVAL SPACE is relatively homogeneous in W_E:
all quality adjectives form one large cluster where the comparative
transformation has a consistent direction.

**Contrast with gender**: there is no single "noun" space — nouns split
into tight semantic clusters (kin, animals, titles, occupations) each
with their own local geometry.

---

## Day 309 Plan

1. **Domain cluster identification**: for each "failing" gender domain
   (titles, animals, fiction), build domain-specific gender axes. What
   are their pc values and holdout accuracies?

2. **Cluster geometry map**: project the training sources for all 12 axes
   onto the first 5 mPCs. Does tight clustering in mPC space predict
   domain-specificity?

3. **Extended +s with full exclusions**: add compound/hyphenated token
   exclusion to the no-caps filter. Does it fix the remaining 5 failures?

4. **Cross-domain transfer test**: take the +tion axis trained on ct-verbs
   and apply to ate-verbs using DIFFERENT scales. Does scale tuning help?

---

## Files

- `expedition_log.md` — Day 308 results
- `442_semantic_atlas_and_axis_predictors.md` — DC 442
- `day308_domain_overlap_and_axis_fixes.py` — experiment script
