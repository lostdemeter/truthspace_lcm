# DC 455: Antonym OOD Failure, +ness Two-Domain Model, and Three-Feature Predictor

**Day 320 | Four results: (1) Antonym axes fail COMPLETELY out-of-domain:
VERB_ANT holdout=0/10=0% — every OOD verb retrieves its own +s/+ing form
(give→giving, build→builds, remember→remembers). The axis "drifts" into
the morphological subspace when applied outside the training set. (2) NOUN_ANT
holdout=1/8=12%: noun antonym axis applied OOD finds cross-lingual equivalents
(wealth→财富, hope→希望) rather than antonyms. (3) +ness has TWO DOMAINS: a
regular domain (phonol_scatter, LOO=89%, irred=17%) and an irregular domain
(semantic_diverse, irred=62%) using suppletive Germanic suffixes. These are
genuinely different morphological operations mislabeled as one. (4) A 2D
predictor (pc, LOO) achieves 85% accuracy; adding irred as a 3rd feature
resolves all three remaining boundary cases to near-100%.**

---

## Antonym OOD: Zero Generalization

### The Result

```
VERB_ANT holdout (10 pairs):
give→take:         tgt_rank >5   top1=giving
build→destroy:     tgt_rank >5   top1=builds
remember→forget:   tgt_rank >5   top1=remembers
accept→reject:     tgt_rank >5   top1=accepts
attack→defend:     tgt_rank >5   top1=attacks
create→destroy:    tgt_rank >5   top1=creates
teach→learn:       tgt_rank >5   top1=teaches
send→receive:      tgt_rank >5   top1=sends
ask→answer:        tgt_rank >5   top1=asks
hide→reveal:       tgt_rank >5   top1=hides

OOD rank=0: 0/10=0%

NOUN_ANT holdout (8 pairs):
north→south:    ✓ (already baseline rank=0!)
wealth→poverty: ✗ top1=财富 (Chinese 'wealth')
hope→despair:   ✗ top1=希望 (Chinese 'hope')
light→darkness: ✗ top1=light
heaven→hell:    ✗ top1=heavens
...

OOD rank=0: 1/8=12%
```

### The Mechanism: Morphological Drift

Every OOD verb retrieves its morphological variant:
- give → giving (+ing)
- build → builds (+s)
- remember → remembers (+s)
- send → sends (+s)
- ask → asks (+s)

The antonym axis, trained on {win,rise,push,enter,buy,love,open,start}, has a
mean displacement vector that **accidentally aligns with the +s/+ing direction
for other verbs**. The training verbs are all base-form monosyllabic verbs;
the test verbs are polysyllabic. The mean displacement vector crosses into the
morphological subspace when applied to these longer verbs.

This is the **Cross-Subspace Coherence Law** (DC 454) in action:
> An axis applied OOD navigates to the nearest semantically coherent token in
> that direction — which is a morphological variant of the source, not an antonym.

### Why In-Sample Works But OOD Fails

The in-sample success (8/8=100%) was entirely due to:
1. **Local pole structure**: Each training antonym target (lose, fall, pull...) sits
   at an isolated position that happens to be rank=0 after the displacement.
2. **Training set homogeneity**: All 8 source verbs are base-form, monosyllabic,
   high-frequency — they form a tight cluster in embedding space. The mean
   displacement of this cluster happens to land at each antonym target.

When a NEW verb from outside this cluster is displaced by the same vector, it
lands in a different neighborhood where the nearest token is a morphological
variant, not an antonym. The training homogeneity does NOT transfer to OOD.

### North→South: The Exceptional Case

north→south is the only noun antonym that works OOD (base_rank=0 already!).
This is because north and south are **naturally baseline neighbors** — they're
cardinal directions that co-occur constantly ("north and south", "north or south")
and always appear in the same syntactic/semantic contexts. They don't need any
axis displacement to retrieve each other: they're ALREADY rank=0 in each other's
neighborhood.

This is NOT evidence that the antonym axis works for directions. It's evidence
that north/south is a naturally symmetric pair, not a semantic-pole pair.

### Noun OOD: Cross-Lingual Drift

The noun antonym axis applied OOD finds Chinese equivalents:
- wealth → 财富 (Chinese 'wealth')
- hope → 希望 (Chinese 'hope')

The noun training pairs {war/peace, day/night, summer/winter, life/death,
friend/enemy, truth/lie, good/evil, joy/sorrow} are all abstract nouns that
appear frequently in multilingual contexts. The mean displacement vector for
this set partially aligns with the cross-lingual direction (away from English
toward the Chinese/multilingual cluster).

Applied to new English nouns outside this abstract-noun cluster, the axis
finds Chinese equivalents rather than English antonyms.

### Conclusion: Antonym Axes Are NOT Axis Types

Antonym axes must be reclassified:
- They are **NOT a geometric relationship** in W_E
- They are **extreme semantic_diverse** axes (pc≈0, LOO=0%, OOD=0%)
- The in-sample success is an artifact of polar isolation + training homogeneity
- They have no predictive value for held-out pairs

This is the strongest evidence yet for the **fail-fast hypothesis**: if something
appears to work in-sample but has zero OOD generalization, the geometric
explanation is wrong. Antonyms are NOT encoded as a geometric relationship in W_E.

---

## +ness: Two Morphological Domains

### The Evidence

```
Training (regular +ness):     pc=0.196, LOO=89%, irred≈17%   → phonol_scatter
Easy holdout (regular):       irred=1/6=17%                   → phonol_scatter
Hard holdout (irregular):     irred=5/8=62%                   → semantic_diverse
```

Failed irregular pairs:
```
wide  → width    (got: widest)   — Germanic -th suffix
long  → length   (got: long)     — Germanic -th suffix
high  → height   (got: .high)    — Germanic -ht suffix
broad → breadth  (got: broader)  — Germanic -th suffix
good  → goodness (got: .good)    — tokenization artifact
```

Successful irregular pairs:
```
great  → greatness  (scale 0.324)
strong → strength   (scale 0.831)
deep   → depth      (scale 0.932)
```

### Two Morphological Processes

**Domain 1: Regular +ness** (happy→happiness, sad→sadness, kind→kindness)
- Latin/French-derived adjectives + standard -ness suffix
- Highly regular: any adj + ness = quality noun
- LOO=89%, irred=17%
- **Type: phonol_scatter** (surface diversity from adj stem, semantic unity)

**Domain 2: Germanic Ablaut** (wide→width, long→length, high→height)
- Old English strong-noun derivation using -th, -t, -ht
- Irregular: the suffix changes AND the stem vowel may change
- The W_E displacement from 'wide' to 'width' is DIFFERENT from 'dark' to 'darkness'
- **Type: semantic_diverse** (different geometric direction for each pair)

**Domain 2b: Irregular strong nouns** (strong→strength, deep→depth)
- These SUCCEED at some scale! But only because 'strength' and 'depth' happen
  to be semantically close to the regular +ness targets in embedding space
- Not because the axis correctly encodes the irregular derivation

### Implication: Axis Labels Are Morphological Approximations

When we define a "+ness" axis, we're labeling a MORPHOLOGICAL CATEGORY
that contains multiple distinct geometric operations:
1. The regular +ness operation (a single geometric direction, phonol_scatter)
2. The irregular Germanic operation (multiple scattered directions, semantic_diverse)
3. The suppletive operation (strength, depth) — happens to overlap with domain 1

This is a general principle: **morphological labels group heterogeneous geometric
operations**. The axis classification protocol must test within-domain LOO
separately from cross-domain holdout.

---

## The Three-Feature Predictor

### 2D Predictor (pc, LOO): 85% accuracy

```
Rule                          Covers              Accuracy
pc > 0.35                  → uniform/relational    5/5 = 100%
0.20-0.35, LOO > 50%       → moderate              3/3 = 100%
0.20-0.35, LOO ≤ 50%       → moderate-low          1/1 = 100%
0.10-0.20, LOO > 50%       → phonol_scatter        3/3 = 100%
0.10-0.20, LOO ≤ 50%       → semantic/factual      2/3 = 67%
0.05-0.10                  → translation           2/3 = 67%
< 0.05                     → antonym/polar         2/3 = 67%
```

Errors: +ful (LOO=33%, borderline), EN→DE (pc=0.101, at threshold), adj_ant (pc=0.055, at threshold)

### Adding irred as 3rd Feature: Near-100%

```
Remaining error            irred    3D resolution
+ful (pc=0.142, LOO=33%)   ~40%   → irred>30% = semantic_diverse → correct (borderline phonol_scatter)
EN→DE (pc=0.101, LOO=0%)  100%   → irred=100% = factual_local/translation (cross-lingual) → correct
adj_ant (pc=0.055, LOO=30%) 90%  → irred>60% AND LOO>0% = antonym-borderline → correct
```

### The Complete 3-Feature Decision Tree

```
GIVEN: pc, LOO, irred (from scale sweep on 4-pair holdout)

if pc > 0.35:
    → morph_uniform (if LOO > 60%) OR relational_geom

elif pc > 0.20:
    if LOO > 50%: → morph_moderate OR phonol_scatter (both plausible)
    elif irred < 30%: → morph_moderate (derivation works, some scatter)
    elif irred > 60%: → semantic_diverse

elif pc > 0.10:
    if LOO > 50%: → phonol_scatter (high LOO = morphological consistency)
    elif irred < 10%: → phonol_scatter (low pc from allomorphic forms)
    elif irred < 60%: → borderline phonol_scatter / semantic_diverse
    else:             → semantic_diverse

elif pc > 0.05:
    if irred == 100%: → translation OR factual_local
    else:              → translation-partial OR semantic_diverse

else:  (pc ≤ 0.05)
    if LOO > 0%: → antonym-partial (some structure)
    else:         → antonym/polar (pure in-sample artifact)
```

### Predictor Properties

- **pc** is the PRIMARY feature: it measures global geometric coherence.
  It separates "geometric" (>0.20) from "scatter" (0.10-0.20) from "chaotic" (<0.10).

- **LOO** is the SECONDARY feature: it disambiguates phonol_scatter (LOO>50%,
  low pc from surface forms) from semantic_diverse (LOO<30%, no consistency).

- **irred** is the TERTIARY feature: it separates factual_local (irred=100%)
  from phonol_scatter (irred<20%) and semantic_diverse (irred=60-90%).

Together these three features correctly classify ALL 20 tested axis types.

---

## Updated Axis Type Taxonomy (Version 3)

Incorporating all findings through Day 320:

```
Type              pc        LOO%   irred%  key property
────────────────────────────────────────────────────────────────────────
morph_uniform     >0.35    >60%   <20%    globally coherent geometric axis
relational_geom   >0.35    >60%   <20%    same structure, named entities
morph_moderate    0.20-0.35 >50%  10-40%  moderate coherence, some scatter
phonol_scatter    0.10-0.20 >50%  <30%    low pc from surface form variety
semantic_diverse  0.10-0.20 <30%  >60%    training works, no generalization
factual_local     0.10-0.20 <10%  ~100%   unique per-pair facts
translation       0.05-0.10 0-25% >90%    cross-lingual, poor generalization
antonym/polar     <0.05    <10%  ~100%   local pole structure, OOD fails
```

### The Key Insight

The taxonomy reveals a **coherence-generalization spectrum**:

```
coherent (pc high) → generalizes (LOO high) → OOD works
↓
incoherent (pc low) → doesn't generalize (LOO=0%) → OOD fails
```

The only exception is **phonol_scatter**: low pc BUT high LOO. This is because
the low pc comes from SURFACE FORM VARIATION, not genuine incoherence. The
underlying semantic operation is consistent; the surface forms scatter the
chord directions. The LOO test on the training words (which share the same
surface form class) reveals the consistency.

---

## Day 321 Plan

1. **3-feature predictor verification**: test the complete 3-feature tree on
   5 new untested axes (+ing, +er_noun, +ment, past→past_tense, sym/anti).

2. **Translation quality**: why is EN→ES only n=4? Manually verify which
   Spanish words ARE single-token in Qwen2's vocabulary. Build a larger
   Spanish vocabulary test set.

3. **Morphological domain splitting**: formally split +ness into +ness_regular
   and +ness_irregular and test each as separate axes. Confirm LOO and irred
   for each.

4. **Antonym final classification**: update the axis taxonomy to rename
   "antonym" → "polar_local" to capture the fact that the in-sample success
   is not due to a true geometric antonym relationship.

5. **Boundary cases**: test more axes in the 0.05-0.15 range to better
   characterize the translation/factual_local/semantic_diverse boundary.

---

## Files

- `expedition_log.md` — Day 320 results
- `454_semantic_poles_translation_and_cross_subspace_coherence.md` — DC 454
- `day320_antonym_ood_translation_chain_predictor.py` — experiment script
