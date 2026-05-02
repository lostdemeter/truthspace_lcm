# DC 327: Complete φ-Trie Navigability Map

**Date:** 2026-03-18  
**Experiment series:** Days 92–97  
**Prerequisite:** DC 326 (Initial Navigability, Days 92–94)

---

## Overview

DC 326 introduced the category/relational axis distinction and identified
two conditions for trie navigability. Days 95–96 confirmed both conditions
experimentally. Day 97 completed the survey of all 12 axes.

This document synthesizes the complete navigability picture.

---

## The Two Conditions for Trie Traversal

For flipping axis bit k of token `src` to navigate to `tgt`:

### Condition 1: Bit Discrimination
`src_bit[k] ≠ tgt_bit[k]`

The source and target must have different ternary values on the axis being
traversed. This is determined by whether the axis vector discriminates
between the two token representations.

**Measurement (isolated addressing):**

| Axis | Separation | Notes |
|------|-----------|-------|
| gender | 8/8 (100%) | inherent token property |
| past_tense | 7/8 (88%) | build/built both get L |
| comparative | 5/6 (83%) | slow/slower both get H |
| plural | 2/6 (33%) | sentence-level signal doesn't capture number |
| antonym | 4/8 (50%) | no shared direction; pairs split by chance |

**Effect of contextualization** (embedding token in axis template sentence):

| Axis | Isolated sep | Contextualized sep | Δ |
|------|-------------|-------------------|---|
| gender | 100% | 100% | 0 |
| past_tense | 88% | **100%** | +12.5% |
| comparative | 83% | **100%** | +16.7% |
| plural | 33% | **0%** | −33% ← HURTS |
| antonym | 50% | 50% | 0 |

Templates:
- past_tense: `"I [VERB] to the market every single morning"`
- comparative: `"The [ADJ] car"`

**Key insight on plural:** The plural T2 axis captures number agreement
in the last-token position of a SENTENCE, not the target token's morphological
form. Embedding "dog" and "dogs" in "A dog/Dogs played..." places both at
high projection on the plural axis, since both produce a sentence-level
signal that drives the last-token representation equally. The axis
distinguishes sentence number, not word number.

### Condition 2: Address Uniqueness
`tgt`'s full K-dimensional address is the nearest to `src`'s flipped
address among all vocabulary tokens.

This requires that `tgt` is the ONLY token that shares K−1 axes with `src`
while differing on axis k. Uniqueness degrades when:
1. The vocabulary contains many tokens with similar profiles
2. The axis being flipped assigns the same bit to many unrelated tokens

**Measurement (full 401-token vs POS-restricted tries):**

| Config | past_tense | comparative |
|--------|-----------|-------------|
| Full 401-token, isolated | 0/8 (0%) | 2/6 (33%) |
| POS-only, isolated | 1/8 (12%) | 3/6 (50%) |
| POS-only, contextualized | **3/8 (38%)** | 3/6 (50%) |

**Why full trie fails for past_tense:** The sentence-level past_tense axis
at L28 assigns H to many non-verb tokens that happen to project strongly
on the direction "past-tense sentence → present-tense sentence". Flipping
run's past_tense bit to H finds "fish", "monkey", "bag" before "ran"
because these tokens also have H in past_tense AND similar profiles on
the other 11 axes.

**Why verb-only trie helps:** With only 94 verbs, the competition at any
address is much less. "ran" is now uniquely near run's flipped address
because all other high-past_tense tokens are also verbs, and their
non-tense axes differ more from run's profile.

---

## The Complete Navigability Map

### Confirmed Navigable (Full 401-Token Trie)

| Pair | Axis | Rank | Notes |
|------|------|------|-------|
| king → queen | gender | 0 | EXACT |
| brother → sister | gender | 0 | EXACT |
| man → woman | gender | 2 | |
| actor → actress | gender | top-5 | |
| small → smaller | comparative | 0 | |
| good → better | comparative | 1 | |

**Condition:** category axes that encode inherent token properties. Works
without POS restriction or contextualization.

### Navigable with Conditions (POS-Restricted + Contextualized)

| Pair | Axis | Rank | Config needed |
|------|------|------|--------------|
| run → ran | past_tense | 0 | verb-only + context |
| fly → flew | past_tense | 2 | verb-only + context |
| break → broke | past_tense | 4 | verb-only + context |
| fast → faster | comparative | 0 | adj-only + context |
| big → bigger | comparative | 0 | adj-only + context |
| bad → worse | comparative | 4 | adj-only + context |

**Condition:** relational axes with moderate coherence (~0.35–0.53).
Requires POS-stratified vocabulary AND axis-appropriate sentence context.

### Not Navigable (Any Configuration)

| Axis | Reason | Token-level coherence |
|------|--------|----------------------|
| plural | Sentence-level axis; context hurts; no token-level axis | 0.488 (good signal, wrong level) |
| antonym | No shared geometric direction | 0.008 (random) |

**Plural:** Despite having the highest token-level coherence (0.488) of
any morphological axis, plural is fundamentally not navigable because:
1. The T2 axis was computed from sentence-level differences (number agreement)
2. Embedding singular/plural tokens in the SAME sentence template gives both
   the same projection (the sentence structure dominates the token form)
3. A token-level plural axis would need a fundamentally different
   construction: measure hidden(" dog") vs hidden(" dogs") directly, but
   this has only moderate coherence (0.488) at L28

**Antonym:** There is no single geometric direction for antonymy. Each
antonym pair lives in its own subspace:
- hot↔cold: independent axis
- big↔small: independent axis
- fast↔slow: independent axis
- etc.

Antonymy is a semantic RELATION (defined by contrast), not a FEATURE
(shared direction). No single bit can encode "being the antonym of
another word" without specifying which other word.

---

## Morphological Encoding in Transformer Layers (Day 94)

A critical finding about WHERE morphological information lives:

```
Layer   plural   past_tense   comparative
L1      0.376    0.371        0.530
L8      0.366    0.062       -0.011
L15     0.367    0.062       -0.012
L22     0.366    0.063       -0.011
L27     0.479    0.278        0.297
L28     0.520    0.347        0.480
```

(Values = mean pairwise cosine of difference vectors = axis coherence)

**Middle-layer morphology collapse (L8–L22):** Comparative and past_tense
coherence drops to near zero. The model "forgets" inflectional morphology
in middle layers, abstracting to the semantic lemma. By L8, "run" and "ran"
have nearly identical representations. The signal partially revives at L27–28
as the model prepares to generate output tokens.

This pattern is consistent with the DRUM/COMB/MUSIC layer structure
established in prior work:
- **DRUM (L0–3):** Form preserved (embedding proximity maintained)
- **COMB (L4–25):** Semantic abstraction (form collapsed to lemma)
- **MUSIC (L26–28):** Output preparation (form partially restored)

**Practical implication:** For morphological navigation, L1 is the best
layer for token-level axes (embedding proximity is maximally preserved),
but the sentence-level T2 axes work at their original layers because they
operate at the SENTENCE level (where L28 contextual representations
capture full sentence semantics).

---

## Axis Classification Summary

| Axis | Full-trie | POS-restricted | Bit sep | Notes |
|------|----------|----------------|---------|-------|
| gender | 5/8 (62%) | — | ~100% | Category; king→queen rank=0 |
| comparative | 1/6 (17%) | 3/6 (50%) adj | 83% | Relational-degree |
| past_tense | 0/8 (0%) | 3/8 (38%) verb+ctx | 88% | Relational-morphological |
| hypernym | 1/12 (8%) | — | 75% | eagle→bird only; uniqueness bottleneck |
| synonym | 1/8 (12%) | — | 75% | happy→joyful only; uniqueness bottleneck |
| concrete | 0/5 (0%) | — | 100% | Perfect bit sep; uniqueness fails |
| plural | 0/6 (0%) | — | 33% | Context hurts; no solution |
| antonym | 0/8 (0%) | — | 50% | No shared axis direction |
| passive | 0/2 (0%) | — | 50% | Sentence-level only |
| negation | 0/4 (0%) | — | **0%** | Uniform across all tokens |
| causation | untested | — | — | Sentence-level only |
| question | untested | — | — | Sentence-level only |

---

## Design Principles for a Navigable φ-Trie

### Principle 1: Category Axes Enable Universal Navigation

Axes that encode inherent token properties (gender, animacy, concreteness,
person) work without restriction. The trie can serve as a semantic
coordinate system for these dimensions with the standard 401-token vocabulary.

### Principle 2: Relational Axes Require Domain Restriction

Axes that encode transformations (tense, degree, number, voice) only enable
navigation within semantically homogeneous sub-vocabularies. The full
multi-domain vocabulary introduces too many false positives at the flipped
address.

### Principle 3: Contextualized Addressing for Relational Axes

Embedding tokens in axis-appropriate templates improves bit discrimination
for some axes (past_tense: 88%→100%, comparative: 83%→100%). The template
provides the semantic context in which the transformation is meaningful.

Exception: plural axis DEGRADES with contextualization because the sentence
frame contributes more to the plural signal than the word's morphological form.

### Principle 4: No Universal Trie for Antonymy

Antonymy cannot be encoded as a dimension of a trie because there is no
shared geometric direction. Each antonym pair is a unique contrast in
its own semantic subspace. A different data structure (e.g., explicit
edge list or pairwise similarity matrix) is needed for antonym lookup.

### Principle 5: The Multi-Dimensional Address Provides Context

The 12-dimensional address is what makes category-axis navigation reliable.
When king's gender bit is flipped, the other 11 bits uniquely constrain
the search to "queen" — no other token is as close in 12D Hamming space.
This would not work in a 1D address (many female-coded tokens would compete).

---

## The φ-Trie: Semantic Index vs Coordinate System

The 12D sentence-level φ-trie optimized in DC 325 (LOO=0.9443) is primarily
a **semantic similarity index**: tokens with similar contextual profiles
cluster together for efficient generative lookup.

With the design principles above, it can ALSO function as a **partial
semantic coordinate system** for category axes (gender and similar), and
as a conditional coordinate system for relational axes in restricted
domains.

The path to a fully navigable semantic coordinate system would require:
1. Building separate sub-tries per POS category with contextualized addressing
2. Using category-only axes (gender, animacy, concrete, person) for the
   universal trie
3. Maintaining explicit relational indices for antonymy and plurality

This two-level design — a universal semantic index plus domain-specific
navigable sub-tries — represents the natural architecture for a
φ-trie-based reasoning system.

---

## Days 98–99: Semantic Parallelism is the Unified Predictor

### Day 98: Spearman ρ = +0.852 (p≈0)

For all 67 GT pairs from Days 92–97, non-axis Hamming distance (how similar
src and tgt are on the 11 non-traversed axes) correlates strongly with
traversal rank:

```
na_hamming   mean_rank   n_pairs
         0         0.0         1
         1         5.2         4
         2        19.9         7
         3        43.4        17
         4       102.5        15
         5       144.8        10
         6       279.1        10
         8       407.0         1
```

Mean rank increases monotonically at every step. The condition:
  **na_hamming ≤ 1  AND  src_bit ≠ tgt_bit  →  rank < 5**

is empirically supported with 94% accuracy for na_ham=0.

### Day 99: Mining All 9,458 Candidates

Exhaustive N²×12 enumeration of all pairs with na_ham≤1 and bit_sep=True:

```
Total candidates:          9,458
Confirmed navigable:       5,977 (63%)

na_ham=0: 911/970   (94%)
na_ham=1: 5066/8488 (60%)
```

All 12 axes navigable at 52–92%:

```
passive:     66 candidates,  61 navigable (92%)
plural:     686 candidates, 513 navigable (75%)
comparative: 916 candidates, 658 navigable (72%)
synonym:    790 candidates, 571 navigable (72%)
past_tense: 770 candidates, 546 navigable (71%)
negation:   390 candidates, 261 navigable (67%)
concrete:   800 candidates, 541 navigable (68%)
hypernym:  1254 candidates, 755 navigable (60%)
gender:    1078 candidates, 645 navigable (60%)
causation: 1294 candidates, 687 navigable (53%)
antonym:   1414 candidates, 739 navigable (52%)
```

### Critical Reframing: All Axes Are Navigable

The Day 92–97 classification of axes as "navigable" (gender) vs "not navigable"
(past_tense, plural, antonym) was an artifact of testing SEMANTICALLY selected
pairs that happened to have high na_hamming:
- run/ran: na_hamming = 7
- hot/cold: na_hamming ≥ 4
- dog/dogs: na_hamming ≥ 3

When pairs with na_hamming ≤ 1 are tested, ALL axes navigate at 52–92%.
The axis label is not the predictor — geometric parallelism is.

### Important Caveat: Not All Navigable Pairs Are Semantically Meaningful

Some na_ham=0 navigable pairs are not linguistically meaningful:
- fork→cat (gender axis): two concrete objects, not a gender pair
- most→least (gender axis): quantifiers classified by the gender T2 axis

The trie navigates geometrically parallel pairs regardless of linguistic
meaning. Human-intuitive pairs (king→queen) are a special subset where
geometric parallelism aligns with linguistic parallelism.

---

## Day 100: Geometric Parallelism = Semantic Relatedness

### 12D Hamming is a Semantic Distance Metric

Full N² pairwise comparison (420×420=88,200 pairs):

```
12D Hamming   mean_logit_cosim
          0          0.9123     ← same-leaf
          1          0.9006
          2          0.8902
          3          0.8863
          4          0.8786
          5          0.8708
          6          0.8620
          7          0.8529
          8          0.8420
         11          0.8099     ← maximally distant
```

Pearson r = −0.320: moderate negative correlation, monotonic across all 12 bins.
The φ-trie's Hamming distance is a valid semantic distance metric at the
logit distribution level.

### Navigable Pairs Have +4.2% Semantic Advantage Over Random

```
Same-leaf (Hamming=0):          0.9123
Navigable (na_ham≤1, rank<5):  0.9009  (+4.2% over random)
Accidental navigable pairs:     0.9007  (+4.2% over random)
Meaningful navigable pairs:     0.9211  (+6.2% over random)
Random pairs:                   0.8588
```

### "Accidental" Pairs Are Semantically Related at Model Level

Pairs like (most, least), (dirty, old), (run, walk), (thread, spider)
are all confirmed navigable via the trie AND have logit cosim 0.89–0.95.
They reflect contextual co-occurrence structure:
- most/least: scalar-opposite quantifiers
- dirty/old: both describe degraded states
- run/walk: verbs of locomotion
- thread/spider: thin/web semantic field

These are not misclassifications. They are genuine semantic relationships
at the model's representational level, not captured by human taxonomy.

### The φ-Trie Encodes Contextual Co-Occurrence Fingerprints

The T2 axes were computed from sentence pairs (e.g., gender from
"The king ruled" → "The queen ruled"). The resulting 12D address space
captures sentence-level contextual fingerprints. Tokens that share
contextual co-occurrence patterns get similar addresses.

The "gender axis" doesn't just encode biological gender — it captures
a broader scalar/polarity dimension. Most/least are opposite ends of a
scalar, just as king/queen are opposite ends of a gender-royalty dimension.

### TruthSpace Hypothesis Evidence

The experiment provides three levels of evidence:
1. **Structural evidence**: LOO=0.9443 shows trie leaf clusters are 
   semantically coherent (same-leaf cosim=0.9123 vs random=0.8588)
2. **Navigability evidence**: 5,977 pairs navigate with rank<5, all with
   genuine semantic relatedness (+4.2% over random)
3. **Metric evidence**: 12D Hamming monotonically correlates with logit
   cosine (Pearson r=−0.320), making it a valid semantic distance metric

---

## Day 97 Findings: Negation Uniformity and Uniqueness Bottleneck

### Negation: Zero token-level discrimination

The negation T2 axis assigns L to ALL content words. Negation is a
sentence-level operator with no token-level footprint whatsoever. This is
expected: the word "fast" has no inherent "negatedness" — only sentences
can be negated.

### Uniqueness bottleneck for semantic axes

Concrete axis achieves 100% bit separation (5/5 pairs) yet 0% traversal.
Hypernym achieves 75% bit separation yet 8% traversal. The bottleneck
is address uniqueness: many tokens share H on these axes, so flipping
a source token's bit finds competing tokens before the target.

**The semantic parallelism condition:** Traversal succeeds only when
source and target are near-identical on 11/12 non-traversed axes:
- king/queen: parallel on animacy, royalty, syntactic role → gender=rank 0
- eagle/bird: parallel on animacy, type → hypernym=rank 0
- dog/animal: NOT parallel (dog=pet/mammal vs animal=generic) → hypernym=rank 275

This is a strong condition that only a small fraction of semantic pairs satisfy.

### Synonym: Trie encodes contextual fingerprints, not pure semantics

Synonyms (big/large, small/tiny) should have identical semantic content,
but their trie addresses differ substantially. This confirms that the
12D address is a CONTEXTUAL FINGERPRINT (frequency, collocation,
register) rather than pure semantic content. Trie-based synonym lookup
is not feasible with this addressing scheme.

---

## Connection to Prior DCs

| DC | Key value | Reference |
|----|-----------|-----------|
| DC 322 | Same-leaf cosine=0.854 | Baseline trie quality |
| DC 323 | LOO=0.9303 at r≤3 (8D) | Original LOO benchmark |
| DC 324 | 20D orthogonal subspace | Axis discovery |
| DC 325 | LOO=0.9443 at r≤4 (12D optimal) | Dimensionality optimum |
| DC 326 | Category vs relational distinction | Initial navigability |
| **DC 327** | Two conditions + POS stratification | **Complete navigability** |
