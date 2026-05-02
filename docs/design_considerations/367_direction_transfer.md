# DC 367: Cross-Domain Direction Transfer

**Day 203 | Morphological directions transfer to unseen vocabulary with
domain-specific reliability. The plural direction is fully universal at
k=1: any single regular noun pair gives acc=0.917 on held-out nouns.
Superlatives are near-universal (k=1 acc=1.000 for regular adjectives).
Past tense is NOT universal — irregular conjugation classes each encode
a distinct geometric transformation. Cross-domain: all morphological
directions share a general "inflection" component in W_E, but only
in one direction (plurals→superlatives, not the reverse).**

---

## Overview

Day 202 tested whether TYPE_BC direction vectors generalize to completely
unseen vocabulary. Training pairs (6) and test pairs (12) were disjoint —
no word appeared in both sets. The candidate retrieval vocabulary was
expanded to 120 common English words covering all morphological classes.

---

## Results

```
EXP 1: Within-domain transfer (k=6 train → held-out test)
  plurals     acc=0.917  H@5=1.000  mean_rank=0.08
  superlative acc=1.000  H@5=1.000  mean_rank=0.00
  past_tense  acc=0.500  H@5=0.500  mean_rank=3.40

EXP 2: Zero-shot from k=1 (single training pair → same held-out test)
  plurals   [cat→cats]     acc=0.917  H@5=1.000  (IDENTICAL to k=6)
  plurals   [dog→dogs]     acc=0.917  H@5=1.000
  plurals   [house→houses] acc=0.917  H@5=1.000
  plurals   [tree→trees]   acc=0.917  H@5=1.000
  superlat. [big→biggest]  acc=1.000  H@5=1.000
  superlat. [fast→fastest] acc=1.000  H@5=1.000
  superlat. [old→oldest]   acc=0.000  H@5=1.000  (polysemy failure)
  past      [run→ran]      acc=0.500  H@5=0.800
  past      [eat→ate]      acc=0.200  H@5=1.000
  past      [go→went]      acc=0.500  H@5=0.900
  past      [see→saw]      acc=0.100  H@5=1.000

EXP 3: Cross-domain direction confusion
  plural_dir     → superlative targets: acc=0.667  (UNEXPECTED)
  superlative_dir → plural targets:     acc=0.000
  past_dir       → plural targets:      acc=0.583  (UNEXPECTED)
```

---

## Finding 1: Plural Direction Is Fully Universal at k=1

All four single-pair exemplars — different nouns, different phonological
patterns — yield **identical accuracy** on 12 held-out nouns:

```
cat→cats:     acc=0.917
dog→dogs:     acc=0.917
house→houses: acc=0.917
tree→trees:   acc=0.917
```

The 0.083 failure rate (1/12 nouns) is constant across all exemplars,
meaning there is exactly one noun in the test set that is hard for any
plural direction. The direction is not just stable — it is **fully
determined** from any single regular noun.

**Why?** The plural morpheme -s applies uniformly across thousands of
regular English nouns. Every occurrence of "cat" in training data co-occurs
with nearby contexts where "cats" also appears. The cat→cats direction is
reinforced by dogs→dog, trees→tree, books→book, etc. — all pointing the
same way. The signal is so redundant that any single pair recovers the
full direction.

This is the strongest evidence yet that **grammar is encoded geometrically
as a universal direction in W_E**, not as word-specific memorization.

---

## Finding 2: Superlative Is Near-Universal, With One Polysemy Failure

```
big→biggest:   acc=1.000  (k=1)
fast→fastest:  acc=1.000  (k=1)
old→oldest:    acc=0.000  (k=1, but H@5=1.000 — rank=1, not 0)
```

"big" and "fast" are unambiguous adjectives. "old" carries extra semantic
weight: age, antiquity, familiarity ("old friend"), experience ("old hand"),
and the proper noun domain ("Old Spice", "Old Testament"). This polysemy
shifts "old" off the clean adjective axis, corrupting the direction derived
from old→oldest.

H@5=1.000 confirms the correct target is rank=1 (second-best), not
completely off. The polysemy adds a small offset that moves the query just
past the correct target.

**Polysemy corrupts k=1 retrieval.** This is the same mechanism seen in
capitals: idiosyncratic associations of a specific word contaminate the
direction estimate. The fix is k≥2 — averaging two adjectives cancels
the polysemy offset.

---

## Finding 3: Past Tense Has Multiple Geometric Classes

```
run→ran:  acc=0.500  H@5=0.800
eat→ate:  acc=0.200  H@5=1.000
go→went:  acc=0.500  H@5=0.900
see→saw:  acc=0.100  H@5=1.000
```

The H@5 scores (0.8–1.0) confirm targets are in the top-5 candidates —
the retrieval is not random. But different irregular patterns produce
different k=1 directions:

```
Ablaut class (vowel change): run/ran, eat/ate, see/saw
Suppletive:                  go/went
```

"run→ran" and "go→went" both give acc=0.500 while "eat→ate" and "see→saw"
give 0.1–0.2. These differences reflect distinct geometric transformations:

- run/ran: front vowel mutation (u→a)
- eat/ate: different vowel pattern
- see/saw: different vowel pattern
- go/went: completely suppletive (from Old English "wendan")

Within-class transfer should work (run→ran predicts take→took?), but
cross-class transfer fails (run→ran cannot predict eat→ate). The "past
tense" relation is NOT a single direction — it is a family of TYPE_BC
directions, one per inflection class.

**This motivates a TYPE_BC subclassification:** within TYPE_BC, some
domains (regular morphology) have a single universal direction; others
(irregular morphology, factual associations) have class-specific directions.

---

## Finding 4: Asymmetric Cross-Domain Transfer

```
plural_dir → superlative targets:  0.667  ✓ transfers
superlative_dir → plural targets:  0.000  ✗ does not transfer
past_dir → plural targets:         0.583  ✓ partial transfer
```

This asymmetry is geometrically meaningful.

**All inflectional morphology shares a "root→inflected" component** in
W_E. Every morphological transformation (cat→cats, big→biggest, run→ran)
moves from a base form to a derived form. In W_E, base forms and their
inflections are displaced in a shared broad direction — let's call it the
**inflection axis**.

The plural direction aligns broadly with the inflection axis because the
plural is the most common English inflection (thousands of noun pairs).
It is a "fat" direction that captures the general inflection component.

The superlative direction aligns with the inflection axis too, but adds
a more specific adjective-space component. It is a "narrow" direction
orthogonal to plurals in the adjective subspace.

When you apply the plural direction to adjectives:
```
"hard" + plural_dir ≈ "hard" displaced toward inflected adjective space
                    → ends up near "hardest" (0.667)
```

When you apply the superlative direction to nouns:
```
"bird" + superlative_dir ≈ "bird" displaced toward adjective superlative space
                          → misses "birds" completely (0.000)
```

The plural direction subsumes the inflection axis; the superlative direction
is an adjective-specific elaboration that does not generalize to nouns.

---

## Revised TYPE_BC Subclassification

Based on Days 196-202, TYPE_BC should be split into three subclasses:

```
TYPE_BC_UNIV (universal direction):
  Marker: k=1 accuracy ≥ 0.85, consistent across all k=1 exemplars
  Domains: plurals (regular nouns)
  Property: single direction applies to ALL instances in the domain
  Min k needed: 1

TYPE_BC_RULE (rule-based, near-universal):
  Marker: k=1 accuracy ≥ 0.85 for unambiguous words, fails on polysemous
  Domains: superlative (regular adjectives)
  Property: universal rule but fails on polysemous anchor words
  Min k needed: 2 (averages out polysemy)

TYPE_BC_CLASS (class-specific direction):
  Marker: k=1 accuracy varies 0.1–0.5 across exemplars
  Domains: past_tense (irregular), capitals
  Property: direction varies by inflection class or cultural context
  Min k needed: 3–6 (must sample across classes or cancel idiosyncrasy)
```

---

## Implications

### Updated Pipeline k-Requirements

```
Subclass         Min k   Detection
──────────────────────────────────────────────────────────────
TYPE_BC_UNIV     1       Any k=1 acc ≥ 0.85
TYPE_BC_RULE     2       k=1 std high; k=2 std collapses
TYPE_BC_CLASS    6       k=1 std high; std high at k=2–4
TYPE_ADJACENT    0       dir_consistency < 0.10
TYPE_ORDINAL     0       Spearman ρ ≥ 0.85
```

### Cross-Domain Transfer as a Probe

The asymmetric cross-domain transfer (plural→superlative but not vice
versa) can be used as a **probe** for the inflection axis:

- Apply the plural direction to any word class
- If acc > 0.5 on held-out inflected forms → word class has a shared
  inflection component with nouns
- If acc ≈ 0 → word class lives in a separate subspace

This gives a geometry-based method to detect morphologically active
word classes without human labels.

---

## Files

- `expedition_day202_direction_transfer.py` — transfer experiment
- `day202_direction_transfer.json` — results
- `366_kshot_scaling.md` — k-shot saturation profiles
- `364_relational_encoding_archetypes.md` — archetype taxonomy
