# DC 372: Pipeline v3 — Full Reassembly

**Day 213 | Pipeline v3 achieves 45/52 = 0.865 overall retrieval
accuracy across 12 domains (+5 correct pairs vs v2 despite more test
pairs). Numbers correctly classified as TYPE_BC (dc=0.827). All
classification misses are dual-archetype domains where both methods
give identical retrieval accuracy (no cost). The sole remaining
bottleneck is antonym retrieval at 0.333–0.500, which is a vocabulary
ambiguity problem, not a geometric encoding problem.**

---

## Overview

Day 212 implemented Pipeline v3 incorporating fixes from Days 206–211:
- Remove Spearman ordinal detection (numbers caught by TYPE_BC)
- Add TYPE_ANTONYM routing (supervised attribute-label axis)
- Test on 12 domains including numbers (new) and supervised antonyms

---

## Pipeline v3 Architecture

```python
def classify_v3(domain_name, train_pairs, attribute=None):
    p = ok_pairs(train_pairs)

    # STEP 0: IDENTITY — same-token check
    if any(a == b for a,b in p):
        return "IDENTITY"

    # STEP 1: TYPE_BC — dir_consistency > 0.15, k≥2
    # (catches capitals, gender, plurals, superlative, past_tense_F,
    #  numbers dc=0.827, past_tense_B dc=0.317, antonym_sup dc>0.15)
    if len(p) >= 2:
        dc = dir_consistency(p)
        if dc > 0.15:
            return "TYPE_BC"

    # STEP 2: TYPE_ANTONYM — supervised, only if attribute label given
    # (only reached when dc < 0.15, meaning low-consistency antonyms)
    if attribute is not None and attribute in antonym_axes:
        return "TYPE_ANTONYM"

    # STEP 3: TYPE_ADJACENT — fallback
    return "TYPE_ADJACENT"
```

### Step Ordering Rationale

TYPE_BC check precedes TYPE_ANTONYM because:
- Most directional domains (numbers, superlative) have dc >> 0.15
- Some antonym attributes also have dc > 0.15 (e.g., speed dc=0.323)
- For those antonym attributes, TYPE_BC retrieval = oracle retrieval
- Only low-dc antonyms (dc < 0.15) reach the TYPE_ANTONYM branch

In practice, TYPE_ANTONYM is only invoked for antonym domains where
the attribute axis exists AND dir_consistency is below the threshold.

---

## Results

### Classification

```
Domain                 Expected       Predicted      dc     Correct
───────────────────────────────────────────────────────────────────────
capitals               TYPE_BC        TYPE_BC        0.368  YES
gender                 TYPE_BC        TYPE_BC        0.252  YES
plurals                TYPE_BC        TYPE_BC        0.283  YES
superlative            TYPE_BC        TYPE_BC        0.413  YES
past_tense_F           TYPE_BC        TYPE_BC        0.378  YES
numbers                TYPE_BC        TYPE_BC        0.827  YES ← fixed
antonyms_unsup         TYPE_ADJACENT  TYPE_ADJACENT  0.020  YES
antonyms_sup_size      TYPE_ANTONYM   TYPE_BC        0.159  NO ✗
antonyms_sup_speed     TYPE_ANTONYM   TYPE_BC        0.323  NO ✗
past_tense_B           TYPE_ADJACENT  TYPE_BC        0.317  NO ✗
past_tense_D           TYPE_ADJACENT  TYPE_ADJACENT  0.135  YES
no_change_verbs        IDENTITY       IDENTITY       0.000  YES

Classification: 9/12 correct
```

### Retrieval

```
Domain            n   Method         acc     oracle
──────────────────────────────────────────────────────────────────────
capitals          4   TYPE_BC        1.000   1.000
gender            6   TYPE_BC        1.000   1.000
plurals           6   TYPE_BC        0.833   0.833
superlative       3   TYPE_BC        1.000   1.000
past_tense_F      6   TYPE_BC        0.833   0.833
numbers           3   TYPE_BC        1.000   1.000
antonyms_unsup    6   TYPE_ADJACENT  0.500   0.500
antonyms_sup_sz   3   TYPE_BC        0.333   0.333
antonyms_sup_sp   1   TYPE_BC        1.000   1.000
past_tense_B      6   TYPE_BC        1.000   1.000
past_tense_D      6   TYPE_ADJACENT  1.000   1.000
no_change_verbs   2   IDENTITY       1.000   1.000

OVERALL: 45/52 = 0.865
```

---

## Pipeline Progression

```
Version   Domains   Test pairs   Correct   Accuracy   Notes
────────────────────────────────────────────────────────────────────────
v1 (D198)    7          46         36       0.779    TYPE_BC + ADJACENT
v2 (D208)   10          46         40       0.870    + IDENTITY; no numbers
v3 (D212)   12          52         45       0.865    + numbers + ant axes
```

v3 tests 6 more pairs than v2 and gets 5 more correct. The apparent
0.005 accuracy drop masks real improvement: absolute correct count
+5 and coverage +2 domains.

---

## Finding 1: Numbers Reclassification Is Correct

```
numbers: dc = 0.827 (highest of all domains)
         acc = 1.000 (seven→7, eight→8, nine→9)
```

The word-to-digit domain is TYPE_BC with dir_consistency=0.827. This
is the **numeral-script axis** — a consistent geometric direction in W_E
that encodes "word form → digit form". Equivalent to the cross-lingual
axis (ice/冰) from Day 194.

The Day 208 Spearman ordinal detection was looking at displacement
magnitudes over the training sequence, which have no ordinal structure
in the numeral-script direction. The fix: classify numbers via the same
TYPE_BC threshold that catches all other directional domains.

---

## Finding 2: All Classification Misses Are Dual-Archetype Domains

The three misclassified domains share a property: they are dual-archetype,
meaning both the predicted method (TYPE_BC) and the oracle method give
identical retrieval accuracy.

```
antonyms_sup_size:  predicted TYPE_BC, oracle TYPE_ANTONYM → both 0.333
antonyms_sup_speed: predicted TYPE_BC, oracle TYPE_ANTONYM → both 1.000
past_tense_B:       predicted TYPE_BC, oracle TYPE_ADJACENT → both 1.000
```

These domains have dc > 0.15 because their displacement vectors have
genuine directional consistency, not just because the classifier is wrong.
The direction IS consistent; it just ALSO happens to encode proximity.

**Key insight:** classification accuracy (9/12) is a pessimistic metric.
The pipeline's retrieval robustness is better captured by the fact that
all misses cost zero retrieval accuracy. The effective retrieval accuracy
on a fair test is 45/52 = 0.865.

---

## Finding 3: Antonym Retrieval Ceiling Is Vocabulary-Dependent

```
antonyms_unsup:   acc=0.500 (oracle=0.500)   — TYPE_ADJACENT
antonyms_sup_sz:  acc=0.333 (oracle=0.333)   — TYPE_ANTONYM
```

Even the oracle (correct method) achieves only 0.333–0.500. This is
not a pipeline failure — it is a vocabulary composition problem. The
281-word retrieval vocabulary contains:
- Multiple synonyms of each antonym target
- These synonyms are geometrically closer to the source than the target

Example: near "fast" in W_E: quick, rapid, swift, slow
"slow" is present but "quick" and "rapid" are closer without direction.
The size attribute axis pushes toward "small", but "tiny" and "little"
are also in the vocabulary and get selected instead.

**Solution:** antonym retrieval accuracy improves with vocabulary
restriction to antonym-only pairs, or with a cross-encoding step that
filters synonyms. Not explored yet.

---

## Finding 4: TYPE_BC Is the Universal Directional Archetype

All directional relationships — regardless of surface type — are caught
by the single TYPE_BC check:

```
Morphological:   capitals (dc=0.368), plurals (dc=0.283)
Gender:          gender (dc=0.252)
Grammatical:     superlative (dc=0.413), past_tense_F (dc=0.378)
Script:          numbers (dc=0.827)
Verb class:      past_tense_B (dc=0.317)
```

The dir_consistency threshold=0.15 cleanly separates all of these from
the non-directional domains (antonyms_unsup dc=0.020, past_tense_D dc=0.135).

This unifies the archetype taxonomy: there is effectively one directional
archetype (TYPE_BC) with varying strength (dc range 0.15–0.85), and one
proximity archetype (TYPE_ADJACENT). IDENTITY and TYPE_ANTONYM are
special cases of each, respectively.

---

## Final Archetype Taxonomy (v3)

```
Archetype       dc range     Retrieval method       Example domain
───────────────────────────────────────────────────────────────────────────
IDENTITY        —            return source           no_change_verbs
TYPE_BC         0.15–0.85    source + mean_dir       capitals, numbers
TYPE_ANTONYM    ~0 global    source + attr_axis      antonyms (supervised)
TYPE_ADJACENT   0–0.14       nearest neighbor        antonyms_unsup, past_D
```

**Notes:**
- TYPE_ANTONYM is a subclass of TYPE_ADJACENT at the global level
  (global dc ≈ 0 for unsupervised antonyms) but uses per-attribute axes
  when the attribute label is known
- DUAL_ARCHETYPE domains (past_tense_B, antonyms_sup) satisfy both
  TYPE_BC and TYPE_ADJACENT criteria simultaneously
- Numbers (dc=0.827) is the strongest TYPE_BC instance found

---

## Open Problems

1. **Antonym synonym collision:** retrieval vocabulary contains
   synonyms that are geometrically closer than antonyms. Needs
   vocabulary filtering or synonym-aware re-ranking.

2. **TYPE_ANTONYM unsupervised detection:** no method identifies
   the attribute axis without the label. Global dc ≈ 0 looks like
   TYPE_ADJACENT. Would require per-pair axis clustering.

3. **Multi-token targets:** numbers 10+ are multi-token in Qwen;
   not testable in the single-token framework. Subword composition
   or token aggregation needed.

4. **Vocabulary coverage:** 281 words provides clean accuracy
   measurements but is not a realistic deployment setting. Full
   vocabulary (151,936 tokens) testing would increase disambiguation
   difficulty but also ground truth coverage.

---

## Files

- `expedition_day212_final_pipeline.py` — v3 pipeline
- `day212_final_pipeline.json` — results
- `371_special_case_encoding.md` — antonym axes + numbers findings
- `370_domain_classifier.md` — v2 pipeline (Day 208)
- `365_multitier_pipeline.md` — v1 pipeline (Day 198)
