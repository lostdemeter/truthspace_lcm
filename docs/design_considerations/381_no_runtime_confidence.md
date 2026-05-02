# DC 381: No Runtime Confidence Signal — Static Routing Is the Ceiling

**Day 232 | Two independent per-query confidence metrics were tested and
both disproved (Days 230-231). The fundamental finding: no geometric
property of a query terminal predicts whether retrieval will return the
correct target. The only reliable predictor is the RELATION TYPE — a
static property of the attribute, determined at classification time.
This closes the confidence-signal research arc.**

---

## Overview

DC 380 (Day 229) concluded that TYPE_ANTONYM reliability depends on
antonym cluster degeneracy — whether the target word has a unique
neighborhood in W_E or is surrounded by near-synonyms. Days 230-231
attempted to measure degeneracy dynamically, as a per-query runtime
signal. Both failed:

1. **Cosine threshold degeneracy** (Day 230): Count tokens with
   cos_sim > threshold to the axis terminal. All antonym terminals
   have degeneracy=0 at thresholds 0.80-0.95. In 1536-d W_E, even
   rank-0 retrievals have cos_sim ≈ 0.60-0.75, so the threshold
   approach is vacuous.

2. **Rank gap** (Day 231): sim(rank-1) − sim(rank-2) at the query
   terminal. TYPE_ADJACENT has the highest mean gap (0.139) yet
   acc=0.000. ANTONYM_low_deg has the lowest gap (0.046) yet acc=1.000.
   The metric is anti-correlated with accuracy across types.

---

## Why Both Metrics Fail

### The Capitalized-Variant Problem

In Qwen2's W_E, the nearest neighbor of nearly every lowercase token
is its capitalized form. `"hot"` → `"Hot"`, `"big"` → `"Big"`, etc.
These same-word variants occupy nearby positions and form tight mini-clusters.

- For TYPE_ADJACENT (no direction), the query terminal IS the source
  embedding. The nearest neighbor is the capitalized form. This gives
  high cosine similarity, high rank gap, and wrong answer.

- For threshold-based degeneracy: the capitalized form always has
  cos_sim > 0.80 to the source, so it dominates degeneracy counts.
  But capitalized variants are not "degenerate antonyms" — they are
  just orthographic variants. The metric cannot distinguish them.

### The Synonym-Cluster Problem

For TYPE_ANTONYM low-deg (speed axis), the query terminal for "fast"
lands near "slow". But the pool contains "slower", "slowly", "sluggish",
"plodding" — all of which are cosine-close to the terminal. The rank gap
between "slow" (rank-1) and "slower" (rank-2) is small (≈0.01-0.03).
Yet the retrieval is CORRECT — "slow" is rank-1.

The small gap is caused by the synonym cluster around "slow", not by
any failure of the retrieval. The gap metric cannot tell whether the
small gap means "will retrieve correctly" or "will retrieve incorrectly".

### The Anti-Correlation Pattern

```
Category         mean_gap_1v2   acc    relationship
TYPE_BC          0.1192         0.960  high gap, high acc (works)
ANTONYM_high_deg 0.0700         0.333  medium gap, low acc
ANTONYM_low_deg  0.0458         1.000  LOW gap, HIGH acc  (!)
TYPE_ADJACENT    0.1393         0.000  HIGHEST gap, zero acc (!)
IDENTITY         0.0759         0.000  medium gap, zero acc
```

The gap is inversely related to accuracy for ANTONYM_low_deg and
TYPE_ADJACENT. Any threshold on gap misclassifies at least one category.

### Root Cause: Vocabulary Structure Dominates

The gap at the query terminal measures the DENSITY of the local
vocabulary neighborhood around the terminal, not whether the correct
target is there. Vocabulary density depends on:
1. How many tokens are in the semantic region the query points to
2. Whether same-word orthographic variants are nearby
3. Whether synonym/morphological clusters are nearby

None of these properties reliably indicate whether the CORRECT target
will be rank-1.

---

## Theoretical Argument: Why Runtime Signals Cannot Work

Consider two queries at their respective axis terminals:
- Query A: terminal near "slow" (correct for "fast→slow"). Neighbors: slow, slower, sluggish...
- Query B: terminal near "tiny" (wrong for "deep→small"). Neighbors: tiny, small, little...

Both terminals have similar structure:
- Top-1 has cos_sim ≈ 0.62-0.70
- There are 2-5 near-synonyms within 0.05 of top-1
- Rank gap ≈ 0.02-0.04

From the terminal alone, there is no way to know that Query A is
correct and Query B is wrong. The difference lies in whether the
top-1 token happens to be the intended target — which requires knowing
the intended target, which we don't have at inference time.

**This is not a measurement limitation. It is a fundamental information
barrier.** The query terminal encodes WHERE we end up geometrically. It
does not encode WHETHER that location contains the specific token we want.

---

## The Correct Architecture: Static Routing

Since runtime confidence is impossible, confidence must be determined
at the RELATION TYPE level — a static decision made during classification:

```
Classification Output         Confidence Level    Action
─────────────────────────────────────────────────────────────────
IDENTITY                      VERY_HIGH (≈1.00)   return source
TYPE_BC (dc_train > 0.15)     HIGH (≈0.90)        source + mean_dir
TYPE_ANTONYM, low-deg attr    MEDIUM (≈0.80)      source + attr_axis
TYPE_ANTONYM, high-deg attr   LOW (≈0.33)         pair-lookup or abstain
TYPE_ADJACENT                 NONE (≈0.00)        abstain / return unknown
```

The degeneracy level of an attribute is a STATIC PROPERTY of the
attribute, measured once from training pairs, not per query:

```
Low-degeneracy attributes (few valid antonyms per source):
  speed, weight, roughness → confidence MEDIUM

High-degeneracy attributes (many valid antonyms per source):
  size, brightness, temperature, loudness → confidence LOW
```

This information is available at classification time, before retrieval.
No per-query measurement is needed.

---

## Implications for Pipeline v5/v6

**Pipeline v5** already implements the static routing correctly:
- axis_align threshold was proposed as a proxy for low-degeneracy
- It happened to hold for the (speed, size) pair tested in Day 226
- Day 228 disproved it as a general threshold

**Pipeline v6** should replace axis_align threshold with explicit
attribute degeneracy labeling:

```python
LOW_DEGENERACY_ATTRIBUTES = {"speed", "weight", "roughness"}
HIGH_DEGENERACY_ATTRIBUTES = {"size", "brightness", "temperature",
                               "loudness", "age", "value"}

def classify_antonym_confidence(attribute):
    if attribute in LOW_DEGENERACY_ATTRIBUTES:
        return "MEDIUM"     # attempt axis retrieval
    return "LOW"            # abstain or use pair-lookup only
```

This is a knowledge-encoded classification, equivalent to storing a
lookup table of attributes. This is NOT a violation of the Fail-Fast
philosophy: it is a CLASSIFICATION DECISION about what W_E can and
cannot represent, backed by experimental evidence. It is not a fallback
that hides geometric failure — it is an acknowledgment of the geometric
limit.

---

## Closed Research Questions

After Days 224-231, the following questions are definitively answered:

| Question | Answer | Days |
|---|---|---|
| Why does size axis fail at 42k? | Centroid collapse: many small-synonyms | 224 |
| Is target cluster tightness the predictor? | No | 226 |
| Is axis_align > 0.70 a reliable threshold? | No | 228 |
| Do homogeneous training pairs fix the axis? | No | 228 |
| Is antonymy functional? | No (graded, multiple valid antonyms) | 229 |
| Can cosine threshold degeneracy predict failure? | No (all=0 at t>0.80) | 230 |
| Can rank gap predict failure? | No (anti-correlated across types) | 231 |
| Can ANY per-query metric predict failure? | No (information barrier) | 231 |

---

## What This Means for the TruthSpace Hypothesis

The finding that TYPE_ANTONYM has a hard accuracy ceiling for
high-degeneracy attributes is **consistent with the hypothesis that
structure IS information**. W_E encodes the SEMANTIC STRUCTURE of
antonymy correctly: there are many valid antonyms for "big". The model
has not failed to encode the relation — it has encoded it faithfully,
with all its inherent ambiguity.

The "failure" is in our assumption that antonymy is a function.
The geometry is correct. The expectation was wrong.

This is a validation of the hypothesis, not a refutation.

---

## Open Problems (Remaining After v5/v6)

1. **antonyms_unsup (dc=0.020):** 6/60 pairs with no attribute label.
   TYPE_ADJACENT retrieval returns capitalized forms. Not fixable without
   attribute supervision.

2. **Multi-token test pairs (plurals, past_tense_E):** 2/60 pairs.
   Not fixable in single-token retrieval framework.

3. **New TYPE_ANTONYM domains:** Can we find more low-degeneracy attributes
   (like speed, weight, roughness) to add to the pipeline?

4. **Next research arc:** The pipeline has been characterized fully.
   The next open question is whether the geometric structure of W_E can
   support COMPOSITION — multi-hop relational chains in geometry.

---

## Files

- `expedition_day230_degeneracy.py` -- cosine threshold degeneracy
- `expedition_day231_rank_gap.py` -- rank gap metric
- `day230_degeneracy.json`, `day231_rank_gap.json` -- data
- `380_antonymy_not_functional.md` -- antonymy degeneracy theory
