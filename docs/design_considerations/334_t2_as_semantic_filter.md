# DC 334: T2 as Semantic Filter

**Days 124-125 | T2 and d_k as candidate ranking signals**

---

## Background

DC 333 established that T2 axes are a mixture of intrinsic syntactic geometry
(stable, universal) and probe-specific semantic directions (variable, content-
dependent). Days 124-125 test the practical consequence: **can T2 and d_k
be used to rank candidate completions?**

---

## Experiment Summary

Both days tested the same 12 factual prompts across 6 categories:
- capitals, languages (semantic/encyclopedic)
- hypernyms, antonyms (semantic/relational)
- tense, gender (syntactic/morphological)

Three tiers of candidates per prompt:
- Correct (1-3 tokens): Paris, dog/animal/mammal, walked/ran/...
- Plausible-wrong (4): London/Rome/Berlin, cat/rabbit/horse
- Semantically-wrong (4): banana, quickly, stone, because

---

## Method A: Isolated T2 vs Context Anchor (Day 124)

Score(candidate) = cosine(T2_isolated(candidate), T2_context_anchor(prompt))

**Result: MRR = 0.540 vs random 0.314 (+72%)**

This method compares the T2 address of the candidate word (standalone) against
the T2 address of the prompt's last token. It works because:
- Words in the same semantic category cluster in T2 space
- City names have similar T2 addresses (cluster together)
- Past-tense verbs share a T2 signature distinct from present-tense
- Semantically-wrong words (banana) are far from city/verb clusters

**Limitation**: Cannot rank WITHIN a category (Paris vs London both cluster
in the "proper noun / city" region of T2 space).

---

## Method B: In-Context T2 (Day 124)

Score(candidate) = cosine(T2_{prompt + candidate}(last_token), T2_context_anchor)

**Result: MRR = 0.244 vs random 0.314 (BELOW random)**

This method FAILS for semantic categories because semantically-wrong words
appended to a retrieval prompt generate LARGER hidden-state activations
(the LM is "surprised"), which produces higher T2 projections.

**Exception**: Works for syntactic categories (tense, gender) — consistent
with Day 123's finding that syntactic axes are stable and intrinsic.

```
Category    Method B  wrong_sim  correct_sim
capitals        FAIL  +0.855      +0.653
tense           PASS  +0.680      +0.745
gender          PASS  +0.718      +0.942
```

---

## Method d_k: Entity Selector Signal (Day 125)

Score(candidate) = |dot(normed_h_L23(prompt + candidate), d_k)|

**Result: MRR = 0.549 vs random 0.314 (+75%)**

d_k at the candidate last-token position is useful but weak. The reason for
weakness: in the test setup, ALL candidates appear in retrieval context
("The capital of France is [candidate]"), so ALL get elevated d_k activation.
The correct/wrong margin is only ~15-20%.

This contrasts with Day 117 where d_k was 4.3x stronger in retrieval context
vs neutral context — that measured ENTITY vs NEUTRAL, not CORRECT vs WRONG
within the same retrieval context.

```
Category  dk_correct  dk_plausible  dk_wrong  order
capitals    0.0628      0.0569       0.0539    1/3
languages   0.0701      0.0701       0.0604    0/2
gender      0.0650      0.0559       0.0346    1/1 ✓
```

---

## Combined T2+d_k Ranking (Day 125)

Score = α × normalize(T2_iso) + (1-α) × normalize(d_k)

**Result: MRR = 0.595 at α=0.9 (T2 dominates)**

```
α=0.0 (d_k only): 0.549
α=0.8:            0.574
α=0.9:            0.595  ← optimum
α=1.0 (T2 only):  0.540
Random baseline:   0.314
```

At α=0.9, T2 contributes 90% of the signal. d_k provides a 10% marginal
improvement but adding more d_k weight degrades performance (d_k is too noisy).

---

## Two-Structure Picture: Roles Confirmed

```
Signal          Range     Role                  Limit
──────────────────────────────────────────────────────────────────────
T2 (12D)        category  CATEGORY FILTER       can't rank within category
d_k (1D)        weak      WITHIN-CAT SUPPLEMENT ~15-20% correct/wrong margin
T2 + d_k (α=0.9) combined BEST COMBINED SCORE  +89% vs random

Both signals from:
  T2: sentence-pair causal ablation → axis at Day78 optimal layer
  d_k: L23 H6 W_k SVD → right singular vector of key projection matrix
```

---

## The Missing Piece: Within-Category Discrimination

The T2+d_k combination achieves MRR=0.595, but it CANNOT reliably distinguish:
- Paris vs London vs Berlin (all cities with similar T2 addresses)
- Portuguese vs Spanish vs Italian (all Romance languages)

To rank WITHIN a category, a higher-resolution signal is needed:
- Full hidden state similarity at optimal layers
- Model log-probability P(candidate | prompt)
- Or a finer-grained representation than the 12D T2 address

The 12D T2 address is a **lossy compression** designed for discrimination
at the CATEGORY level, not the INSTANCE level. This is by design —
the φ-threshold binning deliberately maps many instances to the same address,
trading fine-grained discrimination for robustness and compactness.

---

## Summary

| Signal | MRR | Above Random |
|--------|-----|-------------|
| Random | 0.314 | — |
| T2 (isolated, Method A) | 0.540 | +72% |
| d_k (last token) | 0.549 | +75% |
| T2 + d_k (α=0.9) | **0.595** | **+89%** |
| T2 (in-context, Method B) | 0.244 | BELOW |

T2+d_k combined is the optimal geometric ranking strategy, achieving
+89% above random. The combination works best at α=0.9, confirming
T2 as the primary signal and d_k as a weak supplement.

**Next question**: Does model log-probability P(candidate | prompt) correlate
with T2 ranking? If T2 is a geometric proxy for model probability, this would
validate the TruthSpace hypothesis that geometric structure encodes predictive
information.
