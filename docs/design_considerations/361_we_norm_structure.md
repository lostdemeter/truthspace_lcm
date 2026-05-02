# DC 361: W_E Norm Structure

**Day 191 | W_E embeddings occupy a tight shell near the unit sphere.
Norm encodes token semantic specificity: letters/digits > content words >
function words. Norm does not predict relational retrieval accuracy.
Direction, not magnitude, is the primary information carrier in W_E.**

---

## Overview

Day 190 measured the L2 norm of all 151,936 token embeddings in Qwen2-1.5B-Instruct
and tested whether norm variation encodes token importance, frequency, or
predictive utility for relational retrieval.

---

## Finding 1: W_E Is Near-Unit-Sphere

```
Full norm distribution (V=151,936):
  Mean:  0.6514
  Std:   0.0906
  Min:   0.3764
  Max:   1.1716

Percentiles:
  p1:   0.424     p10: 0.547
  p25:  0.594     p50: 0.648
  p75:  0.713     p90: 0.765
  p95:  0.796     p99: 0.870
```

All 151,936 embedding vectors have norms in [0.38, 1.17]. The distribution
is tight and unimodal, centered near 0.65. There are no outlier tokens
with norm >> 1 or norm ≈ 0.

**Why?** Qwen2 uses RMSNorm throughout the transformer layers. During training,
the embedding matrix receives gradient updates that are normalized by
the downstream RMSNorm operations. This creates a feedback loop that
keeps embedding norms in a bounded range without explicit norm regularization.

The practical consequence: all W_E vectors live in a thin shell in ℝ^1536.
They are approximately (but not exactly) unit-normalized.

---

## Finding 2: Norm Encodes Semantic Specificity

```
Token category     Mean norm    Interpretation
─────────────────────────────────────────────────────────────────
function_words      0.55        Most polysemous, frequent, bleached
content_words       0.59        Moderate specificity
digits (0-9)        0.71        Specific, stable, numeric identity
letters (a-z)       0.72        Maximally specific (26 unique values)
```

**The pattern:** tokens that serve fewer distinct semantic roles have HIGHER
norm. Letters have exactly one role per letter. Digits have one numeric role.
Content words have moderate ambiguity (e.g., "light" = weight OR photons).
Function words ("the", "is", "of") are maximally context-dependent — they
appear in every conceivable semantic context, accumulating contradictory
gradients that partially cancel, leaving shorter embeddings.

**Mechanism:** In SGD training, each occurrence of a token contributes
a gradient update to W_E[token_id]. For a function word used in 10^9
contexts, gradients point in many directions and partially cancel.
For a letter used in 10^7 contexts that always means the same thing,
gradients reinforce. The resulting norm reflects gradient coherence
across training contexts.

---

## Finding 3: Gender Norm Asymmetry

```
Masculine/Feminine norm pairs:
  man   (0.65)  >  woman   (0.55)  Δ = +0.10
  boy   (0.61)  >  girl    (0.55)  Δ = +0.06
  prince(0.58)  >  princess(0.55)  Δ = +0.03
  actor (0.58)  ≈  actress (0.57)  Δ = +0.01
  king  (0.59)  ≈  queen   (0.59)  Δ =  0.00   ← royalty equal
```

Masculine forms have consistently higher norm than feminine counterparts,
with the gap proportional to frequency difference in training data.
"Man" appears far more frequently than "woman" in English corpora;
this frequency advantage translates to more gradient updates and slightly
higher norm. "King" and "queen" appear with roughly equal frequency in
historical/literary text (balanced royal context) → equal norms.

**Important:** this is a corpus bias artifact, not a semantic property.
The norm asymmetry is an imprint of training data distribution.

---

## Finding 4: Norm Does Not Predict Retrieval Accuracy

For the 9 country→capital LOO pairs:

```
France→Paris:     norm_diff = 0.02  ✓
Germany→Berlin:   norm_diff = 0.03  ✓
Italy→Rome:       norm_diff = 0.03  ✗  ← only failure
Spain→Madrid:     norm_diff = 0.02  ✓
Japan→Tokyo:      norm_diff = 0.02  ✓
China→Beijing:    norm_diff = 0.01  ✓
Russia→Moscow:    norm_diff = 0.01  ✓
Greece→Athens:    norm_diff = 0.02  ✓
Sweden→Stockholm: norm_diff = 0.04  ✓

Mean |diff| CORRECT: 0.02
Mean |diff| WRONG:   0.03
```

The single LOO failure (Italy→Rome) has norm diff 0.03, nearly identical
to correct pairs. The failure cannot be predicted from norm information —
it is purely a geometric direction failure (Rome is a common word name
with historical/religious uses that pull its embedding slightly off-axis).

**Norm-as-signal hypothesis: REJECTED.**

W_E norm variation (0.65 ± 0.09) is too small to encode meaningful
magnitude signals for relational retrieval. The tight norm distribution
effectively forces ALL information into the angular direction of embeddings.

---

## Implication: Cosine Similarity Is Optimal

Since all W_E embeddings lie in a thin norm shell (σ/μ = 14%), the dot
product `W_E[a] · W_E[b]` ≈ `|a| × |b| × cos(θ)` ≈ `0.65² × cos(θ)`.
The scalar `0.65²` is approximately constant across all pairs, making
dot product equivalent to cosine similarity (up to a constant factor).

However, the 14% norm variation means tokens at the high end (letters,
digits, norm≈0.72) would be disproportionately favored by dot product
over tokens at the low end (function words, norm≈0.55). For relational
retrieval where we compare words of the same category (country→capital,
both near norm 0.54), the distinction is irrelevant.

**For cross-category retrieval** (e.g., ranking a function word against a
proper noun), cosine similarity would give a more accurate comparison
than raw dot product.

---

## The Direction-Is-Everything Principle

Combining the norm findings with earlier arc results:

```
W_E encodes information as DIRECTION on an approximate unit sphere.

Evidence:
  1. Norms are near-constant (mean=0.65, std=0.09, σ/μ=14%)
  2. Relational directions are stable (0.900 for capitals, constant across layers)
  3. Norm does not predict retrieval accuracy
  4. Sub-token composition fails (sub-tokens have directions, not the right ones)
  5. SVD: signal lives in top-1 direction (mean), noise in remaining dims
  6. Cosine similarity (directional measure) is the right metric

The W_E space is best modeled as a spherical code:
  - Each token occupies a point on an approximate sphere in ℝ^1536
  - Semantic relationships are encoded as angular relationships
  - Adding a mean direction = rotating on the sphere
  - The sphere is high-dimensional enough that many nearly-orthogonal
    directions can coexist (supports ~1536/2 orthogonal relations)
```

---

## Summary

```
Finding                              Value
─────────────────────────────────────────────────────────────────
Norm range                           [0.38, 1.17]
Norm mean / std                      0.65 / 0.09  (σ/μ = 14%)
Function words norm                  0.55 (lowest)
Letter/digit norm                    0.72 (highest)
Gender norm asymmetry                masculine > feminine (corpus bias)
Norm predictive for retrieval        No (norm-as-signal rejected)
Primary information carrier in W_E   Direction (not magnitude)
Optimal similarity metric            Cosine similarity
```

---

## Files

- `expedition_day190_norm_structure.py` — norm distribution experiment
- `day190_norm_structure.json` — results
- `358_activation_vs_token_space.md` — W_E stability
- `359_we_relational_dimensionality.md` — SVD direction analysis
- `360_we_coverage_and_gaps.md` — coverage analysis
