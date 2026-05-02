# DC 359: W_E Relational Dimensionality — SVD Analysis

**Day 187 | SVD of difference-vector matrices across TYPE_BC domains reveals
flat singular spectra, domain-specific directions, and a shared country-axis
between co-domain relations. No universal relational basis exists in W_E.**

---

## Overview

Day 186 applied SVD to matrices of normalized difference vectors
(W_E[target] - W_E[source]) for five relational domains to ask:

1. How many dimensions does each relation occupy?
2. Do different relations share principal directions?
3. Is there a universal relational basis for W_E?

**Answer in brief:** No. Each relation is high-dimensional and domain-specific.
Country-type relations share a partial axis (~0.40). Gender and antonyms are
orthogonal to all other relations.

---

## The Signal/Noise Decomposition

For a matrix D of n normalized difference vectors (each in ℝ^H, H=1536):

```
D = U Σ V^T

σ₁ = signal strength   (how consistent the direction is)
σ₂, σ₃, ... = noise    (per-pair deviations from mean direction)
```

**Signal/noise ratio (SNR) = σ₁² / Σᵢ₌₂ σᵢ²**

The key question: does σ₁ dominate (rank-1 structure, clean direction)
or are all singular values equal (full-rank, random directions)?

---

## Singular Value Spectra

```
Domain            n    σ₁    σ₂/σ₁  σ₃/σ₁  σ₅/σ₁  Interpretation
───────────────────────────────────────────────────────────────────────
capitals         14   (top)  0.429  0.392  0.378   Moderate signal
languages        12   (top)  0.280  0.275  0.257   Stronger signal (sv2 drops)
gender            8   (top)  0.646  0.615  0.585   Weak signal (flat)
country_currency  7   (top)  0.575  0.525  0.298   Moderate, fast tail drop
antonyms          9   (top)  0.879  0.861  0.823   Near-uniform (noise floor)
```

**Interpretation by domain:**

- **languages** (sv2/sv1 = 0.280): The language direction is the most focused
  in W_E. French, German, Italian, Russian all point in nearly the same direction
  from their respective country embeddings. Low noise = high signal.

- **capitals** (sv2/sv1 = 0.429): Capital direction has more variation. Paris is
  in a slightly different direction from France than Tokyo is from Japan.
  Continental differences create sub-clusters.

- **gender** (sv2/sv1 = 0.646): Very flat — king→queen, man→woman, actor→actress
  all point in different micro-directions. The mean direction works (LOO=0.875)
  but the individual directions scatter.

- **antonyms** (sv2/sv1 = 0.879): Near-uniform spectrum. Antonym difference
  vectors are essentially random directions in W_E. The relation is NOT
  directionally encoded (confirms Day 162 and Day 184 findings).

---

## Cross-Domain Principal Direction Alignment

Cosine similarity between top-1 right singular vectors (|cos|):

```
                capitals  languages  gender  currency  antonyms
capitals         1.000      0.430   0.013    0.390     0.018
languages        0.430      1.000   0.005    0.336     0.013
gender           0.013      0.005   1.000    0.001     0.033
currency         0.390      0.336   0.001    1.000     0.008
antonyms         0.018      0.013   0.033    0.008     1.000
```

**Three independent relational subspaces in W_E:**

```
COUNTRY-AXIS (capitals ≈ languages ≈ currency, cos ~0.35-0.43):
  The top-1 direction is shared because all three use country embeddings
  as sources. The "country" cluster occupies a subspace of W_E, and all
  country→X relations start from that subspace → correlated directions.

GENDER-AXIS (cos ~0 to all others):
  King, man, boy, prince, actor are in a completely different region
  of W_E. The gender direction is isolated.

ANTONYM-SPACE (cos ~0 to all others):
  No principal direction — antonym differences are random in W_E.
```

---

## No Universal Relational Basis

The combined SVD across all 41 TYPE_BC difference vectors:

```
k directions needed to explain:
  25%: k = 1
  34%: k = 2
  40%: k = 3
  50%: k = 5
  62%: k = 10
  81%: k = 20
```

A universal W_E relational basis would need ~20 directions to capture 81%
of the variance across all TYPE_BC relations. This is not a compact basis —
it approaches the dimensionality of the sample itself (41 vectors → ~33 dims
needed for 95%).

**Applying the top-1 combined direction to individual domains:**
```
capitals:        0.571  (degrades from 0.900 — mixing country relations hurts)
languages:       1.000  (aligns well with combined because many language pairs)
country_currency: 0.286 (poor — currency direction doesn't align with combined)
```

The combined direction is dominated by the language and capital pairs
(largest groups) and degrades performance on currency pairs.

---

## Why the Mean Direction Still Works

The mean direction LOO approach (DC 354) achieves 0.900 for capitals despite
the high-dimensional noise. Why does the mean work if the spectrum is flat?

**The concentration of measure:** In H=1536 dimensions, the mean of n random
unit vectors has norm ~√n / √H. For n=14 capital pairs, the mean direction
has a meaningful signal because σ₁/H is amplified by the signal component
being consistent while the noise components cancel.

```
Mean direction = (1/n) Σ dᵢ = (1/n)(σ₁ u₁ v₁ᵀ + noise)
                              ≈ (σ₁/n) u₁ v₁ᵀ  (noise cancels as 1/√n)
```

The retrieval accuracy depends on how well the mean direction aligns with
the true direction for the test pair. Since the true direction has a component
along v₁ (shared signal), the mean direction has the right component.

This is why LOO works: **averaging is dimension-independent.** It doesn't
matter if the direction is in 1 or 1536 dimensions — the mean suppresses
individual variation and amplifies the shared signal.

**Minimum sample size** for reliable direction averaging:
```
k required for LOO accuracy ≥ 0.80:
  capitals:  k ≈ 5  (from Day 162 saturation curve)
  languages: k ≈ 3  (strong signal, fast saturation)
  gender:    k ≈ 7  (weak signal, slower saturation)
```

The sample size needed scales with sv2/sv1 — the higher the noise-to-signal
ratio, the more pairs needed to average out the noise.

---

## Implications for TruthSpace Architecture

**1. No global relational index needed.**

W_E does not have a fixed set of "relational axes" that can be precomputed
once and applied universally. Each new relation requires its own direction
estimation from training examples.

**2. Domain separation is structurally free.**

Country-type relations share a principal direction, which means country→capital
training pairs also slightly improve country→language retrieval (and vice versa).
Cross-domain contamination (mixing training pairs from related domains) is
approximately harmless because the shared country-axis helps both.

**3. The signal lives in the low-dimensional component.**

Even though eff_dim = n (all pairs contribute unique dimensions), the
retrieval signal lives in the top-1 direction (25% variance). The remaining
75% is pair-specific variation that doesn't generalize. This means:

```
D_effective = 1 dimension per domain (the mean direction)
D_noise = n-1 dimensions (per-pair deviations)
```

LOO averaging suppresses D_noise by 1/√n, making the 1D signal accessible.

**4. Relations from different source clusters are orthogonal by construction.**

Gender, country, and antonym relations occupy orthogonal subspaces of W_E
because their source words (men/women names, country names, adjectives)
occupy orthogonal subspaces. This is a self-organizing property of W_E:
words of the same semantic category are co-located, and relational directions
stay within those categorical subspaces.

---

## Summary

```
Finding                              Value
─────────────────────────────────────────────────────────────────
Effective dimensionality per domain  = n (all pairs contribute)
Meaningful signal dimensions         = 1 (mean direction)
Cross-domain alignment               capitals↔languages: 0.43
                                     capitals↔currency: 0.39
                                     gender↔all: ~0.01
Universal TYPE_BC basis size         ~20 directions for 81% var
Mean LOO works because               averaging suppresses n-1
                                     noise dimensions in H=1536
```

---

## Files

- `expedition_day186_svd_directions.py` — SVD experiment
- `day186_svd_directions.json` — results
- `358_activation_vs_token_space.md` — W_E stability findings
- `354_multirelation_composition.md` — multi-hop chains
