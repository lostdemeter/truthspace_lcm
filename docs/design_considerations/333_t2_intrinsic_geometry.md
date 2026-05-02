# DC 333: T2 Intrinsic Geometry

**Days 119-123 | Characterization of axis stability, layered architecture, and syntactic/semantic split**

---

## Background

DC 332 characterized the T2 subspace across Days 113-118, confirming that:
- T2 axes correspond to genuine geometric directions in the LM
- The entity selector d_k is orthogonal to the T2 subspace
- The two structures together support factual retrieval

Days 119-123 extend this by asking: **How intrinsic are the T2 axes to the LM?**
Are they universal fixed-point geometric properties, or construction-specific probes?

---

## Finding 1: d_k Has Two Activation Modes (Day 119)

The entity selector d_k activates in two distinct modes:

```
Signal 1: Context-accumulation (last-token)
  Trigger: ANY multi-word prompt
  Magnitude: ~3x vs isolated (0.045 vs 0.017)
  Specificity: NOT retrieval-specific
  Function: general context integration artifact

Signal 2: Entity-retrieval (entity position)
  Trigger: Entity token in retrieval-structured context only
  Magnitude: ~4.3x vs isolated (0.071 vs 0.017)
  Specificity: Dormant in neutral context ("X is a country")
  Function: retrieval pointer — the semantically meaningful signal
```

Few-shot prompting does NOT amplify d_k further. Longer prompts with multiple
examples actually reduce last-token d_k (entity is diluted from final position).

**Implication**: The entity-position signal is the functionally important one
for factual retrieval. The last-token signal is a noise-like context accumulation
artifact that projects onto the same d_k direction.

---

## Finding 2: Layered Semantic Architecture (Day 120b)

Layer-sweep measuring Cohen's d for isolated word pairs reveals three distinct
processing stages:

```
Layer 1-5:   MORPHOLOGICAL IDENTITY
  Axes: comparative, hypernym, synonym, concrete, past_tense, causation, negation
  Peak Cohen's d: comparative=8.453 at L1 (morphological -er suffix detectable immediately)
  Character: surface lexical features, maximally ORTHOGONAL at L3 (offdiag=0.037)

Layer 10-15: SEMANTIC RELATIONS
  Axes: antonym
  Peak Cohen's d: antonym=2.347 at L10
  Character: semantic opposition — a mid-level relational concept

Layer 23-28: PRAGMATIC / DISCOURSE FEATURES
  Axes: gender, plural, passive, question
  Peak Cohen's d: gender=6.257 at L27, question=2.306 at L28
  Character: features requiring cross-reference, syntactic restructuring,
             or discourse-level information
```

The layered architecture explains the Day78 vs Day 120b discrepancy:
- Day78 used sentence-level last-token representations → late-layer optimal
- Day120b uses isolated word tokens → early-layer morphological features dominant
- These are different measurement instruments, not contradictions

**T2 Gram matrix most orthogonal at L3** (offdiag_mean=0.037). Axes become
more correlated as they ascend toward L28 (offdiag=0.088), reflecting
contextual integration of multiple features at higher layers.

---

## Finding 3: T2 Method Is Superior to Mean-Difference (Day 122)

Comparing T2 sentence-pair axes vs independently-derived Mean-Difference (MD)
axes from category word sets:

```
T2 vs MD alignment:
  Mean cos(T2, MD) = 0.155 — mostly divergent
  Aligned (cos > 0.5): 0/12 axes

But MD inter-axis independence WORSE than T2:
  MD offdiag mean: 0.139  (antonym/negation: |cos|=0.830!)
  T2 offdiag mean: 0.062  (antonym/negation: ~0.06)
```

The MD method on isolated words confounds categories that T2 successfully
separates:
- MD antonym direction ≈ MD negation direction (|cos|=0.83)
- T2 antonym and negation axes are nearly orthogonal

Classification accuracy comparison:
```
axis         MD_acc   T2_acc
gender        50.0%    86.7%   ← T2 wins: sentence context disambiguates
plural       100.0%    50.0%   ← MD wins: morphological plurality is lexical
past_tense    70.0%    86.7%   ← T2 wins: temporal context matters
```

**Conclusion**: T2 sentence-pair method is a form of CAUSAL ABLATION —
"change only the target word, hold context constant." This is a more precise
instrument for isolating specific semantic axes than word-mean-difference.
The low T2-MD alignment reflects method difference, not T2 being arbitrary.

---

## Finding 4: Syntactic Axes Are More Intrinsic Than Semantic Axes (Day 123)

Testing whether axis direction is stable across 10 independent sentence-pair
groups (2 pairs each, 20 total per axis):

```
Syntactic transformations (stable):
  question   0.604  STABLE  — inversion is always the same operation
  past_tense 0.486  PARTIAL — present→past morphology consistent
  negation   0.347  PARTIAL — X / not-X has consistent direction
  gender     0.387  PARTIAL — masculine/feminine shift is partially stable

Semantic transformations (variable):
  causation  0.095  VARIABLE — rain→flood ≠ fire→ash geometrically
  antonym    0.136  VARIABLE — hot/cold ≠ fast/slow ≠ good/bad
  hypernym   0.137  VARIABLE — dog→animal ≠ hammer→tool
  comparative 0.158 VARIABLE — fast→faster is morphological, but varies

Overall mean stability: 0.277 (VARIABLE)
```

**Why syntactic axes are stable**: Syntactic transformations apply the SAME
neural operation regardless of content. The question-inversion circuit always
shifts the hidden state in the same direction, regardless of which sentence is
being inverted.

**Why semantic axes are variable**: Semantic relationships are content-specific.
The geometric transformation from "cause" to "effect" depends on what the cause
IS — rain→flood is not the same neural operation as fire→ash. There is no single
"causation direction."

---

## Revised T2 Axis Taxonomy

| Axis | Type | Stability | Layer | Intrinsic? |
|------|------|-----------|-------|-----------|
| question | syntactic | 0.604 (STABLE) | L28 | HIGH |
| past_tense | morphological | 0.486 (PARTIAL) | L28 | MODERATE |
| gender | semantic | 0.387 (PARTIAL) | L27 | MODERATE |
| negation | syntactic | 0.347 (PARTIAL) | L28 | MODERATE |
| passive | syntactic | 0.289 (VARIABLE) | L28 | LOW-MOD |
| plural | morphological | 0.207 (VARIABLE) | L1 | LOW |
| concrete | semantic | 0.221 (VARIABLE) | L28 | LOW |
| synonym | lexical | 0.252 (VARIABLE) | L28 | LOW |
| comparative | morphological | 0.158 (VARIABLE) | L15 | LOW |
| hypernym | semantic | 0.137 (VARIABLE) | L28 | LOW |
| antonym | semantic | 0.136 (VARIABLE) | L28 | LOW |
| causation | semantic | 0.095 (VARIABLE) | L28 | MINIMAL |

---

## Implications for TruthSpace

### What IS intrinsic to the LM

The LM has **stable geometric subspaces** for syntactic transformations:
- Question inversion: a consistent direction at L28
- Tense marking: a consistent direction at L28
- Negation: a consistent direction at L28
- Gender agreement: a partially consistent direction at L27

These can be reliably extracted by ANY sufficiently diverse set of sentence pairs.

### What is NOT a single fixed direction

Semantic relations (causation, antonym, hypernym) are NOT single directions.
They are **relational manifolds** — the transformation depends on the specific
entities involved. The T2 axes for these categories capture ONE representative
slice through this manifold (the specific slice defined by the construction pairs).

This does NOT invalidate T2 as an address system. It means:
- T2 ternary addresses are DISCRIMINATIVE (high LOO accuracy ~94%)
- But they are PROBE-DEPENDENT for semantic axes
- Different probe sentence sets would give different (but equally valid) semantic addresses

### Self-Similarity Principle: Partial Confirmation

The TruthSpace self-similarity principle ("the same transformation works
identically at every scale") holds for syntactic axes. The gender flip
Δx = -2.0 is consistent across pairs (Day118 confirmed this for continuous
projection). But at the discrete address level, the axis is only partially
stable (0.387) — close but not perfect.

For semantic axes, self-similarity does NOT hold. The antonym axis for
"hot/cold" is not the same as for "fast/slow" at the geometric level.

---

## Summary of Days 119-123

| Day | Experiment | Key Finding |
|-----|-----------|-------------|
| 119 | d_k priming | Two modes: last-token general (3x), entity-position retrieval-specific (4.3x) |
| 120b | Layer sweep | L1-5 morphological, L10-15 relational, L23-28 pragmatic. Most orthogonal at L3. |
| 121 | LOO re-opt | Day78 layers not inferior to Day120b peak-Cohen-d layers |
| 122 | PCA validation | T2 (causal ablation) outperforms MD for semantic axes. Low T2-MD alignment reflects method difference. |
| 123 | Axis stability | Syntactic axes stable (question 0.604), semantic axes variable (causation 0.095). Mean 0.277. |

**Overall**: The T2 trie encodes a mixture of intrinsic syntactic geometry
(stable, reliable) and probe-specific semantic directions (variable, content-
dependent). The system works for discrimination (LOO=94%) because the specific
probe directions happen to be diverse enough to create unique addresses, even
if they don't represent universal geometric properties.
