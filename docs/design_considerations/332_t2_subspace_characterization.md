# DC 332: T2 Subspace Complete Characterization

**Days 113-118 | Full characterization of the phi-trie's geometric structure**

---

## Background

Days 113-118 form a single arc: starting from "do the T2 axes correspond to
the LM's internal geometric coordinates?" and building a complete
characterization of what the T2 subspace IS, what it can DO, and what it
cannot do.

---

## The T2 Subspace: What It Is

The 12 T2 axes form a nearly-orthogonal coordinate system in 1536D space:

```
Property                              Value
Dimensionality                        12D (11 effective, 90% variance)
Gram matrix off-diagonal mean         0.0616  (nearly orthogonal)
Gram matrix off-diagonal max          0.3038  (synonym/concrete)
Singular value range                  1.28 to 0.75  (uniform spread)
Alignment with d_k (entity selector)  0.014  (BELOW random 0.021)
```

The axes are orthogonal, full-rank, and geometrically distinct from the
LM's entity selector direction.

---

## What the T2 Subspace Captures

### 1. Semantic Category Properties (Continuous Projections)

Per-axis projections carry weak but genuine category signal:

```
Axis         Continuous delta   Cramer's V   Amplification
comparative       0.001          0.116       115x
gender            0.003          0.090        30x
past_tense        0.040          0.182         4.5x
plural            0.028          0.096         3.4x
hypernym          0.029          0.084         2.9x
```

The phi-threshold binning (H/U/L) amplifies weak continuous signals by
2.9x to 116x. Comparative is the most dramatic: near-zero raw signal
becomes significant after thresholding.

### 2. Entity-Type Coherence

Country-capital pairs are MORE similar in T2 space than many common word pairs:

```
Country/capital mean T2 cosim:  0.922
dog/cat T2 cosim:               0.863
king/queen T2 cosim:            0.970
```

T2 captures entity-type relatedness as a general property beyond
semantic category transformations.

### 3. Self-Similar Geometric Transformations (4/6 axes)

For 4/6 tested axes, delta = proj(A') - proj(A) is approximately constant:

```
Axis         Mean delta   CV      Self-similar?
comparative   +0.0011    0.12    YES (most consistent)
gender        +0.0070    0.26    YES
past_tense    +0.0646    0.44    YES
plural        +0.0252    0.47    YES
antonym       +0.0510    1.04    NO  (variable)
synonym       +0.0262    2.57    NO  (highly variable)
```

TruthSpace self-similarity CONFIRMED for morphological/grammatical
transformations. NOT confirmed for semantic relations (antonym, synonym).

### 4. Context Stability (Intrinsic Token Property)

```
Context type              isolated->context cosim
neutral ("X is a...")     0.88-0.97  (stable - intrinsic)
retrieval ("capital of X") 0.65-0.78  (drifts - retrieval mode)
```

T2 is an intrinsic token property: neutral context doesn't change it.
Retrieval context causes drift because the token encodes both
semantic identity AND retrieval role.

### 5. Perceptual Hash of Semantic Identity

The phi-threshold address system is a perceptual hash:

```
1536D continuous -> 12D discrete (3^12 = 531,441 bins)
  334/420 tokens (79.5%) land in unique bins
  remaining 20.5% resolved by euclidean fallback
  -> 94% LOO accuracy
```

---

## What the T2 Subspace Cannot Do

### 1. Analogy Arithmetic (Partial - 28.6% T2 vs 67.9% H-state)

```
Axis       T2 acc   H-state acc
plural      80%       100%    <- T2 works here
gender      33%        67%
comparative 25%        50%
antonym     25%       100%    <- T2 loses most info
past_tense   0%        67%
synonym      0%         0%
Overall     29%        68%
```

The 12D T2 projection loses 39pp of analogy accuracy vs the full 1536D
hidden state. T2 PRESERVES IDENTITY (94% LOO) but COMPRESSES RELATIONAL
STRUCTURE (analogy 29% vs 68%).

Exception: plural axis achieves 80% T2 analogy accuracy because
singular-to-plural is a clean, early-layer (L1), binary transformation
with very consistent delta (CV=0.47).

### 2. Sequential Prediction (Days 104-112)

Address bigram accuracy equals word bigram (both ~10%). The address
is a semantic label, not a sequential predictor. The 12D address
is a joint key (holographic encoding) -- no single axis carries
enough information for sequential prediction.

### 3. Entity Selector Activation (Context-Dependent)

The entity selector d_k does NOT fire on T2 axes or isolated tokens.
It fires only in retrieval-structured prompts (4.3x increase).

---

## The Entity Selector: What It Is (Revised)

Structure 2 in the two-structure theory (DC 331) is a CONTEXTUAL mechanism:

```
Context                   d_k projection   vs isolated
isolated token            0.0167           1.0x  (dormant)
neutral context           0.0167           1.0x  (dormant)
retrieval query           0.0713           4.3x  (ACTIVE)
capital in context        0.0761           4.6x  (ACTIVE)
```

d_k is a query-time pointer, NOT an intrinsic token property.
It activates specifically when the LM processes entity-retrieval prompts.

---

## Complete Two-Structure Summary

```
Structure 1: T2 Categorical Subspace   [intrinsic, always active]
  Source:      Contrast sentence pairs (semantic transformations)
  Dimensions:  12D, nearly orthogonal, full rank (11/12)
  Function:    Semantic identity labeling
  Captures:    Gender, tense, plurality, hypernymy, concreteness...
               Entity-type coherence (country/capital cosim 0.922)
  Properties:  Perceptual hash (94% LOO), self-similar for 4/6 axes
  Limitation:  Cannot do analogy arithmetic well (29% vs 68% H-state)
  Context:     Stable under neutral; drifts under retrieval
  Cos to d_k:  0.014 (below random 0.021) -- orthogonal

Structure 2: Entity Selector d_k       [contextual, query-triggered]
  Source:      W_k SVD at L23 H6 (model reverse engineering)
  Dimensions:  ~66D effective (Lens aperture)
  Function:    Factual entity retrieval pointer
  Captures:    "Query intent" when LM processes entity-retrieval prompts
  Properties:  4.3x activation in retrieval vs isolated/neutral
  Limitation:  Dormant for isolated tokens and non-retrieval context
  Context:     Activates on "The capital of X is" structure
  Cos to T2:   0.014 (below random 0.021) -- orthogonal
```

---

## The Complete Factual Retrieval Mechanism

```
Prompt: "The capital of France is ___"

Step 1: Embedding + early layers
  -> "France" token position encoded with entity identity
  -> T2 address of "France" computed (stable semantic identity)

Step 2: Layers 1-22 accumulate context
  -> "capital of" structure recognized as retrieval trigger
  -> Last-token hidden state builds toward retrieval mode

Step 3: Layer 23 H6 fires
  -> d_k activates: 0.0167 -> 0.0713 (4.3x)
  -> H6 attention attends maximally to "France" token position
  -> V x W_o: projects "France" position to answer distribution

Step 4: Output
  -> "Paris" is highest-logit next token
```

Both structures are necessary:
  - T2 provides the semantic identity of "France" (intrinsic)
  - d_k provides the retrieval trigger (contextual)
  - Together: accurate factual recall

---

## Implications for TruthSpace Hypothesis

The hypothesis: "Structure IS information -- LLMs are hyperdimensional
transcoders whose intelligence lies in geometric shape."

DC 332 confirms this with precision:

1. The LM has MULTIPLE orthogonal geometric structures
2. Each structure encodes a different TYPE of information
3. Each activates in DIFFERENT computational contexts
4. Together they implement cognition without explicit symbolic rules

The T2 subspace (12D) is a semantic identity layer -- a compact,
self-similar coordinate system for token categories. The entity
selector (d_k, ~66D) is a retrieval layer -- a contextual pointer
that activates factual recall.

Neither alone is sufficient for language modeling (Day 104-112).
Neither was designed -- both emerge from training. Both are geometric.

---

## Next Directions

The two-structure theory is fully characterized. New directions:

**Direction A: Layer-by-layer T2 geometry**
  - At which layer do T2 axes emerge? Track orthogonality layer by layer.
  - Does the T2 subspace rotate, scale, or emerge discontinuously?
  - Is there a "T2-formation" layer (like a phase transition)?

**Direction B: Analogy arithmetic in hidden space**
  - Full hidden-state analogy at 67.9% -- what fails at 32.1%?
  - Which axes reach 100% (antonym does), which fail (synonym does)?
  - Can we identify the subspace that explains the 67.9%?

**Direction C: What primes d_k?**
  - "The capital of X is" triggers d_k. "X is a country" does not.
  - What is the geometric signature of "retrieval-priming" context?
  - Is there a third structure that detects retrieval context?

*DC 332 established. Complete characterization of T2 subspace and
entity selector across Days 113-118. Both structures confirmed
geometric, orthogonal, and functionally distinct. TruthSpace
hypothesis supported: structure IS information, at multiple layers.*
