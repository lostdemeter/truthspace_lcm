# DC 331: Two Orthogonal Geometric Structures in the LM

**Day 113 | T2 semantic axes and entity selector axes occupy orthogonal subspaces**

---

## Background

The TruthSpace project has pursued two parallel experimental arcs:

**Arc 1 — Model Reverse Engineering (Days 1–70)**
Characterized the LM's actual internal geometric operations:
- Spectrometer (per-dimension rules): 14/15 layers
- Geometric Selector (H6 L23 d_k): entity identity retrieval
- Geometric Lens (M_h, 66D aperture): entity identity extraction
- Gyroscope, Resonator: other internal structures

**Arc 2 — φ-Trie (Days 70–112)**
Built and validated a geometric semantic index using T2 axes derived
from contrast sentence pairs:
- 12D nearly orthogonal address space (LOO=94%, analogy=100%)
- Confirms semantic structure is geometric
- But does NOT improve language generation over word bigrams

**Day 113** directly tested the connection between these two arcs:
Are the T2 axes aligned with the LM's actual internal directions?

---

## Key Measurement: T2 vs LM Internal Alignment

```
Metric                          T2 axes    Random baseline
────────────────────────────────────────────────────────
T2 vs d_k (L23 H6 selector)     0.0143        0.0206   ← BELOW RANDOM
T2 vs W_q SVD top-20 (mean)     ~0.0190       0.0200   ≈ RANDOM
T2 vs W_q SVD top-20 (max)       0.0577       0.0809   ← BELOW RANDOM MAX
```

**Conclusion**: The 12 T2 trie axes are geometrically orthogonal to the
LM's entity selector direction (d_k). Their alignment is below the
random vector baseline. The two arcs discovered DIFFERENT geometric
structures in the same 1536D representation space.

---

## T2 Axis Internal Structure

```
Gram matrix off-diagonal:
  Mean:  0.0616  (nearly orthogonal)
  Max:   0.3038  (synonym ↔ concrete, semantically related)
  Effective dimensionality: 11/12 (90% variance)
  Singular values: 1.28 → 0.75 (near-uniform — no dominant axis)
```

The T2 axes form a genuine 12D coordinate system in 1536D space:
- Nearly orthogonal to each other
- Full rank (11 effective dimensions from 12 axes)
- Uniform spread (no dominant direction)

This confirms the trie's 12D address space is a well-conditioned
coordinate system, not a degenerate or redundant representation.

---

## The Two-Structure Theory

The LM's 1536D representation space contains at least two distinct
geometric structures:

### Structure 1: T2 Categorical Axis Space (φ-Trie)

```
Dimension:    12D in 1536D
Source:       Contrast sentence pairs (semantic transformations)
Captures:     Categorical properties: gender, tense, hypernymy,
              plurality, antonymy, concreteness, voice, causality,
              modality, negation
Accuracy:     94% LOO semantic similarity, 100% internal analogy
Function:     Semantic identity labeling (indexing)
NOT used for: Sequential prediction, entity retrieval
```

### Structure 2: Entity Selector Axis Space (Model Reverse Eng.)

```
Dimension:    ~66D in 1536D (Lens aperture, Finding 124)
Source:       W_q, W_k weight SVD at L23 H6
Captures:     Entity identity: France → Paris, Germany → Berlin
              cos(d_q, d_k) = 1.0000 (same-feature detector)
Accuracy:     6/6 capital retrieval, generalizes to 10/12 unseen
Function:     Factual entity identity retrieval (knowledge)
NOT used for: Semantic category classification
```

### Orthogonality

The two subspaces are approximately orthogonal:
```
cos(T2_axis, d_k) = 0.014 ± 0.010   [random: 0.021]
```

This near-zero alignment (below random baseline) indicates the two
structures occupy different, essentially non-overlapping subspaces
of the full 1536D space.

---

## Implications for the TruthSpace Hypothesis

**Hypothesis**: "Structure IS information — LLMs are hyperdimensional
transcoders whose intelligence lies in the geometric shape of their
weights."

**Day 113 refines this**: The LM's geometric structure is not a single
monolithic shape but a **superposition of orthogonal geometric
structures**, each serving a different cognitive function:

```
1536D Hidden Space
├── T2 Categorical Subspace (12D)
│   └── Semantic properties: gender, tense, hypernymy...
│       [accessible via contrast sentence differences]
├── Entity Selector Subspace (~66D)
│   └── Entity identity: France, Germany, Paris...
│       [accessible via W_q/W_k SVD at L23 H6]
├── Spectrometer Subspace (per-dimension, ~1D each)
│   └── Per-token activation rules, layer-by-layer
│       [accessible via per-dimension threshold fitting]
└── ... other structures (Gyroscope, Resonator, Lens)
    [not yet characterized in terms of subspace dimensions]
```

Each arc of the TruthSpace project has been discovering ONE of these
structures. The structures are:
- **Complementary**: each captures different knowledge
- **Orthogonal**: they don't interfere with each other
- **Geometrically real**: not artifacts of the analysis method

This is consistent with the broader hypothesis — but the "shape" is
richer than a single geometric structure. It is a composition of
orthogonal geometric subspaces, each encoding a different type of
information.

---

## The Subspace Decomposition Question

If the LM's knowledge is decomposed into orthogonal subspaces, a key
question emerges: **how many such subspaces are there, and what is
their total dimensionality?**

Known so far:
- T2 categorical: ~12D
- Entity selector: ~66D (Lens aperture from Finding 124)
- Spectrometer: ~1D per dimension (up to 1536 trivially, but effective
  number of semantic dimensions unknown)

Together: 12 + 66 = 78D of confirmed semantically meaningful subspace
out of 1536D total. The remaining ~1458D contains:
- Sequential/syntactic structure (not yet characterized)
- Stylistic/register information
- Numerical and factual knowledge beyond entities
- Noise / unstructured dimensions

---

## Next Experiments

**Day 114**: Vocabulary projection onto both subspaces
- Project all 420 probe tokens onto T2 subspace (12D) and entity
  selector subspace (d_k direction)
- Do proper nouns (king, queen, actor) project strongly onto T2?
- Do concrete entities (France, Germany) project onto d_k?
- Hypothesis: the two projections are complementary — different
  token types are best characterized by different subspaces

**Day 115**: Cross-structure prediction
- Can T2 address + d_k projection together predict token properties
  better than either alone?
- Joint 78D representation vs 12D or 66D alone

---

## Summary

| Arc | Structure | Subspace | Function | Alignment to other |
|-----|-----------|----------|----------|-------------------|
| φ-Trie | T2 categorical | ~12D | Semantic labeling | ⊥ to entity selector |
| Rev. Eng. | Entity selector | ~66D | Factual retrieval | ⊥ to T2 axes |

The LM's 1536D representation space is a superposition of at least two
orthogonal geometric structures. The TruthSpace hypothesis holds —
structure IS information — but the structure is a multi-layer
composition, not a single unified geometry.

*DC 331 established. Two orthogonal geometric structures confirmed:
T2 categorical (12D, φ-trie) and entity selector (66D, H6 L23 d_k).
Cos(T2, d_k) = 0.014 < random 0.021. Both are geometrically real,
functionally distinct, and occupy non-overlapping subspaces.*

---

## Days 114b–115 Addendum: T2 Mechanism Characterization

### Threshold Amplification (Day 115)

φ-thresholding amplifies weak continuous projection signals into strong
discrete class separations:

```
Axis         Continuous Δ  Cramér's V  Amplification
comparative  0.0010         0.1155       115×
gender       0.0030         0.0897        30×
past_tense   0.0400         0.1815         4.5×
plural       0.0280         0.0963         3.4×
hypernym     0.0290         0.0841         2.9×
```

All 5 axes confirmed. The φ-scaled percentile thresholds land at natural
breaks in the projection distribution, converting near-zero continuous
deltas into statistically significant H/U/L class associations.

### The Trie as a Perceptual Hash (Day 115)

```
1536D continuous hidden state
    ↓ project onto 12 T2 axes (correct layer each)
    ↓ φ-threshold bin: H/U/L per axis
12D ternary address (3^12 = 531,441 bins)
    ↓ 334/420 tokens (79.5%) land in unique bins
    ↓ remaining 20.5% resolved by euclidean fallback
94% LOO accuracy
```

The trie is a **perceptual hash of semantic identity**: a compact (12-symbol)
descriptor that uniquely identifies 79.5% of vocabulary tokens, with the
remaining 20.5% resolved by continuous projection distance.

This is structurally identical to perceptual hashing in image retrieval:
- Short hash: captures most of the identity signal
- Hash collisions (shared addresses): resolved by original signal (euclidean)
- Speed/compactness/accuracy tradeoff governed by dimensionality

### Refined Architecture of T2 Categorical Structure

```
Structure 1: T2 Categorical (now fully characterized)
  Continuous layer:  12D projections, weak (Δ=0.001-0.04)
  Discrete layer:    H/U/L bins, strong (V=0.08-0.18)
  Address layer:     12-symbol ternary string (perceptual hash)
  LOO accuracy:      94% (79.5% from uniqueness + 14.5% from fallback)
  Category examples: comparative, gender, past_tense, plural, hypernym
  Function words:    highest T2 magnitude (multi-axis participant)
  Entity pairs:      country/capital mean cosim = 0.922 (> dog/cat 0.863)
```

---

## Day 116 Addendum: Entity Selector Is Context-Dependent

### Result

When isolated single-token hidden states are projected onto d_k:
```
Type           dk_mean   Cohen's d vs common
country         0.0166   ≈ 0 (no separation)
capital         0.0196   ~0.25 (small, outlier-driven)
proper_noun     0.0180   ~0.25 (small)
common_noun     0.0164   (baseline)
function        0.0167   (baseline)
```

All token types project onto d_k with nearly identical magnitudes.
Cohen's d = 0.25 (small effect). The entity selector d_k does NOT
fire on isolated proper nouns.

### Root Cause: d_k Is a Query-Time Direction

Finding 40 (model reverse engineering) discovered d_k in a CONTEXTUAL
setting:
  Prompt: "The capital of [France] is ___"
  d_k fires because the LM is in entity-retrieval mode (context forces
  it to attend to the entity token at its position)

For isolated " France" tokens, no retrieval context exists → d_k
doesn't activate. The direction is:
  - NOT an intrinsic property of entity tokens
  - IS a contextual activation: the LM's pointer direction when
    executing factual retrieval

### Bonus Finding: T2 Captures Entity-Type Coherence Too

Country-capital T2 cosine similarity (0.922) > dog/cat (0.863).
The T2 categorical subspace groups related entities (France, Paris)
as tightly as or more tightly than semantic category pairs.

### Revised Two-Structure Theory

```
Structure 1: T2 Categorical Subspace
  - Encodes: semantic transformations + entity-type coherence
  - Intrinsic token property (fires on isolated tokens)
  - 12D, nearly orthogonal, perceptual hash of semantic identity

Structure 2: Entity Selector Direction (d_k)
  - Encodes: query-time entity retrieval mode
  - Context-dependent (fires on entity-retrieval prompts, not tokens)
  - NOT an intrinsic token property
  - Points TOWARD the answer direction when LM executes factual lookup
```

These are still orthogonal (cos(T2, d_k) = 0.014) but serve different
computational roles: one is a token-time label system, the other is
a query-time retrieval pointer.

*DC 331 updated. Day 116 revises Structure 2 from "entity identity
embedding" to "contextual entity retrieval pointer." T2 subspace
captures entity-type coherence (country/capital cosim 0.922), making
it a broader semantic index than previously characterized.*

---

## Day 117 Addendum: Full Retrieval Mechanism Characterized

### d_k Activation by Context (8 entities, 6 context types)

```
Context                   d_k mean   factor vs isolated
isolated_country           0.0167    1.0×  (baseline)
neutral ("X is country")  0.0167    1.0×  (NO activation)
entity_in_query            0.0713    4.3×  ← LARGE
query_last_token           0.0562    3.4×
capital_in_ctx             0.0761    4.6×  ← LARGEST
```

The neutral context ("France is a country in the world and") does NOT
activate d_k at all. Only retrieval-structured prompts activate it.

### T2 Stability: Intrinsic (neutral) vs Modulated (retrieval)

```
Context              isolated→context T2 cosim
neutral_ctx          0.88–0.97  (stable — intrinsic property preserved)
entity_in_query      0.65–0.78  (drifts — retrieval mode encoded)
query_last_token     0.67–0.83  (moderate drift)
```

T2 coordinate is intrinsic (stable under neutral context) but gets
modulated by retrieval context. The entity token in "The capital of
France is" encodes both its semantic identity AND its retrieval role.

### Complete Factual Retrieval Mechanism

```
[Query structure: "The capital of X is"]
    ↓ Early layers: accumulate entity identity from X position
    ↓ d_k activates at L23 H6 (4.3× above baseline)
    ↓ Attention argmax: entity position ("France") ← H6 selector
    ↓ V×W_o: entity position projects to answer token ("Paris")
    ↓ Output: next-token logits favor "Paris"

[Neutral statement: "X is a country"]
    ↓ Entity position encoded but no retrieval mode primed
    ↓ d_k stays at 0.0167 (random-level)
    ↓ H6 selector does not specifically attend to entity
    ↓ Output: distribution over descriptors, not facts
```

### Final Two-Structure Summary

```
Structure 1: T2 Categorical Subspace  [intrinsic, always active]
  - Semantic transformations: gender, tense, plurality, etc.
  - Entity-type coherence: country/capital cosim 0.922
  - Stable under neutral context (cosim 0.88-0.97)
  - Modulated by retrieval context (cosim 0.65-0.78)
  - Perceptual hash: 94% LOO from 12D ternary address

Structure 2: Entity Selector (d_k)  [contextual, query-triggered]
  - Dormant on isolated tokens (0.0167 = random)
  - Dormant on neutral context (0.0167 = random)
  - Activates on retrieval prompts: 0.0713 (4.3×)
  - Encodes the LM's "query intent" geometry
  - Orthogonal to T2 axes (cos = 0.014, below random 0.021)
```

The TruthSpace hypothesis is confirmed: both structures are geometric,
both are real, both serve distinct and complementary roles. The
LM's knowledge is encoded in multiple orthogonal geometric subspaces,
each activated by different computational contexts.

*DC 331 FINAL. Two orthogonal geometric structures fully characterized.
Structure 1 (T2) is intrinsic and semantic. Structure 2 (d_k) is
contextual and query-triggered. Together they form the geometric
architecture of factual knowledge retrieval in the LM.*
