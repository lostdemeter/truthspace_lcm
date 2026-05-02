# DC 335: Geometric Information Hierarchy

**Days 124-127 | T2, d_k, full hidden state, and log-probability as a hierarchy**

---

## Background

DC 334 established T2 as a semantic category filter and d_k as a weak
within-category ranking signal. Days 126-127 extended this by measuring
Spearman ρ with log-probability and comparing full hidden state similarity.

The central question: **how much of the model's predictive information
is captured by geometric similarity at different levels of compression?**

---

## The Geometric Information Hierarchy

```
Signal          Dim    Layer   MRR     ρ(logprob)   Notes
──────────────────────────────────────────────────────────────────────────
Log-prob        1      N/A    1.000     1.000       oracle (weight matrix)
Full h cos     1536    L25    0.783    +0.355       best geometric ranker
Full h cos     1536    L27    0.711    +0.396       best ρ, lower MRR
Full h cos     1536    L23    0.690    +0.307       entity selector layer
d_k             1      L23    0.549    +0.235       entity selector direction
T2+d_k (α=0.9) 13     multi  0.595      ---        combined
T2             12      multi  0.540    +0.063       category filter
Random          —      —      0.314     0.000       baseline
```

**Key observation**: d_k (1D) retains MORE probability-relevant information
than T2 (12D). A single geometric direction beats a 12-dimensional subspace
for probability prediction.

---

## Finding 1: L25 = Optimal Layer for Geometric Ranking

The full 1536D hidden state cosine achieves maximum MRR=0.783 at L25:

- L1: ρ=-0.174 — early layer is ANTI-correlated with log-prob
- L5-L20: ρ increases monotonically +0.225 → +0.285
- L25: ρ=+0.355, MRR=0.783 — PEAK
- L27: ρ=+0.396 (best ρ) but MRR=0.711 (lower than L25)
- L28: ρ=+0.065, MRR=0.625 — **SHARP DROP**

L25 is the "semantic integration sweet spot":
1. Fully processes morphological, relational, and pragmatic information
2. Has not yet undergone the L28 output transform
3. Isolated word representations are maximally aligned with context

**L28 paradox**: L28 has the best correlation with log-prob for the T2
axes (most T2 axes peak at L28), but the WORST full-cosine-to-log-prob
correlation. This confirms that T2 axes extract DIFFERENT information
from the full hidden state — they capture axis-specific semantic content
while the full cosine at L28 is disrupted by the output projection.

---

## Finding 2: The L28 Output Transform Breaks Cosine Similarity

The final layer L28 applies a transformation preparing the residual stream
for projection to vocabulary logits. This makes isolated word representations
at L28 UNLIKE context representations at L28, even though the model would
assign high probability to those words.

```
Naive intuition: "Paris" at L28 should be similar to context "capital of France"
Reality: L28 full cosine ρ=0.065 — barely above zero correlation

Why: The output transform maps FROM the semantic space INTO the logit space
     This direction is NOT preserved under cosine similarity
```

This explains why the T2 axes (extracted at L28) work well for word-level
CLASSIFICATION but not for context-based RANKING: they capture the post-
transform representation, which is semantically discriminative between word
types but NOT between context-fit levels.

---

## Finding 3: T2 Is Sometimes Anti-Correlated with Log-Probability

T2 isolated cosine ranking is negatively correlated with log-prob for:
- "A rose is a type of": T2 top1="never" (ρ=-0.627)
- "Official language of Egypt": T2 top1="later" (ρ=-0.400)
- "A poodle is a type of": T2 top1="quickly" (ρ=-0.300)

The pattern: common English function words ("never", "later", "quickly")
have T2 addresses close to prompt last-token T2 anchors. The T2 address
of the prompt's last token ("is", "of") reflects the grammatical structure
of the sentence, and function words have similar grammatical profiles.

T2 is a **word-class comparator**, not a **contextual semantic match signal**.

---

## Finding 4: d_k Is a Better 1D Compression Than T2 12D

Information hierarchy for 1D vs 12D compressions:

```
Signal  Dims  ρ(logprob)  What it captures
d_k       1   +0.235      Entity consistency in retrieval context
T2       12   +0.063      Word-class semantic category membership
```

The entity selector direction d_k captures how well a candidate fits
as an ENTITY in the retrieval-primed context. This is more relevant to
probability prediction than categorical membership.

Why: log-probability is dominated by contextual fit (does this word
complete the sentence?), which is closer to what d_k measures (entity
compatibility with retrieval context) than what T2 measures (which
category does this word belong to?).

---

## Finding 5: The 22% Oracle Gap

Full cosine at L25 achieves MRR=0.783 vs oracle 1.000: **22% gap**.

This gap is **irreducible by any geometric similarity measure**. It
requires DIRECTED knowledge:

```
Geometric similarity (symmetric):
  sim(Paris, context) ≈ sim(London, context)  ← both are cities
  Cannot tell which city belongs to France

Directed factual association (asymmetric):
  P(Paris | "capital of France is") >> P(London | ...)
  Encodes France → Paris in weight matrix
```

The 22% gap is the portion of the model's knowledge that is stored as
**directed associations** (country → capital) rather than **undirected
similarity** (Paris and London are both in the city cluster).

TruthSpace implication: Pure geometric traversal (T2 trie navigation)
cannot recover this directed information without additional structure.
The factual associations require either:
1. The full log-probability computation (weight matrix → logits)
2. Or explicit directed edges in the trie (France → Paris)

---

## Revised Picture of the Two Geometric Structures

```
Structure     Signal  Layer  Role                   Limitation
──────────────────────────────────────────────────────────────────────
T2 (12D)      axes    multi  Category filter         Cannot rank within cat
                             Word-class comparator   Sometimes anti-correlated
                             Trie address            22%+ oracle gap

d_k (1D)      entity  L23   Entity compatibility    Weak within-category
              selector      Better prob proxy         All retrieval context
                            than T2                  elevates d_k uniformly

h_L25 (1536D) full    L25   Best geometric ranker   78% oracle ceiling
              hidden        Semantic integration     Requires full inference
              state         sweet spot

log-prob      scalar  N/A   Oracle                  Requires weight matrix
                            100% oracle             computation
```

---

## Summary

The complete ranking picture from Days 124-127:

1. **T2 (12D)**: Category filter — useful for excluding semantically wrong
   words, unreliable for within-category ranking. MRR=0.540.

2. **d_k (1D)**: Better probability proxy — ρ=0.235 vs T2's 0.063. The
   entity selector direction aligns with contextual fit more than T2's
   word-class addresses. MRR=0.549.

3. **Full h_L25 (1536D)**: Best geometric ranker — MRR=0.783, 78% oracle.
   Layer 25 is the semantic integration sweet spot before the output
   transform disrupts isolated-word cosine comparisons.

4. **Log-prob (oracle)**: 100% MRR. The 22% gap from L25 to oracle is
   directed factual knowledge (France→Paris), irreducible by geometric
   similarity alone.

**For TruthSpace**: The trie addresses (T2) capture categorical structure
well, but the system needs additional directed-association machinery to
close the 22% gap for factual knowledge retrieval.
