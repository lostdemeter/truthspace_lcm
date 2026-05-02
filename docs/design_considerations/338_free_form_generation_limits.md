# DC 338: Free-Form Generation — Limits of Geometric Methods

**Days 133-136 | Why geometric similarity cannot replace log-probability**

---

## Summary

Four days of free-form generation experiments (Days 133-136) definitively
establish what geometric methods CAN and CANNOT do for next-token generation.

---

## What Was Tested

```
Day 133: Frequency-based vocab (300 words) → 0% top-1, subword contamination
Day 134: Curated vocab (237 words) → 0% top-1, 2.26x overlap signal
Day 135: Debiased similarity (vocab-mean centering) → L25 worse, T2 slightly better
Day 136: Two-stage T2→L25 (K=10,20,50,100) → 1/21 top-1 (5%), not reliable
```

---

## Definitive Results

| Method | Top-1 agree | Overlap@10 ratio | Verdict |
|--------|-------------|-----------------|---------|
| L25 raw cosine | 0/21 | 0.78x (below random) | FAILS |
| L25 centered | 0/21 | 0.45x (much worse) | FAILS |
| T2 raw | 0/21 | 2.34x | partial signal |
| T2 centered | 0/21 | 2.56x | partial signal |
| struct_axis | 0/21 | 1.11x | marginal |
| T2→L25 two-stage | 1/21 | — | FAILS |
| Log-prob oracle | 21/21 | — | perfect |

---

## Why L25 Fails for Free-Form

L25 cosine similarity is **below random** (0.78x) for vocab-level ranking because:

1. **"is" token bias**: Prompts ending in "is" generate a generic
   "expecting completion" representation at L25. Words like `Portuguese`,
   `Canberra`, `Oslo` have single-token representations that cluster near
   this generic "is" state — they appear at the top for ALL prompts.

2. **Within-category signal is real but small**: The actual factual signal
   (Paris > London for France context) exists but is overwhelmed by the
   frequency bias from steps 1-3 of token generation across training data.

3. **Debiasing fails**: Subtracting vocab mean actually removes some genuine
   signal along with the bias, making results worse. The bias vector aligns
   with the signal vector.

---

## Why T2 Partially Works (2.34x)

T2 succeeds because it encodes **categorical membership** rather than
specific token preferences:

- For "Yesterday he", T2 selects past-tense verbs — correct category
- For "A poodle is a type of", T2 selects animals/categories
- For "The opposite of hot is", T2 selects adjectives

These are 2.34x above random for overlap@10, confirming T2 is a real
category filter. But within category (walked vs ran vs told), T2 cannot
distinguish — all past-tense verbs look identical at the T2 level.

---

## The Two-Stage Failure

The two-stage pipeline (T2 filter K words → L25 rank within) fails because:

1. The oracle word is often OUTSIDE T2's top-K (at T2 rank 50-200)
2. When the oracle IS in top-K, L25 still can't rank it first

Example: "Yesterday he" → oracle=`told`
- T2 rank of `told` = 99 (not in top-50)
- So T2→L25 K=50 never considers `told`

---

## The Fundamental Asymmetry

```
Constrained ranking (5-10 candidates from same category):
  The correct word (Paris) and wrong words (London, Rome) are all cities.
  L25 has Paris > London for France context: 62% oracle MRR ✓

Free-form ranking (237 vocab words across all categories):
  Paris (rank 50), Oslo (rank 1), Portuguese (rank 3)
  The "is" bias promoties Oslo/Portuguese above Paris ✗
```

The constrained setting REMOVES the frequency bias by forcing comparison
within a pre-filtered semantic category. Free-form exposes it.

---

## Complementary Roles Confirmed

```
                 Constrained (5-10)    Free-form (237)
T2 (12D)         MRR=0.540             2.34x overlap
L25 (1536D)      MRR=0.515-0.596       0.78x (below random!)
Combined (auto)  MRR=0.596 (62%)       1/21 top-1

Category filter: T2 (2.34x free-form signal)
Within-category: L25 (best for constrained candidate sets)
```

The pipeline is fundamentally a **semantic reranker**, not a **generator**.

---

## What Would Enable Free-Form Generation

Three approaches could close this gap:

### 1. Directed Knowledge Edges (Explicit)
Build a knowledge graph: (France, Paris), (poodle, dog), etc.
Route: T2 classifies → knowledge graph returns answer → T2 verifies
Cost: requires external knowledge; not emergent geometry

### 2. Attention-Path Factual Probe
Finding 154 showed entity hidden states at L22 contain the answer.
Route: extract entity token h at L22 → use as factual key
This would bypass the "is" bias entirely.
Day 137 will test this.

### 3. Log-Probability Computation
The 38% oracle gap requires weight-matrix computation.
The LM's attention weights at L23 H6 perform the entity lookup.
No geometric approximation can replicate this without the weights.

---

## Implication for TruthSpace

**Partial confirmation of the hypothesis:**

> "Structure IS information"

- ✓ T2 structure encodes CATEGORICAL information (97-100% LOO)
- ✓ T2 axis structure encodes SYNTACTIC transformations (MRR=1.0)
- ✓ L25 structure encodes WITHIN-CATEGORY preferences (62% MRR)
- ✗ No geometric structure encodes SPECIFIC FACTUAL associations
  without directed edges or attention computation

The TruthSpace hypothesis holds for structural/relational knowledge.
For encyclopedic knowledge, the "shape" alone is insufficient — the
directed connections (attention weights mapping entity→fact) are required.

**The architecture of knowledge in the LM:**
```
Syntactic knowledge → T2 axes (extractable, generalizes)
Relational knowledge → struct_axis vectors (partially extractable)
Categorical knowledge → T2 centroid clusters (extractable, near-perfect)
Factual knowledge → attention weight paths (non-extractable without computation)
```

---

## Files

- `expedition_day133_freeform_generation.py` — initial vocab test
- `expedition_day134_curated_vocab_gen.py` — cleaned vocab, 2.26x signal
- `expedition_day135_debiased_generation.py` — debiasing experiment
- `expedition_day136_two_stage_pipeline.py` — T2→L25 two-stage test
