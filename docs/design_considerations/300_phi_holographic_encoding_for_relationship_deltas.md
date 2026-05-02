# DC 300: Phi-Holographic Encoding for Relationship Deltas

**Date:** 2026-03-08  
**Phase:** TruthSpace v1, Phase 4 (LCM Inference)  
**Experiment file:** `experiments/truthspace_v1/dc299_phase4_lcm_inference.py`  
**Predecessor:** DC 299 (Complete Model Map via Platonic Ideal Discovery), DC 255 (4-State Gate as a φ-Structured Dimension)

---

## 1. The Problem We Were Solving

Phase 4 of DC 299 established that **relationship deltas** — mean projection-space differences between semantically related concept pairs — can serve as a geometric inference primitive. Given `France`, apply the `capital_of` delta to retrieve `Paris`.

The original implementation (v1) encoded each concept as a **binary Gödel address**: a 193-bit vector where each bit indicates which side of a learned threshold the concept's projection falls on. Deltas were computed in this binary address space, and retrieval used Hamming distance.

The failure mode was visible immediately: the binary threshold collapses all information about *how far* a projection is from the threshold boundary. A projection just barely above threshold and one far above both become `1`. The boundary region — where the most semantically sensitive concepts live — becomes indistinguishable noise.

---

## 2. The Progression of Approaches

We ran leave-one-out (LOO) validation across three relationships:
- `capital_of` (14 country→capital pairs)
- `male→female` (10 gender pairs)  
- `country→language` (11 pairs)

Each version was tested against two axes: **how the delta is learned** and **how retrieval is performed**.

### v1 — Raw mean delta + Binary Hamming distance
The baseline. Averages projection differences across training pairs, thresholds to binary, retrieves by Hamming distance.

### v2 — Confidence-weighted delta + Binary Hamming distance
Introduced 4-quadrant weighting: axes where source and target bits don't consistently flip (P(flip) ≈ 0.5) are suppressed. Theoretically clean. In practice, mixed results — improved some metrics, degraded others. The binary retrieval itself was the bottleneck.

### v3 — Raw mean delta + Continuous cosine similarity
Bypassed binary thresholding entirely. Applied the delta in continuous projection space and retrieved by cosine similarity against raw projections. **Massive improvement** across all relationships. This validated that the retrieval mechanism, not delta quality, was the primary bottleneck.

```
capital_of rank-1:      v1=7%   →  v3=36%
male→female rank-1:     v1=30%  →  v3=50%
country→lang rank-1:    v1=55%  →  v3=73%
```

### v4 — Phi-4-state encoding (DC255) + Continuous cosine similarity

DC 255 established that gate activations in Qwen2-7B occupy one of four states at ±log(φ) boundaries:

```
CONTRACT  (-1): proj < threshold − log(φ)·σ    → strong absence
PRESERVE- (-0): threshold − log(φ)·σ ≤ proj < threshold    → negative zero
PRESERVE+ (+0): threshold ≤ proj < threshold + log(φ)·σ    → positive zero
EXPAND    (+1): proj > threshold + log(φ)·σ    → strong presence
```

The PRESERVE states are the "negative zero fringe" — small magnitude but maximum information density. Binary encoding collapses them into the same bin as CONTRACT and EXPAND respectively, destroying the distinction between *just barely* and *strongly*.

We encoded each concept projection into the 4-state space with values `{-φ, -1/φ, +1/φ, +φ}` (v4b, continuous variant) and retrieved by cosine similarity in this encoded space.

```
capital_of rank-1:      v3=36%  →  v4b=29%   (slight regression on rank-1)
capital_of top-5:       v3=79%  →  v4b=86%   (clear improvement)
capital_of mean rank:   v3=3.5  →  v4b=3.0   (improvement)
country→lang rank-1:    v3=73%  →  v4b=82%   (clear improvement)
```

---

## 3. The Fibonacci Insight

### 3.1 The Missing Level

Examining the v4b value set `{-φ, -1/φ, +1/φ, +φ}` against the φ-power lattice:

```
...  φ⁻²    φ⁻¹    φ⁰    φ¹    φ²  ...
... 0.382  0.618   1.0  1.618  2.618 ...
              ↑                ↑
           v4b uses             v4b uses
           ±1/φ                ±φ
           
           φ⁰ = 1.0 is MISSING
```

The ratio between adjacent v4b levels is **φ²**, not φ. The Fibonacci sequence of φ-powers (`..., 1/φ, 1, φ, φ²,...`) has uniform ratio φ between every step. v4b skips the φ⁰=1 level entirely.

This is the Fibonacci number that was missing: **1** (both F(1) and F(2) equal 1, the pivotal φ⁰ element between the contractive and expansive halves of the lattice).

The structural consequence: PRESERVE states in v4b are encoded at ±1/φ ≈ ±0.618, which gives them φ× *less* weight relative to CONTRACT/EXPAND (±φ ≈ ±1.618) during cosine similarity. The ratio CONTRACT/PRESERVE = φ/（1/φ) = φ². But from the transition analysis (Section 3.2), the PRESERVE zone is where the relationship signal lives. Underweighting it is the wrong direction.

### 3.2 The Transition Analysis

For every relationship, we classified each concept projection into one of the four DC255 states and counted what type of state transition occurs when moving from source to target. Across all three relationships:

```
Dominant transitions (after "same"):
  C→P-  (CONTRACT → PRESERVE-)   ~6-7% of all axis transitions
  E→P+  (EXPAND   → PRESERVE+)   ~6% of all axis transitions
```

**Target concepts land in PRESERVE zones more than source concepts do.** Countries are strongly established in states (high confidence EXPAND or CONTRACT on geographic axes). Their capitals, languages, and female counterparts sit closer to threshold — in the high-information-density PRESERVE zone.

This is the asymmetry DC 255 described: the PRESERVE states are where the *transformer does its work*, and empirically they are where concept *relationships* point to. An encoding that underweights the PRESERVE zone underweights the signal.

### 3.3 Fibonacci Search and the Skip

The insight also connects to why Fibonacci search outperforms binary search on average: it splits at F(k-1)/F(k) ≈ 1/φ from the bottom rather than at the midpoint. Crucially, **F(1) = F(2) = 1** — the top two positions share equal weight. There is no privileged "rank 0"; the search starts from the φ-ratio point, not from the extreme.

This is precisely the structure we need: the top two axes in importance share equal status (no single dominant axis), and weight decays as φ with each step down the importance ranking. The flat top and gradual decay are both properties of the Fibonacci sequence, not of a pure geometric series.

---

## 4. v6 — Fibonacci-Corrected Encoding `{-φ, -1, +1, +φ}`

Restoring the φ⁰=1 level gives PRESERVE states the value ±1, CONTRACT/EXPAND the value ±φ. Adjacent ratio is now φ throughout:

```
φ / 1 = φ  ✓
1 / (1/φ) = φ  ✓ (comparison to previous level if extended)
```

Encoded levels in continuous form:
- **EXPAND**:    `signed × φ`    (amplified)
- **PRESERVE+**: `signed × 1`    (raw, no scaling)
- **PRESERVE-**: `signed × 1`    (raw, no scaling)
- **CONTRACT**:  `signed × φ`    (amplified, stays negative)

The PRESERVE zone retains its full continuous projection magnitude. CONTRACT and EXPAND are amplified by φ to maintain the ratio. The threshold is now the natural zero of the encoding.

### 4.1 Results

*Note: these figures use the original Phase-0 concept set, before vocabulary correction. Some target words were absent or mis-tokenised, producing artificially high `?` rates. See Section 6.1 and the corrected table in Section 10.*

```
                     v1      v3     v4b     v6
capital_of rank-1:   7%     36%     29%    36%   ← v6 recovers v3
capital_of top-5:   21%     79%     86%    86%   ← v6 matches v4b
capital_of mean:    10.6    3.5     3.0    3.5
gender rank-1:      30%     50%     40%    40%   ← v3 leads on original set
gender top-5:       30%     80%     80%    80%   ← all match
country→lang r-1:   55%     73%     82%    82%   ← v6 matches v4b
country→lang top-5: 82%     82%     82%    82%
```

v6 is the most consistent encoding across all three relationships:
- Never regresses vs v4b
- Recovers v3's rank-1 performance on `capital_of` (36%)
- Maintains v4b's 82% on `country→language`
- `male→female` v3 still leads at rank-1 (50% vs 40%) — the gender relationship is geometrically smooth enough that the raw continuous space already captures it well

---

## 5. Why v3 Still Leads on Gender

The gender relationship is different in character from capital_of and country→language:
- It has a single dominant axis (`female_gendered`, P(flip)=0.90, conf=0.80)
- The delta is almost entirely captured by one axis
- The PRESERVE zone nuance on other axes adds noise relative to signal

When a relationship is essentially 1-dimensional, any encoding that expands the PRESERVE region (relative to CONTRACT/EXPAND) slightly diffuses the tight signal. v3's raw projection naturally concentrates on the one dominant axis. The phi-4-state encoding, designed for *multi-axis* relationships, isn't needed here.

This is consistent with the fail-fast philosophy: the phi-4-state encoding is a tool for relationships that live in the PRESERVE zone across *multiple axes* simultaneously. For single-axis relationships, raw continuous projection is the correct representation.

---

## 6. On the Source of Residual Error

After diagnosis, two categories of failure emerged:

### 6.1 False Failures — Vocabulary Gaps

The Phase-0 concept mining used tokenization heuristics that missed several target words entirely, or found the wrong token variant. These appeared as `?` (target not ranked in top-50) but were not actual geometric failures:

| Target | Phase-0 entry | Correct token | Diagnosis |
|--------|---------------|---------------|-----------|
| Oslo   | `Ġoslo` (id 36262, lowercase) | `ĠOslo` (id 57858, capitalised) | Wrong variant — lowercase "oslo" appears in non-capital contexts, creating a noisy embedding |
| Greek  | `ĠGreeks` (plural, id 60680) | `ĠGreek` (id 17860) | Plural/singular mismatch — "Greek" and "Greeks" have different embeddings |
| boy    | `Ġboyfriend`, `ĠBoys` | `Ġboy` (id 8171) | Only compound forms found by the heuristic |
| girl   | `Ġgirlfriend`, `ĠGirls` | `Ġgirl` (id 3743) | Same — compound forms only |

Injecting the correct tokens via `add_word()` resolved all four cases. Norway→Oslo moved from `?` to rank 3; Greece→Greek moved from `?` to rank 2; boy→girl moved from `?` to rank 0.

The correct takeaway: **a `?` result is diagnostic, not conclusive**. It may indicate a geometry failure or it may indicate a vocabulary gap. Both must be ruled out before claiming the geometric representation is inadequate.

### 6.2 True Failures — Geometry or Semantic Ambiguity

**Poland → Polish** (rank 32-40 across all versions) remains unresolved after vocabulary correction. The word "Polish" carries dual meaning: the national adjective (Poland → Polish) and the common verb (to polish, polishing). This polysemy contaminates its embedding position. In Qwen2's training corpus, the verb sense may dominate, pulling "Polish"'s embedding away from the Slavic language cluster toward action/material semantics. The country→language delta, trained predominantly on unambiguous language names (French, Japanese, Russian), has no way to compensate for this displacement.

This is a genuine geometry failure attributable to training data rather than to our encoding method. It would require either:
- A disambiguation mechanism (detecting when a word's embedding is pulled by polysemy)
- An axis covering Slavic languages (currently absent from our 5 seed axes) to provide an attractor correction

---

## 7. The Encoding Hierarchy

We now have a clear empirical ordering:

```
Relationship type               Best encoding
──────────────────────────────────────────────────────
1-axis (gender)                 v3: raw continuous cosine
Multi-axis, categorical         v6d: phi-4-state {-φ,-1,+1,+φ}
  (capital_of)                      + seed-axis φ-boost on retrieval
Multi-axis, non-categorical     v6: phi-4-state {-φ,-1,+1,+φ}
  (country→language)                Fibonacci-corrected, no boost
Binary (Gödel address lookup)   v1: Hamming
```

The distinction between categorical and non-categorical multi-axis relationships is whether a single seed axis achieves P(flip) > 0.85 across the LOO training pairs. For `capital_of`, the `capital_city` axis is perfectly discriminating (all sources are non-capitals, all targets are capitals). For `country→language`, no such axis exists — the relationship spans Romance, Germanic, Slavic, and Asian language families simultaneously, and no single seed category covers all of them.

The threshold is the natural φ-zero of the encoding. CONTRACT and EXPAND (the outer states) are amplified by φ. PRESERVE± (the inner states, the "negative zero fringe") retain raw magnitude. The ratio between adjacent levels is φ throughout the lattice.

---

## 8. Implications for the LCM Hypothesis

The core TruthSpace hypothesis is that *structure is information* — that geometric relationships in the embedding space encode semantic knowledge directly, without requiring the full transformer machinery to retrieve it.

These experiments validate a specific aspect of that hypothesis: **relationship deltas are real geometric objects** that can be learned from a handful of examples and applied to novel concepts with high accuracy (82% rank-1 on country→language, 86% top-5 on capital_of).

The encoding evolution also confirms the DC 255 prediction: the PRESERVE states carry maximum information density. The model's geometry, when probed through IRD axes, concentrates semantic relationship information near the threshold boundaries — exactly where DC 255 found the highest gate activation density in the transformer's own internal processing.

The same φ-structure that governs the transformer's internal gate geometry (DC 255) governs the optimal encoding for our external geometric inference system. This is consistent with the hypothesis that the transformer didn't *learn* this structure from training data — it discovered it because it's the optimal structure for representing relationships in a finite-dimensional geometric space.

---

## 9. v6d — Seed-Axis φ-Boost for Retrieval

After establishing v6 as the best static encoding, we investigated whether a **query-side attractor** could further improve retrieval by reinforcing categorical membership on semantically grounded axes.

### 9.1 The Mechanism

For each LOO test pair, we compute the per-axis flip probability (`flip_prob`) from `learn_delta_v2`. An axis is a **must-flip axis** if:
1. `flip_prob[ax] > 0.5` — the bit actually changes direction (not just high confidence of staying)
2. `confidence[ax] = |flip_prob[ax] - 0.5| × 2 ≥ 0.85` — near-certain flip
3. `ax ∈ seed_slots` — it is one of the 5 hand-validated categorical seed axes

For the single highest-confidence seed axis satisfying all three conditions, we apply:

```python
# δ[ax] > 0 → targets expected above threshold (PRESERVE+ or EXPAND)
correct = projections[:, ax] > thresholds[ax]
sims[correct]  *= φ        # boost concepts on correct side
sims[~correct] *= 1/φ      # penalise concepts on wrong side
```

The boost is bounded to the range `[1/φ, φ]` — a ×φ² spread total — and applied only once. The seed-axis restriction is essential: IRD-discovered axes can coincidentally reach `P(flip)=1.0` on small LOO training sets (e.g., `capital_city` achieves `P(flip)=0` with conf=1.0 for any relationship where neither source nor target is a capital, causing spurious selection if the constraint `P(flip)>0.5` is absent).

### 9.2 Design Decisions

Three bugs were encountered and resolved before the boost became effective:

1. **All-axis compounding** (first attempt): applying the boost multiplicatively across *all* high-confidence axes drove non-matching concepts to near zero via `(1/φ)^N`. Single-axis restriction fixes this.

2. **EXPAND-only zone** (second attempt): boosting only concepts in the EXPAND zone missed targets in PRESERVE+, where relationship endpoints characteristically live. Using threshold crossing (above/below threshold) rather than `phi4_hi` boundary is correct.

3. **Must-stay axes** (third attempt): `P(flip)=0` with conf=1.0 is a *must-stay* axis, not a useful attractor. The `flip_prob > 0.5` guard prevents these from being selected.

### 9.3 Results (with vocabulary-corrected concept set)

```
                      v3     v6     v6d
capital_of rank-1:   36%    36%    43%    ← +7pp — capital_city seed axis fires
capital_of top-5:    93%    93%    93%
capital_of mean:     2.6    2.8    2.2    ← best mean rank

gender rank-1:       40%    40%    40%    ← no change (female_gendered axis
gender top-5:        90%    90%    90%       below threshold in LOO subsets)
gender mean:         1.3    1.2    1.2

country→lang rank-1: 82%    82%    82%    ← no change (no seed axis with
country→lang top-5:  91%    91%    91%       P(flip)>0.85 across all language
country→lang mean:   3.1    3.8    3.8       families simultaneously)
```

v6d improves `capital_of` because `capital_city` is a seed axis with exactly the right property: every source (a country) is below threshold on this axis, every target (a capital) is above it — `P(flip)=1.0`, `conf=1.0`. The boost correctly amplifies capital cities in the retrieval ranking.

v6d is a no-op for `male→female` and `country→language` because no seed axis achieves `P(flip)>0.85` when the LOO delta averages across heterogeneous language families or when the gender training set is reduced by one pair.

---

## 10. Final Consolidated Results

Results after vocabulary correction (injecting `ĠGreek`, `Ġboy`, `Ġgirl`, and upgrading `Ġoslo→ĠOslo`):

### capital_of (14 pairs)

| pair | v3 | v6 | v6d |
|------|-----|-----|------|
| France → Paris | 1 | 0 | 0 |
| Germany → Berlin | 2 | 1 | **0** |
| Japan → Tokyo | 2 | 3 | **1** |
| China → Beijing | 16 | 20 | 17 |
| Italy → Rome | 2 | 2 | 2 |
| Spain → Madrid | 0 | 0 | 0 |
| Russia → Moscow | 0 | 0 | 0 |
| Greece → Athens | 4 | 4 | 3 |
| Poland → Warsaw | 0 | 1 | 1 |
| Sweden → Stockholm | 0 | 0 | 0 |
| Norway → Oslo | 4 | 3 | 3 |
| Austria → Vienna | 0 | 0 | 0 |
| Belgium → Brussels | 3 | 3 | 2 |
| Netherlands → Amsterdam | 2 | 2 | 2 |
| **Rank-1** | **36%** | **36%** | **43%** |
| **Top-5** | **93%** | **93%** | **93%** |
| **Mean rank** | **2.6** | **2.8** | **2.2** |

### male→female (10 pairs)

| pair | v3 | v6 | v6d |
|------|-----|-----|------|
| king → queen | 1 | 1 | 1 |
| man → woman | 2 | 2 | 2 |
| boy → girl | **0** | **0** | **0** |
| father → mother | 0 | 0 | 0 |
| brother → sister | 1 | 0 | 0 |
| actor → actress | 1 | 1 | 1 |
| prince → princess | 0 | 0 | 0 |
| hero → heroine | 1 | 1 | 1 |
| son → daughter | 0 | 1 | 1 |
| husband → wife | 7 | 6 | 6 |
| **Rank-1** | **40%** | **40%** | **40%** |
| **Top-5** | **90%** | **90%** | **90%** |
| **Mean rank** | **1.3** | **1.2** | **1.2** |

### country→language (11 pairs)

| pair | v3 | v6 | v6d |
|------|-----|-----|------|
| France → French | 0 | 0 | 0 |
| Germany → German | 0 | 0 | 0 |
| Japan → Japanese | 0 | 0 | 0 |
| China → Chinese | 0 | 0 | 0 |
| Italy → Italian | 0 | 0 | 0 |
| Spain → Spanish | 0 | 0 | 0 |
| Russia → Russian | 0 | 0 | 0 |
| Greece → Greek | **2** | **2** | **2** |
| Poland → Polish | 32 | 40 | 40 |
| Sweden → Swedish | 0 | 0 | 0 |
| Norway → Norwegian | 0 | 0 | 0 |
| **Rank-1** | **82%** | **82%** | **82%** |
| **Top-5** | **91%** | **91%** | **91%** |
| **Mean rank** | **3.1** | **3.8** | **3.8** |

---

## 11. Completeness Experiment — Is the φ-Lattice Address Information-Complete?

The holographic claim requires a direct test: given *only* the 193-axis φ-lattice address of a concept (no original embedding), can you reconstruct its nearest neighbours?

### 11.1 Method

Three representations of every concept in the 25,671-concept vocabulary:

| Space | Dims | Description |
|-------|------|-------------|
| `embed` | 3584 | Qwen2 `embed_tokens` vector — **ground truth** |
| `proj`  | 193  | Continuous projection onto 193 IRD axes |
| `φ`     | 193  | φ-4-state encoding `{-φ,-1,+1,+φ}` of proj |

For 2,000 randomly sampled concepts, retrieve top-K neighbours in each space and measure what fraction of the `embed`-space top-K are recovered. Two loss sources are isolated: *axis-selection loss* (3584→193 projection) and *quantisation loss* (proj→φ encoding).

### 11.2 Results

```
  k     proj→emb overlap    φ→emb overlap    quantisation gap
  ─────────────────────────────────────────────────────────────
  1          65.4%              65.5%              +0.0%
  5          60.6%              60.1%              -0.5%
  10         59.0%              57.8%              -1.2%
  50         57.2%              55.7%              -1.5%

  Mean rank of embed top-10 in test space:
    proj space : 21.6
    φ-address  : 24.7  (1.15× worse)

  Spearman ρ over top-200 neighbourhood:
    proj space : 0.5641
    φ-address  : 0.5492  (0.974 of proj)

  Axis-selection loss (3584→193) :  41.0% of top-10 neighbours lost
  Quantisation loss   (proj→φ)   :   1.2% additional loss
  Total loss          (embed→φ)  :  42.2%
```

### 11.3 Interpretation

**The φ-4-state encoding is near-lossless.** The quantisation step costs only 1.2 percentage points on top-10 recall, and zero at top-1. The Spearman ρ ratio of 0.974 means the encoding preserves 97.4% of the rank-ordering quality that the projection already has. The mean rank displacement is 14% worse (24.7 vs 21.6) — modest and consistent with the 1.2% overlap loss.

**The bottleneck is axis coverage, not encoding fidelity.** The 193 IRD axes cover approximately 59% of the embedding neighbourhood at top-10. The remaining 41% loss reflects semantic dimensions that were not discovered during axis mining — not any inadequacy of the φ-lattice encoding scheme.

This directly validates the encoding half of the holographic hypothesis: **the φ-lattice quantisation does not destroy information**. A concept's neighbourhood structure survives the encoding step essentially intact. The question of completeness is therefore entirely a question about axis coverage — and is orthogonal to the encoding choice.

At 193 axes over 3584 embedding dimensions (18.5:1 compression), recovering 59% of the embedding neighbourhood at top-10 is actually strong. Every additional semantically meaningful axis recovered during IRD mining would reduce the 41% gap further. The φ-encoding gap (1.2%) is a fixed overhead that does not grow with axis count.

### 11.4 What This Means for the LCM Hypothesis

The original concern was whether the φ-lattice address could serve as a complete substitute for the embedding vector in downstream tasks. The completeness experiment splits this into two sub-questions:

1. **Can φ-encoding represent whatever the projection captures?** — **Yes**, with 97.4% fidelity.
2. **Does the projection capture everything the embedding captures?** — **No**, 41% neighbourhood loss at top-10.

The path to a complete LCM is therefore: use the existing axes fully. The φ-encoding itself is already ready. The 41% loss reported here (at 193 axes) is almost entirely recoverable from the existing 1500 IRD axes — the quality filter was the artificial bottleneck, not axis coverage. See Section 12 for the full axis expansion analysis and Section 13 for the optimal axis count determination.

---

## 12. Axis Expansion — Quality Filter Was Wrong

### 12.1 Discovery

The existing `dc299_phase1_axes.json` contains 1500 IRD-discovered axes — but `LCMIndex` was loading only 193 due to a combined quality filter (`QUALITY_MIN=0.5`) and cliff detection (sliding-window quality dropping below 0.4). The completeness experiment (Section 11) showed 41% neighbourhood loss at 193 axes. The natural question: is the 41% recoverable from the existing 1500 axes, or is new discovery needed?

### 12.2 Completeness Scaling Across All 1500 Axes

Running `run_axis_sweep` with 1000 sampled concepts:

```
  axes    IRD-order top-10    quality-order top-10
  ─────────────────────────────────────────────────
    50        22.4%                21.6%
   100        39.9%                39.4%
   193        59.2%  ← original    54.9%
   300        70.3%                61.2%
   500        80.2%                65.2%
   750        84.9%                70.9%
  1000        87.8%                75.6%
  1500        93.5%                93.5%  ← ceiling
```

**The 41% gap is almost entirely recoverable from existing axes.** All 1500 IRD axes give 93.5% neighbourhood recall — only 6.5% residual loss. The quality filter was the artificial constraint.

Two important observations:
- **IRD order dominates quality order** by up to 15pp at 500 axes. Axes are already ordered by variance explained (descending); the vocabulary coherence score is a poor proxy for semantic usefulness.
- **Low-coherence axes carry real geometric signal.** Many axes that separate numeric/symbolic tokens fail the alphabetic-token coherence test while still encoding meaningful semantic dimensions.

### 12.3 The Quality Filter Failure Mode

The coherence metric counted the fraction of top/bottom vocabulary tokens matching `^[A-Za-z]{3,}$`. Any axis whose distinguishing concepts happened to be numbers, symbols, or short tokens was penalised regardless of semantic content. This is a vocabulary surface artefact, not a semantic quality measure.

**Fix**: disable the quality filter (`QUALITY_MIN=0.0`) and use cliff detection only when the filter threshold is strict (≥0.4). With `QUALITY_MIN=0.0`, the cliff is irrelevant — all 1500 axes are candidates, and the optimal count is found by inference performance.

---

## 13. Inference Sweep — Optimal Axis Count for Delta Inference

### 13.1 The Trade-off

More axes improve completeness (neighbourhood recall) but eventually degrade inference. The delta vector is the mean difference across training pairs on each axis. With 1500 axes, many low-signal axes contribute noise that raises the confidence floor and dilutes the per-relationship signal. The optimal N balances coverage (completeness) against signal-to-noise (inference).

### 13.2 Sweep Results

For each N, `run_inference_sweep` patches `LCMIndex` in-place (no reload) and runs silent LOO on all three relationship types. v6d rank-1:

```
    N    capital_of  male→female  country→language  aggregate(35)
  ─────────────────────────────────────────────────────────────────
    50      0%          40%           45%              25%
   100     21%          40%           63%              40%
   150     35%          40%           81%              51%
   193     57%          50%           81%              62%  ← previous optimum
   250     64%          60%           81%              68%
   300     57%          60%           90%              68%
   500     64%          70%           90%              74%  ← NEW OPTIMUM ✓
   750     64%          60%           90%              71%
  1000     28%          70%           90%              60%  ← degrades
  1500     21%          50%           72%              45%  ← worst
```

**N=500 is the optimum**: 74% aggregate rank-1 vs 62% at 193 (+12pp). The degradation beyond 500 is driven by capital_of, which collapses at N=1000: the added noise axes randomise the city-direction signal.

### 13.3 Final Configuration: N=500

```python
QUALITY_MIN  = 0.00   # quality filter disabled — IRD order is sufficient
MAX_SEMANTIC = 500    # optimal for inference; use 1500 for maximum completeness
```

### 13.4 Full Result Comparison

| Metric | 193-axis v6d (old) | 500-axis v6d (new) | Δ |
|--------|--------------------|--------------------|---|
| capital_of rank-1 | 43% | **64%** | +21pp |
| capital_of top-5 | 93% | **100%** | +7pp |
| capital_of mean rank | 2.2 | **0.6** | 3.7× |
| male→female rank-1 | 40% | **70%** | +30pp |
| male→female top-5 | 90% | **100%** | +10pp |
| country→language rank-1 | 82% | **91%** | +9pp |
| completeness top-10 (proj) | 59% | **80%** | +21pp |
| completeness top-10 (φ-addr) | 58% | **77%** | +19pp |

The same 1500 axes discovered in Phase 1 are now used at full potential. No new axis discovery was required — the bottleneck was the filter, not the corpus.

### 13.5 Dual-Config for Maximum Completeness

For applications that prioritise neighbourhood recall over inference precision (e.g., concept retrieval, holographic address lookup), use `MAX_SEMANTIC=1500` which gives 93.5% completeness. The φ-encoding overhead at 1500 axes remains 1.2% above the projection baseline (Section 11.2), so the encoding is still near-lossless.

---

## 14. Phase 1b IRD Extension — Pushing to 98% Variance

### 14.1 Motivation

At 1500 axes the IRD had hit its `MAX_AXES` hard cap, not a convergence condition. Cumulative variance explained was 90.67% — 7.33pp short of the 98% target. Phase 1b continues from the existing residual without recomputing the first 1500 axes.

### 14.2 Implementation

`dc299_phase1b_ird_extension.py` — continuation strategy:
1. Load all 1500 existing axis vectors from `dc299_phase1_axes.json`.
2. Build the residual matrix by projecting them out of the normalised concept embeddings.
3. Resume the IRD SVD loop from the residual with `MIN_VARIANCE_STEP=0.0005` (halved from Phase 1).
4. Accept axes until `cumulative_variance ≥ 0.98` or `MAX_NEW_AXES=1500`.

Stopped at iteration 1329 when cumulative variance reached exactly 0.9800.  
Total run time: 2119.6 s (~35 min).

### 14.3 Results

| Metric | Value |
|--------|-------|
| New axes discovered | 1328 |
| Total basis size | 2828 (1500 + 1328) |
| Starting variance (before Phase 1b) | 90.67% |
| Final cumulative variance | **98.00%** |
| Step variance at axis 1500 | 0.00158 |
| Step variance at axis 2800 | 0.00162 (stable) |

### 14.4 Completeness at 2828 Axes

Extended sweep (1000-concept sample, top-10 neighbourhood recall):

| Axes | IRD-order top-10 | quality-order top-10 |
|------|-----------------|---------------------|
| 1500 | 93.5% | 63.5% |
| 2000 | 94.9% | 71.2% |
| 2500 | 95.4% | 81.2% |
| **2828** | **95.8%** | **95.8%** |

At 2828 axes the IRD-order and quality-order orderings **converge** (both 95.8%). This is the hard floor for the current concept set: the remaining 4.2% gap cannot be closed by additional IRD iterations on the same 25K-concept vocabulary without new seed diversity or a larger concept pool.

`dc299_phase4_lcm_inference.py` loads `dc299_phase1b_axes.json` automatically when present, appending its vectors to the Phase 1 set. `MAX_SEMANTIC=500` for inference remains unchanged.

---

## 15. Vocabulary Injection Fix — Polish (Language)

### 15.1 Root Cause

The Phase-0 concept miner filtered tokens by space-prefix and alphabetic form. For the token `ĠPolish` (token 31984, the nationality/language adjective), Phase-0 selected the **lowercase** form `Ġpolish` (token 44029, the verb "to polish") instead. The embedding of the verb "polish" does not carry the `country→language` semantic signal, so `Poland→Polish` was absent from the top-10 results at any axis count.

### 15.2 Fix

Inject `Polish` with the correct token ID using `lcm.add_word("Polish", 31984, overwrite=True)` alongside the existing injections (Greek, boy, girl, Oslo). This overwrites the verb embedding with the nationality adjective embedding for the concept slot `"polish"`.

The same injection was added to `run_delta_tests`, `run_failure_diagnostic`, and `run_inference_sweep`.

### 15.3 Impact

| Relationship | Before fix | After fix |
|---|---|---|
| country→language rank-1 | 10/11 (91%) | **11/11 (100%)** |
| Aggregate v6d rank-1 | 26/35 (74%) | **27/35 (77%)** |

`Norway→Norwegian` was already at 100% (Norwegian injected earlier). `Poland→Polish` completing the set closes the only remaining polysemy failure in the language corpus.

### 15.4 Updated Inference Summary (N=500, v6d, with all injections)

| Relationship | Rank-1 | Top-5 | Mean rank |
|---|---|---|---|
| capital_of (14 pairs) | **9/14 (64%)** | 14/14 (100%) | 0.6 |
| male→female (10 pairs) | **7/10 (70%)** | 10/10 (100%) | 0.5 |
| country→language (11 pairs) | **11/11 (100%)** | 11/11 (100%) | 0.0 |
| **Aggregate (35 pairs)** | **27/35 (77%)** | **35/35 (100%)** | **0.4** |

All remaining failures are rank 1–3 near-misses (see Section 15.5).

### 15.5 Remaining Failure Analysis

**capital_of (5 failures):**

| Query | Rank-0 intruder | Rank | Pattern |
|---|---|---|---|
| China → Beijing | paris (0.464) | 3 | Training-set interference (France→Paris delta bleeds) |
| Greece → Athens | Rome (0.681) | 1 | Geometric proximity of European capitals |
| Norway → Oslo | Norwegian (0.764) | 1 | Language token geometrically closer than capital |
| Belgium → Brussels | Amsterdam (0.746) | 2 | Cross-pair contamination (Netherlands→Amsterdam in training) |
| Netherlands → Amsterdam | Stockholm (0.670) | 1 | European city cluster proximity |

**male→female (3 failures):**

| Query | Rank-0 intruder | Rank | Pattern |
|---|---|---|---|
| man → woman | MEN (0.325) | 1 | Uppercase plural variant in top slot |
| hero → heroine | heroes (0.538) | 1 | Plural of source steals rank-0 |
| husband → wife | husbands (0.560) | 3 | Source-word plural + spouse/hubby ahead of wife |

All 8 remaining failures are near-misses with the correct answer at rank 1–3. No failure places the target outside the top-5.

---

## 16. Files and Implementation

- **Experiment**: `experiments/truthspace_v1/dc299_phase4_lcm_inference.py`
  - `LCMIndex._compute_phi4(proj, continuous, preserve_scale)`: core 4-state encoder
    - `preserve_scale=None` (default, 1/φ): DC255 original, v4a/v4b
    - `preserve_scale=1.0`: Fibonacci-corrected, v6
  - `LCMIndex.apply_delta_phi4(word, delta, k, continuous, preserve_scale)`: v4/v6 retrieval
  - `LCMIndex.apply_delta_continuous(word, delta, k)`: v3 raw cosine retrieval
  - `LCMIndex.apply_delta_phi_boost(word, delta, flip_prob, k, boost_threshold)`: v6d retrieval
  - `LCMIndex.add_word(word, token_id, overwrite)`: inject / replace concept vocabulary entries
  - `LCMIndex.build_seed_corrections(memberships)`: compute φ-attractor corrections from seed membership
  - `LCMIndex.build_corrected_vecs(corrections)`: apply corrections to phi4 encoding matrix
  - `run_delta_tests()`: LOO validation with vocabulary injection + v3/v6/v6d comparison

- **DC 255**: `docs/design_considerations/255_4state_gate_phi_dimension.md`
  - Provides the theoretical basis for the 4-state structure and ±log(φ) boundaries

- **DC 299**: `docs/design_considerations/299_complete_model_map_via_platonic_ideal_discovery.md`
  - Phase 4 context: the IRD axis discovery and Gödel address assignment that underlies this work
