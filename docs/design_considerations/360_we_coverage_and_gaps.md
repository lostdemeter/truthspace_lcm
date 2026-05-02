# DC 360: W_E Coverage and Gaps

**Day 189 | W_E has 99.3% coverage of common English vocabulary and 85.4%
coverage of country→capital pairs. Sub-token composition fails as a gap
workaround. Coverage gaps are genuine and require full transformer fallback.**

---

## Overview

Day 188 measured the accessibility of W_E as a knowledge store:
how many words and relation pairs can be directly embedded as single tokens?
Can missing words be composed from sub-token embeddings?

---

## Finding 1: Near-Complete Common Word Coverage (99.3%)

Of 706 unique common English words tested, 701 (99.3%) tokenize to a single
token in Qwen2-1.5B-Instruct's vocabulary.

```
Single-token (W_E direct access): 99.3%
Multi-token (sub-token only):      0.7%

Multi-token examples:
  equate    → ['Ġequ', 'ate']
  syllable  → ['Ġsyll', 'able']
  excite    → ['Ġexc', 'ite']
  consonant → ['Ġconson', 'ant']
  crease    → ['Ġcre', 'ase']
```

These 5 outliers are mid-frequency English words that the BPE tokenizer split
because shorter sub-tokens ("ate", "ant", "ite") are more frequent as
standalone tokens. The vocabulary is overwhelmingly coverage-complete for
everyday English.

**Implication:** For common English vocabulary, W_E can be treated as a
dense lookup table. There is no need for sub-token approximations for
ordinary language.

---

## Finding 2: Relational Pair Coverage (85.4% for Country→Capital)

Of 41 country→capital pairs:

```
Countries single-token:  41/41 (100%)  ← all country names are single tokens
Capitals single-token:   35/41 (85.4%) ← most capital cities are single tokens
Both accessible in W_E:  35/41 (85.4%)
```

**Multi-token capitals (6/41):**
```
Country      → Capital       → Sub-tokens
Brazil       → Brasilia      → ['ĠBras', 'ilia']
Nigeria      → Abuja         → ['ĠAbu', 'ja']
Bangladesh   → Dhaka         → ['ĠDh', 'aka']
Vietnam      → Hanoi         → ['ĠH', 'anoi']
Ukraine      → Kyiv          → ['ĠKy', 'iv']
Romania      → Bucharest     → ['ĠBuch', 'arest']
```

**Pattern:** Multi-token capitals are less common cities in Western training
corpora (post-Soviet: Kyiv; African: Abuja; Southeast Asian: Dhaka, Hanoi;
South American: Brasilia). The BPE vocabulary reflects training data frequency —
rare city names were never seen often enough to earn a single-token slot.

**Generalization:** The 85% figure applies broadly to geopolitical domains.
For domains involving only common English words (antonyms, gender, colors,
number words), coverage approaches 100%. For proper nouns with geographic
distribution (capitals, person names, company names), expect 75-90%.

---

## Finding 3: Sub-Token Composition Fails

The natural workaround for multi-token words is to average their sub-token
embeddings and use the composite vector:

```
W_E_composite[Brasilia] = mean(W_E['ĠBras'], W_E['ilia'])
```

Testing: cosine similarity between `(W_E[Brazil] + mean_capital_direction)`
and `W_E_composite[target_city]`:

```
Brazil  → Brasilia  (composite): cos = 0.161
Nigeria → Abuja     (composite): cos = 0.129
Vietnam → Hanoi     (composite): cos = 0.116
Ukraine → Kyiv      (composite): cos = 0.082
Romania → Bucharest (composite): cos = 0.199
```

All cosines are in the 0.08–0.20 range — essentially no signal. For context,
cosine similarity of 0.20 in 1536 dimensions is near-random. Sub-token
averages do NOT reconstruct capital city meaning.

**Why composition fails:**

Sub-token embeddings encode **morphological patterns**, not semantic content.
`W_E['ĠBras']` encodes the morpheme "Bras-" as it appears in all contexts
where it occurs (brassieres, brass, Brasilia, Braselton GA...). The average
of "Bras" + "ilia" morpheme embeddings does not converge to the meaning
of the city of Brasilia — the morphemes are polysemous fragments.

This is distinct from compound words in English (e.g., "notebook" = note+book)
where individual sub-words DO carry semantic content. Capital city names are
opaque proper nouns — their meaning cannot be decomposed from their spelling.

---

## The Coverage Architecture

Based on Days 188 findings, W_E coverage follows this hierarchy:

```
Coverage Level               Fraction    Access Method
─────────────────────────────────────────────────────────────────
Common English words         ~99.3%      Direct W_E lookup
Country names / languages    ~100%       Direct W_E lookup
Capital cities (major)       ~85-90%     Direct W_E lookup
Capital cities (minor)       ~10-15%     FAILS — no W_E access
Person names (famous)        ~60-80%*    Direct W_E lookup (estimated)
Scientific terms             ~40-70%*    Depends on frequency
Proper nouns (all)           ~50-65%*    Mixed

*Estimated, not measured
```

---

## The Coverage Gap Problem

For the ~15% of country→capital pairs not accessible in W_E:

**Option 1: Skip** — Report "not in W_E" and move on.
**Option 2: Full transformer** — Run the query through transformer layers
  and use output logits to find the most likely completion token. This works
  but requires full O(L × d²) forward pass — much more expensive.
**Option 3: Nearest-token approximation** — Find the single-token word with
  highest cosine to the sub-token composite. This maps "Ĝbrasilia-ish" to
  some real single-token word. Likely maps to a wrong city.

**Recommendation:** Option 2 (full transformer) is the only correct approach.
The W_E-based approach is appropriate for the 85% coverage case; the 15%
coverage gap falls outside the TruthSpace W_E hypothesis and requires the
full transformer.

**This is consistent with the fail-fast philosophy:** W_E either contains
the token or it doesn't. There is no graceful approximation.

---

## Impact on Multi-Hop Chains (from DC 355)

Multi-hop chain accuracy was measured on the 10-capital subset where all
words were conveniently single-token. The theoretical 85% pair coverage
means multi-hop chains of length k have:

```
Expected coverage = 0.854^k

k=1: 85.4%  (one hop)
k=2: 72.9%  (two hops)
k=3: 62.3%  (three hops)
k=5: 45.3%  (five hops)
```

Multi-hop chains degrade coverage exponentially. For practical multi-hop
retrieval over diverse geopolitical domains, single-token coverage is the
binding constraint.

---

## Summary

```
Finding                              Value
─────────────────────────────────────────────────────────────────
Common English word coverage         99.3%  (701/706)
Country→capital full coverage        85.4%  (35/41)
Sub-token composition cosine         0.08–0.20 (fails)
Why gaps occur                       BPE frequency threshold: rare
                                     proper nouns don't earn single tokens
Workaround for gaps                  Full transformer (only correct option)
Multi-hop k=3 expected coverage      ~62% (for geopolitical domains)
```

---

## Files

- `expedition_day188_we_coverage.py` — coverage measurement
- `day188_we_coverage.json` — results
- `355_multihop_chains.md` — multi-hop chain accuracy
- `357_relational_boundary_revised.md` — encoding type classifier
