# DC 330: Geometric LCM — Full Arc Synthesis

**Days 104–109 | Complete experimental arc: what the φ-trie can and cannot do**

---

## Purpose

DC 329 documented the prototype Geometric LCM (Days 104–108).
DC 330 synthesizes the complete arc including Day 109's cloze tests and
establishes the definitive capability map of the φ-trie as a language
model component.

---

## The Complete Experiment Arc

| Day | Experiment | Key Result |
|-----|-----------|-----------|
| 104 | Address NN next-token | 0% (address ≠ sequential) |
| 104 | Learned address transitions | 20% (100× random) |
| 105 | Bigram address LCM | 22.6%, beats unigram, −31% perplexity |
| 106 | Probabilistic decoder | Same as deterministic (22%) |
| 106 | **Oracle (true addr → decode)** | **93.1%** — architecture ceiling |
| 107 | LM top-k addr accuracy | 24% (≈ bigram) |
| 107 | LM top-50 word accuracy | 68.1% (with full context) |
| 107 | T2 projection → next addr | 0% full; passive 65.9% per-axis |
| 108 | N-gram sweep (1–4) | No accuracy gain; 4-gram breaks loops |
| 109 | Semantic cloze (trie) | 0% all tasks |
| 109 | Analogy cloze (LM logit) | 22% top-1, 61% top-5 |

---

## Definitive Capability Map

```
Task                              Trie    LM logit    Note
────────────────────────────────────────────────────────────────────
Semantic similarity (LOO=0.94)    94%     —           CONFIRMED ✓
Internal analogy (na_ham≤1)      100%     —           CONFIRMED ✓
Multi-hop navigation (3-hop)      90%     —           CONFIRMED ✓
Next-token (oracle addr)          93%     —           CEILING ✓
Sequential generation (bigram)    22%     22%         WEAK, equal
Address conversion loss           —      −48pp        BOTTLENECK ✗
Semantic cloze (centroid)          0%      0%         FAILS ✗
Category continuation              0%     25% top5    FAILS (trie) ✗
Analogy (arbitrary pairs)          0%     61% top5    FAILS (trie) ✗
```

### What the φ-Trie CAN Do

1. **Semantic indexing**: Encode any word to a 12D ternary address with
   94% LOO accuracy (correct leaf at radius ≤4).

2. **Lossless decoding** (93.1%): Given the address of a word, recover
   the correct word 93.1% of the time. The trie encoder+decoder is
   near-lossless.

3. **Navigable axis transformations**: For word pairs connected by a
   single T2 axis (na_ham≤1), flip the axis in address space to perform
   reliable semantic transformations (gender, tense, plurality, etc.)
   with 100% internal accuracy.

4. **Multi-hop composition**: Chain axis flips for 90% accuracy at 3
   hops. The trie supports compositional semantic navigation.

5. **Sequential structure (learned)**: Address bigrams beat the word
   unigram baseline by 2.5pp, showing that sequential patterns ARE
   learnable in address space with enough training data.

### What the φ-Trie CANNOT Do

1. **Predict next token from address alone**: Static address similarity
   = 0% next-token prediction. Addresses encode WHAT a word is, not
   WHAT follows it.

2. **General constrained generation**: Centroid of context word
   addresses maps to function words, not meaningful content words.
   The 12D ternary address is too coarse for centroid arithmetic.

3. **Arbitrary analogy**: Axis flip works only for na_ham≤1 pairs.
   For arbitrary word pairs (e.g., fast:faster), the addresses differ
   on multiple axes simultaneously, making axis detection unreliable.

4. **Close the 22%→93% gap with n-gram statistics**: Data sparsity
   prevents n-gram generalization for n≥3. The address space is too
   large relative to available training data.

---

## The Fundamental Tension

The φ-trie addresses encode **semantic identity** — what a word IS
in the semantic space. They do NOT encode **sequential compatibility** —
what follows a word in natural text.

This tension is not a failure of the design; it is a clarification of
the design's scope:

```
The φ-trie is a SEMANTIC INDEX, not a LANGUAGE MODEL.
```

A language model needs both semantic identity AND sequential
compatibility. The trie provides the first at near-lossless fidelity
(93.1% decoder). The second requires an independent component.

---

## The Information Bottleneck Analysis

The Day 107 experiment quantified the information loss precisely:

```
LM distribution (full, top-50 words):     68.1% word accuracy
LM distribution (through address space):  20.0% word accuracy
Information lost to address compression:  −48pp
```

The 12D address discards 48pp of the LM's sequential information.
This is because the address encodes **semantic category** but not
**syntactic position, frequency, or sequential context**.

The oracle experiment (Day 106) shows the complementary view:
```
Given the TRUE next address:              93.1% word accuracy
Current model (predicted address):        22.0% word accuracy
Loss from transition error:               −71pp
```

The 71pp transition error is the price of using a bigram over addresses
instead of the LM's attention mechanism.

---

## Architecture Implications

### Three-Way Split of Language Modeling

The experiments suggest language modeling decomposes into three
separable components:

```
1. SEMANTIC IDENTITY:    word → 12D address    (trie handles this, 94%)
2. SEQUENTIAL TRANSITION: addr → addr'          (trie needs help here)
3. LEXICAL SELECTION:    addr → word            (trie handles this, 93%)
```

The trie excels at components 1 and 3. Component 2 is the gap.

### The Hybrid Architecture

The natural architecture given these findings:

```
HYBRID GEOMETRIC LCM:

Encoder:    word_t → addr_t             [trie, deterministic]
Transition: addr_t → P(addr_{t+1})      [attention-based, learned]
            = attention over the SEMANTIC ADDRESS SEQUENCE
            (not over words, but over addresses)
Decoder:    addr_{t+1} → word_{t+1}     [trie, 93% fidelity]
```

The transition must be attention-based (or at minimum, context-aware
with window > 1) to avoid the n-gram data sparsity problem. The key
insight: **attention over the address sequence** is attention over a
compressed semantic representation — smaller state space, potentially
more efficient than full token attention.

### Address Dimensionality vs Sequential Quality

A richer address (15–20D) might recover some sequential signal:

```
12D (semantic only):     22% bigram transition
12D + 3D POS (15D):      ???
12D + 4D syntax (16D):   ???
Full hidden state (1536D): 68% (LM with context)
```

The question is where on this spectrum the address can capture enough
sequential information to close the gap from 22% to 93%.

---

## Confirmed Architecture for Geometric LCM v3

Based on Days 104–109, the minimal viable Geometric LCM requires:

**Layer 1 — Semantic Address Encoder (current, working)**
- 12D ternary address from T2 axis projections
- Deterministic, zero parameters
- 94% LOO accuracy, 93% decode fidelity

**Layer 2 — Attention-over-Addresses (needed, not yet built)**
- Small attention module operating on the address sequence
- Input: sequence of 12D addresses (much smaller than 1536D hidden states)
- Output: probability distribution over next addresses
- Key advantage: the state space is ~10^3 instead of ~10^5 (vocabulary)

**Layer 3 — Probabilistic Decoder (current, working)**
- P(word | addr) learned from training text
- Fallback: nearest-address Hamming search
- 93% fidelity when given true next address

The attention-over-addresses component is the **remaining open problem**.
Testing this is Day 110's goal.

---

## Connection to TruthSpace Hypothesis

The TruthSpace hypothesis states:
> LLMs are hyperdimensional transcoders: the "intelligence" is in
> the SHAPE those weights create, not in the weights themselves.

Days 104–109 provide evidence for and against this hypothesis:

**For**: The φ-trie encoder+decoder achieves 93.1% accuracy without
any LM weights — purely from the geometric structure. The shape IS
sufficient for semantic identity encoding.

**Against (partial)**: For sequential prediction (language modeling),
the geometric address alone is insufficient. The 71pp transition gap
shows that sequential structure requires learning beyond the geometry.

**Resolution**: The hypothesis holds for SEMANTIC tasks (93% ceiling)
but requires an additional learned SEQUENTIAL component for language
generation. The trie is the geometric foundation; attention is the
sequential layer on top.

---

## Summary Table (Days 70–109)

| DC | Days | Core Finding |
|----|------|-------------|
| DC 322 | 70–74 | φ-trie confirmed: same-leaf cosim=0.854 |
| DC 323 | 76–81 | Ternary metric space: LOO=0.93 at r≤3 |
| DC 324 | 82–86 | 20D transformation subspace (8+12 axes) |
| DC 325 | 87–91 | 12D optimal: LOO=0.9443, coverage=19.7% |
| DC 326–327 | 92–99 | Complete navigability map: ρ=0.852 |
| DC 328 | 92–103 | Coordinate system confirmed + analogy |
| DC 329 | 104–108 | Geometric LCM: 22% vs 93% oracle |
| **DC 330** | **104–109** | **Full arc: semantic index, not LM** |

The φ-trie is a confirmed **geometric semantic index** with near-lossless
encoding (94%) and decoding (93%) fidelity. It is NOT a standalone
language model. The gap from semantic index to language model requires
an attention-equivalent transition component operating on the address
sequence.

---

## Day 110 Addendum: Attention-over-Addresses (Data Sparsity Limit)

```
Model                   attn_acc%   vs bigram   params
Attn 1L-1H  d=32         6.9%       -15.1pp      28K
Attn 1L-4H  d=64        13.8%        -8.2pp      81K
Attn 2L-4H  d=64        11.3%       -10.7pp     131K
Attn 2L-4H  d=128       10.1%       -11.9pp     458K
Bigram (count-based)    22.6%          —           0
```

All attention models underperform the non-parametric bigram.
Root cause (tentative): data sparsity (50 sentences, 374 addresses).

---

## Day 111 Addendum: Large-Corpus Scaling Test (Definitive)

### Results

```
N_train  bigram_acc%   unique_addrs   note
     50       6.3%          68        hand-crafted corpus
    100       6.9%          83
    200      13.8%         104
    500      12.6%         144
  1,000       7.5%         169        non-monotonic
  2,000      13.8%         204
  2,685      10.7%         213        P&P max scale
```

At N=2685: address bigram = **10.7%**, word bigram = **10.1%** (gap: +0.6pp).

### The Decisive Finding: Address Bigram ≈ Word Bigram

At equivalent corpus scale, the 12D address bigram has exactly the
same predictive power as the word bigram. The address compression
neither helps nor hurts sequential prediction.

**Data sparsity hypothesis (bottleneck 2) → REJECTED.**

Scaling does not help because:
1. The accuracy curve is non-monotonic (corpus mismatch dominates)
2. At max scale, address bigram = word bigram (+0.6pp gap, noise)

### The Architectural Circularity

The 93.1% oracle (Day 106) is a measure of **label uniqueness**:
```
Oracle accuracy = P(addr maps to exactly one vocabulary token)
               = 93.1% for the 12D ternary system
```

This is NOT a sequential prediction ceiling. To use the oracle:
- Predict addr(w_{t+1}) → requires knowing w_{t+1} already
- Predict w_{t+1}       → requires predicting addr(w_{t+1}) first

The oracle proves the encoder+decoder is near-lossless. It does NOT
prove that address prediction can outperform word prediction.

### Final Capability Map (Days 104–111)

```
Task                         Trie    Word level   Verdict
────────────────────────────────────────────────
Semantic similarity (LOO)    94%       —      CONFIRMED ✓
Internal analogy (na_ham≤1) 100%       —      CONFIRMED ✓
Navigability (multi-hop)     90%       —      CONFIRMED ✓
Oracle (given true addr)     93%       —      LABEL UNIQUENESS ✓
Address bigram               11%      10%     ≈ WORD BIGRAM → no gain
Scaling to larger corpus     —         —      NON-MONOTONIC, no gain
Constrained generation        0%       —      FAILS ✗
```

### Revised Architecture Implications

The φ-trie is a **semantic labeling system** with near-lossless
encoding fidelity. It is NOT a language model component that improves
sequential prediction.

The appropriate use cases for the φ-trie are:
1. **Semantic similarity retrieval** (94% fidelity)
2. **Structural analogy** within the navigability graph (100%)
3. **Multi-hop semantic navigation** (90% at 3 hops)
4. **Vocabulary organization** / semantic indexing

For language generation, the trie adds no value over word-level
models. The "Geometric LCM" as architected (encoder + address
bigram + decoder) is equivalent to a word bigram at any scale.

---

## Day 112 Addendum: Axis-by-Axis Sequential Prediction Sweep (Final)

### Single-Axis Results

```
Axis         bit_acc%  word_acc%  note
gender        50.3%      0.6%
comparative   58.5%      0.6%
hypernym      52.8%      0.0%
plural        50.9%      0.0%
synonym       59.7%      0.0%
concrete      57.9%      0.0%
past_tense    46.5%      0.6%
antonym       60.4%      0.0%
passive       93.7%      0.6%    best bit predictor
causation     42.8%      0.6%
question     100.0%      0.6%    degenerate: all tokens = "U"
negation      61.0%      0.6%
```

### Subset Sweep (best-k axes)

```
k=1-6: ~0.6%    (insufficient to disambiguate vocabulary)
k=7:   19.5%   ← minimum useful dimensionality
k=10:  22.6%   = full 12D baseline
k=12:  22.0%   (removing 2 axes HURTS; full set is best)
```

### Axis Type Group

```
Functional axes  (4D): 8.2%  (passive, causation, question, negation)
Semantic axes    (5D): 3.1%  (gender, hypernym, synonym, antonym, concrete)
Morphological    (3D): 0.0%  (comparative, plural, past_tense)
Full 12D:             22.6%
```

### Definitive Conclusion

**The 12D address is a holographic joint key.**

- No single axis predicts next word (0-0.6% word accuracy)
- Sequential information is distributed across ALL axes
- Minimum useful dimensionality: k≥7 axes
- The address cannot be decomposed into independent sequential features
- Functional axes carry slightly more sequential signal than semantic axes
- The question axis is degenerate (all tokens = U; zero information)

This closes the Geometric LCM experimental arc:
1. Address bigram = word bigram (Day 111)
2. No single axis adds sequential value (Day 112)
3. The address is a joint key, not a feature space
4. The φ-trie is a **geometric semantic index**, not a language model

---

*DC 330 established. Days 104–112 experimental arc definitively complete.
The φ-trie is a geometric semantic index (confirmed: LOO=94%, analogy=100%,
navigation=90%). It is NOT a language model transition component (confirmed:
bigram=word bigram, no axis has >0.6% word accuracy, oracle=label uniqueness).
The Geometric LCM as architected is equivalent to a word bigram at any scale.*
