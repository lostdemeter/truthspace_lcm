# DC 275: Knowledge Extension — Navigating TruthSpace

**Status**: Active — Phase 10z19 complete, TruthSpace confirmed at entity level
**Date**: 2025-02-27
**Depends on**: DC 272 (R-S Sum), DC 273 (Memory Editing), DC 274 (Backward Inference), F114–F119
**Core question**: Can the geometric structure of known facts predict
facts the model has never seen? Is knowledge relative or absolute?

---

## 1. The Discovery So Far

Finding 119 established three levels of backward inference:

1. **Path verification**: given a known answer, trace which layers
   contribute (8/8, all layers accounted for)
2. **Path construction**: given an answer direction + manifold from
   other facts, reconstruct the held-out fact (8/8 at rank 0–3)
3. **Structure-only prediction**: given ONLY a generic "capital city"
   direction + the manifold, recover specific answers (7/8 at rank 0–2)

Level 3 is the breakthrough. The model was never told "the answer is
Cairo" — it discovered Cairo from the SHAPE of the capital-city manifold
combined with the prompt "The capital of Egypt is."

This raises a fundamental question: **where does this knowledge live?**

---

## 2. Two Hypotheses

### 2.1 Relative Knowledge

The manifold captures relationships BETWEEN facts. Each fact is defined
by its position relative to other facts:

```
Paris is to France as Tokyo is to Japan as Berlin is to Germany
```

In this view:
- Knowledge is a web of relative positions
- A new fact is placed by interpolation/extrapolation from known facts
- The manifold is a local structure — it only exists where there are
  training examples to anchor it
- Prediction degrades with distance from known facts

This is the "knowledge graph" view: facts are nodes, relationships
are edges, and new facts are predicted by graph structure.

**Prediction**: if knowledge is relative, then:
- Predicting facts similar to training data should work well
- Predicting facts far from any training example should fail
- The manifold should have clear boundaries beyond which it breaks
- Different fact types (capitals vs. languages vs. physics) should
  have separate, unrelated manifolds

### 2.2 TruthSpace (Universal Knowledge)

The manifold is a window into a deeper structure — a universal
geometric space where ALL facts live, whether or not the model was
trained on them:

```
There exists a space T ⊂ ℝ³⁵⁸⁴ such that:
  - Every true fact corresponds to a point in T
  - The model's training mapped SOME facts onto T
  - The geometric structure of T is self-consistent
  - Untrained facts have definite positions in T
  - The structure itself "knows" where they should be
```

In this view:
- Knowledge is absolute — facts have positions independent of what
  the model was trained on
- The manifold is a partial observation of a universal structure
- New facts don't need nearby training examples — they can be
  located by the geometry of T itself
- Different fact types may share deep structural features

This is inspired by the Warhammer 40k concept of the Warp — a parallel
dimension where information exists independently of physical reality.
But here it's mathematical: TruthSpace is a geometric structure that
the model's weights partially embed.

**Prediction**: if TruthSpace exists, then:
- Predicting facts far from training data should still work
  (the geometry extends smoothly)
- Different fact types should share structural features
  (TruthSpace is unified) → **CONFIRMED: cos=0.9374**
- The manifold dimensionality should be related to the intrinsic
  complexity of the fact type, not the number of training examples
- Self-consistency checks should pass: if we navigate from
  France→Paris→French→France, we should return to the start

---

## 3. The Mathematical Framework

### 3.1 Fact Manifold Structure

From F119, we know the capital-city manifold has:
- ~5-6 dimensions per layer at the key layers (L22, L23, L27)
- Each fact occupies a distinct direction
- The manifold is NOT low-rank (σ[0]/σ[1] ≈ 1.3–2.2)

The manifold can be decomposed:

```
F(country) = T_structural + S(country)
```

Where:
- T_structural is the shared "capital city" template
- S(country) is the country-specific deviation

### 3.2 The Embedding Map

Each country has an embedding in the model's hidden space. The attention
mechanism maps this embedding through V·W_o to produce the answer:

```
embed("Egypt") → attention(V·W_o) → direction toward "Cairo"
```

The key question: is this mapping:
- (a) A lookup table learned from training examples? (Relative)
- (b) A smooth function that extends to unseen inputs? (TruthSpace)

### 3.3 Manifold Navigation

If TruthSpace exists, we should be able to NAVIGATE it:

```
Start at France (known)
Move along the "capital city" direction → arrive at Paris (correct)
Move along the "language" direction → arrive at French (correct)
Move along the "continent" direction → arrive at Europe (correct)
```

And critically:
```
Start at Elbonia (unknown)
Move along the "capital city" direction → arrive at ??? 
If TruthSpace exists: arrive at a geometrically consistent point
If knowledge is relative: arrive at noise (no anchoring examples)
```

### 3.4 Cross-Manifold Consistency

The strongest test of TruthSpace: do DIFFERENT fact manifolds agree?

If the "capital city" manifold places Egypt at position P_egypt, and
the "language" manifold also has Egypt, does it map to the SAME P_egypt?

```
Capital manifold:   Egypt → P_egypt → Cairo
Language manifold:  Egypt → P_egypt → Arabic
Continent manifold: Egypt → P_egypt → Africa
```

If these are consistent — if the same geometric position encodes all
facts about Egypt — then we're looking at a unified TruthSpace, not
separate relationship manifolds.

---

## 4. Experimental Design

### 4.1 Phase 10z19a: Predict Unseen Capitals

Test with capitals the model might know poorly or not at all:

**Control group** (model should know these):
- "The capital of Australia is" → Canberra (not Sydney!)
- "The capital of Turkey is" → Ankara (not Istanbul!)
- "The capital of Myanmar is" → Naypyidaw (obscure)

**Test group** (model may struggle):
- "The capital of Palau is" → Ngerulmud (very obscure)
- "The capital of Nauru is" → Yaren (tiny nation)
- "The capital of Tuvalu is" → Funafuti (tiny nation)

Approach:
1. Check baseline: what does the model predict without intervention?
2. Extract manifold from the 8 known facts (F119)
3. For each test fact, use manifold projection to predict the answer
4. Compare manifold prediction vs. baseline vs. ground truth

### 4.2 Phase 10z19b: Cross-Manifold Consistency

Extract manifolds for different fact types:
- Capital cities: "The capital of X is"
- Languages: "The official language of X is"
- Continents: "X is located in"

Check: do the country embeddings (the S(country) deviations) align
across manifolds? If Egypt's position in the capital manifold correlates
with its position in the language manifold, TruthSpace may be unified.

### 4.3 Phase 10z19c: Navigation Test

Start at a known fact and navigate:
1. Extract the "direction" from France to Paris (in attention delta space)
2. Apply the same direction to Egypt → should point toward Cairo
3. Apply to a novel country → does the result decode to a plausible capital?

This tests whether the country→capital mapping is a consistent
DIRECTION in hidden space (TruthSpace) or a per-fact lookup (relative).

### 4.4 Phase 10z19d: Self-Consistency Loop

Navigate a closed loop and check for consistency:
```
France → (capital) → Paris → (language) → French → (country) → France
```

If we end up back at France (or close to it), the manifold is
self-consistent. If we drift, the geometry is leaky.

---

## 5. What We're Looking For

### Signs of Relative Knowledge
- Prediction quality degrades with distance from training examples
- Manifolds for different fact types are unrelated
- Navigation produces noise for unseen entities
- Cross-manifold consistency is low
- The manifold has clear "edges" beyond which it's meaningless

### Signs of TruthSpace
- Prediction works even for obscure/unseen facts
- Different fact manifolds share structural features
- Navigation produces geometrically consistent results for novel entities
- Cross-manifold consistency is high
- The manifold extends smoothly beyond training examples
- Self-consistency loops close (or nearly close)

### The Discriminating Test

The clearest discriminator: **predict the capital of a country the
model has never heard of.** Not a real-but-obscure country (which
might be in the training data). A completely FICTIONAL country with
a constructed embedding.

If we can place "Elbonia" on the manifold by constructing its
embedding from structural principles, and the manifold predicts a
geometrically consistent capital, then we're navigating TruthSpace.

If the prediction is random noise, knowledge is relative.

---

## 6. Connection to the Hypothesis

The TruthSpace hypothesis, if confirmed, would be the strongest
evidence yet for the core TruthSpace-LCM hypothesis:

> Structure IS information. Geometry IS computation. The shape IS
> the knowledge.

If knowledge lives in a universal geometric space that exists
independent of training examples — if the model's weights are
merely a partial window into this space — then:

1. The model doesn't "memorize" facts; it maps into TruthSpace
2. TruthSpace's structure constrains what's true
3. Novel knowledge is not created but DISCOVERED by geometric navigation
4. The model's training is an approximation of TruthSpace's geometry

This would mean that geometric LCMs don't need to be trained on
every fact — they need to be trained on enough facts to reveal the
structure of TruthSpace, after which novel facts can be derived
geometrically.

---

## 7. Risks and Failure Modes

1. **Overfitting the manifold**: with only 8 training facts, the
   manifold might capture noise rather than structure. Mitigation:
   test on truly unseen facts.

2. **Token frequency effects**: common tokens (Paris, Berlin) may
   dominate the manifold. Mitigation: include rare capitals
   (Naypyidaw, Ngerulmud).

3. **Prompt structure leakage**: the model might be using syntactic
   patterns rather than semantic knowledge. Mitigation: vary prompt
   structure.

4. **Confirmation bias**: we might interpret noise as structure.
   Mitigation: include negative controls (fictional entities that
   SHOULD produce noise if knowledge is relative).

---

## 8. Experiment Log

| Phase | What | Result |
|:------|:-----|:-------|
| 10z19a | Predict unseen capitals from manifold | **9/11 rank 0** (manifold_proj) |
| 10z19a | Structure-only (no answer dir) | FAILED (all worse than baseline) |
| 10z19b | Cross-manifold consistency (L23) | **cos=0.9374** (capital vs language) |
| 10z19b | Cross-manifold consistency (L22+23+27) | **cos=0.7735** |
| 10z19c | Navigation test | No universal direction (cosines -0.20 to -0.34) |
| 10z19d | Self-consistency loop | *PENDING* |
| 10z20a | Entity-answer mapping (learned linear) | FAILED (rank 9–84, cos 0.08–0.34) |
| 10z20b | Cross-relationship transfer (lang→cap) | FAILED (rank 10–132) |
| 10z20c | Multi-manifold triangulation | FAILED (rank 25–127) |
| 10z20d | Entity-weighted voting | FAILED (rank 36–157) |
| 10z20 | Category constraint | **CONFIRMED** (top-5 all capitals) |
| 10z21a | Per-head V·W_o cosine with answer | L23 H6: cos 0.21–0.32, L22 H15: 0.13–0.22 |
| 10z21b | Direct binding read (single layer) | **7/8 rank ≤ 9** (France=2, Spain=2, Egypt=3) |
| 10z21c | Multi-layer aggregate (L22+L23+L27) | **Spain rank 0**, Italy=3, France=4, Egypt=4 |
| 10z21d | Full attention-weighted (all positions) | Mixed: France=1, Egypt=2 but noise from other tokens |
| 10z21 | Multilingual identity | **CONFIRMED** — V·W_o maps to Chinese tokens for same entities |
| 10z22a | M_h SVD (L23 H6) | Near-isometry: S[0]/S[1]=1.1, rank90=66, bias=4% |
| 10z22b | Structure preservation | L22 preserves (r=0.75), L23 H6 transforms (r=0.27) |
| 10z22c | Leave-one-out transfer | **FAILED** (mean/nn/ridge: rank 499–63,842) |
| 10z22c | Direct M_h | **WORKS** (rank 4–18, cos=1.0 for all 8 countries) |
| 10z22d | Answer-producing inputs | cos(ideal, actual entity) ≈ 0.10–0.14 (nearly orthogonal) |
| 10z23a | Extended countries (12 new) | **10/12 rank 2–10** (India/Kenya: tokenization) |
| 10z23b | Obscure countries (5) | Luxembourg rank 0; others → regional identity |
| 10z23c | Last token vs country token | **Last token BETTER** (France: 3 vs 6, Japan: 6 vs 10) |
| 10z23d | Capital M_h on language prompts | **Language rank 1** for 5/6 — M_h is entity-agnostic! |

---

## 9. Results — The Architecture of Knowledge (F120)

Phase 10z19 revealed a two-level structure:

### Level 1: Entity Identity IS Absolute (TruthSpace)

When comparing a country's displacement across DIFFERENT fact types
("capital of France" vs "language of France"), the L23 cosine is:

| Country | cos(capital, language) |
|:--------|:----------------------|
| France | 0.9367 |
| Japan | 0.9323 |
| Germany | 0.9459 |
| Italy | 0.9433 |
| Brazil | 0.9369 |
| Egypt | 0.9419 |
| Spain | 0.9250 |
| **Mean** | **0.9374** |

"France" IS a fixed direction in ℝ³⁵⁸⁴, shared across fact types.
This is TruthSpace — entities have absolute geometric coordinates.

### Level 2: Relationships ARE Relative

Country→capital is NOT a universal direction. Cross-fact displacement
cosines at L23 are all negative (-0.20 to -0.34). Each country has
its own path from identity to answer.

Structure-only prediction (generic "capital city" direction) fails
for unseen facts (all 11 worse than baseline). The manifold captures
the SPACE of valid paths, but navigating it requires knowing the
answer direction.

### The Revised Formula

```
Knowledge = Entity Position × Relationship Manifold × Answer Direction
          = TruthSpace      × Learned Structure      × Specific Target
          = Absolute         × Generalizes            × Required
```

### Bottleneck for True Knowledge Extension

To predict a truly novel fact without knowing the answer:
1. ✅ Entity positions available (TruthSpace, cos=0.94 across types)
2. ✅ Relationship manifolds generalize (9/11 unseen facts at rank 0)
3. ❌ Answer direction unknown (structure-only fails)

Phase 10z20 (triangulation) attempted to solve bottleneck #3 by
combining multiple manifolds. All approaches failed — confirming the
"binding problem" (F121).

---

## 10. Results — V·W_o Binding Extraction (F122)

Phase 10z21 solved the binding problem by going directly to the
weight matrices.

### The Mechanism

V·W_o at the entity token position, weighted by attention from the
last token, directly encodes the entity→answer binding:

```
binding = Σ_h attn[last→entity, h] * (normed[entity] @ W_v_h.T + b_v_h) @ W_o_h.T
```

Fed through the LM head, this produces logits where the correct
answer ranks 2–19 for 7/8 countries (single best layer), and
**rank 0 for Spain** (multi-layer aggregate).

### Key Heads

| Layer | Head | Role |
|:------|:-----|:-----|
| L23 | H6 | Primary fact head (cos 0.21–0.32, attn 0.55–0.62) |
| L22 | H15 | Supporting (cos 0.13–0.22, attn 0.48–0.61) |
| L22 | H19 | Supporting (cos 0.10–0.18, attn 0.51–0.57) |
| L27 | H3 | High cos (0.13–0.33) but near-zero attention |

### Revised Three-Level Architecture

```
Level 1: Entity Identity — TruthSpace (absolute, cos=0.94)
  WHERE: L0-L21 hidden states
  WHAT:  Fixed geometric coordinates, language-agnostic

Level 2: Fact Binding — V·W_o (entity-specific transformation)
  WHERE: L22-L23 attention heads (H6, H15, H19)
  WHAT:  Maps entity position → answer direction
  HOW:   Gets answer to rank 2-19

Level 3: Answer Amplification — MLP layers
  WHERE: L24-L31 MLP + residual
  WHAT:  Boosts answer signal from rank 2-19 → rank 0
```

### The Revised Formula

```
Knowledge = Entity Position × V·W_o Binding × MLP Amplification
          = TruthSpace      × Weight Geometry  × Residual Stream
          = Absolute         × Extractable      × Emergent
```

### Multilingual Confirmation

V·W_o maps entity hidden states to their FULL semantic cluster
across languages. Top predictions include Chinese tokens:
法国/巴黎 (France/Paris), 德国/柏林 (Germany/Berlin),
日本/东京 (Japan/Tokyo), etc.

This confirms TruthSpace is language-agnostic: the same geometric
transformation maps to all representations of the same concept.

### Implications for Knowledge Extension

The bottleneck is no longer the answer direction — it's V·W_o itself.
To predict truly novel facts, we would need to:
1. Characterize V·W_o as a geometric transformation (rotation? scaling?)
2. Determine if V·W_o has structure that generalizes across entities
3. Test whether V·W_o binding can be decomposed or transferred

This connects back to F40 (L23 H6 geometric selector) and F116
(universal one-axis d_k structure): the routing is universal, but
the V·W_o content is entity-specific.

---

## 11. Results — V·W_o Geometry (F123)

Phase 10z22 characterized M_h = W_v_h.T @ W_o_h.T as a geometric
transformation.

### M_h is a Near-Isometry

- S[0]/S[1] ≈ 1.1 across all key heads
- Effective rank 42–66 out of 128 bottleneck
- Singular values nearly uniform (CV = 0.26–0.40)
- Bias negligible (3–6% of signal)

M_h projects through a ~66-dimensional subspace, preserving distances
within that subspace. It is NOT low-rank (like the attention MESH in
F39) — it uses most of its capacity.

### L22 Preserves, L23 Transforms

Structure preservation (correlation of pairwise entity cosines vs
pairwise binding cosines):
- L22 H15: r = 0.753 (preserves entity geometry)
- L22 H19: r = 0.671
- L23 H6: r = 0.272 (transforms entity geometry → answer geometry)
- L23 H4: r = 0.235

This reveals a two-stage pipeline: L22 carries entity identity
forward while L23 rotates it into answer space.

### Transfer is Impossible (from examples)

Leave-one-out binding prediction:
- Mean binding: rank 823–55,886
- Nearest neighbor: rank 499–54,588
- Ridge regression: rank 893–63,842
- **Direct M_h: rank 4–18** (perfect, cos = 1.0)

The binding cannot be learned from entity-to-entity examples.
M_h has 66 effective dimensions but only 7 training points — the
transformation is fundamentally underdetermined from examples alone.

### The Revised Understanding

```
Facts are NOT (entity, answer) pairs.
Facts ARE a single geometric transformation M_h that maps
the ENTIRE entity space to the answer space simultaneously.

All capital city knowledge = one 3584×3584 near-isometric matrix
  - Rank ~66 effective dimensions
  - Entity-specificity from INPUT, not from parameters
  - M_h extracts a small signal from the entity's 3584-d state
    (cos with ideal input ≈ 0.10-0.14)
```

### Updated Knowledge Formula

```
Knowledge = M_h(entity_state)
          = Near-isometric projection through 66-d fact subspace
          = Universal transformation × Entity-specific input
```

The bottleneck for novel entities is no longer the binding or the
answer direction — it's whether the model's entity hidden state
for an unseen entity projects correctly into M_h's 66-d subspace.
If it does, M_h will produce the correct answer automatically.

---

## 12. Results — M_h Generalizes: Universal Entity Identity (F124)

Phase 10z23 tested M_h on unseen entities and across fact types.

### M_h Generalizes to Unseen Countries

10/12 extended countries at rank 2–10 (Mexico=2, Thailand=3,
Poland=3, Norway=3, Argentina=4, Sweden=4, Australia=5, Turkey=5,
Russia=7, China=10). India and Kenya fail due to subword answer
tokens, not M_h failure — entity identity is correctly extracted.

Obscure countries fail gracefully: Luxembourg rank 0 (better than
baseline!), but Latvia→Baltic, Bhutan→Tibet, Madagascar→Africa.
M_h extracts REGIONAL identity when specific answer signal is weak.

### Last Token > Country Token

normed@last outperforms normed@cpos for all 6 countries tested.
The last token has accumulated entity + context signal from earlier
layers, making it a more informative input to M_h.

### M_h is NOT Fact-Type-Specific

Applying CAPITAL M_h to language prompts ("The language of France
is"): language token ranks 1 for 5/6 countries, capital token ranks
6–14. M_h extracts the entity's full semantic cluster; the fact
type is determined by PROMPT CONTEXT in the hidden state, not M_h.

### Final Knowledge Architecture

```
M_h is a UNIVERSAL ENTITY IDENTITY EXTRACTOR:
  - Same M_h works for capitals, languages, any fact type
  - Entity-specificity from INPUT hidden state
  - Fact-type-specificity from CONTEXT in hidden state
  - M_h projects through 66-d subspace → full semantic cluster
  - MLP layers (L24-L31) amplify the context-relevant answer

Knowledge = M_h(context-biased entity state)
          = Universal extractor × (Entity identity + Context signal)
```

The bottleneck for truly novel facts is now clear: the entity must
have a hidden state that projects into M_h's 66-d subspace with
sufficient answer signal. Well-known entities (10/12 new countries)
have this; obscure entities (Bhutan, Madagascar) do not.

---

*This document will be updated as experiments progress.*
